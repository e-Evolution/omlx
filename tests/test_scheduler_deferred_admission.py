# SPDX-License-Identifier: Apache-2.0
"""A fits-later request that waits at the head of the queue.

``_release_deferred_admission`` gives back everything a deferred admission
attempt took (restored prefix, block table, SpecPrefill indices), so the
waiting request holds no memory. Two further cases:

- the block-table release queued behind an in-flight prefill commit must not
  free a table the request got after it was re-admitted;
- the deferred head is not retried, and its prefix not restored from SSD
  again, on every step while the memory picture is unchanged.
"""

import concurrent.futures

from test_scheduler_capacity_gate import GB, _guarded_rig
from test_scheduler_park import PROMPT_A, PROMPT_C, step, submit, waiting_ids

import omlx.scheduler as scheduler_module


def _warm_prefix(rig):
    """Finish one request on PROMPT_C so later prompts restore its blocks."""
    scheduler = rig.scheduler
    first = submit(rig, "warm", PROMPT_C, arrival=0.5, max_tokens=1)
    for _ in range(60):
        if first.is_finished():
            break
        step(rig)
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.drain()


# --- A stale deferred release never frees a newer block table -----------------


def test_a_stale_deferred_release_leaves_the_readmitted_table_intact(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    try:
        scheduler = rig.scheduler
        paged = rig.paged
        request = submit(rig, "r", PROMPT_C, arrival=1.0)

        # The deferred attempt's lookup built table A holding one block ref.
        old_table = paged.create_block_table("r")
        old_block = paged.allocate_block()
        paged.add_block_to_table(old_table, old_block, 4)

        # A prefill commit for "r" is still running: the release is queued.
        pending = concurrent.futures.Future()
        commit_running = [True]
        real_inflight = scheduler._inflight_commit_future
        monkeypatch.setattr(
            scheduler,
            "_inflight_commit_future",
            lambda rid: pending if commit_running[0] else real_inflight(rid),
        )
        scheduler._release_deferred_admission(request)
        assert paged.get_block_table("r") is old_table  # still queued

        # The request is re-admitted with a NEW table before the commit ends.
        new_table = paged.create_block_table("r")
        new_block = paged.allocate_block()
        paged.add_block_to_table(new_table, new_block, 4)

        # The commit ends; the queued release runs at the next step start.
        commit_running[0] = False
        pending.set_result(None)
        scheduler._process_finished_commits()

        assert paged.get_block_table("r") is new_table
        assert new_table.block_ids == [new_block.block_id]
        assert new_block.ref_count == 1  # the new run keeps its ref
        assert old_block.ref_count == 0  # the old attempt's ref is released
    finally:
        rig.close()


def test_a_deferred_release_without_a_commit_runs_now(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    try:
        scheduler = rig.scheduler
        paged = rig.paged
        request = submit(rig, "r", PROMPT_C, arrival=1.0)
        table = paged.create_block_table("r")
        block = paged.allocate_block()
        paged.add_block_to_table(table, block, 4)

        scheduler._release_deferred_admission(request)

        assert paged.get_block_table("r") is None
        assert block.ref_count == 0
        assert block.block_id in paged.allocated_blocks  # still findable
    finally:
        rig.close()


# --- The deferred head waits for a memory change before it retries -------------


def _deferring_rig(tmp_path, tokenizer, monkeypatch, *, base_s=60.0, max_s=60.0):
    """A warm prefix, the guard on, one admitted row (constant) and no free
    memory: a request longer than the warm prompt restores a prefix and then
    defers. Returns the rig, the usage cell, the admitted-ids cell and the
    list of the "later" request's restored prefix lengths (one entry per
    restore from cache)."""
    rig, usage = _guarded_rig(
        tmp_path, tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    scheduler = rig.scheduler
    _warm_prefix(rig)
    monkeypatch.setattr(
        scheduler_module, "DEFERRED_ADMISSION_RETRY_BASE_S", base_s, raising=False
    )
    monkeypatch.setattr(
        scheduler_module, "DEFERRED_ADMISSION_RETRY_MAX_S", max_s, raising=False
    )
    scheduler._prefill_memory_guard = True
    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: 1)
    admitted_ids = [frozenset({"busy"})]
    monkeypatch.setattr(
        scheduler, "_admitted_request_ids", lambda: admitted_ids[0], raising=False
    )
    # No idle model to evict: the request defers (an eviction pause keeps
    # the restore on purpose, #2180).
    monkeypatch.setattr(
        scheduler, "_raise_prefill_eviction_if_available", lambda **_kw: None
    )
    usage[0] = scheduler._admission_limit_bytes()
    restores = []
    real_prepare = scheduler._prepare_prefix_cache_for_request

    def prepare(request):
        real_prepare(request)
        if request.request_id == "later" and request.cached_tokens > 0:
            restores.append(request.cached_tokens)

    monkeypatch.setattr(scheduler, "_prepare_prefix_cache_for_request", prepare)
    return rig, usage, admitted_ids, restores


LATER_PROMPT = PROMPT_C + PROMPT_A[:12]  # a prefix hit plus a suffix to prefill
# An exact hit of the warm prompt leaves nothing to prefill, so the preflight
# admits it whatever the memory: only the hold keeps it behind the head.
NEWER_PROMPT = PROMPT_C


def test_an_unchanged_memory_picture_restores_the_deferred_head_once(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage, _ids, restores = _deferring_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        submit(rig, "later", LATER_PROMPT, arrival=1.0)
        submit(rig, "newer", NEWER_PROMPT, arrival=2.0)
        outputs = step(rig, 10)
        assert not [o for o in outputs if o.error]
        assert len(restores) == 1, restores
        # FIFO and the head-of-line hold: nothing overtakes the deferred head.
        assert waiting_ids(rig) == ["later", "newer"]
        assert rig.scheduler.requests["later"].prompt_cache is None
    finally:
        rig.close()


def test_a_finished_admitted_row_triggers_the_next_attempt(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage, admitted_ids, restores = _deferring_rig(
        tmp_path, mock_tokenizer, monkeypatch
    )
    try:
        submit(rig, "later", LATER_PROMPT, arrival=1.0)
        step(rig, 3)
        assert len(restores) == 1
        # The admitted row finishes (or aborts, or is paused): one re-attempt
        # on the next step, then the head waits again.
        admitted_ids[0] = frozenset()
        step(rig)
        assert len(restores) == 2, restores
        step(rig, 5)
        assert len(restores) == 2, restores
        assert waiting_ids(rig) == ["later"]
    finally:
        rig.close()


def test_free_memory_rising_past_the_threshold_triggers_the_next_attempt(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, usage, _ids, restores = _deferring_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        limit = scheduler._admission_limit_bytes()
        delta = scheduler_module.DEFERRED_ADMISSION_FREE_DELTA_FRACTION
        submit(rig, "later", LATER_PROMPT, arrival=1.0)
        step(rig, 2)
        assert len(restores) == 1
        usage[0] = limit - int(limit * delta / 2)  # below the threshold
        step(rig, 3)
        assert len(restores) == 1, restores
        usage[0] = limit - int(limit * delta * 2)  # past it
        step(rig)
        assert len(restores) == 2, restores
        step(rig, 3)  # the new baseline is the freer picture
        assert len(restores) == 2, restores
    finally:
        rig.close()


def test_an_expired_backoff_triggers_the_next_attempt(
    tmp_path, mock_tokenizer, monkeypatch
):
    import time
    import types

    # Advance only the scheduler's clock, so the test does not depend on
    # wall-clock sleeps (which were flaky with a 50 ms margin).
    offset = [0.0]
    fake_time = types.SimpleNamespace(
        **{name: getattr(time, name) for name in dir(time) if not name.startswith("__")}
    )
    fake_time.monotonic = lambda: time.monotonic() + offset[0]
    monkeypatch.setattr(scheduler_module, "time", fake_time)

    rig, _usage, _ids, restores = _deferring_rig(
        tmp_path, mock_tokenizer, monkeypatch, base_s=0.5, max_s=60.0
    )
    try:
        submit(rig, "later", LATER_PROMPT, arrival=1.0)
        step(rig, 2)
        assert len(restores) == 1
        offset[0] += 0.75  # past the 0.5 s backoff
        step(rig)
        assert len(restores) == 2, restores
        # The backoff doubled to 1 s: 0.25 s later it still waits.
        offset[0] += 0.25
        step(rig)
        assert len(restores) == 2, restores
        offset[0] += 1.0  # past the doubled backoff
        step(rig)
        assert len(restores) == 3, restores
    finally:
        rig.close()


def test_the_backoff_doubles_and_is_capped(monkeypatch):
    monkeypatch.setattr(scheduler_module, "DEFERRED_ADMISSION_RETRY_BASE_S", 0.25)
    monkeypatch.setattr(scheduler_module, "DEFERRED_ADMISSION_RETRY_MAX_S", 4.0)
    backoff = scheduler_module._deferred_admission_backoff_s
    assert [backoff(n) for n in range(1, 8)] == [0.25, 0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    assert backoff(10_000) == 4.0  # no overflow on a long wait


def test_a_deferred_head_with_a_pending_teardown_keeps_the_hold(
    tmp_path, mock_tokenizer, monkeypatch
):
    """When its retry is due but its release still waits behind a commit, the
    head keeps the hold instead of being set aside for backfill."""
    rig, _usage, admitted_ids, restores = _deferring_rig(
        tmp_path, mock_tokenizer, monkeypatch
    )
    try:
        scheduler = rig.scheduler
        submit(rig, "later", LATER_PROMPT, arrival=1.0)
        submit(rig, "newer", NEWER_PROMPT, arrival=2.0)
        step(rig)
        assert waiting_ids(rig) == ["later", "newer"]
        real_pending = scheduler._teardown_pending
        monkeypatch.setattr(
            scheduler,
            "_teardown_pending",
            lambda rid: rid == "later" or real_pending(rid),
        )
        admitted_ids[0] = frozenset()  # a retry is due
        step(rig, 2)
        assert waiting_ids(rig) == ["later", "newer"]
        assert "newer" not in scheduler.running
    finally:
        rig.close()


def test_the_deferred_head_is_admitted_once_memory_frees(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, usage, _ids, restores = _deferring_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        later = submit(rig, "later", LATER_PROMPT, arrival=1.0)
        step(rig, 3)
        monkeypatch.setattr(
            scheduler,
            "_num_admitted_requests",
            lambda: len(scheduler.running) + len(scheduler.prefilling),
        )
        usage[0] = scheduler._admission_limit_bytes() - 50 * GB
        for _ in range(60):
            if later.is_finished():
                break
            step(rig)
        assert later.finish_reason != "error"
        assert "later" not in scheduler._deferred_admissions
    finally:
        rig.close()
