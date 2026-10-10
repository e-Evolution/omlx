# SPDX-License-Identifier: Apache-2.0
"""A head that fits alone, waits beside admitted work, then becomes the lone head.

Forced pressure (image decode at a 42 GB ceiling): req1 fits alone and was
correctly deferred while req0 ran ("fits alone but not now"). Its two
prefill-eviction retries were spent in that wait. When req0 finished, req1 was
the only waiting request and was re-checked on a usage reading that still held
pooled Metal buffers and the finished request's KV in the store pipeline
(current 30.27 GB, ~40.05 GB peak over the 31.28 GB line). With no retry left
and the stale estimate over the solo line too, it was rejected by the prefill
memory guard. 0.1 s later the eviction ladder freed 6.21 + 2.55 GB for the
next request.

The tests drive the real scheduler loop on the tiny model against real stores.
The usage reading is controlled; the engine pool's eviction ladder is stood in
by a servicer that frees what the ladder can free when an eviction runs.
"""

import threading

from test_scheduler_capacity_gate import GB, _guarded_rig
from test_scheduler_park import PROMPT_A, PROMPT_C, submit, waiting_ids

import omlx.scheduler as scheduler_module


class _Ladder:
    """Stands in for EnginePool's prefill-eviction ladder (EngineCore runs it
    between steps for the request a step paused)."""

    def __init__(self):
        self.calls = []
        self.frees = None  # callable run when a pass can free memory

    def run(self, eviction_request):
        self.calls.append(eviction_request.request_id)
        if self.frees is not None:
            self.frees()


def _step(rig, ladder, n=1):
    outputs = []
    for _ in range(n):
        result = rig.scheduler.step()
        outputs.extend(result.outputs)
        if result.prefill_eviction_request is not None:
            ladder.run(result.prefill_eviction_request)
        rig.drain()
    return outputs


def _step_until(rig, ladder, predicate, limit=80):
    outputs = []
    for _ in range(limit):
        if predicate():
            return outputs
        outputs.extend(_step(rig, ladder))
    raise AssertionError("condition not reached")


def _done(scheduler, request):
    """Finished, or dropped by a rejection (which leaves no live request)."""
    return request.is_finished() or request.request_id not in scheduler.requests


def _stale_usage(scheduler):
    """A reading whose estimate is over the admission line and the solo line
    (the observed rejection: ~40.05 GB against a 31.28 GB line)."""
    return max(scheduler._admission_limit_bytes(), scheduler._park_resume_lines()[1])


def _defer_beside_admitted_work(rig, ladder, monkeypatch, usage):
    """Phase 1: rows are admitted and hold the memory; the head defers after
    spending both eviction retries on passes that free nothing."""
    scheduler = rig.scheduler
    scheduler._prefill_memory_guard = True
    admitted = [1]
    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: admitted[0])
    monkeypatch.setattr(
        scheduler,
        "_admitted_request_ids",
        lambda: frozenset({"req0"} if admitted[0] else ()),
    )
    usage[0] = scheduler._admission_limit_bytes()
    later = submit(rig, "later", PROMPT_C, arrival=1.0)
    outputs = _step(rig, ladder, 4)
    assert not [o for o in outputs if o.error]
    assert waiting_ids(rig) == ["later"]
    assert "later" in scheduler._deferred_admissions
    assert ladder.calls == ["later", "later"]
    return later, admitted


def test_a_fits_later_wait_does_not_spend_the_lone_heads_eviction_budget(
    tmp_path, mock_tokenizer, monkeypatch
):
    """The stale-reading scenario: once the rows it waited on leave, the head gets a
    fresh eviction budget, the ladder frees the stale memory, and it runs."""
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    ladder = _Ladder()
    try:
        scheduler = rig.scheduler
        later, admitted = _defer_beside_admitted_work(rig, ladder, monkeypatch, usage)

        # req0 finishes. The reading lags its release: only an eviction pass
        # (hot cache, idle model, pooled buffers on the engine thread) frees it.
        admitted[0] = 0
        usage[0] = _stale_usage(scheduler)
        free_now = scheduler._admission_limit_bytes() - 50 * GB
        ladder.frees = lambda: usage.__setitem__(0, free_now)
        outputs = _step_until(rig, ladder, lambda: _done(scheduler, later))
        assert later.finish_reason != "error"
        assert not [o for o in outputs if o.request_id == "later" and o.error]
        assert ladder.calls[2:] == ["later"]  # the ladder ran for it again
    finally:
        rig.close()


def test_a_backoff_retry_beside_the_same_rows_gets_no_fresh_budget(
    tmp_path, mock_tokenizer, monkeypatch
):
    """While the rows it waits on still run, a retry on backoff does not pause
    for the ladder again: the budget renews only when the wait ends."""
    monkeypatch.setattr(scheduler_module, "DEFERRED_ADMISSION_RETRY_BASE_S", 0.0)
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    ladder = _Ladder()
    try:
        later, _admitted = _defer_beside_admitted_work(rig, ladder, monkeypatch, usage)
        outputs = _step(rig, ladder, 6)
        assert not [o for o in outputs if o.error]
        assert waiting_ids(rig) == ["later"]
        assert ladder.calls == ["later", "later"]
        assert later.prefill_eviction_retries == 2
    finally:
        rig.close()


def test_an_unrelated_store_drain_beside_the_same_rows_gets_no_fresh_budget(
    tmp_path, mock_tokenizer, monkeypatch
):
    """An earlier request's store write in flight at deferral is not what the
    head waits on: its drain, while the rows it waits on still run, does not
    end the wait or renew the eviction budget."""
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    ladder = _Ladder()
    try:
        scheduler = rig.scheduler
        gate = scheduler_module._StoreCacheGate(cap=3)
        gate.note_submitted()  # an unrelated earlier request's write
        scheduler._store_cache_gate = gate
        later, _admitted = _defer_beside_admitted_work(rig, ladder, monkeypatch, usage)
        assert scheduler._deferred_admissions["later"].store_jobs == 1

        gate.note_done()  # it drains; req0 still runs
        outputs = _step(rig, ladder, 6)
        assert not [o for o in outputs if o.error]
        assert waiting_ids(rig) == ["later"]
        assert ladder.calls == ["later", "later"]
        assert later.prefill_eviction_retries == 2
    finally:
        rig.close()


def test_a_lone_head_on_a_stale_reading_reclaims_and_rechecks_before_rejecting(
    tmp_path, mock_tokenizer, monkeypatch
):
    """With no eviction left to ask for, the lone head that fits alone first
    returns the pooled buffers on the scheduler thread and re-measures."""
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    ladder = _Ladder()
    try:
        scheduler = rig.scheduler
        scheduler._prefill_memory_guard = True
        monkeypatch.setattr(
            scheduler, "_raise_prefill_eviction_if_available", lambda **_kw: None
        )
        free_now = scheduler._admission_limit_bytes() - 50 * GB
        reclaims = []

        def reclaim():
            reclaims.append(usage[0])
            usage[0] = free_now  # the pooled buffers return
            return usage[0]

        monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", reclaim)
        usage[0] = _stale_usage(scheduler)
        only = submit(rig, "only", PROMPT_C, arrival=1.0)
        outputs = _step_until(rig, ladder, lambda: _done(scheduler, only))
        assert only.finish_reason != "error"
        assert not [o for o in outputs if o.error]
        assert reclaims, "the head was not re-measured after a reclaim"
    finally:
        rig.close()


def test_a_lone_head_that_still_does_not_fit_after_the_reclaim_is_rejected(
    tmp_path, mock_tokenizer, monkeypatch
):
    """Memory nothing gives back (another app, a leak) is not waited on: with
    nothing admitted, nothing draining and the reclaim freeing nothing, the
    older rejection still applies."""
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    ladder = _Ladder()
    try:
        scheduler = rig.scheduler
        scheduler._prefill_memory_guard = True
        usage[0] = _stale_usage(scheduler)
        submit(rig, "only", PROMPT_C, arrival=1.0)
        outputs = _step(rig, ladder, 4)
        errors = [o for o in outputs if o.request_id == "only" and o.error]
        assert len(errors) == 1
        assert errors[0].error_code != "context_exceeds_capacity"
        assert "only" not in scheduler.requests
    finally:
        rig.close()


# --- The finished request's KV in the store pipeline -----------------------------


def _step_unheld(rig, ladder, n=1):
    """Step without waiting for the (held) store-cache worker."""
    outputs = []
    scheduler = rig.scheduler
    for _ in range(n):
        result = scheduler.step()
        outputs.extend(result.outputs)
        if result.prefill_eviction_request is not None:
            ladder.run(result.prefill_eviction_request)
        commit_executor = getattr(scheduler, "_prefill_commit_executor", None)
        if commit_executor is not None:
            commit_executor.submit(lambda: None).result(timeout=20)
        scheduler._process_finished_commits()
    return outputs


def _store_held_rig(tmp_path, tokenizer, monkeypatch, ladder):
    """A request finished and its store-cache write is held in flight.

    While the pipeline owns the payload, the reading is over the solo line;
    once it gives the payload back the reading drops 50 GB under the line.
    """
    rig, _usage = _guarded_rig(
        tmp_path, tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    scheduler = rig.scheduler
    scheduler._prefill_memory_guard = True
    scheduler._store_cache_gate = scheduler_module._StoreCacheGate(cap=3)
    release = threading.Event()
    real_worker = scheduler._async_store_cache_worker

    def held_worker(*args, **kwargs):
        # Only the post-completion store is held; prefill progress commits
        # run on their own executor and are not held.
        if threading.current_thread().name.startswith("omlx-store-cache"):
            release.wait(timeout=30)
        return real_worker(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_async_store_cache_worker", held_worker)
    stale = _stale_usage(scheduler)
    free_now = scheduler._admission_limit_bytes() - 50 * GB

    def usage(**_kw):
        gate = scheduler._store_cache_gate
        owned = (gate is not None and gate.in_flight > 0) or bool(
            scheduler._pending_async_removes
        )
        return stale if owned else free_now

    monkeypatch.setattr(scheduler, "_current_usage_bytes", usage)
    first = submit(rig, "first", PROMPT_A, arrival=0.5, max_tokens=2)
    for _ in range(60):
        if first.is_finished():
            break
        _step_unheld(rig, ladder)
    assert first.is_finished() and first.finish_reason != "error"
    assert scheduler._num_admitted_requests() == 0
    assert scheduler._store_cache_gate.in_flight == 1
    return rig, release


def test_a_lone_head_behind_a_draining_store_write_waits_then_runs(
    tmp_path, mock_tokenizer, monkeypatch
):
    ladder = _Ladder()
    rig, release = _store_held_rig(tmp_path, mock_tokenizer, monkeypatch, ladder)
    try:
        scheduler = rig.scheduler
        second = submit(rig, "second", PROMPT_C, arrival=1.0)
        submit(rig, "newer", PROMPT_A, arrival=2.0)
        outputs = _step_unheld(rig, ladder, 6)
        assert not [o for o in outputs if o.error], [o.error for o in outputs]
        assert waiting_ids(rig) == ["second", "newer"]  # nothing overtakes it

        release.set()  # the write finishes; the pipeline gives the KV back
        for _ in range(200):
            if second.is_finished():
                break
            outputs.extend(_step_unheld(rig, ladder))
            scheduler._store_cache_executor.submit(lambda: None).result(timeout=20)
        assert second.is_finished()
        assert second.finish_reason != "error"
        assert not [o for o in outputs if o.error]
    finally:
        release.set()
        rig.close()


def test_the_head_behind_a_store_write_is_not_retried_every_step(
    tmp_path, mock_tokenizer, monkeypatch
):
    monkeypatch.setattr(scheduler_module, "DEFERRED_ADMISSION_RETRY_BASE_S", 60.0)
    monkeypatch.setattr(scheduler_module, "DEFERRED_ADMISSION_RETRY_MAX_S", 60.0)
    ladder = _Ladder()
    rig, release = _store_held_rig(tmp_path, mock_tokenizer, monkeypatch, ladder)
    try:
        scheduler = rig.scheduler
        attempts = []
        real_check = scheduler._preflight_memory_check

        def check(request):
            if request.request_id == "second":
                attempts.append(request.prefill_eviction_retries)
            return real_check(request)

        monkeypatch.setattr(scheduler, "_preflight_memory_check", check)
        submit(rig, "second", PROMPT_C, arrival=1.0)
        _step_unheld(rig, ladder, 10)
        # Two eviction passes, then one deferral; no retry while it holds.
        assert len(attempts) == 3, attempts
        assert waiting_ids(rig) == ["second"]
    finally:
        release.set()
        rig.close()


# --- Park ---------------------------------------------------------------------------


def test_a_park_resume_starts_a_fresh_eviction_budget(tmp_path, mock_tokenizer):
    """A park is a deliberate wait: retries spent before it do not count
    against the re-admission."""
    from test_scheduler_park import make_rig

    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        request = submit(rig, "parked", PROMPT_C, arrival=1.0)
        request.parked = True
        request.prefill_eviction_retries = 2
        rig.scheduler._finish_park_resume(request, log=False)
        assert request.prefill_eviction_retries == 0
    finally:
        rig.close()
