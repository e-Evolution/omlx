# SPDX-License-Identifier: Apache-2.0
"""Park-instead-of-abort for chunked prefill under memory pressure.

The tests drive the real scheduler step loop on a small initialized Qwen3.5
hybrid model against real paged-SSD, boundary-snapshot and prefix stores. A
parked request keeps its client stream and its place in line, drops its live
cache, and resumes from the blocks an incremental commit stored.
"""

import time
from types import SimpleNamespace

import test_scheduler_prefill_progress_commit as commit_rig
from test_scheduler_prefill_progress_commit import BLOCK, _Rig, _returns_promptly

import omlx.scheduler as scheduler_module
from omlx.request import Request, RequestStatus, SamplingParams

# Controllable commit worker fixture shared with the incremental-commit suite.
gated = commit_rig.gated


def _prompt(seed):
    """24 tokens (23 prefill tokens: five full blocks plus a tail).

    Token ids stay below the tiny model's vocabulary of 32; an out-of-range id
    reads undefined embedding memory and makes the output nondeterministic.
    """
    return [3 + (seed * 5 + i * 7) % 28 for i in range(24)]


PROMPT_A = _prompt(1)
PROMPT_B = _prompt(2)
PROMPT_C = _prompt(3)
PROMPT_D = _prompt(4)


def make_rig(tmp_path, tokenizer, *, park=True, max_parks=3, ok_polls=3, commit=1):
    rig = _Rig(
        tmp_path,
        tokenizer,
        split=False,
        commit_blocks=commit,
        memory_pressure_park=park,
        max_parks_per_request=max_parks,
        park_resume_ok_polls=ok_polls,
    )
    # The pressure monitor is "live": the hold gate waits for ok polls.
    rig.scheduler.note_pressure_poll("soft")
    return rig


def submit(rig, request_id, tokens, arrival, max_tokens=3):
    request = Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        arrival_time=arrival,
    )
    rig.scheduler.add_request(request)
    return request


def step(rig, n=1):
    """Step the scheduler, settling commit workers so cadence is deterministic."""
    outputs = []
    for _ in range(n):
        result = rig.scheduler.step()
        outputs.extend(result.outputs)
        rig.drain()
    return outputs


def prefilling_ids(rig):
    return [r.request_id for r in rig.scheduler.prefilling]


def waiting_ids(rig):
    return [r.request_id for r in rig.scheduler.waiting]


def step_until(rig, predicate, limit=60):
    outputs = []
    for _ in range(limit):
        if predicate():
            return outputs
        outputs.extend(step(rig))
    raise AssertionError("condition not reached")


def prefill_two(rig):
    """Old (A) and young (B) both mid-prefill with committed blocks."""
    old = submit(rig, "old", PROMPT_A, arrival=1.0)
    young = submit(rig, "young", PROMPT_B, arrival=2.0)
    step_until(
        rig,
        lambda: set(prefilling_ids(rig)) == {"old", "young"}
        and rig.table_tokens("young") >= BLOCK,
    )
    return old, young


def run_to_completion(rig, requests, limit=200):
    outputs = []
    for _ in range(limit):
        if all(r.is_finished() or r.request_id not in rig.scheduler.requests
               for r in requests):
            break
        outputs.extend(step(rig))
    for future in list(rig.scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.scheduler._drain_pending_async_removes()
    return outputs


def finish_ok_polls(rig, n):
    for _ in range(n):
        rig.scheduler.note_pressure_poll("ok")


# --- Request side of the enforcer contract ----------------------------------


def test_park_disabled_never_accepts_a_park_request(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, park=False)
    try:
        prefill_two(rig)
        assert rig.scheduler.request_park_prefill() is False
        step(rig)
        assert set(prefilling_ids(rig)) == {"old", "young"}
    finally:
        rig.close()


def test_nothing_parkable_is_reported_so_the_enforcer_aborts(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        assert rig.scheduler.request_park_prefill() is False  # nothing admitted
        request = submit(rig, "only", PROMPT_A, arrival=1.0)
        # Still waiting (not prefilling): not parkable.
        assert rig.scheduler.request_park_prefill() is False
        assert not request.parked
    finally:
        rig.close()


def test_a_request_that_cannot_be_parked_safely_is_not_a_victim(
    tmp_path, mock_tokenizer
):
    """VLM and SpecPrefill prefills run outside the chunked loop or patch the
    whole model; they are never victims."""
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        old, young = prefill_two(rig)
        young.vlm_inputs_embeds = object()
        old.specprefill_indices = [0]
        assert rig.scheduler.request_park_prefill() is False
    finally:
        rig.close()


# --- Victim policy ------------------------------------------------------------


def test_hard_pressure_parks_the_youngest_prefill_without_an_error(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        old, young = prefill_two(rig)
        scheduler = rig.scheduler
        assert scheduler.request_park_prefill() is True
        outputs = step(rig)

        assert not [o for o in outputs if o.finished or o.error]
        assert prefilling_ids(rig) == ["old"]  # the oldest keeps running
        assert waiting_ids(rig) == ["young"]
        assert young.parked and young.park_count == 1
        assert young.status == RequestStatus.WAITING
        assert "young" in scheduler.requests  # the client stream stays open
        assert "young" not in scheduler._prefill_states
        assert young.prompt_cache is None
        assert young.arrival_time == 2.0
        assert young.prompt_token_ids == PROMPT_B
    finally:
        rig.close()


def test_the_oldest_is_protected_while_a_younger_prefill_is_parkable(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        prefill_two(rig)
        rig.scheduler.request_park_prefill()
        step(rig)
        # The younger one was parked; the oldest keeps prefilling.
        assert waiting_ids(rig) == ["young"]
        assert rig.scheduler.requests["old"].park_count == 0
    finally:
        rig.close()


def test_a_sole_prefill_is_parkable_when_pressure_comes_from_elsewhere(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        only = submit(rig, "only", PROMPT_A, arrival=1.0)
        step_until(rig, lambda: prefilling_ids(rig) == ["only"])
        rig.scheduler._park_hold_applies = lambda head: True  # isolate the park
        assert rig.scheduler.request_park_prefill() is True
        step(rig)
        assert only.parked and waiting_ids(rig) == ["only"]
    finally:
        rig.close()


def test_one_victim_is_parked_per_step_and_per_request(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        submit(rig, "r1", PROMPT_A, arrival=1.0)
        submit(rig, "r2", PROMPT_B, arrival=2.0)
        submit(rig, "r3", PROMPT_C, arrival=3.0)
        step_until(rig, lambda: len(prefilling_ids(rig)) == 3)
        rig.scheduler.request_park_prefill()
        step(rig)
        assert waiting_ids(rig) == ["r3"]  # one victim only
        assert set(prefilling_ids(rig)) == {"r1", "r2"}
    finally:
        rig.close()


def test_parked_requests_resume_in_arrival_order_ahead_of_newer_ones(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        submit(rig, "r1", PROMPT_A, arrival=1.0)
        submit(rig, "r2", PROMPT_B, arrival=2.0)
        submit(rig, "r3", PROMPT_C, arrival=3.0)
        step_until(rig, lambda: len(prefilling_ids(rig)) == 3)
        # A newer, never-admitted request is already queued.
        submit(rig, "r4", PROMPT_D, arrival=4.0)
        rig.scheduler.request_park_prefill()
        step(rig)  # parks r3
        rig.scheduler.request_park_prefill()
        step(rig)  # parks r2 (r1 protected)
        assert waiting_ids(rig) == ["r2", "r3", "r4"]
    finally:
        rig.close()


# --- Hold gate ----------------------------------------------------------------


def test_no_resume_before_the_required_consecutive_ok_polls(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=3)
    try:
        old, young = prefill_two(rig)
        rig.scheduler.request_park_prefill()
        step(rig)
        assert waiting_ids(rig) == ["young"]

        finish_ok_polls(rig, 2)
        step(rig, 3)
        assert waiting_ids(rig) == ["young"]  # two ok polls are not enough

        rig.scheduler.note_pressure_poll("soft")  # the streak is broken
        finish_ok_polls(rig, 2)
        step(rig, 3)
        assert waiting_ids(rig) == ["young"]

        finish_ok_polls(rig, 1)  # third consecutive ok poll
        step_until(rig, lambda: "young" in prefilling_ids(rig) or young.output_token_ids)
        assert not young.parked
    finally:
        rig.close()


def _parked_young(rig, level):
    prefill_two(rig)
    rig.scheduler.request_park_prefill()
    step(rig)
    assert waiting_ids(rig) == ["young"]
    rig.scheduler.note_pressure_poll(level)
    return rig.scheduler.requests["young"]


def test_parked_head_passes_when_nothing_is_admitted(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        _parked_young(rig, "soft")
        monkeypatch.setattr(rig.scheduler, "_num_admitted_requests", lambda: 0)
        assert rig.scheduler._park_hold_applies(rig.scheduler.waiting[0]) is False
        step_until(rig, lambda: "young" in prefilling_ids(rig))
    finally:
        rig.close()


def test_parked_head_is_released_after_the_max_hold_unless_hard(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        young = _parked_young(rig, "soft")
        scheduler = rig.scheduler
        assert scheduler._park_hold_applies(young) is True  # fresh park holds
        young.parked_at -= scheduler_module.PARK_MAX_HOLD_S + 1
        assert scheduler._park_hold_applies(young) is False  # soft: bounded
        scheduler.note_pressure_poll("hard")
        assert scheduler._park_hold_applies(young) is True  # hard: still held
    finally:
        rig.close()


def test_new_requests_do_not_overtake_a_parked_one(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        prefill_two(rig)
        rig.scheduler.request_park_prefill()
        step(rig)
        submit(rig, "newer", PROMPT_C, arrival=9.0)
        step(rig, 4)
        assert waiting_ids(rig) == ["young", "newer"]
        assert "newer" not in prefilling_ids(rig)
    finally:
        rig.close()


def test_parked_request_is_exempt_from_the_admission_stall_timeout(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        scheduler = rig.scheduler
        scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S = 0.0
        scheduler._STORE_CACHE_ADMISSION_STALL_TIMEOUT_S = 0.0
        old, young = prefill_two(rig)
        scheduler.request_park_prefill()
        step(rig)
        scheduler._admission_paused = True
        for _ in range(3):
            assert scheduler._memory_admission_stall_output("admission_paused") is None
            time.sleep(0.01)
        outputs = step(rig, 5)
        assert not [o for o in outputs if o.error]
        assert waiting_ids(rig) == ["young"]
        assert "young" in scheduler.requests
    finally:
        rig.close()


def test_the_resume_signal_is_not_awaited_when_no_monitor_reports(
    tmp_path, mock_tokenizer
):
    """Without an enforcer (no poll ever seen) a parked request is not starved."""
    rig = _Rig(
        tmp_path,
        mock_tokenizer,
        split=False,
        commit_blocks=1,
        memory_pressure_park=True,
    )
    try:
        assert rig.scheduler._park_resume_allowed() is True
        rig.scheduler.note_pressure_poll("hard")
        assert rig.scheduler._park_resume_allowed() is False
    finally:
        rig.close()


# --- Resume ------------------------------------------------------------------


def uninterrupted_output(tmp_path, tokenizer):
    rig = make_rig(tmp_path / "baseline", tokenizer, park=False)
    try:
        request = submit(rig, "young", PROMPT_B, arrival=2.0)
        run_to_completion(rig, [request])
        return list(request.output_token_ids)
    finally:
        rig.close()


def test_resume_restores_the_committed_boundary_and_matches_an_uninterrupted_run(
    tmp_path, mock_tokenizer, caplog
):
    import logging

    expected = uninterrupted_output(tmp_path, mock_tokenizer)
    assert expected

    rig = make_rig(tmp_path / "park", mock_tokenizer)
    try:
        old, young = prefill_two(rig)
        scheduler = rig.scheduler
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            scheduler.request_park_prefill()
            step(rig)
            rig.drain()
            committed = rig.probe(PROMPT_B)[0].num_tokens
            assert committed >= BLOCK
            assert any(
                "Parked request young" in r.getMessage()
                and "park 1/3" in r.getMessage()
                for r in caplog.records
            )

            finish_ok_polls(rig, 3)
            step_until(rig, lambda: "young" in prefilling_ids(rig))
            assert young.cached_tokens == committed
            assert any(
                f"Resumed parked request young: cached={committed}" in r.getMessage()
                for r in caplog.records
            )
        outputs = run_to_completion(rig, [old, young])
        assert not [o for o in outputs if o.error]
        assert list(young.output_token_ids) == expected
        assert young.park_count == 1 and not young.parked
        stats = scheduler.get_park_stats()
        assert stats["parks"] == 1 and stats["resumes"] == 1
    finally:
        rig.close()


def test_the_cold_retry_guard_is_bypassed_for_a_parked_request(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        old, young = prefill_two(rig)
        young.prefill_oom_retries = 1  # an earlier internal SDPA-OOM requeue
        assert rig.scheduler._retry_prefills_cold_after_oom(young) is True
        young.parked = True
        assert rig.scheduler._retry_prefills_cold_after_oom(young) is False
    finally:
        rig.close()


def test_park_budget_exhaustion_falls_back_to_the_abort(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, max_parks=1, ok_polls=1)
    try:
        only = submit(rig, "only", PROMPT_A, arrival=1.0)
        step_until(rig, lambda: prefilling_ids(rig) == ["only"])
        assert rig.scheduler.request_park_prefill() is True
        step(rig)
        finish_ok_polls(rig, 1)
        step_until(rig, lambda: "only" in prefilling_ids(rig))
        assert only.park_count == 1
        # The one allowed park is spent: the enforcer is told to abort.
        assert rig.scheduler.request_park_prefill() is False
    finally:
        rig.close()


def test_settling_after_a_park_defers_the_abort_for_a_few_polls(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, max_parks=1)
    try:
        submit(rig, "only", PROMPT_A, arrival=1.0)
        step_until(rig, lambda: prefilling_ids(rig) == ["only"])
        rig.scheduler._park_hold_applies = lambda head: True  # isolate the park
        rig.scheduler.request_park_prefill()
        step(rig)
        # Memory is returning to the OS; the enforcer must not abort yet.
        answers = []
        for _ in range(6):
            rig.scheduler.note_pressure_poll("hard")
            answers.append(rig.scheduler.request_park_prefill())
        assert answers[0] is True
        assert answers[-1] is False
    finally:
        rig.close()


# --- Eviction safety (pinning) --------------------------------------------------


def test_committed_blocks_are_pinned_while_parked_and_released_after_resume(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=1)
    try:
        old, young = prefill_two(rig)
        scheduler = rig.scheduler
        scheduler.request_park_prefill()
        step(rig)
        rig.drain()

        pins = scheduler._park_pins["young"]
        assert pins.block_ids and pins.hashes
        assert rig.ssd.pinned_block_count == len(pins.hashes)
        for block_id in pins.block_ids:
            assert rig.paged.allocated_blocks[block_id].ref_count >= 1

        # Even an unbounded size enforcement keeps the parked prefix on disk.
        rig.ssd._evict_tracked_until_size(0)
        table, remaining = rig.probe(PROMPT_B)
        assert table is not None and table.num_tokens >= BLOCK

        finish_ok_polls(rig, 1)
        step_until(rig, lambda: "young" in prefilling_ids(rig))
        assert "young" not in scheduler._park_pins
        assert rig.ssd.pinned_block_count == 0
    finally:
        rig.close()


def test_pins_are_released_when_a_parked_request_is_aborted(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        prefill_two(rig)
        scheduler = rig.scheduler
        scheduler.request_park_prefill()
        step(rig)
        rig.drain()
        assert "young" in scheduler._park_pins

        assert scheduler.abort_request("young")
        step(rig, 2)
        assert "young" not in scheduler._park_pins
        assert rig.ssd.pinned_block_count == 0
        assert "young" not in scheduler.requests
        assert "young" not in waiting_ids(rig)
    finally:
        rig.close()


def test_pinned_blocks_are_not_evicted_from_the_ssd_index_by_lru(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        old, young = prefill_two(rig)
        rig.drain()
        table, _ = rig.probe(PROMPT_B)
        assert table is not None
        hashes = [
            bytes(rig.paged.allocated_blocks[b].block_hash) for b in table.block_ids
        ]
        rig.ssd.pin_blocks(hashes)
        rig.ssd._evict_tracked_until_size(0)
        for block_hash in hashes:
            assert rig.ssd._index.get(block_hash) is not None
        rig.ssd.unpin_blocks(hashes)
        rig.ssd._evict_tracked_until_size(0)
        assert all(rig.ssd._index.get(h) is None for h in hashes)
        assert rig.ssd.pinned_block_count == 0
    finally:
        rig.close()


# --- Teardown never blocks the MLX thread ------------------------------------------


def test_park_with_a_commit_in_flight_defers_the_teardown_without_blocking(
    tmp_path, mock_tokenizer, gated
):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=1)
    gate = None
    try:
        scheduler = rig.scheduler
        old, young = prefill_two(rig)  # commits ran before the gate was armed
        gate = gated(rig)
        # Run until the young request has a commit that cannot finish.
        for _ in range(8):
            if "young" in scheduler._prefill_commits:
                break
            scheduler.step()
        assert "young" in scheduler._prefill_commits
        assert not scheduler._prefill_commits["young"][0].done()
        scheduler.request_park_prefill()

        assert _returns_promptly(scheduler._process_pending_park)
        assert young.parked and waiting_ids(rig)[0] == "young"
        assert scheduler._teardown_pending("young")  # waits behind the commit

        # A held head is set aside; it is not admitted while its teardown pends.
        finish_ok_polls(rig, 1)
        scheduler.step()
        assert "young" not in prefilling_ids(rig)

        gate.release.set()
        rig.drain()
        assert not scheduler._teardown_pending("young")
        assert "young" in scheduler._park_pins
    finally:
        if gate is not None:
            gate.release.set()
        rig.close()


def test_reset_and_shutdown_release_every_pin(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        prefill_two(rig)
        rig.scheduler.request_park_prefill()
        step(rig)
        rig.drain()
        assert rig.ssd.pinned_block_count > 0
        rig.scheduler.reset()
        assert not rig.scheduler._park_pins
        assert rig.ssd.pinned_block_count == 0
    finally:
        rig.close()


# --- Review follow-ups: leaks, resume bursts, hot cache, settle polls ----------


def test_abort_of_a_parked_request_with_a_commit_in_flight_leaves_no_pins(
    tmp_path, mock_tokenizer, gated
):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=1)
    gate = None
    try:
        scheduler = rig.scheduler
        _, young = prefill_two(rig)
        gate = gated(rig)
        for _ in range(8):
            if "young" in scheduler._prefill_commits:
                break
            scheduler.step()
        assert not scheduler._prefill_commits["young"][0].done()
        scheduler.request_park_prefill()
        scheduler._process_pending_park()
        assert scheduler._teardown_pending("young")

        assert scheduler.abort_request("young")
        scheduler.step()  # the abort runs while the commit is still in flight
        gate.release.set()
        rig.drain()
        step(rig, 3)

        assert not young.parked  # an aborted request is no longer parked
        assert "young" not in scheduler._park_pins
        assert rig.ssd.pinned_block_count == 0
    finally:
        if gate is not None:
            gate.release.set()
        rig.close()


def _park_three(rig):
    """r1 prefilling (protected); r2, r3, r4 parked, all unpinned-ready."""
    for i, (rid, prompt) in enumerate(
        [("r1", PROMPT_A), ("r2", PROMPT_B), ("r3", PROMPT_C), ("r4", PROMPT_D)]
    ):
        submit(rig, rid, prompt, arrival=float(i + 1), max_tokens=50)
    step_until(rig, lambda: len(prefilling_ids(rig)) == 4)
    for _ in range(3):
        rig.scheduler.request_park_prefill()
        step(rig)
    assert waiting_ids(rig) == ["r2", "r3", "r4"]


def _resumed(rig):
    return rig.scheduler.get_park_stats()["resumes"]


def test_parked_requests_resume_one_per_hysteresis_window(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=2)
    try:
        _park_three(rig)
        finish_ok_polls(rig, 2)
        step(rig, 2)
        assert _resumed(rig) == 1  # not a burst of three

        step(rig, 3)  # no new polls: the streak was consumed by the resume
        assert _resumed(rig) == 1

        finish_ok_polls(rig, 2)
        step(rig, 2)
        assert _resumed(rig) == 2
    finally:
        rig.close()


def test_the_max_hold_escape_resumes_one_request_at_a_time(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=3)
    try:
        _park_three(rig)
        rig.scheduler.note_pressure_poll("soft")
        for request in list(rig.scheduler.waiting):
            request.parked_at -= scheduler_module.PARK_MAX_HOLD_S + 1
        step(rig, 2)
        assert _resumed(rig) == 1
        step(rig, 3)  # still no polls: the escape does not chain
        assert _resumed(rig) == 1
    finally:
        rig.close()


def test_a_stale_monitor_resumes_one_request_per_step(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        _park_three(rig)
        rig.scheduler._last_pressure_poll_at = (
            time.monotonic() - scheduler_module.PARK_SIGNAL_STALE_S - 1
        )
        rig.scheduler._schedule_waiting()
        assert _resumed(rig) == 1
    finally:
        rig.close()


def test_a_hot_cache_shrink_reclaims_parked_blocks_that_stay_restorable(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, ok_polls=1)
    try:
        rig.ssd._hot_cache_max_bytes = 64 * 1024**2
        rig.ssd._hot_cache_enabled = True
        old, young = prefill_two(rig)
        scheduler = rig.scheduler
        scheduler.request_park_prefill()
        step(rig)
        rig.drain()
        pins = scheduler._park_pins["young"]
        assert pins.hashes
        assert set(pins.hashes) & set(rig.ssd._hot_cache)  # blocks sit in RAM

        protected = scheduler.get_active_hot_cache_block_hashes()
        assert not protected & set(pins.hashes)  # parked blocks are not RAM-pinned
        rig.ssd.shrink_hot_cache_to(0, protected_hashes=protected)
        assert not set(pins.hashes) & set(rig.ssd._hot_cache)

        # Dirty evictions were written back: SSD LRU still holds them (pinned).
        rig.ssd._evict_tracked_until_size(0)
        table, _ = rig.probe(PROMPT_B)
        assert table is not None and table.num_tokens >= BLOCK

        finish_ok_polls(rig, 1)
        step_until(rig, lambda: "young" in prefilling_ids(rig))
        assert young.cached_tokens >= BLOCK
    finally:
        rig.close()


def test_the_settle_window_counts_enforcer_polls_not_queries(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, max_parks=1)
    try:
        submit(rig, "only", PROMPT_A, arrival=1.0)
        step_until(rig, lambda: prefilling_ids(rig) == ["only"])
        rig.scheduler._park_hold_applies = lambda head: True  # isolate the park
        rig.scheduler.request_park_prefill()
        step(rig)
        # Repeated queries within one poll never consume the window.
        assert all(rig.scheduler.request_park_prefill() for _ in range(10))
        for _ in range(scheduler_module.PARK_SETTLE_POLLS):
            rig.scheduler.note_pressure_poll("hard")
            assert rig.scheduler.request_park_prefill() is True
        rig.scheduler.note_pressure_poll("hard")
        assert rig.scheduler.request_park_prefill() is False
    finally:
        rig.close()


# --- Engine bulk abort -------------------------------------------------------------


def test_is_request_parked_reads_the_request_flag():
    """EngineCore.abort_all_requests uses it to spare parked requests."""
    from omlx.scheduler import Scheduler

    scheduler = SimpleNamespace(
        requests={"p": SimpleNamespace(parked=True), "l": SimpleNamespace(parked=False)}
    )
    assert Scheduler.is_request_parked(scheduler, "p") is True
    assert Scheduler.is_request_parked(scheduler, "l") is False
    assert Scheduler.is_request_parked(scheduler, "missing") is False
