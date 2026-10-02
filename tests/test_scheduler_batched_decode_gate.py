# SPDX-License-Identifier: Apache-2.0
"""Serialize decode when the combined context of a would-be batch is too long.

Two long sequences decoding together merge into a 2-row batch that cannot use
the fused single-row TurboQuant attention kernels and dequantizes the whole KV
every step. ``max_batched_decode_context_tokens`` keeps such a request out of
the batch (it waits for admission) so it later decodes alone. The tests drive
the real scheduler step loop on a tiny hybrid model.
"""

import logging

import test_scheduler_prefill_progress_commit as commit_rig
from test_scheduler_prefill_progress_commit import BLOCK, _Rig
from test_scheduler_park import (
    PROMPT_A,
    PROMPT_B,
    finish_ok_polls,
    prefilling_ids,
    step,
    step_until,
    submit,
    waiting_ids,
)

gated = commit_rig.gated

PROMPT_TOKENS = len(PROMPT_A)  # 24 each; two of them make a 48-token context
TOKENS = 6


def make_rig(tmp_path, tokenizer, *, limit, **overrides):
    rig = _Rig(
        tmp_path,
        tokenizer,
        split=False,
        commit_blocks=0,
        max_batched_decode_context_tokens=limit,
        **overrides,
    )
    return rig


def admitted(rig):
    return len(rig.scheduler.running) + len(rig.scheduler.prefilling)


def drive(rig, requests, limit=200):
    """Step to completion; return the peak number of concurrently admitted."""
    peak = 0
    for _ in range(limit):
        if all(r.is_finished() or r.request_id not in rig.scheduler.requests
               for r in requests):
            break
        step(rig)
        peak = max(peak, admitted(rig))
    for future in list(rig.scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.scheduler._drain_pending_async_removes()
    return peak


def solo_tokens(tmp_path, tokenizer, prompt):
    rig = make_rig(tmp_path / "solo", tokenizer, limit=0)
    try:
        request = submit(rig, "solo", prompt, arrival=1.0, max_tokens=TOKENS)
        drive(rig, [request])
        return list(request.output_token_ids)
    finally:
        rig.close()


def test_combined_context_over_the_limit_never_decodes_in_one_batch(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=2 * PROMPT_TOKENS - 1)
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        step_until(rig, lambda: admitted(rig) >= 1)
        step(rig)
        # The second request is held in the queue, not prefilling beside it.
        assert waiting_ids(rig) == ["b"]
        assert admitted(rig) == 1

        peak = drive(rig, [first, second])
        assert peak == 1
        assert first.is_finished() and second.is_finished()
    finally:
        rig.close()


def test_held_request_output_matches_an_uninterrupted_run(tmp_path, mock_tokenizer):
    expected = solo_tokens(tmp_path, mock_tokenizer, PROMPT_B)
    rig = make_rig(tmp_path / "held", mock_tokenizer, limit=2 * PROMPT_TOKENS - 1)
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        drive(rig, [first, second])
        assert len(expected) == TOKENS
        assert list(second.output_token_ids) == expected
    finally:
        rig.close()


def test_below_the_limit_batching_is_unchanged(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, limit=10 * PROMPT_TOKENS)
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        assert drive(rig, [first, second]) == 2
    finally:
        rig.close()


def test_limit_zero_is_identical_to_before(tmp_path, mock_tokenizer, caplog):
    rig = make_rig(tmp_path, mock_tokenizer, limit=0)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
            second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
            assert drive(rig, [first, second]) == 2
        assert "from batched decode" not in caplog.text
    finally:
        rig.close()


def test_generated_tokens_count_toward_the_combined_context(
    tmp_path, mock_tokenizer
):
    """The running row's context grows as it decodes, so a request that fit
    at its start is held once the sum would pass the limit."""
    rig = make_rig(tmp_path, mock_tokenizer, limit=2 * PROMPT_TOKENS + 1)
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=40)
        step_until(rig, lambda: "a" in rig.scheduler.running and
                   first.num_output_tokens >= 2)
        assert not first.is_finished()
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        step(rig)
        # 48 prompt tokens alone fit the limit; the generated ones tip it over.
        assert waiting_ids(rig) == ["b"]
        assert drive(rig, [first, second]) == 1
    finally:
        rig.close()


def test_hold_and_release_are_logged_once_per_request(
    tmp_path, mock_tokenizer, caplog
):
    limit = 2 * PROMPT_TOKENS - 1
    rig = make_rig(tmp_path, mock_tokenizer, limit=limit)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
            second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
            drive(rig, [first, second])
        holds = [r.getMessage() for r in caplog.records
                 if "Holding b from batched decode" in r.getMessage()]
        releases = [r.getMessage() for r in caplog.records
                    if "Releasing b to decode" in r.getMessage()]
        assert len(holds) == 1 and len(releases) == 1
        assert f"combined context {2 * PROMPT_TOKENS} > limit {limit}" in holds[0]
        assert "single-row decode keeps fused kernels" in holds[0]
        assert not any("Holding a" in r.getMessage() for r in caplog.records)
    finally:
        rig.close()


def test_waiting_requests_stay_in_arrival_order(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, limit=2 * PROMPT_TOKENS - 1)
    try:
        reqs = [
            submit(rig, rid, prompt, arrival=float(i), max_tokens=TOKENS)
            for i, (rid, prompt) in enumerate(
                [("a", PROMPT_A), ("b", PROMPT_B), ("c", PROMPT_A)]
            )
        ]
        finished = []
        for _ in range(300):
            step(rig)
            for r in reqs:
                if r.is_finished() and r.request_id not in finished:
                    finished.append(r.request_id)
            if len(finished) == 3:
                break
        assert finished == ["a", "b", "c"]
    finally:
        rig.close()


# --- No deadlock with the other admission gates --------------------------------


def test_max_concurrent_two_still_serializes_and_drains(tmp_path, mock_tokenizer):
    rig = make_rig(
        tmp_path, mock_tokenizer, limit=2 * PROMPT_TOKENS - 1, max_num_seqs=2
    )
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        assert drive(rig, [first, second]) == 1
    finally:
        rig.close()


def test_a_parked_sole_prefill_resumes_with_a_request_held_behind_it(
    tmp_path, mock_tokenizer
):
    rig = _park_rig(tmp_path, mock_tokenizer)
    try:
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        step_until(rig, lambda: prefilling_ids(rig) == ["a"])
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        rig.scheduler._park_hold_applies = lambda head: True  # isolate the park
        assert rig.scheduler.request_park_prefill() is True
        step(rig)
        assert first.parked and waiting_ids(rig) == ["a", "b"]
        del rig.scheduler._park_hold_applies  # the real hold gate again
        finish_ok_polls(rig, 2)
        assert drive(rig, [first, second]) == 1
        assert first.is_finished() and second.is_finished()
    finally:
        rig.close()


def _park_rig(tmp_path, tokenizer):
    rig = _Rig(
        tmp_path,
        tokenizer,
        split=False,
        commit_blocks=1,
        memory_pressure_park=True,
        park_resume_ok_polls=1,
        max_batched_decode_context_tokens=2 * PROMPT_TOKENS - 1,
    )
    rig.scheduler.note_pressure_poll("soft")
    return rig


def test_a_held_request_behind_a_deferred_teardown_does_not_deadlock(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=2 * PROMPT_TOKENS - 1)
    try:
        scheduler = rig.scheduler
        first = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        second = submit(rig, "b", PROMPT_B, arrival=2.0, max_tokens=TOKENS)
        # "b" pretends a previous run of its id is still tearing down; "a"
        # alone is admitted and "b" must neither block it nor be lost.
        pending = {"b"}
        original = scheduler._teardown_pending
        scheduler._teardown_pending = lambda rid: rid in pending or original(rid)
        step_until(rig, lambda: admitted(rig) >= 1)
        step(rig, 3)
        assert waiting_ids(rig) == ["b"]
        pending.clear()
        assert drive(rig, [first, second]) == 1
        assert second.is_finished()
    finally:
        rig.close()
