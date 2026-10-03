# SPDX-License-Identifier: Apache-2.0
"""Backfill and aging for the batched-decode admission gate (T12b).

``max_batched_decode_context_tokens`` holds a long request out of the decode
batch. The held head must not also block shorter requests behind it whose
combined context still fits: those are admitted past it (EASY backfill), the
held request keeps its place at the front, and once it has waited longer than
``batched_decode_hold_max_s`` backfill stops so it can never starve.
"""

import logging

import test_scheduler_prefill_progress_commit as commit_rig  # noqa: F401
from test_scheduler_batched_decode_gate import (
    PROMPT_TOKENS,
    TOKENS,
    admitted,
    drive,
    make_rig,
)
from test_scheduler_park import (
    PROMPT_A,
    PROMPT_B,
    PROMPT_C,
    PROMPT_D,
    step,
    step_until,
    submit,
    waiting_ids,
)

LONG = PROMPT_C + PROMPT_D  # 48 tokens
# a (24 running) + LONG (48) = 72 > 60 -> held; a + short (24) = 48 <= 60 fits;
# a + short + short = 72 > 60 does not.
LIMIT = 60
RUNNING_TOKENS = 80  # keeps "a" decoding while the queue is exercised


def _running_a(rig):
    submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=RUNNING_TOKENS)
    step_until(
        rig,
        lambda: "a" in rig.scheduler.running
        and rig.scheduler.running["a"].num_output_tokens >= 1,
    )


def _queue(rig):
    long = submit(rig, "long", LONG, arrival=2.0, max_tokens=TOKENS)
    s1 = submit(rig, "s1", PROMPT_B, arrival=3.0, max_tokens=TOKENS)
    s2 = submit(rig, "s2", PROMPT_B, arrival=4.0, max_tokens=TOKENS)
    return long, s1, s2


def _drive_recording_admission(rig, requests, limit=500):
    """Step to completion; return {request_id: first step it was admitted}."""
    first_admitted = {}
    for n in range(limit):
        if all(r.is_finished() for r in requests):
            break
        step(rig)
        seen = {*rig.scheduler.running, *(r.request_id for r in rig.scheduler.prefilling)}
        for rid in seen:
            first_admitted.setdefault(rid, n)
        for r in requests:
            if r.is_finished():
                first_admitted.setdefault(r.request_id, n)
    return first_admitted


def _hold_logs(caplog, needle):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def test_short_request_behind_a_held_head_is_backfilled(
    tmp_path, mock_tokenizer, caplog
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=120)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _running_a(rig)
            _queue(rig)
            step(rig)
        running = set(rig.scheduler.running) | {
            r.request_id for r in rig.scheduler.prefilling
        }
        assert "s1" in running  # fits: admitted past the held head
        # The head keeps the front of the queue; the one that does not fit waits.
        assert waiting_ids(rig) == ["long", "s2"]
        assert _hold_logs(caplog, "Holding long from batched decode")
        assert _hold_logs(caplog, "Backfilling s1 past held long")
    finally:
        rig.close()


def test_backfill_never_exceeds_the_combined_limit(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=120)
    try:
        _running_a(rig)
        _queue(rig)
        for _ in range(4):
            step(rig)
            ctx = sum(
                r.num_tokens
                for r in [*rig.scheduler.running.values(), *rig.scheduler.prefilling]
            )
            assert admitted(rig) <= 2 or ctx <= LIMIT
    finally:
        rig.close()


def test_backfilled_requests_and_held_head_all_finish_in_order_of_room(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=120)
    try:
        a = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=TOKENS)
        step_until(rig, lambda: admitted(rig) >= 1)
        long, s1, s2 = _queue(rig)
        drive(rig, [a, long, s1, s2], limit=400)
        assert all(r.is_finished() for r in (a, long, s1, s2))
    finally:
        rig.close()


def test_aging_stops_backfill_and_the_head_is_admitted_when_room_appears(
    tmp_path, mock_tokenizer, caplog
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=30)
    try:
        scheduler = rig.scheduler
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _running_a(rig)
            long = submit(rig, "long", LONG, arrival=2.0, max_tokens=TOKENS)
            step(rig)
            assert waiting_ids(rig) == ["long"]
            # The head has now waited past the bound.
            scheduler._batched_decode_held_since["long"] -= 1000
            s1 = submit(rig, "s1", PROMPT_B, arrival=3.0, max_tokens=TOKENS)
            step(rig, 2)
            # s1 fits, but backfill is paused behind the aged head.
            assert "s1" not in scheduler.running
            assert waiting_ids(rig) == ["long", "s1"]
            assert _hold_logs(caplog, "Aging: long held")
            assert _hold_logs(caplog, "backfill paused")
            a = scheduler.requests["a"]
            order = _drive_recording_admission(rig, [a, long, s1])
        assert long.is_finished() and s1.is_finished()
        # The aged head is admitted before the request that arrived after it.
        assert order["long"] <= order["s1"]
    finally:
        rig.close()


def test_aging_message_is_logged_once_per_request(tmp_path, mock_tokenizer, caplog):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=30)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _running_a(rig)
            submit(rig, "long", LONG, arrival=2.0, max_tokens=TOKENS)
            step(rig)
            rig.scheduler._batched_decode_held_since["long"] -= 1000
            submit(rig, "s1", PROMPT_B, arrival=3.0, max_tokens=TOKENS)
            step(rig, 4)
        assert len(_hold_logs(caplog, "Aging: long held")) == 1
    finally:
        rig.close()


def test_hold_max_zero_keeps_the_old_head_of_line_behavior(
    tmp_path, mock_tokenizer, caplog
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=0)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _running_a(rig)
            _queue(rig)
            step(rig, 3)
        assert waiting_ids(rig) == ["long", "s1", "s2"]
        assert not _hold_logs(caplog, "Backfilling")
    finally:
        rig.close()


def test_backfill_respects_max_concurrent_requests(tmp_path, mock_tokenizer):
    rig = make_rig(
        tmp_path,
        mock_tokenizer,
        limit=LIMIT,
        batched_decode_hold_max_s=120,
        max_num_seqs=2,
    )
    try:
        a = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=RUNNING_TOKENS)
        step_until(rig, lambda: admitted(rig) >= 1)
        long, s1, s2 = _queue(rig)
        for _ in range(3):
            step(rig)
            assert admitted(rig) <= 2
        assert waiting_ids(rig)[0] == "long"
        drive(rig, [a, long, s1, s2], limit=600)
        assert all(r.is_finished() for r in (a, long, s1, s2))
    finally:
        rig.close()


def test_a_teardown_pending_request_and_a_held_head_do_not_block_backfill(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=120)
    try:
        scheduler = rig.scheduler
        pending = {"s1"}
        original = scheduler._teardown_pending
        scheduler._teardown_pending = lambda rid: rid in pending or original(rid)
        a = submit(rig, "a", PROMPT_A, arrival=1.0, max_tokens=RUNNING_TOKENS)
        step_until(rig, lambda: admitted(rig) >= 1)
        long, s1, s2 = _queue(rig)
        step(rig)
        # long: held by the gate; s1: teardown pending (set aside, not
        # blocking); s2 fits and is backfilled. Nothing is lost, order kept.
        assert waiting_ids(rig) == ["long", "s1"]
        assert "s2" in scheduler.running or any(
            r.request_id == "s2" for r in scheduler.prefilling
        )
        pending.clear()
        drive(rig, [a, long, s1, s2], limit=600)
        assert all(r.is_finished() for r in (a, long, s1, s2))
    finally:
        rig.close()


def test_a_parked_head_that_holds_admission_is_not_overtaken(
    tmp_path, mock_tokenizer
):
    rig = make_rig(tmp_path, mock_tokenizer, limit=LIMIT, batched_decode_hold_max_s=120)
    try:
        scheduler = rig.scheduler
        _running_a(rig)
        long, s1, s2 = _queue(rig)
        long.parked = True
        long.parked_at = 0.0
        scheduler._park_hold_applies = lambda head: True
        step(rig, 2)
        assert waiting_ids(rig) == ["long", "s1", "s2"]  # park hold: no overtake
        scheduler._park_hold_applies = lambda head: False
        step(rig)
        assert waiting_ids(rig)[0] == "long"  # gate-held, set aside, order kept
        assert "s1" in scheduler.running or any(
            r.request_id == "s1" for r in scheduler.prefilling
        )
    finally:
        rig.close()


def test_never_held_requests_stay_fifo(tmp_path, mock_tokenizer):
    rig = make_rig(tmp_path, mock_tokenizer, limit=0, batched_decode_hold_max_s=120)
    try:
        reqs = [
            submit(rig, rid, prompt, arrival=float(i), max_tokens=TOKENS)
            for i, (rid, prompt) in enumerate(
                [("a", PROMPT_A), ("b", PROMPT_B), ("c", PROMPT_C)]
            )
        ]
        order = _drive_recording_admission(rig, reqs, limit=300)
        starts = [order[r.request_id] for r in reqs]
        assert starts == sorted(starts)
    finally:
        rig.close()
