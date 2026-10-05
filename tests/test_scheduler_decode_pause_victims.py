# SPDX-License-Identifier: Apache-2.0
"""Decode-pause victims: who is paused, how the enforcer follows a pause, and
what a pause that fails leaves behind.

Runs the real scheduler step loop on the park suite's ``_Rig`` (tiny Qwen3.5
hybrid model, real paged-SSD stores). A pause that fails before the row
leaves the batch must leave it decoding exactly as if nothing was asked, with
its stored chain unpinned and deleted, and mark it so the enforcer falls back
to the abort instead of choosing it again.
"""

import logging

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park

from omlx.adapter.output_parser import (
    OutputParserFactory,
    OutputParserFinalizeResult,
    OutputParserTokenResult,
)
from omlx.request import Request, SamplingParams

PROMPT, OTHER = dp.PROMPT, dp.OTHER
THIRD = park.PROMPT_D


def submit(rig, request_id, tokens, max_tokens, arrival):
    request = Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        arrival_time=arrival,
    )
    rig.scheduler.add_request(request)
    return request


def step_until_decoding(rig, requests, count=3, limit=400):
    """Step until every row has ``count`` outputs and is pausable."""
    scheduler = rig.scheduler
    for _ in range(limit):
        if all(
            len(r.output_token_ids) >= count
            and r.request_id in scheduler.running
            and scheduler._park_exclusion_reason(r) is None
            for r in requests
        ):
            return
        park.step(rig)
    raise AssertionError("rows are not all decoding")


# --- Victim choice -----------------------------------------------------------------


def _three_decoding(rig, prompts=(PROMPT, OTHER, THIRD), max_tokens=80):
    rows = [
        submit(rig, rid, tokens, max_tokens, arrival=float(i + 1))
        for i, (rid, tokens) in enumerate(zip("abc", prompts))
    ]
    step_until_decoding(rig, rows)
    return rows


def test_the_largest_live_kv_is_paused_and_the_oldest_is_protected(tmp_path):
    rig = dp.make_rig(tmp_path)
    try:
        # ``a`` is oldest and holds the largest KV (a 120-token prompt).
        a, b, c = _three_decoding(
            rig, prompts=(PROMPT * 5, OTHER, THIRD), max_tokens=200
        )
        # Equalize the output counts the prefill order skews, then give
        # ``b`` the larger KV of the two younger rows.
        for row in (a, b, c):
            row.output_token_ids = row.output_token_ids[:3]
        b.output_token_ids = b.output_token_ids + [3, 3]
        assert a.num_tokens > b.num_tokens > c.num_tokens
        victim = rig.scheduler.select_decode_pause_victim()
        assert victim.request_id == "b"
        assert victim.tokens == b.num_tokens
        assert victim.kept == 2
    finally:
        rig.close()


def test_equal_kv_goes_to_the_most_recent_arrival(tmp_path):
    rig = dp.make_rig(tmp_path)
    try:
        a, b, c = _three_decoding(rig, prompts=(PROMPT, PROMPT, PROMPT))
        # Same prompt length; equalize the output counts the prefill order skews.
        for row in (a, b, c):
            row.output_token_ids = row.output_token_ids[:3]
        assert rig.scheduler.select_decode_pause_victim().request_id == "c"
    finally:
        rig.close()


def test_the_oldest_is_paused_when_it_is_the_only_candidate(tmp_path):
    rig = dp.make_rig(tmp_path)
    try:
        (a,) = [submit(rig, "a", PROMPT, 80, arrival=1.0)]
        step_until_decoding(rig, [a])
        assert rig.scheduler.select_decode_pause_victim().request_id == "a"
    finally:
        rig.close()


def test_excluded_and_already_pending_rows_are_never_chosen(tmp_path):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        a, b, c = _three_decoding(rig)
        c._decode_pause_failed = "test"
        assert scheduler.request_decode_pause("b") is True
        # ``a`` is the oldest, but it is the only candidate left.
        assert scheduler.select_decode_pause_victim().request_id == "a"
        scheduler._pending_abort_ids.add("a")
        assert scheduler.select_decode_pause_victim() is None
    finally:
        rig.close()


def test_nothing_is_chosen_when_park_is_disabled(tmp_path):
    rig = dp.make_rig(tmp_path, park_enabled=False)
    try:
        _three_decoding(rig)
        assert rig.scheduler.select_decode_pause_victim() is None
    finally:
        rig.close()


# --- Status the enforcer drains on ---------------------------------------------------


def test_the_status_is_pending_until_the_pause_ran_and_its_chain_is_written(
    tmp_path, monkeypatch
):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        scheduler._park_hold_applies = lambda head: True  # keep it parked
        (a,) = [submit(rig, "a", PROMPT, 80, arrival=1.0)]
        step_until_decoding(rig, [a])
        assert scheduler.decode_pause_status("a") == "done"  # nothing asked

        assert scheduler.request_decode_pause("a") is True
        assert scheduler.decode_pause_status("a") == "pending"

        writes = {"pending": True}
        monkeypatch.setattr(
            rig.ssd, "has_pending_writes", lambda hashes: writes["pending"]
        )
        park.step(rig)
        assert a.parked
        assert scheduler.decode_pause_status("a") == "pending"  # still writing
        writes["pending"] = False
        assert scheduler.decode_pause_status("a") == "done"
    finally:
        rig.close()


def test_the_status_stays_pending_while_the_pause_runs(tmp_path, monkeypatch):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        (a,) = [submit(rig, "a", PROMPT, 80, arrival=1.0)]
        step_until_decoding(rig, [a])
        seen = []
        original = scheduler._pause_decoding_request

        def observe(request, output):
            seen.append(scheduler.decode_pause_status("a"))
            return original(request, output)

        monkeypatch.setattr(scheduler, "_pause_decoding_request", observe)
        assert scheduler.request_decode_pause("a") is True
        park.step(rig)
        assert seen == ["pending"]
        assert not scheduler._pending_decode_pauses
    finally:
        rig.close()


# --- A pause that fails --------------------------------------------------------------


def _ids_of(rig, table):
    hashes = []
    for block_id in table.block_ids:
        block = rig.paged.allocated_blocks.get(block_id)
        hashes.append(bytes(block.block_hash))
    return hashes


def _fail_extract(rig, monkeypatch, store):
    def boom(uids):
        raise RuntimeError("extract exploded")

    monkeypatch.setattr(rig.scheduler.batch_generator, "extract_cache", boom)


def _fail_after_store(rig, monkeypatch, store):
    original = rig.prefix.store_exact_prefix

    def store_then_fail(*args, **kwargs):
        table = original(*args, **kwargs)
        store["hashes"] = _ids_of(rig, table)
        raise OSError("disk vanished mid-store")

    monkeypatch.setattr(rig.prefix, "store_exact_prefix", store_then_fail)


def _fail_pin(rig, monkeypatch, store):
    original = rig.prefix.store_exact_prefix

    def store_and_record(*args, **kwargs):
        table = original(*args, **kwargs)
        store["hashes"] = _ids_of(rig, table)
        return table

    monkeypatch.setattr(rig.prefix, "store_exact_prefix", store_and_record)
    monkeypatch.setattr(rig.scheduler, "_pin_exact_chain", lambda rid, table: False)


@pytest.mark.parametrize(
    "inject, reason",
    [
        (_fail_extract, "RuntimeError: extract exploded"),
        (_fail_after_store, "OSError: disk vanished mid-store"),
        (_fail_pin, "could not be stored and pinned"),
    ],
    ids=["extract", "store", "pin"],
)
def test_a_pause_failing_before_the_row_leaves_keeps_it_decoding_exactly(
    tmp_path, monkeypatch, caplog, inject, reason
):
    (expected,) = dp.baseline(tmp_path, 40)
    rig = dp.make_rig(tmp_path / "paused")
    try:
        scheduler = rig.scheduler
        request = submit(rig, "a", PROMPT, 40, arrival=1.0)
        outputs = []
        dp.step_until_outputs(rig, request, 5, outputs)
        store = {}
        inject(rig, monkeypatch, store)

        assert scheduler.request_decode_pause("a") is True
        with caplog.at_level(logging.WARNING, logger="omlx.scheduler"):
            outputs.extend(park.step(rig))
        monkeypatch.undo()

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "Decode pause declined for a" in r.getMessage()
        ]
        assert len(warnings) == 1 and reason in warnings[0]
        assert "the row keeps decoding" in warnings[0]
        assert "a" in scheduler.running and not request.parked
        assert request.park_count == 0
        assert "a" not in scheduler._decode_resumes
        assert not scheduler._park_pins and rig.ssd.pinned_block_count == 0
        for block_hash in store.get("hashes", ()):
            assert not rig.ssd.has_block(block_hash)
        # Never chosen again: the enforcer falls back to the abort.
        assert "an earlier decode pause failed" in scheduler._park_exclusion_reason(
            request
        )
        assert scheduler.decode_pause_status("a") == "declined"
        assert scheduler.select_decode_pause_victim() is None

        outputs.extend(park.run_to_completion(rig, [request]))
        assert list(request.output_token_ids) == expected
        assert dp.streamed_tokens(outputs, "a") == expected
        assert not [o for o in outputs if o.error]
    finally:
        rig.close()


def test_a_pause_failing_after_the_pending_token_streamed_ends_with_an_error(
    tmp_path, monkeypatch, caplog
):
    (expected,) = dp.baseline(tmp_path, 40)
    rig = dp.make_rig(tmp_path / "paused")
    try:
        scheduler = rig.scheduler
        request = submit(rig, "a", PROMPT, 40, arrival=1.0)
        outputs = []
        dp.step_until_outputs(rig, request, 5, outputs)

        def boom(request_id, uid):
            raise RuntimeError("batch filter exploded")

        monkeypatch.setattr(scheduler, "_remove_decoding_row", boom)
        assert scheduler.request_decode_pause("a") is True
        with caplog.at_level(logging.ERROR, logger="omlx.scheduler"):
            step_outputs = park.step(rig)
        monkeypatch.undo()
        outputs.extend(step_outputs)

        # The pending token reached the client once, then the error.
        assert dp.streamed_tokens(outputs, "a") == expected[:6]
        (error,) = [o for o in outputs if o.error]
        assert error.finished and error.finish_reason == "error"
        assert error.error_code == "decode_pause_failed"
        assert "batch filter exploded" in error.error
        assert "a" not in scheduler.running and "a" not in scheduler.requests
        assert not scheduler._park_pins and rig.ssd.pinned_block_count == 0
        assert "a" not in scheduler._decode_resumes
        assert any(
            "after its pending token was streamed" in r.getMessage()
            for r in caplog.records
        )
        park.step(rig, 2)  # the batch keeps working without the row
    finally:
        rig.close()


class _StopOnDemand:
    """A parser session that stops the request once ``armed`` is set."""

    armed = False

    def process_token(self, token_id):
        return OutputParserTokenResult(
            stream_text=f"<{token_id}>", is_stop=_StopOnDemand.armed
        )

    def finalize(self):
        return OutputParserFinalizeResult()


def test_a_parser_stop_on_the_pending_token_deletes_the_chain_it_stored(
    tmp_path, monkeypatch
):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        scheduler._output_parser_factory = OutputParserFactory(
            kind="test-stop", create_session=lambda tokenizer: _StopOnDemand()
        )
        monkeypatch.setattr(_StopOnDemand, "armed", False)
        request = submit(rig, "a", PROMPT, 40, arrival=1.0)
        dp.step_until_outputs(rig, request, 5, [])
        store = {}
        original = rig.prefix.store_exact_prefix

        def store_and_record(*args, **kwargs):
            table = original(*args, **kwargs)
            store["hashes"] = _ids_of(rig, table)
            return table

        monkeypatch.setattr(rig.prefix, "store_exact_prefix", store_and_record)
        _StopOnDemand.armed = True
        assert scheduler.request_decode_pause("a") is True
        outputs = park.step(rig)

        (final,) = [o for o in outputs if o.request_id == "a" and o.finished]
        assert final.finish_reason == "stop" and not final.error
        assert store["hashes"]
        for block_hash in store["hashes"]:
            assert not rig.ssd.has_block(block_hash)
        assert not scheduler._park_pins and rig.ssd.pinned_block_count == 0
        assert request.park_count == 0
    finally:
        rig.close()
