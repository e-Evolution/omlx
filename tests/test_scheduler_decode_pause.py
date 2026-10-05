# SPDX-License-Identifier: Apache-2.0
"""Pause a decoding request under memory pressure and resume it exactly.

The tests drive the real scheduler step loop on a small initialized Qwen3.5
hybrid model (GatedDeltaNet + full attention) against real paged-SSD,
boundary-snapshot and prefix stores (the ``_Rig`` of the park suite). A
paused row emits its pending token, stores its exact KV under a request
domain, leaves the batch and waits parked with its client stream open.
"""

import logging

import pytest
import test_scheduler_park as park
from conftest import MockTokenizer

import omlx.scheduler as scheduler_module
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import _BoundarySnapshotProvider

PROMPT = park.PROMPT_B  # 24 tokens; every id is below the tiny vocabulary
OTHER = park.PROMPT_C
# Enough paged blocks for prompt plus several hundred generated tokens.
MAX_BLOCKS = 256


def _tokenizer():
    """EOS outside the tiny model's vocabulary: decode runs to max_tokens."""
    tokenizer = MockTokenizer()
    tokenizer.eos_token_id = 99
    return tokenizer


def make_rig(tmp_path, *, split=False, exact_writer=True, park_enabled=True):
    rig = park._Rig(
        tmp_path,
        _tokenizer(),
        split=split,
        commit_blocks=1,
        max_blocks=MAX_BLOCKS,
        memory_pressure_park=park_enabled,
        max_parks_per_request=3,
        park_resume_ok_polls=3,
    )
    if split and exact_writer:
        rig.prefix.set_exact_gdn_checkpoint_writer(
            lambda **kwargs: _BoundarySnapshotProvider.stage_and_commit(
                store=rig.boundary, paged_ssd_manager=rig.ssd, **kwargs
            )
        )
    rig.scheduler.note_pressure_poll("soft")
    return rig


def submit(rig, request_id, tokens, max_tokens, **sampling):
    request = Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(
            max_tokens=max_tokens, temperature=0, **sampling
        ),
        arrival_time=float(len(rig.scheduler.requests) + 1),
    )
    rig.scheduler.add_request(request)
    return request


def step_until_outputs(rig, request, count, outputs, limit=1000):
    for _ in range(limit):
        if len(request.output_token_ids) >= count:
            return
        outputs.extend(park.step(rig))
    raise AssertionError("output count not reached")


def streamed_tokens(outputs, request_id):
    return [t for o in outputs if o.request_id == request_id for t in o.new_token_ids]


def baseline(tmp_path, max_tokens, *, second=False, **sampling):
    """Token ids of an uninterrupted greedy run (optionally beside a second row)."""
    rig = make_rig(tmp_path / "baseline", park_enabled=False)
    try:
        request = submit(rig, "a", PROMPT, max_tokens, **sampling)
        rows = [request]
        if second:
            rows.append(submit(rig, "b", OTHER, max_tokens))
        park.run_to_completion(rig, rows, limit=4 * max_tokens)
        return [list(r.output_token_ids) for r in rows]
    finally:
        rig.close()


# --- Pause primitive ------------------------------------------------------------


def test_pause_emits_the_pending_token_once_and_parks_the_row(tmp_path, caplog):
    (expected,) = baseline(tmp_path, 40)
    rig = make_rig(tmp_path / "pause")
    try:
        scheduler = rig.scheduler
        scheduler._park_hold_applies = lambda head: True  # isolate the pause
        request = submit(rig, "a", PROMPT, 40)
        outputs = []
        step_until_outputs(rig, request, 3, outputs)
        uid = scheduler.request_id_to_uid["a"]

        assert scheduler.request_decode_pause("a") is True
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            pause_outputs = park.step(rig)
        outputs.extend(pause_outputs)

        # The pending token reached the client exactly once, then the row left.
        assert streamed_tokens(pause_outputs, "a") == [expected[3]]
        assert list(request.output_token_ids) == expected[:4]
        assert streamed_tokens(outputs, "a") == expected[:4]
        assert not [o for o in outputs if o.finished or o.error]
        assert request.parked and request.park_count == 1
        assert request.status == RequestStatus.WAITING
        assert park.waiting_ids(rig) == ["a"]
        assert "a" not in scheduler.running
        assert "a" not in scheduler.request_id_to_uid
        assert uid not in scheduler.uid_to_request_id
        assert uid not in scheduler.batch_generator._generation_batch.uids
        assert "a" in scheduler.requests  # the client stream stays open

        record = scheduler._decode_resumes["a"]
        assert record.tokens == PROMPT + expected[:3]  # the KV holds these
        assert record.kickoff == expected[3]  # emitted, not yet in the KV
        assert record.remaining_max_tokens == 40 - 4
        assert scheduler._park_pins["a"].hashes
        assert rig.ssd.pinned_block_count == len(scheduler._park_pins["a"].hashes)
        assert scheduler.get_park_stats()["decode_pauses"] == 1
        assert any(
            "Paused decoding request a at 28 tokens (prompt 24 + output 4), "
            "stored exact prefix" in r.getMessage()
            for r in caplog.records
        )
    finally:
        rig.close()


def test_a_pending_token_that_reaches_max_tokens_finishes_without_a_pause(tmp_path):
    (expected,) = baseline(tmp_path, 5)
    rig = make_rig(tmp_path / "pause")
    try:
        scheduler = rig.scheduler
        request = submit(rig, "a", PROMPT, 5)
        outputs = []
        step_until_outputs(rig, request, 4, outputs)
        assert scheduler.request_decode_pause("a") is True
        outputs.extend(park.run_to_completion(rig, [request]))

        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert list(request.output_token_ids) == expected
        assert streamed_tokens(outputs, "a") == expected
        assert not request.parked and request.park_count == 0
        assert "a" not in scheduler._decode_resumes
        assert rig.ssd.pinned_block_count == 0
    finally:
        rig.close()


def test_a_pending_stop_token_finishes_without_a_pause(tmp_path):
    (free_run,) = baseline(tmp_path, 40)
    # The first token that has not appeared before becomes the stop token.
    index = next(i for i in range(2, 40) if free_run[i] not in free_run[:i])
    stop = free_run[index]
    rig = make_rig(tmp_path / "pause")
    try:
        scheduler = rig.scheduler
        request = submit(rig, "a", PROMPT, 40, stop_token_ids=[stop])
        outputs = []
        step_until_outputs(rig, request, index, outputs)
        assert scheduler.request_decode_pause("a") is True
        outputs.extend(park.run_to_completion(rig, [request]))

        assert request.status == RequestStatus.FINISHED_STOPPED
        assert list(request.output_token_ids) == free_run[:index]
        assert streamed_tokens(outputs, "a") == free_run[:index]
        assert not request.parked and "a" not in scheduler._decode_resumes
        assert rig.ssd.pinned_block_count == 0
    finally:
        rig.close()


def _decoding(rig, request_id="a", max_tokens=40):
    request = submit(rig, request_id, PROMPT, max_tokens)
    step_until_outputs(rig, request, 2, [])
    return request


def test_a_decoding_row_is_pausable_when_nothing_excludes_it(tmp_path):
    rig = make_rig(tmp_path)
    try:
        request = _decoding(rig)
        assert rig.scheduler._park_exclusion_reason(request) is None
    finally:
        rig.close()


@pytest.mark.parametrize(
    "case, expected",
    [
        ("specprefill", "SpecPrefill"),
        ("dflash", "DFlash"),
        ("mtp", "MTP"),
        ("vlm_mtp", "VLM-MTP"),
        ("no_ssd", "no paged SSD cache"),
        ("budget", "park budget spent"),
    ],
)
def test_v1_exclusions_name_why_a_decoding_row_cannot_pause(
    tmp_path, monkeypatch, case, expected
):
    rig = make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        request = _decoding(rig)
        if case == "specprefill":
            request.specprefill_indices = [0]
        elif case == "dflash":
            monkeypatch.setattr(
                scheduler_module, "_block_drafter_for", lambda model: object()
            )
        elif case == "mtp":
            scheduler.batch_generator._generation_batch._omlx_mtp_state = object()
        elif case == "vlm_mtp":
            uid = scheduler.request_id_to_uid["a"]
            scheduler.request_id_to_uid["a"] = -7
        elif case == "no_ssd":
            monkeypatch.setattr(scheduler, "paged_ssd_cache_manager", None)
        elif case == "budget":
            request.park_count = 3
        reason = scheduler._park_exclusion_reason(request)
        assert reason is not None and expected in reason
        assert scheduler.request_decode_pause("a") is False
        if case == "mtp":
            del scheduler.batch_generator._generation_batch._omlx_mtp_state
        if case == "vlm_mtp":
            scheduler.request_id_to_uid["a"] = uid
    finally:
        rig.close()


def test_split_gdn_without_an_exact_sidecar_writer_is_excluded(tmp_path):
    rig = make_rig(tmp_path, split=True, exact_writer=False)
    try:
        request = _decoding(rig)
        reason = rig.scheduler._park_exclusion_reason(request)
        assert reason is not None and "split-GDN" in reason
        assert rig.scheduler.request_decode_pause("a") is False
    finally:
        rig.close()


def test_decode_pause_requires_the_park_feature(tmp_path):
    rig = make_rig(tmp_path, park_enabled=False)
    try:
        _decoding(rig)
        assert rig.scheduler.request_decode_pause("a") is False
    finally:
        rig.close()


# --- Paused requests: abort, lost chain ---------------------------------------------


def _paused(rig, request_id="a", at=3, max_tokens=40):
    scheduler = rig.scheduler
    scheduler._park_hold_applies = lambda head: True  # hold it parked
    request = submit(rig, request_id, PROMPT, max_tokens)
    outputs = []
    step_until_outputs(rig, request, at, outputs)
    assert scheduler.request_decode_pause(request_id) is True
    outputs.extend(park.step(rig))
    assert request.parked and request_id in scheduler._decode_resumes
    return request, outputs


def test_abort_while_paused_leaves_no_pins_and_no_resume_record(tmp_path):
    rig = make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        _paused(rig)
        assert rig.ssd.pinned_block_count > 0
        assert scheduler.abort_request("a")
        park.step(rig, 2)
        assert "a" not in scheduler.requests
        assert "a" not in scheduler._decode_resumes
        assert "a" not in scheduler._park_pins
        assert rig.ssd.pinned_block_count == 0
        assert park.waiting_ids(rig) == []
    finally:
        rig.close()


def test_a_lost_exact_chain_ends_the_request_with_a_structured_error(tmp_path):
    rig = make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        request, _ = _paused(rig)
        # Lose the chain: drop the pins, then every SSD file and pool entry.
        hashes = scheduler._park_pins["a"].hashes
        scheduler._unpin_parked("a")
        rig.ssd._evict_tracked_until_size(0)
        for block_id, block in list(rig.paged.allocated_blocks.items()):
            if block.block_hash in set(hashes):
                rig.paged.allocated_blocks.pop(block_id)
                rig.paged.cached_block_hash_to_block.pop(block.block_hash, block_id)

        del scheduler._park_hold_applies  # resume as soon as admission allows
        outputs = park.step(rig)
        errors = [o for o in outputs if o.request_id == "a" and o.error]
        assert len(errors) == 1
        error = errors[0]
        assert error.finished and error.finish_reason == "error"
        assert error.error_code == "decode_resume_lost"
        assert error.error_metadata["request_id"] == "a"
        assert error.error_metadata["output_tokens"] == 4
        assert "a" not in scheduler.requests
        assert "a" not in scheduler._decode_resumes
        assert rig.ssd.pinned_block_count == 0
        assert scheduler.get_park_stats()["decode_resume_lost"] == 1
        assert not park.waiting_ids(rig) and not scheduler.running
    finally:
        rig.close()
