# SPDX-License-Identifier: Apache-2.0
"""Test-only knob: force decode pauses at fixed output offsets.

``OMLX_DEBUG_FORCE_DECODE_PAUSE_AT`` (comma-separated output-token offsets,
off by default) pauses a decoding request through the normal decode-pause
path (exact prefix stored, row removed, request re-queued parked) when its
output reaches each offset, and resumes it as soon as possible. On a real
model this isolates the pause from batch-size numerics: a B=1 run with forced
pauses is compared with a B=1 run without, everything else equal.

The rig is the real scheduler on a tiny Qwen3.5 hybrid with real paged-SSD
stores (4-token blocks; the prompt is 24 tokens, so output offset 4 puts the
pause on a block edge).
"""

import logging

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park

import omlx.scheduler as scheduler_module

ENV = "OMLX_DEBUG_FORCE_DECODE_PAUSE_AT"
MAX_TOKENS = 40


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ()),
        ("", ()),
        ("100,1000", (100, 1000)),
        (" 20, 5 ,20", (5, 20)),
    ],
)
def test_offsets_parse(raw, expected):
    assert scheduler_module._parse_forced_decode_pause_offsets(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "0", "-3", "5,x", "5,,6"])
def test_invalid_offsets_turn_the_knob_off_with_a_warning(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="omlx.scheduler"):
        assert scheduler_module._parse_forced_decode_pause_offsets(raw) == ()
    assert ENV in caplog.text


def _run(rig, max_tokens=MAX_TOKENS):
    request = dp.submit(rig, "a", dp.PROMPT, max_tokens)
    outputs = park.run_to_completion(rig, [request], limit=8 * max_tokens)
    return request, outputs


def _final(outputs):
    (final,) = [o for o in outputs if o.request_id == "a" and o.finished]
    return final


def test_forced_pauses_give_the_uninterrupted_tokens(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising=False)
    rig = dp.make_rig(tmp_path / "plain")
    try:
        plain, plain_outputs = _run(rig)
        assert plain.decode_pause_count == 0
    finally:
        rig.close()

    # A block edge (24 + 4), mid-block, and back-to-back (20 then 21).
    monkeypatch.setenv(ENV, "4,19,21")
    rig = dp.make_rig(tmp_path / "forced")
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            forced, forced_outputs = _run(rig)
        stats = rig.scheduler.get_park_stats()
    finally:
        rig.close()

    expected = list(plain.output_token_ids)
    assert len(expected) == MAX_TOKENS
    assert list(forced.output_token_ids) == expected
    assert dp.streamed_tokens(forced_outputs, "a") == expected
    assert _final(forced_outputs).finish_reason == _final(plain_outputs).finish_reason
    assert _final(forced_outputs).output_text == _final(plain_outputs).output_text
    assert not [o for o in forced_outputs if o.error]
    assert forced.decode_pause_count == 3
    assert stats["decode_pauses"] == stats["decode_resumes"] == 3
    assert "forcing a decode pause of a at output token 4" in caplog.text
    assert "Resumed decoding request a" in caplog.text


def test_the_knob_unset_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    rig = dp.make_rig(tmp_path / "unset")
    try:
        assert rig.scheduler._forced_pause_offsets == ()
        request, outputs = _run(rig)
        stats = rig.scheduler.get_park_stats()
    finally:
        rig.close()
    assert request.decode_pause_count == 0
    assert stats["decode_pauses"] == 0
    assert len(request.output_token_ids) == MAX_TOKENS
    assert not [o for o in outputs if o.error]


def test_a_forced_pause_resumes_at_once_even_when_the_memory_hold_waits(
    tmp_path, monkeypatch
):
    """The forced pause is not memory driven: its resume skips the memory
    hold, so the run stays comparable to the run without pauses."""
    monkeypatch.setenv(ENV, "4")
    rig = dp.make_rig(tmp_path / "hold")
    try:
        scheduler = rig.scheduler
        scheduler._park_hold_applies = lambda head: True  # memory says wait
        request = dp.submit(rig, "a", dp.PROMPT, MAX_TOKENS)
        outputs = []
        dp.step_until_outputs(rig, request, 4, outputs)
        outputs.extend(park.step(rig))  # the pause runs at step start
        assert request.decode_pause_count == 1
        outputs.extend(park.step(rig, 2))
        assert "a" in scheduler.running
        assert not request.parked
    finally:
        rig.close()


def test_the_knob_needs_the_park_feature(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "4")
    rig = dp.make_rig(tmp_path / "no-park", park_enabled=False)
    try:
        request, outputs = _run(rig)
    finally:
        rig.close()
    assert request.decode_pause_count == 0
    assert len(request.output_token_ids) == MAX_TOKENS
