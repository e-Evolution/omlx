# SPDX-License-Identifier: Apache-2.0
"""A queued head that fits alone runs alone instead of failing.

Forced-pressure image prefill (46 GB): when the request before it finished,
the next image request reached the in-stream preflight with nothing admitted.
Its estimate fits alone (weights + KV + transient under the capacity line),
but the usage reading also held memory no admitted request owns (the queued
requests' vision embeddings and other residue), so ``current + KV +
transient`` crossed the admission line by under 1 GB and the request was
rejected by the prefill memory guard. The peak (~33 GB) was far under the
46 GB ceiling: a parked head in the same position resumes alone (solo),
so a fresh head does too. Nothing is admitted beside it and the enforcer
leaves it to the emergency brake; over the solo line the older rejection
still applies.

The tests drive the real scheduler loop on the tiny hybrid model.
"""

import logging

import pytest
from test_scheduler_capacity_gate import GB, _guarded_rig, gated  # noqa: F401
from test_scheduler_park import PROMPT_A, PROMPT_C, step, submit, waiting_ids

from omlx.exceptions import PrefillMemoryExceededError


def _lone_rig(tmp_path, tokenizer, monkeypatch):
    """Guard on, nothing admitted, no idle model to evict; the usage cell
    starts 1 GB over the admission line (residue nothing admitted owns)."""
    rig, usage = _guarded_rig(
        tmp_path, tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    scheduler = rig.scheduler
    scheduler._prefill_memory_guard = True
    monkeypatch.setattr(
        scheduler, "_raise_prefill_eviction_if_available", lambda **_kw: None
    )
    # As in production: the enforcer's hard watermark (ceiling x hard
    # threshold) sets the admission line well under the ceiling.
    scheduler._memory_hard_watermark_bytes = 80 * GB
    _fit, solo_line = scheduler._park_resume_lines()
    admission = scheduler._admission_limit_bytes()
    assert solo_line - admission > 2 * GB  # room between the two lines
    usage[0] = admission + 1 * GB
    return rig, usage, solo_line


def test_a_lone_head_that_fits_under_the_ceiling_runs_alone(
    tmp_path, mock_tokenizer, monkeypatch, caplog
):
    rig, _usage, _solo = _lone_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            request = submit(rig, "lone", PROMPT_C, arrival=1.0)
            outputs = step(rig)
            assert not [o for o in outputs if o.error], [o.error for o in outputs]
            assert scheduler._solo_request_id == "lone"
            for _ in range(60):
                if request.is_finished():
                    break
                outputs.extend(step(rig))
        assert request.is_finished()
        assert request.finish_reason != "error"
        assert not [o for o in outputs if o.error]
        assert "Admitting request lone alone (solo)" in caplog.text
    finally:
        rig.close()


def test_nothing_is_admitted_beside_the_lone_head(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, usage, _solo = _lone_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        submit(rig, "lone", PROMPT_C + PROMPT_A[:12], arrival=1.0)
        submit(rig, "next", PROMPT_A, arrival=2.0)
        outputs = step(rig)
        assert not [o for o in outputs if o.error]
        assert "lone" in scheduler.running or "lone" in [
            r.request_id for r in scheduler.prefilling
        ]
        assert waiting_ids(rig) == ["next"]
        # Plenty of room now: the solo mark alone keeps "next" waiting.
        usage[0] = scheduler._admission_limit_bytes() - 50 * GB
        step(rig)
        assert "next" in waiting_ids(rig) or scheduler.requests["lone"].is_finished()
    finally:
        rig.close()


def test_over_the_solo_line_the_old_rejection_still_applies(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, usage, solo_line = _lone_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        usage[0] = solo_line
        submit(rig, "lone", PROMPT_C, arrival=1.0)
        outputs = step(rig, 2)
        errors = [o for o in outputs if o.request_id == "lone" and o.error]
        assert len(errors) == 1
        assert "lone" not in scheduler.requests
        assert scheduler._solo_request_id is None
    finally:
        rig.close()


def test_without_the_park_feature_the_old_rejection_still_applies(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage, _solo = _lone_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        scheduler.config.memory_pressure_park = False
        submit(rig, "lone", PROMPT_C, arrival=1.0)
        outputs = step(rig, 2)
        assert [o for o in outputs if o.request_id == "lone" and o.error]
    finally:
        rig.close()


# --- Route preflight: the same predicate --------------------------------------


def test_the_route_preflight_lets_a_lone_request_that_fits_under_the_ceiling_queue(
    gated,  # noqa: F811
):
    scheduler, usage, admitted = gated
    scheduler.config.memory_pressure_park = True
    admitted[0] = 0
    scheduler._memory_hard_watermark_bytes = 80 * GB
    _fit, solo_line = scheduler._park_resume_lines()
    admission = scheduler._admission_limit_bytes()
    assert solo_line - admission > 2 * GB
    usage[0] = admission + 1 * GB
    scheduler.preflight_or_raise(num_prompt_tokens=1_000, max_tokens=100)
    usage[0] = solo_line
    with pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(num_prompt_tokens=1_000, max_tokens=100)
