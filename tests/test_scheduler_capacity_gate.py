# SPDX-License-Identifier: Apache-2.0
"""Never-fits feasibility gate and capacity hint.

The preflight used to compare a request against CURRENT usage, so a request
that would fit once the running ones finish was rejected. The decision is now
split: a request that cannot fit even alone (weights + KV of context plus
max_tokens + transient over the capacity line) is rejected with a structured
``context_exceeds_capacity`` error; one that fits alone but not now waits.
"""

import pytest
from test_scheduler_park import (
    PROMPT_A,
    PROMPT_C,
    make_rig,
    step,
    step_until,
    submit,
    waiting_ids,
)
from test_scheduler_prefill_memory_guard import _make_request, _make_scheduler

import omlx.scheduler as scheduler_module
from omlx.engine_core import _raise_request_output_error
from omlx.exceptions import ContextExceedsCapacityError, PrefillMemoryExceededError
from omlx.request import RequestOutput

GB = 1024**3

CAPACITY_FIELDS = {
    "required_bytes",
    "capacity_bytes",
    "weights_bytes",
    "kv_bytes_per_token",
    "requested_context_tokens",
    "requested_max_tokens",
    "max_viable_context_tokens",
}


@pytest.fixture
def gated(monkeypatch):
    """Guard on, a 100 GB hard limit, 20 GB of weights; ``usage[0]`` and
    ``admitted[0]`` are the current usage and admitted rows it reads."""
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 100 * GB
    scheduler._weights_bytes = 20 * GB
    usage = [0]
    admitted = [0]
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda **_kw: usage[0])
    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: admitted[0])
    return scheduler, usage, admitted


def _required(scheduler, tokens):
    est = scheduler._admission_estimate(
        num_prompt_tokens=tokens,
        cached_tokens=0,
        current=scheduler._model_weights_bytes(),
        text_only=True,
    )
    return est.estimated


# --- Math -----------------------------------------------------------------


def test_capacity_is_the_tighter_of_admission_line_and_safety_cap(gated):
    scheduler, _usage, _admitted = gated
    expected = min(scheduler._admission_limit_bytes(), scheduler._prefill_abort_cap())
    assert scheduler._capacity_limit_bytes() == expected > 0


def test_required_is_weights_plus_kv_of_context_and_max_tokens_plus_transient(
    gated,
):
    scheduler, _usage, _admitted = gated
    cap = scheduler._capacity_estimate(
        context_tokens=30_000, max_tokens=2_000, text_only=True
    )
    monitor = scheduler.memory_monitor
    est = scheduler._admission_estimate(
        num_prompt_tokens=32_000, cached_tokens=0, current=0, text_only=True
    )
    assert est.kv_exact == int(
        monitor.estimate_resident_kv_bytes(32_000, chunk_tokens=est.floor_chunk)
    )
    assert cap.required_bytes == 20 * GB + est.kv_exact + est.transient
    assert cap.weights_bytes == 20 * GB
    assert cap.capacity_bytes == scheduler._capacity_limit_bytes()
    assert cap.requested_context_tokens == 30_000
    assert cap.requested_max_tokens == 2_000
    # 32 layers x 8 KV heads x head_dim 192 x 2 bytes x (K + V)
    assert cap.kv_bytes_per_token == 32 * 8 * 192 * 2 * 2
    assert cap.fits_alone


def test_max_viable_context_is_the_largest_total_that_fits_alone(gated):
    scheduler, _usage, _admitted = gated
    cap = scheduler._capacity_estimate(context_tokens=0, max_tokens=0)
    best = cap.max_viable_context_tokens
    assert best > 1000
    assert _required(scheduler, best) <= cap.capacity_bytes
    assert _required(scheduler, best + 1) > cap.capacity_bytes


def test_weights_over_capacity_leave_no_viable_context(gated):
    scheduler, _usage, _admitted = gated
    scheduler._weights_bytes = 200 * GB
    cap = scheduler._capacity_estimate(context_tokens=10, max_tokens=0)
    assert cap.max_viable_context_tokens == 0
    assert not cap.fits_alone


def test_capacity_is_unavailable_without_weights_or_a_limit(gated):
    scheduler, _usage, _admitted = gated
    scheduler._weights_bytes = None
    assert scheduler._model_weights_bytes() == 0  # MagicMock model: unknown
    assert scheduler._capacity_estimate(context_tokens=10, max_tokens=0) is None
    scheduler._weights_bytes = 20 * GB
    scheduler._memory_hard_limit_bytes = 0
    assert scheduler._capacity_estimate(context_tokens=10, max_tokens=0) is None


def test_weights_are_read_from_the_model_parameters():
    import mlx.core as mx
    import mlx.nn as nn

    scheduler = _make_scheduler()
    scheduler.model = nn.Linear(64, 32)
    mx.eval(scheduler.model.parameters())
    expected = 64 * 32 * 4 + 32 * 4  # fp32 weight and bias
    scheduler._weights_bytes = None
    assert scheduler._model_weights_bytes() == expected


def test_capacity_snapshot_adds_free_bytes_now(gated):
    scheduler, usage, _admitted = gated
    usage[0] = 30 * GB
    snap = scheduler.capacity_snapshot(context_tokens=1000, max_tokens=100)
    assert set(snap) >= CAPACITY_FIELDS
    assert snap["free_bytes_now"] == snap["capacity_bytes"] - 30 * GB
    assert scheduler.max_viable_context_tokens() == snap["max_viable_context_tokens"]


def test_max_viable_context_is_cached(gated, monkeypatch):
    scheduler, _usage, _admitted = gated
    first = scheduler.max_viable_context_tokens()
    calls = []
    real = scheduler._admission_estimate
    monkeypatch.setattr(
        scheduler,
        "_admission_estimate",
        lambda **kw: calls.append(kw) or real(**kw),
    )
    assert scheduler.max_viable_context_tokens() == first
    assert calls == []


# --- Route preflight ------------------------------------------------------------


def test_never_fits_alone_is_rejected_with_the_capacity_numbers(gated):
    scheduler, _usage, _admitted = gated
    best = scheduler.max_viable_context_tokens()
    with pytest.raises(ContextExceedsCapacityError) as info:
        scheduler.preflight_or_raise(
            num_prompt_tokens=best, max_tokens=4096, request_id="huge"
        )
    exc = info.value
    assert isinstance(exc, PrefillMemoryExceededError)  # same HTTP 400 mapping
    assert exc.request_id == "huge"
    assert set(exc.capacity_fields) >= CAPACITY_FIELDS
    assert exc.capacity_fields["requested_context_tokens"] == best
    assert exc.capacity_fields["requested_max_tokens"] == 4096
    assert exc.capacity_fields["max_viable_context_tokens"] == best
    assert exc.estimated_bytes == exc.capacity_fields["required_bytes"]
    assert exc.limit_bytes == exc.capacity_fields["capacity_bytes"]
    assert exc.estimated_bytes > exc.limit_bytes
    assert "compact or shorten the context" in str(exc).lower()


def test_never_fits_is_checked_even_when_nothing_else_uses_memory(gated):
    """Current usage is irrelevant: a request that could start now but whose
    max_tokens cannot fit would otherwise be paused and never resumed."""
    scheduler, usage, _admitted = gated
    usage[0] = 0
    best = scheduler.max_viable_context_tokens()
    scheduler.preflight_or_raise(num_prompt_tokens=best - 10_000, max_tokens=0)
    with pytest.raises(ContextExceedsCapacityError):
        scheduler.preflight_or_raise(num_prompt_tokens=best - 10_000, max_tokens=20_000)


def test_fits_alone_but_not_now_waits_while_other_work_runs(gated):
    scheduler, usage, admitted = gated
    usage[0] = 99 * GB  # almost everything is taken by running requests
    admitted[0] = 2
    scheduler.preflight_or_raise(num_prompt_tokens=50_000, max_tokens=1_000)


def test_fits_alone_but_not_now_on_an_idle_scheduler_keeps_the_old_rejection(gated):
    """Nothing running will free the memory, so waiting cannot help."""
    scheduler, usage, admitted = gated
    usage[0] = 99 * GB
    admitted[0] = 0
    with pytest.raises(PrefillMemoryExceededError) as info:
        scheduler.preflight_or_raise(num_prompt_tokens=50_000, max_tokens=1_000)
    assert not isinstance(info.value, ContextExceedsCapacityError)


def test_unknown_capacity_keeps_the_old_rejection(gated):
    scheduler, usage, admitted = gated
    scheduler._weights_bytes = None  # MagicMock model: weights unknown
    usage[0] = 99 * GB
    admitted[0] = 2
    with pytest.raises(PrefillMemoryExceededError) as info:
        scheduler.preflight_or_raise(num_prompt_tokens=50_000, max_tokens=1_000)
    assert not isinstance(info.value, ContextExceedsCapacityError)


def test_a_never_fits_request_does_not_ask_for_an_eviction(gated):
    scheduler, usage, _admitted = gated
    usage[0] = 99 * GB
    best = scheduler.max_viable_context_tokens()
    assert (
        scheduler.preflight_eviction_request(num_prompt_tokens=best + 1, max_tokens=0)
        is None
    )
    assert (
        scheduler.preflight_eviction_request(num_prompt_tokens=50_000, max_tokens=0)
        is not None
    )


# --- In-stream admission --------------------------------------------------------


def test_in_stream_never_fits_returns_a_rejection_carrying_the_capacity(gated):
    scheduler, _usage, _admitted = gated
    best = scheduler.max_viable_context_tokens()
    request = _make_request(best)
    request.sampling_params.max_tokens = 100
    rejection = scheduler._preflight_memory_check(request)
    assert rejection is not None
    assert rejection.capacity is not None
    assert not rejection.capacity.fits_alone


def test_in_stream_fits_later_defers_while_rows_are_admitted(gated):
    scheduler, usage, admitted = gated
    usage[0] = 99 * GB
    admitted[0] = 1
    request = _make_request(50_000)
    with pytest.raises(scheduler_module._PreflightDeferredError):
        scheduler._preflight_memory_check(request)
    admitted[0] = 0
    rejection = scheduler._preflight_memory_check(request)
    assert rejection is not None and rejection.capacity is None


def test_engine_core_maps_the_output_to_the_typed_error():
    fields = {
        "required_bytes": 9,
        "capacity_bytes": 8,
        "weights_bytes": 5,
        "kv_bytes_per_token": 1,
        "requested_context_tokens": 3,
        "requested_max_tokens": 1,
        "max_viable_context_tokens": 2,
    }
    output = RequestOutput(
        request_id="r",
        finished=True,
        finish_reason="error",
        error="does not fit",
        error_code="context_exceeds_capacity",
        error_metadata={"request_id": "r", **fields},
    )
    with pytest.raises(ContextExceedsCapacityError) as info:
        _raise_request_output_error(output)
    assert info.value.capacity_fields == fields
    assert info.value.request_id == "r"


# --- Real scheduler loop ----------------------------------------------------------


def _guarded_rig(tmp_path, tokenizer, monkeypatch, *, weights_gb, free_gb):
    rig = make_rig(tmp_path, tokenizer)
    scheduler = rig.scheduler
    scheduler._memory_hard_limit_bytes = 100 * GB
    scheduler._weights_bytes = weights_gb * GB
    usage = [scheduler._admission_limit_bytes() - free_gb * GB]
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda **_kw: usage[0])
    return rig, usage


def test_a_request_that_fits_later_waits_in_line_without_an_error(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    try:
        scheduler = rig.scheduler
        scheduler._prefill_memory_guard = True
        # Rows that hold the memory are admitted (the tiny model would end a
        # real one at EOS within a few steps); there is no room right now.
        admitted = [1]
        monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: admitted[0])
        usage[0] = scheduler._admission_limit_bytes()
        later = submit(rig, "later", PROMPT_C, arrival=1.0)
        newer = submit(rig, "newer", PROMPT_A, arrival=2.0)
        outputs = step(rig, 4)
        assert not [o for o in outputs if o.error]
        # It keeps its place and nothing behind it overtakes it.
        assert waiting_ids(rig) == ["later", "newer"]
        assert "later" in scheduler.requests

        # The admitted rows finish and their memory comes back.
        monkeypatch.setattr(
            scheduler,
            "_num_admitted_requests",
            lambda: len(scheduler.running) + len(scheduler.prefilling),
        )
        usage[0] = scheduler._admission_limit_bytes() - 50 * GB
        step_until(
            rig,
            lambda: "later" not in waiting_ids(rig) and "newer" not in waiting_ids(rig),
        )
        assert later.finish_reason != "error"
        assert newer.finish_reason != "error"
    finally:
        rig.close()


def test_a_request_that_never_fits_ends_with_the_structured_error(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, _usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=200, free_gb=50
    )
    try:
        scheduler = rig.scheduler
        scheduler._prefill_memory_guard = True
        submit(rig, "big", PROMPT_C, arrival=1.0, max_tokens=7)
        outputs = step(rig, 2)
        errors = [o for o in outputs if o.request_id == "big" and o.error]
        assert len(errors) == 1
        assert errors[0].error_code == "context_exceeds_capacity"
        meta = errors[0].error_metadata
        assert set(meta) >= CAPACITY_FIELDS
        assert meta["requested_max_tokens"] == 7
        assert meta["weights_bytes"] == 200 * GB
        assert "big" not in scheduler.requests
    finally:
        rig.close()
