# SPDX-License-Identifier: Apache-2.0
"""Memory-aware resume of parked requests (prefill-parked and decode-paused).

A parked head resumes once the estimated free memory covers its need plus a
margin, instead of after a fixed streak of ok pressure polls. While it waits,
the head-of-line hold keeps newer requests out, so nothing consumes the
headroom it is waiting for. The ok-poll rule stays as the fallback when no
estimate is available.
"""

import logging
import time
from types import SimpleNamespace

import pytest
from test_scheduler_park import (
    PROMPT_A,
    PROMPT_B,
    PROMPT_C,
    make_rig,
    prefill_two,
    prefilling_ids,
    step,
    step_until,
    submit,
    waiting_ids,
)
from test_scheduler_prefill_memory_guard import _make_scheduler

import omlx.scheduler as scheduler_module
from omlx.request import Request, SamplingParams
from omlx.scheduler import _DecodeResume

GB = 1024**3


def _parked_request(rid="parked", prompt_tokens=4000, max_tokens=1000, committed=0):
    request = Request(
        request_id=rid,
        prompt=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        arrival_time=1.0,
    )
    request.prompt_token_ids = list(range(prompt_tokens))
    request.num_prompt_tokens = prompt_tokens
    request.parked = True
    request.parked_at = time.monotonic()
    request.park_committed_tokens = committed
    return request


def _decode_record(context_tokens, remaining):
    return _DecodeResume(
        storage_id="paused",
        domain_key=("paused", "1"),
        tokens=list(range(context_tokens)),
        kickoff=7,
        remaining_max_tokens=remaining,
        sampler=None,
        logits_processors=[],
        stop_sequences=None,
    )


@pytest.fixture
def live(monkeypatch):
    """A scheduler with a live pressure monitor, one admitted row and a
    100 GB admission line; ``usage[0]`` is the current usage it reads."""
    scheduler = _make_scheduler()
    scheduler._memory_hard_limit_bytes = 100 * GB
    scheduler.note_pressure_poll("soft")
    usage = [0]
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda **_kw: usage[0])
    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: 1)
    return scheduler, usage


def _margin(scheduler):
    return int(
        scheduler._admission_limit_bytes()
        * scheduler_module.PARK_RESUME_MARGIN_FRACTION
    )


# --- Need ----------------------------------------------------------------


def test_decode_paused_need_is_final_kv_plus_one_decode_transient():
    scheduler = _make_scheduler()
    request = _parked_request("paused")
    scheduler._decode_resumes["paused"] = _decode_record(999, remaining=500)
    monitor = scheduler.memory_monitor
    final = 999 + 1 + 500  # saved context, the kickoff, the remaining tokens
    expected = monitor.estimate_resident_kv_bytes(
        final
    ) + monitor.estimate_chunk_transient_bytes(1, final)
    assert expected > 0
    assert scheduler._parked_resume_need_bytes(request) == int(expected)


def test_prefill_parked_need_is_full_kv_with_max_tokens_plus_prefill_transient():
    """The committed prefix comes back from SSD into memory, so it counts."""
    scheduler = _make_scheduler()
    request = _parked_request(prompt_tokens=4000, max_tokens=1000, committed=2048)
    monitor = scheduler.memory_monitor
    est = scheduler._admission_estimate(
        num_prompt_tokens=4000, cached_tokens=2048, current=0
    )
    expected = monitor.estimate_resident_kv_bytes(4000 + 1000) + est.transient
    assert scheduler._parked_resume_need_bytes(request) == int(expected)


def test_need_is_unavailable_without_a_monitor():
    scheduler = _make_scheduler()
    scheduler.memory_monitor = None
    assert scheduler._parked_resume_need_bytes(_parked_request()) is None


# --- Resume decision -------------------------------------------------------


def test_parked_head_resumes_once_free_memory_covers_need_plus_margin(live):
    scheduler, usage = live
    head = _parked_request()
    need = scheduler._parked_resume_need_bytes(head)
    limit = scheduler._admission_limit_bytes()

    usage[0] = limit - need - _margin(scheduler) + 1  # one byte short
    assert scheduler._park_hold_applies(head) is True

    usage[0] = limit - need - _margin(scheduler)
    assert scheduler._park_hold_applies(head) is False


def test_decode_paused_head_resumes_on_its_own_need(live):
    scheduler, usage = live
    head = _parked_request("paused")
    scheduler._decode_resumes["paused"] = _decode_record(20_000, remaining=4000)
    need = scheduler._parked_resume_need_bytes(head)
    limit = scheduler._admission_limit_bytes()
    usage[0] = limit - need - _margin(scheduler) + 1
    assert scheduler._park_hold_applies(head) is True
    usage[0] -= 1
    assert scheduler._park_hold_applies(head) is False


def test_ok_polls_alone_do_not_resume_a_head_that_does_not_fit(live):
    scheduler, usage = live
    head = _parked_request()
    usage[0] = scheduler._admission_limit_bytes()  # no free memory at all
    for _ in range(10):
        scheduler.note_pressure_poll("ok")
    assert scheduler._park_hold_applies(head) is True


def test_hard_pressure_holds_a_head_the_estimate_says_fits(live):
    scheduler, usage = live
    head = _parked_request()
    usage[0] = 0
    scheduler.note_pressure_poll("hard")
    assert scheduler._park_hold_applies(head) is True


def test_max_hold_does_not_release_a_head_that_does_not_fit(live):
    """Without an estimate the max-hold escape bounds the wait; with one, the
    running rows finish and the nothing-admitted escape releases the head."""
    scheduler, usage = live
    head = _parked_request()
    usage[0] = scheduler._admission_limit_bytes()
    head.parked_at -= scheduler_module.PARK_MAX_HOLD_S + 1
    assert scheduler._park_hold_applies(head) is True


def test_nothing_admitted_releases_only_a_head_that_fits_alone(live, monkeypatch):
    """Nothing admitted used to release any head at once, even one that
    alone overflows the line: it resumed, was paused at the next poll and
    looped until its park budget ran out (C7). Now it waits, bounded by
    PARK_MAX_HOLD_S, unless it fits; the stale-monitor escape stays."""
    scheduler, usage = live
    head = _parked_request()
    usage[0] = scheduler._admission_limit_bytes()
    assert scheduler._park_hold_applies(head) is True

    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: 0)
    assert scheduler._park_hold_applies(head) is True
    head.parked_at -= scheduler_module.PARK_MAX_HOLD_S + 1
    assert scheduler._park_hold_applies(head) is False
    assert scheduler._solo_request_id == head.request_id

    monkeypatch.setattr(scheduler, "_num_admitted_requests", lambda: 1)
    scheduler._last_pressure_poll_at = (
        time.monotonic() - scheduler_module.PARK_SIGNAL_STALE_S - 1
    )
    assert scheduler._park_hold_applies(head) is False


def test_ok_poll_rule_is_the_fallback_without_an_estimate(live):
    scheduler, usage = live
    scheduler._memory_hard_limit_bytes = 0  # no admission line: no estimate
    head = _parked_request()
    assert scheduler._park_hold_applies(head) is True
    for _ in range(scheduler.config.park_resume_ok_polls):
        scheduler.note_pressure_poll("ok")
    assert scheduler._park_hold_applies(head) is False


def test_in_flight_prefill_growth_is_reserved_against_the_head(live):
    """A prefill that already resumed still has KV to allocate: the next
    parked head must not count that memory as free."""
    scheduler, usage = live
    head = _parked_request("head")
    need = scheduler._parked_resume_need_bytes(head)
    limit = scheduler._admission_limit_bytes()
    usage[0] = limit - need - _margin(scheduler)
    assert scheduler._park_hold_applies(head) is False

    growing = _parked_request("growing", prompt_tokens=30_000)
    growing.parked = False
    scheduler.prefilling.append(growing)
    scheduler._prefill_states["growing"] = SimpleNamespace(
        base_size=2000, tokens_processed=8000
    )
    monitor = scheduler.memory_monitor
    reserve = monitor.estimate_resident_kv_bytes(
        30_000
    ) - monitor.estimate_resident_kv_bytes(10_000)
    assert scheduler._inflight_prefill_reserve_bytes() == int(reserve)
    assert scheduler._park_hold_applies(head) is True


def test_wait_and_resume_are_logged_with_need_and_free(live, caplog):
    scheduler, usage = live
    head = _parked_request("big")
    need = scheduler._parked_resume_need_bytes(head)
    limit = scheduler._admission_limit_bytes()
    usage[0] = limit - need // 2
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        scheduler._park_hold_applies(head)
        scheduler._park_hold_applies(head)  # rate-limited: one line
        waits = [r for r in caplog.records if "waiting: need" in r.getMessage()]
        assert len(waits) == 1
        assert "Parked request big waiting: need" in waits[0].getMessage()
        assert "GB, free" in waits[0].getMessage()

        usage[0] = 0
        assert scheduler._park_hold_applies(head) is False
        scheduler._finish_park_resume(head, log=False)
    resumes = [
        r.getMessage() for r in caplog.records if "Resuming parked" in r.getMessage()
    ]
    assert len(resumes) == 1
    assert resumes[0].startswith("Resuming parked request big: need ")
    assert " GB fits in free " in resumes[0]


# --- Reservation on the real scheduler loop ---------------------------------


def _reserved_rig(tmp_path, tokenizer, monkeypatch, *, free_gb, head_need_gb):
    rig = make_rig(tmp_path, tokenizer)
    scheduler = rig.scheduler
    prefill_two(rig)
    scheduler.request_park_prefill()
    step(rig)
    assert waiting_ids(rig) == ["young"]
    # A 100 GB machine whose usage leaves ``free_gb`` under the admission
    # line; the guard prices newer requests against the same usage.
    scheduler._memory_hard_limit_bytes = 100 * GB
    scheduler._prefill_memory_guard = True
    limit = scheduler._admission_limit_bytes()
    monkeypatch.setattr(
        scheduler, "_current_usage_bytes", lambda **_kw: limit - free_gb * GB
    )
    real_need = scheduler._parked_resume_need_bytes
    monkeypatch.setattr(
        scheduler,
        "_parked_resume_need_bytes",
        lambda r: head_need_gb * GB if r.request_id == "young" else real_need(r),
    )
    return rig


def test_a_smaller_newer_request_does_not_take_a_waiting_heads_headroom(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig = _reserved_rig(
        tmp_path, mock_tokenizer, monkeypatch, free_gb=30, head_need_gb=40
    )
    try:
        scheduler = rig.scheduler
        newer = submit(rig, "newer", PROMPT_C, arrival=9.0)
        # On its own the newer request fits the free memory comfortably.
        est = scheduler._admission_estimate(
            num_prompt_tokens=newer.num_prompt_tokens,
            cached_tokens=0,
            current=scheduler._current_usage_bytes(),
        )
        assert est is not None
        assert est.estimated <= scheduler._admission_limit_bytes()
        step(rig, 4)
        assert waiting_ids(rig) == ["young", "newer"]
        assert "newer" not in prefilling_ids(rig)
        assert not newer.output_token_ids
    finally:
        rig.close()


def test_a_newer_request_backfills_once_there_is_room_for_both(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig = _reserved_rig(
        tmp_path, mock_tokenizer, monkeypatch, free_gb=60, head_need_gb=40
    )
    try:
        young = rig.scheduler.requests["young"]
        newer = submit(rig, "newer", PROMPT_C, arrival=9.0)
        step_until(
            rig,
            lambda: (not young.parked)
            and ("newer" in prefilling_ids(rig) or newer.output_token_ids),
        )
        assert "newer" not in waiting_ids(rig)
    finally:
        rig.close()


def test_parked_requests_stay_fifo_under_the_memory_rule(
    tmp_path, mock_tokenizer, monkeypatch
):
    """Only the head is priced: a smaller parked request behind it waits."""
    rig = make_rig(tmp_path, mock_tokenizer)
    try:
        scheduler = rig.scheduler
        submit(rig, "r1", PROMPT_A, arrival=1.0)
        submit(rig, "r2", PROMPT_B, arrival=2.0)
        submit(rig, "r3", PROMPT_C, arrival=3.0)
        step_until(rig, lambda: len(prefilling_ids(rig)) == 3)
        scheduler.request_park_prefill()
        step(rig)
        scheduler.request_park_prefill()
        step(rig)
        assert waiting_ids(rig) == ["r2", "r3"]
        scheduler._memory_hard_limit_bytes = 100 * GB
        limit = scheduler._admission_limit_bytes()
        monkeypatch.setattr(
            scheduler, "_current_usage_bytes", lambda **_kw: limit - 10 * GB
        )
        needs = {"r2": 40 * GB, "r3": 1 * GB}
        monkeypatch.setattr(
            scheduler, "_parked_resume_need_bytes", lambda r: needs[r.request_id]
        )
        step(rig, 4)
        assert waiting_ids(rig) == ["r2", "r3"]
    finally:
        rig.close()
