# SPDX-License-Identifier: Apache-2.0
"""A queued request waits while memory is scheduled on purpose.

Forced pressure (Qwen3.8-27B at a 48 GB ceiling): the third queued
request failed with "memory pressure persisted for 90.6s
(store_cache_backpressure)" (text decode) and with ``admission_paused`` after
67.5 s (image prefill) while the other requests ran. The admission-stall
clock started when the head first met pressure and kept running while a solo
request decoded (the solo hold breaks before the stall check, so the clock is
neither checked nor reset) and while another request prefilled chunk after
chunk. The first check after the solo request finished found 90 s on it and
failed the head, although the server had been making progress all along.

The clock now measures time without progress: a token emitted, a prefill
chunk processed, an external prefill finished, a request admitted or a
store-cache write drained resets it. While memory is scheduled on purpose (a
parked or decode-paused request, a solo request, a deferred fits-later head)
the head waits under a long, configurable bound
(``admission_no_progress_timeout_s``) instead of the 60 s one, so a genuine
deadlock still surfaces with a clear error.
"""

import logging
import time
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park
from test_memory_decode_pause import _busy, tick
from test_memory_resume_thrash import _rig_engine
from test_process_memory_enforcer import (  # noqa: F401 (fixture)
    _make_enforcer,
    mock_engine_pool,
)
from test_scheduler_prefill_memory_guard import _make_scheduler

import omlx.scheduler as scheduler_module
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import SchedulerConfig, _StoreCacheGate
from omlx.settings import GlobalSettings, SchedulerSettings

GB = 1024**3
MAX_TOKENS = 60
STEP_S = 2.0  # controlled clock: every scheduler step takes 2 s


class _Clock:
    """The scheduler's ``time`` module with a controlled monotonic clock."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(scheduler_module, "time", fake)
    return fake


@pytest.fixture(autouse=True)
def single_victim_on(monkeypatch):
    """Every enforcer built here aborts one victim (memory.abort_single_victim)."""
    from omlx.process_memory_enforcer import ProcessMemoryEnforcer

    real_init = ProcessMemoryEnforcer.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("abort_single_victim", True)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ProcessMemoryEnforcer, "__init__", init)


@pytest.fixture
def enforcer(mock_engine_pool):  # noqa: F811
    # Ceiling 100 GB: soft 85 GB pauses admission, hard 95 GB pauses rows.
    return _make_enforcer(
        mock_engine_pool,
        ceiling=100 * GB,
        soft_threshold=0.85,
        hard_threshold=0.95,
        memory_scheduling="on",
    )


# --- Real enforcer, real scheduler, controlled clock ----------------------


def _submit(rig, rid, tokens, arrival):
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0),
        arrival_time=arrival,
    )
    rig.scheduler.add_request(request)
    return request


def _baseline(tmp_path, prompts):
    rig = park._Rig(
        tmp_path / "baseline",
        dp._tokenizer(),
        split=False,
        commit_blocks=1,
        max_blocks=dp.MAX_BLOCKS,
    )
    try:
        rows = [
            _submit(rig, rid, tokens, float(i + 1))
            for i, (rid, tokens) in enumerate(prompts.items())
        ]
        park.run_to_completion(rig, rows, limit=20 * MAX_TOKENS)
        return {r.request_id: list(r.output_token_ids) for r in rows}
    finally:
        rig.close()


def _admitted_ids(scheduler):
    return list(scheduler.running) + [r.request_id for r in scheduler.prefilling]


def _finished(rows, outputs):
    done = {o.request_id for o in outputs if o.finished}
    return all(r.is_finished() or r.request_id in done for r in rows)


async def _drive(rig, enforcer, clock, rows, *, submit_later=(), limit=None):
    """Tick the enforcer on the scripted reading, step, advance the clock.

    ``submit_later`` requests arrive after the first step (the real run
    staggered them 0.5 s), so they meet the pressure the first request causes.
    """
    scheduler = rig.scheduler
    outputs = []
    limit = limit or 12 * MAX_TOKENS
    for index in range(limit):
        if _finished(rows, outputs):
            break
        # One request fits at a time: with one admitted, usage sits in the
        # soft band (admission paused); idle, it is low.
        gb = 90 if _admitted_ids(scheduler) else 20
        await tick(enforcer, gb)
        outputs.extend(park.step(rig))
        clock.advance(STEP_S)
        if index == 0:
            for rid, tokens, arrival in submit_later:
                rows.append(_submit(rig, rid, tokens, arrival))
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    scheduler._drain_pending_async_removes()
    return outputs


@pytest.mark.asyncio
async def test_three_requests_that_fit_one_at_a_time_complete_without_a_stall(
    tmp_path, enforcer, clock, monkeypatch, caplog
):
    """Each request decodes for 120 s of controlled time while the others
    wait under soft pressure. The old 60 s clock failed the second request
    while the first was still emitting tokens."""
    prompts = {"a": dp.PROMPT, "b": dp.OTHER, "c": park.PROMPT_D}
    expected = _baseline(tmp_path, prompts)
    rig = park._Rig(
        tmp_path / "sequential",
        dp._tokenizer(),
        split=False,
        commit_blocks=1,
        max_blocks=dp.MAX_BLOCKS,
        memory_pressure_park=True,
    )
    try:
        rig.scheduler.note_pressure_poll("soft")
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        rows = [_submit(rig, "a", dp.PROMPT, 1.0)]

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            outputs = await _drive(
                rig,
                enforcer,
                clock,
                rows,
                submit_later=[("b", dp.OTHER, 2.0), ("c", park.PROMPT_D, 3.0)],
            )

        errors = [o for o in outputs if o.error]
        assert errors == []
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
        for row in rows:
            rid = row.request_id
            assert list(row.output_token_ids) == expected[rid]
            assert dp.streamed_tokens(outputs, rid) == expected[rid]
            (final,) = [o for o in outputs if o.request_id == rid and o.finished]
            assert final.finish_reason == "length"
        # They ran one at a time: c started only after b finished.
        assert "Admission waiting for memory: admission_paused" in caplog.text
        assert "request(s) progressing" in caplog.text
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_a_queued_request_still_fails_when_nothing_progresses(
    tmp_path, enforcer, clock, monkeypatch
):
    """A genuine stall: the admitted request emits nothing (its decode is
    wedged) while pressure persists. The head fails after the stall timeout
    with a clear error, because nothing progressed."""
    rig = park._Rig(
        tmp_path / "wedged",
        dp._tokenizer(),
        split=False,
        commit_blocks=1,
        max_blocks=dp.MAX_BLOCKS,
    )
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        rows = [_submit(rig, "a", dp.PROMPT, 1.0)]
        outputs = []
        for _ in range(20):  # until a is admitted and decoding
            await tick(enforcer, 90)
            outputs.extend(park.step(rig))
            clock.advance(STEP_S)
            if "a" in scheduler.running:
                break
        assert "a" in scheduler.running
        rows.append(_submit(rig, "b", dp.OTHER, 2.0))
        # From here a's decode emits nothing.
        monkeypatch.setattr(scheduler.batch_generator, "next_generated", lambda: [])

        timeout = scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S
        for _ in range(int(timeout / STEP_S) + 3):
            await tick(enforcer, 90)
            outputs.extend(park.step(rig))
            clock.advance(STEP_S)

        (failed,) = [o for o in outputs if o.request_id == "b" and o.finished]
        assert failed.error_code == "memory_admission_stalled"
        assert "no request made progress" in failed.error
        assert failed.error_metadata["no_progress_seconds"] >= int(timeout)
        assert "b" not in scheduler.requests
    finally:
        rig.close()


# --- Scheduler unit: progress resets the clock -----------------------------


def _request(rid, prompt_tokens=64):
    return Request(
        request_id=rid,
        prompt=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=16),
    )


@pytest.fixture
def pressured(monkeypatch, clock):
    """A scheduler with one running row ``run`` and a queued head ``head``,
    admission paused by memory pressure."""
    scheduler = _make_scheduler()
    running = _request("run")
    scheduler.running[running.request_id] = running
    scheduler.requests[running.request_id] = running
    head = _request("head")
    scheduler.waiting.append(head)
    scheduler.requests[head.request_id] = head
    scheduler._admission_paused = True
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._memory_scheduling = True  # memory.memory_scheduling resolved on
    return scheduler, head, running


def _schedule(scheduler):
    scheduled, rejected = scheduler._schedule_waiting()
    assert scheduled == []
    return rejected


def test_progress_by_others_resets_the_stall_clock(pressured, clock):
    scheduler, head, _ = pressured
    timeout = scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S
    assert _schedule(scheduler) == []  # the head starts waiting

    clock.advance(timeout - 10)
    scheduler._note_admission_progress()  # the running row emitted a token
    clock.advance(timeout - 1)
    assert _schedule(scheduler) == []  # 2 * timeout waiting, 59 s idle

    clock.advance(1)
    (failed,) = _schedule(scheduler)
    assert failed.request_id == "head"
    assert failed.error_code == "memory_admission_stalled"
    assert failed.error_metadata["reason"] == "admission_paused"
    assert failed.error_metadata["no_progress_seconds"] == int(timeout)
    assert failed.error_metadata["stalled_seconds"] == int(2 * timeout - 10)
    assert "no request made progress" in failed.error


def test_a_long_solo_run_does_not_fail_the_head_when_it_finishes(
    pressured, clock, caplog
):
    """Text decode under forced pressure: the head met pressure, then a solo
    request decoded for 90 s; the first check after it finished (store-cache
    backpressure under memory pressure) failed the head."""
    scheduler, head, solo = pressured
    scheduler._solo_request_id = solo.request_id
    scheduler._admission_paused = True
    assert _schedule(scheduler) == []  # clock starts (solo hold)

    for _ in range(45):  # 90 s of solo decode, a token each step
        clock.advance(STEP_S)
        scheduler._note_admission_progress()
        assert _schedule(scheduler) == []
    # The solo request finishes; its store-cache write is in flight.
    del scheduler.running[solo.request_id]
    solo.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
    gate = _StoreCacheGate(cap=1)
    gate.note_submitted()
    scheduler._store_cache_gate = gate
    scheduler._prefill_memory_guard = True
    scheduler._memory_limit_bytes = 100
    scheduler._current_usage_bytes = MagicMock(return_value=101)
    clock.advance(0.6)

    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        assert _schedule(scheduler) == []
    assert list(scheduler.waiting) == [head]
    assert "Memory admission stalled" not in caplog.text


def test_a_solo_hold_waits_under_the_long_bound_and_then_fails(pressured, clock):
    """Deliberate scheduling: the head waits behind a solo request. A solo
    request that emits nothing for ``admission_no_progress_timeout_s`` is a
    genuine deadlock: the head fails with a clear error."""
    scheduler, head, solo = pressured
    scheduler._solo_request_id = solo.request_id
    bound = scheduler.config.admission_no_progress_timeout_s
    assert bound == 600.0
    assert _schedule(scheduler) == []

    clock.advance(bound - 1)
    assert _schedule(scheduler) == []  # far past the 60 s line, still waiting
    clock.advance(1)
    (failed,) = _schedule(scheduler)
    assert failed.error_code == "memory_admission_stalled"
    assert failed.error_metadata["reason"] == "solo_hold"
    assert failed.error_metadata["no_progress_seconds"] == int(bound)
    assert "no request made progress for 600" in failed.error
    assert list(scheduler.waiting) == []


def test_a_deferred_head_waits_under_the_long_bound(pressured, clock):
    scheduler, head, _ = pressured
    scheduler._deferred_admissions[head.request_id] = SimpleNamespace(
        admitted_ids=frozenset({"run"}),
        free_bytes=None,
        retry_at=clock.now + 10_000,
        store_jobs=0,
    )
    assert _schedule(scheduler) == []
    clock.advance(scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S * 3)
    assert _schedule(scheduler) == []
    clock.advance(scheduler.config.admission_no_progress_timeout_s)
    (failed,) = _schedule(scheduler)
    assert failed.error_metadata["reason"] == "deferred_admission"


def test_the_waiting_log_is_rate_limited(pressured, clock, caplog):
    scheduler, _, _ = pressured
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        for _ in range(20):
            _schedule(scheduler)
            scheduler._note_admission_progress()
            clock.advance(1.0)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Admission waiting for memory")
    ]
    assert lines == [
        "Admission waiting for memory: admission_paused; 1 request(s) progressing"
    ]
    clock.advance(scheduler_module._ADMISSION_WAIT_LOG_INTERVAL_S)
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        _schedule(scheduler)
    assert (
        sum(
            r.getMessage().startswith("Admission waiting for memory")
            for r in caplog.records
        )
        == 2
    )


# --- Store-cache backpressure: drains are progress -------------------------


def _drain_one(scheduler):
    """The writer finishes one job; the next one is already queued, so the
    gate stays full."""
    future = Future()
    future.set_result(None)
    scheduler._pending_async_removes = deque([(12345, "gone", future)])
    assert scheduler._drain_pending_async_removes() is True
    scheduler._store_cache_gate.note_submitted()


def test_store_cache_backpressure_waits_while_writes_drain(pressured, clock):
    """Not memory related (no pressure): the store-cache stall fails only
    when the writer stops draining."""
    scheduler, head, running = pressured
    del scheduler.running[running.request_id]
    scheduler._admission_paused = False
    gate = _StoreCacheGate(cap=1)
    gate.note_submitted()
    scheduler._store_cache_gate = gate
    timeout = scheduler._STORE_CACHE_ADMISSION_STALL_TIMEOUT_S
    assert _schedule(scheduler) == []

    for _ in range(3):  # 3 * (timeout - 1) s, the writer drains a job each time
        clock.advance(timeout - 1)
        _drain_one(scheduler)
        assert _schedule(scheduler) == []

    clock.advance(timeout)
    (failed,) = _schedule(scheduler)
    assert failed.error_code == "store_cache_admission_stalled"
    assert failed.error_metadata["no_progress_seconds"] == int(timeout)


def test_memory_related_store_backpressure_waits_while_writes_drain(pressured, clock):
    """The forced-pressure shape: nothing admitted, memory pressure, the finished
    request's write in flight."""
    scheduler, head, running = pressured
    del scheduler.running[running.request_id]
    gate = _StoreCacheGate(cap=1)
    gate.note_submitted()
    scheduler._store_cache_gate = gate
    timeout = scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S
    assert _schedule(scheduler) == []
    for _ in range(3):
        clock.advance(timeout - 1)
        _drain_one(scheduler)
        assert _schedule(scheduler) == []
    clock.advance(timeout)
    (failed,) = _schedule(scheduler)
    assert failed.error_code == "memory_admission_stalled"
    assert failed.error_metadata["reason"] == "store_cache_backpressure"


# --- Configuration ----------------------------------------------------------


def test_the_no_progress_bound_defaults_to_ten_minutes_and_is_configurable(
    tmp_path, monkeypatch
):
    assert SchedulerConfig().admission_no_progress_timeout_s == 600.0
    assert SchedulerSettings().admission_no_progress_timeout_s == 600.0
    custom = SchedulerSettings.from_dict({"admission_no_progress_timeout_s": 1800})
    assert custom.admission_no_progress_timeout_s == 1800.0
    for bad in (0, -1, "long", True, float("inf")):
        junk = SchedulerSettings.from_dict({"admission_no_progress_timeout_s": bad})
        assert junk.admission_no_progress_timeout_s == 600.0

    monkeypatch.setenv("OMLX_ADMISSION_NO_PROGRESS_TIMEOUT_S", "900")
    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.scheduler.admission_no_progress_timeout_s == 900.0
    assert settings.to_scheduler_config().admission_no_progress_timeout_s == 900.0
    settings.scheduler.admission_no_progress_timeout_s = 0
    assert any("admission_no_progress_timeout_s" in e for e in settings.validate())
