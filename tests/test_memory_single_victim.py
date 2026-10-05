# SPDX-License-Identifier: Apache-2.0
"""Single-victim memory abort: pick one request instead of aborting them all.

Covers the scheduler's victim choice (largest live KV, newest arrival breaks
ties, parked and KV-less requests never chosen) and the engine's abort of that
one request, which must hand its client the same error frame as the abort-all
path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.engine_core import EngineCore
from omlx.request import MemoryVictim, Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler

GB = 1024**3


def _req(rid, prompt_tokens, output_tokens=0, arrival=0.0, parked=False):
    request = Request(
        request_id=rid,
        prompt=[1] * prompt_tokens,
        sampling_params=SamplingParams(max_tokens=64),
        arrival_time=arrival,
    )
    request.num_prompt_tokens = prompt_tokens
    request.output_token_ids = [2] * output_tokens
    request.num_computed_tokens = prompt_tokens + output_tokens
    request.status = RequestStatus.RUNNING
    request.parked = parked
    return request


def _fake_scheduler(running=(), prefilling=(), waiting=(), pending=(), monitor=None):
    """The attributes ``Scheduler.select_memory_victim`` reads."""
    everything = [*running, *prefilling, *waiting]
    fake = SimpleNamespace(
        requests={r.request_id: r for r in everything},
        running={r.request_id: r for r in running},
        prefilling=list(prefilling),
        waiting=list(waiting),
        _pending_abort_ids=set(pending),
        _prefill_states={r.request_id: object() for r in prefilling},
        memory_monitor=monitor,
    )
    fake.is_request_parked = lambda rid: bool(fake.requests[rid].parked)
    return fake


def _select(fake, **kwargs):
    return Scheduler.select_memory_victim(fake, **kwargs)


class TestSelectMemoryVictim:
    def test_largest_live_kv_is_chosen_among_three_decoding_requests(self):
        fake = _fake_scheduler(
            running=[
                _req("small", 1_000, 10, arrival=1.0),
                _req("big", 90_000, 500, arrival=2.0),
                _req("mid", 30_000, 200, arrival=3.0),
            ]
        )

        victim = _select(fake)

        assert victim.request_id == "big"
        assert victim.tokens == 90_500
        assert victim.kept == 2

    def test_tie_goes_to_the_most_recent_arrival(self):
        fake = _fake_scheduler(
            running=[
                _req("old", 5_000, 0, arrival=1.0),
                _req("new", 5_000, 0, arrival=9.0),
                _req("mid", 5_000, 0, arrival=4.0),
            ]
        )

        assert _select(fake).request_id == "new"

    def test_a_prefilling_request_counts_by_computed_tokens(self):
        prefill = _req("pf", 200_000, 0, arrival=2.0)
        prefill.num_computed_tokens = 120_000
        fake = _fake_scheduler(
            running=[_req("dec", 50_000, 100, arrival=1.0)], prefilling=[prefill]
        )

        victim = _select(fake)

        assert victim.request_id == "pf"
        assert victim.tokens == 120_000

    def test_parked_requests_are_never_chosen(self):
        fake = _fake_scheduler(
            running=[_req("live", 1_000, 0, arrival=1.0)],
            waiting=[_req("parked", 150_000, 0, arrival=2.0, parked=True)],
        )

        assert _select(fake).request_id == "live"

    def test_requests_without_live_kv_are_never_chosen(self):
        waiting = _req("queued", 10_000, 0, arrival=5.0)
        waiting.num_computed_tokens = 0
        fake = _fake_scheduler(waiting=[waiting])

        assert _select(fake) is None

    def test_a_request_with_a_pending_abort_is_skipped(self):
        fake = _fake_scheduler(
            running=[
                _req("gone", 90_000, 0, arrival=1.0),
                _req("live", 1_000, 0, arrival=2.0),
            ],
            pending={"gone"},
        )

        victim = _select(fake)

        assert victim.request_id == "live"
        assert victim.kept == 0

    def test_estimated_bytes_use_the_kv_estimator_when_available(self):
        monitor = SimpleNamespace(estimate_prompt_kv_bytes=lambda n: n * 1000.0)
        fake = _fake_scheduler(running=[_req("a", 2_000, 0)], monitor=monitor)

        assert _select(fake).est_bytes == 2_000_000


def _engine(scheduler, collectors):
    engine = object.__new__(EngineCore)
    engine.scheduler = scheduler
    engine._output_collectors = collectors
    engine._finished_at = {}
    engine._finished_events = {}
    engine._closed = False
    engine._mark_request_finished = lambda rid: engine._finished_at.setdefault(rid, 1.0)
    engine._wake_engine_loop = lambda: None
    return engine


def _scheduler_for_engine(victim_rid, kept=2):
    sched = MagicMock()
    sched._pending_abort_ids = set()
    sched._memory_hard_limit_bytes = 100 * GB
    sched._memory_hard_watermark_bytes = 95 * GB
    sched.is_request_parked.return_value = False
    sched.select_memory_victim.return_value = (
        None
        if victim_rid is None
        else MemoryVictim(victim_rid, tokens=90_000, est_bytes=6 * GB, kept=kept)
    )
    return sched


class TestEngineAbortOneRequestForMemory:
    @pytest.mark.asyncio
    async def test_only_the_victim_is_aborted_and_the_others_keep_running(self):
        sched = _scheduler_for_engine("big")
        collectors = {rid: MagicMock() for rid in ("big", "b", "c")}
        engine = _engine(sched, collectors)

        with patch("omlx.utils.proc_memory.get_phys_footprint", return_value=97 * GB):
            victim = await engine.abort_one_request_for_memory()

        assert victim.request_id == "big"
        sched.abort_request.assert_called_once_with("big")
        collectors["big"].put.assert_called_once()
        collectors["b"].put.assert_not_called()
        collectors["c"].put.assert_not_called()
        assert set(engine._finished_at) == {"big"}

    @pytest.mark.asyncio
    async def test_victim_gets_the_same_error_frame_as_abort_all(self):
        with patch("omlx.utils.proc_memory.get_phys_footprint", return_value=97 * GB):
            one_sched = _scheduler_for_engine("r")
            one = {"r": MagicMock()}
            await _engine(one_sched, one).abort_one_request_for_memory()

            all_sched = _scheduler_for_engine(None)
            all_sched._pending_abort_ids = set()
            every = {"r": MagicMock()}
            await _engine(all_sched, every).abort_all_requests()

        single = one["r"].put.call_args.args[0]
        legacy = every["r"].put.call_args.args[0]
        assert single.finished is True
        assert single.finish_reason == legacy.finish_reason == "error"
        assert single.error_code == legacy.error_code == "prefill_memory_aborted"
        assert single.error == legacy.error
        assert "process memory limit exceeded" in single.error
        assert single.new_text == legacy.new_text
        assert single.error_metadata == legacy.error_metadata

    @pytest.mark.asyncio
    async def test_no_victim_aborts_nothing_and_returns_none(self):
        sched = _scheduler_for_engine(None)
        collectors = {"a": MagicMock()}
        engine = _engine(sched, collectors)

        assert await engine.abort_one_request_for_memory() is None

        sched.abort_request.assert_not_called()
        collectors["a"].put.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_victim_without_a_collector_is_reported_as_skipped(self):
        """Distinct from "no victim" (None): the caller can tell them apart."""
        sched = _scheduler_for_engine("ghost")
        engine = _engine(sched, {"other": MagicMock()})

        victim = await engine.abort_one_request_for_memory()

        assert victim is not None and victim.request_id == "ghost"
        assert victim.skipped == "no_client"
        sched.abort_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_already_finished_victim_is_reported_as_skipped(self):
        sched = _scheduler_for_engine("done")
        collectors = {"done": MagicMock()}
        engine = _engine(sched, collectors)
        engine._finished_at["done"] = 1.0

        victim = await engine.abort_one_request_for_memory()

        assert victim is not None and victim.skipped == "already_finished"
        sched.abort_request.assert_not_called()
        collectors["done"].put.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_aborted_victim_is_not_skipped(self):
        sched = _scheduler_for_engine("big")
        engine = _engine(sched, {"big": MagicMock()})

        with patch("omlx.utils.proc_memory.get_phys_footprint", return_value=97 * GB):
            victim = await engine.abort_one_request_for_memory()

        assert victim.skipped is None


class TestMemoryVictimPending:
    """The enforcer asks whether a victim's deferred abort has run yet."""

    def test_pending_until_the_scheduler_has_run_the_abort(self):
        sched = SimpleNamespace(_pending_abort_ids={"big"}, requests={"big": 1})
        engine = _engine(sched, {})
        assert engine.memory_victim_pending("big") is True

        sched._pending_abort_ids.clear()  # step popped it, abort in progress
        assert engine.memory_victim_pending("big") is True

        sched.requests.clear()  # _do_abort_request dropped the request
        assert engine.memory_victim_pending("big") is False

    def test_no_scheduler_means_nothing_pending(self):
        engine = _engine(None, {})
        assert engine.memory_victim_pending("big") is False
