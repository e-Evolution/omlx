# SPDX-License-Identifier: Apache-2.0
"""Hard memory pressure pauses a decoding row before it aborts anything.

The enforcer asks for a prefill park first; with no prefill to park it pauses
one decoding row (``request_decode_pause``) and waits for that pause to run
and its exact chain to reach the SSD before it counts the memory as freed.
Only when no row can be paused does it fall back to the single-victim abort.
Emergency pressure still aborts everything.

The first group drives the enforcer against a scripted scheduler; the second
wires the real enforcer to the real scheduler of the park suite's ``_Rig``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park
from test_process_memory_enforcer import (  # noqa: F401 (fixture)
    _make_enforcer,
    _make_entry,
    mock_engine_pool,
)

from omlx.request import MemoryVictim, Request, SamplingParams

GB = 1024**3


@pytest.fixture
def enforcer(mock_engine_pool):  # noqa: F811
    # Hard watermark (95 GB) below the ceiling (100 GB): hard is not an
    # emergency until usage passes the ceiling by the emergency margin.
    return _make_enforcer(
        mock_engine_pool,
        ceiling=100 * GB,
        soft_threshold=0.85,
        hard_threshold=0.95,
    )


async def tick(enforcer, usage_gb):
    with patch("omlx.process_memory_enforcer.mx") as mock_mx:
        mock_mx.get_active_memory.return_value = usage_gb * GB
        mock_mx.get_cache_memory.return_value = 0
        await enforcer._check_and_enforce()


def _busy(enforcer, engine):
    entry = _make_entry("big-model", engine=engine)
    entry.in_use = 1
    enforcer._engine_pool._entries = {"big-model": entry}
    enforcer._engine_pool._find_lru_victim.return_value = None
    return entry


class _ScriptedScheduler:
    """The scheduler surface the enforcer's park and pause paths use."""

    def __init__(self, victims):
        self.victims = list(victims)
        self.paused: list[str] = []
        self.status: dict[str, str] = {}
        self.park_answer = False

    def request_park_prefill(self):
        return self.park_answer

    def select_decode_pause_victim(self):
        return self.victims[0] if self.victims else None

    def request_decode_pause(self, request_id):
        self.paused.append(request_id)
        self.victims = [v for v in self.victims if v.request_id != request_id]
        self.status[request_id] = "pending"
        return True

    def decode_pause_status(self, request_id):
        return self.status.get(request_id, "done")

    def note_pressure_poll(self, level):
        pass

    def adjust_store_cache_cap(self, *args, **kwargs):
        pass


def _engine(scheduler, abort_victims=()):
    engine = MagicMock()
    engine.has_active_requests.return_value = False
    engine.scheduler = scheduler
    engine.abort_all_requests = AsyncMock(return_value=3)
    engine.abort_one_request_for_memory = AsyncMock(
        side_effect=[MemoryVictim(rid, 9_000, GB, 1) for rid in abort_victims] + [None]
    )
    engine.memory_victim_pending = MagicMock(return_value=False)
    return engine


def _victim(rid, kept=2):
    return MemoryVictim(rid, 90_000, 6 * GB, kept)


@pytest.fixture(autouse=True)
def single_victim_on(monkeypatch):
    monkeypatch.setenv("OMLX_MEMORY_ABORT_SINGLE_VICTIM", "true")


class TestEnforcerPausesDecodingVictims:
    @pytest.mark.asyncio
    async def test_hard_pressure_pauses_one_decoding_row_instead_of_aborting(
        self, enforcer, caplog
    ):
        scheduler = _ScriptedScheduler([_victim("big"), _victim("next", kept=1)])
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)

        assert scheduler.paused == ["big"]
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
        assert (
            "Hard memory pressure: pausing decoding victim big instead of aborting"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_a_prefill_park_comes_before_any_decode_pause(self, enforcer):
        scheduler = _ScriptedScheduler([_victim("big")])
        scheduler.park_answer = True
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 97)

        assert scheduler.paused == []
        engine.abort_all_requests.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_second_victim_while_the_pause_or_its_store_drains(self, enforcer):
        scheduler = _ScriptedScheduler([_victim("big"), _victim("next", kept=1)])
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 97)  # pause requested
        for _ in range(3):  # not run yet, then its chain still being written
            await tick(enforcer, 97)
        assert scheduler.paused == ["big"]

        scheduler.status["big"] = "done"
        await tick(enforcer, 97)  # seen done: one settle poll
        assert scheduler.paused == ["big"]
        await tick(enforcer, 97)  # settled and still hard: the next one
        assert scheduler.paused == ["big", "next"]
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_pause_that_never_drains_is_bounded(self, enforcer):
        scheduler = _ScriptedScheduler([_victim("big"), _victim("next", kept=1)])
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 97)
        for _ in range(enforcer._SINGLE_VICTIM_SETTLE_MAX_POLLS):
            await tick(enforcer, 97)
        assert scheduler.paused == ["big"]
        await tick(enforcer, 97)
        assert scheduler.paused == ["big", "next"]

    @pytest.mark.asyncio
    async def test_a_declined_pause_falls_back_to_the_single_victim_abort_at_once(
        self, enforcer, caplog
    ):
        scheduler = _ScriptedScheduler([_victim("big")])
        engine = _engine(scheduler, abort_victims=["big"])
        _busy(enforcer, engine)

        await tick(enforcer, 97)
        assert scheduler.paused == ["big"]
        scheduler.status["big"] = "declined"
        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)

        assert "decode pause of big was declined" in caplog.text
        engine.abort_one_request_for_memory.assert_awaited_once_with(spare_parked=True)
        engine.abort_all_requests.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nothing_pausable_falls_back_to_the_single_victim_abort(
        self, enforcer
    ):
        scheduler = _ScriptedScheduler([])
        engine = _engine(scheduler, abort_victims=["big"])
        _busy(enforcer, engine)

        await tick(enforcer, 97)

        engine.abort_one_request_for_memory.assert_awaited_once_with(spare_parked=True)
        engine.abort_all_requests.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_pause_while_an_aborted_victim_drains(self, enforcer):
        scheduler = _ScriptedScheduler([])
        engine = _engine(scheduler, abort_victims=["big"])
        engine.memory_victim_pending.return_value = True
        _busy(enforcer, engine)
        await tick(enforcer, 97)  # aborts ``big``
        scheduler.victims = [_victim("other")]

        await tick(enforcer, 97)

        assert scheduler.paused == []
        assert engine.abort_one_request_for_memory.await_count == 1

    @pytest.mark.asyncio
    async def test_emergency_aborts_everything_without_pausing(self, enforcer):
        # A pause of ~1 GB cannot bring 103 GB under the 100 GB ceiling, so
        # the emergency pause-first does not apply.
        scheduler = _ScriptedScheduler([MemoryVictim("big", 9_000, GB, 2)])
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 103)

        assert scheduler.paused == []
        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)

    @pytest.mark.asyncio
    async def test_a_failing_scheduler_hook_falls_back_to_the_abort(
        self, enforcer, caplog
    ):
        scheduler = _ScriptedScheduler([_victim("big")])
        scheduler.request_decode_pause = MagicMock(side_effect=RuntimeError("gone"))
        engine = _engine(scheduler, abort_victims=["big"])
        _busy(enforcer, engine)

        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)

        assert "Decode pause request failed on 'big-model': gone" in caplog.text
        engine.abort_one_request_for_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_scheduler_without_the_pause_api_aborts_as_before(self, enforcer):
        scheduler = SimpleNamespace(
            request_park_prefill=lambda: False,
            note_pressure_poll=lambda level: None,
            adjust_store_cache_cap=lambda *a, **k: None,
        )
        engine = _engine(scheduler, abort_victims=["big"])
        _busy(enforcer, engine)

        await tick(enforcer, 97)

        engine.abort_one_request_for_memory.assert_awaited_once()


# --- Real enforcer, real scheduler -------------------------------------------------

PROMPTS = {"a": dp.PROMPT, "b": dp.OTHER, "c": park.PROMPT_D}
MAX_TOKENS = 60


def _submit_three(rig):
    rows = []
    for i, (rid, tokens) in enumerate(PROMPTS.items()):
        request = Request(
            request_id=rid,
            prompt=list(tokens),
            sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0),
            arrival_time=float(i + 1),
        )
        rig.scheduler.add_request(request)
        rows.append(request)
    return rows


def _baseline(tmp_path):
    rig = dp.make_rig(tmp_path / "baseline", park_enabled=False)
    try:
        rows = _submit_three(rig)
        outputs = park.run_to_completion(rig, rows, limit=10 * MAX_TOKENS)
        return {r.request_id: list(r.output_token_ids) for r in rows}, outputs
    finally:
        rig.close()


def _decoding(rig, rows, count=5, outputs=None):
    scheduler = rig.scheduler
    outputs = [] if outputs is None else outputs
    for _ in range(400):
        if all(
            len(r.output_token_ids) >= count and r.request_id in scheduler.running
            for r in rows
        ):
            return
        outputs.extend(park.step(rig))
    raise AssertionError("rows are not all decoding")


def _rig_engine(rig):
    scheduler = rig.scheduler
    engine = SimpleNamespace(
        scheduler=scheduler,
        has_active_requests=lambda: False,
        abort_all_requests=AsyncMock(return_value=0),
        abort_one_request_for_memory=AsyncMock(return_value=None),
        memory_victim_pending=lambda rid: False,
    )
    return engine


def _text(outputs, rid):
    return "".join(o.new_text for o in outputs if o.request_id == rid)


@pytest.mark.asyncio
async def test_three_decoding_rows_at_hard_pressure_pause_one_and_finish_identically(
    tmp_path, enforcer, caplog
):
    expected, base_outputs = _baseline(tmp_path)
    rig = dp.make_rig(tmp_path / "paused")
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        rows = _submit_three(rig)
        outputs = []
        _decoding(rig, rows, outputs=outputs)

        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)
        (paused_id,) = [
            rid for rid in PROMPTS if rid in scheduler._pending_decode_pauses
        ]
        assert paused_id != "a"  # the oldest admitted request is protected
        assert f"pausing decoding victim {paused_id} instead of aborting" in caplog.text
        outputs.extend(park.step(rig))

        paused = scheduler.requests[paused_id]
        others = [r for r in rows if r.request_id != paused_id]
        assert paused.parked and paused_id not in scheduler.running
        assert all(r.request_id in scheduler.running for r in others)
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()

        # Still hard: no second row is paused while the first one settles,
        # and the others keep decoding.
        before = {r.request_id: len(r.output_token_ids) for r in others}
        frozen = len(paused.output_token_ids)
        await tick(enforcer, 97)
        outputs.extend(park.step(rig, 3))
        assert not scheduler._pending_decode_pauses
        assert all(len(r.output_token_ids) > before[r.request_id] for r in others)
        assert len(paused.output_token_ids) == frozen

        # Pressure clears: the paused row resumes and every row completes.
        for _ in range(4):
            await tick(enforcer, 1)
            outputs.extend(park.step(rig))
        outputs.extend(park.run_to_completion(rig, rows, limit=10 * MAX_TOKENS))

        for row in rows:
            rid = row.request_id
            assert list(row.output_token_ids) == expected[rid]
            assert dp.streamed_tokens(outputs, rid) == expected[rid]
            assert _text(outputs, rid) == _text(base_outputs, rid)
            (final,) = [o for o in outputs if o.request_id == rid and o.finished]
            assert final.finish_reason == "length" and not final.error
        stats = scheduler.get_park_stats()
        assert stats["decode_pauses"] == stats["decode_resumes"] == 1
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_nothing_pausable_on_the_real_scheduler_aborts_a_single_victim(
    tmp_path, enforcer, monkeypatch
):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        engine.abort_one_request_for_memory = AsyncMock(
            return_value=MemoryVictim("b", 30, 0, 2)
        )
        _busy(enforcer, engine)
        rows = _submit_three(rig)
        _decoding(rig, rows)
        monkeypatch.setattr(
            scheduler, "_decode_pause_exclusion_reason", lambda request: "test"
        )

        await tick(enforcer, 97)

        assert not scheduler._pending_decode_pauses
        engine.abort_one_request_for_memory.assert_awaited_once_with(spare_parked=True)
        engine.abort_all_requests.assert_not_awaited()
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_emergency_on_the_real_scheduler_still_aborts_everything(
    tmp_path, enforcer
):
    rig = dp.make_rig(tmp_path)
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        rows = _submit_three(rig)
        _decoding(rig, rows)

        await tick(enforcer, 103)
        park.step(rig)

        assert not scheduler._pending_decode_pauses
        assert not any(r.parked for r in rows)
        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)
        engine.abort_one_request_for_memory.assert_not_awaited()
    finally:
        rig.close()
