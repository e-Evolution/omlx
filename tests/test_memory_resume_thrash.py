# SPDX-License-Identifier: Apache-2.0
"""A paused request does not thrash between pause and resume.

Forced pressure (Qwen3.8-27B at a 42 GB ceiling): one 130k request decoded
alone above the hard watermark (34-37 GB against 33.6 GB); the other two were
held out of the batch. The enforcer paused it, nothing else was admitted, and
the nothing-admitted escape of the park hold resumed it on the next step
without asking whether it fit. Pressure returned, it was paused again, and
after ``park budget spent (3/3)`` the single-victim abort killed it. Not one
"waiting: need" or "fits in free" line was logged in any branch run: the
memory-aware resume never ran.

The resume now measures every parked head:

- beside admitted work it resumes only under the hard watermark minus a
  hysteresis margin, and a head paused twice within a short window waits for
  an admitted row to leave or free memory to rise (anti-thrash);
- with nothing admitted, a head that fits only under the ceiling runs alone
  (solo): nothing else is admitted beside it and the enforcer does not pause
  or abort it below the emergency threshold. The emergency brake stays.
- decode pauses have their own, larger budget.

The real-rig tests wire the real enforcer to the real scheduler (tiny Qwen3.5
hybrid, real paged SSD stores) and script the memory readings that run's log
shows; only the byte sizes are scripted, at the scale of the real run.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park
from test_memory_decode_pause import (
    _busy,
    _engine,
    _ScriptedScheduler,
    _victim,
    tick,
)
from test_process_memory_enforcer import (  # noqa: F401 (fixture)
    _make_enforcer,
    mock_engine_pool,
)
from test_scheduler_park_resume_priority import _parked_request
from test_scheduler_prefill_memory_guard import _make_scheduler

import omlx.scheduler as scheduler_module
from omlx.request import Request, SamplingParams
from omlx.settings import GlobalSettings, SchedulerSettings

GB = 1024**3


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
    # Ceiling 100 GB, soft 85 GB, hard watermark 95 GB: hard is not an
    # emergency until usage passes the ceiling by the emergency margin. The
    # physical cap is not binding (200 GB), whatever the host's memory.
    enforcer = _make_enforcer(
        mock_engine_pool,
        ceiling=100 * GB,
        soft_threshold=0.85,
        hard_threshold=0.95,
    )
    enforcer._get_abort_limit_bytes = lambda: 200 * GB
    return enforcer


# --- Resume line and anti-thrash (scheduler unit) --------------------------


@pytest.fixture
def lined(monkeypatch):
    """A live scheduler with the enforcer's lines: 100 GB ceiling, 95 GB hard
    watermark; ``usage[0]`` is its usage, ``admitted`` its admitted ids."""
    scheduler = _make_scheduler()
    scheduler._memory_hard_limit_bytes = 100 * GB
    scheduler._memory_hard_watermark_bytes = 95 * GB
    scheduler._memory_abort_limit_bytes = 200 * GB  # physical cap not binding
    scheduler.note_pressure_poll("soft")
    usage = [0]
    admitted = {"ids": frozenset({"other"})}
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda **_kw: usage[0])
    monkeypatch.setattr(
        scheduler, "_num_admitted_requests", lambda: len(admitted["ids"])
    )
    monkeypatch.setattr(scheduler, "_admitted_request_ids", lambda: admitted["ids"])
    return scheduler, usage, admitted


def _fit_line(scheduler):
    return scheduler._park_resume_lines()[0]


def test_resume_beside_admitted_work_stays_under_the_hard_watermark_margin(lined):
    scheduler, usage, _ = lined
    head = _parked_request()
    need = scheduler._parked_resume_need_bytes(head)
    watermark = 95 * GB
    hard_line = watermark - int(
        watermark * scheduler_module.PARK_RESUME_HARD_MARGIN_FRACTION
    )
    assert _fit_line(scheduler) == hard_line

    # The old line (admission line minus 2%) would resume this head.
    admission = scheduler._admission_limit_bytes()
    old_line = admission - int(admission * scheduler_module.PARK_RESUME_MARGIN_FRACTION)
    usage[0] = old_line - need
    assert scheduler._park_hold_applies(head) is True

    usage[0] = hard_line - need + 1
    assert scheduler._park_hold_applies(head) is True
    usage[0] = hard_line - need
    assert scheduler._park_hold_applies(head) is False


def test_a_head_that_fits_only_alone_waits_while_others_run(lined):
    scheduler, usage, admitted = lined
    head = _parked_request()
    need = scheduler._parked_resume_need_bytes(head)
    usage[0] = _fit_line(scheduler) - need + GB  # over the fit line, under the ceiling

    assert scheduler._park_hold_applies(head) is True
    assert scheduler._solo_request_id is None

    admitted["ids"] = frozenset()
    assert scheduler._park_hold_applies(head) is False
    assert scheduler._solo_request_id == head.request_id


def test_a_head_paused_twice_quickly_waits_for_a_row_to_leave(lined, caplog):
    scheduler, usage, admitted = lined
    head = _parked_request()
    now = time.monotonic()
    head.park_times = [now - 5, now - 1]
    head.park_count = 2
    usage[0] = 0  # fits easily

    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        assert scheduler._park_hold_applies(head) is True
        assert scheduler._park_hold_applies(head) is True
    assert "was paused 2 times within" in caplog.text

    admitted["ids"] = frozenset()  # the row it waited on left
    admitted["ids"] = frozenset({"another"})
    assert scheduler._park_hold_applies(head) is False


def test_a_thrashing_head_resumes_when_free_memory_rises_enough(lined):
    scheduler, usage, _ = lined
    head = _parked_request()
    now = time.monotonic()
    head.park_times = [now - 5, now - 1]
    head.park_count = 2
    usage[0] = 40 * GB
    assert scheduler._park_hold_applies(head) is True  # records the picture

    delta = int(_fit_line(scheduler) * scheduler_module.PARK_THRASH_FREE_DELTA_FRACTION)
    usage[0] -= delta
    assert scheduler._park_hold_applies(head) is True  # not more than the delta
    usage[0] -= 1
    assert scheduler._park_hold_applies(head) is False


def test_pauses_outside_the_window_are_not_thrash(lined):
    scheduler, usage, _ = lined
    head = _parked_request()
    now = time.monotonic()
    window = scheduler_module.PARK_THRASH_WINDOW_S
    head.park_times = [now - window - 5, now - 1]
    usage[0] = 0
    assert scheduler._park_hold_applies(head) is False


def test_a_lone_head_paused_twice_runs_solo_even_when_it_fits(lined):
    """The estimate said it fits, yet it was paused again while alone: the
    estimate under-counts it, so it runs solo instead of looping."""
    scheduler, usage, admitted = lined
    admitted["ids"] = frozenset()
    head = _parked_request()
    now = time.monotonic()
    head.park_times = [now - 5, now - 1]
    usage[0] = 0
    assert scheduler._park_hold_applies(head) is False
    assert scheduler._solo_request_id == head.request_id


def test_hard_pressure_still_holds_a_lone_head(lined):
    scheduler, usage, admitted = lined
    admitted["ids"] = frozenset()
    head = _parked_request()
    usage[0] = 0
    scheduler.note_pressure_poll("hard")
    assert scheduler._park_hold_applies(head) is True


# --- Budgets ----------------------------------------------------------------


def test_prefill_parks_and_decode_pauses_have_separate_budgets():
    scheduler = _make_scheduler()
    request = _parked_request()
    request.parked = False
    request.park_count = 5
    request.decode_pause_count = 3  # two prefill parks
    assert "budget" not in (scheduler._park_exclusion_reason(request) or "")
    request.park_count = 6
    assert scheduler._park_exclusion_reason(request) == "park budget spent (3/3)"


def test_decode_pause_budget_defaults_to_eight_and_is_configurable(
    tmp_path, monkeypatch
):
    assert scheduler_module.SchedulerConfig().max_decode_pauses_per_request == 8
    assert SchedulerSettings().max_decode_pauses_per_request == 8
    custom = SchedulerSettings.from_dict({"max_decode_pauses_per_request": 12})
    assert custom.max_decode_pauses_per_request == 12
    for bad in (0, -1, "many", True):
        junk = SchedulerSettings.from_dict({"max_decode_pauses_per_request": bad})
        assert junk.max_decode_pauses_per_request == 8

    monkeypatch.setenv("OMLX_MAX_DECODE_PAUSES_PER_REQUEST", "5")
    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.scheduler.max_decode_pauses_per_request == 5
    assert settings.to_scheduler_config().max_decode_pauses_per_request == 5
    settings.scheduler.max_decode_pauses_per_request = 0
    assert any("max_decode_pauses_per_request" in e for e in settings.validate())


# --- Enforcer: the solo request is exempt below the emergency threshold ---


class _SoloScheduler(_ScriptedScheduler):
    def __init__(self, victims, solo):
        super().__init__(victims)
        self.solo = solo

    def memory_pressure_solo_request(self):
        return self.solo


class TestEnforcerLeavesTheSoloRequestAlone:
    @pytest.mark.asyncio
    async def test_hard_pressure_neither_pauses_nor_aborts_it(self, enforcer, caplog):
        scheduler = _SoloScheduler([_victim("a", kept=0)], solo="a")
        engine = _engine(scheduler, abort_victims=["a"])
        _busy(enforcer, engine)

        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)
            await tick(enforcer, 98)

        assert scheduler.paused == []
        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
        assert caplog.text.count("a runs alone (solo) on 'big-model'") == 1

    @pytest.mark.asyncio
    async def test_the_emergency_brake_still_aborts_it(self, enforcer):
        # The emergency brake measures against the physical cap (#4312); here
        # it is the ceiling.
        enforcer._get_abort_limit_bytes = lambda: 100 * GB
        scheduler = _SoloScheduler([_victim("a", kept=0)], solo="a")
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 103)

        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)

    @pytest.mark.asyncio
    async def test_another_busy_engine_removes_the_exemption(self, enforcer):
        scheduler = _SoloScheduler([_victim("a", kept=0)], solo="a")
        engine = _engine(scheduler)
        entry = _busy(enforcer, engine)
        other = SimpleNamespace(
            engine=SimpleNamespace(has_active_requests=lambda: True),
            is_pinned=True,
            is_loading=False,
            in_use=1,
            last_access=0.0,
            model_id="other",
        )
        enforcer._engine_pool._entries = {"big-model": entry, "other": other}
        # Usage is read from the scripted MLX sample, not the executor cache.
        enforcer._has_active_requests = lambda: False

        await tick(enforcer, 97)

        assert scheduler.paused == ["a"]

    @pytest.mark.asyncio
    async def test_a_failing_probe_gives_no_exemption(self, enforcer):
        scheduler = _SoloScheduler([_victim("a", kept=0)], solo="a")
        scheduler.memory_pressure_solo_request = lambda: 1 / 0
        engine = _engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 97)

        assert scheduler.paused == ["a"]


# --- Real enforcer, real scheduler: the forced-pressure sequence -----------

MAX_TOKENS = 60


def _rig(tmp_path, *, park_enabled=True, **config):
    rig = park._Rig(
        tmp_path,
        dp._tokenizer(),
        split=False,
        commit_blocks=1,
        max_blocks=dp.MAX_BLOCKS,
        memory_pressure_park=park_enabled,
        max_parks_per_request=3,
        park_resume_ok_polls=3,
        **config,
    )
    rig.scheduler.note_pressure_poll("soft")
    return rig


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
    rig = _rig(tmp_path / "baseline", park_enabled=False)
    try:
        rows = [
            _submit(rig, rid, tokens, float(i + 1))
            for i, (rid, tokens) in enumerate(prompts.items())
        ]
        park.run_to_completion(rig, rows, limit=20 * MAX_TOKENS)
        return {r.request_id: list(r.output_token_ids) for r in rows}
    finally:
        rig.close()


def _rig_engine(rig):
    return SimpleNamespace(
        scheduler=rig.scheduler,
        has_active_requests=lambda: False,
        abort_all_requests=AsyncMock(return_value=0),
        abort_one_request_for_memory=AsyncMock(return_value=None),
        memory_victim_pending=lambda rid: False,
    )


def _decoding_ids(scheduler):
    return list(scheduler.running)


def _script(rig, monkeypatch, *, reading, need):
    """Scripted memory: ``reading(decoding_ids)`` is the usage both the
    enforcer and the scheduler read; ``need`` maps a request to its resume
    need. The run's lone request: 21.6 GB of weights plus 8-10 GB of KV, close
    to the hard watermark."""
    scheduler = rig.scheduler

    def usage(**_kw):
        return reading(_decoding_ids(scheduler))

    monkeypatch.setattr(scheduler, "_current_usage_bytes", usage)
    monkeypatch.setattr(
        scheduler, "_parked_resume_need_bytes", lambda r: need[r.request_id]
    )
    return lambda: usage() / GB


async def _drive(rig, enforcer, gb, rows, limit=40 * MAX_TOKENS):
    outputs = []
    for _ in range(limit):
        if all(r.is_finished() for r in rows):
            break
        await tick(enforcer, gb())
        outputs.extend(park.step(rig))
    for future in list(rig.scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.scheduler._drain_pending_async_removes()
    return outputs


def _assert_no_abort(engine, rows, expected, outputs):
    engine.abort_one_request_for_memory.assert_not_awaited()
    engine.abort_all_requests.assert_not_awaited()
    for row in rows:
        rid = row.request_id
        assert list(row.output_token_ids) == expected[rid]
        assert dp.streamed_tokens(outputs, rid) == expected[rid]
        (final,) = [o for o in outputs if o.request_id == rid and o.finished]
        assert final.finish_reason == "length" and not final.error


@pytest.mark.asyncio
async def test_lone_request_over_the_hard_watermark_pauses_once_then_runs_solo(
    tmp_path, enforcer, monkeypatch, caplog
):
    """pause -> resume -> pressure -> pause used to loop to the abort."""
    prompts = {"a": dp.PROMPT}
    expected = _baseline(tmp_path, prompts)
    rig = _rig(tmp_path / "solo")
    try:
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        # Decoding alone: 97 GB, over the 95 GB hard watermark; paused: 20 GB.
        gb = _script(
            rig,
            monkeypatch,
            reading=lambda ids: 97 * GB if ids else 20 * GB,
            need={"a": 74 * GB},  # 20 + 74 > 90.25 (fit line), < 98 (ceiling)
        )
        (row,) = [_submit(rig, "a", dp.PROMPT, 1.0)]

        with caplog.at_level(logging.INFO):
            outputs = await _drive(rig, enforcer, gb, [row])

        scheduler = rig.scheduler
        _assert_no_abort(engine, [row], expected, outputs)
        assert row.decode_pause_count == 1
        stats = scheduler.get_park_stats()
        assert stats["decode_pauses"] == stats["decode_resumes"] == 1
        assert "Resuming parked request a alone (solo)" in caplog.text
        assert "a runs alone (solo)" in caplog.text
        assert "budget spent" not in caplog.text
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_requests_that_fit_one_at_a_time_all_complete_without_aborts(
    tmp_path, enforcer, monkeypatch
):
    """The forced-pressure decode run: three long requests, one admitted at a time,
    each over the hard watermark while it decodes. Total demand fits
    sequentially, so every request completes and nothing is aborted."""
    prompts = {"a": dp.PROMPT, "b": dp.OTHER, "c": park.PROMPT_D}
    expected = _baseline(tmp_path, prompts)
    rig = _rig(tmp_path / "sequential", max_num_seqs=1)
    try:
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        gb = _script(
            rig,
            monkeypatch,
            reading=lambda ids: 97 * GB if ids else 20 * GB,
            need={"a": 74 * GB, "b": 72 * GB, "c": 73 * GB},
        )
        rows = [
            _submit(rig, rid, tokens, float(i + 1))
            for i, (rid, tokens) in enumerate(prompts.items())
        ]

        outputs = await _drive(rig, enforcer, gb, rows)

        _assert_no_abort(engine, rows, expected, outputs)
        assert all(r.decode_pause_count <= 1 for r in rows)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_a_row_paused_twice_waits_for_the_other_row_instead_of_thrashing(
    tmp_path, enforcer, monkeypatch, caplog
):
    """Two rows over the watermark: the newer one is paused, fits beside the
    older one, resumes, and pressure returns. After its second pause it waits
    until the older row finishes, then completes; nothing is aborted."""
    prompts = {"a": dp.PROMPT, "b": dp.OTHER}
    expected = _baseline(tmp_path, prompts)
    rig = _rig(tmp_path / "thrash")
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        decoding = {"both": False}

        def reading(ids):
            if len(ids) >= 2:
                decoding["both"] = True
                return 97 * GB
            return 60 * GB if ids else 20 * GB

        gb = _script(
            rig,
            monkeypatch,
            reading=reading,
            need={"a": 30 * GB, "b": 30 * GB},  # 60 + 30 = 90 <= 90.25
        )
        rows = [_submit(rig, "a", dp.PROMPT, 1.0), _submit(rig, "b", dp.OTHER, 2.0)]

        with caplog.at_level(logging.INFO):
            outputs = await _drive(rig, enforcer, gb, rows)

        _assert_no_abort(engine, rows, expected, outputs)
        b = rows[1]
        assert b.decode_pause_count == 2
        assert rows[0].decode_pause_count == 0  # the oldest row is protected
        assert "Parked request b was paused 2 times within" in caplog.text
        # It resumed only after ``a`` finished.
        assert len(b.output_token_ids) == MAX_TOKENS
        assert scheduler.get_park_stats()["decode_resumes"] == 2
    finally:
        rig.close()
