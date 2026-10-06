# SPDX-License-Identifier: Apache-2.0
"""Memory pressure while a media prompt prefills in the external loop.

C7 image-decode (2026-10-05, 42 GB ceiling): a 130k image prompt prefilled
alone in the external loop while two requests waited in the queue. At the end
of its prefill usage reached 35.3 GB, over the 33.6 GB hard watermark but
under the ceiling, so it was NOT the emergency brake. The non-emergency hard
path found nothing to park ("multimodal prompt ... prefills in the external
loop") and nothing to pause; the single-victim selection found no live
request (the external prefill sits in neither ``running`` nor a chunked
prefill state, and queued requests hold no KV), so the enforcer fell back to
abort-all: "Aborted 2 requests due to memory pressure", then 1 more. Every
request was lost and nothing was freed by aborting the queued ones.

Now:

- a request in the external loop with nothing else admitted runs alone: the
  enforcer neither pauses nor aborts it below the emergency threshold;
- with a decoding row beside it, the decoding row is paused instead;
- with no live request at all, the non-emergency path aborts nothing;
- at emergency pressure one decode pause that brings usage back under the
  ceiling comes before abort-all, which stays the last resort;
- beside admitted work, a media prefill whose estimated peak crosses the hard
  watermark line waits at the head instead of spiking past it.

The tests drive the real enforcer and the real scheduler (tiny Qwen3.5 hybrid
behind an mlx-vlm style ``inputs_embeds`` call, real paged SSD stores); the
memory readings are scripted at the scale of the real run.
"""

import asyncio
import concurrent.futures
import functools
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park
from test_memory_decode_pause import _busy, _ScriptedScheduler, tick
from test_process_memory_enforcer import (  # noqa: F401 (fixture)
    _make_enforcer,
    mock_engine_pool,
)
from test_scheduler_prefill_commit_vlm import IMAGE_KEY, IMAGE_START, _VLMLikeModel
from test_scheduler_prefill_progress_commit import _Rig

from omlx.request import MemoryVictim, Request, SamplingParams

GB = 1024**3
MAX_TOKENS = 30
TEXT = dp.PROMPT
MEDIA = park.PROMPT_D


@pytest.fixture(autouse=True)
def single_victim_on(monkeypatch):
    monkeypatch.setenv("OMLX_MEMORY_ABORT_SINGLE_VICTIM", "true")


@pytest.fixture
def enforcer(mock_engine_pool):  # noqa: F811
    # Ceiling 100 GB, soft 85 GB, hard watermark 95 GB; emergency from 102 GB.
    return _make_enforcer(
        mock_engine_pool,
        ceiling=100 * GB,
        soft_threshold=0.85,
        hard_threshold=0.95,
    )


def _rig(tmp_path, *, park_enabled=True, max_num_seqs=None):
    config = {"max_num_seqs": max_num_seqs} if max_num_seqs else {}
    rig = _Rig(
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
    rig.wrapped = _VLMLikeModel(rig.model)
    rig.scheduler.model = rig.wrapped
    rig.scheduler.note_pressure_poll("soft")
    return rig


def _text(rig, rid, arrival, tokens=TEXT):
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0),
        arrival_time=arrival,
    )
    rig.scheduler.add_request(request)
    return request


def _media(rig, rid, arrival, tokens=MEDIA):
    embeds = rig.wrapped.embed(list(tokens))
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0),
        arrival_time=arrival,
        vlm_inputs_embeds=embeds,
        vlm_image_hash=IMAGE_KEY[0],
        vlm_cache_key_start=IMAGE_START,
    )
    rig.scheduler.add_request(request)
    return request


def _baseline(tmp_path, build):
    rig = _rig(tmp_path / "baseline", park_enabled=False)
    try:
        rows = build(rig)
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


class _Readings:
    """One scripted usage figure read by the enforcer and the scheduler."""

    def __init__(self, rig, monkeypatch, gb):
        self.gb = gb
        monkeypatch.setattr(
            rig.scheduler, "_current_usage_bytes", lambda **_kw: int(self.gb * GB)
        )


def _hold_external_prefill(rig, monkeypatch, rid):
    """Block the MLX thread inside ``rid``'s external prefill until released,
    so the enforcer can poll while it runs (as in production)."""
    entered, release = threading.Event(), threading.Event()
    scheduler = rig.scheduler
    original = scheduler._do_external_prefill

    def held(request, *args, **kwargs):
        if request.request_id == rid:
            entered.set()
            assert release.wait(30)
        return original(request, *args, **kwargs)

    monkeypatch.setattr(scheduler, "_do_external_prefill", held)
    return entered, release


class _MLXThread:
    """The scheduler's own thread: its MLX stream is thread-local, so the rig
    is built, stepped and closed there while the test polls the enforcer."""

    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="test-mlx"
        )

    async def run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, functools.partial(fn, *args, **kwargs)
        )

    def close(self):
        self._pool.shutdown(wait=True)


@pytest.fixture
def mlx_thread():
    thread = _MLXThread()
    yield thread
    thread.close()


def _drain_stores(rig):
    for future in list(rig.scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.scheduler._drain_pending_async_removes()


async def _finish(rig, enforcer, readings, rows, gb_when_done=20, mlx=None):
    outputs = []
    for _ in range(40 * MAX_TOKENS):
        if all(r.is_finished() for r in rows):
            break
        readings.gb = gb_when_done
        await tick(enforcer, readings.gb)
        if mlx is None:
            outputs.extend(park.step(rig))
        else:
            outputs.extend(await mlx.run(park.step, rig))
    if mlx is None:
        _drain_stores(rig)
    else:
        await mlx.run(_drain_stores, rig)
    return outputs


def _assert_completed(rows, expected, outputs):
    for row in rows:
        rid = row.request_id
        assert list(row.output_token_ids) == expected[rid], rid
        assert dp.streamed_tokens(outputs, rid) == expected[rid], rid
        (final,) = [o for o in outputs if o.request_id == rid and o.finished]
        assert final.finish_reason == "length" and not final.error


def _decoding(rig, rows, outputs):
    for _ in range(400):
        if all(r.request_id in rig.scheduler.running for r in rows):
            return
        outputs.extend(park.step(rig))
    raise AssertionError("rows are not all decoding")


# --- Lone external prefill: the C7 image-decode case ----------------------


@pytest.mark.asyncio
async def test_lone_external_media_prefill_at_hard_pressure_is_not_aborted(
    tmp_path, enforcer, monkeypatch, caplog, mlx_thread
):
    def build(rig):
        return [_media(rig, "m", 1.0), _text(rig, "q1", 2.0), _text(rig, "q2", 3.0)]

    expected = _baseline(tmp_path, build)
    rig = await mlx_thread.run(_rig, tmp_path / "lone", max_num_seqs=1)
    try:
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        readings = _Readings(rig, monkeypatch, 30)
        entered, release = _hold_external_prefill(rig, monkeypatch, "m")
        rows = await mlx_thread.run(build, rig)

        stepping = asyncio.create_task(mlx_thread.run(park.step, rig))
        assert await asyncio.to_thread(entered.wait, 30)
        readings.gb = 97  # the end-of-prefill spike: over hard, under the ceiling
        with caplog.at_level(logging.INFO):
            await tick(enforcer, 97)
            await tick(enforcer, 98)
        engine.abort_all_requests.assert_not_awaited()  # C7: abort-all here
        assert rig.scheduler.memory_pressure_solo_request() == "m"
        release.set()
        outputs = list(await stepping)
        outputs.extend(await _finish(rig, enforcer, readings, rows, mlx=mlx_thread))

        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
        assert "m runs alone (solo)" in caplog.text
        _assert_completed(rows, expected, outputs)
    finally:
        await mlx_thread.run(rig.close)


# --- A decoding row beside the external prefill is paused instead ---------


@pytest.mark.asyncio
async def test_spike_during_external_prefill_pauses_the_decoding_row(
    tmp_path, enforcer, monkeypatch, caplog, mlx_thread
):
    def build(rig):
        return [_text(rig, "a", 1.0), _media(rig, "m", 2.0)]

    expected = _baseline(tmp_path, build)
    rig = await mlx_thread.run(_rig, tmp_path / "beside")
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        readings = _Readings(rig, monkeypatch, 30)
        outputs = []
        a = await mlx_thread.run(_text, rig, "a", 1.0)
        await mlx_thread.run(_decoding, rig, [a], outputs)
        entered, release = _hold_external_prefill(rig, monkeypatch, "m")
        m = await mlx_thread.run(_media, rig, "m", 2.0)

        stepping = asyncio.create_task(mlx_thread.run(park.step, rig))
        assert await asyncio.to_thread(entered.wait, 30)
        readings.gb = 97  # the spike, over hard and under the ceiling
        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 97)
        assert scheduler.memory_pressure_solo_request() is None
        assert scheduler._pending_decode_pauses == {"a"}
        assert "pausing decoding victim a instead of aborting" in caplog.text
        release.set()
        outputs.extend(await stepping)
        outputs.extend(await mlx_thread.run(park.step, rig))  # the pause runs
        assert scheduler.requests["a"].parked

        outputs.extend(await _finish(rig, enforcer, readings, [a, m], mlx=mlx_thread))

        engine.abort_one_request_for_memory.assert_not_awaited()
        engine.abort_all_requests.assert_not_awaited()
        assert a.decode_pause_count == 1
        _assert_completed([a, m], expected, outputs)
    finally:
        await mlx_thread.run(rig.close)


# --- Emergency: one pause first, abort-all as the last resort --------------


def _emergency_victim(scheduler, monkeypatch, est_gb):
    """The decode-pause victim's estimate at the real run's scale."""
    select = scheduler.select_decode_pause_victim

    def scaled():
        victim = select()
        if victim is None:
            return None
        return MemoryVictim(victim.request_id, victim.tokens, int(est_gb * GB), 1)

    monkeypatch.setattr(scheduler, "select_decode_pause_victim", scaled)


@pytest.mark.asyncio
async def test_emergency_pauses_one_row_when_that_gets_back_under_the_ceiling(
    tmp_path, enforcer, monkeypatch, caplog
):
    def build(rig):
        return [_text(rig, "a", 1.0), _text(rig, "b", 2.0, dp.OTHER)]

    expected = _baseline(tmp_path, build)
    rig = _rig(tmp_path / "emergency")
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        readings = _Readings(rig, monkeypatch, 30)
        rows = build(rig)
        outputs = []
        _decoding(rig, rows, outputs)
        _emergency_victim(scheduler, monkeypatch, est_gb=6)

        readings.gb = 103  # beyond ceiling + 2 GB: emergency
        with caplog.at_level(logging.WARNING):
            await tick(enforcer, 103)
        assert scheduler._pending_decode_pauses == {"b"}  # the oldest is kept
        assert "Emergency memory pressure: pausing decoding request b" in caplog.text
        engine.abort_all_requests.assert_not_awaited()

        outputs.extend(park.step(rig))  # the pause runs
        readings.gb = 97  # back under the ceiling
        outputs.extend(await _finish(rig, enforcer, readings, rows))

        engine.abort_all_requests.assert_not_awaited()
        engine.abort_one_request_for_memory.assert_not_awaited()
        _assert_completed(rows, expected, outputs)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_emergency_aborts_all_when_no_pause_gets_under_the_ceiling(
    tmp_path, enforcer, monkeypatch
):
    rig = _rig(tmp_path)
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        readings = _Readings(rig, monkeypatch, 30)
        rows = [_text(rig, "a", 1.0), _text(rig, "b", 2.0, dp.OTHER)]
        _decoding(rig, rows, [])
        _emergency_victim(scheduler, monkeypatch, est_gb=1)

        readings.gb = 103
        await tick(enforcer, 103)

        assert not scheduler._pending_decode_pauses
        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)
    finally:
        rig.close()


@pytest.mark.asyncio
async def test_emergency_aborts_all_when_nothing_is_pausable(
    tmp_path, enforcer, monkeypatch
):
    rig = _rig(tmp_path)
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        readings = _Readings(rig, monkeypatch, 30)
        rows = [_text(rig, "a", 1.0), _text(rig, "b", 2.0, dp.OTHER)]
        _decoding(rig, rows, [])
        monkeypatch.setattr(
            scheduler, "_decode_pause_exclusion_reason", lambda request: "test"
        )

        readings.gb = 103
        await tick(enforcer, 103)

        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)
    finally:
        rig.close()


class TestEmergencyPauseFirstBounds:
    @pytest.mark.asyncio
    async def test_a_pause_still_pending_after_the_bound_gives_way_to_abort_all(
        self, enforcer
    ):
        scheduler = _ScriptedScheduler([MemoryVictim("b", 9_000, 6 * GB, 1)])
        engine = _scripted_engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 103)  # pause requested
        for _ in range(enforcer._EMERGENCY_PAUSE_MAX_POLLS):
            await tick(enforcer, 103)  # still pending: waits
        engine.abort_all_requests.assert_not_awaited()
        await tick(enforcer, 103)

        assert scheduler.paused == ["b"]
        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)

    @pytest.mark.asyncio
    async def test_growing_usage_does_not_wait_for_the_pause(self, enforcer):
        scheduler = _ScriptedScheduler([MemoryVictim("b", 9_000, 6 * GB, 1)])
        engine = _scripted_engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 103)
        await tick(enforcer, 105)

        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)

    @pytest.mark.asyncio
    async def test_a_declined_pause_gives_way_to_abort_all(self, enforcer):
        scheduler = _ScriptedScheduler([MemoryVictim("b", 9_000, 6 * GB, 1)])
        engine = _scripted_engine(scheduler)
        _busy(enforcer, engine)

        await tick(enforcer, 103)
        scheduler.status["b"] = "declined"
        await tick(enforcer, 103)

        engine.abort_all_requests.assert_awaited_once_with(spare_parked=False)


def _scripted_engine(scheduler):
    engine = SimpleNamespace(
        scheduler=scheduler,
        has_active_requests=lambda: False,
        abort_all_requests=AsyncMock(return_value=2),
        abort_one_request_for_memory=AsyncMock(return_value=None),
        memory_victim_pending=lambda rid: False,
    )
    return engine


# --- Transient reservation before an external media prefill ---------------


@pytest.mark.asyncio
async def test_media_prefill_whose_peak_crosses_the_hard_line_waits_beside_work(
    tmp_path, enforcer, monkeypatch, caplog
):
    def build(rig):
        return [_text(rig, "a", 1.0), _media(rig, "m", 2.0)]

    expected = _baseline(tmp_path, build)
    rig = _rig(tmp_path / "reserve")
    try:
        scheduler = rig.scheduler
        engine = _rig_engine(rig)
        _busy(enforcer, engine)
        await tick(enforcer, 30)  # propagate the lines
        readings = _Readings(rig, monkeypatch, 30)
        outputs = []
        a = _text(rig, "a", 1.0)
        _decoding(rig, [a], outputs)

        # Usage under the soft line (admission is open) and a media prefill
        # whose KV lands between the resume line and the admission line: the
        # plain preflight passes it, the reservation does not.
        fit_line = scheduler._park_resume_lines()[0]
        admission = scheduler._admission_limit_bytes()
        readings.gb = 80
        kv = (fit_line + admission) // 2 - 80 * GB
        monitor = scheduler.memory_monitor
        monkeypatch.setattr(
            monitor, "estimate_resident_kv_bytes", lambda n, chunk_tokens=1: kv
        )
        m = _media(rig, "m", 2.0)

        with caplog.at_level(logging.INFO):
            outputs.extend(park.step(rig, 3))
        assert "m" not in scheduler.running
        assert [r.request_id for r in scheduler.waiting] == ["m"]
        assert "Deferring media prefill m" in caplog.text

        outputs.extend(await _finish(rig, enforcer, readings, [a, m]))
        engine.abort_all_requests.assert_not_awaited()
        engine.abort_one_request_for_memory.assert_not_awaited()
        _assert_completed([a, m], expected, outputs)
    finally:
        rig.close()


def test_the_reservation_needs_the_park_feature(tmp_path, monkeypatch):
    rig = _rig(tmp_path, park_enabled=False)
    try:
        scheduler = rig.scheduler
        m = _media(rig, "m", 1.0)
        est = SimpleNamespace(estimated=10**15)
        assert scheduler._external_media_reservation_short(m, est) is False
    finally:
        rig.close()


def test_a_media_prompt_whose_images_are_cached_is_not_reserved(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_MEDIA_TEXT_SUFFIX_CHUNKED", "true")
    rig = _rig(tmp_path)
    try:
        scheduler = rig.scheduler
        scheduler._num_admitted_requests = lambda: 1
        scheduler._memory_hard_limit_bytes = 100 * GB
        m = _media(rig, "m", 1.0)
        m.vlm_media_end = 8
        m.cached_tokens = 8
        est = SimpleNamespace(estimated=10**15)
        assert scheduler._external_media_reservation_short(m, est) is False
        m.cached_tokens = 4
        assert scheduler._external_media_reservation_short(m, est) is True
    finally:
        rig.close()
