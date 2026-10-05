# SPDX-License-Identifier: Apache-2.0
"""A media prompt whose images are all cached prefills its text suffix chunked.

Any image in a conversation's history gives the request fused embeddings, and
those used to force the synchronous external prefill loop even when every
image sat inside the restored prefix. Such a request froze concurrent decode
and could not be parked. With ``OMLX_MEDIA_TEXT_SUFFIX_CHUNKED`` on, a request
whose last media token is cached prefills the uncached text suffix through the
chunked text path with the request's mRoPE delta bound per chunk.

The tests drive the real scheduler, the real ``VLMModelAdapter`` and real
paged-SSD stores. The language model is a small Qwen3.5 hybrid that also adds
a term computed from every mRoPE plane of its ``position_ids``, so a position
that differs between the two paths changes the logits and the greedy tokens.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_scheduler_prefill_progress_commit import BLOCK, _Rig

from omlx.models.vlm import VLMModelAdapter
from omlx.request import Request, SamplingParams

FLAG = "OMLX_MEDIA_TEXT_SUFFIX_CHUNKED"
IMAGE_TOKEN = 30
IMAGE_START = 2
IMAGE_LEN = 4  # one 1x2x2 grid image
MEDIA_END = IMAGE_START + IMAGE_LEN
DELTA = -2  # Qwen get_rope_index: max(image positions) + 1 - MEDIA_END
IMAGE_HASH = "img-a"


def _text(seed, n):
    return [3 + (seed * 5 + i * 7) % 26 for i in range(n)]


TURN1 = _text(1, 25)  # 24 prefill tokens: the stored prefix is block aligned
TURN1[IMAGE_START:MEDIA_END] = [IMAGE_TOKEN] * IMAGE_LEN
TURN2 = TURN1 + _text(2, 18)


def _mrope_positions(n, image_start=IMAGE_START):
    """(3, 1, n) positions the way Qwen get_rope_index lays out one image."""
    planes = [[], [], []]
    for k in range(image_start):
        for p in planes:
            p.append(k)
    for dh, dw in ((0, 0), (0, 1), (1, 0), (1, 1)):
        planes[0].append(image_start)
        planes[1].append(image_start + dh)
        planes[2].append(image_start + dw)
    for k in range(image_start + IMAGE_LEN, n):
        for p in planes:
            p.append(k + DELTA)
    return mx.array([[p] for p in planes], dtype=mx.int32)


class _MRoPELanguageModel(nn.Module):
    """Tiny hybrid whose output depends on all three mRoPE planes."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.model = inner.language_model.model
        self._position_ids = None
        self._rope_deltas = None
        self.unpositioned_calls = 0
        hidden = self.model.embed_tokens.weight.shape[1]
        self._planes = mx.random.normal((3, hidden), key=mx.random.key(7)) * 0.2

    def make_cache(self):
        return self.inner.make_cache()

    def _position_term(self, position_ids, batch, length, cache):
        if position_ids is None:
            self.unpositioned_calls += 1
            offset = next(c.offset for c in cache if hasattr(c, "offset"))
            position_ids = mx.arange(length)[None] + offset
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None], (3, position_ids.shape[0], length)
            )
        freqs = mx.array([0.3, 0.7, 1.1])[:, None, None]
        waves = mx.sin(position_ids.astype(mx.float32) * freqs)  # (3, B, T)
        term = (waves[..., None] * self._planes[:, None, None, :]).sum(axis=0)
        return mx.broadcast_to(term, (batch, length, term.shape[-1]))

    def __call__(self, inputs, inputs_embeds=None, cache=None, position_ids=None, **_):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(inputs)
        batch, length = inputs_embeds.shape[:2]
        hidden = inputs_embeds + self._position_term(
            position_ids, batch, length, cache
        ).astype(inputs_embeds.dtype)
        return self.inner(inputs, cache=cache, input_embeddings=hidden)


class _FakeQwenVL(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.language_model = _MRoPELanguageModel(inner)
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            text_config=SimpleNamespace(
                rope_scaling={"mrope_section": [8, 4, 4]}, rope_parameters=None
            ),
        )


def _rig(tmp_path, tokenizer, **overrides):
    rig = _Rig(tmp_path, tokenizer, split=False, commit_blocks=1, **overrides)
    rig.adapter = VLMModelAdapter(_FakeQwenVL(rig.model))
    rig.scheduler.model = rig.adapter
    return rig


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "true")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv(FLAG, "false")


def _embeds(rig, tokens):
    lm = rig.adapter._language_model
    embeds = lm.model.embed_tokens(mx.array([tokens]))
    image = mx.random.normal((1, IMAGE_LEN, embeds.shape[-1]), key=mx.random.key(3))
    return mx.concatenate(
        [embeds[:, :IMAGE_START], image.astype(embeds.dtype), embeds[:, MEDIA_END:]],
        axis=1,
    )


def _media_request(rig, rid, tokens, *, max_tokens=4, arrival=None, **overrides):
    extra = {
        "position_ids": _mrope_positions(len(tokens)),
        "_captured_rope_deltas": mx.array([[DELTA]]),
    }
    fields = dict(
        vlm_inputs_embeds=_embeds(rig, tokens),
        vlm_extra_kwargs=extra,
        vlm_image_hash=IMAGE_HASH,
        vlm_cache_key_start=IMAGE_START,
        vlm_media_end=MEDIA_END,
    )
    fields.update(overrides)
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        **({"arrival_time": arrival} if arrival is not None else {}),
        **fields,
    )
    rig.scheduler.add_request(request)
    return request


def _text_request(rig, rid, tokens, *, max_tokens, arrival):
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        arrival_time=arrival,
    )
    rig.scheduler.add_request(request)
    return request


def _step(rig, n=1):
    outputs = []
    for _ in range(n):
        outputs.extend(rig.scheduler.step().outputs)
        rig.drain()
    return outputs


def _run(rig, requests, limit=200, on_step=None):
    for _ in range(limit):
        if all(
            r.is_finished() or r.request_id not in rig.scheduler.requests
            for r in requests
        ):
            break
        _step(rig)
        if on_step is not None:
            on_step()
    for future in list(rig.scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    rig.scheduler._drain_pending_async_removes()
    rig.drain()


class _Spy:
    """Record which prefill path each admission took."""

    def __init__(self, rig, monkeypatch):
        self.external_media = []
        self.begun = []
        scheduler = rig.scheduler
        real_external = scheduler._do_external_prefill
        real_begin = scheduler._begin_prefill

        def external(request, tokens, cache, vlm_embeds=None):
            if vlm_embeds is not None:
                self.external_media.append(request.request_id)
            return real_external(request, tokens, cache, vlm_embeds=vlm_embeds)

        def begin(request, tokens, cache):
            self.begun.append((request.request_id, request.rope_deltas))
            return real_begin(request, tokens, cache)

        monkeypatch.setattr(scheduler, "_do_external_prefill", external)
        monkeypatch.setattr(scheduler, "_begin_prefill", begin)


def _warm_turn(rig):
    """Turn one prefills cold (external) and stores its media-keyed prefix."""
    first = _media_request(rig, "t1", TURN1, max_tokens=1)
    _run(rig, [first])
    return first


def _probe(rig, tokens):
    return rig.prefix.fetch_cache(
        "probe",
        list(tokens),
        extra_keys=(IMAGE_HASH,),
        extra_key_token_start=IMAGE_START,
    )


# --- 1. Routing --------------------------------------------------------------


def test_last_image_cached_takes_the_chunked_text_path(
    tmp_path, mock_tokenizer, flag_on, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        spy = _Spy(rig, monkeypatch)
        second = _media_request(rig, "t2", TURN2)
        _step(rig)
        assert second.cached_tokens >= MEDIA_END
        assert second.media_text_suffix is True
        assert spy.external_media == []
        # The delta is bound before the first chunk forms its positions.
        assert spy.begun == [("t2", float(DELTA))]
        assert "t2" in rig.scheduler._prefill_states
        # Embeddings stay on the request (park re-admit and OOM requeue).
        assert second.vlm_inputs_embeds is not None
        _run(rig, [second])
        assert second.vlm_inputs_embeds is None  # freed at the first token
    finally:
        rig.close()


def _route_after_warm(rig, monkeypatch, **overrides):
    _warm_turn(rig)
    spy = _Spy(rig, monkeypatch)
    second = _media_request(rig, "t2", TURN2, **overrides)
    _step(rig)
    return spy, second


@pytest.mark.parametrize(
    "case",
    ["image_in_suffix", "unknown_media_end", "no_captured_delta", "failed_proof"],
)
def test_fallbacks_take_the_external_media_path(
    tmp_path, mock_tokenizer, flag_on, monkeypatch, case
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        overrides = {}
        if case == "image_in_suffix":
            overrides["vlm_media_end"] = len(TURN2) - 2
        elif case == "unknown_media_end":
            overrides["vlm_media_end"] = None
        elif case == "no_captured_delta":
            overrides["vlm_extra_kwargs"] = {
                "position_ids": _mrope_positions(len(TURN2))
            }
        elif case == "failed_proof":
            # A suffix plane that is not text positions (an image grid there).
            pid = _mrope_positions(len(TURN2))
            planes = pid.tolist()
            planes[1][0][-3] += 1
            overrides["vlm_extra_kwargs"] = {
                "position_ids": mx.array(planes, dtype=mx.int32),
                "_captured_rope_deltas": mx.array([[DELTA]]),
            }
        spy, second = _route_after_warm(rig, monkeypatch, **overrides)
        assert second.cached_tokens > 0
        assert spy.external_media == ["t2"]
        assert not second.media_text_suffix
        assert "t2" not in rig.scheduler._prefill_states
    finally:
        rig.close()


def test_flag_off_keeps_the_external_media_path(
    tmp_path, mock_tokenizer, flag_off, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        spy, second = _route_after_warm(rig, monkeypatch)
        assert second.cached_tokens >= MEDIA_END
        assert spy.external_media == ["t2"]
        assert not second.media_text_suffix
    finally:
        rig.close()


def test_flag_is_off_by_default(tmp_path, mock_tokenizer, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        spy, _ = _route_after_warm(rig, monkeypatch)
        assert spy.external_media == ["t2"]
    finally:
        rig.close()


def test_minimum_prefill_prefix_keeps_the_external_path(
    tmp_path, mock_tokenizer, flag_on, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        spy = _Spy(rig, monkeypatch)
        second = _media_request(rig, "t2", TURN2)
        route = rig.scheduler._media_text_suffix_route
        monkeypatch.setattr(
            type(rig.adapter), "minimum_prefill_prefix", lambda self, tokens: 3
        )
        assert route(second, list(TURN2[20:])) is False
        monkeypatch.setattr(
            type(rig.adapter), "minimum_prefill_prefix", lambda self, tokens: 0
        )
        second.cached_tokens = 20
        assert route(second, list(TURN2[20:])) is True
        assert route(second, list(TURN2[-1:])) is False  # one token left
        assert spy.external_media == []
    finally:
        rig.close()


# --- 2. Position equality ----------------------------------------------------


def test_chunked_positions_equal_the_external_slice(
    tmp_path, mock_tokenizer, flag_on, monkeypatch
):
    """Every suffix forward sees the positions the external slice would pass."""
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        lm = rig.adapter._language_model
        seen = []
        real_call = type(lm).__call__

        def record(
            self, inputs, inputs_embeds=None, cache=None, position_ids=None, **kw
        ):
            offset = next(c.offset for c in cache if hasattr(c, "offset"))
            if isinstance(offset, int) and inputs.shape[1] > 1:
                seen.append((offset, position_ids))
            return real_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                cache=cache,
                position_ids=position_ids,
                **kw,
            )

        monkeypatch.setattr(type(lm), "__call__", record)
        second = _media_request(rig, "t2", TURN2)
        _run(rig, [second])
        assert second.media_text_suffix is True
        full = _mrope_positions(len(TURN2))
        chunks = [(o, p) for o, p in seen if o >= second.cached_tokens]
        assert chunks, "the suffix never reached the model"
        for offset, positions in chunks:
            n = positions.shape[-1]
            expected = full[:, :, offset : offset + n]
            if positions.ndim == 2:
                positions = mx.broadcast_to(positions[None], expected.shape)
            assert positions.tolist() == expected.tolist()
        assert lm.unpositioned_calls == 0
    finally:
        rig.close()


# --- 3. Greedy equality end to end, 5. D1 keys and next-turn lookup -----------


def _second_turn(tmp_path, tokenizer, monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    rig = _rig(tmp_path, tokenizer)
    _warm_turn(rig)
    second = _media_request(rig, "t2", TURN2, max_tokens=6)
    _run(rig, [second])
    return rig, second


def _restored_arrays(rig, tokens):
    table, _ = _probe(rig, tokens)
    assert table is not None
    hashes = [rig.paged.allocated_blocks[b].block_hash for b in table.block_ids]
    cache = rig.prefix.reconstruct_cache(table)
    arrays = []
    for layer in cache:
        arrays.extend(a for a in layer.state if isinstance(a, mx.array))
    return table.num_tokens, hashes, arrays


def test_greedy_tokens_and_final_cache_equal_the_external_path(
    tmp_path, mock_tokenizer, monkeypatch
):
    on, on_req = _second_turn(tmp_path / "on", mock_tokenizer, monkeypatch, "true")
    off, off_req = _second_turn(tmp_path / "off", mock_tokenizer, monkeypatch, "false")
    try:
        assert on_req.media_text_suffix is True
        assert not off_req.media_text_suffix
        assert on_req.cached_tokens == off_req.cached_tokens >= MEDIA_END
        assert len(on_req.output_token_ids) == 6
        assert on_req.output_token_ids == off_req.output_token_ids
        assert on_req.rope_deltas == off_req.rope_deltas == float(DELTA)

        next_turn = TURN2 + on_req.output_token_ids + _text(3, 8)
        n_on, hashes_on, arrays_on = _restored_arrays(on, next_turn)
        n_off, hashes_off, arrays_off = _restored_arrays(off, next_turn)
        # The next turn hits everything the second turn prefilled.
        assert n_on == n_off >= (len(TURN2) - 1) // BLOCK * BLOCK
        assert hashes_on == hashes_off
        assert len(arrays_on) == len(arrays_off) > 0
        for a, b in zip(arrays_on, arrays_off):
            assert mx.allclose(a, b, atol=1e-5)
    finally:
        on.close()
        off.close()


# --- 4. Fairness ---------------------------------------------------------------


def test_a_decode_row_emits_tokens_between_suffix_chunks(
    tmp_path, mock_tokenizer, flag_on
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        talker = _text_request(rig, "dec", _text(7, 9), max_tokens=40, arrival=1.0)
        _step(rig, 3)
        assert "dec" in rig.scheduler.running
        second = _media_request(rig, "t2", TURN2, arrival=2.0)
        interleaved = []

        def watch():
            if second.request_id in rig.scheduler._prefill_states:
                interleaved.append(len(talker.output_token_ids))

        _run(rig, [second], on_step=watch)
        assert second.media_text_suffix is True
        assert len(interleaved) >= 2
        assert interleaved[-1] > interleaved[0]  # decode advanced mid-prefill
    finally:
        rig.close()


# --- 6. Park --------------------------------------------------------------------


def test_text_suffix_media_request_parks_and_resumes_with_its_commits(
    tmp_path, mock_tokenizer, monkeypatch
):
    baseline, base_req = _second_turn(
        tmp_path / "base", mock_tokenizer, monkeypatch, "true"
    )
    baseline.close()
    rig = _rig(
        tmp_path / "park",
        mock_tokenizer,
        memory_pressure_park=True,
        park_resume_ok_polls=1,
    )
    try:
        rig.scheduler.note_pressure_poll("soft")
        _warm_turn(rig)
        second = _media_request(rig, "t2", TURN2, max_tokens=6)
        admitted_cached = []
        for _ in range(40):
            _step(rig)
            if second.request_id in rig.scheduler._prefill_states:
                admitted_cached.append(second.cached_tokens)
                if rig.table_tokens("t2") >= second.cached_tokens + BLOCK:
                    break
        assert second.media_text_suffix is True
        assert rig.scheduler._park_exclusion_reason(second) is None
        assert rig.scheduler.request_park_prefill() is True  # hard pressure
        _step(rig)
        # Parked at the step start; alone, it is re-admitted at once.
        assert second.park_count == 1
        committed = second.park_committed_tokens
        assert committed > admitted_cached[0]
        assert second.vlm_inputs_embeds is not None  # kept for the re-admit

        rig.scheduler.note_pressure_poll("ok")
        _run(rig, [second])
        assert second.park_count == 1
        assert second.cached_tokens == committed  # resumed from its commits
        assert second.media_text_suffix is True
        assert second.output_token_ids == base_req.output_token_ids
    finally:
        rig.close()


def test_external_media_prefill_stays_excluded_from_park(
    tmp_path, mock_tokenizer, flag_on
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        media = _media_request(rig, "m", TURN1)
        rig.scheduler.waiting.clear()
        rig.scheduler.prefilling.append(media)
        rig.scheduler._prefill_states["m"] = object()
        assert "multimodal" in rig.scheduler._park_exclusion_reason(media)
        media.media_text_suffix = True
        assert rig.scheduler._park_exclusion_reason(media) is None
    finally:
        rig.close()


# --- 7. Abort, OOM requeue, teardown -------------------------------------------


def test_abort_mid_suffix_cleans_up(tmp_path, mock_tokenizer, flag_on):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        second = _media_request(rig, "t2", TURN2)
        _step(rig)
        assert "t2" in rig.scheduler._prefill_states
        rig.scheduler.abort_request("t2")
        _step(rig, 2)
        assert "t2" not in rig.scheduler._prefill_states
        assert "t2" not in rig.scheduler.requests
        assert not rig.scheduler.prefilling
        assert second.media_text_suffix is True
    finally:
        rig.close()


def test_oom_requeue_goes_back_to_the_external_path(
    tmp_path, mock_tokenizer, flag_on, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer)
    try:
        _warm_turn(rig)
        spy = _Spy(rig, monkeypatch)
        scheduler = rig.scheduler
        real_chunk = scheduler._step_prefill_chunk
        failed = []

        def chunk(state):
            if not failed:
                failed.append(state.request.request_id)
                raise RuntimeError("Memory limit exceeded during chunked prefill")
            return real_chunk(state)

        monkeypatch.setattr(scheduler, "_step_prefill_chunk", chunk)
        second = _media_request(rig, "t2", TURN2)
        _step(rig)
        assert failed == ["t2"]
        assert second.prefill_oom_retries == 1
        assert second.cached_tokens == 0
        assert second.vlm_inputs_embeds is not None
        _run(rig, [second])
        assert spy.external_media == ["t2"]  # cold retry: images are uncached
        assert not second.media_text_suffix
        assert second.output_token_ids
    finally:
        rig.close()


def test_teardown_with_a_suffix_in_flight(tmp_path, mock_tokenizer, flag_on):
    rig = _rig(tmp_path, mock_tokenizer)
    _warm_turn(rig)
    _media_request(rig, "t2", TURN2)
    _step(rig)
    assert "t2" in rig.scheduler._prefill_states
    rig.close()  # shutdown with a suffix chunk state and commits in flight
    assert not rig.scheduler._prefill_commits


# --- Media end from the processor's token ids (engine/vlm.py) -------------------


def _config(**ids):
    return SimpleNamespace(**ids)


@pytest.mark.parametrize(
    "tokens,config,expected",
    [
        ([1, 9, 9, 2, 3], _config(image_token_id=9), 3),
        ([1, 9, 2, 8, 8, 4, 5], _config(image_token_id=9, video_token_id=8), 5),
        ([7, 1, 2], _config(image_token_index=9, audio_token_id=7), 1),
        ([7, 7, 1], _config(audio_token_index=7), 2),
        ([1, 2, 3], _config(image_token_id=9), None),
        ([1, 9, 3], _config(), None),
    ],
)
def test_media_end_is_the_index_after_the_last_media_token(tokens, config, expected):
    from omlx.engine.vlm import _vlm_media_end

    assert _vlm_media_end(tokens, config) == expected


def test_engine_core_threads_media_end_onto_the_request():
    import asyncio
    from unittest.mock import MagicMock

    from omlx.engine_core import EngineCore

    core = EngineCore.__new__(EngineCore)
    core.scheduler = MagicMock()
    core.scheduler._specprefill_draft_model = None
    core.config = SimpleNamespace(stream_interval=1)
    core._mlx_executor = None
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._wake_engine_loop = lambda: None

    asyncio.run(core.add_request([1, 2, 3], request_id="r", vlm_media_end=2))
    request = core.scheduler.add_request.call_args.args[0]
    assert request.vlm_media_end == 2


async def test_vlm_engine_streams_the_media_end_to_the_core():
    from unittest.mock import AsyncMock, MagicMock

    from test_vlm_engine import _make_loaded_engine

    engine = _make_loaded_engine(model_type="test-vlm")
    engine._vlm_model.config.image_token_id = 9
    engine._engine = MagicMock()
    engine._engine.add_request = AsyncMock(return_value="req-1")
    engine._engine.abort_request = AsyncMock(return_value=True)

    async def one_output(_request_id):
        yield SimpleNamespace(
            output_text="ok",
            new_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine.stream_outputs = one_output
    async for _ in engine.stream_generate(
        [1, 9, 9, 2, 3], vlm_inputs_embeds=mx.zeros((1, 5, 4))
    ):
        pass
    assert engine._engine.add_request.call_args.kwargs["vlm_media_end"] == 3

    # Text-only turns carry no media end (the core default None applies).
    async for _ in engine.stream_generate([1, 9, 2]):
        pass
    assert "vlm_media_end" not in engine._engine.add_request.call_args.kwargs


async def test_vlm_engine_generate_forwards_the_media_end():
    from unittest.mock import AsyncMock

    from test_vlm_engine import _make_loaded_engine

    engine = _make_loaded_engine(model_type="test-vlm")
    engine._vlm_model.config.video_token_id = 8
    output = SimpleNamespace(
        output_text="ok",
        prompt_tokens=4,
        completion_tokens=1,
        finish_reason="stop",
        tool_calls=None,
        cached_tokens=0,
        first_token_at=None,
    )
    engine._engine = SimpleNamespace(generate=AsyncMock(return_value=output))
    await engine.generate([8, 8, 1, 2], vlm_inputs_embeds=mx.zeros((1, 4, 4)))
    assert engine._engine.generate.call_args.kwargs["vlm_media_end"] == 2
