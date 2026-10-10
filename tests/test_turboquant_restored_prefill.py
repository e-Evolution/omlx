# SPDX-License-Identifier: Apache-2.0
"""Suffix prefill after a restored TurboQuant prefix.

A paged-SSD hit restores TurboQuant (TQ) layers as quantized
``TurboQuantKVCache`` objects (#1842). Prefilling a long uncached suffix over
that quantized prefix makes every chunk of every TQ layer attend through the
tiled quantized-attention path, which is several times slower than the dense
SDPA a cold prefill uses at the same positions. These tests drive both prefill
loops (external and resumable/chunked) with a fake model that records the
cache type each layer sees per chunk.
"""

import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_vlm.turboquant import TurboQuantKVCache

from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

STEP = 6
PREFIX = 8
HEADS = 2
DIM = 32


def _kv_for(token_ids: list[int]) -> mx.array:
    """Deterministic per-token K/V rows, shape (1, HEADS, n, DIM)."""
    pos = mx.array(token_ids, dtype=mx.float32)[:, None]
    freqs = mx.arange(1, DIM + 1, dtype=mx.float32)[None, :] * 0.37
    rows = mx.sin(pos * freqs) + 0.1 * mx.cos(pos * freqs * 1.7)
    heads = mx.stack([rows, -0.5 * rows + 0.25], axis=0)
    return heads[None].astype(mx.float16)


def _restored_cache(tq_bits: float | None = 4.0) -> list:
    """A restored [TQ, plain KV] prefix of PREFIX tokens (skip-last layout)."""
    kv = _kv_for(list(range(PREFIX)))
    first = KVCache()
    last = KVCache()
    # Small growth steps leave the restored buffers exactly full, as a
    # paged-SSD restore does, so the prefill reservation has work to plan.
    first.step = last.step = 4
    first.update_and_fetch(kv, kv)
    last.update_and_fetch(kv, kv)
    if tq_bits is None:
        mx.eval(first.keys, last.keys)
        return [first, last]
    restored = TurboQuantKVCache.from_cache(first, bits=tq_bits)
    mx.eval(restored.keys, restored.values, last.keys, last.values)
    return [restored, last]


def _state_arrays(cache_obj: TurboQuantKVCache) -> list[mx.array]:
    keys, values = cache_obj.state
    return [a for _, a in tree_flatten([keys, values])]


def _token_slice(arrays: list[mx.array], start: int, stop: int) -> list[mx.array]:
    return [a[:, :, start:stop] for a in arrays]


@pytest.fixture
def harness():
    model = MagicMock()
    model.layers = []
    del model._omlx_prefill
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=STEP,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler.block_aware_cache = None
    scheduler._memory_limit_bytes = 0
    scheduler._turboquant_kv_bits = 4.0
    scheduler._turboquant_skip_last = True
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [42]
    seen: list[list[str]] = []
    seen_ids: list[list[int]] = []

    def forward(tokens, *, cache, **kwargs):
        row = tokens.tolist()[0]
        seen.append([type(c).__name__ for c in cache])
        seen_ids.append([id(c) for c in cache])
        kv = _kv_for(row)
        for layer in cache:
            layer.update_and_fetch(kv, kv)

    model.side_effect = forward
    get_prefill_tracker().clear()
    with ExitStack() as patches:
        patches.enter_context(patch("omlx.scheduler._sync_and_clear_cache"))
        patches.enter_context(
            patch("omlx.scheduler.get_phys_footprint", return_value=0)
        )
        patches.enter_context(
            patch.object(
                scheduler, "_adaptive_chunk_size", side_effect=lambda n, **_: n
            )
        )
        patches.enter_context(
            patch.object(
                scheduler, "_guard_prefill_chunk", side_effect=lambda n, **_: n
            )
        )
        patches.enter_context(
            patch.object(scheduler, "_supports_skip_lm_head", return_value=False)
        )
        patches.enter_context(patch.object(scheduler, "_record_chunk_transient"))
        patches.enter_context(
            patch.object(scheduler, "_maybe_record_fixed_state_bytes")
        )
        patches.enter_context(
            patch.object(scheduler, "_should_clear_after_chunk", return_value=False)
        )
        patches.enter_context(
            patch.object(scheduler, "_turboquant_eligible", return_value=True)
        )
        yield SimpleNamespace(scheduler=scheduler, seen=seen, seen_ids=seen_ids)
    get_prefill_tracker().clear()


def _run_prefill(scheduler: Scheduler, path: str, cache: list, suffix: int) -> list:
    """Prefill PREFIX cached + ``suffix`` new tokens; return the decode cache."""
    tokens = list(range(PREFIX + suffix))
    request = Request(
        request_id=f"tq-{path}-{suffix}",
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request.cached_tokens = PREFIX
    request.remaining_tokens = tokens[PREFIX:]
    if path == "external":
        returned, last_token = scheduler._do_external_prefill(
            request, request.remaining_tokens, cache
        )
        assert last_token == tokens[-1:]
        return returned
    state = scheduler._begin_prefill(request, request.remaining_tokens, cache)
    state.sampler = MagicMock()
    state.sm = MagicMock()
    state.per_row_lps = []
    while not scheduler._step_prefill_chunk(state):
        pass
    scheduler._insert_prefilled_request(request, state, [])
    return scheduler.batch_generator.insert.call_args.kwargs["caches"][0]


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_long_suffix_runs_dense_and_requantizes_onto_restored_codes(harness, path):
    """A multi-chunk suffix sees a dense prefix; decode gets TQ again.

    The restored prefix is dequantized once, every suffix chunk attends a
    plain KVCache, and the decode cache is the original TQ object with its
    restored prefix codes untouched (no second quantization) plus the suffix
    quantized exactly like a cold prefill's end-of-prefill conversion.
    """
    scheduler = harness.scheduler
    cache = _restored_cache()
    restored, plain = cache
    prefix_codes = [mx.array(a) for a in _state_arrays(restored)]
    suffix = 2 * STEP + 1  # 12 prefill tokens: two full chunks

    original_dequantize = TurboQuantKVCache.dequantize
    with patch.object(
        TurboQuantKVCache,
        "dequantize",
        autospec=True,
        side_effect=original_dequantize,
    ) as dequantize:
        decode_cache = _run_prefill(scheduler, path, cache, suffix)

    assert harness.seen == [["KVCache", "KVCache"], ["KVCache", "KVCache"]]
    assert dequantize.call_count == 1
    assert dequantize.call_args.args[0] is restored

    total = PREFIX + suffix - 1
    assert decode_cache[0] is restored
    assert decode_cache[0].offset == total
    assert decode_cache[1] is plain
    assert type(decode_cache[1]) is KVCache
    assert decode_cache[1].offset == total

    after = _state_arrays(decode_cache[0])
    for before_arr, after_arr in zip(
        prefix_codes, _token_slice(after, 0, PREFIX), strict=True
    ):
        assert mx.array_equal(before_arr, after_arr).item()

    exact = KVCache()
    full = _kv_for(list(range(total)))
    exact.update_and_fetch(full, full)
    cold = TurboQuantKVCache.from_cache(exact, bits=4.0)
    for want, got in zip(
        _token_slice(_state_arrays(cold), PREFIX, total),
        _token_slice(after, PREFIX, total),
        strict=True,
    ):
        assert mx.array_equal(want, got).item()


@pytest.mark.parametrize("path", ["external", "resumable"])
@pytest.mark.parametrize("suffix", [STEP, STEP + 1])
def test_short_suffix_keeps_quantized_prefix(harness, path, suffix):
    """A suffix that fits in one prefill chunk stays on the quantized path."""
    scheduler = harness.scheduler
    cache = _restored_cache()
    restored, plain = cache

    original_dequantize = TurboQuantKVCache.dequantize
    with patch.object(
        TurboQuantKVCache,
        "dequantize",
        autospec=True,
        side_effect=original_dequantize,
    ) as dequantize:
        decode_cache = _run_prefill(scheduler, path, cache, suffix)

    assert harness.seen == [["TurboQuantKVCache", "KVCache"]]
    assert dequantize.call_count == 0
    assert decode_cache[0] is restored
    assert decode_cache[0].offset == PREFIX + suffix - 1
    assert decode_cache[1] is plain


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_restored_dense_prefix_is_untouched_until_cold_epilogue(harness, path):
    """Non-TQ restored layers keep their objects; the cold epilogue still runs."""
    scheduler = harness.scheduler
    cache = _restored_cache(tq_bits=None)
    first, plain = cache

    decode_cache = _run_prefill(scheduler, path, cache, 2 * STEP + 1)

    assert harness.seen == [["KVCache", "KVCache"], ["KVCache", "KVCache"]]
    assert harness.seen_ids == [[id(first), id(plain)], [id(first), id(plain)]]
    assert isinstance(decode_cache[0], TurboQuantKVCache)
    assert decode_cache[0] is not first
    assert decode_cache[1] is plain


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_turboquant_off_leaves_restored_quantized_layers_alone(harness, path):
    """Without TurboQuant KV configured no layer is swapped or converted."""
    scheduler = harness.scheduler
    scheduler._turboquant_kv_bits = None
    cache = _restored_cache()
    restored, plain = cache

    decode_cache = _run_prefill(scheduler, path, cache, 2 * STEP + 1)

    assert harness.seen == [
        ["TurboQuantKVCache", "KVCache"],
        ["TurboQuantKVCache", "KVCache"],
    ]
    assert decode_cache[0] is restored
    assert decode_cache[1] is plain


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_dense_prefix_is_visible_to_prefill_capacity_reserve(harness, path, caplog):
    """The dequantized layers are planned by the #3933 prefill reservation."""
    scheduler = harness.scheduler
    cache = _restored_cache()
    suffix = 2 * STEP + 1

    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        _run_prefill(scheduler, path, cache, suffix)

    horizon = PREFIX + suffix
    reserve_lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Reserved prefill capacity")
    ]
    assert reserve_lines
    assert reserve_lines[0].startswith(
        f"Reserved prefill capacity for {horizon} tokens across 2 caches"
    )
    assert any(
        "dequantized 1 restored TurboQuant layer" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_dense_prefix_that_does_not_fit_keeps_quantized_path(harness, path, caplog):
    """Under memory pressure the compact quantized prefix is kept."""
    scheduler = harness.scheduler
    cache = _restored_cache()
    restored, _ = cache

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(Scheduler, "_prefill_peak_target", return_value=1 << 20)
        )
        stack.enter_context(
            patch.object(scheduler, "_current_usage_bytes", return_value=(1 << 20) - 64)
        )
        stack.enter_context(caplog.at_level(logging.INFO, logger="omlx.scheduler"))
        decode_cache = _run_prefill(scheduler, path, cache, 2 * STEP + 1)

    assert harness.seen == [
        ["TurboQuantKVCache", "KVCache"],
        ["TurboQuantKVCache", "KVCache"],
    ]
    assert decode_cache[0] is restored
    assert any(
        r.getMessage().startswith("Kept restored TurboQuant prefix quantized")
        for r in caplog.records
    )


@pytest.mark.parametrize("path", ["external", "resumable"])
def test_all_turboquant_layout_takes_dense_dtype_from_the_embedding(harness, path):
    """With no plain KV layer left (skip-last off), use the activation dtype.

    The admission estimate reads the activation dtype from ``model.dtype`` or
    the embedding weight; the dense prefix must use the same dtype a cold
    prefill writes.
    """
    scheduler = harness.scheduler
    scheduler._turboquant_skip_last = False
    model = scheduler.model
    model.dtype = None
    model.model.embed_tokens.weight = mx.zeros((4, DIM), dtype=mx.float16)
    restored = _restored_cache()[0]
    dense_dtypes = []
    forward = model.side_effect

    def recording_forward(tokens, *, cache, **kwargs):
        dense_dtypes.append(
            [c.keys.dtype for c in cache if type(c) is KVCache and c.keys is not None]
        )
        forward(tokens, cache=cache, **kwargs)

    model.side_effect = recording_forward

    decode_cache = _run_prefill(scheduler, path, [restored], 2 * STEP + 1)

    assert harness.seen == [["KVCache"], ["KVCache"]]
    assert dense_dtypes[0] == [mx.float16]
    assert decode_cache[0] is restored
    assert decode_cache[0].offset == PREFIX + 2 * STEP
