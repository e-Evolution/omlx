# SPDX-License-Identifier: Apache-2.0
"""Batched (B>1, left-padded) TurboQuant attention must equal per-row attention.

Shapes follow Qwen3.8-27B: 24 query heads, 4 KV heads, head_dim 256, 4-bit.
The cache is oMLX's ``BatchTurboQuantKVCache`` (rows end-aligned, zero-filled
left padding), driven the way the scheduler drives it: merge per-request
caches, then ``update_and_fetch`` the new decode token(s) and attend.
"""

import logging

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen3_5 import language as q35_lang
from mlx_vlm.turboquant import TurboQuantKVCache

from omlx.patches import turboquant_attention as tqa
from omlx.turboquant_kv import (
    BatchTurboQuantKVCache,
    carry_own_padding_mask,
    is_own_padding_mask,
)

pytestmark = pytest.mark.turboquant

N_Q, N_KV, D, BITS = 24, 4, 256, 4.0
SCALE = D**-0.5
LENGTHS = [2000, 6000, 9000]


@pytest.fixture(scope="module", autouse=True)
def _patched():
    tqa.apply_turboquant_attention_patch()
    tqa._PATCHED = True


def _kv(b, n, seed):
    mx.random.seed(seed)
    k = mx.random.normal((b, N_KV, n, D)).astype(mx.float16)
    v = mx.random.normal((b, N_KV, n, D)).astype(mx.float16)
    return k, v


def _row_cache(k, v):
    c = TurboQuantKVCache(bits=BITS)
    c.update_and_fetch(k, v)
    return c


def _build(lengths, new_len, seed=0):
    """Return (batch_cache, per_row_caches, queries, kb, vb) after appending
    `new_len` new tokens to every row (so each row holds lengths[i]+new_len)."""
    rows = []
    for i, n in enumerate(lengths):
        k, v = _kv(1, n, seed + i)
        rows.append(_row_cache(k, v))
    batch = BatchTurboQuantKVCache.merge(rows)
    mx.random.seed(seed + 100)
    nk = mx.random.normal((len(lengths), N_KV, new_len, D)).astype(mx.float16)
    nv = mx.random.normal((len(lengths), N_KV, new_len, D)).astype(mx.float16)
    q = mx.random.normal((len(lengths), N_Q, new_len, D)).astype(mx.float16)
    # The model builds its mask BEFORE the layers append the new tokens.
    batch._test_mask = batch.make_mask(new_len)
    keys, values = batch.update_and_fetch(nk, nv)
    row_states = []
    for i, c in enumerate(rows):
        rk, rv = c.update_and_fetch(nk[i : i + 1], nv[i : i + 1])
        row_states.append((c, rk, rv))
    return batch, row_states, q, keys, values


def _single(row_state, q_row, mask):
    c, rk, rv = row_state
    if q_row.shape[2] == 1:
        return c.decode_attention(q_row, keys_state=rk, values_state=rv, scale=SCALE)
    return tqa._decode_multirow_attention(c, q_row, rk, rv, SCALE)


def _batched(batch, q, keys, values, mask=None):
    return q35_lang._qwen3_5_left_padded_attention(
        q, keys, values, cache=batch, scale=SCALE, mask=mask
    )


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


@pytest.mark.parametrize("new_len", [1, 2, 3, 4])
def test_padded_batch_matches_single_row_attention(new_len):
    batch, rows, q, keys, values = _build(LENGTHS, new_len)
    out = _batched(batch, q, keys, values)
    mx.eval(out)
    assert out.shape == q.shape
    for i, rs in enumerate(rows):
        ref = _single(rs, q[i : i + 1], None)
        diff = _max_abs(out[i : i + 1], ref)
        assert diff < 2e-3, f"row {i} (len {LENGTHS[i]}) max|diff|={diff:.5f}"


def test_short_row_is_not_polluted_by_padding():
    # Row 0 has 7000 padded columns; zero-key columns must not enter softmax.
    batch, rows, q, keys, values = _build(LENGTHS, 1)
    out = _batched(batch, q, keys, values)
    ref = _single(rows[0], q[0:1], None)
    assert _max_abs(out[0:1], ref) < 2e-3
    ratio = mx.linalg.norm(out[0:1].astype(mx.float32)) / mx.linalg.norm(
        ref.astype(mx.float32)
    )
    assert ratio.item() == pytest.approx(1.0, abs=1e-2)


def test_equal_length_rows_unpadded_batch_matches_single_rows():
    batch, rows, q, keys, values = _build([3000, 3000], 1)
    out = _batched(batch, q, keys, values)
    for i, rs in enumerate(rows):
        assert _max_abs(out[i : i + 1], _single(rs, q[i : i + 1], None)) < 2e-3


def test_b1_cache_is_unchanged():
    k, v = _kv(1, 3000, 7)
    c = BatchTurboQuantKVCache([0], bits=BITS)
    c.update_and_fetch(k, v)
    mx.random.seed(11)
    q = mx.random.normal((1, N_Q, 1, D)).astype(mx.float16)
    nk, nv = _kv(1, 1, 8)
    keys, values = c.update_and_fetch(nk, nv)
    out = _batched(c, q, keys, values)
    ref = c.decode_attention(q, keys_state=keys, values_state=values, scale=SCALE)
    assert _max_abs(out, ref) < 2e-3


def test_logs_per_row_path_once(caplog):
    tqa._LOGGED_PATHS.clear()
    batch, rows, q, keys, values = _build([1500, 2500, 3500], 1)
    with caplog.at_level(logging.INFO, logger=tqa.logger.name):
        _batched(batch, q, keys, values)
        _batched(batch, q, keys, values)
    msgs = [
        r.getMessage() for r in caplog.records if "batched attention" in r.getMessage()
    ]
    assert msgs == ["TurboQuant batched attention: per-row fused path (B=3)"]


# ---------------------------------------------------------------------------
# patched_sdpa branch (explicit padding+causal mask from cache.make_mask)
# ---------------------------------------------------------------------------


def _sdpa(batch, q, keys, values, mask):
    from mlx_lm.models import base as mlx_base

    return mlx_base.scaled_dot_product_attention(q, keys, values, batch, SCALE, mask)


@pytest.mark.parametrize("new_len", [1, 3])
def test_patched_sdpa_array_mask_matches_single_row(new_len):
    batch, rows, q, keys, values = _build(LENGTHS, new_len)
    mask = batch._test_mask
    assert isinstance(mask, mx.array)
    tqa._LOGGED_PATHS.clear()
    out = _sdpa(batch, q, keys, values, mask)
    assert "fused" in tqa._LOGGED_PATHS  # the per-row branch engaged
    for i, rs in enumerate(rows):
        ref = _single(rs, q[i : i + 1], None)
        assert _max_abs(out[i : i + 1], ref) < 2e-3


# ---------------------------------------------------------------------------
# Fallback (always unpadded + correct)
# ---------------------------------------------------------------------------


def test_foreign_array_mask_is_honored_not_replaced():
    batch, rows, q, keys, values = _build([1200, 2400], 1)
    mask = mx.ones((2, 1, 1, 2401), dtype=mx.bool_)  # attends the padding too
    out = _sdpa(batch, q, keys, values, mask)
    dk, dv = batch.dequantize(keys, values)
    ref = mx.fast.scaled_dot_product_attention(
        q, dk.astype(q.dtype), dv.astype(q.dtype), scale=SCALE, mask=mask
    )
    assert _max_abs(out, ref) < 2e-3


def test_row_dequant_fallback_matches_single_row():
    batch, rows, q, keys, values = _build([1200, 2400], 1)
    out = tqa._per_row_batched_attention(batch, q, keys, values, SCALE)
    # Force the dequantize fallback for every row.
    ks, vs = batch._unwrap(keys), batch._unwrap(values)
    from omlx.turboquant_kv import _filter_state, _slice_state_range, _state_length

    total = _state_length(ks)
    pads = batch.left_padding.tolist()
    for i, pad in enumerate(pads):
        row_k = _filter_state(_slice_state_range(ks, pad, total), slice(i, i + 1))
        row_v = _filter_state(_slice_state_range(vs, pad, total), slice(i, i + 1))
        fb = tqa._row_dequant_attention(batch, q[i : i + 1], row_k, row_v, SCALE)
        assert _max_abs(fb, out[i : i + 1]) < 5e-3


def test_row_dequant_fallback_zero_pads_queries_inside_padding():
    batch, rows, q, keys, values = _build([1200, 1], 3)
    # Row 1 holds 1 + 3 = 4 tokens; with q_len 3 every query still has keys.
    out = tqa._per_row_batched_attention(batch, q, keys, values, SCALE)
    assert out.shape == q.shape
    assert not mx.any(mx.isnan(out)).item()


def test_unsupported_envelope_logs_fallback_reason(caplog):
    tqa._LOGGED_PATHS.clear()
    batch, rows, q, keys, values = _build([800, 1600], 16)
    with caplog.at_level(logging.INFO, logger=tqa.logger.name):
        out = tqa._per_row_batched_attention(batch, q, keys, values, SCALE)
    assert out.shape == q.shape
    assert any(
        "per-row dequantize fallback" in r.getMessage() and "q_len 16" in r.getMessage()
        for r in caplog.records
    )


def test_non_tq_cache_is_unchanged():
    from mlx_lm.models.cache import BatchKVCache

    cache = BatchKVCache([3, 0])
    mx.random.seed(5)
    k = mx.random.normal((2, N_KV, 40, D)).astype(mx.float16)
    v = mx.random.normal((2, N_KV, 40, D)).astype(mx.float16)
    q = mx.random.normal((2, N_Q, 1, D)).astype(mx.float16)
    mask = cache.make_mask(40)
    keys, values = cache.update_and_fetch(k, v)
    q = mx.random.normal((2, N_Q, 40, D)).astype(mx.float16)
    out = _sdpa(cache, q, keys, values, mask)
    ref = mx.fast.scaled_dot_product_attention(q, keys, values, scale=SCALE, mask=mask)
    assert _max_abs(out, ref) == 0.0


# ---------------------------------------------------------------------------
# End-to-end greedy decode on a tiny hybrid (GatedDeltaNet + attention) model
# ---------------------------------------------------------------------------


def _tiny_model():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(3)
    model = Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                full_attention_interval=2,
                vocab_size=64,
                linear_num_value_heads=4,
                linear_num_key_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                partial_rotary_factor=1.0,
            ),
        )
    )
    mx.eval(model.parameters())
    return model


def _prefilled_row_caches(model, prompt):
    caches = []
    for c in model.make_cache():
        caches.append(
            TurboQuantKVCache(bits=BITS) if type(c).__name__ == "KVCache" else c
        )
    logits = model(mx.array(prompt)[None], cache=caches)
    return caches, int(mx.argmax(logits[0, -1]).item())


def _merge(row_caches):
    from mlx_lm.models.cache import ArraysCache

    merged = []
    for layer in range(len(row_caches[0])):
        group = [rc[layer] for rc in row_caches]
        if isinstance(group[0], TurboQuantKVCache):
            merged.append(BatchTurboQuantKVCache.merge(group))
        else:
            merged.append(ArraysCache.merge(group))
    return merged


def test_greedy_decode_b3_equals_b1_mlx_lm_hybrid_array_mask_path():
    model = _tiny_model()
    mx.random.seed(21)
    lengths = [300, 900, 1500]
    prompts = [mx.random.randint(0, 64, (n,)).tolist() for n in lengths]
    steps = 24

    singles, single_logits = [], []
    for p in prompts:
        caches, tok = _prefilled_row_caches(model, p)
        toks, lg = [tok], []
        for _ in range(steps):
            logits = model(mx.array([[toks[-1]]]), cache=caches)
            lg.append(logits[0, -1])
            toks.append(int(mx.argmax(logits[0, -1]).item()))
        singles.append(toks)
        single_logits.append(mx.stack(lg))

    row_caches, first = zip(*[_prefilled_row_caches(model, p) for p in prompts])
    batch = _merge(list(row_caches))
    toks, batch_logits = [list(first)], []
    for _ in range(steps):
        logits = model(mx.array(toks[-1])[:, None], cache=batch)
        batch_logits.append(logits[:, -1])
        toks.append(mx.argmax(logits[:, -1], axis=-1).tolist())
    batched = [[t[i] for t in toks] for i in range(3)]
    blg = mx.stack(batch_logits, axis=1)
    worst = max(_max_abs(blg[i], single_logits[i]) for i in range(3))
    assert worst < 5e-2, f"max logits diff {worst:.4f}"
    assert batched == singles


def _tiny_vlm_language_model():
    """mlx-vlm's qwen3_5 LanguageModel: the "left_padded_decode" runtime path."""
    from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    mx.random.seed(3)
    text = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=128,
        intermediate_size=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=32,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [8, 4, 4],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    cfg = ModelConfig(
        text_config=text, vision_config=VisionConfig(), model_type="qwen3_5"
    )
    model = LanguageModel(text, cfg)
    mx.eval(model.parameters())
    return model


def _logits(out):
    return out.logits if hasattr(out, "logits") else out


def test_greedy_decode_b3_equals_b1_vlm_left_padded_decode_path():
    model = _tiny_vlm_language_model()
    mx.random.seed(21)
    prompts = [mx.random.randint(0, 64, (n,)).tolist() for n in (300, 900, 1500)]
    steps = 24

    def prefill(prompt):
        caches = [
            TurboQuantKVCache(bits=BITS) if type(c).__name__ == "KVCache" else c
            for c in model.make_cache()
        ]
        out = _logits(model(mx.array(prompt)[None], cache=caches))
        return caches, int(mx.argmax(out[0, -1]).item())

    singles, single_logits = [], []
    for p in prompts:
        caches, tok = prefill(p)
        toks, lg = [tok], []
        for _ in range(steps):
            out = _logits(model(mx.array([[toks[-1]]]), cache=caches))
            lg.append(out[0, -1])
            toks.append(int(mx.argmax(out[0, -1]).item()))
        singles.append(toks)
        single_logits.append(mx.stack(lg))

    rows, first = zip(*[prefill(p) for p in prompts])
    batch = []
    for layer in range(len(rows[0])):
        group = [r[layer] for r in rows]
        cls = (
            BatchTurboQuantKVCache
            if isinstance(group[0], TurboQuantKVCache)
            else type(group[0])
        )
        batch.append(cls.merge(group))
    toks, batch_logits = [list(first)], []
    for _ in range(steps):
        out = _logits(model(mx.array(toks[-1])[:, None], cache=batch))
        batch_logits.append(out[:, -1])
        toks.append(mx.argmax(out[:, -1], axis=-1).tolist())
    batched = [[t[i] for t in toks] for i in range(3)]
    blg = mx.stack(batch_logits, axis=1)
    worst = max(_max_abs(blg[i], single_logits[i]) for i in range(3))
    assert worst < 5e-2, f"max logits diff {worst:.4f}"
    assert batched == singles


# ---------------------------------------------------------------------------
# MTP verify shape (B>1, L>1, left-padded) and mlx-vlm's per-layer mask slice
# ---------------------------------------------------------------------------
# mlx-vlm's attention layer re-slices the model's mask (``mask[..., :kv]``) on
# every layer, so the TurboQuant layer receives a new array object, not the one
# ``make_mask`` returned. A padded B>1 call with L > 1 must still attend each
# row's own slice only, instead of dequantizing the whole padded context.


def _spy_per_row(monkeypatch):
    calls = []
    real = tqa._per_row_batched_attention

    def spy(*args, **kwargs):
        calls.append(args[1].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(tqa, "_per_row_batched_attention", spy)
    return calls


@pytest.mark.parametrize("new_len", [1, 2, 3, 5])
def test_resliced_own_mask_takes_per_row_path_and_matches_single_row(
    new_len, monkeypatch
):
    batch, rows, q, keys, values = _build(LENGTHS, new_len)
    own = batch._test_mask
    derived = own[..., : own.shape[-1]]  # what mlx-vlm hands to each layer
    assert derived is not own
    # The wrapped ``_prepare_projected_qkv`` carries the tag onto the slice.
    carry_own_padding_mask(own, derived)
    calls = _spy_per_row(monkeypatch)
    out = _batched(batch, q, keys, values, mask=derived)
    mx.eval(out)
    assert len(calls) == 1, "padded B>1 call left the per-row path"
    for i, rs in enumerate(rows):
        ref = _single(rs, q[i : i + 1], None)
        diff = _max_abs(out[i : i + 1], ref)
        assert diff < 2e-3, f"row {i} (len {LENGTHS[i]}) max|diff|={diff:.5f}"


@pytest.mark.parametrize("verify_len", [2, 3, 5])
def test_real_model_verify_forward_takes_per_row_path(verify_len, monkeypatch):
    model = _tiny_vlm_language_model()
    mx.random.seed(33)
    prompts = [mx.random.randint(0, 64, (n,)).tolist() for n in (300, 900, 1500)]
    draft = mx.random.randint(0, 64, (3, verify_len))

    def prefill(prompt):
        caches = [
            TurboQuantKVCache(bits=BITS) if type(c).__name__ == "KVCache" else c
            for c in model.make_cache()
        ]
        model(mx.array(prompt)[None], cache=caches)
        return caches

    rows = [prefill(p) for p in prompts]
    refs = [
        _logits(model(draft[i : i + 1], cache=prefill(prompts[i])))[0] for i in range(3)
    ]
    batch = []
    for layer in range(len(rows[0])):
        group = [r[layer] for r in rows]
        cls = (
            BatchTurboQuantKVCache
            if isinstance(group[0], TurboQuantKVCache)
            else type(group[0])
        )
        batch.append(cls.merge(group))

    calls = _spy_per_row(monkeypatch)
    out = _logits(model(draft, cache=batch, speculative_verify=True))
    mx.eval(out)

    n_tq_layers = sum(isinstance(c, BatchTurboQuantKVCache) for c in batch)
    assert n_tq_layers > 0
    assert len(calls) == n_tq_layers, (
        "verify forward left the per-row path: "
        f"{len(calls)} per-row calls for {n_tq_layers} TurboQuant layers"
    )
    for i in range(3):
        assert _max_abs(out[i], refs[i]) < 5e-2


def test_wrapped_prepare_keeps_own_mask_tag_for_the_real_attention_layer():
    attn_cls = q35_lang.Qwen3_5Attention
    assert getattr(attn_cls._prepare_projected_qkv, "_omlx_carries_own_mask", False)


def test_carried_own_mask_tag_is_scoped_to_full_width_slices_of_own_masks():
    batch, rows, q, keys, values = _build([1200, 2400], 2)
    own = batch._test_mask
    total = own.shape[-1]

    derived = own[..., :total]
    assert derived is not own
    assert not is_own_padding_mask(derived, 2, total, 2)
    carry_own_padding_mask(own, derived)
    assert is_own_padding_mask(derived, 2, total, 2)

    # A truncated slice is a different mask: never tagged.
    short = own[..., : total - 1]
    carry_own_padding_mask(own, short)
    assert not is_own_padding_mask(short, 2, total - 1, 2)

    # A foreign source never lends its (absent) tag.
    foreign = mx.array(own)
    foreign_derived = foreign[..., :total]
    carry_own_padding_mask(foreign, foreign_derived)
    assert not is_own_padding_mask(foreign_derived, 2, total, 2)


@pytest.mark.parametrize("new_len", [1, 3])
def test_foreign_mask_with_own_mask_shape_is_honored(new_len):
    # A foreign boolean mask with exactly the shape of the cache's own mask
    # must not be mistaken for it: it attends the padding columns.
    batch, rows, q, keys, values = _build([1200, 2400], new_len)
    own = batch._test_mask
    total = own.shape[-1]
    causal = (
        mx.arange(total)[None, :] <= (total - new_len + mx.arange(new_len))[:, None]
    )
    foreign = mx.broadcast_to(causal, own.shape)
    assert not mx.array_equal(foreign, own).item()
    out = _batched(batch, q, keys, values, mask=foreign)
    dk, dv = batch.dequantize(keys, values)
    ref = mx.fast.scaled_dot_product_attention(
        q, dk.astype(q.dtype), dv.astype(q.dtype), scale=SCALE, mask=foreign
    )
    assert _max_abs(out, ref) < 2e-3
