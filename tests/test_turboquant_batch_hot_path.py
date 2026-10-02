# SPDX-License-Identifier: Apache-2.0
"""Batched TurboQuant decode/verify hot path: no host syncs, no re-quantize.

Regression guards for the "B=3 decode is ~20x slower than B=1" report:

* the left-padded verify call (explicit padding mask, 1 < L <= 15) must run the
  per-row fused kernels, not dequantize the whole batch to float;
* per-layer attention must not read ``left_padding`` back to the host (each
  read is a GPU sync that breaks MLX lazy-graph pipelining) nor compare masks
  with ``mx.array_equal(...).item()``;
* the ragged rollback (``prepare(right_padding)`` + ``finalize``) must move the
  already-quantized KV, not dequantize and re-quantize the whole batch.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import dynamic_roll
from mlx_vlm.models.qwen3_5 import language as q35_lang
from mlx_vlm.turboquant import TurboQuantKVCache

from omlx.patches import turboquant_attention as tqa
from omlx.turboquant_kv import BatchTurboQuantKVCache

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


def _build(lengths, new_len, seed=0):
    rows = []
    for i, n in enumerate(lengths):
        k, v = _kv(1, n, seed + i)
        c = TurboQuantKVCache(bits=BITS)
        c.update_and_fetch(k, v)
        rows.append(c)
    batch = BatchTurboQuantKVCache.merge(rows)
    mx.random.seed(seed + 100)
    nk = mx.random.normal((len(lengths), N_KV, new_len, D)).astype(mx.float16)
    nv = mx.random.normal((len(lengths), N_KV, new_len, D)).astype(mx.float16)
    q = mx.random.normal((len(lengths), N_Q, new_len, D)).astype(mx.float16)
    # The model builds its mask BEFORE the layers append the new tokens.
    mask = batch.make_mask(new_len)
    keys, values = batch.update_and_fetch(nk, nv)
    refs = []
    for i, c in enumerate(rows):
        rk, rv = c.update_and_fetch(nk[i : i + 1], nv[i : i + 1])
        refs.append((c, rk, rv))
    return batch, refs, q, keys, values, mask


def _single(ref, q_row):
    c, rk, rv = ref
    if q_row.shape[2] == 1:
        return c.decode_attention(q_row, keys_state=rk, values_state=rv, scale=SCALE)
    return tqa._decode_multirow_attention(c, q_row, rk, rv, SCALE)


def _max_abs(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


class _HostReadCounter:
    """Stand-in for ``left_padding`` that counts host readbacks."""

    def __init__(self, array):
        self._array = array
        self.reads = 0

    def __getattr__(self, name):
        attr = getattr(self._array, name)
        if name in {"tolist", "item", "max", "min", "sum", "any", "all"}:

            def counted(*a, **k):
                self.reads += 1
                return attr(*a, **k)

            return counted
        return attr

    def __getitem__(self, idx):
        return self._array[idx]

    def __int__(self):
        self.reads += 1
        return int(self._array)

    def __add__(self, other):
        return self._array + other

    def __radd__(self, other):
        return other + self._array


@pytest.mark.parametrize("new_len", [1, 3])
def test_verify_with_padding_mask_does_not_dequantize_the_batch(new_len, monkeypatch):
    batch, refs, q, keys, values, mask = _build(LENGTHS, new_len)
    calls = {"n": 0}
    original = BatchTurboQuantKVCache.dequantize

    def spy(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(BatchTurboQuantKVCache, "dequantize", spy)
    out = q35_lang._qwen3_5_left_padded_attention(
        q, keys, values, cache=batch, scale=SCALE, mask=mask
    )
    mx.eval(out)
    assert calls["n"] == 0, "padded verify dequantized the whole batch"
    for i, ref in enumerate(refs):
        diff = _max_abs(out[i : i + 1], _single(ref, q[i : i + 1]))
        assert diff < 2e-3, f"row {i} max|diff|={diff:.5f}"


@pytest.mark.parametrize("new_len,use_mask", [(1, False), (3, True), (1, True)])
def test_attention_hot_path_reads_nothing_back_to_the_host(
    new_len, use_mask, monkeypatch
):
    batch, refs, q, keys, values, mask = _build(LENGTHS, new_len)
    call = lambda: q35_lang._qwen3_5_left_padded_attention(  # noqa: E731
        q, keys, values, cache=batch, scale=SCALE, mask=mask if use_mask else None
    )
    mx.eval(call())  # first call may seed per-cache host metadata
    counter = _HostReadCounter(batch.left_padding)
    batch._set_left_padding(counter, batch.host_left_padding())
    equal_calls = {"n": 0}
    real_equal = mx.array_equal

    def spy_equal(*a, **k):
        equal_calls["n"] += 1
        return real_equal(*a, **k)

    monkeypatch.setattr(mx, "array_equal", spy_equal)
    mx.eval(call())
    assert counter.reads == 0, f"{counter.reads} left_padding host reads per layer"
    assert equal_calls["n"] == 0, "mask array_equal sync on the hot path"


def test_patched_sdpa_padding_mask_has_no_array_equal_sync(monkeypatch):
    from mlx_lm.models import base as mlx_base

    batch, refs, q, keys, values, mask = _build(LENGTHS, 1)
    equal_calls = {"n": 0}
    real_equal = mx.array_equal

    def spy_equal(*a, **k):
        equal_calls["n"] += 1
        return real_equal(*a, **k)

    monkeypatch.setattr(mx, "array_equal", spy_equal)
    out = mlx_base.scaled_dot_product_attention(
        q, keys, values, batch, SCALE, mask
    )
    mx.eval(out)
    assert equal_calls["n"] == 0
    for i, ref in enumerate(refs):
        assert _max_abs(out[i : i + 1], _single(ref, q[i : i + 1])) < 2e-3


def _dequant(cache):
    k, v = cache.dequantize()
    mx.eval(k, v)
    return k, v


def test_ragged_finalize_moves_packed_kv_without_requantizing(monkeypatch):
    batch, _, _, _, _, _ = _build([3000, 3000, 3000], 3)
    k0, v0 = _dequant(batch)
    right = [0, 2, 1]
    expected_k = dynamic_roll(k0, mx.array(right)[:, None], axis=2)
    expected_v = dynamic_roll(v0, mx.array(right)[:, None], axis=2)
    pads_before = [int(p) for p in batch.left_padding.tolist()]

    quantize_calls = {"n": 0}
    for codec in (batch.key_codec, batch.value_codec):
        original = codec.quantize

        def counted(x, _orig=original):
            quantize_calls["n"] += 1
            return _orig(x)

        monkeypatch.setattr(codec, "quantize", counted)

    batch.prepare(right_padding=right)
    batch.finalize()
    k1, v1 = _dequant(batch)
    assert quantize_calls["n"] == 0, "finalize re-quantized the batch"
    pads_after = [int(p) for p in batch.left_padding.tolist()]
    assert pads_after == [p + r for p, r in zip(pads_before, right)]
    # Upstream's packed roll (#3127) rotates the step-allocated buffer, so the
    # columns that become left padding may differ from a roll of the written
    # region; they are masked. Every unmasked column must match exactly.
    for b, pad in enumerate(pads_after):
        assert _max_abs(k1[b : b + 1, :, pad:], expected_k[b : b + 1, :, pad:]) == 0.0
        assert _max_abs(v1[b : b + 1, :, pad:], expected_v[b : b + 1, :, pad:]) == 0.0
    # The written length is unchanged.
    assert batch._phys_end == k0.shape[2]


def test_ragged_finalize_keeps_attention_equal_to_single_rows():
    # Roll a verify block's ragged tail away, then check each row still attends
    # exactly its own (now shorter) history.
    batch, refs, q, keys, values, _ = _build([1500, 2500, 3500], 4)
    accepted = [4, 2, 3]  # tokens kept out of the 4 appended
    right = [max(accepted) - a for a in accepted]
    trim = 4 - max(accepted)
    batch.trim(trim)
    batch.prepare(right_padding=right)
    batch.finalize()
    for i, (c, _, _) in enumerate(refs):
        c.trim(4 - accepted[i])
    mx.random.seed(77)
    q1 = mx.random.normal((3, N_Q, 1, D)).astype(mx.float16)
    nk, nv = _kv(3, 1, 55)
    mask = batch.make_mask(1)
    k2, v2 = batch.update_and_fetch(nk, nv)
    out = q35_lang._qwen3_5_left_padded_attention(
        q1, k2, v2, cache=batch, scale=SCALE, mask=mask
    )
    mx.eval(out)
    for i, (c, _, _) in enumerate(refs):
        rk, rv = c.update_and_fetch(nk[i : i + 1], nv[i : i + 1])
        ref = c.decode_attention(
            q1[i : i + 1], keys_state=rk, values_state=rv, scale=SCALE
        )
        assert _max_abs(out[i : i + 1], ref) < 2e-3, f"row {i}"


def test_batch_policy_parks_after_two_hopeless_cycles():
    from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

    policy = BatchPolicy(("a", "b", "c"), 4)
    for _ in range(6):
        policy.observe_standard(88.0)
    # A depth-4 cycle can emit at most 5 tokens per row; at 2.5 s against an
    # 88 ms ordinary step it can never win, however well drafts are accepted.
    policy.observe_mtp(4, [2, 1, 1], 2500.0, stable=True)
    policy.observe_mtp(4, [2, 1, 1], 2500.0, stable=True)
    assert policy.should_park()


def test_batch_policy_keeps_profitable_mtp():
    from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

    policy = BatchPolicy(("a", "b", "c"), 4)
    for _ in range(6):
        policy.observe_standard(88.0)
    for _ in range(6):
        policy.observe_mtp(4, [3, 2, 3], 150.0, stable=True)
    assert not policy.should_park()


# ---------------------------------------------------------------------------
# Ragged single-launch kernel (decode and short verify blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_len", [1, 2, 3, 4, 5, 6, 8])
def test_ragged_kernel_matches_single_rows_and_is_one_launch(new_len, monkeypatch):
    batch, refs, q, keys, values, _ = _build([1800, 5200, 9100], new_len)
    tqa._LOGGED_PATHS.clear()
    launches = {"n": 0}
    original = TurboQuantKVCache.decode_attention

    def spy(self, *a, **k):
        launches["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(TurboQuantKVCache, "decode_attention", spy)
    out = tqa._per_row_batched_attention(
        batch, q, keys, values, SCALE
    )
    mx.eval(out)
    assert "ragged" in tqa._LOGGED_PATHS, tqa._LOGGED_PATHS
    assert launches["n"] == 0, "per-row decode_attention ran next to the ragged kernel"
    monkeypatch.undo()
    for i, ref in enumerate(refs):
        diff = _max_abs(out[i : i + 1], _single(ref, q[i : i + 1]))
        assert diff < 2e-3, f"row {i} q_len {new_len} max|diff|={diff:.5f}"


def test_ragged_kernel_row_shorter_than_query_block_is_zero_not_nan():
    batch, refs, q, keys, values, _ = _build([1500, 2500], 4)
    # Make row 0 own only 2 columns (< q_len 4): its first two queries sit in
    # the padding and attend nothing.
    ks, vs = batch._unwrap(keys), batch._unwrap(values)
    from omlx.turboquant_kv import _state_length

    total = _state_length(ks)
    batch.left_padding = mx.array([total - 2, int(batch.left_padding[1].item())])
    out = tqa._ragged_batched_attention(batch, q, ks, vs, SCALE)
    mx.eval(out)
    assert not mx.any(mx.isnan(out)).item()
    assert mx.max(mx.abs(out[0:1, :, :2, :].astype(mx.float32))).item() == 0.0
    assert mx.max(mx.abs(out[0:1, :, 2:, :].astype(mx.float32))).item() > 0.0


def test_ragged_kernel_all_padding_row_is_zero():
    batch, refs, q, keys, values, _ = _build([1500, 2500], 1)
    ks, vs = batch._unwrap(keys), batch._unwrap(values)
    from omlx.turboquant_kv import _state_length

    total = _state_length(ks)
    batch.left_padding = mx.array([total, 0])
    out = tqa._ragged_batched_attention(batch, q, ks, vs, SCALE)
    mx.eval(out)
    assert not mx.any(mx.isnan(out)).item()
    assert mx.max(mx.abs(out[0].astype(mx.float32))).item() == 0.0
    assert mx.max(mx.abs(out[1].astype(mx.float32))).item() > 0.0


def test_ragged_kernel_declines_wider_blocks_and_non_mse(monkeypatch):
    batch, refs, q, keys, values, _ = _build([1500, 2500], 9)
    ks, vs = batch._unwrap(keys), batch._unwrap(values)
    assert tqa._ragged_batched_attention(batch, q, ks, vs, SCALE) is None


@pytest.mark.parametrize("new_len", [1, 3])
def test_ragged_kernel_unrolled_and_single_token_loops_agree(new_len, monkeypatch):
    batch, refs, q, keys, values, _ = _build([1800, 5200, 9100], new_len)
    ks, vs = batch._unwrap(keys), batch._unwrap(values)
    outs = []
    for unroll in (1, 2, 4):
        monkeypatch.setattr(tqa, "_RAGGED_UNROLL", unroll)
        out = tqa._ragged_batched_attention(batch, q, ks, vs, SCALE)
        mx.eval(out)
        outs.append(out)
    assert _max_abs(outs[0], outs[1]) < 1e-3
    assert _max_abs(outs[0], outs[2]) < 1e-3


# ---------------------------------------------------------------------------
# Dense (non-TurboQuant) layer next to the TurboQuant ones
# ---------------------------------------------------------------------------


def _dense_batch(pads, total, new_len, seed=0):
    from mlx_lm.models.cache import BatchKVCache

    mx.random.seed(seed)
    cache = BatchKVCache(list(pads))
    k = mx.random.normal((len(pads), N_KV, total - new_len, D)).astype(mx.float16)
    v = mx.random.normal((len(pads), N_KV, total - new_len, D)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    mask = cache.make_mask(new_len)
    nk = mx.random.normal((len(pads), N_KV, new_len, D)).astype(mx.float16)
    nv = mx.random.normal((len(pads), N_KV, new_len, D)).astype(mx.float16)
    q = mx.random.normal((len(pads), N_Q, new_len, D)).astype(mx.float16)
    keys, values = cache.update_and_fetch(nk, nv)
    return cache, q, keys, values, mask


@pytest.mark.parametrize("new_len", [1, 3])
def test_dense_padded_rows_attend_their_own_slice_without_copying_the_cache(
    new_len, monkeypatch
):
    pads = [1500, 400, 0]  # different SDPA plans per row: no shared ragged kernel
    cache, q, keys, values, mask = _dense_batch(pads, 2600, new_len)
    takes = {"n": 0}
    real_take = mx.take

    def spy_take(*a, **k):
        takes["n"] += 1
        return real_take(*a, **k)

    monkeypatch.setattr(mx, "take", spy_take)
    out = q35_lang._qwen3_5_left_padded_attention(
        q, keys, values, cache=cache, scale=SCALE, mask=None
    )
    mx.eval(out)
    assert takes["n"] == 0, "dense KV rows were gathered (copied) per pad group"
    ref = mx.fast.scaled_dot_product_attention(q, keys, values, scale=SCALE, mask=mask)
    assert _max_abs(out, ref) < 2e-3


def test_dense_unpadded_batch_keeps_the_upstream_path():
    cache, q, keys, values, mask = _dense_batch([0, 0], 1200, 1)
    assert tqa._dense_padded_attention(q, keys, values, cache=cache, scale=SCALE) is None
