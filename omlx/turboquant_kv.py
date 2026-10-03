# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache — thin wrapper around mlx_vlm.turboquant.

Core implementation (codecs, Metal kernels, TurboQuantKVCache) lives in
mlx-vlm.  This module re-exports the public API and adds
BatchTurboQuantKVCache (inherits TurboQuantKVCache) for omlx's
continuous-batching scheduler.
"""

from __future__ import annotations

import logging
import math
import weakref
from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.cache import (
    KVCache,
    _BaseCache,
    create_attention_mask,
    create_causal_mask,
    dynamic_roll,
)
from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    TurboQuantMSEState,
    TurboQuantPolarProdState,
    TurboQuantPolarState,
    TurboQuantProdState,
    TurboQuantSplitState,
    _allocate_state_like,
    _build_codec,
    _concat_state,
    _QuantizedStateProxy,
    _reserve_state_capacity,
    _slice_state,
    _slice_state_range,
    _state_length,
    _state_nbytes,
    _validate_bits,
    _write_state,
    turboquant_enabled,
)

logger = logging.getLogger(__name__)


@classmethod
def _from_cache(cls, cache, bits, seed=0):
    result = cls(bits=bits, seed=seed)
    if callable(getattr(cache, "keys_and_values", None)):
        keys, values = (
            cache.keys_and_values() if cache.keys is not None else (None, None)
        )
    else:
        keys, values = cache.state
    if keys is not None:
        result.update_and_fetch(keys, values)
    return result


TurboQuantKVCache.from_cache = _from_cache


__all__ = [
    "TurboQuantKVCache",
    "BatchTurboQuantKVCache",
    "turboquant_enabled",
]


# ---------------------------------------------------------------------------
# Codec rebuild for SSD cache reconstruction
# ---------------------------------------------------------------------------


def _infer_head_dim(state, bits: int) -> int:
    """Infer head_dim from a TQ quantized state's packed tensor width.

    MSEState.indices has shape (..., packed_width) where
    packed_width = ceil(head_dim * bits / 32).
    """
    if isinstance(state, TurboQuantMSEState):
        packed_width = state.indices.shape[-1]
    elif isinstance(state, TurboQuantProdState):
        packed_width = state.mse_indices.shape[-1]
        bits = max(bits - 1, 1)
    else:
        raise TypeError(
            f"Cannot infer head_dim from state type: {type(state).__name__}"
        )
    return packed_width * 32 // bits


def _rebuild_codecs(tq_cache: TurboQuantKVCache, key_state, value_state) -> None:
    """Rebuild TQ codecs deterministically from (head_dim, bits, seed).

    TQ codecs (rotation matrices, codebooks) are fully determined by
    (head_dim, bits, seed) for integer bit-widths — no data dependency.
    This allows rebuilding codecs without the original fp16 tensors,
    which is needed when reconstructing from SSD cache.
    """
    bits = tq_cache.bits
    seed = tq_cache.seed
    fractional = not math.isclose(bits, round(bits), abs_tol=1e-6)
    key_bits = int(math.floor(bits) if fractional else bits)
    val_bits = int(math.ceil(bits) if fractional else bits)

    head_dim = _infer_head_dim(key_state, key_bits)

    dummy = mx.zeros((1, 1, 1, head_dim))
    tq_cache.key_codec = _build_codec(dummy, key_bits, mode="mse", seed=seed)
    tq_cache.value_codec = _build_codec(dummy, val_bits, mode="mse", seed=seed + 1)


def _concat_state_token_axis(states):
    """Concatenate TurboQuant states along token axis with a low-churn fast path."""
    if not states:
        return None
    if len(states) == 1:
        state = states[0]
        return state._state if isinstance(state, _QuantizedStateProxy) else state

    unwrapped = [
        state._state if isinstance(state, _QuantizedStateProxy) else state
        for state in states
    ]
    first = unwrapped[0]
    if isinstance(first, TurboQuantMSEState) and all(
        isinstance(state, TurboQuantMSEState) for state in unwrapped
    ):
        return TurboQuantMSEState(
            mx.concatenate([state.norms for state in unwrapped], axis=2),
            mx.concatenate([state.indices for state in unwrapped], axis=2),
        )

    result = first
    for state in unwrapped[1:]:
        result = _concat_state(result, state)
    return result


# ---------------------------------------------------------------------------
# Batch-level state helpers (axis-0 operations)
# ---------------------------------------------------------------------------


def _filter_state(state, indices):
    """Index-select along batch dimension (axis 0)."""
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(state.norms[indices], state.indices[indices])
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[indices],
            state.mse_indices[indices],
            state.residual_norms[indices],
            state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[indices],
            tuple(level[indices] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[indices],
            _filter_state(state.polar_state, indices),
            state.residual_norms[indices],
            state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _filter_state(state.low, indices),
            _filter_state(state.high, indices),
        )
    raise TypeError(f"Unsupported state type: {type(state)!r}")


def _map_state_arrays(state, fn):
    """Apply ``fn`` to every token-axis array of a state (shape [B, H, T(, X)])."""
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(fn(state.norms), fn(state.indices))
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            fn(state.norms),
            fn(state.mse_indices),
            fn(state.residual_norms),
            fn(state.qjl_signs),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            fn(state.radii), tuple(fn(level) for level in state.level_indices)
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            fn(state.norms),
            _map_state_arrays(state.polar_state, fn),
            fn(state.residual_norms),
            fn(state.qjl_signs),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _map_state_arrays(state.low, fn), _map_state_arrays(state.high, fn)
        )
    raise TypeError(f"Unsupported state type: {type(state)!r}")


def _concat_state_batch(states):
    """Concatenate a list of states along batch dimension (axis 0)."""
    if not states:
        return None
    first = states[0]
    if isinstance(first, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.concatenate([s.norms for s in states], axis=0),
            mx.concatenate([s.indices for s in states], axis=0),
        )
    if isinstance(first, TurboQuantProdState):
        return TurboQuantProdState(
            mx.concatenate([s.norms for s in states], axis=0),
            mx.concatenate([s.mse_indices for s in states], axis=0),
            mx.concatenate([s.residual_norms for s in states], axis=0),
            mx.concatenate([s.qjl_signs for s in states], axis=0),
        )
    if isinstance(first, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.concatenate([s.radii for s in states], axis=0),
            tuple(
                mx.concatenate(
                    [states[j].level_indices[i] for j in range(len(states))], axis=0
                )
                for i in range(len(first.level_indices))
            ),
        )
    if isinstance(first, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.concatenate([s.norms for s in states], axis=0),
            _concat_state_batch([s.polar_state for s in states]),
            mx.concatenate([s.residual_norms for s in states], axis=0),
            mx.concatenate([s.qjl_signs for s in states], axis=0),
        )
    if isinstance(first, TurboQuantSplitState):
        return TurboQuantSplitState(
            _concat_state_batch([s.low for s in states]),
            _concat_state_batch([s.high for s in states]),
        )
    raise TypeError(f"Unsupported state type: {type(first)!r}")


def _pad_state_left(state, pad_length: int):
    """Prepend zeros along the token dimension (axis 2) of a state."""
    if state is None or pad_length <= 0:
        return state
    pad = _allocate_state_like(state, pad_length)
    return _concat_state(pad, state)


def _roll_state(state, shifts, axis: int = 2):
    """Roll a quantized state along its token axis by per-row ``shifts``.

    Mirrors dynamic_roll's per-batch-row dynamic shift (same axis/shifts
    convention as _slice_state/_concat_state above), applied directly to the
    packed representation instead of dequantize -> roll -> requantize: roll
    is pure reindexing along the token axis, so applying it independently to
    every field that shares that axis (norms, packed indices, radii, ...)
    preserves their per-token pairing exactly, with no fp16 materialization
    and no re-rounding through the codec on the way back in.
    """
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            dynamic_roll(state.norms, shifts, axis=axis),
            dynamic_roll(state.indices, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            dynamic_roll(state.norms, shifts, axis=axis),
            dynamic_roll(state.mse_indices, shifts, axis=axis),
            dynamic_roll(state.residual_norms, shifts, axis=axis),
            dynamic_roll(state.qjl_signs, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            dynamic_roll(state.radii, shifts, axis=axis),
            tuple(
                dynamic_roll(level, shifts, axis=axis)
                for level in state.level_indices
            ),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            dynamic_roll(state.norms, shifts, axis=axis),
            _roll_state(state.polar_state, shifts, axis=axis),
            dynamic_roll(state.residual_norms, shifts, axis=axis),
            dynamic_roll(state.qjl_signs, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _roll_state(state.low, shifts, axis=axis),
            _roll_state(state.high, shifts, axis=axis),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _empty_state_batch_like(state, batch_size: int):
    """Allocate an empty token state with the requested batch size."""
    if state is None:
        return None
    row = _filter_state(_allocate_state_like(state, 0), slice(0, 1))
    if batch_size == 1:
        return row
    return _concat_state_batch([row] * batch_size)


# ---------------------------------------------------------------------------
# Own padding-mask tag
# ---------------------------------------------------------------------------
# The decode/verify attention replacement may only substitute per-row slices
# for a mask when that mask IS the left-padding + causal mask the cache built.
# Comparing arrays (``array_equal(...).item()``) is a GPU sync on every layer,
# so ``make_mask`` tags the mask OBJECT it returns (weak identity registry:
# mx.array takes no attributes) together with its structure (query rows,
# physical column count BEFORE the append, batch size), and the attention side
# checks both. The model builds the mask once per forward from the first
# cache, before any layer appends, and hands the same object to every layer.
# A foreign array with the same shape is a different object, so it is never
# mistaken for the cache's own mask.

_OWN_PADDING_MASKS: dict = {}


def mark_own_padding_mask(mask, n: int, phys: int, batch: int, window_size=None) -> None:
    """Tag ``mask`` as the padding mask this cache built for ``n`` rows."""
    if window_size is not None or not isinstance(mask, mx.array):
        return
    key = id(mask)

    def _forget(_ref, key=key):
        _OWN_PADDING_MASKS.pop(key, None)

    _OWN_PADDING_MASKS[key] = (weakref.ref(mask, _forget), (n, phys, batch))


def is_own_padding_mask(mask, n: int, total: int, batch: int) -> bool:
    """True iff ``mask`` is a padding mask built for ``n`` rows ending at ``total``."""
    entry = _OWN_PADDING_MASKS.get(id(mask))
    if entry is None:
        return False
    ref, signature = entry
    return ref() is mask and signature == (n, total - n, batch)


# ---------------------------------------------------------------------------
# BatchTurboQuantKVCache — inherits TurboQuantKVCache
# ---------------------------------------------------------------------------


class BatchTurboQuantKVCache(TurboQuantKVCache):
    """TurboQuantKVCache with batch operations for continuous batching.

    Inherits update_and_fetch, decode_attention, _ensure_codecs, state,
    and all decode logic from TurboQuantKVCache with ZERO overhead.
    Only adds batch-specific methods (merge/extract/extend/filter) and
    overrides make_mask for per-request left_padding support.
    """

    def __init__(self, left_padding: List[int], bits: float = 4.0, seed: int = 0):
        super().__init__(bits=bits, seed=seed)
        self.group_size = 0
        self._pads_host = None
        self.left_padding = mx.array(left_padding)
        # Known from the Python list: no device readback needed for the
        # per-layer attention routing (see host_left_padding()).
        self._pads_host = tuple(int(p) for p in left_padding)
        self._batch_size = len(left_padding)
        # B=1: offset is int (parent-compatible, zero overhead decode)
        # B>1: offset is mx.array (per-request, needs override)
        if self._batch_size > 1:
            self.offset = mx.array([-l for l in left_padding])
        else:
            self.offset = -left_padding[0]
        self._right_padding = None
        self._right_padding_host = None
        # Written physical column count (B>1 only; B=1 uses the parent's int
        # offset). Rows are end-aligned, but this must NOT be derived from
        # offset.max(): once filter() removes the last zero-left-padding row,
        # every logical offset is short of the written end, and an
        # offset-derived position writes INSIDE the survivors' live KV. It
        # also must not be derived from _state_length(self.keys): that is the
        # step-allocated capacity, not the written end.
        self._phys_end = 0

    # ---- left_padding with a host mirror ----------------------------------
    # Attention routing needs the per-row padding as Python ints on EVERY
    # attention layer of EVERY decode step. Reading the device array back
    # (``tolist``/``max().item()``) is a GPU sync that breaks MLX lazy-graph
    # pipelining, so the host copy is derived once per change instead and
    # carried along by the mutators below, which know the values.

    @property
    def left_padding(self):
        return self._left_padding

    @left_padding.setter
    def left_padding(self, value):
        self._left_padding = value
        self._pads_host = None

    def _set_left_padding(self, array, host):
        self._left_padding = array
        self._pads_host = None if host is None else tuple(int(p) for p in host)

    def host_left_padding(self):
        """Per-row left padding as a tuple of ints (one readback per change)."""
        if self._pads_host is None:
            self._pads_host = tuple(int(p) for p in self._left_padding.tolist())
        return self._pads_host

    def max_left_padding(self) -> int:
        pads = self.host_left_padding()
        return max(pads) if pads else 0

    @property
    def _idx(self):
        # mlx-vlm uses this cursor for rollback and padded attention masks.
        return self.offset if isinstance(self.offset, int) else self._phys_end

    def trim(self, n):
        position = self._idx
        n = min(position, n)
        self.offset -= n
        self._phys_end = position - n
        self._cached_state = None
        self._cached_state_offset = -1
        return n

    # ---- update_and_fetch override for B>1 only ----------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if isinstance(self.offset, int):
            # B=1: parent's method directly (zero overhead)
            return super().update_and_fetch(keys, values)
        # B>1: track per-request offset separately from state offset
        T_new = keys.shape[2]
        # Append at the written physical end (see __init__._phys_end note).
        int_offset = self._phys_end
        self.offset += T_new
        saved_offset = self.offset
        self.offset = int_offset
        result = super().update_and_fetch(keys, values)
        self.offset = saved_offset
        self._phys_end = int_offset + T_new
        return result

    # ---- state override for B>1 (offset is mx.array) -----------------------

    @property
    def state(self):
        if isinstance(self.offset, int):
            return super().state
        # B>1: slice to the written end — _state_length(self.keys) is the
        # step-allocated capacity and would expose unwritten columns to
        # attention after the buffer grows.
        if self.keys is None:
            return None, None
        length = self._phys_end
        return _slice_state(self.keys, length), _slice_state(self.values, length)

    @state.setter
    def state(self, value):
        TurboQuantKVCache.state.fset(self, value)
        # The parent fset resets to int-offset (B=1) bookkeeping, where the
        # parent's offset is the write cursor (and trim() may move it back).
        # Drop any stale batch-mode value so _ensure_array_offset re-derives
        # _phys_end from that cursor at the B>1 switch.
        self._phys_end = 0

    # ---- make_mask override (batch-aware) ----------------------------------

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ):
        offset = self.offset
        if isinstance(offset, int):
            return create_attention_mask(N, offset, return_array, window_size)
        if (
            isinstance(offset, mx.array)
            and offset.size == 1
            and self.max_left_padding() == 0
        ):
            return create_attention_mask(N, offset.item(), return_array, window_size)
        # B>1 (or a left-padded survivor after filter()): delegate to mlx-lm's
        # create_causal_mask with the physical column count + per-request
        # left_padding, exactly like BatchKVCache. The old hand-rolled term
        # compared each request's sequence length (offset) against the column
        # index, which masked out valid left-padded tokens — so left-padded
        # requests attended to ~nothing and decoded garbage. The column count
        # is the WRITTEN end, not offset.max(): after the zero-left-padding
        # row departs, offset.max() undercounts and blinds the survivors to
        # their own tail context.
        phys = self._phys_end
        mask = create_causal_mask(
            N, offset=phys, window_size=window_size, left_padding=self.left_padding
        )
        mark_own_padding_mask(
            mask, N, phys, len(self.host_left_padding()), window_size
        )
        return mask

    # prefill_attention and dequantize inherited from TurboQuantKVCache

    # ---- batch operations --------------------------------------------------

    def _ensure_array_offset(self):
        if isinstance(self.offset, int):
            # B=1 tracks written columns in the parent's int offset (plus any
            # left padding); sync the physical end before switching to
            # per-request array offsets, where the parent no longer maintains
            # it.
            lp0 = int(self.left_padding[0].item()) if self.left_padding is not None else 0
            self._phys_end = max(self._phys_end, self.offset + lp0)
            self.offset = mx.array([self.offset])

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            host_before = self._pads_host
            host_add = [int(p) for p in left_padding]
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            if host_before is not None and len(host_before) == len(host_add):
                self._pads_host = tuple(a + b for a, b in zip(host_before, host_add))
            self.offset -= (
                left_padding
                if isinstance(self.offset, mx.array)
                else host_add[0]
            )
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
            self._right_padding_host = [int(p) for p in right_padding]

    def finalize(self):
        if self._right_padding is None:
            return
        padding = self._right_padding
        host_padding = getattr(self, "_right_padding_host", None)
        if host_padding is None:
            host_padding = [int(p) for p in padding.tolist()]
        if self.keys is not None:
            # Roll the already-quantized state directly (_roll_state) rather
            # than dequantize -> dynamic_roll -> requantize: roll only
            # reindexes along the token axis, so it's exact on the packed
            # representation and skips both the fp16 materialization and the
            # requantization codec's re-rounding of an already-quantized
            # value.
            shifts = padding[:, None]
            self.keys = _roll_state(self.keys, shifts, axis=2)
            self.values = _roll_state(self.values, shifts, axis=2)
            mx.eval(self.keys, self.values)
        self.offset -= (
            padding if isinstance(self.offset, mx.array) else host_padding[0]
        )
        host_before = self._pads_host
        self.left_padding += padding
        if host_before is not None and len(host_before) == len(host_padding):
            self._pads_host = tuple(a + b for a, b in zip(host_before, host_padding))
        self._right_padding = None
        self._right_padding_host = None
        self._cached_state = None
        self._cached_state_offset = -1

    def filter(self, batch_indices):
        self._ensure_array_offset()
        if self.keys is not None:
            self.keys = _filter_state(self.keys, batch_indices)
            self.values = _filter_state(self.values, batch_indices)
        self.offset = self.offset[batch_indices]
        host_before = self._pads_host
        self.left_padding = self.left_padding[batch_indices]
        if host_before is not None and isinstance(batch_indices, (list, tuple, range)):
            self._pads_host = tuple(host_before[int(i)] for i in batch_indices)
        # Shift left to drop shared padding, mirroring BatchKVCache.filter().
        # The turboquant_skip_last layer is a stock BatchKVCache that compacts
        # here, and the model feeds every attention layer ONE mask built from
        # the first (TQ) layer's _phys_end. Skipping the same compaction
        # leaves that mask min_left_pad columns wider than the dense layer's
        # keys and crashes the next decode step (#2237).
        host_pads = self.host_left_padding()
        min_left_pad = min(host_pads) if host_pads else 0
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = _slice_state_range(
                    self.keys, min_left_pad, self._phys_end
                )
                self.values = _slice_state_range(
                    self.values, min_left_pad, self._phys_end
                )
            self._phys_end -= min_left_pad
            self._set_left_padding(
                self.left_padding - min_left_pad,
                [p - min_left_pad for p in host_pads],
            )
        self._cached_state = None
        self._cached_state_offset = -1

    def extend(self, other: "BatchTurboQuantKVCache"):
        if not isinstance(other, BatchTurboQuantKVCache):
            raise TypeError(
                "BatchTurboQuantKVCache.extend expected BatchTurboQuantKVCache, "
                f"got {type(other).__name__}"
            )
        self._ensure_array_offset()
        other._ensure_array_offset()
        max_off = max(self.offset.max().item(), other.offset.max().item())
        # Align on the WRITTEN ends: _state_length is step-allocated capacity,
        # and padding a joining row by capacity difference would bury its
        # content behind unwritten columns. _pad_and_trim also slices each
        # side down to its written end, normalizing any over-allocation.
        s_idx = self._phys_end if self.keys is not None else 0
        o_idx = other._phys_end if other.keys is not None else 0
        max_idx = max(s_idx, o_idx)
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values

        def _pad_and_trim(c, idx):
            batch_size = int(c.offset.shape[0])
            if c.keys is None:
                if max_idx > 0 and ref_keys is not None:
                    ks = _empty_state_batch_like(ref_keys, batch_size)
                    vs = _empty_state_batch_like(ref_values, batch_size)
                else:
                    ks = None
                    vs = None
            else:
                ks = _slice_state(c.keys, idx)
                vs = _slice_state(c.values, idx)
            left = max_idx - idx
            if left > 0 and ks is not None:
                ks = _pad_state_left(ks, left)
                vs = _pad_state_left(vs, left)
            return ks, vs, c.offset, c.left_padding + left

        s_ks, s_vs, s_off, s_lp = _pad_and_trim(self, s_idx)
        o_ks, o_vs, o_off, o_lp = _pad_and_trim(other, o_idx)

        if s_ks is not None and o_ks is not None:
            self.keys = _concat_state_batch([s_ks, o_ks])
            self.values = _concat_state_batch([s_vs, o_vs])
        elif o_ks is not None:
            self.keys = o_ks
            self.values = o_vs

        host_s, host_o = self._pads_host, other._pads_host
        self.offset = mx.concatenate([s_off, o_off])
        self.left_padding = mx.concatenate([s_lp, o_lp])
        if host_s is not None and host_o is not None:
            self._pads_host = tuple(p + (max_idx - s_idx) for p in host_s) + tuple(
                p + (max_idx - o_idx) for p in host_o
            )
        self._phys_end = max_idx
        self._cached_state = None
        self._cached_state_offset = -1

        if self.key_codec is None:
            self.key_codec = other.key_codec
            self.value_codec = other.value_codec

    def extract(self, idx: int) -> TurboQuantKVCache:
        padding = self.host_left_padding()[idx]
        total = (
            self.offset[idx].item()
            if isinstance(self.offset, mx.array)
            else self.offset
        )
        end = padding + total

        tq = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        if self.keys is not None:
            ks = _slice_state_range(self.keys, padding, end)
            vs = _slice_state_range(self.values, padding, end)
            tq.keys = _filter_state(ks, slice(idx, idx + 1))
            tq.values = _filter_state(vs, slice(idx, idx + 1))
            tq.offset = total
        tq.key_codec = self.key_codec
        tq.value_codec = self.value_codec
        return tq

    @classmethod
    def merge(cls, caches: List[TurboQuantKVCache]) -> "BatchTurboQuantKVCache":
        for cache in caches:
            if not isinstance(cache, TurboQuantKVCache):
                raise TypeError(
                    "BatchTurboQuantKVCache.merge expected TurboQuantKVCache "
                    f"entries, got {type(cache).__name__}"
                )
        bits = caches[0].bits
        seed = caches[0].seed
        configs = {(c.bits, c.seed) for c in caches}
        if len(configs) > 1:
            # Packed state width is ceil(head_dim * bits / 32) and codecs are
            # rebuilt from (head_dim, bits, seed), so members quantized under
            # different configs cannot share a batch. Without this guard the
            # mismatch surfaces as a raw mx.concatenate shape error (or, for
            # equal widths, silent garbage decode) deep in
            # _concat_state_batch (#2045).
            raise ValueError(
                "Cannot batch TurboQuant caches with mixed quantization "
                f"configs (bits, seed): {sorted(configs)}. A request restored "
                "from cache blocks written at another turboquant_kv_bits "
                "depth cannot share a batch with fresh requests; clear the "
                "paged SSD cache for this model if this persists."
            )
        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        batch = cls(padding, bits=bits, seed=seed)

        for c in caches:
            if c.key_codec is not None:
                batch.key_codec = c.key_codec
                batch.value_codec = c.value_codec
                break

        key_states = []
        value_states = []
        reference_key_state = None
        reference_value_state = None
        for c in caches:
            ks, vs = c.state
            if ks is not None:
                reference_key_state = (
                    ks._state if isinstance(ks, _QuantizedStateProxy) else ks
                )
                reference_value_state = (
                    vs._state if isinstance(vs, _QuantizedStateProxy) else vs
                )
                break

        for p, c in zip(padding, caches):
            ks, vs = c.state
            if ks is None:
                if max_length > 0 and reference_key_state is not None:
                    key_states.append(
                        _allocate_state_like(reference_key_state, max_length)
                    )
                    value_states.append(
                        _allocate_state_like(reference_value_state, max_length)
                    )
                continue
            ks = ks._state if isinstance(ks, _QuantizedStateProxy) else ks
            vs = vs._state if isinstance(vs, _QuantizedStateProxy) else vs
            if p > 0:
                ks = _pad_state_left(ks, p)
                vs = _pad_state_left(vs, p)
            key_states.append(ks)
            value_states.append(vs)

        if key_states:
            batch.keys = _concat_state_batch(key_states)
            batch.values = _concat_state_batch(value_states)
            mx.eval(batch.keys, batch.values)

        batch.offset += max_length
        batch._phys_end = max_length
        return batch
