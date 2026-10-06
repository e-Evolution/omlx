# SPDX-License-Identifier: Apache-2.0
"""The never-fits gate on real VLM and text model structures.

On ``VLMBatchedEngine`` the scheduler's model is the ``VLMModelAdapter``. It
keeps the mlx-vlm model in ``_vlm_model``, and ``nn.Module.parameters()``
skips attributes that start with an underscore, so the weights summed to 0,
``capacity_snapshot`` returned None, ``/v1/omlx/capacity`` said
``available: false`` and a 60k prompt with ``max_tokens`` 2,000,000 passed the
route with HTTP 200 on a real model.

These tests build a tiny but real mlx-vlm Qwen3.5 model (vision tower plus the
hybrid GDN/full-attention language model), wrap it in the real adapter, and
read the capacity the way the engine does.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5 import Model as VLMModel
from mlx_vlm.models.qwen3_5 import ModelConfig
from test_scheduler_prefill_progress_commit import _tiny_hybrid_model

from omlx.engine.base import preflight_max_tokens
from omlx.exceptions import ContextExceedsCapacityError
from omlx.models.vlm import VLMModelAdapter
from omlx.scheduler import Scheduler, SchedulerConfig

MB = 1024**2
TQ_BITS = 4.0
HEAD_DIM = 16
KV_HEADS = 1
FULL_ATTENTION_LAYERS = 1  # two layers, full attention every second one


def _tiny_vlm_model(num_layers: int = 2) -> VLMModel:
    mx.random.seed(5)
    text = dict(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        full_attention_interval=2,
        vocab_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [4, 2, 2],
        },
    )
    vision = dict(
        model_type="qwen3_5",
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        out_hidden_size=32,
        num_heads=2,
        patch_size=2,
        spatial_patch_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
        fullatt_block_indexes=[0],
    )
    return VLMModel(
        ModelConfig.from_dict(
            dict(
                model_type="qwen3_5",
                text_config=text,
                vision_config=vision,
                vocab_size=32,
                image_token_id=30,
                video_token_id=31,
            )
        )
    )


def _unique_nbytes(module: nn.Module) -> int:
    seen: dict[int, int] = {}
    for _, value in tree_flatten(module.parameters()):
        seen[id(value)] = int(value.nbytes)
    return sum(seen.values())


def _scheduler(model, tokenizer, *, tq_bits=TQ_BITS, capacity=160 * MB):
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(max_num_seqs=4, paged_cache_block_size=0),
    )
    if tq_bits is not None:
        scheduler._turboquant_kv_bits = float(tq_bits)
        scheduler._set_model_info_for_monitor()
    # What the enforcer propagates: guard on, memory scheduling on, a
    # capacity line.
    scheduler._prefill_memory_guard = True
    scheduler._memory_scheduling = True
    scheduler._memory_hard_limit_bytes = capacity
    scheduler._memory_hard_watermark_bytes = capacity
    scheduler._prefill_headroom_safety = 1.0
    scheduler._current_usage_bytes = lambda **_kw: 0
    return scheduler


@pytest.fixture
def vlm():
    return _tiny_vlm_model()


def test_vlm_adapter_hides_its_model_from_parameters(vlm):
    """The premise: the adapter alone reports no parameters at all."""
    adapter = VLMModelAdapter(vlm)
    assert tree_flatten(adapter.parameters()) == []
    assert _unique_nbytes(vlm) > 0


def test_capacity_snapshot_is_available_behind_the_real_vlm_adapter(
    vlm, mock_tokenizer
):
    scheduler = _scheduler(VLMModelAdapter(vlm), mock_tokenizer)

    snapshot = scheduler.capacity_snapshot(context_tokens=1000, max_tokens=500)

    assert snapshot is not None
    # Vision tower and language model, each array counted once (the adapter
    # also holds ``_language_model``, the same arrays again).
    assert snapshot["weights_bytes"] == _unique_nbytes(vlm)
    assert snapshot["capacity_bytes"] == scheduler._capacity_limit_bytes() > 0
    assert snapshot["requested_context_tokens"] == 1000
    assert snapshot["requested_max_tokens"] == 500
    assert snapshot["required_bytes"] > snapshot["weights_bytes"]
    assert snapshot["max_viable_context_tokens"] > 0


def test_capacity_kv_per_token_prices_the_turboquant_prefill_peak(vlm, mock_tokenizer):
    """TurboQuant quantizes after prefill: the peak holds the full-width KV
    and the quantized copy. A lone full-attention layer is quantized (the
    last-layer skip applies only with more than one)."""
    scheduler = _scheduler(VLMModelAdapter(vlm), mock_tokenizer)

    snapshot = scheduler.capacity_snapshot()

    base = 2.0  # bf16/fp16 activations
    tq = TQ_BITS / 8.0 + 2.0 / HEAD_DIM
    per_token = FULL_ATTENTION_LAYERS * KV_HEADS * HEAD_DIM * 2 * (base + tq)
    assert snapshot["kv_bytes_per_token"] == round(per_token)
    monitor = scheduler.memory_monitor
    assert monitor.estimate_prompt_kv_bytes(1024) == pytest.approx(1024 * per_token)


def test_capacity_kv_per_token_with_turboquant_on_two_full_attention_layers(
    mock_tokenizer,
):
    """Same check where TurboQuant actually shrinks a layer: four layers, two
    full attention, the last one kept at full width."""
    four = _tiny_vlm_model(num_layers=4)
    scheduler = _scheduler(VLMModelAdapter(four), mock_tokenizer)

    snapshot = scheduler.capacity_snapshot()

    base = 2.0
    tq = TQ_BITS / 8.0 + 2.0 / HEAD_DIM
    stored = (1 * tq + base) / 2  # one quantized layer, the last at full width
    per_token = 2 * KV_HEADS * HEAD_DIM * 2 * (base + stored)
    assert snapshot is not None
    assert snapshot["weights_bytes"] == _unique_nbytes(four)
    assert snapshot["kv_bytes_per_token"] == round(per_token)


def test_quantized_vlm_weights_count_scales_and_biases(vlm, mock_tokenizer):
    nn.quantize(vlm.language_model, group_size=32, bits=4)
    scheduler = _scheduler(VLMModelAdapter(vlm), mock_tokenizer)

    weights = scheduler._model_weights_bytes()

    assert weights == _unique_nbytes(vlm)
    names = {name for name, _ in tree_flatten(vlm.parameters())}
    assert any(name.endswith(".scales") for name in names)


def test_text_model_on_the_batched_engine_has_a_capacity(mock_tokenizer):
    model = _tiny_hybrid_model()
    scheduler = _scheduler(model, mock_tokenizer)

    snapshot = scheduler.capacity_snapshot(context_tokens=100, max_tokens=10)

    assert snapshot is not None
    assert snapshot["weights_bytes"] == _unique_nbytes(model)


def test_max_tokens_two_million_is_priced_unclamped_and_never_fits(vlm, mock_tokenizer):
    """No route clamps max_tokens to the context window (only the prompt is
    validated), so the gate prices the value the request asked for and
    rejects it in one admission with the structured error."""
    scheduler = _scheduler(VLMModelAdapter(vlm), mock_tokenizer)
    max_tokens = preflight_max_tokens({"max_tokens": 2_000_000})
    assert max_tokens == 2_000_000

    with pytest.raises(ContextExceedsCapacityError) as excinfo:
        scheduler.preflight_or_raise(
            num_prompt_tokens=60_000,
            request_id="never",
            max_tokens=max_tokens,
        )

    fields = excinfo.value.capacity_fields
    assert fields["requested_context_tokens"] == 60_000
    assert fields["requested_max_tokens"] == 2_000_000
    assert fields["required_bytes"] > fields["capacity_bytes"]
    assert fields["capacity_bytes"] == scheduler._capacity_limit_bytes()
    assert 0 < fields["max_viable_context_tokens"] < 2_060_000


def test_the_same_prompt_with_a_small_max_tokens_passes_the_gate(vlm, mock_tokenizer):
    scheduler = _scheduler(VLMModelAdapter(vlm), mock_tokenizer)

    scheduler.preflight_or_raise(
        num_prompt_tokens=60_000, request_id="fits", max_tokens=256
    )


@pytest.mark.parametrize("engine_cls", ["vlm", "batched"])
def test_engine_capacity_reaches_the_scheduler_behind_the_core(
    engine_cls, vlm, mock_tokenizer
):
    """No engine type check stands in the way: both engines read the
    scheduler behind ``_engine.engine`` (AsyncEngineCore -> EngineCore)."""
    from types import SimpleNamespace

    from omlx.engine.batched import BatchedEngine
    from omlx.engine.vlm import VLMBatchedEngine

    cls = VLMBatchedEngine if engine_cls == "vlm" else BatchedEngine
    model = VLMModelAdapter(vlm) if engine_cls == "vlm" else _tiny_hybrid_model()
    scheduler = _scheduler(model, mock_tokenizer)
    engine = SimpleNamespace(
        _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    )
    engine._capacity_scheduler = lambda: cls._capacity_scheduler(engine)

    snapshot = cls.capacity_snapshot(engine, context_tokens=10, max_tokens=5)
    hint = cls.max_context_tokens_hint(engine)

    assert snapshot is not None and snapshot["weights_bytes"] > 0
    assert isinstance(hint, int) and hint == snapshot["max_viable_context_tokens"]
