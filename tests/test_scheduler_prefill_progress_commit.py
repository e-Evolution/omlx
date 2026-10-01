# SPDX-License-Identifier: Apache-2.0
"""Incremental prefill commit: completed blocks become durable during prefill.

The tests drive the real chunked-prefill step against a small initialized
Qwen3.5 hybrid model (GatedDeltaNet + full attention) with real paged-SSD,
boundary-snapshot and prefix-cache stores, then compare restored-prefix logits
with a cold forward pass.
"""

import concurrent.futures

import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

BLOCK = 4


def _tiny_hybrid_model():
    return Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                full_attention_interval=2,
                vocab_size=32,
                linear_num_value_heads=2,
                linear_num_key_heads=2,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_conv_kernel_dim=4,
                partial_rotary_factor=1.0,
            ),
        )
    )


class _Rig:
    """A scheduler wired to real stores, plus helpers to drive prefill."""

    def __init__(self, tmp_path, tokenizer, *, split, commit_blocks, tq_bits=None):
        mx.random.seed(11)
        self.model = _tiny_hybrid_model()
        self.scheduler = Scheduler(
            model=self.model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                paged_cache_block_size=BLOCK,
                gdn_ssd_split_enabled=split,
                chunked_prefill=True,
                prefill_step_size=BLOCK,
                prefill_progress_commit_blocks=commit_blocks,
            ),
        )
        self.paged = PagedCacheManager(
            block_size=BLOCK, max_blocks=64, initial_blocks=64, model_name="tiny"
        )
        kv_type = "TurboQuantKVCache" if tq_bits else "KVCache"
        self.ssd = PagedSSDCacheManager(
            cache_dir=tmp_path / "cache",
            max_size_bytes=16 * 1024**2,
            expected_model_name="tiny",
            expected_num_layers=2,
            expected_block_size=BLOCK,
            expected_layer_cache_types=["ArraysCache", kv_type],
            gdn_ssd_split_enabled=split,
        )
        self.boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
        self.prefix = BlockAwarePrefixCache(
            model=self.model,
            paged_cache_manager=self.paged,
            paged_ssd_cache_manager=self.ssd,
            gdn_ssd_split_enabled=split,
        )
        self.prefix.set_gdn_checkpoint_loader(self.boundary.load_file)
        s = self.scheduler
        s.paged_cache_manager = self.paged
        s.paged_ssd_cache_manager = self.ssd
        s._boundary_snapshot_store = self.boundary
        s.block_aware_cache = self.prefix
        s._store_cache_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="omlx-store-cache"
        )
        if tq_bits:
            s._turboquant_kv_bits = float(tq_bits)
            s.refresh_ssd_layer_signature()

    def start(self, request_id, tokens):
        request = Request(
            request_id=request_id,
            prompt=list(tokens),
            sampling_params=SamplingParams(max_tokens=1),
        )
        self.scheduler.add_request(request)
        self.scheduler.waiting.clear()
        request.remaining_tokens = list(tokens)
        return request, self.scheduler._begin_prefill(request, list(tokens), None)

    def step(self, state, chunks):
        done = False
        for _ in range(chunks):
            done = self.scheduler._step_prefill_chunk(state)
        self.drain()
        return done

    def drain(self):
        """Wait for every in-flight store or commit on the store executor."""
        self.scheduler._store_cache_executor.submit(lambda: None).result(timeout=20)

    def close(self):
        self.scheduler.shutdown()

    def table_tokens(self, request_id):
        table = self.paged.get_block_table(request_id)
        return table.num_tokens if table else 0

    def probe(self, tokens):
        """Prefix lookup a fresh request with the same prompt would perform."""
        table, remaining = self.prefix.fetch_cache("probe", list(tokens))
        return table, remaining


PROMPT = list(range(3, 20))  # 17 tokens -> 16 prefill tokens -> four blocks


@pytest.fixture(params=[False, True], ids=["inline-gdn", "split-gdn"])
def split(request):
    return request.param


def test_setting_off_stores_nothing_during_prefill(tmp_path, mock_tokenizer, split):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=0)
    try:
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 3)
        assert rig.table_tokens("r1") == 0
        table, remaining = rig.probe(PROMPT)
        assert table is None
        assert remaining == PROMPT
    finally:
        rig.close()


def test_commit_every_n_blocks_stores_completed_blocks(tmp_path, mock_tokenizer, split):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    try:
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 1)
        assert rig.table_tokens("r1") == 0  # one block done, below the threshold
        rig.step(state, 1)
        assert rig.table_tokens("r1") == 2 * BLOCK
        rig.step(state, 1)
        assert rig.table_tokens("r1") == 2 * BLOCK  # third block waits for N=2
        rig.step(state, 1)
        assert rig.table_tokens("r1") == 4 * BLOCK
    finally:
        rig.close()


def test_abort_mid_prefill_keeps_committed_blocks_for_a_reissue(
    tmp_path, mock_tokenizer, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    try:
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 3)  # blocks 1-2 committed, block 3 only in flight
        assert rig.scheduler._do_abort_request("r1")

        table, remaining = rig.probe(PROMPT)
        assert table is not None and table.num_tokens == 2 * BLOCK
        assert remaining == PROMPT[2 * BLOCK :]

        restored = rig.prefix.reconstruct_cache(table)
        assert restored is not None
        model = rig.model
        with mx.stream(rig.scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)
            cold = model(mx.array([PROMPT]), cache=model.make_cache())
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
    finally:
        rig.close()


def test_commit_is_skipped_under_memory_pressure(
    tmp_path, mock_tokenizer, monkeypatch, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=1)
    try:
        scheduler = rig.scheduler
        pressured = {"on": True}
        monkeypatch.setattr(
            scheduler, "_prefill_commit_blocked_by_pressure", lambda: pressured["on"]
        )
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 2)
        assert rig.table_tokens("r1") == 0

        pressured["on"] = False
        rig.step(state, 1)
        # Everything completed so far is committed once pressure clears.
        assert rig.table_tokens("r1") == 3 * BLOCK
    finally:
        rig.close()


@pytest.mark.parametrize(
    "usage,limit,expected", [(500, 1000, False), (1000, 1000, True), (900, 0, False)]
)
def test_pressure_gate_follows_the_soft_limit(
    tmp_path, mock_tokenizer, monkeypatch, usage, limit, expected
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=1)
    try:
        scheduler = rig.scheduler
        scheduler._memory_limit_bytes = limit
        monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda **kw: usage)
        assert scheduler._prefill_commit_blocked_by_pressure() is expected
    finally:
        rig.close()


def _packed(layer):
    """Flat list of the arrays that make up a restored KV layer."""
    out = []

    def walk(x):
        if isinstance(x, mx.array):
            out.append(x)
        elif hasattr(x, "_fields"):
            for name in x._fields:
                walk(getattr(x, name))
        elif isinstance(x, (list, tuple)):
            for item in x:
                walk(item)

    walk(layer.state)
    return out


def test_turboquant_blocks_committed_per_window_match_whole_prefix_conversion(
    tmp_path, mock_tokenizer, split
):
    from mlx_vlm.turboquant import TurboQuantKVCache

    import omlx.turboquant_kv  # noqa: F401  (installs TurboQuantKVCache.from_cache)

    # One block per commit: the 8 stored tokens come from two separate windows.
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=1, tq_bits=4)
    try:
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 2)
        assert rig.table_tokens("r1") == 2 * BLOCK

        table, _ = rig.probe(PROMPT)
        restored = rig.prefix.reconstruct_cache(table)
        assert restored is not None
        assert type(restored[1]).__name__ == "TurboQuantKVCache"

        live = state.cache[1]  # still fp16: conversion happens after prefill
        assert type(live).__name__ == "KVCache"
        whole = TurboQuantKVCache.from_cache(_window(live, 0, 2 * BLOCK), bits=4.0)
        for got, want in zip(_packed(restored[1]), _packed(whole), strict=True):
            assert mx.array_equal(got, want).item()
    finally:
        rig.close()


def _window(layer, start, end):
    piece = type(layer)()
    piece.keys = layer.keys[..., start:end, :]
    piece.values = layer.values[..., start:end, :]
    piece.offset = end - start
    return piece


def test_window_longer_than_the_restored_prefix_is_stored_by_position(
    tmp_path, mock_tokenizer, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=1)
    try:
        # Seed the cache with the first block, as an earlier request would.
        _, state = rig.start("seed", PROMPT)
        rig.step(state, 1)
        assert rig.table_tokens("seed") == BLOCK
        assert rig.scheduler._do_abort_request("seed")

        # The re-issue restores one block; its commit window (3 blocks) is
        # then longer than the restored prefix (1 block).
        rig.scheduler.config.prefill_progress_commit_blocks = 3
        request = Request(
            request_id="again",
            prompt=list(PROMPT),
            sampling_params=SamplingParams(max_tokens=1),
        )
        rig.scheduler.add_request(request)
        rig.scheduler.waiting.clear()
        rig.scheduler._prepare_prefix_cache_for_request(request)
        assert request.cached_tokens == BLOCK
        state = rig.scheduler._begin_prefill(
            request, request.remaining_tokens[: len(PROMPT)], request.prompt_cache
        )
        rig.step(state, 3)
        assert rig.table_tokens("again") == 4 * BLOCK
        assert rig.scheduler._do_abort_request("again")

        table, remaining = rig.probe(PROMPT)
        assert table.num_tokens == 4 * BLOCK
        restored = rig.prefix.reconstruct_cache(table)
        model = rig.model
        with mx.stream(rig.scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)
            cold = model(mx.array([PROMPT]), cache=model.make_cache())
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
    finally:
        rig.close()


def test_committed_boundary_markers_are_retired_once_stored(
    tmp_path, mock_tokenizer, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    try:
        scheduler = rig.scheduler
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 3)
        scheduler._reap_prefill_commit("r1")
        # Boundaries 4 and 8 were committed (a split-GDN commit moves their
        # staged files); only the in-flight boundary 12 may remain.
        assert sorted(scheduler._boundary_cache_snapshots["r1"]) == [3 * BLOCK]
    finally:
        rig.close()


def _finish(rig, request, state):
    """Run prefill to the end, insert, decode one token and finalize."""
    from mlx_lm.generate import BatchGenerator

    scheduler = rig.scheduler
    while not rig.step(state, 1):
        pass
    scheduler._emit_final_boundary_if_needed(state)
    scheduler.batch_generator = BatchGenerator(
        rig.model,
        max_tokens=1,
        sampler=lambda logits: mx.zeros(logits.shape[:-1], dtype=mx.int32),
        stream=scheduler._stream,
    )
    sampler, per_row = scheduler._build_sampler_and_processors(request.sampling_params)
    state.sampler = sampler
    state.sm = scheduler._build_state_machine(request)
    state.per_row_lps = per_row
    scheduler._insert_prefilled_request(request, state, [])
    for _ in range(6):
        _, responses = scheduler.batch_generator.next()
        if responses:
            break
    _, finished = scheduler._process_batch_responses(responses)
    scheduler._cleanup_finished(finished)
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=20)
    scheduler._drain_pending_async_removes()
    rig.drain()


@pytest.mark.parametrize("commit_blocks", [0, 1, 2, 3])
def test_completion_after_commits_stores_the_same_prefix_once(
    tmp_path, mock_tokenizer, split, commit_blocks
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=commit_blocks)
    try:
        request, state = rig.start("r1", PROMPT)
        _finish(rig, request, state)

        table, remaining = rig.probe(PROMPT)
        assert table is not None and table.num_tokens == 4 * BLOCK
        assert remaining == PROMPT[4 * BLOCK :]
        assert len(table.block_ids) == len(set(table.block_ids)) == 4
        restored = rig.prefix.reconstruct_cache(table)
        assert restored is not None
        model = rig.model
        with mx.stream(rig.scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)
            cold = model(mx.array([PROMPT]), cache=model.make_cache())
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
        # No commit or snapshot bookkeeping outlives the request.
        assert "r1" not in rig.scheduler._prefill_commits
        assert "r1" not in rig.scheduler._boundary_cache_snapshots
    finally:
        rig.close()


def test_retry_after_prefill_oom_prefills_cold_even_with_committed_blocks(
    tmp_path, mock_tokenizer
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=1)
    try:
        _, state = rig.start("r1", PROMPT)
        rig.step(state, 2)
        assert rig.scheduler._do_abort_request("r1")

        def lookup(request_id, retries):
            request = Request(
                request_id=request_id,
                prompt=list(PROMPT),
                sampling_params=SamplingParams(max_tokens=1),
            )
            request.prefill_oom_retries = retries
            rig.scheduler.add_request(request)
            rig.scheduler._prepare_prefix_cache_for_request(request)
            return request

        # A client re-issue resumes from the committed boundary...
        assert lookup("reissue", 0).cached_tokens == 2 * BLOCK
        # ...but the scheduler's own retry after a prefill memory failure
        # stays cold (see _requeue_or_fail_prefill).
        retry = lookup("retry", 1)
        assert retry.cached_tokens == 0
        assert retry.remaining_tokens == PROMPT
    finally:
        rig.close()


def test_external_prefill_loop_commits_progress_too(tmp_path, mock_tokenizer, split):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    try:
        scheduler = rig.scheduler
        scheduler.config.chunked_prefill = False
        request = Request(
            request_id="ext",
            prompt=list(PROMPT),
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(request)
        scheduler.waiting.clear()
        request.remaining_tokens = list(PROMPT)
        scheduler.requests[request.request_id] = request
        scheduler._do_external_prefill(request, list(PROMPT), None)
        rig.drain()
        assert rig.table_tokens("ext") == 4 * BLOCK
        assert scheduler._do_abort_request("ext")
        table, remaining = rig.probe(PROMPT)
        assert table.num_tokens == 4 * BLOCK
        assert remaining == PROMPT[4 * BLOCK :]
    finally:
        rig.close()


def test_dense_attention_model_commits_without_snapshots(tmp_path, mock_tokenizer):
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(5)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    try:
        # Swap the hybrid model for a dense one under the same stores.
        rig.model = rig.scheduler.model = model
        rig.prefix.model = model
        rig.ssd._expected_layer_cache_types = ["KVCache", "KVCache"]
        _, state = rig.start("r1", PROMPT)
        assert not state.boundary_enabled
        rig.step(state, 3)
        assert rig.table_tokens("r1") == 2 * BLOCK
        assert rig.scheduler._do_abort_request("r1")

        table, remaining = rig.probe(PROMPT)
        assert table.num_tokens == 2 * BLOCK
        restored = rig.prefix.reconstruct_cache(table)
        with mx.stream(rig.scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)
            cold = model(mx.array([PROMPT]), cache=model.make_cache())
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
    finally:
        rig.close()


def test_window_of_a_restored_turboquant_prefix_slices_the_packed_state(
    tmp_path, mock_tokenizer
):
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache, _slice_state_range

    import omlx.turboquant_kv  # noqa: F401

    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=1)
    try:
        mx.random.seed(2)
        k = mx.random.normal((1, 1, 12, 16)).astype(mx.float16)
        fp16 = KVCache()
        fp16.update_and_fetch(k, k)
        packed = TurboQuantKVCache.from_cache(fp16, bits=4.0)
        tail = KVCache()  # the skipped last KV layer stays fp16
        tail.update_and_fetch(k, k)

        window = rig.scheduler._prefill_commit_window([packed, tail], 4, 12)

        assert window is not None
        assert type(window[0]).__name__ == "TurboQuantKVCache"
        assert window[0].offset == 8
        want = _slice_state_range(packed.keys, 4, 12)
        assert mx.array_equal(window[0].keys.norms[:, :, :8], want.norms[:, :, :8])
        assert type(window[1]).__name__ == "KVCache"
        assert window[1].offset == 8
    finally:
        rig.close()


def test_window_refuses_composite_and_unknown_sliceable_layers(
    tmp_path, mock_tokenizer
):
    from mlx_lm.models.cache import CacheList, KVCache

    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=1)
    try:
        k = mx.zeros((1, 1, 8, 16), dtype=mx.float16)
        kv = KVCache()
        kv.update_and_fetch(k, k)
        composite = CacheList(KVCache(), KVCache())
        assert rig.scheduler._prefill_commit_window([composite], 0, 8) is None
        # A cache that has not reached the window end yet cannot be sliced.
        assert rig.scheduler._prefill_commit_window([kv], 0, 12) is None
    finally:
        rig.close()
