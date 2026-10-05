# SPDX-License-Identifier: Apache-2.0
"""Incremental prefill commit: completed blocks become durable during prefill.

The tests drive the real chunked-prefill step against a small initialized
Qwen3.5 hybrid model (GatedDeltaNet + full attention) with real paged-SSD,
boundary-snapshot and prefix-cache stores, then compare restored-prefix logits
with a cold forward pass.
"""

import concurrent.futures
import threading
import time

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

    def __init__(
        self,
        tmp_path,
        tokenizer,
        *,
        split,
        commit_blocks,
        tq_bits=None,
        max_blocks=64,
        **config_overrides,
    ):
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
                **config_overrides,
            ),
        )
        self.paged = PagedCacheManager(
            block_size=BLOCK,
            max_blocks=max_blocks,
            initial_blocks=max_blocks,
            model_name="tiny",
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
            # A commit still in flight makes the next chunk skip its own, so
            # settle each one: the commit cadence stays deterministic.
            self.drain()
        return done

    def drain(self):
        """Wait for every in-flight store or commit, then retire the commits."""
        s = self.scheduler
        s._store_cache_executor.submit(lambda: None).result(timeout=20)
        commit_executor = getattr(s, "_prefill_commit_executor", None)
        if commit_executor is not None:
            commit_executor.submit(lambda: None).result(timeout=20)
        s._process_finished_commits()

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


def _decode_to_finish(rig, request, state):
    """Insert a fully prefilled request, decode one token, return finished ids."""
    from mlx_lm.generate import BatchGenerator

    scheduler = rig.scheduler
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
    return finished


def _finish(rig, request, state):
    """Run prefill to the end, insert, decode one token and finalize."""
    scheduler = rig.scheduler
    while not rig.step(state, 1):
        pass
    scheduler._cleanup_finished(_decode_to_finish(rig, request, state))
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


# --- Teardown while a commit is in flight --------------------------------------


class _CommitGate:
    """Holds the commit worker on an Event so a test controls its lifetime."""

    def __init__(self, scheduler, monkeypatch, fail=False):
        self.release = threading.Event()
        self.entered = threading.Event()
        real = scheduler._commit_prefill_worker

        def gated(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(30)
            if fail:
                raise RuntimeError("commit exploded")
            return real(*args, **kwargs)

        monkeypatch.setattr(scheduler, "_commit_prefill_worker", gated)


@pytest.fixture
def gated(monkeypatch):
    """Factory: rig with a controllable commit worker; released on teardown."""
    gates = []

    def make(rig, fail=False):
        gate = _CommitGate(rig.scheduler, monkeypatch, fail=fail)
        gates.append(gate)
        return gate

    yield make
    for gate in gates:
        gate.release.set()


def _returns_promptly(fn, *args, timeout=5.0):
    """Run ``fn`` on a helper thread; fail if it blocks past ``timeout``."""
    box = {}

    def run():
        box["value"] = fn(*args)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"{fn.__name__} blocked on the in-flight commit"
    return box.get("value")


def _inflight_commit(rig, gate, split_chunks=2):
    """Prefill until the first commit is submitted and held by the gate."""
    _, state = rig.start("r1", PROMPT)
    for _ in range(split_chunks):
        rig.scheduler._step_prefill_chunk(state)
    assert gate.entered.wait(20)
    future, _ = rig.scheduler._prefill_commits["r1"]
    assert not future.done()
    return state


def _await_teardown(rig, gate):
    gate.release.set()
    rig.drain()
    assert not rig.scheduler._deferred_teardowns


def test_abort_with_a_commit_in_flight_defers_cleanup_until_it_completes(
    tmp_path, mock_tokenizer, gated, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)
        cleanups = []
        real_cleanup = rig.boundary.cleanup_request
        rig.boundary.cleanup_request = lambda rid: (
            cleanups.append(rid),
            real_cleanup(rid),
        )

        assert _returns_promptly(scheduler._do_abort_request, "r1")
        # The request is gone from the scheduler, but the snapshots and the
        # block table the worker is using are untouched while it runs.
        assert "r1" not in scheduler.requests
        assert scheduler._boundary_cache_snapshots.get("r1")
        assert cleanups == []
        assert scheduler.has_requests()

        _await_teardown(rig, gate)
        assert cleanups == ["r1"]
        assert "r1" not in scheduler._boundary_cache_snapshots
        assert "r1" not in rig.prefix._request_tables
        assert not scheduler._deferred_teardowns

        table, remaining = rig.probe(PROMPT)
        assert table is not None and table.num_tokens == 2 * BLOCK
        assert remaining == PROMPT[2 * BLOCK :]
        restored = rig.prefix.reconstruct_cache(table)
        assert restored is not None
        model = rig.model
        with mx.stream(scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)
            cold = model(mx.array([PROMPT]), cache=model.make_cache())
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
    finally:
        gate.release.set()
        rig.close()


def test_prefill_failure_with_a_commit_in_flight_defers_cleanup(
    tmp_path, mock_tokenizer, gated, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)

        _returns_promptly(scheduler._drop_boundary_snapshots_for_request, "r1")
        _returns_promptly(scheduler._release_paged_cache_for_request, "r1")
        assert scheduler._boundary_cache_snapshots.get("r1")
        assert scheduler.has_requests()

        _await_teardown(rig, gate)
        assert "r1" not in scheduler._boundary_cache_snapshots
        # The rejection path frees the table (release_cache), as before.
        assert rig.paged.get_block_table("r1") is None
        assert "r1" not in rig.prefix._request_tables
    finally:
        gate.release.set()
        rig.close()


def test_overflow_retry_defers_snapshot_cleanup_and_holds_readmission(
    tmp_path, mock_tokenizer, gated
):
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        request, state = rig.start("r1", PROMPT)
        for _ in range(2):
            scheduler._step_prefill_chunk(state)
        assert gate.entered.wait(20)

        _returns_promptly(scheduler._drop_boundary_snapshots_for_request, "r1")
        scheduler.waiting.append(request)
        scheduled, _ = scheduler._schedule_waiting()
        # Same request id: the head of the queue waits for its own teardown.
        assert scheduled == []
        assert list(scheduler.waiting) == [request]

        _await_teardown(rig, gate)
        assert not scheduler._teardown_pending("r1")
    finally:
        gate.release.set()
        rig.close()


def test_finish_with_a_commit_in_flight_chains_the_final_store_after_it(
    tmp_path, mock_tokenizer, monkeypatch, gated, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        request, state = rig.start("r1", PROMPT)
        while not scheduler._step_prefill_chunk(state):
            pass
        assert gate.entered.wait(20)
        events = []
        real_store = rig.prefix.store_cache

        def traced(*args, **kwargs):
            kind = "commit" if kwargs.get("cache_covers_suffix_only") else "final"
            events.append(f"{kind}-start")
            try:
                return real_store(*args, **kwargs)
            finally:
                events.append(f"{kind}-end")

        monkeypatch.setattr(rig.prefix, "store_cache", traced)
        finished = _decode_to_finish(rig, request, state)

        # MLX streams are thread-local, so this runs on the test thread; the
        # watchdog frees a regressed implementation that would block instead.
        watchdog = threading.Timer(8.0, gate.release.set)
        watchdog.start()
        started = time.monotonic()
        scheduler._cleanup_finished(finished)
        elapsed = time.monotonic() - started
        watchdog.cancel()
        assert elapsed < 4.0, "_cleanup_finished blocked on the in-flight commit"
        future = scheduler._inflight_store_futures["r1"]
        time.sleep(0.3)
        assert not future.done()  # waiting for the commit, not racing it
        assert events == []  # the final store has not touched the table yet

        gate.release.set()
        future.result(timeout=20)
        scheduler._drain_pending_async_removes()
        rig.drain()

        # Strictly sequential: the commit ends before the final store starts.
        assert events == ["commit-start", "commit-end", "final-start", "final-end"]
        table, remaining = rig.probe(PROMPT)
        assert table is not None and table.num_tokens == 4 * BLOCK
        assert len(table.block_ids) == len(set(table.block_ids)) == 4
        assert rig.prefix.reconstruct_cache(table) is not None
        assert not scheduler._deferred_teardowns
        assert "r1" not in scheduler._prefill_commits
    finally:
        gate.release.set()
        rig.close()


def test_commit_that_raises_still_runs_the_teardown_and_leaves_no_blocks(
    tmp_path, mock_tokenizer, gated, split
):
    rig = _Rig(tmp_path, mock_tokenizer, split=split, commit_blocks=2)
    gate = gated(rig, fail=True)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")

        gate.release.set()
        rig.drain()
        assert not scheduler._deferred_teardowns
        assert "r1" not in scheduler._prefill_commits
        assert "r1" not in scheduler._boundary_cache_snapshots
        assert rig.paged.get_block_table("r1") is None
        table, remaining = rig.probe(PROMPT)
        assert table is None
        assert remaining == PROMPT
    finally:
        gate.release.set()
        rig.close()


def test_step_retires_completed_commits_and_runs_their_teardown(
    tmp_path, mock_tokenizer, gated
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")
        gate.release.set()
        scheduler._prefill_commits["r1"][0].result(timeout=20)

        scheduler.step()
        assert not scheduler._deferred_teardowns
        assert "r1" not in scheduler._boundary_cache_snapshots
    finally:
        gate.release.set()
        rig.close()


def test_commits_do_not_queue_behind_a_final_store_of_another_request(
    tmp_path, mock_tokenizer
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    blocker = threading.Event()
    try:
        scheduler = rig.scheduler
        # A long post-finish store occupies the store executor.
        scheduler._store_cache_executor.submit(blocker.wait, 30)
        _, state = rig.start("r1", PROMPT)
        scheduler._step_prefill_chunk(state)
        scheduler._step_prefill_chunk(state)
        future, _ = scheduler._prefill_commits["r1"]
        future.result(timeout=10)  # would time out behind the blocked store
        scheduler._process_finished_commits()
        assert "r1" not in scheduler._prefill_commits
        assert rig.table_tokens("r1") == 2 * BLOCK
    finally:
        blocker.set()
        rig.close()


def test_same_prompt_reissue_during_a_pending_teardown_keeps_blocks_alive(
    tmp_path, mock_tokenizer, gated
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")
        gate.release.set()
        scheduler._prefill_commits["r1"][0].result(timeout=20)

        # A re-issue matches the committed blocks before the teardown runs.
        table, _ = rig.probe(PROMPT)
        assert table is not None and table.num_tokens == 2 * BLOCK
        scheduler._process_finished_commits()

        for block_id in table.block_ids:
            assert rig.paged.allocated_blocks[block_id].ref_count == 1
        assert "r1" not in rig.prefix._request_tables
    finally:
        gate.release.set()
        rig.close()


def test_shutdown_completes_pending_teardowns_within_the_bounded_wait(
    tmp_path, mock_tokenizer, gated, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_SHUTDOWN_WAIT_S", 5.0)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    _inflight_commit(rig, gate)
    _returns_promptly(scheduler._do_abort_request, "r1")
    threading.Timer(0.3, gate.release.set).start()

    scheduler.shutdown()

    assert not scheduler._deferred_teardowns
    assert not scheduler._prefill_commits
    assert "r1" not in scheduler._boundary_cache_snapshots


def test_shutdown_abandons_a_commit_that_never_completes(
    tmp_path, mock_tokenizer, gated, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_SHUTDOWN_WAIT_S", 0.2)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    _inflight_commit(rig, gate)
    _returns_promptly(scheduler._do_abort_request, "r1")

    started = time.monotonic()
    scheduler.shutdown()
    assert time.monotonic() - started < 10
    assert not scheduler._deferred_teardowns
    # The snapshot directory is gone even though the commit never finished.
    assert not any((tmp_path / "cache").rglob("*r1*"))


# --- Reaping, admission, shutdown, reset barrier -------------------------------


def _orphans(scheduler):
    """Deferred teardowns with neither a tracked commit nor a completion to run on."""
    return [
        rid
        for rid in scheduler._deferred_teardowns
        if rid not in scheduler._prefill_commits
    ]


def test_teardowns_drain_when_another_path_reaps_the_commit_first(
    tmp_path, mock_tokenizer, gated
):
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")
        assert scheduler._teardown_pending("r1")
        gate.release.set()
        scheduler._prefill_commits["r1"][0].result(timeout=20)

        # _cleanup_finished-style reap, before _process_finished_commits runs.
        assert scheduler._inflight_commit_future("r1") is None

        assert _orphans(scheduler) == []
        assert not scheduler._teardown_pending("r1")
        assert "r1" not in scheduler._boundary_cache_snapshots
        assert not scheduler._deferred_teardowns  # has_requests() is not pinned
    finally:
        gate.release.set()
        rig.close()


def test_orphaned_deferred_teardowns_are_swept_at_the_next_step(
    tmp_path, mock_tokenizer
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    try:
        scheduler = rig.scheduler
        ran = []
        scheduler._deferred_teardowns["ghost"] = [lambda: ran.append("ghost")]
        scheduler._deferred_teardown_since["ghost"] = time.monotonic()

        scheduler._process_finished_commits()

        assert ran == ["ghost"]
        assert _orphans(scheduler) == []
        assert not scheduler._teardown_pending("ghost")
    finally:
        rig.close()


def test_unrelated_request_is_admitted_behind_a_held_head(
    tmp_path, mock_tokenizer, gated
):
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        request, state = rig.start("r1", PROMPT)
        for _ in range(2):
            scheduler._step_prefill_chunk(state)
        assert gate.entered.wait(20)
        _returns_promptly(scheduler._drop_boundary_snapshots_for_request, "r1")
        other = Request(
            request_id="r2",
            prompt=list(range(40, 57)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        third = Request(
            request_id="r3",
            prompt=list(range(60, 77)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(other)
        scheduler.add_request(third)
        scheduler.waiting.clear()
        scheduler.waiting.extend([request, other, third])

        scheduled, _ = scheduler._schedule_waiting()

        assert [r.request_id for r in scheduled] + [
            r.request_id for r in scheduler.prefilling
        ] != []
        queued = [r.request_id for r in scheduler.waiting]
        assert queued[0] == "r1"  # the held head keeps its place
        assert "r2" not in queued  # the unrelated request got through
    finally:
        gate.release.set()
        rig.close()


class _BlockingStore:
    """Replaces store_cache with a call that blocks until released."""

    def __init__(self, rig, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        real = rig.prefix.store_cache

        def blocking(*args, **kwargs):
            self.calls += 1
            self.entered.set()
            assert self.release.wait(30)
            return real(*args, **kwargs)

        monkeypatch.setattr(rig.prefix, "store_cache", blocking)


def test_shutdown_cancels_a_commit_that_has_not_reached_the_store(
    tmp_path, mock_tokenizer, gated, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_SHUTDOWN_WAIT_S", 0.2)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    _inflight_commit(rig, gate)
    future, _ = scheduler._prefill_commits["r1"]
    stores = []
    real_store = rig.prefix.store_cache
    monkeypatch.setattr(
        rig.prefix,
        "store_cache",
        lambda *a, **k: (stores.append(1), real_store(*a, **k))[1],
    )

    started = time.monotonic()
    scheduler.shutdown()
    assert time.monotonic() - started < 10

    gate.release.set()
    assert future.result(timeout=10) is False  # bailed out, no store attempted
    assert stores == []


def test_shutdown_does_not_close_the_stores_under_an_active_store_call(
    tmp_path, mock_tokenizer, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_SHUTDOWN_WAIT_S", 0.2)
    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_CANCEL_WAIT_S", 0.3)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    blocking = _BlockingStore(rig, monkeypatch)
    scheduler = rig.scheduler
    closed = []
    real_ssd_close = rig.ssd.close
    real_boundary_shutdown = rig.boundary.shutdown
    monkeypatch.setattr(
        rig.ssd,
        "close",
        lambda *a, **k: (closed.append("ssd"), real_ssd_close(*a, **k)),
    )
    monkeypatch.setattr(
        rig.boundary,
        "shutdown",
        lambda *a, **k: (closed.append("boundary"), real_boundary_shutdown(*a, **k)),
    )
    try:
        _, state = rig.start("r1", PROMPT)
        for _ in range(2):
            scheduler._step_prefill_chunk(state)
        assert blocking.entered.wait(20)

        started = time.monotonic()
        scheduler.shutdown()
        assert time.monotonic() - started < 10  # bounded
        assert closed == []  # worker is still inside the store call
    finally:
        blocking.release.set()
        time.sleep(0.2)
        real_ssd_close()
        real_boundary_shutdown()


def test_commit_executor_threads_cannot_block_interpreter_exit(
    tmp_path, mock_tokenizer
):
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    try:
        executor = rig.scheduler._ensure_prefill_commit_executor()
        seen = executor.submit(lambda: threading.current_thread()).result(timeout=10)
        assert seen.daemon
        assert seen.name.startswith("omlx-prefill-commit")
    finally:
        rig.close()


def test_reset_runs_deferred_teardowns_before_clearing_state(
    tmp_path, mock_tokenizer, gated, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "FATAL_TEARDOWN_TIMEOUT_S", 10.0)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    try:
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")
        events = []
        real_cleanup = rig.boundary.cleanup_request
        real_clear = rig.prefix.clear
        monkeypatch.setattr(
            rig.boundary,
            "cleanup_request",
            lambda rid: (events.append("teardown"), real_cleanup(rid))[1],
        )
        monkeypatch.setattr(
            rig.prefix, "clear", lambda: (events.append("clear"), real_clear())[1]
        )
        threading.Timer(0.3, gate.release.set).start()

        scheduler.reset()

        assert events[:2] == ["teardown", "clear"]
        assert not scheduler._deferred_teardowns
        assert not scheduler._prefill_commits
    finally:
        gate.release.set()
        rig.close()


def test_reset_is_fatal_when_a_commit_outlives_the_barrier(
    tmp_path, mock_tokenizer, gated, monkeypatch
):
    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "FATAL_TEARDOWN_TIMEOUT_S", 0.2)
    fatal = []
    monkeypatch.setattr(scheduler_module, "fatal_exit", lambda msg: fatal.append(msg))
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    try:
        _inflight_commit(rig, gate)
        scheduler.reset()
        assert len(fatal) == 1
        assert "prefill commit" in fatal[0]
        assert "r1" in scheduler._prefill_commits  # nothing was cleared
    finally:
        gate.release.set()
        rig.close()


def test_stuck_teardown_warning_is_logged_once(
    tmp_path, mock_tokenizer, gated, monkeypatch, caplog
):
    import logging

    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_STUCK_LOG_S", 0.0)
    rig = _Rig(tmp_path, mock_tokenizer, split=True, commit_blocks=2)
    gate = gated(rig)
    scheduler = rig.scheduler
    try:
        _inflight_commit(rig, gate)
        _returns_promptly(scheduler._do_abort_request, "r1")
        with caplog.at_level(logging.WARNING, logger="omlx.scheduler"):
            for _ in range(3):
                scheduler._process_finished_commits()
        stuck = [r for r in caplog.records if "still waits" in r.getMessage()]
        assert len(stuck) == 1
        assert "r1" in stuck[0].getMessage()
    finally:
        gate.release.set()
        rig.close()


# --- Bounded wait of the final store on an in-flight commit --------------------


def test_stuck_commit_times_out_the_final_store_and_frees_the_slot(
    tmp_path, mock_tokenizer, monkeypatch, gated, caplog
):
    """A commit that never ends must not hold the final-store worker or gate.

    The final store is skipped (logged once), the store-gate slot is released,
    and a later request's final store proceeds on the same worker thread.
    """
    import logging

    import omlx.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PREFILL_COMMIT_FINAL_STORE_WAIT_S", 0.3)
    rig = _Rig(tmp_path, mock_tokenizer, split=False, commit_blocks=2)
    gate = gated(rig)
    try:
        scheduler = rig.scheduler
        request, state = rig.start("r1", PROMPT)
        while not scheduler._step_prefill_chunk(state):
            pass
        assert gate.entered.wait(20)
        finished = _decode_to_finish(rig, request, state)

        class _Gate:
            submitted = 0
            done = 0

            def note_submitted(self):
                self.submitted += 1

            def note_done(self):
                self.done += 1

        store_gate = _Gate()
        scheduler._store_cache_gate = store_gate
        with caplog.at_level(logging.WARNING, logger="omlx.scheduler"):
            scheduler._cleanup_finished(finished)
            scheduler._inflight_store_futures["r1"].result(timeout=10)
            scheduler._drain_pending_async_removes()

        # Skipped, logged once, slot released although the commit still runs.
        timeouts = [r for r in caplog.records if "final store" in r.getMessage()]
        assert len(timeouts) == 1
        assert store_gate.submitted == 1 and store_gate.done == 1
        assert "r1" not in scheduler._inflight_store_futures
        assert not scheduler._pending_async_removes

        # Table state is released through the deferred teardown once it ends.
        assert scheduler._teardown_pending("r1")
        gate.release.set()
        rig.drain()
        assert not scheduler._teardown_pending("r1")
        assert "r1" not in rig.prefix._request_tables
        # The blocks the commit stored stay reusable by a re-issue.
        table, _ = rig.probe(PROMPT)
        assert table is not None and table.num_tokens >= BLOCK

        # A later final store proceeds.
        request2, state2 = rig.start("r2", list(range(40, 57)))
        _finish(rig, request2, state2)
        table, _ = rig.probe(list(range(40, 57)))
        assert table is not None and table.num_tokens == 4 * BLOCK
    finally:
        gate.release.set()
        rig.close()
