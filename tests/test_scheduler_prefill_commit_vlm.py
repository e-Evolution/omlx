# SPDX-License-Identifier: Apache-2.0
"""Incremental prefill commit on the VLM engine path.

``VLMBatchedEngine`` runs a text prompt through the same chunked prefill as any
other engine, but a prompt whose conversation carries an image (pi resends the
whole history every turn) arrives with fused embeddings and request-wide media
cache keys. It takes the external prefill loop instead. These tests drive that
path against a small initialized Qwen3.5 hybrid wrapped to accept
``inputs_embeds`` the way the mlx-vlm language model does.
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_scheduler_prefill_progress_commit import (  # noqa: F401
    BLOCK,
    PROMPT,
    _decode_to_finish,
    _Rig,
    gated,
)

from omlx.request import Request, SamplingParams

IMAGE_KEY = ("image-a",)
IMAGE_START = 6  # inside the second block: block one stays a plain text prefix


class _VLMLikeModel(nn.Module):
    """Hybrid text model with the mlx-vlm language-model call signature."""

    def __init__(self, inner, minimum_prefix=0):
        super().__init__()
        self.inner = inner
        self._minimum_prefix = minimum_prefix

    @property
    def layers(self):
        return self.inner.layers

    def make_cache(self):
        return self.inner.make_cache()

    def embed(self, tokens):
        return self.inner.language_model.model.embed_tokens(mx.array([tokens]))

    def clear_vlm_position_state(self):
        pass

    def __call__(self, inputs, cache=None, inputs_embeds=None, **_):
        return self.inner(inputs, cache=cache, input_embeddings=inputs_embeds)

    def minimum_prefill_prefix(self, tokens):
        return self._minimum_prefix


@pytest.fixture
def split(request):
    return request.param


def _rig(tmp_path, tokenizer, split, *, commit_blocks=2, minimum_prefix=0):
    rig = _Rig(tmp_path, tokenizer, split=split, commit_blocks=commit_blocks)
    wrapped = _VLMLikeModel(rig.model, minimum_prefix)
    rig.scheduler.model = wrapped
    rig.wrapped = wrapped
    return rig


def _media_request(rig, rid="mm", prompt=PROMPT, **overrides):
    embeds = rig.wrapped.embed(list(prompt))
    request = Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=1),
        vlm_inputs_embeds=embeds,
        vlm_image_hash=IMAGE_KEY[0],
        vlm_cache_key_start=IMAGE_START,
        **overrides,
    )
    rig.scheduler.add_request(request)
    rig.scheduler.waiting.clear()
    request.remaining_tokens = list(prompt)
    return request, embeds


def _prefill_externally(rig, request, embeds):
    rig.scheduler._do_external_prefill(
        request, list(request.prompt_token_ids), None, vlm_embeds=(embeds, {}, 0)
    )
    rig.drain()


def _lookup(rig, tokens, keyed):
    kwargs = (
        {"extra_keys": IMAGE_KEY, "extra_key_token_start": IMAGE_START} if keyed else {}
    )
    return rig.prefix.fetch_cache("probe", list(tokens), **kwargs)


@pytest.mark.parametrize(
    "split", [False, True], ids=["inline-gdn", "split-gdn"], indirect=True
)
def test_media_prompt_commits_progress_under_its_own_cache_keys(
    tmp_path, mock_tokenizer, split
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        request, embeds = _media_request(rig)
        _prefill_externally(rig, request, embeds)
        assert rig.table_tokens("mm") == 4 * BLOCK

        # A re-issue of the same media prompt resumes at the committed boundary
        # and the restored prefix equals a cold forward pass.
        assert rig.scheduler._do_abort_request("mm")
        table, remaining = _lookup(rig, PROMPT, keyed=True)
        assert table is not None and table.num_tokens == 4 * BLOCK
        assert remaining == PROMPT[4 * BLOCK :]
        restored = rig.prefix.reconstruct_cache(table)
        assert restored is not None
        with mx.stream(rig.scheduler._stream):
            warm = rig.wrapped(
                mx.array([remaining]),
                cache=restored,
                inputs_embeds=rig.wrapped.embed(remaining),
            )
            cold = rig.wrapped(
                mx.array([PROMPT]),
                cache=rig.wrapped.make_cache(),
                inputs_embeds=rig.wrapped.embed(PROMPT),
            )
            mx.eval(warm, cold)
        assert mx.allclose(warm[:, -1], cold[:, -1], atol=1e-4)
    finally:
        rig.close()


@pytest.mark.parametrize(
    "split", [False, True], ids=["inline-gdn", "split-gdn"], indirect=True
)
def test_committed_media_blocks_are_not_visible_to_a_text_only_lookup(
    tmp_path, mock_tokenizer, split
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        request, embeds = _media_request(rig)
        _prefill_externally(rig, request, embeds)
        assert rig.scheduler._do_abort_request("mm")

        # Block one ends before the image starts, so it is an ordinary text
        # prefix; every later block carries the media key and must not match.
        table, remaining = _lookup(rig, PROMPT, keyed=False)
        assert table is not None and table.num_tokens == BLOCK
        assert remaining == PROMPT[BLOCK:]
    finally:
        rig.close()


@pytest.mark.parametrize(
    "split", [False, True], ids=["inline-gdn", "split-gdn"], indirect=True
)
def test_media_commit_matches_the_final_store_keys(tmp_path, mock_tokenizer, split):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        request, embeds = _media_request(rig)
        _prefill_externally(rig, request, embeds)
        committed = [
            rig.paged.allocated_blocks[b].block_hash
            for b in rig.paged.get_block_table("mm").block_ids
        ]
        # The chain the final store would produce for the same prompt and keys.
        from omlx.cache.paged_cache import compute_block_hash, resolve_block_extra_keys

        parent, expected = None, []
        for start in range(0, 4 * BLOCK, BLOCK):
            parent = compute_block_hash(
                parent,
                PROMPT[start : start + BLOCK],
                extra_keys=resolve_block_extra_keys(
                    start + BLOCK,
                    extra_keys=IMAGE_KEY,
                    extra_key_token_start=IMAGE_START,
                ),
                model_name=rig.paged.model_name,
            )
            expected.append(parent)
        assert committed == expected
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_models_needing_an_atomic_media_prefix_do_not_commit(
    tmp_path, mock_tokenizer, split, caplog
):
    rig = _rig(tmp_path, mock_tokenizer, split, minimum_prefix=BLOCK)
    try:
        request, embeds = _media_request(rig)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _prefill_externally(rig, request, embeds)
        assert rig.table_tokens("mm") == 0
        assert any(
            "skipped for mm" in r.getMessage() and "atomic" in r.getMessage()
            for r in caplog.records
        )
    finally:
        rig.close()


@pytest.mark.parametrize(
    "split", [False, True], ids=["inline-gdn", "split-gdn"], indirect=True
)
def test_text_prompt_on_the_vlm_engine_commits_through_chunked_prefill(
    tmp_path, mock_tokenizer, split
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        request = Request(
            request_id="txt",
            prompt=list(PROMPT),
            sampling_params=SamplingParams(max_tokens=1),
        )
        rig.scheduler.add_request(request)
        rig.scheduler.waiting.clear()
        request.remaining_tokens = list(PROMPT)
        state = rig.scheduler._begin_prefill(request, list(PROMPT), None)
        rig.step(state, 3)
        assert rig.table_tokens("txt") == 2 * BLOCK
    finally:
        rig.close()


# --- Observability: a prefill that is not durable always says why ---------------


def _info(caplog, needle):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and needle in r.getMessage()
    ]


def _chunked_text_prefill(rig, rid="r1", chunks=4):
    request = Request(
        request_id=rid,
        prompt=list(PROMPT),
        sampling_params=SamplingParams(max_tokens=1),
    )
    rig.scheduler.add_request(request)
    rig.scheduler.waiting.clear()
    request.remaining_tokens = list(PROMPT)
    state = rig.scheduler._begin_prefill(request, list(PROMPT), None)
    rig.step(state, chunks)
    return request


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_each_successful_commit_logs_its_token_range_once(
    tmp_path, mock_tokenizer, split, caplog
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _chunked_text_prefill(rig)
        assert _info(caplog, "Committing prefill progress for r1") == [
            "Committing prefill progress for r1: tokens [0, 8)",
            "Committing prefill progress for r1: tokens [8, 16)",
        ]
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_media_commit_log_names_the_media_keys(tmp_path, mock_tokenizer, split, caplog):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        request, embeds = _media_request(rig)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _prefill_externally(rig, request, embeds)
        assert _info(
            caplog, "Committing prefill progress for mm: tokens [0, 8) under media"
        )
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_waiting_for_the_first_commit_is_logged_once_per_request(
    tmp_path, mock_tokenizer, split, caplog
):
    rig = _rig(tmp_path, mock_tokenizer, split, commit_blocks=3)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _chunked_text_prefill(rig, chunks=2)
        lines = _info(caplog, "Prefill progress commit skipped for r1")
        assert len(lines) == 1
        assert "of 3 blocks complete" in lines[0]
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_pressure_skip_is_logged_once_with_the_usage(
    tmp_path, mock_tokenizer, split, caplog, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        monkeypatch.setattr(
            rig.scheduler, "_prefill_commit_blocked_by_pressure", lambda: True
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _chunked_text_prefill(rig, chunks=4)
        lines = [
            m
            for m in _info(caplog, "Prefill progress commit skipped for r1")
            if "soft limit" in m
        ]
        assert len(lines) == 1
        assert rig.table_tokens("r1") == 0
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_missing_snapshot_skip_is_logged_once(
    tmp_path, mock_tokenizer, split, caplog, monkeypatch
):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        monkeypatch.setattr(
            rig.scheduler, "_get_boundary_store_override", lambda *a: None
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            _chunked_text_prefill(rig, chunks=4)
        lines = _info(caplog, "no recurrent-state snapshot")
        assert len(lines) == 1 and "skipped for r1" in lines[0]
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_opt_out_and_specprefill_skips_say_so(tmp_path, mock_tokenizer, split, caplog):
    rig = _rig(tmp_path, mock_tokenizer, split)
    try:
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            request, embeds = _media_request(rig, rid="opt", skip_cache_store=True)
            _prefill_externally(rig, request, embeds)
            request, embeds = _media_request(rig, rid="sp")
            request.specprefill_indices = mx.array([0, 1])
            _prefill_externally(rig, request, embeds)
        assert _info(caplog, "skipped for opt: the request opted out")
        assert _info(caplog, "skipped for sp: sparse prefill")
        assert rig.table_tokens("opt") == 0 and rig.table_tokens("sp") == 0
    finally:
        rig.close()


# --- Park on the VLM engine ------------------------------------------------------


def _park_rig(tmp_path, tokenizer):
    import test_scheduler_park as park

    rig = park.make_rig(tmp_path, tokenizer)
    rig.wrapped = rig.scheduler.model = _VLMLikeModel(rig.model)
    return park, rig


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_text_prompt_in_chunked_prefill_on_the_vlm_engine_is_a_park_victim(
    tmp_path, mock_tokenizer, split
):
    park, rig = _park_rig(tmp_path, mock_tokenizer)
    try:
        request = park.submit(rig, "txt", park.PROMPT_A, arrival=1.0)
        park.step_until(
            rig,
            lambda: park.prefilling_ids(rig) == ["txt"]
            and rig.table_tokens("txt") >= BLOCK,
        )
        assert rig.scheduler._park_exclusion_reason(request) is None
        assert rig.scheduler._select_park_victim() is request
        assert rig.scheduler.request_park_prefill() is True
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_exclusion_reasons_name_each_unparkable_case(tmp_path, mock_tokenizer, split):
    park, rig = _park_rig(tmp_path, mock_tokenizer)
    try:
        scheduler = rig.scheduler
        media, _ = _media_request(rig, rid="media")
        scheduler.prefilling.append(media)
        scheduler._prefill_states["media"] = object()
        assert "multimodal" in scheduler._park_exclusion_reason(media)

        sparse = Request(
            request_id="sparse", prompt=[3, 4], sampling_params=SamplingParams()
        )
        sparse.specprefill_indices = mx.array([0])
        assert "SpecPrefill" in scheduler._park_exclusion_reason(sparse)

        external = Request(
            request_id="ext", prompt=[3, 4], sampling_params=SamplingParams()
        )
        assert "external loop" in scheduler._park_exclusion_reason(external)

        spent = Request(
            request_id="spent", prompt=[3, 4], sampling_params=SamplingParams()
        )
        spent.park_count = scheduler.config.max_parks_per_request
        assert "park budget spent" in scheduler._park_exclusion_reason(spent)

        decoding = Request(
            request_id="dec", prompt=[3, 4], sampling_params=SamplingParams()
        )
        scheduler.running["dec"] = decoding
        assert scheduler._park_exclusion_reason(decoding) == (
            "decoding, not in chunked prefill"
        )
    finally:
        rig.close()


@pytest.mark.parametrize("split", [False], ids=["inline-gdn"], indirect=True)
def test_no_parkable_victim_is_logged_once_per_request_with_its_reason(
    tmp_path, mock_tokenizer, split, caplog
):
    park, rig = _park_rig(tmp_path, mock_tokenizer)
    try:
        scheduler = rig.scheduler
        decoding = Request(
            request_id="dec", prompt=[3, 4], sampling_params=SamplingParams()
        )
        scheduler.requests["dec"] = decoding
        scheduler.running["dec"] = decoding
        external = Request(
            request_id="ext", prompt=[3, 4], sampling_params=SamplingParams()
        )
        scheduler.requests["ext"] = external  # admitted, in the external loop
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            assert scheduler.request_park_prefill() is False
            assert scheduler.request_park_prefill() is False
        lines = _info(caplog, "Park skipped for")
        assert len(lines) == 2
        assert any("dec: decoding, not in chunked prefill" in m for m in lines)
        assert any("ext: prefill runs in the external loop" in m for m in lines)
    finally:
        rig.close()
