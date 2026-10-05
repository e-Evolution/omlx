# SPDX-License-Identifier: Apache-2.0
"""HTTP surface of the never-fits gate: the 400 body, the capacity endpoint
and the ``X-oMLX-Max-Context-Tokens`` header."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.server as server
from omlx.engine.base import BaseEngine, GenerationOutput
from omlx.exceptions import ContextExceedsCapacityError, PrefillMemoryExceededError

FIELDS = {
    "required_bytes": 140_000_000_000,
    "capacity_bytes": 96_000_000_000,
    "weights_bytes": 28_000_000_000,
    "kv_bytes_per_token": 65_536,
    "requested_context_tokens": 1_200_000,
    "requested_max_tokens": 32_768,
    "max_viable_context_tokens": 1_036_000,
}


def _never_fits():
    return ContextExceedsCapacityError(
        message="Request needs ~130.4 GB but capacity is 89.4 GB.",
        request_id="req-huge",
        capacity_fields=dict(FIELDS),
    )


# --- 400 body ---------------------------------------------------------------------


def _handler_app():
    app = FastAPI()
    app.add_exception_handler(
        PrefillMemoryExceededError, server.prefill_memory_exceeded_handler
    )

    @app.get("/v1/raise")
    def raise_api():
        raise _never_fits()

    @app.get("/health/raise")
    def raise_other():
        raise _never_fits()

    return app


def test_never_fits_maps_to_400_with_the_existing_envelope_and_capacity_fields():
    with TestClient(_handler_app()) as client:
        resp = client.get("/v1/raise")
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    error = body["error"]
    # Existing envelope, unchanged for clients that already parse it.
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None
    assert "Request needs ~130.4 GB" in error["message"]
    assert error["estimated_bytes"] == FIELDS["required_bytes"]
    assert error["limit_bytes"] == FIELDS["capacity_bytes"]
    # New classification and numbers.
    assert error["code"] == "context_exceeds_capacity"
    assert error["omlx_code"] == "context_exceeds_capacity"
    for name, value in FIELDS.items():
        assert error[name] == value
    assert error["retryable"] is False


def test_non_api_routes_carry_the_same_fields():
    with TestClient(_handler_app()) as client:
        resp = client.get("/health/raise")
    assert resp.status_code == 400
    body = resp.json()
    assert body["omlx_code"] == "context_exceeds_capacity"
    assert body["retryable"] is False
    for name, value in FIELDS.items():
        assert body[name] == value


def test_streaming_error_payload_keeps_the_capacity_fields():
    body = server._streaming_error_payload(_never_fits(), "chat")
    assert body["error"]["omlx_code"] == "context_exceeds_capacity"
    assert body["error"]["max_viable_context_tokens"] == 1_036_000


# --- Engine hooks -------------------------------------------------------------------


def _engine_with_scheduler(scheduler):
    engine = MagicMock(spec=BaseEngine)
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    engine._capacity_scheduler = BaseEngine._capacity_scheduler.__get__(engine)
    engine.capacity_snapshot = BaseEngine.capacity_snapshot.__get__(engine)
    engine.max_context_tokens_hint = BaseEngine.max_context_tokens_hint.__get__(engine)
    return engine


def test_base_engine_reads_the_scheduler_and_ignores_non_numbers():
    scheduler = SimpleNamespace(
        capacity_snapshot=lambda **kw: {"free_bytes_now": 1, **kw},
        max_viable_context_tokens=lambda: 4096,
    )
    engine = _engine_with_scheduler(scheduler)
    assert engine.capacity_snapshot(context_tokens=5, max_tokens=6) == {
        "free_bytes_now": 1,
        "context_tokens": 5,
        "max_tokens": 6,
    }
    assert engine.max_context_tokens_hint() == 4096

    broken = _engine_with_scheduler(MagicMock())
    assert broken.capacity_snapshot() is None
    assert broken.max_context_tokens_hint() is None
    assert _engine_with_scheduler(None).max_context_tokens_hint() is None


# --- Capacity endpoint -----------------------------------------------------------


@pytest.fixture
def capacity_env(monkeypatch):
    snapshot = {**FIELDS, "free_bytes_now": 12_000_000_000}
    engine = MagicMock()
    engine.capacity_snapshot = MagicMock(return_value=snapshot)
    entries = {
        "loaded-model": SimpleNamespace(engine=engine),
        "cold-model": SimpleNamespace(engine=None),
    }
    pool = MagicMock()
    pool.get_entry = MagicMock(side_effect=entries.get)
    monkeypatch.setattr(server._server_state, "engine_pool", pool)
    monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(server._server_state, "api_key", None)
    monkeypatch.setattr(server._server_state, "global_settings", None)
    return engine, snapshot


def test_capacity_endpoint_returns_the_numbers_and_free_bytes(capacity_env):
    engine, snapshot = capacity_env
    client = TestClient(server.app)
    resp = client.get(
        "/v1/omlx/capacity",
        params={"model": "loaded-model", "context_tokens": 1000, "max_tokens": 50},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "loaded-model"
    assert body["available"] is True
    for name, value in snapshot.items():
        assert body[name] == value
    engine.capacity_snapshot.assert_called_once_with(context_tokens=1000, max_tokens=50)


def test_capacity_endpoint_unknown_cold_and_unavailable(capacity_env):
    engine, _snapshot = capacity_env
    client = TestClient(server.app)
    assert client.get("/v1/omlx/capacity", params={"model": "nope"}).status_code == 404
    assert (
        client.get("/v1/omlx/capacity", params={"model": "cold-model"}).status_code
        == 409
    )
    engine.capacity_snapshot.return_value = None
    resp = client.get("/v1/omlx/capacity", params={"model": "loaded-model"})
    assert resp.status_code == 200
    assert resp.json()["available"] is False


def test_capacity_endpoint_requires_the_api_key(capacity_env, monkeypatch):
    monkeypatch.setattr(server._server_state, "api_key", "secret-key")
    monkeypatch.setattr(server, "allows_unauthenticated_inference", lambda: False)
    client = TestClient(server.app)
    params = {"model": "loaded-model"}
    assert client.get("/v1/omlx/capacity", params=params).status_code == 401
    resp = client.get(
        "/v1/omlx/capacity",
        params=params,
        headers={"Authorization": "Bearer secret-key"},
    )
    assert resp.status_code == 200


# --- Header on chat completions -------------------------------------------------


def _chat_env(monkeypatch, hint):
    from omlx.engine.vlm import VLMBatchedEngine

    engine = MagicMock(spec=VLMBatchedEngine)
    engine.tokenizer = SimpleNamespace(
        encode=lambda t, **kw: t.split(), tool_call_start=None, tool_call_end=None
    )
    engine.start = AsyncMock()
    engine.preflight_chat = AsyncMock()
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.chat = AsyncMock(
        return_value=GenerationOutput(
            text="hello", prompt_tokens=3, completion_tokens=1, finish_reason="stop"
        )
    )
    engine.is_diffusion_model = False
    engine.message_extractor = None
    engine.supports_multimodal_fallback = False
    engine.max_context_tokens_hint = MagicMock(return_value=hint)

    async def _get_engine(model_id, *, lease=None):
        return engine

    pool = MagicMock()
    pool.get_entry = MagicMock(
        return_value=SimpleNamespace(preserve_thinking_default=False)
    )
    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(server, "validate_context_window", lambda *a, **k: None)
    monkeypatch.setattr(server, "get_engine_pool", lambda: pool)
    monkeypatch.setitem(
        server.app.dependency_overrides, server.verify_inference_api_key, lambda: True
    )
    return engine


def _chat(client):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def test_chat_completion_carries_the_max_context_header(monkeypatch):
    _chat_env(monkeypatch, hint=123_456)
    resp = _chat(TestClient(server.app))
    assert resp.status_code == 200
    assert resp.headers["X-oMLX-Max-Context-Tokens"] == "123456"


def test_chat_completion_omits_the_header_without_an_estimate(monkeypatch):
    _chat_env(monkeypatch, hint=None)
    resp = _chat(TestClient(server.app))
    assert resp.status_code == 200
    assert "X-oMLX-Max-Context-Tokens" not in resp.headers


def test_streaming_chat_completion_carries_the_max_context_header(monkeypatch):
    _chat_env(monkeypatch, hint=65_536)
    client = TestClient(server.app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.headers["X-oMLX-Max-Context-Tokens"] == "65536"
