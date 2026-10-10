# SPDX-License-Identifier: Apache-2.0
"""Memory scheduling is on only where memory pressure is planned for.

``memory.memory_scheduling`` is ``auto`` (default: on with an explicit custom
ceiling), ``on`` or ``off``. The process memory enforcer resolves it against
the prefill memory guard and the ceiling, logs the result and propagates it to
every scheduler. The admission features that plan against a fixed memory
budget consult it; off, admission behaves as on upstream ``main``.
"""

import logging
from types import SimpleNamespace

import pytest
import test_process_memory_enforcer as enforcer_tests
from test_process_memory_enforcer import _make_enforcer
from test_scheduler_prefill_memory_guard import _make_scheduler

import omlx.server as server
from omlx.process_memory_enforcer import resolve_memory_scheduling
from omlx.settings import GlobalSettings, MemorySettings

GB = 1024**3

# Shared fixtures: an enforcer's engine pool.
mock_engine_pool = enforcer_tests.mock_engine_pool

# --- Resolution -------------------------------------------------------------


def _resolve(mode, *, guard=True, tier="custom", ceiling_gb=48.0):
    return resolve_memory_scheduling(
        mode,
        prefill_memory_guard=guard,
        memory_guard_tier=tier,
        custom_ceiling_bytes=int(ceiling_gb * GB),
    )


def test_auto_is_on_with_an_explicit_ceiling():
    state = _resolve("auto")
    assert state.enabled is True
    assert state.mode == "auto"
    assert state.reason == "explicit ceiling 48.0 GB"
    assert state.warn is False


@pytest.mark.parametrize("tier", ["safe", "balanced", "aggressive"])
def test_auto_is_off_without_an_explicit_ceiling(tier):
    """The custom ceiling only applies under the custom tier: a value left
    over under another tier pins nothing."""
    for ceiling_gb in (0.0, 48.0):
        state = _resolve("auto", tier=tier, ceiling_gb=ceiling_gb)
        assert state.enabled is False
        assert "no explicit memory ceiling" in state.reason
        assert state.warn is False


def test_auto_is_off_with_the_guard_off():
    state = _resolve("auto", guard=False)
    assert state.enabled is False
    assert state.reason == "prefill memory guard is off"
    assert state.warn is False


def test_on_forces_it_on_without_a_ceiling():
    state = _resolve("on", tier="balanced", ceiling_gb=0)
    assert state.enabled is True
    assert state.reason == "memory_scheduling=on"


def test_on_still_needs_the_guard_and_warns():
    state = _resolve("on", guard=False)
    assert state.enabled is False
    assert state.warn is True
    assert "prefill memory guard is off" in state.reason


def test_off_forces_it_off_with_a_ceiling():
    state = _resolve("off")
    assert state.enabled is False
    assert state.reason == "memory_scheduling=off"
    assert state.warn is False


def test_an_unknown_mode_resolves_as_auto():
    assert _resolve("sometimes").mode == "auto"
    assert _resolve(" ON ").mode == "on"


# --- Settings ----------------------------------------------------------------


def test_the_setting_defaults_to_auto_and_round_trips():
    assert MemorySettings().memory_scheduling == "auto"
    assert MemorySettings().to_dict()["memory_scheduling"] == "auto"
    assert MemorySettings.from_dict({"memory_scheduling": "On"}).memory_scheduling == (
        "on"
    )
    assert MemorySettings.from_dict(
        {"memory_scheduling": "bogus"}
    ).memory_scheduling == ("auto")
    assert MemorySettings.from_dict({}).memory_scheduling == "auto"


@pytest.mark.parametrize(
    "raw,expected", [("on", "on"), ("OFF", "off"), ("auto", "auto")]
)
def test_the_env_override_sets_the_mode(tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv("OMLX_MEMORY_SCHEDULING", raw)
    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.memory.memory_scheduling == expected


def test_an_invalid_env_value_is_ignored_with_a_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OMLX_MEMORY_SCHEDULING", "maybe")
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.memory.memory_scheduling == "auto"
    assert "Invalid OMLX_MEMORY_SCHEDULING value: maybe" in caplog.text


def test_validate_rejects_an_unknown_mode(tmp_path):
    settings = GlobalSettings.load(base_path=tmp_path)
    settings.memory.memory_scheduling = "sometimes"
    assert any("memory_scheduling" in e for e in settings.validate())
    settings.memory.memory_scheduling = "off"
    assert not any("memory_scheduling" in e for e in settings.validate())


# --- Enforcer: resolve at startup and on live changes, then propagate --------


def test_the_enforcer_resolves_and_logs_at_startup(
    mock_engine_pool,
    caplog,
):
    with caplog.at_level(logging.INFO, logger="omlx.process_memory_enforcer"):
        enforcer = _make_enforcer(
            mock_engine_pool,
            tier="custom",
            memory_guard_custom_ceiling_gb=48.0,
        )
    assert enforcer.memory_scheduling == "auto"
    assert enforcer.memory_scheduling_enabled is True
    assert "Memory scheduling: on (explicit ceiling 48.0 GB)" in caplog.text


def test_the_enforcer_defaults_to_off_on_a_tier(mock_engine_pool, caplog):
    with caplog.at_level(logging.INFO, logger="omlx.process_memory_enforcer"):
        enforcer = _make_enforcer(mock_engine_pool)
    assert enforcer.memory_scheduling_enabled is False
    assert "Memory scheduling: off (no explicit memory ceiling" in caplog.text


def test_on_with_the_guard_off_logs_a_warning(mock_engine_pool, caplog):
    with caplog.at_level(logging.WARNING, logger="omlx.process_memory_enforcer"):
        enforcer = _make_enforcer(
            mock_engine_pool, memory_scheduling="on", prefill_memory_guard=False
        )
    assert enforcer.memory_scheduling_enabled is False
    assert any(
        r.levelno == logging.WARNING and "Memory scheduling: off" in r.getMessage()
        for r in caplog.records
    )


def test_live_changes_re_resolve_and_log(mock_engine_pool, caplog):
    enforcer = _make_enforcer(
        mock_engine_pool, tier="custom", memory_guard_custom_ceiling_gb=48.0
    )
    caplog.clear()  # only the lines logged by the live changes below
    with caplog.at_level(logging.INFO, logger="omlx.process_memory_enforcer"):
        enforcer.memory_guard_tier = "balanced"
        assert enforcer.memory_scheduling_enabled is False
        enforcer.memory_scheduling = "on"
        assert enforcer.memory_scheduling_enabled is True
        enforcer.prefill_memory_guard = False
        assert enforcer.memory_scheduling_enabled is False
        enforcer.prefill_memory_guard = True
        enforcer.memory_scheduling = "auto"
        enforcer.memory_guard_tier = "custom"
        enforcer.memory_guard_custom_ceiling_bytes = 40 * GB
    assert enforcer.memory_scheduling_enabled is True
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Memory scheduling:")
    ]
    assert messages[0].startswith("Memory scheduling: off (no explicit memory ceiling")
    assert messages[1] == "Memory scheduling: on (memory_scheduling=on)"
    assert messages[-1] == "Memory scheduling: on (explicit ceiling 40.0 GB)"


def test_the_enforcer_propagates_the_state_to_schedulers(
    mock_engine_pool,
):
    from unittest.mock import MagicMock

    scheduler = _make_scheduler()
    assert scheduler._memory_scheduling is False  # upstream until told
    engine = MagicMock(spec=[])
    engine.scheduler = scheduler
    mock_engine_pool._entries = {"m": SimpleNamespace(engine=engine)}
    enforcer = _make_enforcer(
        mock_engine_pool, tier="custom", memory_guard_custom_ceiling_gb=48.0
    )
    enforcer._propagate_memory_limit()
    assert scheduler._memory_scheduling is True
    enforcer.memory_scheduling = "off"
    enforcer._propagate_memory_limit()
    assert scheduler._memory_scheduling is False


# --- Admin settings -----------------------------------------------------------


def test_admin_saves_the_mode_and_applies_it_live(
    mock_engine_pool,
    monkeypatch,
):
    import asyncio

    from fastapi import HTTPException
    from test_admin_server_aliases import (
        _make_global_settings,
        _patched_global_settings,
    )

    import omlx.admin.routes as admin_routes
    from omlx.admin.routes import GlobalSettingsRequest

    enforcer = _make_enforcer(mock_engine_pool)
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", enforcer)
    gs = _make_global_settings()
    gs.memory.memory_scheduling = "auto"
    with _patched_global_settings(gs):
        result = asyncio.run(
            admin_routes.update_global_settings(
                request=GlobalSettingsRequest(memory_scheduling="ON"), is_admin=True
            )
        )
        assert "memory_scheduling" in result["runtime_applied"]
        assert gs.memory.memory_scheduling == "on"
        assert enforcer.memory_scheduling == "on"
        assert enforcer.memory_scheduling_enabled is True
        with pytest.raises(HTTPException) as info:
            asyncio.run(
                admin_routes.update_global_settings(
                    request=GlobalSettingsRequest(memory_scheduling="maybe"),
                    is_admin=True,
                )
            )
    assert info.value.status_code == 400
    assert enforcer.memory_scheduling == "on"


def test_admin_lists_the_mode_with_the_memory_settings(tmp_path):
    import asyncio

    from test_admin_server_aliases import _patched_global_settings

    import omlx.admin.routes as admin_routes

    gs = GlobalSettings.load(base_path=tmp_path)
    gs.memory.memory_scheduling = "off"
    with _patched_global_settings(gs):
        payload = asyncio.run(admin_routes.get_global_settings(is_admin=True))
    assert payload["memory"]["memory_scheduling"] == "off"
