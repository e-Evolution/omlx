# SPDX-License-Identifier: Apache-2.0
"""Memory scheduling across admin settings saves, with the fork defaults layer.

The live settings object keeps the ``OMLX_DEFAULTS_FILE`` values through every
admin save; a save leaves the scheduling mode alone unless it changes the mode
(the tier, ceiling and guard re-resolve ``auto`` only); an explicit mode POST
re-resolves, logs and propagates even when the mode is unchanged; and the
admin settings report the live effective state, so it can be checked without
the log.

Only the enforcer that drives the engine pool logs its state. The settings
page's tier previews build one throwaway enforcer per tier with the default
mode; their "Memory scheduling: off" lines used to read as the live enforcer
turning off after every settings page load.
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import test_fork_defaults as fork_tests
import test_process_memory_enforcer as enforcer_tests
from test_admin_server_aliases import _patched_global_settings
from test_process_memory_enforcer import _make_enforcer
from test_scheduler_prefill_memory_guard import _make_scheduler

import omlx.admin.routes as admin_routes
import omlx.server as server
from omlx.admin.routes import GlobalSettingsRequest
from omlx.process_memory_enforcer import preview_tier_ceilings
from omlx.settings import GlobalSettings

clean_env = fork_tests.clean_env
defaults_file = fork_tests.defaults_file
mock_engine_pool = enforcer_tests.mock_engine_pool

ENFORCER_LOGGER = "omlx.process_memory_enforcer"


def _scheduling_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == ENFORCER_LOGGER and r.getMessage().startswith("Memory scheduling:")
    ]


@pytest.fixture
def live(tmp_path, defaults_file, mock_engine_pool, monkeypatch):
    """Real settings over the fork defaults file, a live enforcer built from
    them as the server builds it, and one scheduler it propagates to."""
    base = fork_tests._base(tmp_path, {"memory": {"memory_guard_tier": "aggressive"}})
    settings = GlobalSettings.load(base_path=base)
    scheduler = _make_scheduler()
    engine = MagicMock(spec=[])
    engine.scheduler = scheduler
    mock_engine_pool._entries = {"m": SimpleNamespace(engine=engine)}
    enforcer = _make_enforcer(
        mock_engine_pool,
        tier=settings.memory.memory_guard_tier,
        prefill_memory_guard=settings.memory.prefill_memory_guard,
        memory_scheduling=settings.memory.memory_scheduling,
    )
    enforcer._running = True  # setters propagate only while running
    enforcer._propagate_memory_limit()
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", enforcer)
    monkeypatch.setattr(server._server_state, "engine_pool", None)
    with _patched_global_settings(settings):
        yield SimpleNamespace(
            base=base, settings=settings, enforcer=enforcer, scheduler=scheduler
        )
    enforcer._running = False


def _post(**fields) -> dict:
    return asyncio.run(
        admin_routes.update_global_settings(
            request=GlobalSettingsRequest(**fields), is_admin=True
        )
    )


def _get() -> dict:
    return asyncio.run(admin_routes.get_global_settings(is_admin=True))


def test_the_live_enforcer_starts_on_from_the_fork_defaults(live):
    assert live.settings.memory.memory_scheduling == "on"
    assert live.enforcer.memory_scheduling_enabled is True
    assert live.scheduler._memory_scheduling is True


@pytest.mark.parametrize(
    "fields",
    [
        {"max_concurrent_requests": 7},
        {"chunked_prefill": False},
        {"memory_guard_tier": "balanced"},
    ],
    ids=["max_concurrent_requests", "chunked_prefill", "memory_guard_tier"],
)
def test_an_admin_save_keeps_the_fork_defaults_and_the_mode(live, caplog, fields):
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        _post(**fields)
    for section, key, default, *_ in fork_tests.KEYS:
        assert fork_tests._value(live.settings, section, key) == default, key
    assert live.enforcer.memory_scheduling == "on"
    assert live.enforcer.memory_scheduling_enabled is True
    assert live.scheduler._memory_scheduling is True
    assert _scheduling_lines(caplog) == []
    # The defaults still never reach settings.json, and a restart sees them.
    saved = json.loads((live.base / "settings.json").read_text())
    for section, key, *_ in fork_tests.KEYS:
        assert key not in saved.get(section, {}), f"{section}.{key} persisted"
    again = GlobalSettings.load(base_path=live.base)
    for section, key, default, *_ in fork_tests.KEYS:
        assert fork_tests._value(again, section, key) == default, key


def test_loading_the_settings_page_logs_no_scheduling_change(live, caplog):
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        _get()
    assert _scheduling_lines(caplog) == []


def test_tier_previews_never_log_a_scheduling_state(caplog):
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        previews = preview_tier_ceilings()
    assert set(previews) == {"safe", "balanced", "aggressive", "custom"}
    assert _scheduling_lines(caplog) == []


def test_the_settings_report_the_live_effective_state(live):
    payload = _get()
    assert payload["memory"]["memory_scheduling"] == "on"
    assert payload["memory"]["memory_scheduling_effective"] == {
        "enabled": True,
        "reason": "memory_scheduling=on",
    }
    live.enforcer.prefill_memory_guard = False
    effective = _get()["memory"]["memory_scheduling_effective"]
    assert effective["enabled"] is False
    assert effective["reason"].startswith("memory_scheduling=on needs the prefill")


def test_the_effective_state_is_null_without_a_live_enforcer(live, monkeypatch):
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", None)
    assert _get()["memory"]["memory_scheduling_effective"] is None


def test_the_factory_defaults_carry_no_live_state(live):
    payload = asyncio.run(admin_routes.get_global_settings_defaults(is_admin=True))
    assert "memory_scheduling_effective" not in payload["memory"]


def test_an_explicit_mode_post_re_resolves_logs_and_propagates(live, caplog):
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        result = _post(memory_scheduling="off")
    assert _scheduling_lines(caplog) == [
        "Memory scheduling: off (memory_scheduling=off)"
    ]
    assert live.scheduler._memory_scheduling is False
    assert result["memory_scheduling_effective"] == {
        "enabled": False,
        "reason": "memory_scheduling=off",
    }

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        result = _post(memory_scheduling="on")
    assert _scheduling_lines(caplog) == ["Memory scheduling: on (memory_scheduling=on)"]
    assert live.scheduler._memory_scheduling is True
    assert result["memory_scheduling_effective"]["enabled"] is True


def test_an_explicit_post_of_the_current_mode_still_logs_and_propagates(live, caplog):
    live.scheduler._memory_scheduling = False  # a scheduler that missed a tick
    with caplog.at_level(logging.INFO, logger=ENFORCER_LOGGER):
        _post(memory_scheduling="on")
    assert _scheduling_lines(caplog) == ["Memory scheduling: on (memory_scheduling=on)"]
    assert live.scheduler._memory_scheduling is True


def test_a_save_without_a_live_enforcer_reports_no_effective_state(live, monkeypatch):
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", None)
    result = _post(max_concurrent_requests=3)
    assert result["memory_scheduling_effective"] is None
