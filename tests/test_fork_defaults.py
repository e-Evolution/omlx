# SPDX-License-Identifier: Apache-2.0
"""Fork defaults layer: ``OMLX_DEFAULTS_FILE``.

Precedence: explicit env > settings.json > fork defaults file > built-in
defaults. The defaults never reach settings.json through a save.
"""

import json
import logging
import os
from argparse import Namespace
from unittest.mock import patch

import pytest

import omlx.settings as settings_mod
from omlx.settings import GlobalSettings, scheduler_switch_env

# (section, key, fork default, upstream default, settings.json value,
#  env var, env value, value the env value parses to)
# fmt: off
KEYS = [
    ("scheduler", "prefill_progress_commit_blocks", 4, 0, 2,
     "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS", "6", 6),
    ("scheduler", "memory_pressure_park", True, False, False,
     "OMLX_MEMORY_PRESSURE_PARK", "true", True),
    ("scheduler", "memory_abort_single_victim", True, False, False,
     "OMLX_MEMORY_ABORT_SINGLE_VICTIM", "true", True),
    ("scheduler", "media_text_suffix_chunked", True, False, False,
     "OMLX_MEDIA_TEXT_SUFFIX_CHUNKED", "true", True),
    ("memory", "mlx_cache_limit_gb", 16.0, 0.0, 8.0,
     "OMLX_MLX_CACHE_LIMIT_GB", "24", 24.0),
    ("memory", "memory_scheduling", "on", "auto", "off",
     "OMLX_MEMORY_SCHEDULING", "auto", "auto"),
    ("server", "mid_system_policy", "inline_note", None, "strict",
     "OMLX_MID_SYSTEM_POLICY", "user_note_safe", "user_note_safe"),
    ("server", "thinking_retention_guard_tokens", 20000, None, 5000,
     "OMLX_THINKING_RETENTION_GUARD_TOKENS", "30000", 30000),
]
# fmt: on
KEY_IDS = [f"{section}.{key}" for section, key, *_ in KEYS]
ENV_NAMES = [row[5] for row in KEYS] + ["OMLX_DEFAULTS_FILE"]


def _defaults() -> dict:
    data: dict = {}
    for section, key, default, *_ in KEYS:
        data.setdefault(section, {})[key] = default
    return data


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def defaults_file(tmp_path, clean_env):
    path = tmp_path / "fork-defaults.json"
    path.write_text(json.dumps(_defaults()))
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(path))
    return path


def _base(tmp_path, settings_data=None):
    base = tmp_path / "omlx-home"
    base.mkdir()
    if settings_data is not None:
        (base / "settings.json").write_text(json.dumps(settings_data))
    return base


def _value(settings, section, key):
    return getattr(getattr(settings, section), key)


# ------------------------------------------------------------- precedence


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_fork_default_applies_when_settings_json_lacks_the_key(
    tmp_path, defaults_file, row
):
    section, key, default, *_ = row
    settings = GlobalSettings.load(base_path=_base(tmp_path, {"version": "1.0"}))
    assert _value(settings, section, key) == default


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_fork_default_applies_without_any_settings_json(tmp_path, defaults_file, row):
    section, key, default, *_ = row
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert _value(settings, section, key) == default


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_settings_json_value_wins_over_the_fork_default(tmp_path, defaults_file, row):
    section, key, _default, _upstream, explicit, *_ = row
    base = _base(tmp_path, {section: {key: explicit}})
    settings = GlobalSettings.load(base_path=base)
    assert _value(settings, section, key) == explicit


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_explicit_env_wins_over_settings_json_and_fork_default(
    tmp_path, defaults_file, row
):
    section, key, _default, _upstream, explicit, env_name, env_raw, parsed = row
    base = _base(tmp_path, {section: {key: explicit}})
    with patch.dict(os.environ, {env_name: env_raw}):
        settings = GlobalSettings.load(base_path=base)
    assert _value(settings, section, key) == parsed


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_unset_defaults_file_keeps_upstream_defaults(tmp_path, clean_env, row):
    section, key, _default, upstream, *_ = row
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert _value(settings, section, key) == upstream
    assert settings.fork_defaults_report is None


# ------------------------------------------------------- bad defaults file


def test_missing_defaults_file_warns_and_falls_back(tmp_path, clean_env, caplog):
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(tmp_path / "nope.json"))
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.scheduler.memory_pressure_park is False
    assert "nope.json" in caplog.text
    assert "nope.json" in settings.fork_defaults_report


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"scheduler": 3}'])
def test_malformed_defaults_file_warns_and_falls_back(
    tmp_path, clean_env, caplog, content
):
    path = tmp_path / "bad.json"
    path.write_text(content)
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.scheduler.prefill_progress_commit_blocks == 0
    assert settings.memory.memory_scheduling == "auto"
    assert "bad.json" in caplog.text


def test_unknown_section_or_key_is_skipped_with_a_warning(tmp_path, clean_env, caplog):
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {
                "scheduler": {"memory_pressure_park": True, "no_such_key": 1},
                "no_such_section": {"x": 1},
            }
        )
    )
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.scheduler.memory_pressure_park is True
    assert "scheduler.no_such_key" in caplog.text
    assert "no_such_section" in caplog.text


def test_a_default_rejected_by_its_section_is_skipped_with_a_warning(
    tmp_path, clean_env, caplog
):
    """A well-formed file with one value a section's from_dict rejects skips
    that key only: settings.json and the other defaults still load."""
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {
                "memory": {"soft_threshold": "high", "mlx_cache_limit_gb": 16.0},
                "scheduler": {"memory_pressure_park": True},
            }
        )
    )
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(path))
    base = _base(tmp_path, {"scheduler": {"prefill_progress_commit_blocks": 2}})
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=base)
    assert settings.memory.soft_threshold == 0.85
    assert settings.memory.mlx_cache_limit_gb == 16.0
    assert settings.scheduler.memory_pressure_park is True
    assert settings.scheduler.prefill_progress_commit_blocks == 2
    assert "memory.soft_threshold" in caplog.text
    assert settings.fork_defaults_report == (
        "Fork defaults: 2 applied, 1 rejected (memory.soft_threshold), "
        "0 overridden by settings.json, 0 overridden by env"
    )
    assert settings.fork_defaults_report_level == logging.WARNING
    # The rejected default is never saved either.
    settings.save()
    saved = json.loads((base / "settings.json").read_text())
    assert saved["memory"]["soft_threshold"] == 0.85


def test_a_default_its_section_coerces_is_flagged(tmp_path, clean_env, caplog):
    """from_dict silently coerces some values (a non-boolean switch reads as
    off); the defaults layer names them instead of applying them unflagged."""
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {
                "scheduler": {
                    "memory_pressure_park": "yes",
                    "prefill_progress_commit_blocks": 4,
                }
            }
        )
    )
    clean_env.setenv("OMLX_DEFAULTS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="omlx.settings"):
        settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.scheduler.memory_pressure_park is False
    assert settings.scheduler.prefill_progress_commit_blocks == 4
    assert "scheduler.memory_pressure_park='yes' was applied as False" in caplog.text
    assert "prefill_progress_commit_blocks" not in caplog.text


def test_relative_path_resolves_next_to_the_omlx_package(
    tmp_path, clean_env, monkeypatch
):
    resources = tmp_path / "Resources"
    resources.mkdir()
    (resources / "omlx-fork-defaults.json").write_text(json.dumps(_defaults()))
    monkeypatch.setattr(settings_mod, "_PACKAGE_PARENT", resources)
    clean_env.setenv("OMLX_DEFAULTS_FILE", "omlx-fork-defaults.json")
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.memory.memory_scheduling == "on"


# ------------------------------------------------------------- persistence


def _saved(base):
    return json.loads((base / "settings.json").read_text())


def test_save_never_writes_fork_defaults_into_settings_json(tmp_path, defaults_file):
    base = _base(tmp_path, {"server": {"port": 9000}})
    settings = GlobalSettings.load(base_path=base)
    settings.save()
    saved = _saved(base)
    for section, key, *_ in KEYS:
        assert key not in saved.get(section, {}), f"{section}.{key} persisted"
    assert saved["server"]["port"] == 9000
    # A reload with the same defaults file sees the same values.
    again = GlobalSettings.load(base_path=base)
    for section, key, default, *_ in KEYS:
        assert _value(again, section, key) == default


def test_changed_fork_default_is_persisted_as_a_user_value(tmp_path, defaults_file):
    base = _base(tmp_path)
    settings = GlobalSettings.load(base_path=base)
    settings.memory.memory_scheduling = "auto"  # e.g. from the admin UI
    settings.save()
    assert _saved(base)["memory"]["memory_scheduling"] == "auto"
    assert GlobalSettings.load(base_path=base).memory.memory_scheduling == "auto"


@pytest.mark.parametrize("row", KEYS, ids=KEY_IDS)
def test_settings_json_value_survives_a_save(tmp_path, defaults_file, row):
    section, key, _default, _upstream, explicit, *_ = row
    base = _base(tmp_path, {section: {key: explicit}})
    GlobalSettings.load(base_path=base).save()
    assert _saved(base)[section][key] == explicit


@pytest.mark.parametrize(
    "key,env_name,env_raw",
    [
        ("mid_system_policy", "OMLX_MID_SYSTEM_POLICY", "inline_note"),
        (
            "thinking_retention_guard_tokens",
            "OMLX_THINKING_RETENTION_GUARD_TOKENS",
            "30000",
        ),
    ],
)
def test_runtime_only_env_override_is_never_persisted(
    tmp_path, clean_env, key, env_name, env_raw
):
    base = _base(tmp_path)
    with patch.dict(os.environ, {env_name: env_raw}):
        GlobalSettings.load(base_path=base).save()
    assert key not in _saved(base)["server"]
    explicit = {"mid_system_policy": "strict", "thinking_retention_guard_tokens": 5000}[
        key
    ]
    (base / "settings.json").write_text(json.dumps({"server": {key: explicit}}))
    with patch.dict(os.environ, {env_name: env_raw}):
        GlobalSettings.load(base_path=base).save()
    assert _saved(base)["server"][key] == explicit


def test_save_cli_overrides_never_writes_fork_defaults(tmp_path, defaults_file):
    base = _base(tmp_path, {"server": {"port": 9000}})
    settings = GlobalSettings.load(base_path=base)
    settings.save_cli_overrides(Namespace(port=9100))
    saved = _saved(base)
    assert saved["server"]["port"] == 9100
    for section, key, *_ in KEYS:
        assert key not in saved.get(section, {}), f"{section}.{key} persisted"


# ------------------------------------------------------------------ report


def test_report_counts_applied_and_overridden_defaults(tmp_path, defaults_file):
    base = _base(tmp_path, {"memory": {"memory_scheduling": "auto"}})
    with patch.dict(os.environ, {"OMLX_MLX_CACHE_LIMIT_GB": "24"}):
        settings = GlobalSettings.load(base_path=base)
    assert settings.fork_defaults_report == (
        "Fork defaults: 7 applied, 1 overridden by settings.json "
        "(memory.memory_scheduling), 1 overridden by env (memory.mlx_cache_limit_gb)"
    )


def test_report_with_nothing_overridden(tmp_path, defaults_file):
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert settings.fork_defaults_report == (
        "Fork defaults: 8 applied, 0 overridden by settings.json, "
        "0 overridden by env"
    )


# --------------------------------------------- env-only switches as settings


def test_switch_env_exports_enabled_scheduler_switches(tmp_path, defaults_file):
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert scheduler_switch_env(settings.scheduler) == {
        "OMLX_MEMORY_ABORT_SINGLE_VICTIM": "true",
        "OMLX_MEDIA_TEXT_SUFFIX_CHUNKED": "true",
    }


def test_switch_env_is_empty_with_upstream_defaults(tmp_path, clean_env):
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert scheduler_switch_env(settings.scheduler) == {}


def test_switch_env_leaves_runtime_readers_off_when_disabled(tmp_path, defaults_file):
    base = _base(tmp_path, {"scheduler": {"memory_abort_single_victim": False}})
    settings = GlobalSettings.load(base_path=base)
    assert "OMLX_MEMORY_ABORT_SINGLE_VICTIM" not in scheduler_switch_env(
        settings.scheduler
    )


# ------------------------------------------------------ shared truthy set

SWITCH_ENV = [
    ("memory_pressure_park", "OMLX_MEMORY_PRESSURE_PARK"),
    ("memory_abort_single_victim", "OMLX_MEMORY_ABORT_SINGLE_VICTIM"),
    ("media_text_suffix_chunked", "OMLX_MEDIA_TEXT_SUFFIX_CHUNKED"),
]


def _runtime_readers():
    from omlx.process_memory_enforcer import ProcessMemoryEnforcer
    from omlx.scheduler import _media_text_suffix_chunked_enabled

    return {
        "OMLX_MEMORY_ABORT_SINGLE_VICTIM": (
            ProcessMemoryEnforcer._abort_single_victim_enabled
        ),
        "OMLX_MEDIA_TEXT_SUFFIX_CHUNKED": _media_text_suffix_chunked_enabled,
    }


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("on", True),
        (" On ", True),
        ("yes", True),
        ("1", True),
        ("TRUE", True),
        ("off", False),
        ("0", False),
        ("no", False),
    ],
)
@pytest.mark.parametrize("attr, env_name", SWITCH_ENV, ids=[e for _a, e in SWITCH_ENV])
def test_every_scheduler_switch_env_reads_the_same_truthy_set(
    tmp_path, clean_env, attr, env_name, raw, expected
):
    clean_env.setenv(env_name, raw)
    settings = GlobalSettings.load(base_path=_base(tmp_path))
    assert getattr(settings.scheduler, attr) is expected
    reader = _runtime_readers().get(env_name)
    if reader is not None:
        assert reader() is expected


def test_the_truthy_set_is_shared():
    from omlx.utils.env_flags import TRUTHY_ENV_VALUES

    assert frozenset({"1", "true", "yes", "on"}) == TRUTHY_ENV_VALUES
