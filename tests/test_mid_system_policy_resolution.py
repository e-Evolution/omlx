# SPDX-License-Identifier: Apache-2.0
"""Resolution order of the mid-system policy used by the server."""

from types import SimpleNamespace

import pytest

import omlx.server as server


class _State:
    def __init__(self, preserve=True, global_policy=None):
        self.global_settings = SimpleNamespace(
            server=SimpleNamespace(
                preserve_mid_system_cache=preserve,
                mid_system_policy=global_policy,
            )
        )


@pytest.fixture
def patch_state(monkeypatch):
    def apply(state, model_policy=None):
        monkeypatch.setattr(server, "_server_state", state)
        monkeypatch.setattr(
            server,
            "get_model_settings_for_request",
            lambda model_id: SimpleNamespace(mid_system_policy=model_policy)
            if model_id
            else None,
        )

    return apply


def test_legacy_default_is_user_note_safe(patch_state):
    patch_state(_State())
    assert server._unsupported_mid_system_policy() == "user_note_safe"


def test_legacy_flag_off_is_strict(patch_state):
    patch_state(_State(preserve=False))
    assert server._unsupported_mid_system_policy("m") == "strict"


def test_global_policy_beats_legacy_flag(patch_state):
    patch_state(_State(preserve=False, global_policy="inline_note"))
    assert server._unsupported_mid_system_policy("m") == "inline_note"


def test_model_policy_beats_global(patch_state):
    patch_state(_State(global_policy="inline_note"), model_policy="strict")
    assert server._unsupported_mid_system_policy("m") == "strict"


def test_invalid_model_policy_is_ignored(patch_state):
    patch_state(_State(global_policy="inline_note"), model_policy="bogus")
    assert server._unsupported_mid_system_policy("m") == "inline_note"
