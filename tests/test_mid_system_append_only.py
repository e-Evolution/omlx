# SPDX-License-Identifier: Apache-2.0
"""Append-only rendering of mid-transcript system updates (Qwen3.8 template).

pi appends ``{role: "system"}`` messages mid-transcript ("Updated system
prompt section ...").  The rendered prompt of turn N must stay a strict prefix
of turn N+1 so the prefix cache is reusable.  The Qwen3.8 template rejects
non-leading system messages, so oMLX must rewrite them; the rewrite must not
mutate anything before the update's own position.
"""

from pathlib import Path

import pytest
from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment

from omlx.api.utils import (
    MID_SYSTEM_NOTE_KEY,
    normalize_mid_system_policy,
    prepare_system_messages_for_template,
)

TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen3_8_chat_template.jinja"
).read_text()

SYSTEM = {"role": "system", "content": "You are a coding agent."}
U1 = {"role": "user", "content": "Refactor the parser module."}
A_TOOL1 = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Need to inspect the file.",
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": {"path": "a.py"}},
        }
    ],
}
TOOL1 = {"role": "tool", "tool_call_id": "c1", "content": "file contents"}
UPDATE = {
    "role": "system",
    "content": "Updated system prompt section 'todo': 2 items stale.",
}
A_TOOL2 = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Now edit it.",
    "tool_calls": [
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "edit_file", "arguments": {"path": "a.py"}},
        }
    ],
}
TOOL2 = {"role": "tool", "tool_call_id": "c2", "content": "ok"}

POLICIES = ["strict", "user_note_safe", "inline_note"]


def _raise(msg):
    raise ValueError(msg)


class _Tokenizer:
    """Minimal tokenizer rendering through the vendored Qwen3.8 template."""

    def __init__(self):
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        env.globals["raise_exception"] = _raise
        import json

        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
        self._tmpl = env.from_string(TEMPLATE)
        self.chat_template = TEMPLATE

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        kwargs.pop("continue_final_message", None)
        kwargs.setdefault("tools", None)
        return self._tmpl.render(messages=messages, **kwargs)


def render(messages, policy, **template_kwargs):
    tok = _Tokenizer()
    prepared = prepare_system_messages_for_template(
        messages,
        tok,
        unsupported_mid_system_policy=policy,
    )
    return tok.apply_chat_template(
        prepared, add_generation_prompt=False, **template_kwargs
    )


TURN_N = [SYSTEM, U1, A_TOOL1, TOOL1]
TURN_N1 = [SYSTEM, U1, A_TOOL1, TOOL1, UPDATE, A_TOOL2, TOOL2]


class TestMidSystemAppendOnly:
    def test_inline_note_update_after_tool_result_is_append_only(self):
        assert render(TURN_N1, "inline_note").startswith(
            render(TURN_N, "inline_note")
        )

    @pytest.mark.parametrize("policy", ["strict", "user_note_safe"])
    def test_legacy_policies_mutate_the_system_head(self, policy):
        # Documents the measured mechanism (plan section 3.3): at a tool-call
        # boundary user_note_safe refuses and falls back to strict, which
        # merges the update into the leading system message, so the rendered
        # prefix changes at the end of the system head.
        n, n1 = render(TURN_N, policy), render(TURN_N1, policy)
        assert not n1.startswith(n)
        assert "Updated system prompt section" not in n
        head_of_n1 = n1[: n1.index("<|im_end|>")]
        assert "Updated system prompt section" in head_of_n1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A2 pending: with preserve_thinking=false the Qwen template anchors "
            "last_query_index on the user-role note and strips earlier thinking"
        ),
    )
    def test_inline_note_without_thinking_retention_is_append_only(self):
        kw = {"preserve_thinking": False}
        assert render(TURN_N1, "inline_note", **kw).startswith(
            render(TURN_N, "inline_note", **kw)
        )

    def test_update_at_tail_is_append_only_for_next_turn(self):
        tail = [SYSTEM, U1, A_TOOL1, TOOL1, UPDATE]
        later = [SYSTEM, U1, A_TOOL1, TOOL1, UPDATE, A_TOOL2, TOOL2]
        assert render(later, "inline_note").startswith(render(tail, "inline_note"))

    def test_second_update_does_not_touch_the_first(self):
        update2 = {"role": "system", "content": "Updated section 'memory'."}
        one = [SYSTEM, U1, A_TOOL1, TOOL1, UPDATE, A_TOOL2, TOOL2]
        two = one + [update2]
        assert render(two, "inline_note").startswith(render(one, "inline_note"))

    def test_adjacent_updates_stay_separate_messages(self):
        update2 = {"role": "system", "content": "Second."}
        prepared = prepare_system_messages_for_template(
            [SYSTEM, U1, A_TOOL1, TOOL1, UPDATE, update2, A_TOOL2, TOOL2],
            _Tokenizer(),
            unsupported_mid_system_policy="inline_note",
        )
        notes = [m for m in prepared if m.get(MID_SYSTEM_NOTE_KEY)]
        assert len(notes) == 2
        assert [m["role"] for m in prepared].count("system") == 1

    def test_head_is_never_mutated(self):
        prepared = prepare_system_messages_for_template(
            TURN_N1, _Tokenizer(), unsupported_mid_system_policy="inline_note"
        )
        assert prepared[0] == SYSTEM

    def test_note_wrapper_marker_and_position(self):
        prepared = prepare_system_messages_for_template(
            TURN_N1, _Tokenizer(), unsupported_mid_system_policy="inline_note"
        )
        idx = [i for i, m in enumerate(prepared) if m.get(MID_SYSTEM_NOTE_KEY)]
        assert idx == [4]
        note = prepared[4]
        assert note["role"] == "user"
        assert note["content"] == (
            "<system-update>\n" + UPDATE["content"] + "\n</system-update>"
        )
        assert prepared[3]["role"] == "tool"
        assert prepared[5] is not None and prepared[5]["role"] == "assistant"

    def test_update_between_users_not_merged_with_neighbour(self):
        msgs = [
            SYSTEM,
            U1,
            {"role": "assistant", "content": "ok", "reasoning_content": "r"},
            UPDATE,
            {"role": "user", "content": "next question"},
        ]
        prepared = prepare_system_messages_for_template(
            msgs, _Tokenizer(), unsupported_mid_system_policy="inline_note"
        )
        assert [m["role"] for m in prepared] == [
            "system",
            "user",
            "assistant",
            "user",
            "user",
        ]
        assert prepared[4]["content"] == "next question"

    def test_multimodal_neighbour_is_left_untouched(self):
        image_user = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
        msgs = [SYSTEM, image_user, A_TOOL1, TOOL1, UPDATE, A_TOOL2, TOOL2]
        prepared = prepare_system_messages_for_template(
            msgs, _Tokenizer(), unsupported_mid_system_policy="inline_note"
        )
        assert prepared[1] == image_user
        assert prepared[1]["content"][1]["type"] == "image_url"
        assert sum(1 for m in prepared if m.get(MID_SYSTEM_NOTE_KEY)) == 1

    @pytest.mark.parametrize("policy", POLICIES)
    def test_no_mid_system_means_identical_output_across_policies(self, policy):
        msgs = [SYSTEM, U1, A_TOOL1, TOOL1, A_TOOL2, TOOL2]
        baseline = prepare_system_messages_for_template(
            msgs, _Tokenizer(), unsupported_mid_system_policy="strict"
        )
        assert (
            prepare_system_messages_for_template(
                msgs, _Tokenizer(), unsupported_mid_system_policy=policy
            )
            == baseline
        )

    def test_partial_request_behaves_as_strict(self):
        strict = prepare_system_messages_for_template(
            TURN_N1,
            _Tokenizer(),
            is_partial=True,
            unsupported_mid_system_policy="strict",
        )
        inline = prepare_system_messages_for_template(
            TURN_N1,
            _Tokenizer(),
            is_partial=True,
            unsupported_mid_system_policy="inline_note",
        )
        assert inline == strict

    def test_unknown_policy_falls_back_to_strict(self):
        strict = prepare_system_messages_for_template(
            TURN_N1, _Tokenizer(), unsupported_mid_system_policy="strict"
        )
        assert (
            prepare_system_messages_for_template(
                TURN_N1, _Tokenizer(), unsupported_mid_system_policy="bogus"
            )
            == strict
        )

    def test_strict_and_user_note_safe_unchanged(self):
        # Pin today's behaviour: both fold the update into the head here.
        for policy in ("strict", "user_note_safe"):
            prepared = prepare_system_messages_for_template(
                TURN_N1, _Tokenizer(), unsupported_mid_system_policy=policy
            )
            assert "Updated system prompt section" in prepared[0]["content"]


class TestMidSystemPolicyNormalization:
    @pytest.mark.parametrize("value", ["strict", "user_note_safe", "inline_note"])
    def test_valid_values(self, value):
        assert normalize_mid_system_policy(value) == value
        assert normalize_mid_system_policy(f"  {value.upper()} ") == value

    @pytest.mark.parametrize("value", [None, "", "bogus", 3, "inline"])
    def test_invalid_values_are_none(self, value):
        assert normalize_mid_system_policy(value) is None
