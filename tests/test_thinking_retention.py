# SPDX-License-Identifier: Apache-2.0
"""Deterministic thinking retention (plan A2) on the Qwen3.8 template.

The Qwen3.8 template strips ``<think>`` from every assistant turn at or before
``last_query_index`` (the last user message not wrapped in ``<tool_response>``)
unless ``preserve_thinking`` is on.  pi appends a synthetic user message after
tool results that contain images, and oMLX renders mid-transcript system
updates as user notes, so each of those strips the agent loop's thinking and
breaks the prefix cache.  ``apply_thinking_retention`` classifies those
messages as non-query and replays a deterministic strip boundary instead.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

import omlx.server as server
from omlx.api.utils import (
    MID_SYSTEM_NOTE_KEY,
    prepare_system_messages_for_template,
)
from omlx.api.thinking_retention import (
    IMAGE_MARKER_TEXT,
    apply_thinking_retention,
    is_non_query_user_message,
    normalize_guard_tokens,
    replay_strip_boundaries,
)

TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen3_8_chat_template.jinja"
).read_text()

GUARD = 20000


def _raise(msg):
    raise ValueError(msg)


class _Tokenizer:
    """Minimal tokenizer rendering through the vendored Qwen3.8 template."""

    def __init__(self):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
        self._tmpl = env.from_string(TEMPLATE)
        self.chat_template = TEMPLATE

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        kwargs.pop("continue_final_message", None)
        kwargs.setdefault("tools", None)
        return self._tmpl.render(messages=messages, **kwargs)


def count_words(text):
    return len(text.split())


def render_today(messages, policy="inline_note"):
    """What the server renders today (preserve_thinking off, no retention)."""
    tok = _Tokenizer()
    prepared = prepare_system_messages_for_template(
        messages, tok, unsupported_mid_system_policy=policy
    )
    return tok.apply_chat_template(
        prepared, add_generation_prompt=False, preserve_thinking=False
    )


def render(messages, guard=GUARD, policy="inline_note", count=count_words):
    """Render the way the server does with the module wired in."""
    tok = _Tokenizer()
    prepared = prepare_system_messages_for_template(
        messages, tok, unsupported_mid_system_policy=policy
    )
    result = apply_thinking_retention(prepared, count, guard)
    kwargs = {"preserve_thinking": True if result.took_control else False}
    return tok.apply_chat_template(
        result.messages, add_generation_prompt=False, **kwargs
    )


SYSTEM = {"role": "system", "content": "You are a coding agent."}
U1 = {"role": "user", "content": "Refactor the parser module."}
U2 = {"role": "user", "content": "Now add tests."}
UPDATE = {
    "role": "system",
    "content": "Updated system prompt section 'todo': 2 items stale.",
}
IMG_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
MARK_TEXT = {"role": "user", "content": IMAGE_MARKER_TEXT}
MARK_VLM = {
    "role": "user",
    "content": [{"type": "text", "text": IMAGE_MARKER_TEXT}, IMG_PART],
}


def asst(idx, reasoning, content=""):
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": f"c{idx}",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": f"{idx}.py"}},
            }
        ],
    }


def tool(idx):
    return {"role": "tool", "tool_call_id": f"c{idx}", "content": f"result {idx}"}


def final(reasoning, content="All done."):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning}


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Key RED: the image marker / the system-update note must not break the prefix.
# ---------------------------------------------------------------------------

TURN_N = [SYSTEM, U1, asst(1, "think-one"), tool(1)]


@pytest.mark.parametrize("marker", [MARK_TEXT, MARK_VLM], ids=["text", "vlm"])
def test_image_marker_keeps_prefix_append_only(marker):
    turn_n1 = TURN_N + [
        marker,
        asst(2, "think-two"),
        tool(2),
        marker,
        final("think-three"),
    ]
    assert render(turn_n1).startswith(render(TURN_N))


def test_image_marker_breaks_prefix_without_the_module():
    # Documents the measured mechanism (plan 3.2): today each marker strips the
    # thinking of the loop, so the previous prompt is no longer a prefix.
    turn_n1 = TURN_N + [MARK_TEXT, asst(2, "think-two"), tool(2)]
    assert not render_today(turn_n1).startswith(render_today(TURN_N))
    assert render(turn_n1, guard=None) == render_today(turn_n1)


def test_system_update_note_keeps_prefix_append_only():
    turn_n1 = TURN_N + [UPDATE, asst(2, "think-two"), tool(2)]
    assert not render_today(turn_n1).startswith(render_today(TURN_N))
    assert render(turn_n1).startswith(render(TURN_N))


def test_marker_and_note_together_keep_prefix_append_only():
    turn_n1 = TURN_N + [MARK_VLM, UPDATE, asst(2, "think-two"), tool(2)]
    assert render(turn_n1).startswith(render(TURN_N))


def test_thinking_text_survives_after_marker():
    out = render(TURN_N + [MARK_TEXT, asst(2, "think-two"), tool(2)])
    assert "think-one" in out and "think-two" in out


# ---------------------------------------------------------------------------
# Real user messages behave exactly as today.
# ---------------------------------------------------------------------------

LOOP = [SYSTEM, U1, asst(1, "think-one"), tool(1), final("think-two", "first answer")]


def test_conversation_without_markers_is_byte_identical_to_today():
    convo = LOOP + [U2, asst(3, "think-three"), tool(3)]
    assert render(convo) == render_today(convo)
    assert not apply_thinking_retention(convo, count_words, GUARD).took_control


def test_real_user_message_strips_thinking_before_it():
    convo = LOOP + [U2, asst(3, "think-three"), tool(3), MARK_TEXT]
    out = render(convo)
    assert "think-one" not in out and "think-two" not in out
    assert "think-three" in out
    # The segment from the real user message on matches today's rendering of
    # the same conversation (the marker itself is the only addition).
    header = "<|im_start|>user\nNow add tests."
    today = render_today(LOOP + [U2, asst(3, "think-three"), tool(3)])
    assert out.split(header)[1].startswith(today.split(header)[1])


def test_stripped_turn_exact_rendered_form():
    # Pins the shape of a stripped turn. The template cannot emit its bare
    # stripped form (no <think> block) while preserve_thinking is on, so a
    # stripped turn carries an empty think block. Any change here must be
    # deliberate (plan A2 risk: unmeasured quality impact of the empty block).
    convo = LOOP + [U2, asst(3, "think-three"), tool(3), MARK_TEXT]
    out = render(convo)
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\nfirst answer<|im_end|>\n"
        in out
    )
    assert out.count("<think>\n\n</think>\n\n") == 2  # asst(1) + final()
    today = render_today(convo[:5] + [U2])
    assert "<|im_start|>assistant\nfirst answer<|im_end|>\n" in today


# ---------------------------------------------------------------------------
# Guard.
# ---------------------------------------------------------------------------


def loop_history(n_turns, w=10):
    msgs = [SYSTEM, U1]
    for i in range(1, n_turns + 1):
        msgs += [asst(i, words(f"t{i}_", w)), tool(i)]
    return msgs


GUARD_N = 55  # 10-word turns: the 6th turn crosses it (60 > 55)


def test_guard_not_crossed_means_no_strip_and_no_control():
    hist = loop_history(5)
    res = apply_thinking_retention(hist, count_words, GUARD_N)
    assert not res.took_control
    assert res.messages == hist
    assert render(hist, guard=GUARD_N) == render_today(hist)


def test_guard_crossing_strips_once_then_append_only_again():
    renders = {k: render(loop_history(k), guard=GUARD_N) for k in range(1, 12)}
    # Before the crossing: append-only.
    for k in range(2, 6):
        assert renders[k].startswith(renders[k - 1])
    # The 6th turn crosses: turns 1..5 are stripped, turn 6 is kept.
    assert not renders[6].startswith(renders[5])
    assert "t1_0" not in renders[6] and "t5_0" not in renders[6]
    assert "t6_0" in renders[6]
    # Then append-only again for the next turns (R restarts at 10).
    for k in range(7, 10):
        assert renders[k].startswith(renders[k - 1])
    assert "t6_0" in renders[9] and "t1_0" not in renders[9]


def test_guard_strip_is_exactly_one_boundary_move():
    hist = loop_history(9)
    boundaries = replay_strip_boundaries(hist, count_words, GUARD_N)
    assert len(set(boundaries[1:])) == 2  # initial (real user) + one guard strip
    assert boundaries[-1] == hist.index(asst(6, words("t6_", 10)))


def test_retained_thinking_never_exceeds_guard_by_more_than_one_turn():
    hist = loop_history(40)
    for k in range(2, 41):
        res = apply_thinking_retention(hist[: 2 + 2 * k], count_words, GUARD_N)
        retained = sum(
            count_words(m.get("reasoning_content") or "")
            for m in res.messages
            if m["role"] == "assistant"
        )
        assert retained <= GUARD_N + 10


def test_boundary_resets_on_real_user_message():
    hist = loop_history(7) + [U2, asst(8, words("t8_", 10)), tool(8)]
    out = render(hist, guard=GUARD_N)
    assert "t6_0" not in out and "t7_0" not in out  # before the real user
    assert "t8_0" in out
    more = hist + [asst(9, words("t9_", 10)), tool(9)]
    assert render(more, guard=GUARD_N).startswith(out)


def test_guard_with_marker_in_history_still_strips_at_crossing():
    hist = [SYSTEM, U1]
    for i in range(1, 9):
        hist += [asst(i, words(f"t{i}_", 10)), tool(i), MARK_TEXT]
    out = render(hist, guard=GUARD_N)
    assert "t1_0" not in out and "t8_0" in out
    assert render(hist + [asst(9, "x y z"), tool(9), MARK_TEXT], guard=GUARD_N).startswith(
        out
    )


# ---------------------------------------------------------------------------
# Near-miss markers are real user messages.
# ---------------------------------------------------------------------------


def _seq(user_msg, before=None):
    before = before if before is not None else [asst(1, "r"), tool(1)]
    return [SYSTEM, U1, *before, user_msg]


@pytest.mark.parametrize(
    "msgs",
    [
        _seq(MARK_TEXT),
        _seq(MARK_VLM),
        [SYSTEM, U1, asst(1, "r"), tool(1), tool(2), MARK_VLM],
        _seq({"role": "user", "content": [{"type": "text", "text": IMAGE_MARKER_TEXT}, IMG_PART, IMG_PART]}),
        _seq({"role": "user", "content": [IMG_PART, {"type": "text", "text": IMAGE_MARKER_TEXT}]}),
        [SYSTEM, U1, asst(1, "r"), tool(1), {"role": "user", "content": "x", MID_SYSTEM_NOTE_KEY: True}],
    ],
)
def test_genuine_non_query_messages(msgs):
    assert is_non_query_user_message(msgs, len(msgs) - 1)


@pytest.mark.parametrize(
    "msgs",
    [
        _seq({"role": "user", "content": IMAGE_MARKER_TEXT + " Please compare."}),
        _seq({"role": "user", "content": "Attached image from tool result:"}),
        _seq({"role": "user", "content": IMAGE_MARKER_TEXT.lower()}),
        _seq(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_MARKER_TEXT},
                    {"type": "text", "text": "also this"},
                    IMG_PART,
                ],
            }
        ),
        _seq(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_MARKER_TEXT + " extra"},
                    IMG_PART,
                ],
            }
        ),
        # not directly after a tool message
        [SYSTEM, MARK_TEXT],
        [SYSTEM, U1, MARK_VLM],
        [SYSTEM, U1, asst(1, "r"), MARK_TEXT],
        # marker text with no image in list form
        _seq({"role": "user", "content": [{"type": "text", "text": IMAGE_MARKER_TEXT}]}),
    ],
)
def test_near_miss_markers_are_real_user_messages(msgs):
    assert not is_non_query_user_message(msgs, len(msgs) - 1)


def test_near_miss_marker_moves_the_boundary_like_a_real_user():
    near = {"role": "user", "content": IMAGE_MARKER_TEXT + " Please compare."}
    convo = TURN_N + [near, asst(2, "think-two"), tool(2)]
    assert render(convo) == render_today(convo)  # module inert
    assert "think-one" not in render(convo)


def test_real_user_query_text_is_never_classified_non_query():
    assert not is_non_query_user_message([SYSTEM, U1], 1)
    assert not is_non_query_user_message(TURN_N + [U2], len(TURN_N))


# ---------------------------------------------------------------------------
# Disabled, inert and determinism.
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    TURN_N,
    TURN_N + [MARK_TEXT, asst(2, "think-two"), tool(2)],
    TURN_N + [MARK_VLM, UPDATE, asst(2, "think-two"), tool(2), MARK_VLM],
    LOOP + [U2, asst(3, "think-three"), tool(3), MARK_TEXT],
    loop_history(12),
]


@pytest.mark.parametrize("convo", CONVERSATIONS)
def test_guard_none_is_byte_identical_to_today(convo):
    res = apply_thinking_retention(convo, count_words, None)
    assert not res.took_control
    assert res.messages is convo or res.messages == convo
    assert render(convo, guard=None) == render_today(convo)


def test_input_messages_are_never_mutated():
    convo = TURN_N + [MARK_TEXT, asst(2, "think-two"), tool(2)]
    snapshot = json.dumps(convo, sort_keys=True)
    apply_thinking_retention(convo, count_words, GUARD)
    apply_thinking_retention(loop_history(12), count_words, GUARD_N)
    assert json.dumps(convo, sort_keys=True) == snapshot


def test_same_input_same_output():
    convo = loop_history(12)
    a = apply_thinking_retention(convo, count_words, GUARD_N)
    b = apply_thinking_retention(convo, count_words, GUARD_N)
    assert a.messages == b.messages and a.took_control == b.took_control


def test_boundary_decisions_depend_only_on_the_history_prefix():
    full = loop_history(14)
    full[8:8] = []  # keep as is
    for k in range(2, len(full)):
        short = replay_strip_boundaries(full[:k], count_words, GUARD_N)
        longer = replay_strip_boundaries(full[: k + 5], count_words, GUARD_N)
        assert longer[:k] == short


def test_no_reasoning_anywhere_is_inert_even_with_notes_and_markers():
    plain = {"role": "assistant", "content": "ok"}
    convo = [SYSTEM, U1, plain, UPDATE, MARK_TEXT]
    assert not apply_thinking_retention(convo, count_words, GUARD).took_control


def test_missing_or_non_string_reasoning_is_tolerated():
    convo = [
        SYSTEM,
        U1,
        {"role": "assistant", "content": "x", "reasoning_content": None},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": None},
    ]
    assert not apply_thinking_retention(convo, count_words, GUARD).took_control


# ---------------------------------------------------------------------------
# Settings and env parsing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (20000, 20000),
        (1, 1),
        ("20000", 20000),
        (" 15 ", 15),
        (2000.0, 2000),
        (None, None),
        ("", None),
        (0, None),
        (-1, None),
        ("-5", None),
        (True, None),
        (False, None),
        ("true", None),
        (1.5, None),
        (math.inf, None),
        (math.nan, None),
        ("nan", None),
        ("inf", None),
        ("1e3", None),
        ([], None),
    ],
)
def test_normalize_guard_tokens(value, expected):
    assert normalize_guard_tokens(value) == expected


def test_model_settings_field_default_and_round_trip():
    from omlx.model_settings import ModelSettings

    assert ModelSettings().thinking_retention_guard_tokens is None
    assert "thinking_retention_guard_tokens" not in ModelSettings().to_dict()
    ms = ModelSettings.from_dict({"thinking_retention_guard_tokens": 20000})
    assert ms.to_dict()["thinking_retention_guard_tokens"] == 20000


def test_field_is_a_universal_profile_field():
    from omlx.model_profiles import UNIVERSAL_PROFILE_FIELDS

    assert "thinking_retention_guard_tokens" in UNIVERSAL_PROFILE_FIELDS


def test_env_override_is_applied_and_not_persisted(tmp_path):
    from omlx.settings import GlobalSettings

    with patch.dict(
        "os.environ", {"OMLX_THINKING_RETENTION_GUARD_TOKENS": " 20000 "}
    ):
        settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.server.thinking_retention_guard_tokens == 20000
    assert "thinking_retention_guard_tokens" not in settings.server.to_dict()


@pytest.mark.parametrize("raw", ["bogus", "0", "-3", "1.5", "nan", "true"])
def test_env_override_rejects_invalid_values(tmp_path, raw):
    from omlx.settings import GlobalSettings

    with patch.dict("os.environ", {"OMLX_THINKING_RETENTION_GUARD_TOKENS": raw}):
        settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.server.thinking_retention_guard_tokens is None


def test_unset_env_leaves_guard_off(tmp_path):
    from omlx.settings import GlobalSettings

    settings = GlobalSettings.load(base_path=tmp_path)
    assert settings.server.thinking_retention_guard_tokens is None


# ---------------------------------------------------------------------------
# Server wiring.
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self):
        self.tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())


@pytest.fixture
def server_state(monkeypatch):
    def apply(global_guard=None, model_guard=None, preserve_default=True):
        monkeypatch.setattr(
            server,
            "_server_state",
            SimpleNamespace(
                global_settings=SimpleNamespace(
                    server=SimpleNamespace(thinking_retention_guard_tokens=global_guard)
                )
            ),
        )
        monkeypatch.setattr(
            server,
            "get_model_settings_for_request",
            lambda model_id: SimpleNamespace(
                thinking_retention_guard_tokens=model_guard
            ),
        )
        entry = SimpleNamespace(preserve_thinking_default=preserve_default)
        monkeypatch.setattr(
            server,
            "get_engine_pool",
            lambda: SimpleNamespace(get_entry=lambda model_id: entry),
        )

    return apply


def test_guard_resolution_model_beats_global(server_state):
    server_state(global_guard=20000, model_guard=500)
    assert server._thinking_retention_guard_tokens("m") == 500
    server_state(global_guard=20000, model_guard=None)
    assert server._thinking_retention_guard_tokens("m") == 20000
    server_state(global_guard=None, model_guard=None)
    assert server._thinking_retention_guard_tokens("m") is None
    server_state(global_guard=20000, model_guard=-4)  # invalid -> ignored
    assert server._thinking_retention_guard_tokens("m") == 20000


NOTE_CONVO = TURN_N + [MARK_TEXT, asst(2, "think-two"), tool(2)]


def test_server_override_replaces_forced_preserve_thinking_false(server_state):
    server_state(global_guard=GUARD)
    msgs, kwargs, took = server._apply_thinking_retention(
        NOTE_CONVO,
        _Engine(),
        "m",
        {"preserve_thinking": False, "reasoning_effort": "high"},
    )
    assert took is True
    assert kwargs == {"preserve_thinking": True, "reasoning_effort": "high"}


def test_server_override_does_not_mutate_the_input_kwargs(server_state):
    server_state(global_guard=GUARD)
    forced = {"preserve_thinking": False}
    server._apply_thinking_retention(NOTE_CONVO, _Engine(), "m", forced)
    assert forced == {"preserve_thinking": False}


def test_server_leaves_kwargs_alone_when_module_off_or_inert(server_state):
    server_state(global_guard=None)
    forced = {"preserve_thinking": False}
    assert server._apply_thinking_retention(NOTE_CONVO, _Engine(), "m", forced) == (
        NOTE_CONVO,
        forced,
        False,
    )
    server_state(global_guard=GUARD)
    assert server._apply_thinking_retention(TURN_N, _Engine(), "m", None) == (
        TURN_N,
        None,
        False,
    )


def test_server_skips_templates_without_preserve_thinking_support(server_state):
    server_state(global_guard=GUARD, preserve_default=False)
    forced = {"preserve_thinking": False}
    assert server._apply_thinking_retention(NOTE_CONVO, _Engine(), "m", forced)[2] is False


def test_server_skips_when_thinking_is_disabled(server_state):
    server_state(global_guard=GUARD)
    kwargs = {"enable_thinking": False}
    assert server._apply_thinking_retention(NOTE_CONVO, _Engine(), "m", kwargs)[2] is False


def test_server_override_feeds_reasoning_output_caching(server_state):
    from omlx.api.utils import cache_reasoning_output

    server_state(global_guard=GUARD)
    _, kwargs, took = server._apply_thinking_retention(
        NOTE_CONVO, _Engine(), "m", {"preserve_thinking": False}
    )
    assert took
    # The override happens before cache_reasoning_output(): the engine caches
    # reasoning output because the next prompt really keeps it.
    assert cache_reasoning_output(
        None, native_reasoning=True, chat_template_kwargs=kwargs
    )


def test_server_counts_tokens_with_the_engine_tokenizer(server_state):
    server_state(global_guard=3)
    convo = [
        SYSTEM,
        U1,
        asst(1, "a b c"),
        tool(1),
        asst(2, "d e f"),
        tool(2),
    ]
    msgs, kwargs, took = server._apply_thinking_retention(
        convo, _Engine(), "m", {"preserve_thinking": False}
    )
    assert took
    assert msgs[2]["reasoning_content"] == ""
    assert msgs[4]["reasoning_content"] == "d e f"


def test_server_wires_the_module_at_every_template_call_site():
    import inspect

    src = inspect.getsource(server)
    calls = src.count("_apply_thinking_retention(")
    assert calls == 4  # definition + chat completions, Anthropic, Responses
    for fn in (
        server.create_chat_completion,
        server.create_anthropic_message,
        server.create_response,
    ):
        body = inspect.getsource(fn)
        prep = body.index("prepare_system_messages_for_template(")
        apply = body.index("_apply_thinking_retention(")
        assert prep < apply
        # Always after the forced chat_template_kwargs merge.
        assert body.index("merge_chat_template_request_kwargs(") < apply
