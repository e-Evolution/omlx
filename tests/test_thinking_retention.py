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
    # definition + chat completions, Anthropic, Responses, /tokenize
    assert calls >= 5
    for fn in (
        server.create_chat_completion,
        server.create_anthropic_message,
        server.create_response,
        server._tokenize_chat_messages,
    ):
        body = inspect.getsource(fn)
        prep = body.index("prepare_system_messages_for_template(")
        apply = body.index("_apply_thinking_retention(")
        assert prep < apply
        # Always after the forced chat_template_kwargs merge (inline, or
        # through upstream's _merge_chat_template_kwargs helper).
        merge = [
            body.find(name)
            for name in (
                "merge_chat_template_request_kwargs(",
                "_merge_chat_template_kwargs(",
            )
        ]
        assert min(i for i in merge if i >= 0) < apply


# ---------------------------------------------------------------------------
# Graceful degradation: retention must never fail a request.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_retention_warnings():
    warned = getattr(server, "_retention_warned_reasons", set())
    warned.clear()
    yield
    warned.clear()


def _degraded(engine, server_state, caplog):
    server_state(global_guard=GUARD)
    with caplog.at_level("WARNING"):
        out = server._apply_thinking_retention(
            NOTE_CONVO, engine, "m", {"preserve_thinking": False}
        )
    assert out == (NOTE_CONVO, {"preserve_thinking": False}, False)
    assert out[0] is NOTE_CONVO
    return [r for r in caplog.records if "Thinking retention skipped" in r.message]


def test_processor_without_encode_degrades_with_a_warning(server_state, caplog):
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(), processor=SimpleNamespace()
    )
    assert len(_degraded(engine, server_state, caplog)) == 1


@pytest.mark.parametrize("exc", [ValueError("bad"), RuntimeError("boom")])
def test_encode_raising_degrades(server_state, caplog, exc):
    def encode(text, **kw):
        raise exc

    engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=encode))
    assert len(_degraded(engine, server_state, caplog)) == 1


@pytest.mark.parametrize("bad", [None, "abc", 7])
def test_encode_returning_non_sequence_degrades(server_state, caplog, bad):
    engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t, **kw: bad))
    assert len(_degraded(engine, server_state, caplog)) == 1


def test_degradation_warns_once_per_process_and_reason(server_state, caplog):
    engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t, **kw: None))
    assert len(_degraded(engine, server_state, caplog)) == 1
    with caplog.at_level("WARNING"):
        server._apply_thinking_retention(NOTE_CONVO, engine, "m", None)
    assert len([r for r in caplog.records if "retention skipped" in r.message]) == 1


def test_processor_tokenizer_is_used_when_engine_tokenizer_cannot_encode(server_state):
    server_state(global_guard=GUARD)
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(),
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda t, **kw: t.split())
        ),
    )
    _, kwargs, took = server._apply_thinking_retention(
        NOTE_CONVO, engine, "m", {"preserve_thinking": False}
    )
    assert took and kwargs == {"preserve_thinking": True}


def test_encode_without_add_special_tokens_kwarg_still_counts(server_state):
    server_state(global_guard=GUARD)
    engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t: t.split()))
    assert server._apply_thinking_retention(NOTE_CONVO, engine, "m", None)[2] is True


# ---------------------------------------------------------------------------
# Only the tail after the last real user message is tokenized.
# ---------------------------------------------------------------------------


def test_turns_before_last_real_user_message_are_never_counted():
    convo = [
        SYSTEM,
        U1,
        asst(1, "old-think-one"),
        tool(1),
        asst(2, "old-think-two"),
        tool(2),
        {"role": "user", "content": "next question"},
        asst(3, "tail-think"),
        tool(3),
        MARK_TEXT,
        asst(4, "tail-think-b"),
        tool(4),
    ]
    seen = []

    def spy(text):
        seen.append(text)
        return 1

    result = apply_thinking_retention(convo, spy, GUARD)
    assert seen == ["tail-think", "tail-think-b"]
    assert result.took_control
    assert result.messages[2]["reasoning_content"] == ""
    assert result.messages[4]["reasoning_content"] == ""
    assert result.messages[7]["reasoning_content"] == "tail-think"


def test_skipping_old_turns_does_not_change_the_final_boundary():
    for n in (3, 6, 9):
        convo = loop_history(n)
        convo.insert(4, {"role": "user", "content": "follow-up"})
        full = replay_strip_boundaries(convo, count_words, GUARD_N)[-1]
        got = apply_thinking_retention(convo, count_words, GUARD_N)
        stripped = [
            i
            for i, (a, b) in enumerate(zip(convo, got.messages))
            if a is not b
        ]
        assert all(i < full for i in stripped)


# ---------------------------------------------------------------------------
# Endpoint level: engine and token counter receive the rebound kwargs/messages.
# ---------------------------------------------------------------------------


def _endpoint_env(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from omlx.engine.vlm import VLMBatchedEngine

    engine = MagicMock(spec=VLMBatchedEngine)
    engine.tokenizer = _Tokenizer()
    engine.tokenizer.encode = lambda t, **kw: t.split()
    engine.start = AsyncMock()
    engine.preflight_chat = AsyncMock()
    engine.count_chat_tokens = MagicMock(return_value=10)
    engine.chat = AsyncMock(side_effect=RuntimeError("stop-after-capture"))
    engine.is_diffusion_model = False
    engine.message_extractor = None
    engine.supports_multimodal_fallback = False

    async def _get_engine(model_id, *, lease=None):
        return engine

    entry = SimpleNamespace(preserve_thinking_default=True)
    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=entry)
    monkeypatch.setattr(server, "get_engine_for_model", _get_engine)
    monkeypatch.setattr(server, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(server, "validate_context_window", lambda *a, **k: None)
    monkeypatch.setattr(server, "get_engine_pool", lambda: pool)
    monkeypatch.setattr(server, "_thinking_retention_guard_tokens", lambda m=None: 3)
    monkeypatch.setitem(
        server.app.dependency_overrides, server.verify_inference_api_key, lambda: True
    )
    return engine


def test_chat_completions_engine_and_counter_get_rebound_kwargs(monkeypatch):
    from fastapi.testclient import TestClient

    engine = _endpoint_env(monkeypatch)
    body = {
        "model": "m",
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": "think one",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
            {"role": "user", "content": [{"type": "text", "text": IMAGE_MARKER_TEXT},
                                          IMG_PART]},
            {"role": "assistant", "content": "", "reasoning_content": "think two",
             "tool_calls": [{"id": "c2", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "r"},
        ],
    }
    with TestClient(server.app, raise_server_exceptions=False) as client:
        client.post("/v1/chat/completions", json=body)
    counted = engine.count_chat_tokens.call_args.args[0]
    ct = engine.count_chat_tokens.call_args.kwargs["chat_template_kwargs"]
    assert ct["preserve_thinking"] is True
    assert [m.get("reasoning_content") for m in counted if m["role"] == "assistant"] == [
        "",
        "think two",
    ]
    chat = engine.chat.await_args.kwargs
    assert chat["chat_template_kwargs"]["preserve_thinking"] is True
    assert chat["preserve_reasoning"] is True
    assert engine.chat.await_args.kwargs["messages"] == counted


_ANTHROPIC_BODY = {
    "model": "m",
    "max_tokens": 64,
    "stream": False,
    "tools": [
        {"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ],
    "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "think one", "signature": "s"},
            {"type": "tool_use", "id": "c1", "name": "f", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "r"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "think two", "signature": "s"},
            {"type": "tool_use", "id": "c2", "name": "f", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c2", "content": "r"}]},
    ],
}

_RESPONSES_BODY = {
    "model": "m",
    "stream": False,
    "tools": [
        {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}
    ],
    "input": [
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think one"}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "r"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think two"}]},
        {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "r"},
    ],
}


@pytest.mark.parametrize(
    "path,body",
    [("/v1/messages", _ANTHROPIC_BODY), ("/v1/responses", _RESPONSES_BODY)],
)
def test_other_endpoints_engine_and_counter_get_rebound_kwargs(monkeypatch, path, body):
    from fastapi.testclient import TestClient

    engine = _endpoint_env(monkeypatch)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        client.post(path, json=body)
    assert engine.chat.await_count == 1, "request never reached engine.chat"
    counted = engine.count_chat_tokens.call_args.args[0]
    ct = engine.count_chat_tokens.call_args.kwargs["chat_template_kwargs"]
    assert ct["preserve_thinking"] is True
    assert [m.get("reasoning_content") for m in counted if m["role"] == "assistant"] == [
        "",
        "think two",
    ]
    chat = engine.chat.await_args.kwargs
    assert chat["chat_template_kwargs"]["preserve_thinking"] is True
    assert chat["preserve_reasoning"] is True
    assert engine.chat.await_args.kwargs["messages"] == counted


def test_anthropic_retention_runs_before_chat_kwargs_template_kwargs_are_set():
    import inspect

    body = inspect.getsource(server.create_anthropic_message)
    apply = body.index("_apply_thinking_retention(")
    assert apply < body.index('chat_kwargs["chat_template_kwargs"]')
    assert apply < body.index('chat_kwargs["preserve_reasoning"]')


# ---------------------------------------------------------------------------
# took_control is monotone in the history prefix (one rendering-mode switch).
# ---------------------------------------------------------------------------


def _took(convo, counter=count_words, guard=GUARD_N):
    return apply_thinking_retention(convo, counter, guard).took_control


def _assert_monotone(convo, counter=count_words, guard=GUARD_N):
    seen = False
    for k in range(1, len(convo) + 1):
        took = _took(convo[:k], counter, guard)
        assert took or not seen, f"took_control flipped back at prefix {k}"
        seen = seen or took


def test_text_only_takeover_survives_a_later_small_loop():
    big = [asst(i, words(f"a{i}_", 10)) for i in range(1, 9)]
    convo = [SYSTEM, U1]
    for a in big:
        convo += [a, tool(int(a["tool_calls"][0]["id"][1:]))]
    convo += [U2, asst(20, "tiny"), tool(20), {"role": "user", "content": "more"},
              asst(21, "small"), tool(21)]
    first = next(
        k for k in range(1, len(convo) + 1) if _took(convo[:k])
    )
    assert first <= 2 + 2 * 8  # took control inside loop 1
    for k in range(first, len(convo) + 1):
        assert _took(convo[:k]), f"flipped back at prefix {k}"


def _convo_mixed():
    c = [SYSTEM, U1]
    for i in range(1, 8):
        c += [asst(i, words(f"x{i}_", 10)), tool(i)]
    c += [MARK_TEXT, U2, asst(8, "s"), tool(8), UPDATE,
          {"role": "user", "content": "q3"}, asst(9, words("y_", 40)), tool(9),
          MARK_VLM, asst(10, "z"), tool(10), final("done")]
    return c


def _convo_notes_only():
    c = [SYSTEM, U1, asst(1, "a b"), tool(1), U2]
    for i in range(2, 9):
        c += [asst(i, words(f"n{i}_", 9)), tool(i)]
    c += [{"role": "user", "content": "q"}, asst(9, "t"), tool(9)]
    return c


def _convo_text_only():
    c = [SYSTEM, U1]
    for i in range(1, 8):
        c += [asst(i, words(f"p{i}_", 10)), tool(i)]
    c += [U2, asst(8, "x"), tool(8), {"role": "user", "content": "q"},
          asst(9, "y"), tool(9), final("end")]
    return c


def _convo_markers_then_big():
    c = [SYSTEM, U1, asst(1, "a"), tool(1), MARK_TEXT, U2]
    for i in range(2, 6):
        c += [asst(i, words(f"m{i}_", 30)), tool(i), MARK_VLM]
    c += [{"role": "user", "content": "q"}, asst(9, "t"), tool(9)]
    return c


@pytest.mark.parametrize(
    "build",
    [_convo_mixed, _convo_notes_only, _convo_text_only, _convo_markers_then_big],
)
def test_took_control_is_monotone_over_all_prefixes(build):
    _assert_monotone(build())


def test_exact_tokenization_still_only_on_the_tail_with_head_proxy():
    convo = [SYSTEM, U1]
    for i in range(1, 9):
        convo += [asst(i, words(f"h{i}_", 10)), tool(i)]
    convo += [U2, asst(20, "tail-a"), tool(20), asst(21, "tail-b"), tool(21)]
    seen = []

    def spy(text):
        seen.append(text)
        return 1

    result = apply_thinking_retention(convo, spy, GUARD_N)
    assert seen == ["tail-a", "tail-b"]
    assert result.took_control  # head loop was oversized by the char proxy


def test_warned_reasons_are_bounded_by_category(server_state, caplog):
    server_state(global_guard=GUARD)
    for n in range(50):
        def encode(text, _n=n, **kw):
            raise ValueError(f"distinct message {_n}")

        engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=encode))
        with caplog.at_level("WARNING"):
            server._apply_thinking_retention(NOTE_CONVO, engine, "m", None)
    assert len(server._retention_warned_reasons) == 1
    assert "encode-raised:ValueError" in server._retention_warned_reasons
    assert any("distinct message 0" in r.message for r in caplog.records)
