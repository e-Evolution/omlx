# SPDX-License-Identifier: Apache-2.0
"""Deterministic thinking retention for agent loops.

Templates in the Qwen3.6+ family strip ``<think>`` from every assistant turn at
or before ``last_query_index``, the last user message that is not wrapped in
``<tool_response>``, unless ``preserve_thinking`` is on.  Some clients append
synthetic user messages inside an agent loop (pi-ai adds
``Attached image(s) from tool result:`` after tool results that carry images;
oMLX renders mid-transcript system updates as ``<system-update>`` user notes).
Each one moves ``last_query_index`` and strips the loop's thinking, so the
rendered prompt shrinks mid-history and the prefix cache breaks.

Turning ``preserve_thinking`` on globally is not an option: with about 50k
tokens of retained old thinking the model produced corrupted output.  This
module instead decides, as a pure function of the message history, which
assistant turns keep their thinking:

* A *real* user message strips all thinking before it, exactly as the template
  does today.  Image markers and system-update notes are *non-query* messages
  and never move the boundary.
* A guard bounds retention.  When the thinking retained since the boundary
  would exceed ``guard_tokens``, the boundary moves to the turn that crossed
  it: everything before that turn is stripped and the count restarts.

Decisions only depend on the history prefix, so a decision taken for a shorter
history replays identically on a longer one and the prompt stays append-only
between strips.  The caller renders with ``preserve_thinking=True`` when
``took_control`` is set; stripped turns carry an empty ``reasoning_content``
and so render ``<think>\\n\\n</think>\\n\\n`` (the template has no way to emit
its bare stripped form while ``preserve_thinking`` is on).

The module is inert (``took_control`` False, messages returned unchanged) when
it is disabled, when no non-query message and no guard strip is involved, and
when no assistant turn carries reasoning.  That makes disabled and unaffected
conversations render byte-for-byte as before.

Invariant: ``took_control`` is a monotone function of the history prefix, so a
conversation switches rendering mode at most once.  Exact tokenization only
runs on the live tail (after the last real user message); older loops are
judged by a cheap deterministic proxy, ``ceil(chars / 2)``, which never
underestimates tokens, replayed with the same boundary rules.  If any earlier
loop could have exceeded the guard, takeover stays on even after a new real
user message moves that loop out of the tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# Exact text pi-ai puts in front of the images of a tool result.
IMAGE_MARKER_TEXT = "Attached image(s) from tool result:"

# Key set by ``prepare_system_messages_for_template`` (policy inline_note) on
# user notes standing in for mid-transcript system messages. Mirrors
# ``omlx.api.utils.MID_SYSTEM_NOTE_KEY`` (not imported: utils is heavy and this
# module has to stay a pure leaf).
MID_SYSTEM_NOTE_KEY = "_omlx_mid_system_note"

_IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})


@dataclass(frozen=True)
class ThinkingRetentionResult:
    """Outcome of one retention pass."""

    messages: list[dict]
    # True when the caller must render with preserve_thinking=True.
    took_control: bool = False
    # Assistant turns whose reasoning was cleared.
    stripped_turns: int = 0


def normalize_guard_tokens(value: Any) -> int | None:
    """Return a positive int guard, or None for anything not valid.

    Rejects bool, non-positive, non-finite and non-integral values. Strings
    must be plain decimal integers (surrounding whitespace allowed).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer() and value > 0:
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.isascii() and text.isdigit():
            parsed = int(text)
            return parsed if parsed > 0 else None
    return None


def _is_marker_content(content: Any) -> bool:
    if isinstance(content, str):
        return content == IMAGE_MARKER_TEXT
    if not isinstance(content, list) or not content:
        return False
    marker_seen = False
    images = 0
    for part in content:
        if not isinstance(part, dict):
            return False
        ptype = part.get("type")
        if ptype in _IMAGE_PART_TYPES:
            images += 1
        elif ptype == "text" and part.get("text") == IMAGE_MARKER_TEXT:
            if marker_seen:
                return False
            marker_seen = True
        else:
            return False
    return marker_seen and images > 0


def is_non_query_user_message(messages: list[dict], index: int) -> bool:
    """Whether ``messages[index]`` is a user message that is not a real query.

    True for a system-update note, or for pi's image marker: a user message
    directly after a tool message whose only text is the marker and whose
    other parts are all images.  Everything else with role user is real.
    """
    msg = messages[index]
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    if msg.get(MID_SYSTEM_NOTE_KEY):
        return True
    if index == 0 or messages[index - 1].get("role") != "tool":
        return False
    return _is_marker_content(msg.get("content"))


def _reasoning_text(msg: dict) -> str:
    value = msg.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _proxy_tokens(text: str) -> int:
    """Cheap token upper bound: never below the word count or ``chars / 2``."""
    return math.ceil(len(text) / 2.0)


def _replay(
    messages: list[dict],
    count_tokens: Callable[[str], int],
    guard_tokens: int,
    count_from: int = 0,
) -> tuple[list[int], bool, bool, bool]:
    """Replay the history in order.

    Assistant turns before ``count_from`` are not tokenized: the caller passes
    the index of the last real user message, which strips everything before it
    regardless of size.  Only the final boundary is consumed by
    ``apply_thinking_retention``, so skipping them cannot change the boundary.
    They are still sized with ``_proxy_tokens`` so that ``guard_stripped``
    (which feeds ``took_control``) stays monotone in the history prefix.

    Returns ``(boundaries, has_reasoning, has_non_query, guard_stripped)``,
    where ``boundaries[i]`` is the strip boundary after processing message
    ``i``: assistant turns with an index below it are stripped.
    """
    boundary = 0
    retained = 0
    live_reasoning_turns = 0
    has_reasoning = False
    has_non_query = False
    guard_stripped = False
    boundaries: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "user":
            if is_non_query_user_message(messages, i):
                has_non_query = True
            else:
                boundary = i
                retained = 0
                live_reasoning_turns = 0
        elif role == "assistant":
            text = _reasoning_text(msg)
            if text.strip():
                has_reasoning = True
                if i < count_from:
                    tokens = _proxy_tokens(text)
                else:
                    tokens = int(count_tokens(text))
                retained += tokens
                if retained > guard_tokens:
                    if live_reasoning_turns:
                        guard_stripped = True
                    boundary = i
                    retained = tokens
                    live_reasoning_turns = 0
                live_reasoning_turns += 1
        boundaries.append(boundary)
    return boundaries, has_reasoning, has_non_query, guard_stripped


def replay_strip_boundaries(
    messages: list[dict],
    count_tokens: Callable[[str], int],
    guard_tokens: int,
) -> list[int]:
    """Strip boundary after each message (exposed for tests and diagnostics)."""
    return _replay(messages, count_tokens, guard_tokens)[0]


def apply_thinking_retention(
    messages: list[dict],
    count_tokens: Callable[[str], int],
    guard_tokens: int | None,
) -> ThinkingRetentionResult:
    """Clear ``reasoning_content`` on the turns to strip.

    ``messages`` is never mutated; modified messages are copies.  When the
    returned ``took_control`` is True the caller must render with
    ``preserve_thinking=True`` so retained turns keep their thinking.
    """
    guard = normalize_guard_tokens(guard_tokens)
    if guard is None or not messages:
        return ThinkingRetentionResult(messages)
    last_real_user = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and not is_non_query_user_message(messages, i)
        ):
            last_real_user = i
            break
    boundaries, has_reasoning, has_non_query, guard_stripped = _replay(
        messages, count_tokens, guard, count_from=last_real_user
    )
    if not has_reasoning or not (has_non_query or guard_stripped):
        return ThinkingRetentionResult(messages)
    boundary = boundaries[-1]
    out: list[dict] = []
    stripped = 0
    for i, msg in enumerate(messages):
        if (
            i < boundary
            and isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and "reasoning_content" in msg
            and msg["reasoning_content"] != ""
        ):
            msg = {**msg, "reasoning_content": ""}
            stripped += 1
        out.append(msg)
    return ThinkingRetentionResult(out, True, stripped)
