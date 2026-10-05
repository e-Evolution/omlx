# SPDX-License-Identifier: Apache-2.0
"""Stop, parser and logits-processor state carry across a decode pause.

Each case runs the same greedy request twice on the real scheduler step loop
(the ``_Rig`` of the park suite: a tiny Qwen3.5 hybrid model against real
paged-SSD stores): once uninterrupted and once paused at a chosen output
token and resumed. The token ids, the streamed text, the final text, the tool
calls and the finish reason must be identical.
"""

import re

import pytest
import test_scheduler_decode_pause as dp
import test_scheduler_park as park
from conftest import MockTokenizer

from omlx.adapter.output_parser import (
    OutputParserFactory,
    OutputParserFinalizeResult,
    OutputParserTokenResult,
)
from omlx.request import Request, SamplingParams

PROMPT = dp.PROMPT
EOS = 99  # outside the tiny vocabulary: decode runs to a stop or max_tokens


class TextTokenizer(MockTokenizer):
    """Token ``t`` reads as ``<t>``: stop strings encode back to their ids."""

    def __init__(self):
        super().__init__()
        self.eos_token_id = EOS

    def encode(self, text, add_special_tokens=True):
        ids = [int(match) for match in re.findall(r"<(\d+)>", text)]
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(
            f"<{int(t)}>"
            for t in token_ids
            if not (skip_special_tokens and int(t) == self.eos_token_id)
        )


class _SpanParserSession:
    """A stateful protocol parser.

    Tokens between two ``toggle`` tokens are buffered as one tool call instead
    of being streamed; the second ``stop`` token ends the request.
    """

    def __init__(self, toggle, stop):
        self.toggle, self.stop = toggle, stop
        self.in_span = False
        self.span: list[int] = []
        self.stops_seen = 0
        self.tool_calls: list[dict[str, str]] = []

    def process_token(self, token_id):
        token_id = int(token_id)
        if token_id == self.stop:
            self.stops_seen += 1
            if self.stops_seen == 2:
                return OutputParserTokenResult(is_stop=True)
        if token_id == self.toggle:
            if self.in_span:
                arguments = "".join(f"<{t}>" for t in self.span)
                self.tool_calls.append({"name": "span", "arguments": arguments})
                self.span = []
            self.in_span = not self.in_span
            return OutputParserTokenResult()
        if self.in_span:
            self.span.append(token_id)
            return OutputParserTokenResult()
        text = f"<{token_id}>"
        return OutputParserTokenResult(stream_text=text, visible_text=text)

    def finalize(self):
        tail = "".join(f"<{t}>" for t in self.span)
        return OutputParserFinalizeResult(
            stream_text=tail, visible_text=tail, tool_calls=list(self.tool_calls)
        )


def _install_span_parser(toggle, stop):
    def setup(rig):
        rig.scheduler._output_parser_factory = OutputParserFactory(
            kind="test-span",
            create_session=lambda tokenizer: _SpanParserSession(toggle, stop),
        )

    return setup


def run(
    tmp_path,
    name,
    *,
    max_tokens,
    pauses=(),
    tokenizer=None,
    setup=None,
    request_attrs=None,
    **sampling,
):
    """Run request ``a``; pause it once its output reaches each count in ``pauses``."""
    rig = dp.make_rig(tmp_path / name, park_enabled=bool(pauses), tokenizer=tokenizer)
    if setup is not None:
        setup(rig)
    request = Request(
        request_id="a",
        prompt=list(PROMPT),
        sampling_params=SamplingParams(
            max_tokens=max_tokens, temperature=0, **sampling
        ),
        arrival_time=1.0,
    )
    for attr, value in (request_attrs or {}).items():
        setattr(request, attr, value)
    rig.scheduler.add_request(request)
    outputs = []
    for at in pauses:
        _step_until_pausable(rig, request, at, outputs)
        # The pending token the pause emits is output token ``at``.
        assert len(request.output_token_ids) == at
        assert rig.scheduler.request_decode_pause("a") is True
        outputs.extend(park.step(rig))
        park.finish_ok_polls(rig, 3)
    outputs.extend(park.run_to_completion(rig, [request], limit=4 * max_tokens))
    return rig, request, outputs


def _step_until_pausable(rig, request, count, outputs, limit=1000):
    """Step until the output reaches ``count`` and the row decodes again."""
    scheduler = rig.scheduler
    for _ in range(limit):
        if (
            len(request.output_token_ids) >= count
            and "a" in scheduler.running
            and scheduler._park_exclusion_reason(request) is None
        ):
            return
        outputs.extend(park.step(rig))
    raise AssertionError("pause point not reached")


def _final(outputs):
    (final,) = [o for o in outputs if o.request_id == "a" and o.finished]
    return final


def assert_same_run(baseline, paused):
    (_, base_row, base_outputs), (_, row, outputs) = baseline, paused
    assert list(row.output_token_ids) == list(base_row.output_token_ids)
    assert dp.streamed_tokens(outputs, "a") == dp.streamed_tokens(base_outputs, "a")
    assert "".join(o.new_text for o in outputs if o.request_id == "a") == "".join(
        o.new_text for o in base_outputs if o.request_id == "a"
    )
    final, base_final = _final(outputs), _final(base_outputs)
    assert not final.error and not base_final.error
    assert final.finish_reason == base_final.finish_reason
    assert final.output_text == base_final.output_text
    assert final.tool_calls == base_final.tool_calls
    assert final.completion_tokens == base_final.completion_tokens


def close(*runs):
    for rig, _, _ in runs:
        rig.close()


@pytest.fixture
def free_run(tmp_path):
    """Greedy output ids of an unconstrained 80-token run."""
    rig, row, _ = run(tmp_path, "free", max_tokens=80)
    rig.close()
    return list(row.output_token_ids)


def _first_new_ngram(tokens, n=3):
    """Index of the first n-gram that never appeared before (nor overlaps)."""
    for i in range(2, len(tokens) - n):
        gram = tokens[i : i + n]
        earlier = tokens[: i + n - 1]
        if not any(earlier[j : j + n] == gram for j in range(len(earlier) - n + 1)):
            return i
    raise AssertionError("no fresh n-gram in the free run")


# --- Stop sequences --------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 1], ids=["after-first", "after-second"])
def test_a_stop_sequence_spanning_the_pause_matches_like_an_uninterrupted_run(
    tmp_path, free_run, offset
):
    i = _first_new_ngram(free_run)
    stop = "".join(f"<{t}>" for t in free_run[i : i + 3])
    common = dict(max_tokens=80, tokenizer=TextTokenizer(), stop=[stop])
    baseline = run(tmp_path, "base", **common)
    # The pending token emitted at the pause is stop token ``offset``.
    paused = run(tmp_path, "paused", pauses=[i + offset], **common)
    try:
        assert _final(baseline[2]).finish_reason == "stop"
        assert list(baseline[1].output_token_ids) == free_run[: i + 2]
        assert paused[1].park_count == 1
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


def test_a_held_stop_prefix_is_released_when_it_diverges_after_the_resume(
    tmp_path, free_run
):
    i = _first_new_ngram(free_run)
    # Token 50 is never generated: the prefix is held, then released.
    stop = f"<{free_run[i]}><{free_run[i + 1]}><50>"
    common = dict(max_tokens=60, tokenizer=TextTokenizer(), stop=[stop])
    baseline = run(tmp_path, "base", **common)
    paused = run(tmp_path, "paused", pauses=[i], **common)
    try:
        assert _final(baseline[2]).finish_reason == "length"
        assert paused[1].park_count == 1
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


def test_a_text_stop_completed_by_the_pending_token_finishes_without_a_pause(
    tmp_path, free_run
):
    """A stop string with no token form (it straddles two tokens' text)."""
    tokenizer = TextTokenizer()
    text = tokenizer.decode(free_run)
    for i in range(3, 60):
        stop = f"{str(free_run[i - 1])[-1]}><{str(free_run[i])[0]}"
        if text.find(stop) == len(tokenizer.decode(free_run[:i])) - 2:
            break
    else:
        raise AssertionError("no straddling stop string in the free run")
    assert tokenizer.encode(stop, add_special_tokens=False) == []
    common = dict(max_tokens=60, tokenizer=TextTokenizer(), stop=[stop])
    baseline = run(tmp_path, "base", **common)
    paused = run(tmp_path, "paused", pauses=[i], **common)
    try:
        assert _final(baseline[2]).finish_reason == "stop"
        # The pending token ended the request: no pause, nothing pinned.
        assert paused[1].park_count == 0
        assert "a" not in paused[0].scheduler._decode_resumes
        assert paused[0].ssd.pinned_block_count == 0
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


# --- Protocol parser ---------------------------------------------------------------


def _occurrences(tokens, token):
    return [i for i, t in enumerate(tokens) if t == token]


def _span_parser_case(free_run):
    """A toggle token with a span before a stop token seen twice."""
    for toggle in dict.fromkeys(free_run):
        spots = _occurrences(free_run, toggle)
        if len(spots) < 2 or spots[1] - spots[0] < 3:
            continue
        for stop in dict.fromkeys(free_run[spots[1] + 1 :]):
            stops = _occurrences(free_run, stop)
            if stop != toggle and len(stops) >= 2 and stops[0] > spots[1]:
                return toggle, spots, stop, stops
    raise AssertionError("no parser case in the free run")


@pytest.mark.parametrize("where", ["inside-span", "between-stops"])
def test_parser_state_carries_across_the_pause(tmp_path, free_run, where):
    toggle, spots, stop, stops = _span_parser_case(free_run)
    pause_at = spots[0] + 1 if where == "inside-span" else stops[0] + 1
    common = dict(max_tokens=80, setup=_install_span_parser(toggle, stop))
    baseline = run(tmp_path, "base", **common)
    paused = run(tmp_path, "paused", pauses=[pause_at], **common)
    try:
        final = _final(baseline[2])
        assert final.finish_reason == "stop"
        assert final.tool_calls
        assert paused[1].park_count == 1
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


def test_a_parser_stop_on_the_pending_token_finishes_without_a_pause(
    tmp_path, free_run
):
    toggle, _, stop, stops = _span_parser_case(free_run)
    common = dict(max_tokens=80, setup=_install_span_parser(toggle, stop))
    baseline = run(tmp_path, "base", **common)
    # The pending token is the second stop token: the parser ends the request.
    paused = run(tmp_path, "paused", pauses=[stops[1]], **common)
    try:
        assert _final(baseline[2]).finish_reason == "stop"
        assert paused[1].park_count == 0
        assert "a" not in paused[0].scheduler._decode_resumes
        assert paused[0].ssd.pinned_block_count == 0
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


# --- Logits processors and max_tokens ----------------------------------------------

THINK_END, THINK_LEAD, THINK_TRAIL = 5, 6, [7, 8]
BUDGET = 10


def _thinking_setup(rig):
    rig.scheduler._resolve_think_close_pattern = lambda *args: (
        [THINK_LEAD],
        list(THINK_TRAIL),
    )


# Before the budget, on it, and inside the forced close sequence.
@pytest.mark.parametrize("pause_at", [BUDGET - 3, BUDGET - 1, BUDGET + 1, BUDGET + 3])
def test_the_thinking_budget_forces_its_close_at_the_same_position(tmp_path, pause_at):
    common = dict(
        max_tokens=40,
        setup=_thinking_setup,
        request_attrs={"needs_think_prefix": True, "think_end_token_id": THINK_END},
        thinking_budget=BUDGET,
    )
    baseline = run(tmp_path, "base", **common)
    paused = run(tmp_path, "paused", pauses=[pause_at], **common)
    try:
        forced = [THINK_LEAD, THINK_END, *THINK_TRAIL]
        tokens = list(baseline[1].output_token_ids)
        start = next(
            i for i in range(len(tokens)) if tokens[i : i + len(forced)] == forced
        )
        assert start <= BUDGET  # the budget closed thinking here
        assert paused[1].park_count == 1
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


@pytest.mark.parametrize("pause_at", [5, 40])
def test_repetition_penalty_context_carries_across_the_pause(tmp_path, pause_at):
    common = dict(max_tokens=80, repetition_penalty=1.5, repetition_context_size=16)
    baseline = run(tmp_path, "base", **common)
    paused = run(tmp_path, "paused", pauses=[pause_at], **common)
    try:
        assert paused[1].park_count == 1
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


def test_two_pauses_keep_the_max_tokens_budget_exact(tmp_path):
    baseline = run(tmp_path, "base", max_tokens=90)
    paused = run(tmp_path, "paused", max_tokens=90, pauses=[3, 50])
    try:
        assert paused[1].park_count == 2
        stats = paused[0].scheduler.get_park_stats()
        assert stats["decode_pauses"] == stats["decode_resumes"] == 2
        assert len(paused[1].output_token_ids) == 90
        assert _final(paused[2]).finish_reason == "length"
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)


def test_a_stop_sequence_survives_two_pauses_inside_it(tmp_path, free_run):
    i = _first_new_ngram(free_run, n=4)
    stop = "".join(f"<{t}>" for t in free_run[i : i + 4])
    common = dict(max_tokens=80, tokenizer=TextTokenizer(), stop=[stop])
    baseline = run(tmp_path, "base", **common)
    # A resumed row emits one token in its resume step, so the second pause
    # comes two tokens later; both land inside the four-token sequence.
    paused = run(tmp_path, "paused", pauses=[i, i + 2], **common)
    try:
        assert paused[1].park_count == 2
        assert_same_run(baseline, paused)
    finally:
        close(baseline, paused)
