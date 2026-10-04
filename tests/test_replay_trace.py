# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmarks/unattended/replay_trace.py (no servers, no models)."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "unattended")
)

import replay_trace as rt  # noqa: E402


def rec(ts, session, messages, **extra):
    payload = {"model": "m", "messages": messages, "max_tokens": 4096, **extra}
    return {"ts": ts, "session": session, "provider": "omlx", "model": "m", "payload": payload}


def U(text):
    return {"role": "user", "content": text}


def A(text):
    return {"role": "assistant", "content": text}


def S(text):
    return {"role": "system", "content": text}


class TestParseAndFilter:
    def test_parses_jsonl_skipping_blank_and_malformed(self):
        lines = [json.dumps(rec(1, "a", [U("x")])), "", "{not json", json.dumps(rec(2, "a", [U("y")]))]
        out = rt.parse_trace(lines)
        assert [r["ts"] for r in out] == [1, 2]

    def test_filters_by_session_and_time_range_keeping_order(self):
        rs = [rec(1, "a", []), rec(2, "b", []), rec(3, "a", []), rec(9, "a", [])]
        assert [r["ts"] for r in rt.filter_records(rs, session="a")] == [1, 3, 9]
        assert [r["ts"] for r in rt.filter_records(rs, ts_from=2, ts_to=3)] == [2, 3]
        assert len(rt.filter_records(rs)) == 4


class TestBuildRequest:
    def test_reuse_mode_forces_one_token_non_streaming(self):
        r = rec(1, "a", [U("x")], stream=True, stream_options={"include_usage": True},
                tools=[{"type": "function"}], chat_template_kwargs={"enable_thinking": True})
        body = rt.build_request(r, "reuse")
        assert body["max_tokens"] == 1 and body["stream"] is False
        assert "stream_options" not in body
        assert body["tools"] == [{"type": "function"}]
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["messages"] == [U("x")]

    def test_full_mode_keeps_recorded_max_tokens_and_streams(self):
        body = rt.build_request(rec(1, "a", [U("x")]), "full")
        assert body["max_tokens"] == 4096
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_max_completion_tokens_is_normalised_to_max_tokens(self):
        r = rec(1, "a", [U("x")])
        del r["payload"]["max_tokens"]
        r["payload"]["max_completion_tokens"] = 77
        body = rt.build_request(r, "full")
        assert body["max_tokens"] == 77 and "max_completion_tokens" not in body
        assert "max_completion_tokens" not in rt.build_request(r, "reuse")

    def test_does_not_mutate_the_record_and_strips_auth(self):
        r = rec(1, "a", [U("x")], apiKey="sk-secret", headers={"Authorization": "Bearer sk-secret"})
        before = copy.deepcopy(r)
        body = rt.build_request(r, "reuse")
        assert r == before
        assert "sk-secret" not in json.dumps(body)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            rt.build_request(rec(1, "a", []), "bogus")


class TestLogParsing:
    DIV = (
        "2026-10-04 10:00:00,000 - omlx.scheduler - INFO - [-] - prefix cache: request r1 "
        "re-prefills 52000 of 60000 tokens (reused 8000); closest stored sequence r0 shares "
        "the first 8192 of 59000 comparable tokens before diverging\n"
    )
    RESTORE = (
        "x - omlx.scheduler - INFO - [-] - Prefix cache restore for r1: source=paged cached=8000 suffix=52000 blocks=4 lookup=1ms\n"
    )

    def test_divergence_line(self):
        d = rt.parse_request_log(self.DIV + self.RESTORE)
        assert d["divergence_position"] == 8192
        assert d["comparable_tokens"] == 59000
        assert d["log_reprefill_tokens"] == 52000
        assert d["restore_source"] == "paged"

    def test_missing_lines_give_none(self):
        d = rt.parse_request_log("nothing here\n")
        assert d["divergence_position"] is None and d["restore_source"] is None
        assert d["holds"] == 0 and d["agings"] == 0 and d["tool_parser_failures"] == 0

    def test_hold_aging_and_tool_parser_failures(self):
        text = (
            "Holding r1 from batched decode: combined context 9 > limit 1\n"
            "Aging: r1 held 40 s\n"
            "Native tool parser failed (ValueError: bad)\n"
            "Tool call markers found but parsing failed, returning text\n"
        )
        d = rt.parse_request_log(text)
        assert (d["holds"], d["agings"], d["tool_parser_failures"]) == (1, 1, 2)


class TestClassifier:
    def test_cold_start_when_no_previous_request_in_session(self):
        assert rt.classify_reprefill(None, rec(1, "a", [S("s"), U("x")]), 1000, 0) == "cold_start"

    def test_no_reprefill_is_hit(self):
        prev = rec(1, "a", [S("s"), U("x")])
        cur = rec(2, "a", [S("s"), U("x"), A("y"), U("z")])
        assert rt.classify_reprefill(prev, cur, 1000, 1000) == "hit"

    def test_compaction_when_prompt_or_messages_shrink(self):
        prev = rec(1, "a", [S("s"), U("1"), A("2"), U("3"), A("4"), U("5")])
        cur = rec(2, "a", [S("s"), U("summary")])
        assert rt.classify_reprefill(prev, cur, 400, 100, prev_prompt_tokens=900) == "compaction"

    def test_image_marker_in_new_turn(self):
        prev = rec(1, "a", [S("s"), U("x")])
        img = {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}
        cur = rec(2, "a", [S("s"), U("x"), A("y"), img])
        assert rt.classify_reprefill(prev, cur, 2000, 500, prev_prompt_tokens=900) == "image"

    def test_system_update_when_system_text_changed_or_added_mid_conversation(self):
        prev = rec(1, "a", [S("rules v1"), U("x")])
        changed = rec(2, "a", [S("rules v2"), U("x"), A("y"), U("z")])
        assert rt.classify_reprefill(prev, changed, 2000, 100, prev_prompt_tokens=900) == "system_update"
        added = rec(2, "a", [S("rules v1"), U("x"), A("y"), S("reminder"), U("z")])
        assert rt.classify_reprefill(prev, added, 2000, 100, prev_prompt_tokens=900) == "system_update"

    def test_unknown_when_nothing_matches(self):
        prev = rec(1, "a", [S("s"), U("x")])
        cur = rec(2, "a", [S("s"), U("x"), A("y"), U("z")])
        assert rt.classify_reprefill(prev, cur, 2000, 100, prev_prompt_tokens=900) == "unknown"


def row(prompt, cached, cls, ttft=None, flags=None, ok=True, **kw):
    return {"ok": ok, "prompt_tokens": prompt, "cached_tokens": cached, "class": cls,
            "ttft_s": ttft, "quality_flags": flags or [], "holds": 0, "agings": 0,
            "tool_parser_failures": kw.get("tpf", 0), "aborts": kw.get("aborts", 0)}


class TestReducer:
    ROWS = [
        row(1000, 0, "cold_start", 10.0),
        row(1100, 1000, "unknown", 1.0),
        row(2000, 1000, "image", 6.0),
        row(2100, 2000, "hit", 0.5),
    ]

    def test_token_weighted_hit_ratio_and_reprefill_totals(self):
        s = rt.reduce_results(self.ROWS, "full")
        assert s["prompt_tokens"] == 6200
        assert s["cached_tokens"] == 4000
        assert s["hit_ratio"] == pytest.approx(4000 / 6200)
        assert s["reprefilled_tokens"] == 2200
        assert s["by_class"]["cold_start"] == {"requests": 1, "reprefilled_tokens": 1000}
        assert s["by_class"]["image"]["reprefilled_tokens"] == 1000
        assert s["by_class"]["unknown"]["reprefilled_tokens"] == 100
        assert s["by_class"]["hit"]["reprefilled_tokens"] == 100

    def test_ttft_percentiles_only_in_full_mode(self):
        full = rt.reduce_results(self.ROWS, "full")
        assert full["ttft_p50_s"] == pytest.approx(3.5)
        assert full["ttft_p90_s"] == pytest.approx(rt.percentile([0.5, 1.0, 6.0, 10.0], 90))
        assert rt.reduce_results(self.ROWS, "reuse")["ttft_p50_s"] is None

    def test_percentile_edges(self):
        assert rt.percentile([], 50) is None
        assert rt.percentile([4.0], 90) == 4.0
        assert rt.percentile([1.0, 3.0], 50) == pytest.approx(2.0)

    def test_failures_aborts_and_quality_flags(self):
        rows = [row(10, 0, "cold_start", flags=["literal_think_close"]),
                row(10, 0, "unknown", ok=False, aborts=2, tpf=1)]
        s = rt.reduce_results(rows, "full")
        assert s["failed_requests"] == 1
        assert s["aborts"] == 2
        assert s["quality"]["literal_think_close"] == 1
        assert s["quality"]["tool_parser_failures"] == 1

    def test_empty_results(self):
        s = rt.reduce_results([], "reuse")
        assert s["requests"] == 0 and s["hit_ratio"] is None


class TestQualityFlags:
    def test_flags(self):
        assert rt.quality_flags("ok text") == []
        assert rt.quality_flags("a </think> b") == ["literal_think_close"]
        assert sorted(rt.quality_flags("<function=x> <parameter=y>")) == ["literal_function", "literal_parameter"]


class TestDeterminism:
    def test_identical_results_compare_equal_ignoring_timings(self):
        a = [{"prompt_tokens": 10, "cached_tokens": 5, "class": "unknown", "ttft_s": 1.0, "wall_s": 2.0}]
        b = [{"prompt_tokens": 10, "cached_tokens": 5, "class": "unknown", "ttft_s": 9.0, "wall_s": 7.0}]
        assert rt.results_equal(a, b)

    def test_different_cached_tokens_not_equal(self):
        a = [{"prompt_tokens": 10, "cached_tokens": 5, "class": "unknown"}]
        b = [{"prompt_tokens": 10, "cached_tokens": 6, "class": "unknown"}]
        assert not rt.results_equal(a, b)
        assert not rt.results_equal(a, a + a)


class TestGuards:
    def test_refuses_port_8001_unless_allowed(self, capsys):
        argv = ["--trace", "/nonexistent", "--variant", "fork", "--port", "8001"]
        assert rt.main(argv) == 2
        assert "8001" in capsys.readouterr().err

    def test_missing_trace_is_an_error_before_any_server_work(self, capsys):
        assert rt.main(["--trace", "/nonexistent/x.jsonl", "--variant", "fork"]) == 2


class TestReplayLoop:
    class FakeClient:
        def __init__(self):
            self.bodies = []

        def send(self, body, mode):
            self.bodies.append(copy.deepcopy(body))
            n = sum(len(m["content"]) if isinstance(m["content"], str) else 5 for m in body["messages"])
            return {"ok": True, "prompt_tokens": n, "cached_tokens": 0, "ttft_s": 0.1, "wall_s": 0.2, "text": "fine </think>", "error": None}

    def test_replays_in_order_without_mutating_records(self):
        records = [rec(1, "a", [U("aaaa")]), rec(2, "a", [U("aaaa"), A("b"), U("cc")])]
        before = copy.deepcopy(records)
        client = self.FakeClient()
        rows = rt.replay(records, client, "full", log_window=lambda: "")
        assert records == before
        assert [b["messages"] for b in client.bodies] == [r["payload"]["messages"] for r in records]
        assert [r["class"] for r in rows] == ["cold_start", "unknown"]
        assert rows[0]["quality_flags"] == ["literal_think_close"]
