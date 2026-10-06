# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure logic of benchmarks/unattended (no servers)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "unattended")
)

import bench_metrics as bm  # noqa: E402

LOG = """\
2026-10-02 04:25:24,438 - omlx.process_memory_enforcer - INFO - [-] - MLX buffer cache limit set to 16.0 GB (was 128.0 GB)
2026-10-02 04:25:29,888 - omlx.cache.paged_ssd_cache - INFO - [-] - PagedSSDCacheManager initialized: dir=/Volumes/X/c, max_size=150.00 GB
2026-10-02 04:26:00,000 - omlx.process_memory_enforcer - INFO - [-] - Memory pressure level: normal -> warn (current=80.0GB, soft=90.0GB, hard=100.0GB, ceiling=110.0GB)
2026-10-02 04:26:01,000 - omlx.process_memory_enforcer - INFO - [-] - Memory pressure level: warn -> critical (current=95.0GB, soft=90.0GB, hard=100.0GB, ceiling=110.0GB)
2026-10-02 04:26:02,000 - omlx.process_memory_enforcer - INFO - [-] - Requested pressure cache reclaim on 1 scheduler(s) (freed_hot=0.0GB, pool=12.5GB)
2026-10-02 04:26:03,000 - omlx.process_memory_enforcer - INFO - [-] - Requested pressure cache reclaim on 1 scheduler(s) (freed_hot=1.0GB, pool=85.3GB)
2026-10-02 04:26:04,000 - omlx.scheduler - INFO - [-] - Parked request abc at 8192 tokens committed (park 1/3), reason: pressure
2026-10-02 04:26:05,000 - omlx.scheduler - INFO - [-] - Park skipped for def: no committed progress (memory pressure falls back to abort)
2026-10-02 04:26:06,000 - omlx.scheduler - INFO - [-] - Resumed parked request abc: cached=8192
2026-10-02 04:26:07,000 - omlx.scheduler - INFO - [-] - Holding r1 from batched decode: combined context 260000 > limit 200000 (single-row decode keeps fused kernels)
2026-10-02 04:26:08,000 - omlx.scheduler - INFO - [-] - Releasing r1 to decode
2026-10-02 04:26:09,000 - omlx.scheduler - INFO - [-] - Prefix cache restore for r2: source=paged cached=129024 suffix=5949 blocks=63 lookup=11.1ms reconstruct=4494.7ms promote=True
2026-10-02 04:26:10,000 - omlx.scheduler - INFO - [-] - Prefix cache restore for r3: source=paged cached=40960 suffix=4028 blocks=20 lookup=2.4ms reconstruct=1.4ms promote=True
2026-10-02 04:26:11,000 - omlx.engine_core - WARNING - [-] - Aborted 3 requests due to memory pressure
2026-10-02 04:26:12,000 - omlx.engine_core - WARNING - [-] - Aborted 1 requests: client disconnected
2026-10-02 04:26:13,000 - omlx.patches.turboquant_attention - INFO - [-] - TurboQuant batched attention: per-row fused path (B=3)
2026-10-02 04:26:14,000 - omlx.process_memory_enforcer - WARNING - [-] - Hard memory pressure: aborted 2 request(s) on m
"""


class TestParseServerLogRagged:
    def test_ragged_path_counts_as_batch_but_forced_b1_does_not(self):
        text = (
            "x - omlx.patches.turboquant_attention - INFO - [-] - "
            "TurboQuant batched attention: ragged single-launch path (B=2)\n"
            "x - omlx.patches.turboquant_attention - INFO - [-] - "
            "TurboQuant batched attention: ragged single-launch path (B=3)\n"
            "x - omlx.patches.turboquant_attention - INFO - [-] - "
            "TurboQuant batched attention: ragged single-launch path (B=1, forced)\n"
        )
        m = bm.parse_server_log(text)
        assert m["tq_batched_fused_rows"] == [2, 3]
        assert m["batch_max_b"] == 3


class TestParseServerLogObservability:
    EVICT = (
        "x - omlx.cache.paged_ssd_cache - INFO - [-] - SSD cache evicted: blocks=11 "
        "bytes=260 sidecars=1 window_s=60.0 total_blocks=12 total_bytes=360 "
        "total_sidecars=1\n"
        "x - omlx.cache.paged_ssd_cache - INFO - [-] - SSD cache evicted: blocks=4 "
        "bytes=40 sidecars=0 window_s=61.0 total_blocks=16 total_bytes=400 "
        "total_sidecars=1\n"
    )

    def test_ssd_evictions_are_summed_from_summary_lines(self):
        m = bm.parse_server_log(self.EVICT)
        assert m["ssd_evicted_blocks"] == 15
        assert m["ssd_evicted_bytes"] == 300
        assert m["ssd_evicted_sidecars"] == 1

    def test_no_eviction_lines_give_zero(self):
        m = bm.parse_server_log("nothing here\n")
        assert (m["ssd_evicted_blocks"], m["ssd_evicted_bytes"]) == (0, 0)


class TestParseServerLog:
    def test_aborts_split_memory_from_other(self):
        m = bm.parse_server_log(LOG)
        assert m["aborts_total"] == 3 + 1 + 2
        assert m["aborts_memory"] == 3 + 2

    def test_pool_max_and_values(self):
        m = bm.parse_server_log(LOG)
        assert m["pool_gb_max"] == pytest.approx(85.3)
        assert m["pool_gb_values"] == [12.5, 85.3]

    def test_pool_absent_is_none(self):
        assert bm.parse_server_log("nothing here\n")["pool_gb_max"] is None

    def test_pressure_levels(self):
        m = bm.parse_server_log(LOG)
        assert m["pressure_transitions"] == [["normal", "warn"], ["warn", "critical"]]
        assert m["pressure_levels_seen"] == ["normal", "warn", "critical"]

    def test_parks_and_skips(self):
        m = bm.parse_server_log(LOG)
        assert m["parks"] == 1
        assert m["parked_tokens"] == [8192]
        assert m["park_skips"] == 1
        assert m["park_resumes"] == 1

    def test_hold_release(self):
        m = bm.parse_server_log(LOG)
        assert m["holds"] == 1
        assert m["releases"] == 1

    def test_prefix_cache_restores(self):
        m = bm.parse_server_log(LOG)
        assert m["restores"] == [129024, 40960]
        assert m["restore_cached_last"] == 40960
        assert m["restore_cached_max"] == 129024

    def test_path_lines(self):
        m = bm.parse_server_log(LOG)
        assert m["mlx_cache_limit_gb"] == pytest.approx(16.0)
        assert m["tq_batched_fused_rows"] == [3]
        assert m["tq_batched_fused_max_b"] == 3
        assert m["tq_batched_fallback"] is False

    def test_tq_fallback_flag(self):
        line = "x - TurboQuant batched attention: per-row dequantize fallback (B=2)\n"
        m = bm.parse_server_log(line)
        assert m["tq_batched_fallback"] is True

    def test_ssd_initialized(self):
        m = bm.parse_server_log(LOG)
        assert m["ssd_initialized"] is True
        assert m["ssd_disabled"] is False

    @pytest.mark.parametrize(
        "line",
        [
            "SSD cache disabled: x",
            "OSError: [Errno 1] Operation not permitted: '/Volumes/x'",
        ],
    )
    def test_ssd_disabled_markers(self, line):
        m = bm.parse_server_log(line + "\n")
        assert m["ssd_disabled"] is True
        assert m["ssd_initialized"] is False

    def test_empty_log(self):
        m = bm.parse_server_log("")
        assert m["aborts_total"] == 0 and m["restores"] == [] and m["parks"] == 0


def _frame(obj):
    import json

    return "data: " + json.dumps(obj)


class TestStreamAccumulator:
    def test_ok_stream_with_usage(self):
        acc = bm.StreamAccumulator()
        assert (
            acc.feed(_frame({"choices": [{"delta": {"role": "assistant"}}]}), now=1.0)
            is False
        )
        assert (
            acc.feed(_frame({"choices": [{"delta": {"content": "Hel"}}]}), now=2.0)
            is True
        )
        acc.feed(_frame({"choices": [{"delta": {"reasoning_content": "lo"}}]}), now=3.0)
        acc.feed(
            _frame(
                {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
            ),
            now=4.0,
        )
        acc.feed("data: [DONE]", now=4.1)
        r = acc.finalize()
        assert r["ok"] is True
        assert r["text"] == "Hello"
        assert r["first_token_at"] == 2.0
        assert r["prompt_tokens"] == 100 and r["completion_tokens"] == 5

    def test_keepalive_and_comments_do_not_set_first_token(self):
        acc = bm.StreamAccumulator()
        acc.feed(": keep-alive", now=1.0)
        acc.feed(_frame({"choices": [{"delta": {"content": ""}}]}), now=1.5)
        assert acc.first_token_at is None

    def test_error_frame_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed(_frame({"choices": [{"delta": {"content": "x"}}]}), now=1.0)
        acc.feed(
            _frame({"error": {"message": "Memory limit exceeded", "type": "x"}}),
            now=2.0,
        )
        acc.feed(
            _frame(
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            ),
            now=3.0,
        )
        r = acc.finalize()
        assert r["ok"] is False
        assert "Memory limit exceeded" in r["error"]

    def test_event_error_line_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed("event: error", now=1.0)
        acc.feed(_frame({"message": "boom"}), now=1.1)
        assert acc.finalize()["ok"] is False

    def test_missing_usage_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed(_frame({"choices": [{"delta": {"content": "x"}}]}), now=1.0)
        acc.feed("data: [DONE]", now=2.0)
        r = acc.finalize()
        assert r["ok"] is False
        assert "usage" in r["error"]

    def test_missing_done_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed(_frame({"choices": [{"delta": {"content": "x"}}]}), now=1.0)
        acc.feed(
            _frame(
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            ),
            now=2.0,
        )
        r = acc.finalize()
        assert r["ok"] is False
        assert "DONE" in r["error"]

    def test_no_tokens_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed(
            _frame(
                {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0}}
            ),
            now=2.0,
        )
        acc.feed("data: [DONE]", now=2.1)
        assert acc.finalize()["ok"] is False

    def test_cached_tokens_from_usage_details(self):
        acc = bm.StreamAccumulator()
        acc.feed(_frame({"choices": [{"delta": {"content": "x"}}]}), now=1.0)
        acc.feed(
            _frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                }
            ),
            now=2.0,
        )
        acc.feed("data: [DONE]", now=2.1)
        assert acc.finalize()["cached_tokens"] == 4

    def test_invalid_json_is_failure(self):
        acc = bm.StreamAccumulator()
        acc.feed("data: {not json", now=1.0)
        assert acc.finalize()["ok"] is False


class TestRequestMetrics:
    def test_rates(self):
        m = bm.request_metrics(
            t_start=10.0,
            t_first=20.0,
            t_end=30.0,
            prompt_tokens=1000,
            completion_tokens=101,
        )
        assert m["ttft_s"] == pytest.approx(10.0)
        assert m["prefill_tps"] == pytest.approx(100.0)
        assert m["decode_tps"] == pytest.approx(10.0)  # (101-1)/10
        assert m["total_s"] == pytest.approx(20.0)

    def test_single_token_decode_is_none(self):
        m = bm.request_metrics(0.0, 1.0, 1.0, 10, 1)
        assert m["decode_tps"] is None

    def test_missing_first_token(self):
        m = bm.request_metrics(0.0, None, 5.0, 10, 0)
        assert m["ttft_s"] is None and m["prefill_tps"] is None


class TestFootprint:
    def test_parse_units(self):
        out = "zsh [94571]: 64-bit    Footprint: 2016 KB (16384 bytes per page)\n"
        assert bm.parse_footprint_gb(out) == pytest.approx(2016 / 1024 / 1024)
        assert (
            bm.parse_footprint_gb(
                "p [1]: 64-bit    Footprint: 32 GB (16384 bytes per page)"
            )
            == 32
        )
        assert bm.parse_footprint_gb(
            "p [1]: 64-bit    Footprint: 512 MB (x)"
        ) == pytest.approx(0.5)

    def test_sum_multiple_processes(self):
        out = "a [1]: 64-bit    Footprint: 1 GB (x)\nb [2]: 64-bit    Footprint: 512 MB (x)\n"
        assert bm.parse_footprint_gb(out) == pytest.approx(1.5)

    def test_unparseable_is_none(self):
        assert bm.parse_footprint_gb("garbage") is None


class TestStats:
    def test_median(self):
        assert bm.median([3, 1, 2]) == 2
        assert bm.median([1, 2, 3, 4]) == 2.5
        assert bm.median([]) is None
        assert bm.median([None, 4.0]) == 4.0

    def test_summarize_samples(self):
        s = bm.summarize_samples(
            [
                {"t": 0, "footprint_gb": 10.0, "health_ok": True, "health_s": 0.01},
                {"t": 1, "footprint_gb": 40.0, "health_ok": False, "health_s": 2.0},
                {"t": 2, "footprint_gb": None, "health_ok": True, "health_s": 0.02},
            ]
        )
        assert s["footprint_peak_gb"] == 40.0
        assert s["health_failures"] == 1
        assert s["health_max_s"] == 2.0
        assert s["n"] == 3

    def test_summarize_empty(self):
        s = bm.summarize_samples([])
        assert s["footprint_peak_gb"] is None and s["n"] == 0


def _req(pre=1000.0, dec=20.0, ok=True, **kw):
    r = {
        "ok": ok,
        "prompt_tokens": 110000,
        "completion_tokens": 256,
        "prefill_tps": pre,
        "decode_tps": dec,
        "ttft_s": 100.0,
        "total_s": 110.0,
        "error": None if ok else "boom",
    }
    r.update(kw)
    return r


def _run(scenario, log=None, ok=True):
    return {"ok": ok, "scenario": scenario, "log": log or bm.parse_server_log("")}


class TestEvalP6:
    def test_pass_within_tolerance(self):
        up = [_run({"request": _req(1000, 20)}) for _ in range(3)]
        fk = [_run({"request": _req(960, 19.2)}) for _ in range(3)]
        r = bm.eval_p6(up, fk)
        assert r["verdict"] == "PASS"

    def test_fail_on_prefill_regression(self):
        up = [_run({"request": _req(1000, 20)})]
        fk = [_run({"request": _req(900, 20)})]
        r = bm.eval_p6(up, fk)
        assert r["verdict"] == "FAIL"
        assert any(
            c["name"] == "prefill_tps >= -5%" and not c["ok"] for c in r["checks"]
        )

    def test_median_is_used(self):
        up = [_run({"request": _req(1000, 20)}) for _ in range(3)]
        fk = [_run({"request": _req(p, 20)}) for p in (500, 990, 995)]
        assert bm.eval_p6(up, fk)["verdict"] == "PASS"

    def test_failed_fork_request_fails(self):
        up = [_run({"request": _req()})]
        fk = [_run({"request": _req(ok=False)})]
        assert bm.eval_p6(up, fk)["verdict"] == "FAIL"

    def test_no_upstream_is_inconclusive(self):
        fk = [_run({"request": _req()})]
        assert bm.eval_p6([], fk)["verdict"] == "INCONCLUSIVE"

    def test_no_fork_runs(self):
        assert bm.eval_p6([_run({"request": _req()})], [])["verdict"] == "INCONCLUSIVE"


def _p1(cached, progress=66000, ok=True):
    return _run(
        {
            "prompt_tokens": 110000,
            "progress_tokens_est": progress,
            "reissue": _req(ok=ok),
            "reissue_cached": cached,
            "reissue_reprefilled": 110000 - (cached or 0),
        }
    )


class TestEvalP1:
    def test_pass_when_half_of_progress_cached(self):
        r = bm.eval_p1([_p1(40960)], [_p1(0)])
        assert r["verdict"] == "PASS"

    def test_fail_when_below_half(self):
        assert bm.eval_p1([_p1(20000)], [_p1(0)])["verdict"] == "FAIL"

    def test_fail_when_reissue_failed(self):
        assert bm.eval_p1([_p1(66000, ok=False)], [])["verdict"] == "FAIL"

    def test_none_cached_is_zero(self):
        assert bm.eval_p1([_p1(None)], [])["verdict"] == "FAIL"

    def test_upstream_reported(self):
        r = bm.eval_p1([_p1(40960)], [_p1(0)])
        assert r["upstream"]["cached_median"] == 0
        assert r["fork"]["cached_median"] == 40960

    def test_no_fork_runs_inconclusive(self):
        assert bm.eval_p1([], [_p1(0)])["verdict"] == "INCONCLUSIVE"


def _p2(aborts_mem=0, pool=None, oks=(True, True)):
    log = bm.parse_server_log("")
    log["aborts_memory"] = aborts_mem
    log["pool_gb_max"] = pool
    return _run(
        {"requests": [_req(ok=o) for o in oks], "wall_s": 900.0},
        log=log,
    )


class TestEvalP2:
    def test_pass(self):
        assert (
            bm.eval_p2([_p2(0, 16.5)], [_p2(3, 85.0, (False, False))])["verdict"]
            == "PASS"
        )

    def test_fail_on_abort(self):
        assert bm.eval_p2([_p2(1, 10.0)], [])["verdict"] == "FAIL"

    def test_fail_on_pool_over_limit(self):
        assert bm.eval_p2([_p2(0, 17.5)], [])["verdict"] == "FAIL"

    def test_fail_on_failed_request(self):
        assert bm.eval_p2([_p2(0, 10.0, (True, False))], [])["verdict"] == "FAIL"

    def test_no_pool_evidence_still_passes_but_noted(self):
        r = bm.eval_p2([_p2(0, None)], [])
        assert r["verdict"] == "PASS"
        assert any(
            "pool" in c["name"] and "no pool" in c["detail"] for c in r["checks"]
        )

    def test_pool_limit_configurable(self):
        assert bm.eval_p2([_p2(0, 20.0)], [], pool_limit_gb=21.0)["verdict"] == "PASS"


BASE_TEXT = " ".join(f"w{i}" for i in range(200))
CHUNK = 5  # chars per streamed chunk in the fixtures -> 32 tokens = 160 chars


def _out(text, ok=True):
    return {
        "ok": ok,
        "text": text,
        "completion_tokens": 200,
        "chunk_chars": [CHUNK] * 100,
    }


def _p3row(name, diverge_at=None, degenerate=False, ok=True, cold_differs=False):
    b3 = BASE_TEXT
    if degenerate:
        b3 = "alpha beta gamma delta " * 60
    elif diverge_at is not None:
        b3 = BASE_TEXT[:diverge_at] + "#" + BASE_TEXT[diverge_at + 1 :]
    return {
        "name": name,
        "alone": _out(BASE_TEXT + (" x" if cold_differs else ""), ok),
        "alone_restored": _out(BASE_TEXT, ok),
        "batched": _out(b3, ok),
    }


def _p3(diverge_at=None, degenerate=False, batch_b=3, rows_ok=True, cold_differs=False):
    rows = [
        _p3row(n, diverge_at, degenerate, rows_ok, cold_differs)
        for n in ("20k", "45k", "135k")
    ]
    log = bm.parse_server_log("")
    log["batch_max_b"] = batch_b
    return _run({"rows": rows}, log=log)


class TestTextMetrics:
    def test_distinct4_all_unique(self):
        assert bm.distinct_ngram_ratio(BASE_TEXT, 4) == 1.0

    def test_distinct4_loop_is_low(self):
        assert bm.distinct_ngram_ratio("a b c d " * 50, 4) < 0.05

    def test_distinct4_too_short_is_none(self):
        assert bm.distinct_ngram_ratio("a b c", 4) is None

    def test_is_degenerate_relative_and_floor(self):
        assert bm.is_degenerate(0.9, 0.4)  # < 0.5 * 0.9
        assert not bm.is_degenerate(0.9, 0.46)
        assert bm.is_degenerate(0.4, 0.25)  # below absolute floor 0.3
        assert not bm.is_degenerate(0.2, 0.19)  # baseline already below floor
        assert not bm.is_degenerate(None, 0.9)
        assert not bm.is_degenerate(0.9, None)

    def test_first_divergence(self):
        assert bm.first_divergence("abcdef", "abcxyz") == 3
        assert bm.first_divergence("abc", "abc") is None
        assert bm.first_divergence("abc", "abcd") == 3

    def test_prefix_chars_counts_first_n_chunks(self):
        assert bm.prefix_chars([3, 4, 5, 6], 2) == 7
        assert bm.prefix_chars([3, 4], 32) == 7

    def test_chunk_index_at_char(self):
        assert bm.chunk_index_at_char([5, 5, 5], 0) == 0
        assert bm.chunk_index_at_char([5, 5, 5], 7) == 1
        assert bm.chunk_index_at_char([5, 5, 5], 99) == 3

    def test_spread(self):
        assert bm.spread([3.0, None, 1.0, 2.0]) == (1.0, 3.0)
        assert bm.spread([]) == (None, None)


def _chunks(start, rate, n):
    return [start + k / rate for k in range(n)]


class TestAggregateDecodeRate:
    def test_sum_of_row_rates_over_common_window(self):
        rows = [
            {"chunk_times": _chunks(0.0, 10.0, 101), "completion_tokens": 101}
            for _ in range(3)
        ]
        assert bm.aggregate_decode_rate(rows)["rate"] == pytest.approx(30.0)

    def test_window_excludes_rows_not_yet_decoding(self):
        rows = [
            {"chunk_times": _chunks(0.0, 10.0, 301), "completion_tokens": 301},
            {"chunk_times": _chunks(0.0, 10.0, 301), "completion_tokens": 301},
            {"chunk_times": _chunks(10.0, 10.0, 301), "completion_tokens": 301},
        ]
        r = bm.aggregate_decode_rate(rows)
        assert r["window_s"] == pytest.approx(20.0)
        assert r["rate"] == pytest.approx(30.0, rel=0.01)

    def test_tokens_per_chunk_scaling(self):
        rows = [
            {"chunk_times": _chunks(0.0, 10.0, 101), "completion_tokens": 202}
            for _ in range(3)
        ]
        assert bm.aggregate_decode_rate(rows)["rate"] == pytest.approx(60.0, rel=0.02)

    def test_no_common_window(self):
        rows = [
            {"chunk_times": _chunks(0.0, 10.0, 11), "completion_tokens": 11},
            {"chunk_times": _chunks(100.0, 10.0, 11), "completion_tokens": 11},
        ]
        r = bm.aggregate_decode_rate(rows)
        assert r["rate"] is None and r["window_s"] is None

    def test_missing_timestamps(self):
        assert bm.aggregate_decode_rate([{"completion_tokens": 5}])["rate"] is None


class TestParseBatchLog:
    def test_mtp_batch_line(self):
        m = bm.parse_server_log("INFO Lightning MTP batch activated for 3 sequences\n")
        assert m["mtp_batch_max"] == 3 and m["batch_max_b"] == 3

    def test_batch_max_uses_both_sources(self):
        m = bm.parse_server_log(
            "Lightning MTP batch activated for 2 sequences\n"
            "TurboQuant batched attention: per-row fused path (B=3)\n"
        )
        assert m["batch_max_b"] == 3
        assert bm.parse_server_log("")["batch_max_b"] == 0


class TestEvalP3:
    def test_pass(self):
        assert bm.eval_p3([_p3()], [_p3(diverge_at=10)])["verdict"] == "PASS"

    def test_fail_on_divergence_inside_first_32_tokens(self):
        r = bm.eval_p3([_p3(diverge_at=100)], [])
        assert r["verdict"] == "FAIL"
        assert any("32" in c["name"] and not c["ok"] for c in r["checks"])

    def test_late_divergence_is_allowed(self):
        r = bm.eval_p3([_p3(diverge_at=300)], [])
        assert r["verdict"] == "PASS"
        assert r["fork"]["detail"][0]["first_divergence_chars"] == 300

    def test_fail_on_degeneration(self):
        r = bm.eval_p3([_p3(degenerate=True)], [])
        assert r["verdict"] == "FAIL"
        assert any("degenerat" in c["name"] and not c["ok"] for c in r["checks"])

    def test_fail_when_batch_not_exercised(self):
        r = bm.eval_p3([_p3(batch_b=2)], [])
        assert r["verdict"] == "FAIL"
        assert any("batch" in c["name"] and not c["ok"] for c in r["checks"])

    def test_fail_when_request_failed(self):
        assert bm.eval_p3([_p3(rows_ok=False)], [])["verdict"] == "FAIL"

    def test_upstream_detail_is_reported_not_judged(self):
        r = bm.eval_p3([_p3()], [_p3(diverge_at=10, degenerate=False)])
        d = r["upstream"]["detail"][0]
        assert d["first_divergence_chars"] == 10
        assert d["diverged_in_prefix"] is True
        assert r["upstream"]["batch_max_b"] == 3
        up = bm.eval_p3([_p3()], [_p3(degenerate=True)])["upstream"]
        assert up["degenerate_rows"] == 3
        assert up["detail"][0]["d4_b3"] < up["detail"][0]["d4_b1"]

    def test_cold_vs_restored_is_informational(self):
        r = bm.eval_p3([_p3(cold_differs=True)], [])
        assert r["verdict"] == "PASS"
        assert r["fork"]["cold_vs_restored_mismatched_rows"] == 3


def _rec(tok, lp=-0.1, gap=3.0):
    """A token record whose top-2 margin is `gap` nats."""
    return {"t": tok, "lp": lp, "top": [[tok, lp], [tok + 1000, lp - gap]]}


def _recs(n, **kw):
    return [_rec(i, **kw) for i in range(n)]


class TestLogprobComparison:
    def test_identical_has_no_divergence(self):
        c = bm.compare_logprobs(_recs(40), _recs(40))
        assert c["div_index"] is None
        assert c["mean_abs_dlp"] == 0.0
        assert c["n_shared"] == 40

    def test_margin_taken_from_reference_top2_at_divergence(self):
        ref, test = _recs(40), _recs(40)
        ref[10] = _rec(10, lp=-0.2, gap=0.3)
        test[10] = {"t": 1010, "lp": -0.5, "top": [[1010, -0.5], [10, -0.8]]}
        c = bm.compare_logprobs(ref, test)
        assert c["div_index"] == 10
        assert c["margin"] == pytest.approx(0.3)
        assert c["n_shared"] == 10

    def test_mean_and_max_abs_dlp_over_shared_prefix(self):
        ref, test = _recs(20), _recs(20)
        test[2]["lp"] = ref[2]["lp"] - 0.4  # |dlp| 0.4 on one of 5 shared tokens
        test[5] = {"t": 999, "lp": -9.0, "top": [[999, -9.0]]}
        c = bm.compare_logprobs(ref, test)
        assert c["div_index"] == 5
        assert c["mean_abs_dlp"] == pytest.approx(0.4 / 5)
        assert c["max_abs_dlp"] == pytest.approx(0.4)

    def test_length_mismatch_without_token_difference_is_not_divergence(self):
        c = bm.compare_logprobs(_recs(40), _recs(30))
        assert c["div_index"] is None and c["n_shared"] == 30

    def test_missing_second_candidate_gives_none_margin(self):
        ref, test = _recs(5), _recs(5)
        ref[3] = {"t": 3, "lp": -0.1, "top": [[3, -0.1]]}
        test[3] = _rec(77)
        assert bm.compare_logprobs(ref, test)["margin"] is None


class TestClassifyDivergence:
    def _c(self, margin, mean):
        return {"div_index": 5, "margin": margin, "mean_abs_dlp": mean}

    def test_no_divergence(self):
        assert bm.classify_divergence({"div_index": None}) is None

    def test_small_margin_small_drift_is_noise(self):
        assert bm.classify_divergence(self._c(0.1, 0.01)) == "noise"

    def test_large_margin_is_bug(self):
        assert bm.classify_divergence(self._c(2.5, 0.01)) == "bug"

    def test_large_prefix_drift_is_bug(self):
        assert bm.classify_divergence(self._c(0.1, 0.5)) == "bug"

    def test_grey_zone_is_inconclusive(self):
        assert bm.classify_divergence(self._c(1.0, 0.01)) == "inconclusive"
        assert bm.classify_divergence(self._c(0.1, 0.1)) == "inconclusive"

    def test_unknown_margin_is_inconclusive(self):
        assert bm.classify_divergence(self._c(None, 0.01)) == "inconclusive"


def _p3lp(ref_gap_at_div=0.2, div_at=10, drift=0.0, batch_b=3, degenerate=False):
    p = _p3(batch_b=batch_b, degenerate=degenerate)
    for row in p["scenario"]["rows"]:
        ref, b3 = _recs(64), _recs(64)
        for r in b3:
            r["lp"] -= drift
        if div_at is not None:
            ref[div_at] = _rec(div_at, gap=ref_gap_at_div)
            b3[div_at] = {"t": 5000, "lp": -0.6, "top": [[5000, -0.6]]}
        row["logprobs"] = {"alone_restored": ref, "batched": b3}
    return p


class TestEvalP3WithLogprobs:
    def test_noise_divergence_passes_even_inside_prefix(self):
        p = _p3lp(ref_gap_at_div=0.2, div_at=10)
        p["scenario"]["rows"][0]["batched"]["text"] = "totally different"
        r = bm.eval_p3([p], [])
        assert r["verdict"] == "PASS"
        d = r["fork"]["detail"][0]
        assert d["classification"] == "noise"
        assert d["margin"] == pytest.approx(0.2)
        assert d["div_token"] == 10

    def test_large_margin_divergence_fails(self):
        r = bm.eval_p3([_p3lp(ref_gap_at_div=4.0, div_at=10)], [])
        assert r["verdict"] == "FAIL"
        assert any("noise" in c["name"] and not c["ok"] for c in r["checks"])

    def test_inconclusive_margin_fails(self):
        assert bm.eval_p3([_p3lp(ref_gap_at_div=1.0)], [])["verdict"] == "FAIL"

    def test_divergence_after_token_32_is_reported_not_judged(self):
        r = bm.eval_p3([_p3lp(ref_gap_at_div=4.0, div_at=40)], [])
        assert r["verdict"] == "PASS"
        assert r["fork"]["detail"][0]["classification"] == "bug"

    def test_no_divergence_passes(self):
        assert bm.eval_p3([_p3lp(div_at=None)], [])["verdict"] == "PASS"

    def test_still_requires_batch_and_no_degeneration(self):
        assert bm.eval_p3([_p3lp(batch_b=2)], [])["verdict"] == "FAIL"
        assert bm.eval_p3([_p3lp(degenerate=True)], [])["verdict"] == "FAIL"

    def test_controls_are_reported(self):
        p = _p3lp(div_at=None)
        for row in p["scenario"]["rows"]:
            row["logprobs"]["alone_restored2"] = _recs(64)
            rg = _recs(64)
            rg[7] = _rec(7, gap=0.1)
            rg[7] = {"t": 4242, "lp": -1.0, "top": [[4242, -1.0]]}
            row["logprobs"]["ragged_b1"] = rg
        r = bm.eval_p3([p], [])
        ctl = r["fork"]["detail"][0]["controls"]
        assert ctl["rerun"]["div_index"] is None
        assert ctl["ragged_b1"]["div_index"] == 7
        assert r["verdict"] == "PASS"  # controls are informational


def _p4(
    serial_wall,
    conc_wall,
    decs_serial=(20.0, 20.0, 20.0),
    rates_conc=(10.0, 10.0, 10.0),
    aborts=0,
    ok=True,
    start_offsets=(0.0, 0.0, 0.0),
):
    log = bm.parse_server_log("")
    log["aborts_total"] = aborts
    conc = [
        _req(
            dec=r,
            ok=ok,
            completion_tokens=201,
            chunk_times=_chunks(off, r, 201),
        )
        for r, off in zip(rates_conc, start_offsets)
    ]
    return _run(
        {
            "serial": {
                "wall_s": serial_wall,
                "requests": [_req(dec=d, ok=ok) for d in decs_serial],
            },
            "concurrent": {"wall_s": conc_wall, "requests": conc},
        },
        log=log,
    )


class TestEvalP4:
    def test_pass(self):
        r = bm.eval_p4([_p4(600, 400)], [_p4(600, 500)])
        assert r["verdict"] == "PASS"
        assert r["fork"]["aggregate_decode_tps"] == pytest.approx(30.0, rel=0.02)

    def test_equal_wall_passes(self):
        assert bm.eval_p4([_p4(600, 600)], [])["verdict"] == "PASS"

    def test_fail_when_concurrent_slower(self):
        assert bm.eval_p4([_p4(600, 650)], [])["verdict"] == "FAIL"

    def test_fail_on_abort(self):
        assert bm.eval_p4([_p4(600, 400, aborts=1)], [])["verdict"] == "FAIL"

    def test_fail_when_aggregate_below_single_row(self):
        r = bm.eval_p4([_p4(600, 400, rates_conc=(5.0, 5.0, 5.0))], [])
        assert r["verdict"] == "FAIL"
        assert any("aggregate" in c["name"] and not c["ok"] for c in r["checks"])

    def test_per_row_ratio_is_information_only(self):
        r = bm.eval_p4([_p4(600, 400, rates_conc=(10.0, 10.0, 10.0))], [])
        assert r["verdict"] == "PASS"
        assert r["row_ratios"] == [0.5, 0.5, 0.5]
        assert all("per-row" not in c["name"] for c in r["checks"])

    def test_fail_without_common_decode_window(self):
        r = bm.eval_p4([_p4(600, 400, start_offsets=(0.0, 0.0, 500.0))], [])
        assert r["verdict"] == "FAIL"
        assert any("no window" in c["detail"] for c in r["checks"])

    def test_fail_when_request_failed(self):
        assert bm.eval_p4([_p4(600, 400, ok=False)], [])["verdict"] == "FAIL"


class TestEvalP6Attribution:
    def test_spread_and_attribution(self):
        up = [_run({"request": _req(p, 20)}) for p in (990, 1000, 1010)]
        off = [_run({"request": _req(p, 20)}) for p in (940, 950, 960)]
        fk = [_run({"request": _req(p, 20)}) for p in (900, 910, 920)]
        r = bm.eval_p6(up, fk, off)
        assert r["fork"]["prefill_tps_min"] == 900
        assert r["fork"]["prefill_tps_max"] == 920
        assert r["fork_off"]["prefill_tps"] == 950
        a = r["attribution"]
        assert a["prefill_pct_forkoff_vs_upstream"] == pytest.approx(-5.0)
        assert a["prefill_pct_fork_vs_forkoff"] == pytest.approx(-4.2105, abs=0.01)
        assert r["verdict"] == "FAIL"  # verdict stays fork vs upstream

    def test_without_fork_off(self):
        up = [_run({"request": _req(1000, 20)})]
        fk = [_run({"request": _req(1000, 20)})]
        r = bm.eval_p6(up, fk)
        assert r["fork_off"] is None and r["attribution"] == {}


class TestPromptSizing:
    def test_scale_lengths_for_smoke(self):
        assert bm.scale_tokens(110_000, smoke=False) == 110_000
        assert 100 <= bm.scale_tokens(20_000, smoke=True) <= 1000
        assert bm.scale_tokens(135_000, smoke=True) > bm.scale_tokens(
            20_000, smoke=True
        )

    def test_nonce_prefix_unique(self):
        a = bm.make_nonce("seed-a")
        b = bm.make_nonce("seed-b")
        assert a != b and a.startswith("[bench-nonce ")

    def test_nonce_is_deterministic_per_seed(self):
        assert bm.make_nonce("x") == bm.make_nonce("x")

    def test_build_prompt_prefix_and_target(self):
        ids = list(range(1000))
        text = bm.build_prompt_text(
            "N:", ids, start=10, n_tokens=100, decode=lambda xs: ",".join(map(str, xs))
        )
        assert text.startswith("N:")
        assert text.count(",") == 99  # 100 ids joined

    def test_build_prompt_wraps_corpus(self):
        ids = list(range(10))
        text = bm.build_prompt_text(
            "", ids, start=8, n_tokens=5, decode=lambda xs: ",".join(map(str, xs))
        )
        assert text == "8,9,0,1,2"


class TestSafety:
    def test_cache_dir_guard(self):
        root = "/Volumes/ExternalSSD/oLMXCache-bench"
        assert bm.is_safe_bench_path(f"{root}/fork", [root])
        assert not bm.is_safe_bench_path(
            "/Volumes/ExternalSSD/oLMXCache", [root]
        )
        assert not bm.is_safe_bench_path(f"{root}/../oLMXCache", [root])
        assert not bm.is_safe_bench_path(root, [root])  # never the root itself

    def test_refuses_protected_port(self):
        assert bm.port_allowed(8002, allow=[])
        assert not bm.port_allowed(8001, allow=[])
        assert bm.port_allowed(8001, allow=[8001])


import bench_premises as bp  # noqa: E402

REAL = {
    "server": {"port": 8001},
    "cache": {"ssd_cache_dir": "/real", "enabled": True},
    "memory": {"memory_guard_tier": "balanced", "mlx_cache_limit_gb": 16.0},
    "scheduler": {
        "max_concurrent_requests": 3,
        "prefill_progress_commit_blocks": 4,
        "memory_pressure_park": True,
        "max_parks_per_request": 3,
        "park_resume_ok_polls": 3,
        "max_batched_decode_context_tokens": 200000,
    },
    "auth": {"api_key": "k"},
}


class TestVariantConfig:
    def test_env_fork(self):
        env = bp.variant_env("fork", "P6")
        assert env == {
            "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "4",
            "OMLX_MEMORY_PRESSURE_PARK": "true",
            "OMLX_MLX_CACHE_LIMIT_GB": "16",
            "OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS": "200000",
        }

    def test_env_fork_p3_disables_batch_gate(self):
        assert (
            bp.variant_env("fork", "P3")["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"]
            == "0"
        )
        assert (
            bp.variant_env("fork", "P4")["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"]
            == "200000"
        )

    def test_env_upstream_empty_and_fork_off_all_off(self):
        assert bp.variant_env("upstream", "P2") == {}
        off = bp.variant_env("fork-off", "P2")
        assert off["OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS"] == "0"
        assert off["OMLX_MEMORY_PRESSURE_PARK"] == "false"
        assert off["OMLX_MLX_CACHE_LIMIT_GB"] == "0"
        assert off["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] == "0"

    def test_settings_isolated_and_real_untouched(self):
        import copy

        real = copy.deepcopy(REAL)
        s = bp.build_settings(
            real, "fork", 8002, "/bench/cache", 2, bp.variant_env("fork", "P2")
        )
        assert real == REAL
        assert s["server"]["port"] == 8002
        assert s["cache"]["ssd_cache_dir"] == "/bench/cache"
        assert s["memory"]["memory_guard_tier"] == "aggressive"
        assert s["scheduler"]["max_concurrent_requests"] == 2
        assert s["scheduler"]["prefill_progress_commit_blocks"] == 4
        assert s["memory"]["mlx_cache_limit_gb"] == 16.0

    def test_upstream_settings_drop_fork_only_keys(self):
        s = bp.build_settings(REAL, "upstream", 8002, "/c", 1, {})
        assert "prefill_progress_commit_blocks" not in s["scheduler"]
        assert "max_batched_decode_context_tokens" not in s["scheduler"]
        assert "mlx_cache_limit_gb" not in s["memory"]

    def test_fork_off_settings_all_off(self):
        s = bp.build_settings(
            REAL, "fork-off", 8002, "/c", 1, bp.variant_env("fork-off", "P6")
        )
        assert s["scheduler"]["prefill_progress_commit_blocks"] == 0
        assert s["scheduler"]["memory_pressure_park"] is False
        assert s["scheduler"]["max_batched_decode_context_tokens"] == 0
        assert s["memory"]["mlx_cache_limit_gb"] == 0.0


class TestPerFeatureVariants:
    def test_d1_only(self):
        env = bp.variant_env("fork-d1only", "P6")
        assert env["OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS"] == "4"
        assert env["OMLX_MEMORY_PRESSURE_PARK"] == "false"
        assert env["OMLX_MLX_CACHE_LIMIT_GB"] == "0"
        assert env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] == "0"

    def test_cachelimit_only(self):
        env = bp.variant_env("fork-cachelimit-only", "P6")
        assert env["OMLX_MLX_CACHE_LIMIT_GB"] == "16"
        assert env["OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS"] == "0"

    def test_fork_off_p3_batching_allowed(self):
        assert (
            bp.variant_env("fork-off", "P3")["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"]
            == "0"
        )

    def test_variant_names_include_all(self):
        assert {"upstream", "fork", "fork-off", "fork-d1only"} <= set(bp.VARIANTS)

    def test_p3_plan_cold_then_restored_then_batched(self):
        assert bp.P3_PHASES == ("cold", "alone_restored", "batched")


def test_fork_overlay_keeps_app_kernels_and_takes_fork_sources(tmp_path):
    import bench_premises as bp

    app = tmp_path / "Resources"
    (app / "omlx/custom_kernels/k").mkdir(parents=True)
    (app / "omlx/custom_kernels/k/_ext.so").write_text("compiled")
    (app / "omlx/a.py").write_text("app")
    (app / "Python").mkdir()
    fork = tmp_path / "fork"
    (fork / "omlx").mkdir(parents=True)
    (fork / "omlx/a.py").write_text("fork")
    (fork / "omlx/b.py").write_text("new")
    root = tmp_path / "overlay"
    launcher = bp.build_fork_overlay(app, fork, root)
    launcher = bp.build_fork_overlay(app, fork, root)  # idempotent
    assert (root / "omlx/custom_kernels/k/_ext.so").read_text() == "compiled"
    assert (root / "omlx/a.py").read_text() == "fork"
    assert (root / "omlx/b.py").read_text() == "new"
    assert "omlx.cli" in launcher.read_text()


class TestBenchOverrides:
    def test_extra_env_parsed_and_restricted_to_omlx_keys(self):
        import bench_premises as bp

        env = {"OMLX_BENCH_EXTRA_ENV": "OMLX_A=1, OMLX_B=x=y"}
        assert bp.bench_extra_env(env) == {"OMLX_A": "1", "OMLX_B": "x=y"}
        assert bp.bench_extra_env({}) == {}
        with pytest.raises(ValueError):
            bp.bench_extra_env({"OMLX_BENCH_EXTRA_ENV": "PATH=/x"})
        with pytest.raises(ValueError):
            bp.bench_extra_env({"OMLX_BENCH_EXTRA_ENV": "OMLX_A"})

    def test_fork_repo_defaults_to_harness_tree(self, tmp_path):
        import bench_premises as bp

        assert bp.bench_fork_repo({}) == bp.REPO
        (tmp_path / "omlx").mkdir()
        got = bp.bench_fork_repo({"OMLX_BENCH_FORK_REPO": str(tmp_path)})
        assert got == tmp_path
        with pytest.raises(ValueError):
            bp.bench_fork_repo({"OMLX_BENCH_FORK_REPO": str(tmp_path / "nope")})

    def test_server_merges_extra_env(self, monkeypatch, tmp_path):
        import bench_premises as bp

        monkeypatch.setenv("OMLX_BENCH_EXTRA_ENV", "OMLX_MID_SYSTEM_POLICY=inline_note")
        srv = bp.Server("fork", "REPLAY", 8002, 1, tmp_path, False)
        assert srv.env["OMLX_MID_SYSTEM_POLICY"] == "inline_note"
        assert srv.env["OMLX_MLX_CACHE_LIMIT_GB"] == "16"


def test_fork_overlay_refuses_to_delete_foreign_dir(tmp_path):
    import bench_premises as bp
    import pytest

    foreign = tmp_path / "x"
    foreign.mkdir()
    with pytest.raises(RuntimeError):
        bp.build_fork_overlay(tmp_path, tmp_path, foreign)


class TestLogprobDumpHelpers:
    def test_load_dump_parses_jsonl_and_skips_torn_last_line(self, tmp_path):
        import json

        f = tmp_path / "r.jsonl"
        f.write_text(
            json.dumps({"pt": 5, "t": 1, "lp": -0.1, "top": [[1, -0.1]]})
            + "\n"
            + json.dumps({"pt": 5, "t": 2, "lp": -0.2, "top": [[2, -0.2]]})
            + '\n{"pt": 5, "t"'
        )
        recs = bm.load_logprob_dump(f)
        assert [r["t"] for r in recs] == [1, 2]

    def test_new_dumps_returns_only_files_added_since(self, tmp_path):
        (tmp_path / "a.jsonl").write_text("{}\n")
        before = bm.dump_listing(tmp_path)
        (tmp_path / "b.jsonl").write_text("{}\n")
        assert [p.name for p in bm.new_dumps(tmp_path, before)] == ["b.jsonl"]

    def test_pick_dump_by_prompt_tokens(self, tmp_path):
        import json

        for name, pt in (("x", 100), ("y", 250)):
            (tmp_path / f"{name}.jsonl").write_text(
                json.dumps({"pt": pt, "t": 1, "lp": 0.0, "top": []}) + "\n"
            )
        paths = sorted(tmp_path.iterdir())
        assert bm.pick_dump(paths, 250).name == "y.jsonl"
        assert bm.pick_dump(paths, 999) is None


class TestP3DebugEnv:
    def test_fork_p3_gets_dump_and_ragged_flag_paths(self, tmp_path):
        import bench_premises as bp

        env = bp.debug_env("fork", "P3", tmp_path)
        assert env["OMLX_DEBUG_LOGPROB_DUMP"].startswith(str(tmp_path))
        assert env["OMLX_DEBUG_FORCE_RAGGED_B1"].startswith(str(tmp_path))

    def test_upstream_and_other_premises_get_none(self, tmp_path):
        import bench_premises as bp

        assert bp.debug_env("upstream", "P3", tmp_path) == {}
        assert bp.debug_env("fork", "P4", tmp_path) == {}

    def test_p3_control_phases(self):
        import bench_premises as bp

        assert bp.P3_CONTROL_PHASES == ("alone_restored2", "ragged_b1")


class TestReportP3Margins:
    def test_margins_and_controls_tables_render(self):
        import report

        p = _p3lp(ref_gap_at_div=0.2, div_at=10)
        for row in p["scenario"]["rows"]:
            row["logprobs"]["alone_restored2"] = _recs(64)
        md = report.section_p3({"fork": {"P3": [p]}})
        assert "Near-tie analysis" in md
        assert "0.200 nats" in md
        assert "noise" in md
        assert "Controls" in md and "restored again" in md

    def test_no_margins_without_logprobs(self):
        import report

        assert "Near-tie" not in report.section_p3({"fork": {"P3": [_p3()]}})


class TestNoMtpVariant:
    def test_variant_exists_and_uses_fork_on_env(self):
        import bench_premises as bp

        assert "fork-nomtp" in bp.VARIANTS
        assert bp.variant_env("fork-nomtp", "P3") == bp.variant_env("fork", "P3")

    def test_model_settings_turn_mtp_off_only_for_the_bench_model(self):
        import bench_premises as bp

        ms = {
            "version": 1,
            "models": {
                bp.MODEL: {"mtp_enabled": True, "x": 1},
                "other": {"mtp_enabled": True},
            },
        }
        out = bp.variant_model_settings(ms, "fork-nomtp")
        assert out["models"][bp.MODEL]["mtp_enabled"] is False
        assert out["models"][bp.MODEL]["x"] == 1
        assert out["models"]["other"]["mtp_enabled"] is True
        assert ms["models"][bp.MODEL]["mtp_enabled"] is True  # input untouched

    def test_other_variants_keep_model_settings(self):
        import bench_premises as bp

        ms = {"models": {bp.MODEL: {"mtp_enabled": True}}}
        assert bp.variant_model_settings(ms, "fork") == ms

    def test_report_renders_nomtp_control(self):
        import report

        p = _p3lp(ref_gap_at_div=0.2, div_at=10)
        md = report.section_p3({"fork": {"P3": [p]}, "fork-nomtp": {"P3": [p]}})
        assert "MTP off" in md


def _with_rerun(p, identical):
    for row in p["scenario"]["rows"]:
        rr = _recs(64)
        if not identical:
            rr[20] = {"t": 777, "lp": -1.0, "top": [[777, -1.0]]}
        row["logprobs"]["alone_restored2"] = rr
    return p


class TestEvalP3MtpAttribution:
    def test_batching_noise_with_irreproducible_mtp_reference_passes(self):
        mtp_on = _with_rerun(_p3lp(ref_gap_at_div=1.0, div_at=8), identical=False)
        off = _p3lp(ref_gap_at_div=0.0, div_at=40)
        r = bm.eval_p3([mtp_on], [], nomtp=[off])
        assert r["verdict"] == "PASS"
        names = [c["name"] for c in r["checks"]]
        assert any("MTP off" in n for n in names)
        assert r["nomtp"]["detail"][0]["classification"] == "noise"

    def test_reproducible_mtp_reference_is_judged_on_its_own(self):
        mtp_on = _with_rerun(_p3lp(ref_gap_at_div=1.0, div_at=8), identical=True)
        off = _p3lp(ref_gap_at_div=0.0, div_at=40)
        assert bm.eval_p3([mtp_on], [], nomtp=[off])["verdict"] == "FAIL"

    def test_batching_bug_with_mtp_off_fails(self):
        mtp_on = _with_rerun(_p3lp(ref_gap_at_div=0.0, div_at=40), identical=False)
        off = _p3lp(ref_gap_at_div=4.0, div_at=10)
        assert bm.eval_p3([mtp_on], [], nomtp=[off])["verdict"] == "FAIL"

    def test_mtp_off_batch_must_be_exercised(self):
        mtp_on = _with_rerun(_p3lp(div_at=None), identical=True)
        off = _p3lp(div_at=None, batch_b=2)
        assert bm.eval_p3([mtp_on], [], nomtp=[off])["verdict"] == "FAIL"

    def test_without_nomtp_behaviour_is_unchanged(self):
        mtp_on = _with_rerun(_p3lp(ref_gap_at_div=1.0, div_at=8), identical=False)
        assert bm.eval_p3([mtp_on], [])["verdict"] == "FAIL"


class TestReportRaggedControlEngaged:
    def _md(self, logged):
        import report

        p = _p3lp(div_at=None)
        p["scenario"]["ragged_b1_forced_logged"] = logged
        for row in p["scenario"]["rows"]:
            row["logprobs"]["ragged_b1"] = _recs(64)
        return report.section_p3({"fork": {"P3": [p]}})

    def test_engaged(self):
        assert "engaged (server log): yes" in self._md(True)

    def test_not_engaged_is_flagged_invalid(self):
        assert "not valid" in self._md(False)


class TestClassifyP5Session:
    GUARD = (
        "oMLX memory guard aborted this request mid-prefill: Request aborted: "
        "process memory limit exceeded (usage 106.2 GB)"
    )
    # Normal answer that merely talks about the guard (the original false positive).
    TALKS = (
        "The process_memory_enforcer module enforces a memory limit exceeded "
        "policy and aborted requests are reported with prefill_memory_exceeded. "
    ) * 5

    def test_answer_that_discusses_memory_code_is_success(self):
        r = bm.classify_p5_session(self.TALKS, "Warning: x\n", 0)
        assert r["ok"] is True
        assert r["memory_failure"] is False
        assert r["recovered"] is False

    def test_guard_abort_in_stderr_is_failure(self):
        r = bm.classify_p5_session("", self.GUARD, 1)
        assert r["ok"] is False
        assert r["memory_failure"] is True
        assert r["memory_signature"] is True

    def test_error_line_on_stdout_is_failure(self):
        out = "Error: prefill_memory_exceeded for request 12\n"
        r = bm.classify_p5_session(out, "", 1)
        assert r["memory_failure"] is True and r["ok"] is False

    def test_signature_mid_line_on_stdout_is_not_a_failure(self):
        out = "x" * 300 + "\nthe text says Error: prefill_memory_exceeded here"
        r = bm.classify_p5_session(out, "", 0)
        assert r["memory_failure"] is False and r["ok"] is True

    @pytest.mark.parametrize(
        "sig",
        [
            "process memory limit exceeded",
            "prefill_memory_exceeded",
            "prefill_memory_aborted",
            "memory_admission_stalled",
            "Model 'Qwen3.8' is busy",
        ],
    )
    def test_each_signature_detected(self, sig):
        assert bm.classify_p5_session("", f"boom {sig}", 1)["memory_failure"]

    def test_recovered_when_signature_but_full_output_exit_zero(self):
        r = bm.classify_p5_session("a" * 400, self.GUARD, 0)
        assert r["recovered"] is True
        assert r["memory_failure"] is False
        assert r["ok"] is True

    def test_exit_nonzero_or_short_output_is_not_ok(self):
        assert bm.classify_p5_session("a" * 400, "", 1)["ok"] is False
        assert bm.classify_p5_session("a" * 50, "", 0)["ok"] is False
        assert bm.classify_p5_session("a" * 400, "", None)["ok"] is False


def _p5s(variant, mode, completed, manual, n=3):
    return {
        "variant": variant,
        "mode": mode,
        "sessions": n,
        "completed": completed,
        "manual_interventions": manual,
        "rows": [],
    }


class TestEvalP5:
    def test_pass_when_fork_with_clean_and_baseline_fails(self):
        v = bm.eval_p5(_p5s("fork", "with", 3, 0), _p5s("upstream", "without", 0, 3))
        assert v["verdict"] == "PASS"

    def test_fail_when_fork_with_needs_a_human(self):
        v = bm.eval_p5(_p5s("fork", "with", 2, 1), _p5s("upstream", "without", 0, 3))
        assert v["verdict"] == "FAIL"

    def test_inconclusive_when_baseline_never_failed(self):
        v = bm.eval_p5(_p5s("fork", "with", 3, 0), _p5s("upstream", "without", 3, 0))
        assert v["verdict"] != "PASS"

    def test_missing_summary_is_inconclusive(self):
        assert bm.eval_p5(None, None)["verdict"] == "INCONCLUSIVE"


# --- PL (batched-decode limit probe, T13) -----------------------------------

PL_LOG = """\
2026-10-03 10:00:00,000 - omlx.scheduler - INFO - [-] - Holding r2 from batched decode: combined context 260000 > limit 200000 (single-row decode keeps fused kernels)
2026-10-03 10:00:01,000 - omlx.scheduler - INFO - [-] - Backfilling r3 past held r2
2026-10-03 10:00:02,000 - omlx.scheduler - INFO - [-] - Aging: r2 held 120 s, backfill paused
2026-10-03 10:00:03,000 - omlx.scheduler - INFO - [-] - Releasing r2 to decode
"""


class TestPlLog:
    def test_backfill_and_aging_counts(self):
        m = bm.parse_server_log(PL_LOG)
        assert m["holds"] == 1
        assert m["backfills"] == 1
        assert m["agings"] == 1

    def test_empty_log_zero(self):
        m = bm.parse_server_log("")
        assert m["backfills"] == 0 and m["agings"] == 0


class TestPlEnv:
    def test_pl_env_override(self):
        env = bp.variant_env("fork", "PL", batched_limit=300000)
        assert env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] == "300000"

    def test_zero_override_is_kept(self):
        env = bp.variant_env("fork", "PL", batched_limit=0)
        assert env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] == "0"

    def test_no_override_keeps_default(self):
        env = bp.variant_env("fork", "PL")
        assert env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] == "200000"

    def test_pl_registered_with_two_slots(self):
        assert bp.PREMISE_CONCURRENCY["PL"] == 2
        assert "PL" in bp.SCENARIOS
        assert "PL" not in bp.ALL_PREMISES

    def test_override_reaches_settings(self):
        env = bp.variant_env("fork", "PL", batched_limit=300000)
        s = bp.build_settings(REAL, "fork", 8002, "/c", 2, env)
        assert s["scheduler"]["max_batched_decode_context_tokens"] == 300000


def _pl_req(name, start, first, end, n, ctoks, text="a b c d e f g h i j k l"):
    times = [first + (end - first) * i / (n - 1) for i in range(n)]
    return {
        "name": name,
        "ok": True,
        "start_s": start,
        "end_s": end,
        "prompt_tokens": 100,
        "completion_tokens": ctoks,
        "decode_tps": (ctoks - 1) / (end - first),
        "chunk_times": times,
        "text": text,
    }


def _pl_run(requests, wall=100.0, peak=70.0, aborts=0, log=None):
    return {
        "ok": True,
        "wall_s": wall,
        "scenario": {"requests": requests},
        "log": {
            "aborts_total": aborts,
            "holds": 0,
            "backfills": 0,
            "agings": 0,
            "batch_max_b": 2,
            **(log or {}),
        },
        "samples": {"footprint_peak_gb": peak},
    }


class TestPlSummary:
    def test_row_fields(self):
        reqs = [
            _pl_req("165k", 0, 10, 60, 101, 1000),
            _pl_req("95k", 0, 20, 70, 101, 1000),
        ]
        row = bm.pl_row(_pl_run(reqs, wall=72.0, peak=71.5, log={"holds": 3}))
        assert row["wall_s"] == 72.0
        assert row["peak_gb"] == 71.5
        assert row["holds"] == 3
        assert row["max_b"] == 2
        assert row["aborts"] == 0
        assert row["req_end_s"] == [60, 70]
        assert [round(t, 1) for t in row["req_decode_tps"]] == [20.0, 20.0]
        # window = [20, 60]; each row decodes 1000 tok / 50 s = 20 tok/s
        assert row["aggregate_tps"] == pytest.approx(40.0, rel=0.1)
        assert row["degenerate"] is False

    def test_degenerate_flag_on_repetition(self):
        loop = "x y z w " * 50
        reqs = [
            _pl_req("165k", 0, 10, 60, 101, 1000, text=loop),
            _pl_req("95k", 0, 20, 70, 101, 1000),
        ]
        assert bm.pl_row(_pl_run(reqs))["degenerate"] is True

    def test_no_overlap_gives_none_aggregate(self):
        reqs = [
            _pl_req("165k", 0, 10, 30, 21, 100),
            _pl_req("95k", 0, 40, 60, 21, 100),
        ]
        assert bm.pl_row(_pl_run(reqs))["aggregate_tps"] is None

    def test_failed_request_is_not_ok(self):
        reqs = [_pl_req("165k", 0, 10, 60, 101, 1000), {"name": "95k", "ok": False}]
        assert bm.pl_row(_pl_run(reqs))["ok"] is False


# --------------------------------------------------------------------------
# P7: park-instead-of-abort validation (text vs image-bearing prompts)
# --------------------------------------------------------------------------

P7_LOG = """\
2026-10-05 04:00:01,000 - omlx.process_memory_enforcer - INFO - [-] - Memory pressure level: ok -> soft (current=29.0GB, soft=27.4GB, hard=29.5GB, ceiling=30.4GB)
2026-10-05 04:00:05,000 - omlx.process_memory_enforcer - INFO - [-] - Memory pressure level: soft -> hard (current=29.7GB, soft=27.4GB, hard=29.5GB, ceiling=30.4GB)
2026-10-05 04:00:05,100 - omlx.process_memory_enforcer - INFO - [-] - Hard memory pressure: parking a prefill victim on 1 scheduler(s) instead of aborting (poll 1/30)
2026-10-05 04:00:05,300 - omlx.scheduler - INFO - [-] - Parked request rid-aaa at 24576 tokens committed (park 1/3), reason: memory pressure
2026-10-05 04:00:06,200 - omlx.scheduler - INFO - [-] - Park skipped for rid-bbb: decoding, not in chunked prefill (memory pressure falls back to abort)
2026-10-05 04:00:35,300 - omlx.scheduler - INFO - [-] - Resumed parked request rid-aaa: cached=24576
2026-10-05 04:00:40,000 - omlx.engine_core - WARNING - [-] - Aborted 2 requests due to memory pressure
"""


class TestParseParkEvents:
    def test_extracts_each_event_kind_with_timestamps(self):
        ev = bm.parse_park_events(P7_LOG)
        kinds = [e["kind"] for e in ev]
        assert kinds == [
            "pressure",
            "pressure",
            "hard_park_poll",
            "parked",
            "park_skipped",
            "resumed",
            "abort",
        ]
        parked = ev[3]
        assert parked["rid"] == "rid-aaa"
        assert parked["committed"] == 24576
        assert parked["park_n"] == 1 and parked["budget"] == 3
        assert parked["reason"] == "memory pressure"
        assert ev[2]["poll"] == 1 and ev[2]["max"] == 30
        assert ev[4]["rid"] == "rid-bbb" and "decoding" in ev[4]["reason"]
        assert ev[5]["cached"] == 24576
        assert ev[6]["n"] == 2
        assert ev[1]["to"] == "hard" and ev[1]["current_gb"] == 29.7
        assert ev[5]["ts"] - ev[3]["ts"] == pytest.approx(30.0, abs=0.01)

    def test_empty_and_unrelated_lines(self):
        assert bm.parse_park_events("") == []
        assert bm.parse_park_events("2026-10-05 04:00:01,000 - x - INFO - hello\n") == []

    def test_tolerates_unparsable_timestamp(self):
        ev = bm.parse_park_events("Resumed parked request r1: cached=5\n")
        assert ev == [{"kind": "resumed", "ts": None, "rid": "r1", "cached": 5}]


class TestParkTracks:
    def test_tracks_pair_parks_with_resumes_and_time_parked(self):
        tr = bm.build_park_tracks(bm.parse_park_events(P7_LOG))
        a = tr["rid-aaa"]
        assert a["parks"] == 1
        assert a["parked_tokens"] == [24576]
        assert a["resume_cached"] == [24576]
        assert a["parked_s"] == pytest.approx(30.0, abs=0.01)
        assert a["unresumed"] == 0
        assert "rid-bbb" not in tr  # a skip is not a park

    def test_unresumed_park_is_counted_and_has_no_time(self):
        log = (
            "2026-10-05 04:00:05,000 - s - INFO - [-] - Parked request r9 at 100 "
            "tokens committed (park 1/3), reason: x\n"
        )
        a = bm.build_park_tracks(bm.parse_park_events(log))["r9"]
        assert a["unresumed"] == 1
        assert a["parked_s"] is None


class TestAttributeRids:
    def test_matches_by_resume_cached_equal_to_client_cached(self):
        tracks = bm.build_park_tracks(bm.parse_park_events(P7_LOG))
        reqs = [
            {"name": "r0", "cached_tokens": 0},
            {"name": "r1", "cached_tokens": 24576},
        ]
        assert bm.attribute_rids(tracks, reqs) == {"r1": "rid-aaa"}

    def test_ambiguous_match_is_left_unattributed(self):
        tracks = {
            "a": {"resume_cached": [100]},
            "b": {"resume_cached": [100]},
        }
        reqs = [{"name": "r0", "cached_tokens": 100}]
        assert bm.attribute_rids(tracks, reqs) == {}


def _p7req(name, ok=True, cached=0, error=None, text="a b c d e f g h", ttft=10.0):
    return {
        "name": name,
        "ok": ok,
        "error": error,
        "cached_tokens": cached,
        "prompt_tokens": 100_000,
        "completion_tokens": 8,
        "start_s": 0.0,
        "end_s": 60.0,
        "ttft_s": ttft,
        "text": text,
    }


class TestSummarizeP7:
    def test_park_fired_all_ok(self):
        reqs = [_p7req("r0"), _p7req("r1", cached=24576), _p7req("r2")]
        s = bm.summarize_p7(reqs, {}, bm.parse_park_events(P7_LOG))
        assert s["parks"] == 1
        assert s["resumes"] == 1
        assert s["park_skips"] == 1
        assert s["hard_pressure_seen"] is True
        assert s["rows"][1]["rid"] == "rid-aaa"
        assert s["rows"][1]["parks"] == 1
        assert s["rows"][1]["parked_s"] == pytest.approx(30.0, abs=0.01)
        assert s["rows"][0]["parks"] == 0
        assert s["all_final_ok"] is True
        assert s["verdict"] == "park_fired_ok"
        assert s["aborts_logged"] == 2

    def test_reissue_recorded_for_failed_request(self):
        reqs = [_p7req("r0", ok=False, error="aborted"), _p7req("r1")]
        re_ = {"r0": _p7req("r0", cached=40_000, ttft=3.0)}
        s = bm.summarize_p7(reqs, re_, [])
        row = s["rows"][0]
        assert row["aborted"] is True
        assert row["reissue_ok"] is True
        assert row["reissue_cached"] == 40_000
        assert row["final_ok"] is True
        assert s["aborted_requests"] == 1
        assert s["reissue_cached"] == [40_000]
        assert s["all_final_ok"] is True

    def test_failed_reissue_makes_run_not_ok(self):
        reqs = [_p7req("r0", ok=False, error="aborted")]
        re_ = {"r0": _p7req("r0", ok=False, error="aborted again")}
        s = bm.summarize_p7(reqs, re_, [])
        assert s["all_final_ok"] is False

    def test_sanity_uses_distinct_4gram_ratio(self):
        loop = "x y z w " * 50
        s = bm.summarize_p7([_p7req("r0", text=loop)], {}, [])
        assert s["rows"][0]["distinct_4gram"] < 0.1
        assert s["rows"][0]["degenerate"] is True
        ok = bm.summarize_p7([_p7req("r0", text="one two three four five six")], {}, [])
        assert ok["rows"][0]["degenerate"] is False

    def test_text_is_not_kept_in_rows(self):
        s = bm.summarize_p7([_p7req("r0")], {}, [])
        assert "text" not in s["rows"][0]


class TestClassifyP7:
    def _s(self, **kw):
        base = {
            "parks": 0,
            "resumes": 0,
            "park_skips": 0,
            "hard_pressure_seen": False,
            "aborts_logged": 0,
            "aborted_requests": 0,
            "all_final_ok": True,
        }
        base.update(kw)
        return base

    def test_no_pressure(self):
        assert bm.classify_p7(self._s()) == "no_pressure"

    def test_park_fired_ok(self):
        s = self._s(parks=1, resumes=1, hard_pressure_seen=True)
        assert bm.classify_p7(s) == "park_fired_ok"

    def test_park_fired_with_aborts(self):
        s = self._s(parks=1, resumes=1, hard_pressure_seen=True, aborted_requests=1)
        assert bm.classify_p7(s) == "park_fired_with_aborts"

    def test_parked_not_resumed(self):
        s = self._s(parks=2, resumes=1, hard_pressure_seen=True)
        assert bm.classify_p7(s) == "parked_not_resumed"

    def test_park_skipped_then_abort(self):
        s = self._s(park_skips=2, hard_pressure_seen=True, aborted_requests=2)
        assert bm.classify_p7(s) == "park_skipped_then_abort"

    def test_park_skipped_survived(self):
        s = self._s(park_skips=1, hard_pressure_seen=True)
        assert bm.classify_p7(s) == "park_skipped_survived"

    def test_hard_pressure_without_park_path(self):
        s = self._s(hard_pressure_seen=True, aborts_logged=1, aborted_requests=1)
        assert bm.classify_p7(s) == "hard_pressure_aborted"
        s = self._s(hard_pressure_seen=True)
        assert bm.classify_p7(s) == "hard_pressure_survived"

    def test_abort_without_pressure_signal(self):
        s = self._s(aborted_requests=1)
        assert bm.classify_p7(s) == "aborted_without_pressure_log"


class TestP7Ceiling:
    def test_default_sits_between_model_and_measured_peak(self):
        c = bm.p7_default_ceiling_gb(model_gb=16.8, peak_gb=44.0)
        assert 16.8 < c < 44.0
        assert c == pytest.approx(30.4, abs=0.05)

    def test_rejects_peak_not_above_model(self):
        with pytest.raises(ValueError):
            bm.p7_default_ceiling_gb(model_gb=20.0, peak_gb=20.0)

    def test_settings_override_is_custom_tier_with_ceiling(self):
        o = bm.p7_settings_override(30.4)
        assert o == {
            "memory": {
                "memory_guard_tier": "custom",
                "memory_guard_custom_ceiling_gb": 30.4,
            }
        }

    def test_override_can_widen_the_hard_band(self):
        o = bm.p7_settings_override(42.0, hard_threshold=0.8, soft_threshold=0.7)
        assert o["memory"]["hard_threshold"] == 0.8
        assert o["memory"]["soft_threshold"] == 0.7
        assert o["memory"]["memory_guard_tier"] == "custom"

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError):
            bm.p7_settings_override(42.0, hard_threshold=0.7, soft_threshold=0.8)
        with pytest.raises(ValueError):
            bm.p7_settings_override(42.0, hard_threshold=1.2)

    def test_deep_merge_keeps_siblings_and_input(self):
        base = {"memory": {"memory_guard_tier": "aggressive", "x": 1}, "a": {"b": 1}}
        out = bm.deep_merge(base, bm.p7_settings_override(30.0))
        assert out["memory"]["memory_guard_tier"] == "custom"
        assert out["memory"]["x"] == 1
        assert out["a"] == {"b": 1}
        assert base["memory"]["memory_guard_tier"] == "aggressive"


class TestP7Content:
    def test_png_is_valid_and_distinct_per_seed(self):
        import struct
        import zlib

        a = bm.make_png(32, 24, "a")
        b = bm.make_png(32, 24, "b")
        assert a[:8] == b"\x89PNG\r\n\x1a\n"
        assert a != b
        assert bm.make_png(32, 24, "a") == a  # deterministic
        # IHDR then IDAT inflate to (1 + 3*w) * h bytes
        w, h = struct.unpack(">II", a[16:24])
        assert (w, h) == (32, 24)
        pos, idat = 8, b""
        while pos < len(a):
            n, typ = struct.unpack(">I4s", a[pos : pos + 8])
            if typ == b"IDAT":
                idat += a[pos + 8 : pos + 8 + n]
            pos += 12 + n
        assert len(zlib.decompress(idat)) == (1 + 3 * w) * h

    def test_text_mode_content_is_a_plain_string(self):
        assert bm.p7_content("hello", "text", "s") == "hello"

    def test_image_mode_puts_image_before_the_long_text(self):
        parts = bm.p7_content("LONG", "image", "s")
        assert [p["type"] for p in parts] == ["text", "image_url", "text"]
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert parts[2]["text"] == "LONG"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            bm.p7_content("x", "video", "s")


class TestP7Wiring:
    def test_p7_registered_with_concurrency_three(self):
        assert "P7" in bp.SCENARIOS
        assert bp.PREMISE_CONCURRENCY["P7"] == 3
        assert "P7" not in bp.ALL_PREMISES  # opt-in only, never part of "all"

    def test_scenario_ok_requires_every_request_to_finish(self):
        good = {"summary": {"all_final_ok": True}}
        bad = {"summary": {"all_final_ok": False}}
        assert bp.scenario_ok("P7", good) is True
        assert bp.scenario_ok("P7", bad) is False

    def test_server_applies_settings_override(self, tmp_path):
        env = bp.variant_env("fork", "P7")
        s = bp.build_settings(REAL, "fork", 8002, "/c", 3, env)
        out = bm.deep_merge(s, bm.p7_settings_override(31.0))
        assert out["memory"]["memory_guard_tier"] == "custom"
        assert out["memory"]["memory_guard_custom_ceiling_gb"] == 31.0
        assert out["memory"]["mlx_cache_limit_gb"] == 16.0  # untouched


class TestScenarioP7:
    def _ctx(self, fail_first, mode="text"):
        import argparse
        import asyncio

        calls = []

        class Srv:
            def log_size(self):
                return 0

            def log_since(self, off):
                return P7_LOG

        class Corpus:
            def prompt(self, seed, n):
                return f"PROMPT[{seed}:{n}]"

        class Ctx:
            server = Srv()
            corpus = Corpus()
            args = argparse.Namespace(
                p7_mode=mode,
                p7_requests=3,
                p7_stagger_s=0.0,
                p7_max_tokens=256,
                p7_ceiling_gb=30.4,
                settle_s=0.0,
                smoke=False,
            )

            def n(self, nominal):
                return nominal

            def mt(self, nominal):
                return nominal

            def seed(self, *parts):
                return "|".join(map(str, parts))

            async def chat(self, prompt, max_tokens, sampling=None, **kw):
                calls.append((kw["name"], prompt, max_tokens))
                name = kw["name"]
                failed = name in fail_first
                return {
                    "name": name,
                    "ok": not failed,
                    "error": "aborted" if failed else None,
                    "cached_tokens": 24576 if name == "req1" else 0,
                    "text": "alpha beta gamma delta epsilon zeta",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "ttft_s": 0.5,
                }

        return Ctx(), calls, asyncio

    def test_failed_request_is_reissued_once_with_same_prompt(self):
        ctx, calls, asyncio = self._ctx({"req2"})
        sc = asyncio.run(bp.scenario_p7(ctx))
        names = [c[0] for c in calls]
        assert names.count("req2-reissue") == 1
        orig = next(c for c in calls if c[0] == "req2")
        again = next(c for c in calls if c[0] == "req2-reissue")
        assert orig[1] == again[1]
        assert sc["summary"]["aborted_requests"] == 1
        assert bp.scenario_ok("P7", sc) is True
        assert all("text" not in r for r in sc["requests"])

    def test_distinct_prompts_and_image_mode_parts(self):
        ctx, calls, asyncio = self._ctx(set(), mode="image")
        asyncio.run(bp.scenario_p7(ctx))
        prompts = [c[1] for c in calls if c[0].startswith("req")]
        assert len(prompts) == 3
        assert all(p[1]["type"] == "image_url" for p in prompts)
        assert len({p[2]["text"] for p in prompts}) == 3
        assert len({p[1]["image_url"]["url"] for p in prompts}) == 3
        assert not any(n.endswith("-reissue") for n, _, _ in calls)


def test_collect_setup_records_p7_params_and_override():
    import argparse

    class Srv:
        variant, premise = "fork", "P7"
        settings = {}
        env = {}
        max_concurrent = 3
        cache_dir = "/c"
        settings_override = bm.p7_settings_override(30.4)

        def command(self):
            return ["x", "serve"]

    args = argparse.Namespace(
        smoke=False,
        p1_cancel_fraction=0.6,
        p3_max_tokens=1,
        p4_max_tokens=1,
        p7_mode="image",
        p7_ceiling_gb=30.4,
        p7_requests=3,
        p7_stagger_s=5.0,
        p7_max_tokens=256,
    )
    setup = bp.collect_setup(args, Srv())
    assert setup["p7"]["mode"] == "image"
    assert setup["settings_override"]["memory"]["memory_guard_tier"] == "custom"


# --------------------------------------------------------------------------
# C7: pause-resume cycle validation (decode pressure, images, never-fits)
# --------------------------------------------------------------------------

C7_LOG = """\
2026-10-05 10:00:01,000 - omlx.process_memory_enforcer - INFO - [-] - Memory pressure level: soft -> hard (current=34.0GB, soft=33.6GB, hard=33.6GB, ceiling=42.0GB)
2026-10-05 10:00:01,100 - omlx.process_memory_enforcer - WARNING - [-] - Hard memory pressure: pausing decoding victim rid-a instead of aborting (120000 tokens, ~9.5 GB) on 'm'; 2 request(s) kept
2026-10-05 10:00:02,000 - omlx.scheduler - INFO - [-] - Paused decoding request rid-a at 120640 tokens (prompt 120000 + output 640), stored exact prefix
2026-10-05 10:00:02,500 - omlx.process_memory_enforcer - INFO - [-] - Hard memory pressure: waiting for victim rid-a memory to drain on 'm' (poll 1/10, hard 34.0GB)
2026-10-05 10:00:03,000 - omlx.scheduler - INFO - [-] - Parked request rid-a waiting: need 12.3 GB, free 4.1 GB
2026-10-05 10:00:20,000 - omlx.scheduler - INFO - [-] - Resuming parked request rid-a: need 12.3 GB fits in free 13.0 GB
2026-10-05 10:00:21,000 - omlx.scheduler - INFO - [-] - Resumed decoding request rid-a: restored 120640 tokens exactly
2026-10-05 10:00:30,000 - omlx.process_memory_enforcer - WARNING - [-] - Hard memory pressure: aborted 1 victim rid-b (90000 tokens, ~7.0 GB) on 'm'; 1 request(s) kept
2026-10-05 10:00:40,000 - omlx.process_memory_enforcer - WARNING - [-] - Hard memory pressure: aborted 3 request(s) on 'm' and kept model loaded
2026-10-05 10:00:41,000 - omlx.scheduler - INFO - [-] - Parked request rid-c at 24576 tokens committed (park 1/3), reason: memory pressure
2026-10-05 10:00:42,000 - omlx.scheduler - INFO - [-] - Resumed parked request rid-c: cached=24576
"""


class TestParseC7Events:
    def test_each_new_line_becomes_one_event_in_order(self):
        ev = bm.parse_c7_events(C7_LOG)
        kinds = [e["kind"] for e in ev]
        assert kinds == [
            "pressure",
            "victim_pause",
            "decode_paused",
            "drain_wait",
            "park_wait",
            "resume_fit",
            "decode_resumed",
            "victim_abort",
            "abort_all",
            "parked",
            "resumed",
        ]

    def test_fields(self):
        ev = {e["kind"]: e for e in bm.parse_c7_events(C7_LOG)}
        assert ev["decode_paused"]["rid"] == "rid-a"
        assert ev["decode_paused"]["tokens"] == 120640
        assert ev["decode_paused"]["prompt"] == 120000
        assert ev["decode_paused"]["output"] == 640
        assert ev["decode_resumed"]["restored"] == 120640
        assert ev["victim_pause"]["rid"] == "rid-a"
        assert ev["victim_pause"]["tokens"] == 120000
        assert ev["victim_abort"]["rid"] == "rid-b"
        assert ev["abort_all"]["n"] == 3
        assert ev["drain_wait"]["rid"] == "rid-a"
        assert (ev["park_wait"]["need_gb"], ev["park_wait"]["free_gb"]) == (12.3, 4.1)
        assert (ev["resume_fit"]["need_gb"], ev["resume_fit"]["free_gb"]) == (12.3, 13.0)
        assert ev["decode_paused"]["ts"] is not None

    def test_old_parser_ignores_new_lines(self):
        assert [e["kind"] for e in bm.parse_park_events(C7_LOG)] == [
            "pressure",
            "parked",
            "resumed",
        ]

    def test_waiting_line_is_not_a_prefill_park(self):
        ev = bm.parse_c7_events(
            "x - Parked request r1 waiting: need 1.0 GB, free 0.5 GB\n"
        )
        assert [e["kind"] for e in ev] == ["park_wait"]


def _c7req(name, ptok, ok=True, text="a b c", times=None, **kw):
    return {
        "name": name,
        "ok": ok,
        "error": None if ok else "aborted",
        "prompt_tokens": ptok,
        "completion_tokens": 3,
        "cached_tokens": 0,
        "text": text,
        "start_s": 0.0,
        "end_s": 10.0,
        "ttft_s": 1.0,
        "chunk_times": times if times is not None else [1.0, 1.1, 1.2],
        **kw,
    }


class TestC7Attribution:
    def test_decode_pause_matched_by_unique_prompt_tokens(self):
        ev = bm.parse_c7_events(C7_LOG)
        reqs = [_c7req("req0", 120000), _c7req("req1", 90000)]
        got = bm.attribute_decode_pauses(ev, reqs)
        assert got["req0"] == {"pauses": 1, "resumes": 1, "restored": [120640]}
        assert "req1" not in got

    def test_ambiguous_prompt_tokens_are_left_out(self):
        ev = bm.parse_c7_events(C7_LOG)
        reqs = [_c7req("req0", 120000), _c7req("req1", 120000)]
        assert bm.attribute_decode_pauses(ev, reqs) == {}


class TestMaxDecodeStall:
    def test_largest_gap(self):
        assert bm.max_decode_stall([1.0, 1.1, 4.1, 4.2]) == pytest.approx(3.0)

    def test_none_when_fewer_than_two_chunks(self):
        assert bm.max_decode_stall([]) is None
        assert bm.max_decode_stall([1.0]) is None


class TestIdentityVerdict:
    def test_identical(self):
        v = bm.identity_verdict(_c7req("a", 1, text="x y z"), _c7req("b", 1, text="x y z"))
        assert v["identical"] is True
        assert v["first_divergence_chars"] is None
        assert v["classification"] is None

    def test_divergence_without_logprobs_is_unclassified(self):
        v = bm.identity_verdict(_c7req("a", 1, text="x y z"), _c7req("b", 1, text="x y q"))
        assert v["identical"] is False
        assert v["first_divergence_chars"] == 4
        assert v["classification"] == "unclassified"

    def test_near_tie_divergence_is_noise(self):
        ref = [{"t": 1, "lp": -0.1, "top": [[1, -0.1], [2, -0.2]]}] * 3 + [
            {"t": 5, "lp": -0.7, "top": [[5, -0.7], [6, -0.8]]}
        ]
        test = [dict(r) for r in ref]
        test[3] = {"t": 6, "lp": -0.8, "top": [[5, -0.7], [6, -0.8]]}
        v = bm.identity_verdict(
            _c7req("a", 1, text="x y z"),
            _c7req("b", 1, text="x y q"),
            ref_lp=ref,
            test_lp=test,
        )
        assert v["classification"] == "noise"
        assert v["div_token"] == 3

    def test_large_margin_divergence_is_bug(self):
        ref = [{"t": 5, "lp": -0.1, "top": [[5, -0.1], [6, -4.0]]}]
        test = [{"t": 6, "lp": -4.0, "top": [[5, -0.1], [6, -4.0]]}]
        v = bm.identity_verdict(
            _c7req("a", 1, text="x"), _c7req("b", 1, text="y"), ref_lp=ref, test_lp=test
        )
        assert v["classification"] == "bug"


NF_FIELDS = {
    "required_bytes": 90_000_000_000,
    "capacity_bytes": 42_000_000_000,
    "weights_bytes": 17_000_000_000,
    "kv_bytes_per_token": 60_000,
    "requested_context_tokens": 60_000,
    "requested_max_tokens": 2_000_000,
    "max_viable_context_tokens": 400_000,
    "retryable": False,
}


def _nf_body(**over):
    err = {"omlx_code": "context_exceeds_capacity", **NF_FIELDS, **over}
    return {"type": "error", "error": err}


class TestEvaluateNeverFits:
    def test_structured_error_is_correct(self):
        r = bm.evaluate_never_fits(400, _nf_body(), 1.2, [])
        assert r["correct"] is True
        assert r["reasons"] == []
        assert r["fields"]["max_viable_context_tokens"] == 400_000

    def test_wrong_status_or_code(self):
        r = bm.evaluate_never_fits(500, _nf_body(), 1.0, [])
        assert r["correct"] is False
        assert any("status" in x for x in r["reasons"])
        body = _nf_body()
        body["error"]["omlx_code"] = "prefill_memory_exceeded"
        r = bm.evaluate_never_fits(400, body, 1.0, [])
        assert any("omlx_code" in x for x in r["reasons"])

    def test_missing_field_and_retryable(self):
        body = _nf_body(retryable=True)
        del body["error"]["weights_bytes"]
        r = bm.evaluate_never_fits(400, body, 1.0, [])
        assert any("weights_bytes" in x for x in r["reasons"])
        assert any("retryable" in x for x in r["reasons"])

    def test_inconsistent_numbers(self):
        r = bm.evaluate_never_fits(
            400, _nf_body(required_bytes=1, capacity_bytes=2), 1.0, []
        )
        assert any("required_bytes" in x for x in r["reasons"])

    def test_park_events_or_slow_answer_fail(self):
        ev = bm.parse_c7_events(C7_LOG)
        r = bm.evaluate_never_fits(400, _nf_body(), 1.0, ev)
        assert any("park" in x or "pause" in x for x in r["reasons"])
        r = bm.evaluate_never_fits(400, _nf_body(), 999.0, [])
        assert any("admission" in x for x in r["reasons"])

    def test_non_dict_body(self):
        r = bm.evaluate_never_fits(None, None, 120.0, [])
        assert r["correct"] is False


def _c7sum(scenario, reqs, log=C7_LOG, **kw):
    return bm.summarize_c7(scenario, reqs, bm.parse_c7_events(log), **kw)


class TestSummarizeC7:
    def test_counters(self):
        reqs = [_c7req("req0", 120000), _c7req("req1", 90000, ok=False)]
        s = _c7sum("decode", reqs, peak_gb=41.2, wall_s=300.0)
        assert s["pauses_prefill"] == 1
        assert s["pauses_decode"] == 1
        assert s["resumes_prefill"] == 1
        assert s["resumes_decode"] == 1
        assert s["restored_tokens"] == 120640
        assert s["aborts_single"] == 1
        assert s["aborts_all"] == 3
        assert s["completions"] == 1
        assert s["aborted_requests"] == 1
        assert s["peak_gb"] == 41.2
        assert s["wall_s"] == 300.0
        assert s["hard_pressure_seen"] is True

    def test_rows_carry_pause_attribution_and_stall(self):
        reqs = [_c7req("req0", 120000, times=[1.0, 1.1, 6.1])]
        s = _c7sum("decode", reqs)
        row = s["rows"][0]
        assert row["decode_pauses"] == 1
        assert row["max_decode_stall_s"] == pytest.approx(5.0)
        assert s["max_decode_stall_s"] == pytest.approx(5.0)

    def test_identity_rollup(self):
        ident = {
            "req0": {"identical": True, "paused": True, "classification": None},
            "req1": {"identical": False, "paused": True, "classification": "noise"},
            "req2": {"identical": False, "paused": False, "classification": "bug"},
        }
        s = _c7sum("decode", [_c7req("req0", 120000)], identity=ident)
        assert s["identity"] == {
            "checked": 3,
            "paused_checked": 2,
            "identical": 1,
            "noise": 1,
            "bug": 1,
            "inconclusive": 0,
            "unclassified": 0,
        }

    def test_never_fits_correctness_passthrough(self):
        s = _c7sum("never-fits", [], log="", never_fits={"correct": True})
        assert s["never_fits_correct"] is True
        assert s["verdict"] == "never_fits_ok"


class TestClassifyC7:
    def _s(self, **kw):
        base = {
            "aborted_requests": 0,
            "pauses_prefill": 0,
            "pauses_decode": 0,
            "resumes_prefill": 0,
            "resumes_decode": 0,
            "hard_pressure_seen": True,
            "all_final_ok": True,
            "identity": {"bug": 0},
            "never_fits_correct": None,
        }
        base.update(kw)
        return base

    def test_decode_ok(self):
        s = self._s(pauses_decode=2, resumes_decode=2)
        assert bm.classify_c7("decode", s) == "decode_pause_ok"
        assert bm.classify_c7("image-decode", s) == "decode_pause_ok"

    def test_decode_failures(self):
        assert bm.classify_c7("decode", self._s(aborted_requests=1, pauses_decode=1, resumes_decode=1)) == "aborts"
        assert bm.classify_c7("decode", self._s(hard_pressure_seen=False)) == "no_pressure"
        assert bm.classify_c7("decode", self._s()) == "no_decode_pause"
        assert bm.classify_c7("decode", self._s(pauses_decode=2, resumes_decode=1)) == "paused_not_resumed"
        assert bm.classify_c7("decode", self._s(pauses_decode=1, resumes_decode=1, identity={"bug": 1})) == "identity_bug"

    def test_image_prefill(self):
        assert bm.classify_c7("image-prefill", self._s(pauses_prefill=1, resumes_prefill=1)) == "prefill_park_ok"
        assert bm.classify_c7("image-prefill", self._s()) == "no_park"
        assert bm.classify_c7("image-prefill", self._s(pauses_prefill=2, resumes_prefill=1)) == "parked_not_resumed"
        assert bm.classify_c7("image-prefill", self._s(aborted_requests=1, pauses_prefill=1, resumes_prefill=1)) == "aborts"

    def test_never_fits(self):
        assert bm.classify_c7("never-fits", self._s(never_fits_correct=True)) == "never_fits_ok"
        assert bm.classify_c7("never-fits", self._s(never_fits_correct=False)) == "never_fits_wrong"

    def test_unknown_scenario(self):
        with pytest.raises(ValueError):
            bm.classify_c7("nope", self._s())


class TestC7Defaults:
    def test_decode_defaults_match_the_p7_standard_config(self):
        d = bm.c7_defaults("decode")
        assert d["ceiling_gb"] == 42.0
        assert d["hard_threshold"] == 0.8
        assert d["max_tokens"] >= 1500
        assert d["requests"] == 3

    def test_unknown(self):
        with pytest.raises(ValueError):
            bm.c7_defaults("x")


class _C7Resp:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    async def aread(self):
        import json as _json

        return _json.dumps(self._body).encode()

    def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class TestScenarioC7:
    def _ctx(self, kind, fail=(), paused_log=C7_LOG, nf=None):
        import argparse
        import asyncio

        calls = []
        posts = []

        class Srv:
            base_url = "http://x"
            dump_dir = None

            def log_size(self):
                return 0

            def log_since(self, off):
                return paused_log

            def headers(self):
                return {}

        class Corpus:
            def prompt(self, seed, n):
                return f"PROMPT[{seed}:{n}]"

        class Client:
            def stream(self, method, url, json=None, headers=None):
                posts.append(("stream", url, json))
                return _C7Resp(
                    nf[0] if nf else 400,
                    nf[1] if nf else _nf_body(),
                    {"X-oMLX-Max-Context-Tokens": "400000"},
                )

            async def post(self, url, json=None, headers=None, timeout=None):
                posts.append(("post", url, json))
                return _C7Resp(200, {}, {"X-oMLX-Max-Context-Tokens": "400000"})

            async def get(self, url, headers=None, params=None, timeout=None):
                posts.append(("get", url, params))
                return _C7Resp(200, {"available": True, "max_viable_context_tokens": 400000})

        class Ctx:
            server = Srv()
            corpus = Corpus()
            client = Client()
            args = argparse.Namespace(
                c7_scenario=kind,
                c7_requests=3,
                c7_stagger_s=0.0,
                c7_max_tokens=2000,
                c7_warm_tokens=2000,
                c7_never_prompt_tokens=60000,
                c7_never_max_tokens=2_000_000,
                c7_no_identity=False,
                p7_ceiling_gb=42.0,
                settle_s=0.0,
                smoke=False,
            )

            def n(self, nominal):
                return nominal

            def mt(self, nominal):
                return nominal

            def seed(self, *parts):
                return "|".join(map(str, parts))

            async def chat(self, prompt, max_tokens, sampling=None, **kw):
                name = kw["name"]
                calls.append((name, prompt, max_tokens, sampling))
                idx = int(name[3]) if name.startswith("req") else 0
                ptok = {0: 120000, 1: 90000, 2: 100000}.get(idx, 1000)
                return _c7req(
                    name,
                    ptok,
                    ok=name not in fail,
                    text="alpha beta gamma",
                )

        return Ctx(), calls, posts, asyncio

    def test_decode_runs_pressure_phase_then_alone_references_greedy(self):
        ctx, calls, _, asyncio = self._ctx("decode")
        sc = asyncio.run(bp.scenario_c7(ctx))
        names = [c[0] for c in calls]
        assert names[:3] == ["req0", "req1", "req2"]
        assert names[3:] == ["req0-ref", "req1-ref", "req2-ref"]
        assert all(c[3] == bp.SAMPLING_GREEDY for c in calls)
        assert all(c[2] == 2000 for c in calls)
        assert sc["summary"]["identity"]["checked"] == 3
        assert sc["summary"]["pauses_decode"] == 1
        assert sc["summary"]["rows"][0]["decode_pauses"] == 1
        assert sc["identity"]["req0"]["paused"] is True
        assert sc["identity"]["req1"]["paused"] is False
        assert all("text" not in r and "chunk_times" not in r for r in sc["requests"])
        assert bp.scenario_ok("C7", sc) is True

    def test_no_identity_flag_skips_references(self):
        ctx, calls, _, asyncio = self._ctx("decode")
        ctx.args.c7_no_identity = True
        asyncio.run(bp.scenario_c7(ctx))
        assert not any(c[0].endswith("-ref") for c in calls)

    def test_failed_request_is_not_reissued_and_fails_scenario(self):
        ctx, calls, _, asyncio = self._ctx("decode", fail={"req1"})
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert not any("reissue" in c[0] for c in calls)
        assert "req1-ref" not in [c[0] for c in calls]
        assert sc["summary"]["aborted_requests"] == 1
        assert bp.scenario_ok("C7", sc) is False

    def test_image_decode_puts_an_image_in_every_prompt(self):
        ctx, calls, _, asyncio = self._ctx("image-decode")
        asyncio.run(bp.scenario_c7(ctx))
        prompts = [c[1] for c in calls if c[0] in ("req0", "req1", "req2")]
        assert all(p[1]["type"] == "image_url" for p in prompts)
        assert len({p[1]["image_url"]["url"] for p in prompts}) == 3

    def test_image_prefill_warms_each_image_prefix_first(self):
        ctx, calls, _, asyncio = self._ctx("image-prefill")
        ctx.args.c7_max_tokens = 32
        asyncio.run(bp.scenario_c7(ctx))
        names = [c[0] for c in calls]
        assert names[:3] == ["warm0", "warm1", "warm2"]
        assert names[3:6] == ["req0", "req1", "req2"]
        assert not any(n.endswith("-ref") for n in names)
        warm = calls[0][1]
        long_ = calls[3][1]
        assert warm[1] == long_[1]  # same image part
        assert long_[2]["text"].startswith(warm[2]["text"])  # text suffix after it
        assert len(long_[2]["text"]) > len(warm[2]["text"])

    def test_never_fits_checks_status_body_capacity_and_header(self):
        ctx, calls, posts, asyncio = self._ctx("never-fits", paused_log="")
        sc = asyncio.run(bp.scenario_c7(ctx))
        body = next(p[2] for p in posts if p[0] == "stream")
        assert body["max_tokens"] == 2_000_000
        assert any(p[0] == "get" and "omlx/capacity" in p[1] for p in posts)
        assert sc["never_fits"]["status"] == 400
        assert sc["capacity"]["status"] == 200
        assert sc["max_context_header"] == "400000"
        assert sc["followup"]["ok"] is True
        assert sc["summary"]["never_fits_correct"] is True
        assert sc["summary"]["verdict"] == "never_fits_ok"
        assert bp.scenario_ok("C7", sc) is True

    def test_never_fits_wrong_status_fails_scenario(self):
        ctx, _, _, asyncio = self._ctx("never-fits", nf=(500, {"error": {}}), paused_log="")
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert sc["summary"]["never_fits_correct"] is False
        assert bp.scenario_ok("C7", sc) is False


    def test_never_fits_with_park_events_in_the_window_is_wrong(self):
        ctx, _, _, asyncio = self._ctx("never-fits")  # window holds park events
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert sc["summary"]["never_fits_correct"] is False


class TestC7Wiring:
    def test_registered_opt_in_with_concurrency_three(self):
        assert "C7" in bp.SCENARIOS
        assert bp.PREMISE_CONCURRENCY["C7"] == 3
        assert "C7" not in bp.ALL_PREMISES

    def test_logprob_dump_enabled_for_c7_on_non_upstream_only(self, tmp_path):
        assert "OMLX_DEBUG_LOGPROB_DUMP" in bp.debug_env("fork", "C7", tmp_path)
        assert bp.debug_env("upstream", "C7", tmp_path) == {}

    def test_cli_defaults_fill_from_the_scenario(self):
        a = bp.resolve_c7_args(
            __import__("argparse").Namespace(
                c7_scenario="decode",
                p7_ceiling_gb=None,
                p7_hard_threshold=None,
                c7_stagger_s=None,
                c7_max_tokens=None,
                c7_requests=None,
            )
        )
        assert (a.p7_ceiling_gb, a.p7_hard_threshold) == (42.0, 0.8)
        assert a.c7_max_tokens >= 1500 and a.c7_requests == 3

    def test_cli_overrides_win(self):
        a = bp.resolve_c7_args(
            __import__("argparse").Namespace(
                c7_scenario="decode",
                p7_ceiling_gb=30.0,
                p7_hard_threshold=0.7,
                c7_stagger_s=2.0,
                c7_max_tokens=100,
                c7_requests=2,
            )
        )
        assert (a.p7_ceiling_gb, a.p7_hard_threshold, a.c7_max_tokens) == (30.0, 0.7, 100)



# --------------------------------------------------------------------------
# C7 re-run (D4): ceiling-headroom references, new log lines, capacity checks
# --------------------------------------------------------------------------

C7R_LOG = """\
2026-10-06 10:00:01,000 - omlx.process_memory_enforcer - WARNING - [-] - Hard memory pressure: rid-a runs alone (solo) on 'm'; it is not paused or aborted below the emergency threshold (ceiling 48.0GB)
2026-10-06 10:00:02,000 - omlx.scheduler - INFO - [-] - Resuming parked request rid-b alone (solo): it fits under the ceiling only with nothing else running; nothing else is admitted until it finishes, and memory pressure below the emergency threshold does not pause it again
2026-10-06 10:00:03,000 - omlx.scheduler - INFO - [-] - Decode pause skipped for rid-c: decode pause budget spent (8/8)
2026-10-06 10:00:04,000 - omlx.process_memory_enforcer - WARNING - [-] - Emergency memory pressure: waiting for the decode pause of rid-a on 'm' before aborting (poll 1/2)
2026-10-06 10:00:05,000 - omlx.scheduler - INFO - [-] - Deferring media prefill rid-d: its external-loop peak ~20.1 GB would cross the hard watermark line 36.5 GB beside admitted work; it waits for admitted requests to finish
"""


class TestParseC7RerunEvents:
    def test_new_lines_become_events(self):
        ev = bm.parse_c7_events(C7R_LOG)
        assert [e["kind"] for e in ev] == [
            "solo",
            "solo",
            "decode_budget_spent",
            "emergency_pause_wait",
            "media_deferred",
        ]
        assert ev[0]["rid"] == "rid-a" and ev[1]["rid"] == "rid-b"
        assert ev[2]["rid"] == "rid-c" and (ev[2]["used"], ev[2]["budget"]) == (8, 8)
        assert ev[3]["rid"] == "rid-a"
        assert ev[4]["rid"] == "rid-d" and ev[4]["peak_gb"] == 20.1

    def test_resume_solo_line_is_not_a_fit_resume(self):
        ev = bm.parse_c7_events(C7R_LOG)
        assert "resume_fit" not in {e["kind"] for e in ev}

    def test_summary_counts_them(self):
        s = bm.summarize_c7("decode", [], bm.parse_c7_events(C7R_LOG))
        assert s["solo_events"] == 2
        assert s["decode_budget_spent"] == 1
        assert s["emergency_pause_waits"] == 1
        assert s["media_deferrals"] == 1


class TestCapacityCheck:
    def test_available_with_header(self):
        c = bm.capacity_check({"status": 200, "body": {"available": True}}, "123456")
        assert c == {"capacity_available": True, "header_present": True, "header": "123456"}

    def test_unavailable_and_no_header(self):
        c = bm.capacity_check({"status": 200, "body": {"available": False}}, None)
        assert c["capacity_available"] is False and c["header_present"] is False

    def test_error_or_missing_body(self):
        assert bm.capacity_check({"status": None, "error": "x"}, "1")["capacity_available"] is False
        assert bm.capacity_check({"status": 200, "body": None}, "")["header_present"] is False


class TestC7References:
    def _ref(self, text="a b c", ptok=100):
        return {"name": "req0", "ok": True, "text": text, "prompt_tokens": ptok, "completion_tokens": 3}

    def test_round_trip(self, tmp_path):
        p = tmp_path / "references.json"
        bm.save_c7_refs(p, {"req0": {"result": self._ref(), "lp": [{"t": 1}]}}, ceiling_gb=56.0)
        data = bm.load_c7_refs(p)
        assert data["ceiling_gb"] == 56.0
        assert data["refs"]["req0"]["result"]["text"] == "a b c"
        assert data["refs"]["req0"]["lp"] == [{"t": 1}]

    def test_identity_against_refs_identical(self):
        refs = {"req0": {"result": self._ref(), "lp": None}}
        got = bm.identity_against_refs(refs, [_c7req("req0", 100, text="a b c")], {}, paused=set())
        assert got["req0"]["identical"] is True
        assert got["req0"]["paused"] is False
        assert got["req0"]["ref_ok"] is True and got["req0"]["ref_clean"] is True

    def test_identity_against_refs_divergent_marks_paused(self):
        refs = {"req0": {"result": self._ref(), "lp": None}}
        got = bm.identity_against_refs(
            refs, [_c7req("req0", 100, text="a b q")], {}, paused={"req0"}
        )
        assert got["req0"]["identical"] is False
        assert got["req0"]["classification"] == "unclassified"
        assert got["req0"]["paused"] is True

    def test_failed_or_unreferenced_requests_are_skipped(self):
        refs = {"req0": {"result": self._ref(), "lp": None}}
        got = bm.identity_against_refs(
            refs,
            [_c7req("req0", 100, ok=False), _c7req("req1", 100)],
            {},
            paused=set(),
        )
        assert got == {}

    def test_prompt_mismatch_is_flagged(self):
        refs = {"req0": {"result": self._ref(ptok=100), "lp": None}}
        got = bm.identity_against_refs(refs, [_c7req("req0", 999, text="a b c")], {}, paused=set())
        assert got["req0"]["prompt_mismatch"] is True


class TestScenarioC7Refs(TestScenarioC7):
    def test_refs_only_runs_alone_serially_and_saves_references(self, tmp_path):
        ctx, calls, _, asyncio = self._ctx("decode", paused_log="")
        ctx.server.out = tmp_path
        ctx.args.c7_refs_only = True
        ctx.args.p7_ceiling_gb = 56.0
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert [c[0] for c in calls] == ["req0", "req1", "req2"]
        assert all(c[3] == bp.SAMPLING_GREEDY for c in calls)
        data = bm.load_c7_refs(tmp_path / "references.json")
        assert data["ceiling_gb"] == 56.0
        assert set(data["refs"]) == {"req0", "req1", "req2"}
        assert data["refs"]["req0"]["result"]["text"] == "alpha beta gamma"
        assert data["refs"]["req0"]["clean"] is True
        assert sc["summary"]["verdict"] == "refs_ok"
        assert bp.scenario_ok("C7", sc) is True

    def test_refs_only_flags_a_pressured_reference(self, tmp_path):
        ctx, _, _, asyncio = self._ctx("decode")  # C7_LOG has hard pressure + pause
        ctx.server.out = tmp_path
        ctx.args.c7_refs_only = True
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert sc["summary"]["verdict"] == "refs_pressured"

    def test_refs_from_replaces_the_alone_phase(self, tmp_path):
        ctx, calls, _, asyncio = self._ctx("decode")
        ref = _c7req("req0", 120000, text="alpha beta gamma")
        refs = {
            f"req{i}": {"result": {**ref, "name": f"req{i}", "prompt_tokens": p}, "lp": None, "clean": True}
            for i, p in enumerate((120000, 90000, 100000))
        }
        bm.save_c7_refs(tmp_path / "references.json", refs, 56.0)
        ctx.args.c7_refs_from = str(tmp_path)
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert [c[0] for c in calls] == ["req0", "req1", "req2"]
        assert sc["summary"]["identity"]["checked"] == 3
        assert sc["summary"]["identity"]["identical"] == 3
        assert sc["identity"]["req0"]["paused"] is True
        assert sc["identity"]["req0"]["ref_ceiling_gb"] == 56.0

    def test_never_fits_summary_carries_capacity_fields(self):
        ctx, _, _, asyncio = self._ctx("never-fits", paused_log="")
        sc = asyncio.run(bp.scenario_c7(ctx))
        assert sc["summary"]["capacity_available"] is True
        assert sc["summary"]["header_present"] is True
