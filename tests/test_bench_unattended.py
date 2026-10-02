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


def _p3(identical=True, b3_line=3, rows_ok=True):
    rows = []
    for name in ("20k", "45k", "135k"):
        alone = {"ok": rows_ok, "text": "abc def", "completion_tokens": 7}
        batched = {
            "ok": rows_ok,
            "text": "abc def" if identical else "abc xyz",
            "completion_tokens": 7,
        }
        rows.append({"name": name, "alone": alone, "batched": batched})
    log = bm.parse_server_log("")
    log["tq_batched_fused_max_b"] = b3_line
    return _run({"rows": rows}, log=log)


class TestEvalP3:
    def test_pass(self):
        assert bm.eval_p3([_p3()], [_p3(identical=False)])["verdict"] == "PASS"

    def test_fail_on_mismatch(self):
        r = bm.eval_p3([_p3(identical=False)], [])
        assert r["verdict"] == "FAIL"

    def test_fail_when_batch_not_exercised(self):
        r = bm.eval_p3([_p3(b3_line=0)], [])
        assert r["verdict"] == "FAIL"
        assert any("batch" in c["name"] and not c["ok"] for c in r["checks"])

    def test_fail_when_request_failed(self):
        assert bm.eval_p3([_p3(rows_ok=False)], [])["verdict"] == "FAIL"

    def test_upstream_mismatch_count_reported(self):
        r = bm.eval_p3([_p3()], [_p3(identical=False)])
        assert r["upstream"]["mismatched_rows"] == 3
        assert r["fork"]["mismatched_rows"] == 0

    def test_common_prefix_len(self):
        assert bm.common_prefix_len("abcdef", "abcxyz") == 3
        assert bm.common_prefix_len("", "a") == 0


def _p4(
    serial_wall,
    conc_wall,
    decs_serial=(20.0, 20.0, 20.0),
    decs_conc=(12.0, 12.0, 12.0),
    aborts=0,
    ok=True,
):
    log = bm.parse_server_log("")
    log["aborts_total"] = aborts
    return _run(
        {
            "serial": {
                "wall_s": serial_wall,
                "requests": [_req(dec=d, ok=ok) for d in decs_serial],
            },
            "concurrent": {
                "wall_s": conc_wall,
                "requests": [_req(dec=d, ok=ok) for d in decs_conc],
            },
        },
        log=log,
    )


class TestEvalP4:
    def test_pass(self):
        assert bm.eval_p4([_p4(600, 400)], [_p4(600, 500)])["verdict"] == "PASS"

    def test_fail_when_concurrent_not_faster(self):
        assert bm.eval_p4([_p4(600, 650)], [])["verdict"] == "FAIL"

    def test_fail_on_abort(self):
        assert bm.eval_p4([_p4(600, 400, aborts=1)], [])["verdict"] == "FAIL"

    def test_fail_when_row_below_half(self):
        assert (
            bm.eval_p4([_p4(600, 400, decs_conc=(12.0, 9.0, 12.0))], [])["verdict"]
            == "FAIL"
        )

    def test_fail_when_request_failed(self):
        assert bm.eval_p4([_p4(600, 400, ok=False)], [])["verdict"] == "FAIL"

    def test_exactly_half_passes(self):
        assert (
            bm.eval_p4([_p4(600, 400, decs_conc=(10.0, 10.0, 10.0))], [])["verdict"]
            == "PASS"
        )


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


class TestP3CacheRestoreInfo:
    def test_alone_cached_mismatch_is_counted_separately(self):
        run = _p3()
        for row in run["scenario"]["rows"]:
            row["alone_cached"] = {
                "ok": True,
                "text": "different",
                "completion_tokens": 7,
            }
        s = bm.eval_p3([run], [])
        assert s["verdict"] == "PASS"  # spec compares B=1 vs B=3 only
        assert s["fork"]["cache_restore_mismatched_rows"] == 3


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
