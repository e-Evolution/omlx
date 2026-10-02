#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a markdown report from bench_premises.py JSON results.

Usage: report.py --out DIR [--md FILE]
Reads DIR/<variant>/<premise>/run*.json (and optional DIR/p5/*.json).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_metrics as bm  # noqa: E402

VARIANTS = ["upstream", "fork-off", "fork", "fork-nomtp", "fork-d1only", "fork-cachelimit-only"]
PREMISES = ["P1", "P2", "P3", "P4", "P6"]
TITLES = {
    "P1": "P1 - no lost work (cancel mid-prefill, re-issue)",
    "P2": "P2 - 2 x ~130k concurrent, no aborts",
    "P3": "P3 - greedy batch correctness (B=3 restored vs B=1 restored)",
    "P4": "P4 - concurrency pays (3 mixed requests)",
    "P5": "P5 - unattended pi batch",
    "P6": "P6 - single 110k request, no regression (upstream vs fork-off vs fork)",
}


def load(out: Path) -> dict[str, dict[str, list[dict]]]:
    data: dict[str, dict[str, list[dict]]] = {}
    for v in VARIANTS:
        for p in PREMISES:
            files = sorted((out / v / p).glob("run*.json"), key=lambda f: f.name)
            runs = [
                json.loads(f.read_text())
                for f in files
                if not f.name.endswith(".samples.json")
            ]
            if runs:
                data.setdefault(v, {})[p] = runs
    return data


def f(v: Any, nd: int = 1, unit: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def med(runs: list[dict], getter) -> float | None:
    vals = []
    for r in runs:
        with contextlib.suppress(KeyError, TypeError, IndexError):
            vals.append(getter(r))
    return bm.median(vals)


def req_med(runs: list[dict], key: str) -> float | None:
    return med(runs, lambda r: r["scenario"]["request"][key])


def scenario_med(runs: list[dict], key: str) -> float | None:
    return med(runs, lambda r: r["scenario"][key])


def table(header: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def setup_block(data: dict) -> str:
    runs = [r for v in data.values() for rs in v.values() for r in rs]
    if not runs:
        return "_No results._"
    s = runs[0]["setup"]
    fork = next((r["setup"] for r in runs if r["variant"] != "upstream"), s)
    app = s.get("upstream_app") or {}
    lines = [
        f"- Upstream: `{s['upstream_tag']}` commit `{s['upstream_commit'][:10]}`"
        f" (official oMLX.app {app.get('version', '?')} build {app.get('build', '?')}), run via `omlx-cli serve`",
        f"- Fork: branch `{fork.get('fork_branch')}` HEAD `{fork['fork_commit'][:10]}` (`{fork.get('fork_describe')}`)",
        f"- Hardware: {s['chip']}, {s['ram_gb']} GB unified memory, macOS {s['macos']}",
        f"- Model: `{s['model']}` (oQ4e + MTP); TurboQuant KV: {s['model_settings'].get('turboquant_kv_enabled')} "
        f"({s['model_settings'].get('turboquant_kv_bits')} bit); MTP: {s['model_settings'].get('mtp_enabled')}; "
        f"thinking: {s['model_settings'].get('enable_thinking')}",
        f"- Sampling: default {s['sampling']['default']}; greedy (P3) {s['sampling']['greedy']}",
        f"- Memory: guard tier `{s['memory']['memory_guard_tier']}`; cache dirs isolated per variant, "
        f"SSD max {s['cache'].get('ssd_cache_max_size')}, hot {s['cache'].get('hot_cache_max_size')}",
        f"- Cache/warm-up conditions: {s['cache_condition']}; warm-up: {s['warmup']}",
    ]
    envs = {}
    for r in runs:
        envs[r["variant"]] = r["setup"]["feature_env"]
    for v, env in envs.items():
        lines.append(
            f"- Feature env `{v}`: "
            + (
                ", ".join(f"`{k}={val}`" for k, val in env.items())
                or "none (upstream code)"
            )
        )
    conc = sorted({(r["premise"], r["setup"]["max_concurrent_requests"]) for r in runs})
    lines.append(
        "- `max_concurrent_requests`: " + ", ".join(f"{p}={c}" for p, c in conc)
    )
    counts = sorted({(r["variant"], r["premise"], r["runs_requested"]) for r in runs})
    lines.append(
        "- Runs per variant/premise: " + ", ".join(f"{v}/{p}={n}" for v, p, n in counts)
    )
    lines += [
        "- Run counts: P6 and the other premises report the MEDIAN over the runs above, with min/max "
        "spread; one server start per (variant, premise), cold cache per run (unique nonce per run).",
        "- P3 flow (like-for-like): cold B=1 per prompt only populates the prefix cache; then B=1 restored "
        "(reference), then B=3 restored started together so all rows decode in one batch "
        "(`max_concurrent_requests=3`; fork variants run with `OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS=0`). "
        "The chat API exposes no logprobs, so fork variants run with the opt-in debug dump "
        "(`OMLX_DEBUG_LOGPROB_DUMP`) and tokens are compared with their top-5 logprobs. PASS = 3-row "
        "batch in the log, no degeneration (distinct word-4gram ratio < 0.5x the B=1 value, or < 0.3 "
        "when B=1 is above it), and every B=3 vs B=1-restored divergence inside the first 32 tokens "
        "is numerical noise: top1-top2 margin of the B=1 reference < 0.5 nats and mean |dlp| of the "
        "shared prefix < 0.05 (bug: margin > 2 or mean |dlp| >= 0.25; between: inconclusive = FAIL). "
        "Controls: B=1 restored twice (determinism), B=1 forced through the ragged B>1 kernel, and the "
        "`fork-nomtp` variant (MTP off) which isolates batching from MTP verify numerics; when the "
        "MTP-on reference is itself not reproducible, the batching-only (MTP off) judgement decides. "
        "Divergence after token 32 is reported, not judged. Upstream is reported, not judged.",
        "- P4 criterion: with B=3 the non-attention compute alone sets a ~49 ms step (~20 tok/s per row), "
        "so 'per-row >= 50% of single-row' is physically unreachable; the premise is not wasting memory and "
        "time. PASS = 0 aborts, concurrent wall <= serial wall, and the aggregate decode throughput over "
        "the window where all three rows decode (per-chunk timestamps) >= the median single-row serial "
        "decode rate. Per-row ratios are informational.",
        "- P6 attribution: `upstream` -> `fork-off` is the fork code-path overhead with all four features "
        "off; `fork-off` -> `fork` is the features' cost; `fork-d1only` / `fork-cachelimit-only` isolate "
        "single features when run.",
    ]
    if any(r["setup"].get("smoke") for r in runs):
        lines.append(
            "- **SMOKE RUN: tiny prompts, results are NOT valid premise evidence.**"
        )
    return "\n".join(lines)


def _rng(runs: list[dict], key: str, nd: int) -> str:
    vals = []
    for r in runs:
        q = r["scenario"]["request"]
        if q.get("ok") and q.get(key) is not None:
            vals.append(q[key])
    lo, hi = bm.spread(vals)
    return "n/a" if lo is None else f"{lo:.{nd}f}-{hi:.{nd}f}"


def section_p6(data: dict) -> str:
    rows = []
    for v in VARIANTS:
        runs = data.get(v, {}).get("P6")
        if not runs:
            continue
        ok = [r for r in runs if r["scenario"]["request"].get("ok")]
        q = functools.partial(req_med, ok)
        rows.append(
            [
                v,
                f"{len(ok)}/{len(runs)}",
                f(q("prompt_tokens"), 0),
                f(q("ttft_s"), 1, " s"),
                f(q("prefill_tps"), 1),
                _rng(runs, "prefill_tps", 1),
                f(q("decode_tps"), 2),
                _rng(runs, "decode_tps", 2),
                f(q("total_s"), 1, " s"),
                f(med(runs, lambda r: r["samples"]["footprint_peak_gb"]), 1, " GB"),
            ]
        )
    return table(
        [
            "variant",
            "ok/runs",
            "prompt tok",
            "TTFT (median)",
            "prefill tok/s (median)",
            "prefill min-max",
            "decode tok/s (median)",
            "decode min-max",
            "total",
            "peak footprint",
        ],
        rows,
    )


def section_p1(data: dict) -> str:
    rows = []
    for v in VARIANTS:
        runs = data.get(v, {}).get("P1")
        if not runs:
            continue
        sc = functools.partial(scenario_med, runs)
        rows.append(
            [
                v,
                f(sc("prompt_tokens"), 0),
                f(sc("progress_tokens_est"), 0),
                f(sc("cancel_after_s"), 1, " s"),
                f(sc("reissue_cached"), 0),
                f(sc("reissue_reprefilled"), 0),
                f(med(runs, lambda r: r["scenario"]["reissue"]["ttft_s"]), 1, " s"),
                f(sum(len(r["log"]["parked_tokens"]) for r in runs), 0),
            ]
        )
    return table(
        [
            "variant",
            "prompt tok",
            "est. progress at cancel",
            "cancel after",
            "re-issue cached",
            "re-prefilled",
            "re-issue TTFT",
            "parks",
        ],
        rows,
    )


def section_p2(data: dict) -> str:
    rows = []
    for v in VARIANTS:
        runs = data.get(v, {}).get("P2")
        if not runs:
            continue
        s = bm._p2_summary(runs)
        lv = sorted({lvl for r in runs for lvl in r["log"]["pressure_levels_seen"]})
        rows.append(
            [
                v,
                f"{s['completed']}/{s['total']}",
                f(s["aborts_memory"], 0),
                f(s["aborts_total"], 0),
                f(s["footprint_peak_gb"], 1, " GB"),
                f(s["pool_gb_max"], 1, " GB"),
                f(s["wall_s_median"], 1, " s"),
                f(sum(r["log"]["parks"] for r in runs), 0),
                f(sum(r["log"]["holds"] for r in runs), 0),
                ",".join(lv) or "-",
            ]
        )
    return table(
        [
            "variant",
            "completed",
            "memory aborts",
            "all aborts",
            "peak footprint",
            "max pool",
            "wall",
            "parks",
            "holds",
            "pressure levels",
        ],
        rows,
    )


def section_p3(data: dict) -> str:
    rows = []
    for v in VARIANTS:
        runs = data.get(v, {}).get("P3")
        if not runs:
            continue
        s = bm._p3_summary(runs)
        for d in s["detail"]:
            div = d["first_divergence_chars"]
            rows.append(
                [
                    v,
                    d["name"],
                    "identical"
                    if div is None
                    else f"char {div} (~token {f(d['first_divergence_token_est'], 0)})",
                    "yes" if d["diverged_in_prefix"] else "no",
                    f(d["d4_b1"], 2),
                    f(d["d4_b3"], 2),
                    "yes" if d["degenerate"] else "no",
                    f(s["batch_max_b"], 0),
                    f(s["cold_vs_restored_mismatched_rows"], 0),
                ]
            )
    main = table(
        [
            "variant",
            "row",
            "first divergence B=3 vs B=1-restored",
            "inside first 32 tok",
            "distinct-4gram B=1",
            "distinct-4gram B=3",
            "degenerate",
            "max B in log",
            "cold vs restored B=1 mismatches (all rows)",
        ],
        rows,
    )
    return main + _p3_margins(data, "fork") + _p3_margins(data, "fork-nomtp")


def _p3_margins(data: dict, variant: str = "fork") -> str:
    """Near-tie analysis and controls (only when logprob dumps were captured)."""
    runs = data.get(variant, {}).get("P3")
    if not runs:
        return ""
    detail = [d for d in bm._p3_summary(runs)["detail"] if d["classification_available"]]
    if not detail:
        return ""
    rows = [
        [
            d["name"],
            "identical" if d["div_token"] is None else f(d["div_token"], 0),
            f(d["margin"], 3, " nats"),
            f(d["test_margin"], 3, " nats"),
            f(d["mean_abs_dlp"], 4),
            f(d["max_abs_dlp"], 4),
            d["classification"] or "-",
            "yes" if d["diverged_in_prefix"] else "no",
        ]
        for d in detail
    ]
    label = "fork" if variant == "fork" else "fork with MTP off (control)"
    out = f"\n\n**Near-tie analysis, B=3 vs B=1-restored (token level, {label}).** " + (
        f"Noise: margin < {bm.P3_NOISE_MARGIN} nats and mean |dlp| < {bm.P3_NOISE_DLP}; "
        f"bug: margin > {bm.P3_BUG_MARGIN} or mean |dlp| >= {bm.P3_BUG_DLP}.\n\n"
    )
    out += table(
        [
            "row",
            "first divergent token",
            "B=1 top1-top2 margin",
            "B=3 top1-top2 margin",
            "mean abs dlp (shared prefix)",
            "max abs dlp",
            "class",
            "inside first 32 tok",
        ],
        rows,
    )
    crow = []
    for d in detail:
        for label, desc in (
            ("rerun", "B=1 restored vs B=1 restored again"),
            ("ragged_b1", "B=1 fused vs B=1 forced through ragged kernel"),
            ("ragged_vs_b3", "B=1 ragged vs B=3"),
        ):
            c = d.get("controls", {}).get(label)
            if c is None:
                continue
            crow.append(
                [
                    d["name"],
                    desc,
                    "identical" if c["div_index"] is None else f(c["div_index"], 0),
                    f(c["margin"], 3, " nats"),
                    f(c["mean_abs_dlp"], 4),
                    c["classification"] or "-",
                ]
            )
    if crow:
        out += "\n\n**Controls.**\n\n" + table(
            ["row", "comparison", "first divergent token", "margin", "mean abs dlp", "class"],
            crow,
        )
    return out


def section_p4(data: dict) -> str:
    rows = []
    for v in VARIANTS:
        runs = data.get(v, {}).get("P4")
        if not runs:
            continue
        s = bm._p4_summary(runs)
        sw, cw = s["serial_wall_s"], s["concurrent_wall_s"]
        ratios = [
            (c / x) if (c is not None and x) else None
            for x, c in zip(s["serial_decode_tps"], s["concurrent_decode_tps"])
        ]
        rows.append(
            [
                v,
                f(sw, 1, " s"),
                f(cw, 1, " s"),
                f(sw / cw if sw and cw else None, 2, "x"),
                " / ".join(f(x, 2) for x in s["serial_decode_tps"]),
                f(s["serial_single_row_median_tps"], 2),
                f(s["aggregate_decode_tps"], 2),
                f(s["aggregate_window_s"], 1, " s"),
                " / ".join(f(x, 2) for x in s["concurrent_decode_tps"]),
                " / ".join(f(x, 2) for x in ratios),
                f(s["aborts_total"], 0),
                f"{s['completed']}/{s['total']}",
            ]
        )
    return table(
        [
            "variant",
            "serial wall",
            "concurrent wall",
            "speedup",
            "serial decode tok/s (20k/45k/90k)",
            "median single-row",
            "aggregate decode (all 3 decoding)",
            "window",
            "per-row concurrent tok/s",
            "per-row ratio (info)",
            "aborts",
            "ok",
        ],
        rows,
    )


def verdicts(data: dict) -> dict[str, dict]:
    g = lambda v, p: data.get(v, {}).get(p, [])  # noqa: E731
    out = {}
    if any(g(v, "P6") for v in data):
        out["P6"] = bm.eval_p6(
            g("upstream", "P6"), g("fork", "P6"), g("fork-off", "P6")
        )
    if any(g(v, "P1") for v in data):
        out["P1"] = bm.eval_p1(g("fork", "P1"), g("upstream", "P1"))
    if any(g(v, "P2") for v in data):
        out["P2"] = bm.eval_p2(g("fork", "P2"), g("upstream", "P2"))
    if any(g(v, "P3") for v in data):
        out["P3"] = bm.eval_p3(
            g("fork", "P3"), g("upstream", "P3"), g("fork-nomtp", "P3") or None
        )
    if any(g(v, "P4") for v in data):
        out["P4"] = bm.eval_p4(g("fork", "P4"), g("upstream", "P4"))
    return out


def attribution(data: dict, ver: dict[str, dict]) -> list[str]:
    """P6 prefill/decode attribution: upstream -> fork-off -> fork."""
    lines = []
    a = (ver.get("P6") or {}).get("attribution") or {}
    if not a:
        return lines

    def pct(v):
        return "n/a" if v is None else f"{v:+.1f}%"

    for tag in ("prefill", "decode"):
        lines.append(
            f"- P6 {tag} attribution: fork-off vs upstream {pct(a.get(tag + '_pct_forkoff_vs_upstream'))} "
            f"(fork code-path overhead, features off); fork vs fork-off "
            f"{pct(a.get(tag + '_pct_fork_vs_forkoff'))} (cost of the four features)."
        )
    base = bm.median(
        [
            r["scenario"]["request"].get("prefill_tps")
            for r in data.get("fork-off", {}).get("P6", [])
            if r["scenario"]["request"].get("ok")
        ]
    )
    for v in ("fork-d1only", "fork-cachelimit-only"):
        runs = data.get(v, {}).get("P6")
        if runs and base:
            m = bm.median(
                [
                    r["scenario"]["request"].get("prefill_tps")
                    for r in runs
                    if r["scenario"]["request"].get("ok")
                ]
            )
            if m is not None:
                lines.append(
                    f"- P6 prefill `{v}` vs fork-off: {pct((m / base - 1) * 100)}."
                )
    lines.append(
        "- The T5d per-row TurboQuant fix cannot be disabled and is present in every fork variant."
    )
    return lines


def tradeoffs(data: dict, ver: dict[str, dict]) -> list[str]:
    notes = []
    for p, v in ver.items():
        for c in v["checks"]:
            if not c["ok"]:
                notes.append(f"- **{p} FAIL** - {c['name']}: {c['detail']}")
    runs = [r for vv in data.values() for rs in vv.values() for r in rs]
    bad = [r for r in runs if not r["ok"]]
    if bad:
        notes.append(
            f"- {len(bad)} run(s) had a failed request (SSE error frame or missing usage counts as failure): "
            + ", ".join(f"{r['variant']}/{r['premise']}/run{r['run']}" for r in bad)
        )
    fk = data.get("fork", {})
    if "P6" in fk and "P6" in data.get("upstream", {}) and "P6" in ver:
        for c in ver["P6"]["checks"]:
            if c["name"].startswith(("prefill", "decode")):
                notes.append(f"- P6 {c['name']}: {c['detail']}")
    if (
        "P4" in ver
        and (ver["P4"].get("upstream") or {}).get("serial_wall_s") is not None
    ):
        u = ver["P4"]["upstream"]
        notes.append(
            f"- P4 upstream reference: serial {f(u['serial_wall_s'], 1)} s vs concurrent {f(u['concurrent_wall_s'], 1)} s, {u['aborts_total']} aborts."
        )
    if (
        "P2" in ver
        and ver["P2"].get("fork", {}).get("total")
        and ver["P2"]["fork"].get("pool_gb_max") is None
    ):
        notes.append(
            "- P2 pool: no `pool=` line was logged (the log only prints the pool during pressure episodes), so the pool limit is unverified by log; "
            "the MLX cache limit setting is confirmed by the `MLX buffer cache limit` startup line."
        )
    notes += attribution(data, ver)
    if not notes:
        notes.append("- None observed.")
    return notes


def p5_section(out: Path) -> str:
    files = sorted((out / "p5").glob("*summary*.json"))
    if not files:
        return "_Not run (see p5_pi_unattended.sh)._"
    rows = []
    for fp in files:
        d = json.loads(fp.read_text())
        rows.append(
            [
                d.get("mode", fp.stem),
                f(d.get("sessions"), 0),
                f(d.get("completed"), 0),
                f(d.get("manual_interventions"), 0),
                f(d.get("memory_errors_seen"), 0),
                f(d.get("wall_s"), 0, " s"),
            ]
        )
    return table(
        [
            "mode",
            "sessions",
            "completed",
            "manual interventions",
            "memory errors in output",
            "wall",
        ],
        rows,
    )


def render(out: Path) -> str:
    data = load(out)
    ver = verdicts(data)
    smoke = any(
        r["setup"].get("smoke") for v in data.values() for rs in v.values() for r in rs
    )
    md = [
        "# Unattended-scheduling premise benchmark",
        "",
        "## Setup",
        "",
        setup_block(data),
        "",
    ]
    sections = {
        "P6": section_p6,
        "P1": section_p1,
        "P2": section_p2,
        "P3": section_p3,
        "P4": section_p4,
    }
    md += ["## Verdicts", ""]
    rows = []
    for p in ["P1", "P2", "P3", "P4", "P6"]:
        if p in ver:
            v = ver[p]["verdict"] + (" (smoke, not valid)" if smoke else "")
            rows.append(
                [
                    p,
                    v,
                    "; ".join(
                        f"{'ok' if c['ok'] else 'FAIL'}: {c['name']} ({c['detail']})"
                        for c in ver[p]["checks"]
                    ),
                ]
            )
    rows.append(["P5", "see below", "pi-driven, separate script"])
    md += [table(["premise", "verdict", "checks"], rows), ""]
    md += ["## Results", ""]
    for p in ["P6", "P1", "P2", "P3", "P4"]:
        if p in ver or any(p in d for d in data.values()):
            md += [f"### {TITLES[p]}", "", sections[p](data), ""]
    md += [f"### {TITLES['P5']}", "", p5_section(out), ""]
    md += ["## Regressions and tradeoffs", "", *tradeoffs(data, ver), ""]
    md += [
        "## Reproduction",
        "",
        "See `benchmarks/unattended/README.md`. Raw JSON, samples and server logs live in the output directory (not in git).",
        "",
    ]
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/omlx-unattended-bench")
    ap.add_argument(
        "--md", default="", help="markdown output file (default: DIR/report.md)"
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    text = render(out)
    dest = Path(args.md) if args.md else out / "report.md"
    dest.write_text(text)
    print(text)
    print(f"\n[report written to {dest}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
