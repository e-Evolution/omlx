#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a recorded pi trace against an isolated oMLX bench server.

Input is the JSONL written by the pi-omlx-unattended trace recorder (one line
per provider request: ``{ts, session, provider, model, payload}``). Payloads
are sent in recorded order to ``/v1/chat/completions`` with messages, tools and
template kwargs untouched, and the prefix-cache behaviour is measured.

Modes:
  reuse (default)  max_tokens=1, stream=false: measures prefix-cache reuse only
  full             recorded max_tokens, streaming: adds TTFT, decode and quality

Safety (same rules as bench_premises.py): port 8001 (the user's app) is refused
unless ``--allow-port 8001``; a port that already accepts connections is
refused when this script would start the server; the server is always an
isolated bench instance with a fresh cache dir (or an attached one you started
with ``bench_premises.py --serve``); auth headers are never read from the trace.
Traces contain full prompts and code: keep them and the results outside git.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics as bm  # noqa: E402

MODES = ("reuse", "full")
SECRET_KEYS = {"apikey", "api_key", "authorization", "headers"}
IMAGE_PART_TYPES = {"image_url", "image", "input_image"}
CLASSES = ("cold_start", "compaction", "system_update", "image", "unknown", "hit")

LOG_UNAVAILABLE_WARNING = (
    "--attach without --server-log: log-derived fields (divergence position, restore "
    "source, holds, agings, aborts, tool-parser failures, pool memory) are unavailable "
    "and reported as null/n/a. Pass --server-log to get them."
)
LOG_FIELDS = (
    "divergence_position", "comparable_tokens", "log_reprefill_tokens", "restore_source",
    "holds", "agings", "aborts", "tool_parser_failures", "pool_gb_max",
    "cache_outcome", "ttft_hold_s", "ttft_restore_s", "ttft_prefill_s",
    "ttft_first_decode_s",
)

_RE_DIVERGE = re.compile(
    r"prefix cache: request \S+ re-prefills (\d+) of (\d+) tokens \(reused (\d+)\); "
    r"closest stored sequence \S+ shares the first (\d+) of (\d+) comparable tokens"
)
_RE_TTFT = re.compile(
    r"TTFT breakdown: request=\S+ prompt_tokens=\d+ cached_tokens=\d+ cache=(\S+) "
    r"hold_s=([\d.]+) restore_s=([\d.]+) prefill_s=([\d.]+) first_decode_s=([\d.]+)"
)
_RE_RESTORE_SRC = re.compile(r"Prefix cache restore for \S+: source=(\S+) cached=(\d+)")
_RE_TOOL_FAIL = re.compile(r"tool parser failed|Tool call (?:start )?markers? found but parsing failed", re.I)


# --------------------------------------------------------------------------
# Trace input
# --------------------------------------------------------------------------


def parse_trace(lines: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("payload"), dict):
            out.append(obj)
    return out


def filter_records(
    records: list[dict[str, Any]],
    session: str | None = None,
    ts_from: float | None = None,
    ts_to: float | None = None,
) -> list[dict[str, Any]]:
    out = []
    for r in records:
        if session is not None and r.get("session") != session:
            continue
        ts = r.get("ts")
        if ts_from is not None and (ts is None or ts < ts_from):
            continue
        if ts_to is not None and (ts is None or ts > ts_to):
            continue
        out.append(r)
    return out


def build_request(record: dict[str, Any], mode: str) -> dict[str, Any]:
    """Request body for one record. Deep copy: the record is never mutated."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    body = {
        k: copy.deepcopy(v)
        for k, v in record["payload"].items()
        if k.lower() not in SECRET_KEYS
    }
    recorded = body.pop("max_completion_tokens", None)
    if mode == "reuse":
        body["max_tokens"] = 1
        body["stream"] = False
        body.pop("stream_options", None)
    else:
        if recorded is not None and "max_tokens" not in body:
            body["max_tokens"] = recorded
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


# --------------------------------------------------------------------------
# Log + classification
# --------------------------------------------------------------------------


def parse_request_log(text: str) -> dict[str, Any]:
    """Facts from the server-log window of ONE request (replay is sequential)."""
    parsed = bm.parse_server_log(text)
    div = _RE_DIVERGE.search(text)
    src = _RE_RESTORE_SRC.search(text)
    ttft = _RE_TTFT.search(text)
    return {
        "cache_outcome": ttft.group(1) if ttft else None,
        "ttft_hold_s": float(ttft.group(2)) if ttft else None,
        "ttft_restore_s": float(ttft.group(3)) if ttft else None,
        "ttft_prefill_s": float(ttft.group(4)) if ttft else None,
        "ttft_first_decode_s": float(ttft.group(5)) if ttft else None,
        "divergence_position": int(div.group(4)) if div else None,
        "comparable_tokens": int(div.group(5)) if div else None,
        "log_reprefill_tokens": int(div.group(1)) if div else None,
        "restore_source": src.group(1) if src else None,
        "holds": parsed["holds"],
        "agings": parsed["agings"],
        "aborts": parsed["aborts_total"],
        "tool_parser_failures": sum(1 for ln in text.splitlines() if _RE_TOOL_FAIL.search(ln)),
        "pool_gb_max": parsed["pool_gb_max"],
    }


def _messages(record: dict[str, Any] | None) -> list[Any]:
    return list(((record or {}).get("payload") or {}).get("messages") or [])


def _has_image(msg: Any) -> bool:
    c = msg.get("content") if isinstance(msg, dict) else None
    return isinstance(c, list) and any(
        isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES for p in c
    )


def classify_reprefill(
    prev: dict[str, Any] | None,
    cur: dict[str, Any],
    prompt_tokens: int,
    cached_tokens: int,
    prev_prompt_tokens: int | None = None,
) -> str:
    """Heuristic cause of a re-prefill (priority: cold, compaction, system, image).

    ``prev`` is the previous request of the same session. The message-level
    common prefix is compared; the "new turn" is everything after it.
    """
    if prev is None:
        return "cold_start"
    if prompt_tokens - cached_tokens <= 0:
        return "hit"
    pm, cm = _messages(prev), _messages(cur)
    if len(cm) < len(pm) or (
        prev_prompt_tokens is not None and prompt_tokens < prev_prompt_tokens
    ):
        return "compaction"
    common = 0
    while common < min(len(pm), len(cm)) and pm[common] == cm[common]:
        common += 1
    changed = cm[common:]
    if any(_is_system(m) for m in changed):
        return "system_update"
    if any(_has_image(m) for m in changed):
        return "image"
    return "unknown"


def _is_system(msg: Any) -> bool:
    return isinstance(msg, dict) and msg.get("role") in ("system", "developer")


def quality_flags(text: str) -> list[str]:
    flags = []
    if "</think>" in text:
        flags.append("literal_think_close")
    if "<function=" in text:
        flags.append("literal_function")
    if "<parameter=" in text:
        flags.append("literal_parameter")
    return flags


# --------------------------------------------------------------------------
# Reducer
# --------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def reduce_results(
    rows: list[dict[str, Any]],
    mode: str,
    partial: bool = False,
    log_available: bool = True,
) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    missing = [r for r in ok if r.get("usage_missing")]
    measured = [r for r in ok if not r.get("usage_missing")]
    prompt = sum(r["prompt_tokens"] or 0 for r in measured)
    cached = sum(r["cached_tokens"] or 0 for r in measured)
    by_class: dict[str, dict[str, int]] = {}
    for r in measured:
        slot = by_class.setdefault(r["class"], {"requests": 0, "reprefilled_tokens": 0})
        slot["requests"] += 1
        slot["reprefilled_tokens"] += max(0, (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0))
    walls = [r["wall_s"] for r in ok if r.get("wall_s") is not None]

    def logsum(key: str) -> int | None:
        return sum(r.get(key) or 0 for r in rows) if log_available else None

    quality: dict[str, int | None] = {"tool_parser_failures": logsum("tool_parser_failures")}
    for r in rows:
        for flag in r.get("quality_flags", []):
            quality[flag] = (quality.get(flag) or 0) + 1
    out: dict[str, Any] = {
        "mode": mode,
        "partial": partial,
        "log_available": log_available,
        "requests": len(rows),
        "failed_requests": len(rows) - len(ok),
        "usage_missing": len(missing),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_ratio": (cached / prompt) if prompt else None,
        "reprefilled_tokens": prompt - cached,
        "by_class": by_class,
        "wall_p50_s": percentile(walls, 50),
        "wall_p90_s": percentile(walls, 90),
        "aborts": logsum("aborts"),
        "holds": logsum("holds"),
        "agings": logsum("agings"),
        "quality": quality,
    }
    if log_available:
        outcomes: dict[str, int] = {}
        for r in rows:
            if r.get("cache_outcome"):
                outcomes[r["cache_outcome"]] = outcomes.get(r["cache_outcome"], 0) + 1
        out["cache_outcomes"] = outcomes
        for stage in ("hold", "restore", "prefill", "first_decode"):
            vals = [r[f"ttft_{stage}_s"] for r in rows if r.get(f"ttft_{stage}_s") is not None]
            out[f"ttft_{stage}_p50_s"] = percentile(vals, 50)
    else:
        out["cache_outcomes"] = None
    if mode == "full":  # reuse mode is non-streaming: its latency is wall time, not TTFT
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        out["ttft_p50_s"] = percentile(ttfts, 50)
        out["ttft_p90_s"] = percentile(ttfts, 90)
    return out


_DETERMINISTIC_KEYS = ("prompt_tokens", "cached_tokens", "class")


def results_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Compare two replays on the deterministic fields only (no timings)."""
    if len(a) != len(b):
        return False
    return all(
        all(x.get(k) == y.get(k) for k in _DETERMINISTIC_KEYS) for x, y in zip(a, b)
    )


# --------------------------------------------------------------------------
# Replay loop (transport injected so it is testable with fakes)
# --------------------------------------------------------------------------


def replay(
    records: list[dict[str, Any]],
    client: Any,
    mode: str,
    log_window: Callable[[], str],
    progress: Callable[[int, int], None] | None = None,
    log_available: bool = True,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    done: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Replay sequentially. ``done`` maps trace index -> row already measured
    (resume): those requests are not re-sent. Each new row goes to ``on_row``
    as soon as it completes so an interruption loses nothing."""
    rows: list[dict[str, Any]] = []
    last: dict[Any, tuple[dict[str, Any], int | None]] = {}
    done = done or {}
    log_window()  # discard anything logged before the first request
    for i, record in enumerate(records):
        session = record.get("session")
        if i in done:
            prior = done[i]
            rows.append(prior)
            if prior.get("ok"):
                last[session] = (record, prior.get("prompt_tokens"))
            continue
        body = build_request(record, mode)
        res = client.send(body, mode)
        logtxt = log_window()
        info = (
            parse_request_log(logtxt) if log_available else {k: None for k in LOG_FIELDS}
        )
        prev, prev_pt = last.get(session, (None, None))
        ok = bool(res.get("ok"))
        missing = ok and bool(res.get("usage_missing"))
        pt, ct = res.get("prompt_tokens") or 0, res.get("cached_tokens") or 0
        if not ok:
            cls = "failed"
        elif missing:
            cls = "usage_missing"
        else:
            cls = classify_reprefill(prev, record, pt, ct, prev_pt)
        row = {
            "index": i,
            "ts": record.get("ts"),
            "session": session,
            "ok": ok,
            "error": res.get("error"),
            "usage_missing": missing,
            "prompt_tokens": None if missing else res.get("prompt_tokens"),
            "cached_tokens": None if missing else res.get("cached_tokens"),
            "ttft_s": res.get("ttft_s") if mode == "full" else None,
            "wall_s": res.get("wall_s"),
            "class": cls,
            "quality_flags": quality_flags(res.get("text") or "") if mode == "full" else [],
            **info,
        }
        rows.append(row)
        if on_row:
            on_row(row)
        if ok:
            last[session] = (record, None if missing else res.get("prompt_tokens"))
        if progress:
            progress(i + 1, len(records))
    return rows


def run_replay(
    records: list[dict[str, Any]],
    client: Any,
    mode: str,
    log_window: Callable[[], str],
    out: Path,
    resume: bool = False,
    log_available: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Replay with incremental ``<out>/rows.jsonl``. Returns (rows, partial).

    A KeyboardInterrupt or any exception stops the loop but keeps the rows
    gathered so far (``partial=True``) so the caller still writes a report.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if resume and path.exists():
        for ln in path.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # torn last line from a hard kill: that request is re-sent
            if isinstance(r, dict) and isinstance(r.get("index"), int):
                done[r["index"]] = r
    gathered: list[dict[str, Any]] = [done[k] for k in sorted(done) if k < len(records)]
    partial = False
    with open(path, "w" if not resume else "a") as fh:
        if resume:
            fh.seek(0, 2)

        def on_row(row: dict[str, Any]) -> None:
            gathered.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        try:
            replay(records, client, mode, log_window, progress, log_available, on_row, done)
        except (KeyboardInterrupt, Exception) as exc:
            partial = True
            print(f"[replay] interrupted ({type(exc).__name__}); writing partial report", file=sys.stderr)
    gathered.sort(key=lambda r: r["index"])
    return gathered, partial


class HttpClient:
    """Real transport: sync httpx against the bench server."""

    def __init__(self, base_url: str, headers: dict[str, str], timeout_s: float = 3600.0):
        import httpx

        self._httpx = httpx
        self.url = f"{base_url}/v1/chat/completions"
        self.headers = headers
        self.timeout = httpx.Timeout(timeout_s, connect=60.0)

    def send(self, body: dict[str, Any], mode: str) -> dict[str, Any]:
        t0 = time.time()
        try:
            if mode == "reuse":
                r = self._httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
                wall = time.time() - t0
                if r.status_code != 200:
                    return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "wall_s": wall}
                usage = r.json().get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                pt, ct = usage.get("prompt_tokens"), details.get("cached_tokens")
                # non-streaming: only wall time is measured, never TTFT
                return {
                    "ok": True, "error": None, "text": "", "wall_s": wall,
                    "usage_missing": pt is None or ct is None,
                    "prompt_tokens": pt, "cached_tokens": ct,
                }
            acc = bm.StreamAccumulator()
            with self._httpx.stream("POST", self.url, json=body, headers=self.headers, timeout=self.timeout) as r:
                if r.status_code != 200:
                    r.read()
                    return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "wall_s": time.time() - t0}
                for line in r.iter_lines():
                    acc.feed(line, time.time())
            fin = acc.finalize()
            pt, ct = fin["prompt_tokens"], fin["cached_tokens"]
            usage_missing = acc.usage is None or pt is None or ct is None
            ok, error = fin["ok"], fin["error"]
            if not ok and usage_missing and not acc.errors and acc.done and fin["text"]:
                ok, error = True, None  # only the usage chunk is absent: flagged, not a hit
            return {
                "ok": ok, "error": error, "text": fin["text"],
                "usage_missing": ok and usage_missing,
                "wall_s": time.time() - t0,
                "ttft_s": None if fin["first_token_at"] is None else fin["first_token_at"] - t0,
                "prompt_tokens": pt, "cached_tokens": ct,
            }
        except Exception as exc:  # network failure is a failed request, not a crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "wall_s": time.time() - t0}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    def f(v: Any, nd: int = 3) -> str:
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [f"# Trace replay: {meta.get('variant')} / {summary['mode']}", ""]
    if summary.get("partial"):
        lines += ["**PARTIAL RESULT: the run was interrupted; these rows are incomplete.**", ""]
    if summary.get("usage_missing"):
        lines += [
            f"**WARNING: {summary['usage_missing']} request(s) returned no usage block "
            "(usage_missing). They are excluded from hit ratio and re-prefill totals.**",
            "",
        ]
    lines += [
        f"- requests: {summary['requests']} (failed {summary['failed_requests']}, "
        f"usage_missing {summary.get('usage_missing', 0)}), aborts {f(summary['aborts'])}",
        f"- prompt tokens {summary['prompt_tokens']}, cached {summary['cached_tokens']}, "
        f"hit ratio (token-weighted) {f(summary['hit_ratio'])}",
        f"- re-prefilled tokens {summary['reprefilled_tokens']}",
    ]
    if "ttft_p50_s" in summary:
        lines.append(f"- TTFT p50 {f(summary['ttft_p50_s'])} s, p90 {f(summary['ttft_p90_s'])} s")
    lines += [
        f"- wall time p50 {f(summary.get('wall_p50_s'))} s, p90 {f(summary.get('wall_p90_s'))} s",
        f"- peak memory {f(meta.get('peak_memory_gb'))} GB, holds {f(summary['holds'])}, agings {f(summary['agings'])}",
        f"- quality: {json.dumps(summary['quality'], sort_keys=True)}",
    ]
    if summary.get("cache_outcomes"):
        counts = ", ".join(f"{k} {v}" for k, v in sorted(summary["cache_outcomes"].items()))
        lines.append(f"- cache outcomes: {counts}")
    stages = [
        f"{st} {f(summary.get(f'ttft_{st}_p50_s'))} s"
        for st in ("hold", "restore", "prefill", "first_decode")
        if summary.get(f"ttft_{st}_p50_s") is not None
    ]
    if stages:
        lines.append(f"- TTFT stage p50: {', '.join(stages)}")
    if not summary.get("log_available", True):
        lines.append("- log_available: false (log-derived fields are n/a)")
    lines += [
        "",
        "| class | requests | re-prefilled tokens |",
        "|---|---:|---:|",
    ]
    for cls in CLASSES:
        if cls in summary["by_class"]:
            s = summary["by_class"][cls]
            lines.append(f"| {cls} | {s['requests']} | {s['reprefilled_tokens']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="recorder JSONL")
    ap.add_argument("--variant", required=True, help="fork | upstream | fork-off | ... (bench_premises variants)")
    ap.add_argument("--mode", choices=MODES, default="reuse")
    ap.add_argument("--session", default=None, help="only this session id")
    ap.add_argument("--from-ts", type=float, default=None, help="only records with ts >= this (ms epoch)")
    ap.add_argument("--to-ts", type=float, default=None, help="only records with ts <= this (ms epoch)")
    ap.add_argument("--limit", type=int, default=0, help="replay at most N records (0 = all)")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--allow-port", type=int, action="append", default=[])
    ap.add_argument("--attach", action="store_true",
                    help="use a server already running on --port (started with bench_premises.py --serve)")
    ap.add_argument("--server-log", default=None, help="with --attach: that server's server.log")
    ap.add_argument("--api-key", default="", help="with --attach: bearer key of that server")
    ap.add_argument("--cache-root", default=None, help="parent of the bench SSD cache dir")
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--out", default=None, help="output dir (default: DEFAULT_OUT/replay)")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip trace rows already in <out>/rows.jsonl (same trace and filters; "
                         "the server cache is cold after a restart, so resumed rows are not "
                         "comparable to an uninterrupted run)")
    args = ap.parse_args(argv)

    if not bm.port_allowed(args.port, args.allow_port):
        print(f"refusing protected port {args.port} (pass --allow-port {args.port} to override)", file=sys.stderr)
        return 2
    trace = Path(args.trace)
    if not trace.is_file():
        print(f"trace not found: {trace}", file=sys.stderr)
        return 2
    records = filter_records(
        parse_trace(trace.read_text().splitlines()), args.session, args.from_ts, args.to_ts
    )
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("no records after filtering", file=sys.stderr)
        return 2

    import bench_premises as bp  # heavy import; only after cheap checks pass

    if not args.attach and bp.port_in_use(args.port):
        print(f"port {args.port} already accepts connections; refusing to start a server there", file=sys.stderr)
        return 2
    if args.attach and not bp.port_in_use(args.port):
        print(f"--attach but nothing listens on port {args.port}", file=sys.stderr)
        return 2
    if args.cache_root:
        bp.CACHE_ROOT = args.cache_root
    out = Path(args.out or Path(bp.DEFAULT_OUT) / "replay")
    (out / "logs").mkdir(parents=True, exist_ok=True)

    srv = None
    sampler = None
    if args.attach:
        base_url, headers = f"http://127.0.0.1:{args.port}", (
            {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        )
        log_path = Path(args.server_log) if args.server_log else None
        if log_path is None:
            print(f"[replay] WARNING: {LOG_UNAVAILABLE_WARNING}", file=sys.stderr)
    else:
        srv = bp.Server(args.variant, "REPLAY", args.port, args.max_concurrent, out, args.keep_cache)
        srv.prepare()
        srv.start()
        base_url, headers, log_path = srv.base_url, srv.headers(), srv.log_path
    try:
        if srv is not None:
            srv.wait_ready()
            import asyncio

            asyncio.run(bp._load_model(srv))
            sampler = bp.Sampler(srv)
            sampler.start()
        offset = [log_path.stat().st_size if log_path and log_path.exists() else 0]

        def window() -> str:
            if not log_path or not log_path.exists():
                return ""
            with open(log_path, "rb") as fh:
                fh.seek(offset[0])
                data = fh.read()
            offset[0] += len(data)
            return data.decode("utf-8", "replace")

        log_available = log_path is not None
        rows, partial = run_replay(
            records, HttpClient(base_url, headers), args.mode, window, out,
            resume=args.resume, log_available=log_available,
            progress=lambda i, n: print(f"[replay] {i}/{n}", flush=True),
        )
    finally:
        samples = sampler.stop() if sampler else []
        if srv is not None:
            srv.stop()
            srv.archive_and_clean()
    peaks = [s["footprint_gb"] for s in samples if s.get("footprint_gb") is not None]
    summary = reduce_results(rows, args.mode, partial=partial, log_available=log_available)
    meta = {
        "variant": args.variant, "trace": str(trace), "session": args.session,
        "peak_memory_gb": max(peaks) if peaks else None,
        "port": args.port, "attached": args.attach,
        "partial": partial, "log_available": log_available,
    }
    stem = f"replay-{args.variant}-{args.mode}-{int(time.time())}"
    (out / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    (out / f"{stem}.md").write_text(render_markdown(summary, meta))
    print(render_markdown(summary, meta))
    print(f"[replay] wrote {out / stem}.json / .md", flush=True)
    if summary["usage_missing"]:
        print(f"[replay] WARNING: {summary['usage_missing']} request(s) had no usage block", file=sys.stderr)
    return 0 if summary["failed_requests"] == 0 and not partial else 1


if __name__ == "__main__":
    sys.exit(main())
