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

_RE_DIVERGE = re.compile(
    r"prefix cache: request \S+ re-prefills (\d+) of (\d+) tokens \(reused (\d+)\); "
    r"closest stored sequence \S+ shares the first (\d+) of (\d+) comparable tokens"
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
    return {
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


def _text(msg: Any) -> str:
    c = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


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


def reduce_results(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    cached = sum(r["cached_tokens"] or 0 for r in ok)
    by_class: dict[str, dict[str, int]] = {}
    for r in ok:
        slot = by_class.setdefault(r["class"], {"requests": 0, "reprefilled_tokens": 0})
        slot["requests"] += 1
        slot["reprefilled_tokens"] += max(0, (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0))
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None] if mode == "full" else []
    quality: dict[str, int] = {"tool_parser_failures": sum(r.get("tool_parser_failures", 0) for r in rows)}
    for r in rows:
        for flag in r.get("quality_flags", []):
            quality[flag] = quality.get(flag, 0) + 1
    return {
        "mode": mode,
        "requests": len(rows),
        "failed_requests": len(rows) - len(ok),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_ratio": (cached / prompt) if prompt else None,
        "reprefilled_tokens": prompt - cached,
        "by_class": by_class,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p90_s": percentile(ttfts, 90),
        "aborts": sum(r.get("aborts", 0) for r in rows),
        "holds": sum(r.get("holds", 0) for r in rows),
        "agings": sum(r.get("agings", 0) for r in rows),
        "quality": quality,
    }


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last: dict[Any, tuple[dict[str, Any], int | None]] = {}
    log_window()  # discard anything logged before the first request
    for i, record in enumerate(records):
        body = build_request(record, mode)
        res = client.send(body, mode)
        logtxt = log_window()
        info = parse_request_log(logtxt)
        session = record.get("session")
        prev, prev_pt = last.get(session, (None, None))
        pt, ct = res.get("prompt_tokens") or 0, res.get("cached_tokens") or 0
        row = {
            "index": i,
            "ts": record.get("ts"),
            "session": session,
            "ok": bool(res.get("ok")),
            "error": res.get("error"),
            "prompt_tokens": res.get("prompt_tokens"),
            "cached_tokens": res.get("cached_tokens"),
            "ttft_s": res.get("ttft_s"),
            "wall_s": res.get("wall_s"),
            "class": classify_reprefill(prev, record, pt, ct, prev_pt) if res.get("ok") else "failed",
            "quality_flags": quality_flags(res.get("text") or "") if mode == "full" else [],
            **info,
        }
        rows.append(row)
        if res.get("ok"):
            last[session] = (record, res.get("prompt_tokens"))
        if progress:
            progress(i + 1, len(records))
    return rows


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
                return {
                    "ok": True, "error": None, "text": "", "wall_s": wall, "ttft_s": wall,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": details.get("cached_tokens"),
                }
            acc = bm.StreamAccumulator()
            with self._httpx.stream("POST", self.url, json=body, headers=self.headers, timeout=self.timeout) as r:
                if r.status_code != 200:
                    r.read()
                    return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "wall_s": time.time() - t0}
                for line in r.iter_lines():
                    acc.feed(line, time.time())
            fin = acc.finalize()
            return {
                "ok": fin["ok"], "error": fin["error"], "text": fin["text"],
                "wall_s": time.time() - t0,
                "ttft_s": None if fin["first_token_at"] is None else fin["first_token_at"] - t0,
                "prompt_tokens": fin["prompt_tokens"], "cached_tokens": fin["cached_tokens"],
            }
        except Exception as exc:  # network failure is a failed request, not a crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "wall_s": time.time() - t0}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    def f(v: Any, nd: int = 3) -> str:
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [
        f"# Trace replay: {meta.get('variant')} / {summary['mode']}",
        "",
        f"- requests: {summary['requests']} (failed {summary['failed_requests']}), aborts {summary['aborts']}",
        f"- prompt tokens {summary['prompt_tokens']}, cached {summary['cached_tokens']}, "
        f"hit ratio (token-weighted) {f(summary['hit_ratio'])}",
        f"- re-prefilled tokens {summary['reprefilled_tokens']}",
        f"- TTFT p50 {f(summary['ttft_p50_s'])} s, p90 {f(summary['ttft_p90_s'])} s",
        f"- peak memory {f(meta.get('peak_memory_gb'))} GB, holds {summary['holds']}, agings {summary['agings']}",
        f"- quality: {json.dumps(summary['quality'], sort_keys=True)}",
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

        rows = replay(
            records, HttpClient(base_url, headers), args.mode, window,
            progress=lambda i, n: print(f"[replay] {i}/{n}", flush=True),
        )
    finally:
        samples = sampler.stop() if sampler else []
        if srv is not None:
            srv.stop()
            srv.archive_and_clean()
    peaks = [s["footprint_gb"] for s in samples if s.get("footprint_gb") is not None]
    summary = reduce_results(rows, args.mode)
    meta = {
        "variant": args.variant, "trace": str(trace), "session": args.session,
        "peak_memory_gb": max(peaks) if peaks else None,
        "port": args.port, "attached": args.attach,
    }
    stem = f"replay-{args.variant}-{args.mode}-{int(time.time())}"
    (out / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    (out / f"{stem}.md").write_text(render_markdown(summary, meta))
    print(render_markdown(summary, meta))
    print(f"[replay] wrote {out / stem}.json / .md", flush=True)
    return 0 if summary["failed_requests"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
