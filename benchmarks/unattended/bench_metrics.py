# SPDX-License-Identifier: Apache-2.0
"""Pure logic for the unattended-scheduling premise benchmark.

Everything here is side-effect free so it can be unit tested with fixtures:
server-log parsing, SSE stream accounting, rate computation, `footprint`
output parsing, prompt sizing helpers, safety guards and the PASS/FAIL
evaluation of each premise.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import struct
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Server log parsing
# --------------------------------------------------------------------------

_RE_ABORT = re.compile(r"[Aa]borted (\d+) request")
_RE_ABORT_HARD = re.compile(r"aborted (\d+) request\(s\)")
_RE_POOL = re.compile(r"pool=([\d.]+)\s*(GB|MB)")
_RE_PRESSURE = re.compile(r"Memory pressure level: (\w+) -> (\w+)")
_RE_PARKED = re.compile(r"Parked request \S+ at (\d+) tokens committed")
_RE_PARK_SKIP = re.compile(r"Park skipped for ")
_RE_RESUMED = re.compile(r"Resumed parked request ")
_RE_HOLD = re.compile(r"Holding \S+ from batched decode")
_RE_RELEASE = re.compile(r"Releasing \S+ to decode")
_RE_BACKFILL = re.compile(r"Backfilling \S+ past held ")
_RE_AGING = re.compile(r"Aging: \S+ held \d+ s")
_RE_RESTORE = re.compile(r"Prefix cache restore for \S+: source=\S+ cached=(\d+)")
_RE_SSD_EVICTED = re.compile(
    r"SSD cache evicted: blocks=(\d+) bytes=(\d+) sidecars=(\d+)"
)
_RE_CACHE_LIMIT = re.compile(r"MLX buffer cache limit set to ([\d.]+) GB")
_RE_TQ_FUSED = re.compile(
    r"TurboQuant batched attention: (?:per-row fused|ragged single-launch) path "
    r"\(B=(\d+)\)"
)
_RE_MTP_BATCH = re.compile(r"Lightning MTP batch activated for (\d+) sequences")
_RE_TQ_FALLBACK = re.compile(
    r"TurboQuant batched attention: per-row dequantize fallback"
)


def parse_server_log(text: str) -> dict[str, Any]:
    """Extract the benchmark-relevant facts from a window of server log."""
    aborts_total = 0
    aborts_memory = 0
    pool: list[float] = []
    transitions: list[list[str]] = []
    levels: list[str] = []
    parked_tokens: list[int] = []
    park_skips = park_resumes = holds = releases = 0
    backfills = agings = 0
    restores: list[int] = []
    cache_limit: float | None = None
    tq_rows: list[int] = []
    tq_fallback = False
    mtp_rows: list[int] = []
    ssd_init = False
    ssd_disabled = False
    evicted = [0, 0, 0]  # blocks, bytes, sidecars from rate-limited summaries

    for line in text.splitlines():
        m = _RE_ABORT.search(line) or _RE_ABORT_HARD.search(line)
        if m:
            n = int(m.group(1))
            aborts_total += n
            low = line.lower()
            if "memory" in low or "pressure" in low:
                aborts_memory += n
        m = _RE_POOL.search(line)
        if m:
            v = float(m.group(1))
            pool.append(round(v / 1024, 3) if m.group(2) == "MB" else v)
        m = _RE_PRESSURE.search(line)
        if m:
            transitions.append([m.group(1), m.group(2)])
            for lvl in (m.group(1), m.group(2)):
                if lvl not in levels:
                    levels.append(lvl)
        m = _RE_PARKED.search(line)
        if m:
            parked_tokens.append(int(m.group(1)))
        if _RE_PARK_SKIP.search(line):
            park_skips += 1
        if _RE_RESUMED.search(line):
            park_resumes += 1
        if _RE_HOLD.search(line):
            holds += 1
        if _RE_RELEASE.search(line):
            releases += 1
        if _RE_BACKFILL.search(line):
            backfills += 1
        if _RE_AGING.search(line):
            agings += 1
        m = _RE_RESTORE.search(line)
        if m:
            restores.append(int(m.group(1)))
        m = _RE_SSD_EVICTED.search(line)
        if m:
            for i in range(3):
                evicted[i] += int(m.group(i + 1))
        m = _RE_CACHE_LIMIT.search(line)
        if m:
            cache_limit = float(m.group(1))
        m = _RE_TQ_FUSED.search(line)
        if m:
            tq_rows.append(int(m.group(1)))
        m = _RE_MTP_BATCH.search(line)
        if m:
            mtp_rows.append(int(m.group(1)))
        if _RE_TQ_FALLBACK.search(line):
            tq_fallback = True
        if "PagedSSDCacheManager initialized" in line:
            ssd_init = True
        if "cache disabled" in line.lower() or "Operation not permitted" in line:
            ssd_disabled = True

    return {
        "aborts_total": aborts_total,
        "aborts_memory": aborts_memory,
        "pool_gb_values": pool,
        "pool_gb_max": max(pool) if pool else None,
        "pressure_transitions": transitions,
        "pressure_levels_seen": levels,
        "parks": len(parked_tokens),
        "parked_tokens": parked_tokens,
        "park_skips": park_skips,
        "park_resumes": park_resumes,
        "holds": holds,
        "releases": releases,
        "backfills": backfills,
        "agings": agings,
        "restores": restores,
        "restore_cached_last": restores[-1] if restores else None,
        "restore_cached_max": max(restores) if restores else None,
        "mlx_cache_limit_gb": cache_limit,
        "tq_batched_fused_rows": tq_rows,
        "tq_batched_fused_max_b": max(tq_rows) if tq_rows else 0,
        "tq_batched_fallback": tq_fallback,
        "mtp_batch_rows": mtp_rows,
        "mtp_batch_max": max(mtp_rows) if mtp_rows else 0,
        # Largest batch seen by either log signal (fork TQ path or MTP batch).
        "batch_max_b": max([*tq_rows, *mtp_rows], default=0),
        "ssd_initialized": ssd_init and not ssd_disabled,
        "ssd_disabled": ssd_disabled,
        "ssd_evicted_blocks": evicted[0],
        "ssd_evicted_bytes": evicted[1],
        "ssd_evicted_sidecars": evicted[2],
    }


# --------------------------------------------------------------------------
# P7: park-instead-of-abort validation
# --------------------------------------------------------------------------

_RE_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_RE_EV_PRESSURE = re.compile(
    r"Memory pressure level: (\w+) -> (\w+) \(current=([\d.]+)GB"
    r"(?:.*?ceiling=([\d.]+)GB)?"
)
_RE_EV_HARD_POLL = re.compile(
    r"Hard memory pressure: parking a prefill victim on (\d+) scheduler\(s\) "
    r"instead of aborting \(poll (\d+)/(\d+)\)"
)
_RE_EV_PARKED = re.compile(
    r"Parked request (\S+) at (\d+) tokens committed "
    r"\(park (\d+)/(\d+)\), reason: (.*)$"
)
_RE_EV_RESUMED = re.compile(r"Resumed parked request (\S+): cached=(\d+)")
_RE_EV_SKIPPED = re.compile(
    r"Park skipped for (\S+): (.*?) \(memory pressure falls back to abort\)"
)
_RE_EV_ABORT = re.compile(r"Aborted (\d+) requests? due to memory pressure")


def _log_ts(line: str) -> float | None:
    m = _RE_LOG_TS.match(line)
    if not m:
        return None
    try:
        base = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return None
    return base + int(m.group(2)) / 1000.0


def parse_park_events(text: str) -> list[dict[str, Any]]:
    """Ordered park-relevant events (with timestamps) from a server log window."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        ts = _log_ts(line)
        m = _RE_EV_PRESSURE.search(line)
        if m:
            events.append(
                {
                    "kind": "pressure",
                    "ts": ts,
                    "from": m.group(1),
                    "to": m.group(2),
                    "current_gb": float(m.group(3)),
                    "ceiling_gb": float(m.group(4)) if m.group(4) else None,
                }
            )
            continue
        m = _RE_EV_HARD_POLL.search(line)
        if m:
            events.append(
                {
                    "kind": "hard_park_poll",
                    "ts": ts,
                    "schedulers": int(m.group(1)),
                    "poll": int(m.group(2)),
                    "max": int(m.group(3)),
                }
            )
            continue
        m = _RE_EV_PARKED.search(line)
        if m:
            events.append(
                {
                    "kind": "parked",
                    "ts": ts,
                    "rid": m.group(1),
                    "committed": int(m.group(2)),
                    "park_n": int(m.group(3)),
                    "budget": int(m.group(4)),
                    "reason": m.group(5).strip(),
                }
            )
            continue
        m = _RE_EV_RESUMED.search(line)
        if m:
            events.append(
                {"kind": "resumed", "ts": ts, "rid": m.group(1), "cached": int(m.group(2))}
            )
            continue
        m = _RE_EV_SKIPPED.search(line)
        if m:
            events.append(
                {"kind": "park_skipped", "ts": ts, "rid": m.group(1), "reason": m.group(2)}
            )
            continue
        m = _RE_EV_ABORT.search(line)
        if m:
            events.append({"kind": "abort", "ts": ts, "n": int(m.group(1))})
    return events


def build_park_tracks(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per server request id: parks, tokens committed, resumes, time parked."""
    tracks: dict[str, dict[str, Any]] = {}
    open_at: dict[str, list[float | None]] = {}
    for e in events:
        rid = e.get("rid")
        if e["kind"] == "parked":
            t = tracks.setdefault(
                rid,
                {
                    "parks": 0,
                    "parked_tokens": [],
                    "resume_cached": [],
                    "parked_s": None,
                    "unresumed": 0,
                },
            )
            t["parks"] += 1
            t["parked_tokens"].append(e["committed"])
            open_at.setdefault(rid, []).append(e["ts"])
        elif e["kind"] == "resumed" and rid in tracks:
            t = tracks[rid]
            t["resume_cached"].append(e["cached"])
            pending = open_at.get(rid) or []
            started = pending.pop(0) if pending else None
            if started is not None and e["ts"] is not None:
                t["parked_s"] = round((t["parked_s"] or 0.0) + e["ts"] - started, 3)
    for rid, pending in open_at.items():
        tracks[rid]["unresumed"] = len(pending)
    return tracks


def attribute_rids(
    tracks: dict[str, dict[str, Any]], requests: list[dict[str, Any]]
) -> dict[str, str]:
    """Best-effort client request name -> server rid.

    The server never exposes its internal request id to the client, so a
    resumed request is matched through its restored token count: the resume
    ``cached=N`` equals the ``cached_tokens`` of exactly one client request.
    Ambiguous or missing matches are left out rather than guessed.
    """
    out: dict[str, str] = {}
    for req in requests:
        cached = req.get("cached_tokens") or 0
        if cached <= 0:
            continue
        rids = [
            rid for rid, t in tracks.items() if cached in (t.get("resume_cached") or [])
        ]
        clash = [
            r
            for r in requests
            if r is not req and (r.get("cached_tokens") or 0) == cached
        ]
        if len(rids) == 1 and not clash:
            out[req["name"]] = rids[0]
    return out


def classify_p7(s: dict[str, Any]) -> str:
    """One-word outcome of a P7 run, from a summary's counters."""
    aborted = s.get("aborted_requests", 0) > 0
    if s.get("parks", 0) > 0:
        if s.get("resumes", 0) < s["parks"]:
            return "parked_not_resumed"
        return "park_fired_with_aborts" if aborted else "park_fired_ok"
    if s.get("park_skips", 0) > 0:
        return "park_skipped_then_abort" if aborted else "park_skipped_survived"
    if s.get("hard_pressure_seen"):
        return "hard_pressure_aborted" if aborted else "hard_pressure_survived"
    if aborted:
        return "aborted_without_pressure_log"
    return "no_pressure"


def summarize_p7(
    requests: list[dict[str, Any]],
    reissues: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce a P7 run: per-request rows plus run-level park counters."""
    tracks = build_park_tracks(events)
    names = attribute_rids(tracks, requests)
    rows: list[dict[str, Any]] = []
    for r in requests:
        rid = names.get(r["name"])
        track = tracks.get(rid, {}) if rid else {}
        re_ = reissues.get(r["name"])
        ratio = distinct_ngram_ratio(r.get("text") or "")
        wall = None
        if r.get("end_s") is not None and r.get("start_s") is not None:
            wall = round(r["end_s"] - r["start_s"], 2)
        ok = bool(r.get("ok"))
        rows.append(
            {
                "name": r["name"],
                "rid": rid,
                "ok": ok,
                "error": r.get("error"),
                "aborted": not ok,
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "cached_tokens": r.get("cached_tokens"),
                "ttft_s": r.get("ttft_s"),
                "wall_s": wall,
                "parks": track.get("parks", 0),
                "parked_tokens": track.get("parked_tokens", []),
                "resume_cached": track.get("resume_cached", []),
                "parked_s": track.get("parked_s"),
                "distinct_4gram": round(ratio, 4) if ratio is not None else None,
                "degenerate": ratio is not None and ratio < P3_DEGEN_FLOOR,
                "reissue_ok": bool(re_.get("ok")) if re_ else None,
                "reissue_cached": re_.get("cached_tokens") if re_ else None,
                "reissue_ttft_s": re_.get("ttft_s") if re_ else None,
                "final_ok": ok or bool(re_ and re_.get("ok")),
            }
        )
    summary: dict[str, Any] = {
        "rows": rows,
        "parks": sum(1 for e in events if e["kind"] == "parked"),
        "resumes": sum(1 for e in events if e["kind"] == "resumed"),
        "park_skips": sum(1 for e in events if e["kind"] == "park_skipped"),
        "park_skip_reasons": sorted(
            {e["reason"] for e in events if e["kind"] == "park_skipped"}
        ),
        "hard_park_polls": sum(1 for e in events if e["kind"] == "hard_park_poll"),
        "hard_pressure_seen": any(
            e["kind"] == "pressure" and e["to"] == "hard" for e in events
        ),
        "aborts_logged": sum(e["n"] for e in events if e["kind"] == "abort"),
        "aborted_requests": sum(1 for row in rows if row["aborted"]),
        "reissue_cached": [
            row["reissue_cached"] for row in rows if row["reissue_cached"] is not None
        ],
        "all_final_ok": all(row["final_ok"] for row in rows),
        "events": events,
    }
    summary["verdict"] = classify_p7(summary)
    return summary


# --------------------------------------------------------------------------
# C7: pause-resume cycle validation (decode pressure, images, never-fits)
# --------------------------------------------------------------------------
# Log lines come from the feat/pause-resume-cycle branch:
#   scheduler.py       "Paused decoding request", "Resumed decoding request",
#                      "Parked request ... waiting", "Resuming parked request"
#   process_memory_enforcer.py  "Hard memory pressure: pausing decoding victim",
#                      "... aborted 1 victim", "... waiting for victim",
#                      "... aborted N request(s)" (abort-all)

_RE_EV_DEC_PAUSED = re.compile(
    r"Paused decoding request (\S+) at (\d+) tokens "
    r"\(prompt (\d+) \+ output (\d+)\), stored exact prefix"
)
_RE_EV_DEC_RESUMED = re.compile(
    r"Resumed decoding request (\S+): restored (\d+) tokens exactly"
)
_RE_EV_VICTIM_PAUSE = re.compile(
    r"Hard memory pressure: pausing decoding victim (\S+) instead of "
    r"aborting \((\d+) tokens, ~([\d.]+) GB\)"
)
_RE_EV_VICTIM_ABORT = re.compile(
    r"Hard memory pressure: aborted 1 victim (\S+) \((\d+) tokens, ~([\d.]+) GB\)"
)
_RE_EV_DRAIN = re.compile(
    r"Hard memory pressure: waiting for victim (\S+) memory to drain"
)
_RE_EV_PARK_WAIT = re.compile(
    r"Parked request (\S+) waiting: need ([\d.]+) GB, free ([\d.]+) GB"
)
_RE_EV_RESUME_FIT = re.compile(
    r"Resuming parked request (\S+): need ([\d.]+) GB fits in free ([\d.]+) GB"
)
_RE_EV_ABORT_ALL = re.compile(r"Hard memory pressure: aborted (\d+) request\(s\) on")


def parse_c7_events(text: str) -> list[dict[str, Any]]:
    """Ordered pause/park/abort events: the P7 kinds plus the C7 log lines."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        ts = _log_ts(line)
        m = _RE_EV_DEC_PAUSED.search(line)
        if m:
            events.append(
                {
                    "kind": "decode_paused",
                    "ts": ts,
                    "rid": m.group(1),
                    "tokens": int(m.group(2)),
                    "prompt": int(m.group(3)),
                    "output": int(m.group(4)),
                }
            )
            continue
        m = _RE_EV_DEC_RESUMED.search(line)
        if m:
            events.append(
                {
                    "kind": "decode_resumed",
                    "ts": ts,
                    "rid": m.group(1),
                    "restored": int(m.group(2)),
                }
            )
            continue
        m = _RE_EV_VICTIM_PAUSE.search(line) or _RE_EV_VICTIM_ABORT.search(line)
        if m:
            kind = "victim_pause" if "pausing" in line else "victim_abort"
            events.append(
                {
                    "kind": kind,
                    "ts": ts,
                    "rid": m.group(1),
                    "tokens": int(m.group(2)),
                    "est_gb": float(m.group(3)),
                }
            )
            continue
        m = _RE_EV_DRAIN.search(line)
        if m:
            events.append({"kind": "drain_wait", "ts": ts, "rid": m.group(1)})
            continue
        m = _RE_EV_PARK_WAIT.search(line)
        if m:
            events.append(
                {
                    "kind": "park_wait",
                    "ts": ts,
                    "rid": m.group(1),
                    "need_gb": float(m.group(2)),
                    "free_gb": float(m.group(3)),
                }
            )
            continue
        m = _RE_EV_RESUME_FIT.search(line)
        if m:
            events.append(
                {
                    "kind": "resume_fit",
                    "ts": ts,
                    "rid": m.group(1),
                    "need_gb": float(m.group(2)),
                    "free_gb": float(m.group(3)),
                }
            )
            continue
        m = _RE_EV_ABORT_ALL.search(line)
        if m:
            events.append({"kind": "abort_all", "ts": ts, "n": int(m.group(1))})
            continue
        events.extend(parse_park_events(line))
    return events


def attribute_decode_pauses(
    events: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Client request name -> its decode pauses, matched by prompt tokens.

    The log prints the server's prompt token count; it equals the client's
    ``prompt_tokens`` usage. A count shared by two requests is ambiguous and
    left out rather than guessed.
    """
    by_prompt: dict[int, list[str]] = {}
    for r in requests:
        if r.get("prompt_tokens"):
            by_prompt.setdefault(r["prompt_tokens"], []).append(r["name"])
    rid_name: dict[str, str] = {}
    out: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["kind"] != "decode_paused":
            continue
        names = by_prompt.get(e["prompt"], [])
        if len(names) != 1:
            continue
        rid_name[e["rid"]] = names[0]
        t = out.setdefault(names[0], {"pauses": 0, "resumes": 0, "restored": []})
        t["pauses"] += 1
    for e in events:
        if e["kind"] == "decode_resumed" and e["rid"] in rid_name:
            t = out[rid_name[e["rid"]]]
            t["resumes"] += 1
            t["restored"].append(e["restored"])
    return out


def max_decode_stall(chunk_times: list[float]) -> float | None:
    """Largest gap between two streamed chunks of one request (seconds)."""
    if len(chunk_times) < 2:
        return None
    return max(b - a for a, b in zip(chunk_times, chunk_times[1:], strict=False))


def identity_verdict(
    ref: dict[str, Any],
    test: dict[str, Any],
    ref_lp: list[dict] | None = None,
    test_lp: list[dict] | None = None,
) -> dict[str, Any]:
    """Greedy identity of a request against its uninterrupted run.

    ``classification`` is None when identical; with logprob dumps it is the
    near-tie verdict (noise / bug / inconclusive), otherwise ``unclassified``.
    """
    ref_text, test_text = ref.get("text") or "", test.get("text") or ""
    div = first_divergence(ref_text, test_text)
    out: dict[str, Any] = {
        "identical": div is None,
        "first_divergence_chars": div,
        "ref_tokens": ref.get("completion_tokens"),
        "test_tokens": test.get("completion_tokens"),
        "classification": None,
    }
    if div is None:
        return out
    out["classification"] = "unclassified"
    if ref_lp and test_lp:
        cmp = compare_logprobs(ref_lp, test_lp)
        out.update(
            classification=classify_divergence(cmp) or "unclassified",
            div_token=cmp["div_index"],
            margin=cmp["margin"],
            mean_abs_dlp=cmp["mean_abs_dlp"],
        )
    return out


NEVER_FITS_FIELDS = (
    "required_bytes",
    "capacity_bytes",
    "weights_bytes",
    "kv_bytes_per_token",
    "requested_context_tokens",
    "requested_max_tokens",
    "max_viable_context_tokens",
)
NEVER_FITS_MAX_S = 60.0  # "within one admission": the answer must be prompt
_NEVER_FITS_FORBIDDEN = ("parked", "decode_paused", "park_wait", "victim_pause")


def evaluate_never_fits(
    status: int | None,
    body: Any,
    elapsed_s: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Did a never-fits request get the structured 400 in one admission?"""
    reasons: list[str] = []
    err = body.get("error") if isinstance(body, dict) else None
    err = err if isinstance(err, dict) else {}
    if status != 400:
        reasons.append(f"status {status} != 400")
    if err.get("omlx_code") != "context_exceeds_capacity":
        reasons.append(f"omlx_code {err.get('omlx_code')!r} != context_exceeds_capacity")
    for f in NEVER_FITS_FIELDS:
        if not isinstance(err.get(f), int) or isinstance(err.get(f), bool):
            reasons.append(f"missing or non-integer field {f}")
    if err.get("retryable") is not False:
        reasons.append("retryable is not false")
    req, cap = err.get("required_bytes"), err.get("capacity_bytes")
    if isinstance(req, int) and isinstance(cap, int) and req <= cap:
        reasons.append("required_bytes <= capacity_bytes (it would fit)")
    ctx_t, mt = err.get("requested_context_tokens"), err.get("requested_max_tokens")
    viable = err.get("max_viable_context_tokens")
    if all(isinstance(x, int) for x in (ctx_t, mt, viable)) and viable >= ctx_t + mt:
        reasons.append("max_viable_context_tokens >= requested tokens")
    if elapsed_s > NEVER_FITS_MAX_S:
        reasons.append(f"answered after {elapsed_s:.1f}s, not within one admission")
    bad = sorted({e["kind"] for e in events if e["kind"] in _NEVER_FITS_FORBIDDEN})
    if bad:
        reasons.append(f"park or pause events during a never-fits request: {bad}")
    return {
        "correct": not reasons,
        "reasons": reasons,
        "status": status,
        "elapsed_s": round(elapsed_s, 2),
        "fields": {f: err.get(f) for f in (*NEVER_FITS_FIELDS, "retryable")},
    }


C7_DEFAULTS: dict[str, dict[str, Any]] = {
    # Standard P7 config that fired parks on the fork (status.log 2026-10-05):
    # ceiling 42 GB, hard 0.8, pool reclaim disabled, 3 staggered long prompts.
    "decode": {
        "ceiling_gb": 42.0,
        "hard_threshold": 0.8,
        "stagger_s": 0.5,
        "max_tokens": 2000,
        "requests": 3,
    },
    "image-decode": {
        "ceiling_gb": 42.0,
        "hard_threshold": 0.8,
        "stagger_s": 0.5,
        "max_tokens": 2000,
        "requests": 3,
    },
    "image-prefill": {
        "ceiling_gb": 42.0,
        "hard_threshold": 0.8,
        "stagger_s": 0.5,
        "max_tokens": 32,
        "requests": 3,
    },
    "never-fits": {
        "ceiling_gb": 42.0,
        "hard_threshold": None,
        "stagger_s": 0.0,
        "max_tokens": 16,
        "requests": 1,
    },
}


def c7_defaults(scenario: str) -> dict[str, Any]:
    if scenario not in C7_DEFAULTS:
        raise ValueError(f"unknown C7 scenario: {scenario!r}")
    return dict(C7_DEFAULTS[scenario])


def classify_c7(scenario: str, s: dict[str, Any]) -> str:
    """One-word outcome of a C7 scenario, from a summary's counters."""
    if scenario == "never-fits":
        return "never_fits_ok" if s.get("never_fits_correct") else "never_fits_wrong"
    if scenario not in ("decode", "image-decode", "image-prefill"):
        raise ValueError(f"unknown C7 scenario: {scenario!r}")
    if s.get("aborted_requests", 0) > 0:
        return "aborts"
    if scenario == "image-prefill":
        pauses, resumes = s.get("pauses_prefill", 0), s.get("resumes_prefill", 0)
        if pauses == 0:
            return "no_park"
        return "parked_not_resumed" if resumes < pauses else "prefill_park_ok"
    if not s.get("hard_pressure_seen"):
        return "no_pressure"
    pauses, resumes = s.get("pauses_decode", 0), s.get("resumes_decode", 0)
    if pauses == 0:
        return "no_decode_pause"
    if resumes < pauses:
        return "paused_not_resumed"
    if (s.get("identity") or {}).get("bug", 0) > 0:
        return "identity_bug"
    return "decode_pause_ok"


def summarize_c7(
    scenario: str,
    requests: list[dict[str, Any]],
    events: list[dict[str, Any]],
    identity: dict[str, dict[str, Any]] | None = None,
    never_fits: dict[str, Any] | None = None,
    peak_gb: float | None = None,
    wall_s: float | None = None,
) -> dict[str, Any]:
    """Reduce a C7 scenario: per-request rows plus the run-level counters."""
    paused = attribute_decode_pauses(events, requests)
    rows = []
    for r in requests:
        p = paused.get(r["name"], {})
        gap = max_decode_stall(r.get("chunk_times") or [])
        wall = None
        if r.get("end_s") is not None and r.get("start_s") is not None:
            wall = round(r["end_s"] - r["start_s"], 2)
        rows.append(
            {
                "name": r["name"],
                "ok": bool(r.get("ok")),
                "error": r.get("error"),
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "ttft_s": r.get("ttft_s"),
                "wall_s": wall,
                "decode_pauses": p.get("pauses", 0),
                "decode_resumes": p.get("resumes", 0),
                "restored_tokens": p.get("restored", []),
                "max_decode_stall_s": None if gap is None else round(gap, 3),
            }
        )
    gaps = [row["max_decode_stall_s"] for row in rows if row["max_decode_stall_s"]]
    ident = identity or {}
    classes = [v.get("classification") for v in ident.values() if not v.get("identical")]
    summary: dict[str, Any] = {
        "scenario": scenario,
        "rows": rows,
        "pauses_prefill": sum(1 for e in events if e["kind"] == "parked"),
        "pauses_decode": sum(1 for e in events if e["kind"] == "decode_paused"),
        "resumes_prefill": sum(1 for e in events if e["kind"] == "resumed"),
        "resumes_decode": sum(1 for e in events if e["kind"] == "decode_resumed"),
        "restored_tokens": sum(
            e["restored"] for e in events if e["kind"] == "decode_resumed"
        ),
        "aborts_single": sum(1 for e in events if e["kind"] == "victim_abort"),
        "aborts_all": sum(e["n"] for e in events if e["kind"] == "abort_all"),
        "victim_pauses": sum(1 for e in events if e["kind"] == "victim_pause"),
        "drain_waits": sum(1 for e in events if e["kind"] == "drain_wait"),
        "park_waits": sum(1 for e in events if e["kind"] == "park_wait"),
        "resume_fits": sum(1 for e in events if e["kind"] == "resume_fit"),
        "hard_pressure_seen": any(
            e["kind"] == "pressure" and e["to"] == "hard" for e in events
        ),
        "completions": sum(1 for row in rows if row["ok"]),
        "aborted_requests": sum(1 for row in rows if not row["ok"]),
        "all_final_ok": all(row["ok"] for row in rows),
        "max_decode_stall_s": max(gaps) if gaps else None,
        "peak_gb": peak_gb,
        "wall_s": wall_s,
        "identity": {
            "checked": len(ident),
            "paused_checked": sum(1 for v in ident.values() if v.get("paused")),
            "identical": sum(1 for v in ident.values() if v.get("identical")),
            "noise": classes.count("noise"),
            "bug": classes.count("bug"),
            "inconclusive": classes.count("inconclusive"),
            "unclassified": classes.count("unclassified"),
        },
        "never_fits_correct": None if never_fits is None else bool(never_fits.get("correct")),
        "events": events,
    }
    summary["verdict"] = classify_c7(scenario, summary)
    return summary


def p7_default_ceiling_gb(
    model_gb: float = 16.8, peak_gb: float = 44.0, frac: float = 0.5
) -> float:
    """Custom-tier ceiling that lands hard pressure mid-prefill.

    Between the resident model and the working-set peak measured with no
    guard pressure (P2 on the fork: 44 GB), so the KV growth of two or three
    long prefills crosses the hard watermark (97% of a custom ceiling)
    before the prefills finish.
    """
    if peak_gb <= model_gb:
        raise ValueError("peak_gb must be above model_gb")
    return round(model_gb + frac * (peak_gb - model_gb), 1)


def p7_settings_override(
    ceiling_gb: float,
    hard_threshold: float | None = None,
    soft_threshold: float | None = None,
) -> dict[str, Any]:
    """settings.json overlay: custom-tier ceiling, optionally a wider hard band.

    Prefill chunk transients can jump straight from below the hard watermark
    to beyond ``ceiling + 2 GB`` (the emergency brake, which never parks).
    Lowering ``hard_threshold`` widens the band in which a park may fire. A
    lone hard threshold drags the soft one 0.1 below it so the order holds.
    """
    mem: dict[str, Any] = {
        "memory_guard_tier": "custom",
        "memory_guard_custom_ceiling_gb": ceiling_gb,
    }
    if hard_threshold is not None:
        if not 0.0 < hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be in (0, 1]")
        if soft_threshold is None:
            soft_threshold = round(max(0.05, hard_threshold - 0.1), 2)
        mem["hard_threshold"] = hard_threshold
        mem["soft_threshold"] = soft_threshold
    elif soft_threshold is not None:
        mem["soft_threshold"] = soft_threshold
    if "hard_threshold" in mem and not mem["soft_threshold"] < mem["hard_threshold"]:
        raise ValueError("soft_threshold must be below hard_threshold")
    return {"memory": mem}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """New dict: ``override`` merged into ``base`` recursively (inputs untouched)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def make_png(width: int, height: int, seed: str) -> bytes:
    """Small deterministic RGB PNG (stdlib only); differs per ``seed``."""
    digest = hashlib.sha256(seed.encode()).digest()
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter: none
        for x in range(width):
            rows += bytes(
                (
                    (x * 255 // max(1, width - 1) + digest[0]) % 256,
                    (y * 255 // max(1, height - 1) + digest[1]) % 256,
                    digest[2 + (x + y) % 20] ^ ((x * y) & 0xFF),
                )
            )

    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def p7_content(prompt: str, mode: str, seed: str) -> Any:
    """Chat message content for a P7 request.

    ``text``: the prompt as is. ``image``: an OpenAI ``image_url`` data-URI
    part placed BEFORE the long text, so the image sits at the start of the
    history and the request carries ``vlm_inputs_embeds``.
    """
    if mode == "text":
        return prompt
    if mode != "image":
        raise ValueError(f"unknown P7 mode: {mode!r}")
    b64 = base64.b64encode(make_png(112, 112, seed)).decode()
    return [
        {"type": "text", "text": "Reference image for the task below."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]


# --------------------------------------------------------------------------
# SSE stream accounting
# --------------------------------------------------------------------------


class StreamAccumulator:
    """Accumulate an OpenAI-style SSE chat stream line by line.

    An error frame, a missing usage frame, a missing ``[DONE]`` terminator or
    a stream with no generated text is a FAILURE (an earlier script treated
    these as success).
    """

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.first_token_at: float | None = None
        self.chunk_times: list[float] = []  # one entry per text-bearing frame
        self.chunk_chars: list[int] = []
        self.usage: dict[str, Any] | None = None
        self.errors: list[str] = []
        self.done = False
        self._event: str | None = None

    def feed(self, line: str, now: float) -> bool:
        """Feed one SSE line. Returns True when it carried the first text."""
        line = line.rstrip("\r\n")
        if not line or line.startswith(":"):
            return False
        if line.startswith("event:"):
            self._event = line[6:].strip()
            return False
        if not line.startswith("data:"):
            return False
        payload = line[5:].strip()
        event, self._event = self._event, None
        if payload == "[DONE]":
            self.done = True
            return False
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            self.errors.append(f"invalid JSON frame: {payload[:120]}")
            return False
        if event == "error" or (isinstance(obj, dict) and "error" in obj):
            err = obj.get("error", obj) if isinstance(obj, dict) else obj
            msg = err.get("message") if isinstance(err, dict) else None
            self.errors.append(str(msg or err)[:300])
            return False
        if isinstance(obj, dict) and obj.get("usage"):
            self.usage = obj["usage"]
        got_text = False
        n_chars = 0
        for choice in (obj.get("choices") or []) if isinstance(obj, dict) else []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                piece = delta.get(key)
                if piece:
                    self.parts.append(piece)
                    got_text = True
                    n_chars += len(piece)
        if got_text:
            self.chunk_times.append(now)
            self.chunk_chars.append(n_chars)
        if got_text and self.first_token_at is None:
            self.first_token_at = now
            return True
        return False

    def finalize(self) -> dict[str, Any]:
        text = "".join(self.parts)
        usage = self.usage or {}
        error: str | None = None
        if self.errors:
            error = "; ".join(self.errors)
        elif self.usage is None:
            error = "missing usage frame"
        elif not self.done:
            error = "stream ended without [DONE]"
        elif not text:
            error = "no generated text"
        details = usage.get("prompt_tokens_details") or {}
        return {
            "ok": error is None,
            "error": error,
            "text": text,
            "first_token_at": self.first_token_at,
            "chunk_times": list(self.chunk_times),
            "chunk_chars": list(self.chunk_chars),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens")
            if isinstance(details, dict)
            else None,
        }


def request_metrics(
    t_start: float,
    t_first: float | None,
    t_end: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> dict[str, float | None]:
    ttft = None if t_first is None else t_first - t_start
    prefill = None
    if ttft and prompt_tokens:
        prefill = prompt_tokens / ttft
    decode = None
    if (
        t_first is not None
        and completion_tokens
        and completion_tokens > 1
        and t_end > t_first
    ):
        decode = (completion_tokens - 1) / (t_end - t_first)
    return {
        "ttft_s": ttft,
        "prefill_tps": prefill,
        "decode_tps": decode,
        "total_s": t_end - t_start,
    }


# --------------------------------------------------------------------------
# Process sampling helpers
# --------------------------------------------------------------------------

_RE_FOOTPRINT = re.compile(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB|TB|B)\b")
_UNIT_GB = {
    "B": 1 / 1024**3,
    "KB": 1 / 1024**2,
    "MB": 1 / 1024,
    "GB": 1.0,
    "TB": 1024.0,
}


def parse_footprint_gb(output: str) -> float | None:
    """Sum the ``Footprint:`` figures of `footprint` output, in GB."""
    total = 0.0
    found = False
    for m in _RE_FOOTPRINT.finditer(output):
        total += float(m.group(1)) * _UNIT_GB[m.group(2)]
        found = True
    return total if found else None


def median(values: list[Any]) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def spread(values: list[Any]) -> tuple[float | None, float | None]:
    xs = [v for v in values if v is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fps = [s["footprint_gb"] for s in samples if s.get("footprint_gb") is not None]
    hs = [s["health_s"] for s in samples if s.get("health_s") is not None]
    return {
        "n": len(samples),
        "footprint_peak_gb": max(fps) if fps else None,
        "footprint_median_gb": median(fps),
        "health_failures": sum(1 for s in samples if not s.get("health_ok", True)),
        "health_max_s": max(hs) if hs else None,
    }


# --------------------------------------------------------------------------
# Prompt sizing and safety guards
# --------------------------------------------------------------------------

SMOKE_DIVISOR = 200


def scale_tokens(n: int, smoke: bool) -> int:
    """Nominal prompt length -> actual target (smoke runs use tiny prompts)."""
    return max(100, n // SMOKE_DIVISOR) if smoke else n


def make_nonce(seed: str) -> str:
    digest = hashlib.sha1(seed.encode()).hexdigest()[:16]
    return f"[bench-nonce {digest}]\n"


def build_prompt_text(
    prefix: str,
    corpus_ids: list[int],
    start: int,
    n_tokens: int,
    decode: Callable[[list[int]], str],
) -> str:
    """Nonce prefix + `n_tokens` corpus tokens starting at `start` (wrapping)."""
    n = len(corpus_ids)
    sel = [corpus_ids[(start + i) % n] for i in range(n_tokens)]
    return prefix + decode(sel)


def is_safe_bench_path(path: str, roots: list[str]) -> bool:
    """True only for a path strictly inside one of the bench roots."""
    norm = os.path.normpath(path)
    if ".." in path.split(os.sep):
        return False
    for root in roots:
        r = os.path.normpath(root)
        if norm != r and norm.startswith(r + os.sep):
            return True
    return False


PROTECTED_PORTS = (8001,)


def port_allowed(port: int, allow: list[int]) -> bool:
    return port not in PROTECTED_PORTS or port in allow


# --------------------------------------------------------------------------
# Premise evaluation
# --------------------------------------------------------------------------


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _verdict(checks: list[dict[str, Any]]) -> str:
    return "PASS" if all(c["ok"] for c in checks) else "FAIL"


def _inconclusive(reason: str) -> dict[str, Any]:
    return {"verdict": "INCONCLUSIVE", "checks": [_check("data", False, reason)]}


def _fmt(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _pct(a: float | None, b: float | None) -> float | None:
    """Percent change of `a` relative to `b`."""
    return (a / b - 1) * 100 if a is not None and b else None


def eval_p6(
    upstream: list[dict],
    fork: list[dict],
    fork_off: list[dict] | None = None,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Verdict is fork vs upstream; fork-off only feeds the attribution."""
    if not fork:
        return _inconclusive("no fork runs")
    if not upstream:
        return _inconclusive("no upstream runs to compare against")

    def meds(runs: list[dict]) -> dict[str, Any]:
        reqs = [
            r["scenario"]["request"] for r in runs if r["scenario"]["request"].get("ok")
        ]
        out: dict[str, Any] = {"n_ok": len(reqs), "n_runs": len(runs)}
        for key in ("prefill_tps", "decode_tps", "ttft_s"):
            vals = [q[key] for q in reqs]
            out[key] = median(vals)
            out[f"{key}_min"], out[f"{key}_max"] = spread(vals)
        return out

    up, fk = meds(upstream), meds(fork)
    off = meds(fork_off) if fork_off else None
    checks = [
        _check(
            "all fork requests ok",
            all(r["scenario"]["request"].get("ok") for r in fork),
            f"{fk['n_ok']}/{len(fork)} ok",
        )
    ]
    for key in ("prefill_tps", "decode_tps"):
        u, f = up[key], fk[key]
        if u and f is not None:
            ratio = f / u
            ok = ratio >= 1 - tolerance
            detail = f"fork {f:.2f} vs upstream {u:.2f} ({(ratio - 1) * 100:+.1f}%)"
        else:
            ok, detail = False, "missing median"
        checks.append(_check(f"{key} >= -{int(tolerance * 100)}%", ok, detail))
    attribution: dict[str, float | None] = {}
    if off:
        for key, tag in (("prefill_tps", "prefill"), ("decode_tps", "decode")):
            attribution[f"{tag}_pct_forkoff_vs_upstream"] = _pct(off[key], up[key])
            attribution[f"{tag}_pct_fork_vs_forkoff"] = _pct(fk[key], off[key])
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "upstream": up,
        "fork": fk,
        "fork_off": off,
        "attribution": attribution,
    }


def _p1_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {
            "cached_median": None,
            "reprefilled_median": None,
            "progress_median": None,
        }
    return {
        "cached_median": median(
            [r["scenario"].get("reissue_cached") or 0 for r in runs]
        ),
        "reprefilled_median": median(
            [r["scenario"].get("reissue_reprefilled") for r in runs]
        ),
        "progress_median": median(
            [r["scenario"].get("progress_tokens_est") for r in runs]
        ),
        "ttft_median": median([r["scenario"]["reissue"].get("ttft_s") for r in runs]),
    }


def eval_p1(
    fork: list[dict], upstream: list[dict], min_ratio: float = 0.5
) -> dict[str, Any]:
    if not fork:
        return _inconclusive("no fork runs")
    ratios = []
    for r in fork:
        s = r["scenario"]
        prog = s.get("progress_tokens_est") or 0
        ratios.append((s.get("reissue_cached") or 0) / prog if prog else 0.0)
    med_ratio = median(ratios)
    checks = [
        _check(
            "re-issue completed",
            all(r["scenario"]["reissue"].get("ok") for r in fork),
            f"{sum(1 for r in fork if r['scenario']['reissue'].get('ok'))}/{len(fork)} ok",
        ),
        _check(
            f"cached >= {int(min_ratio * 100)}% of cancelled progress",
            med_ratio is not None and med_ratio >= min_ratio,
            f"median cached/progress = {_fmt(med_ratio)}",
        ),
    ]
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "fork": _p1_summary(fork),
        "upstream": _p1_summary(upstream),
    }


def _p2_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"aborts_memory": None, "completed": None, "total": None}
    reqs = [q for r in runs for q in r["scenario"]["requests"]]
    pools = [
        r["log"].get("pool_gb_max")
        for r in runs
        if r["log"].get("pool_gb_max") is not None
    ]
    return {
        "aborts_memory": sum(r["log"].get("aborts_memory", 0) for r in runs),
        "aborts_total": sum(r["log"].get("aborts_total", 0) for r in runs),
        "completed": sum(1 for q in reqs if q.get("ok")),
        "total": len(reqs),
        "pool_gb_max": max(pools) if pools else None,
        "footprint_peak_gb": max(
            [r.get("samples", {}).get("footprint_peak_gb") or 0 for r in runs],
            default=None,
        ),
        "wall_s_median": median([r["scenario"].get("wall_s") for r in runs]),
    }


def eval_p2(
    fork: list[dict], upstream: list[dict], pool_limit_gb: float = 17.0
) -> dict[str, Any]:
    if not fork:
        return _inconclusive("no fork runs")
    fk = _p2_summary(fork)
    checks = [
        _check(
            "0 memory aborts",
            fk["aborts_memory"] == 0,
            f"{fk['aborts_memory']} memory aborts",
        ),
        _check(
            "both requests completed per run",
            fk["completed"] == fk["total"],
            f"{fk['completed']}/{fk['total']} completed",
        ),
    ]
    if fk["pool_gb_max"] is None:
        checks.append(
            _check(
                f"max pool <= {pool_limit_gb:g} GB",
                True,
                "no pool= evidence in log (no pressure episode)",
            )
        )
    else:
        checks.append(
            _check(
                f"max pool <= {pool_limit_gb:g} GB",
                fk["pool_gb_max"] <= pool_limit_gb,
                f"max pool {fk['pool_gb_max']:.1f} GB",
            )
        )
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "fork": fk,
        "upstream": _p2_summary(upstream),
    }


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# --- text metrics (P3) ----------------------------------------------------

P3_PREFIX_TOKENS = 32
P3_DEGEN_REL = 0.5  # B=3 distinct-4gram ratio below this x the B=1 value
P3_DEGEN_FLOOR = 0.3  # ... or below this absolute floor (when B=1 is above it)


def distinct_ngram_ratio(text: str, n: int = 4) -> float | None:
    """Distinct word n-grams / total word n-grams (1.0 = no repetition)."""
    words = text.split()
    total = len(words) - n + 1
    if total < 1:
        return None
    return len({tuple(words[i : i + n]) for i in range(total)}) / total


def is_degenerate(
    baseline: float | None,
    candidate: float | None,
    rel: float = P3_DEGEN_REL,
    floor: float = P3_DEGEN_FLOOR,
) -> bool:
    """Flag `candidate` when far worse than the calibrating `baseline`."""
    if baseline is None or candidate is None:
        return False
    if candidate < rel * baseline:
        return True
    return candidate < floor <= baseline


# Near-tie classification of the first token divergence (needs logprob dumps).
P3_NOISE_MARGIN = 0.5  # nats: top1-top2 gap of the reference at the divergence
P3_BUG_MARGIN = 2.0
P3_NOISE_DLP = 0.05  # mean |delta logprob| of the chosen tokens before it
P3_BUG_DLP = 0.25


def compare_logprobs(
    ref: list[dict], test: list[dict], limit: int | None = None
) -> dict[str, Any]:
    """First token divergence of two logprob dumps and its near-tie evidence.

    Records are ``{"t": token, "lp": chosen logprob, "top": [[id, lp], ...]}``
    sorted by logprob. ``margin`` is the REFERENCE top1-top2 gap at the
    divergence position; the drift is measured on the shared prefix.
    """
    n = min(len(ref), len(test))
    if limit is not None:
        n = min(n, limit)
    div = next((i for i in range(n) if ref[i]["t"] != test[i]["t"]), None)
    shared = n if div is None else div
    drifts = [abs(ref[i]["lp"] - test[i]["lp"]) for i in range(shared)]

    def gap(rec: dict) -> float | None:
        top = rec.get("top") or []
        return top[0][1] - top[1][1] if len(top) >= 2 else None

    return {
        "n_ref": len(ref),
        "n_test": len(test),
        "n_shared": shared,
        "div_index": div,
        "margin": None if div is None else gap(ref[div]),
        "test_margin": None if div is None else gap(test[div]),
        "mean_abs_dlp": sum(drifts) / len(drifts) if drifts else 0.0,
        "max_abs_dlp": max(drifts, default=0.0),
    }


def classify_divergence(cmp: dict[str, Any]) -> str | None:
    """noise / bug / inconclusive for a divergence; None when there is none."""
    if cmp.get("div_index") is None:
        return None
    margin, drift = cmp.get("margin"), cmp.get("mean_abs_dlp") or 0.0
    if (margin is not None and margin > P3_BUG_MARGIN) or drift >= P3_BUG_DLP:
        return "bug"
    if margin is not None and margin < P3_NOISE_MARGIN and drift < P3_NOISE_DLP:
        return "noise"
    return "inconclusive"


def load_logprob_dump(path) -> list[dict]:
    """Records of one request's OMLX_DEBUG_LOGPROB_DUMP file (torn tail skipped)."""
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            break
    return out


def dump_listing(directory) -> set[str]:
    d = Path(directory)
    return {p.name for p in d.glob("*.jsonl")} if d.exists() else set()


def new_dumps(directory, before: set[str]) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.glob("*.jsonl") if p.name not in before)


def pick_dump(paths: list[Path], prompt_tokens: int | None) -> Path | None:
    """The dump whose records carry this prompt-token count."""
    for p in paths:
        recs = load_logprob_dump(p)
        if recs and recs[0].get("pt") == prompt_tokens:
            return p
    return None


def first_divergence(a: str, b: str) -> int | None:
    """Index of the first differing char, None when the texts are identical."""
    if a == b:
        return None
    return common_prefix_len(a, b)


def prefix_chars(chunk_chars: list[int], n: int = P3_PREFIX_TOKENS) -> int:
    """Characters carried by the first `n` streamed chunks (~tokens)."""
    return sum(chunk_chars[:n])


def chunk_index_at_char(chunk_chars: list[int], pos: int) -> int:
    """How many whole streamed chunks end at or before char `pos`."""
    total = 0
    for i, c in enumerate(chunk_chars):
        total += c
        if total > pos:
            return i
    return len(chunk_chars)


def _row_prefix_chars(out: dict) -> int:
    cc = out.get("chunk_chars")
    if cc:
        return prefix_chars(cc)
    n = out.get("completion_tokens") or 0
    return int(len(out.get("text", "")) * P3_PREFIX_TOKENS / n) if n else 0


def _p3_row_detail(row: dict) -> dict[str, Any]:
    ref, b3 = row["alone_restored"], row["batched"]
    div = first_divergence(ref["text"], b3["text"])
    pre = _row_prefix_chars(ref)
    d1 = distinct_ngram_ratio(ref["text"])
    d3 = distinct_ngram_ratio(b3["text"])
    cc = ref.get("chunk_chars") or []
    detail = {
        "name": row["name"],
        "first_divergence_chars": div,
        "first_divergence_token_est": (
            None if div is None or not cc else chunk_index_at_char(cc, div)
        ),
        "prefix32_chars": pre,
        "diverged_in_prefix": div is not None and div < pre,
        "d4_b1": d1,
        "d4_b3": d3,
        "degenerate": is_degenerate(d1, d3),
        "alone_tokens": ref.get("completion_tokens"),
        "batched_tokens": b3.get("completion_tokens"),
        "classification_available": False,
    }
    lps = row.get("logprobs") or {}
    if lps.get("alone_restored") and lps.get("batched"):
        c = compare_logprobs(lps["alone_restored"], lps["batched"])
        detail.update(
            classification_available=True,
            classification=classify_divergence(c),
            div_token=c["div_index"],
            margin=c["margin"],
            test_margin=c["test_margin"],
            mean_abs_dlp=c["mean_abs_dlp"],
            max_abs_dlp=c["max_abs_dlp"],
            n_shared=c["n_shared"],
            diverged_in_prefix=(
                c["div_index"] is not None and c["div_index"] < P3_PREFIX_TOKENS
            ),
        )
        controls = {}
        pairs = (
            ("rerun", "alone_restored", "alone_restored2"),
            ("ragged_b1", "alone_restored", "ragged_b1"),
            ("ragged_vs_b3", "ragged_b1", "batched"),
        )
        for label, a, b in pairs:
            if lps.get(a) and lps.get(b):
                cc2 = compare_logprobs(lps[a], lps[b])
                cc2["classification"] = classify_divergence(cc2)
                controls[label] = cc2
        detail["controls"] = controls
    return detail


def _p3_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"rows": None, "detail": []}
    rows = [row for r in runs for row in r["scenario"]["rows"]]
    usable = [
        row
        for row in rows
        if row["alone_restored"].get("ok") and row["batched"].get("ok")
    ]
    detail = [_p3_row_detail(row) for row in usable]
    cold = [
        row
        for row in rows
        if row.get("alone", {}).get("ok")
        and row["alone_restored"].get("ok")
        and row["alone"]["text"] != row["alone_restored"]["text"]
    ]
    return {
        "rows": len(rows),
        "detail": detail,
        "diverged_rows": sum(
            1 for d in detail if d["first_divergence_chars"] is not None
        ),
        "diverged_in_prefix_rows": sum(1 for d in detail if d["diverged_in_prefix"]),
        "degenerate_rows": sum(1 for d in detail if d["degenerate"]),
        # Informational: B=1 cold vs B=1 restored from the prefix cache.
        "cold_vs_restored_mismatched_rows": len(cold),
        "batch_max_b": max((r["log"].get("batch_max_b", 0) for r in runs), default=0),
    }


def _noise_check(
    detail: list[dict],
    all_ok: bool,
    label: str = "B=3 vs B=1 (restored)",
    waive: str | None = None,
) -> dict[str, Any]:
    """Divergences inside the first 32 tokens must be near-tie numerical noise.

    `waive` (a reason) turns a failing judgement informational: it is only
    passed when the reference itself is shown not to be reproducible.
    """
    bad = [
        d
        for d in detail
        if d["diverged_in_prefix"] and d["classification"] != "noise"
    ]

    def one(d: dict) -> str:
        if d["div_token"] is None:
            return f"{d['name']}: identical"
        return (
            f"{d['name']}: token {d['div_token']} margin {_fmt(d['margin'])} nats, "
            f"mean|dlp| {_fmt(d['mean_abs_dlp'], 3)} -> {d['classification']}"
            + ("" if d["diverged_in_prefix"] else " (after token 32, not judged)")
        )

    detail_txt = f"{len(bad)}/{len(detail)} rows not noise; " + "; ".join(
        one(d) for d in detail
    )
    ok = all_ok and not bad
    if bad and waive and all_ok:
        ok, detail_txt = True, f"informational ({waive}); " + detail_txt
    return _check(
        f"{label}: divergence inside the first {P3_PREFIX_TOKENS} tokens "
        "is numerical noise (near-tie)",
        ok,
        detail_txt,
    )


def _reference_irreproducible(detail: list[dict]) -> bool:
    """True when B=1 restored vs B=1 restored again already diverges."""
    return any(
        (d.get("controls", {}).get("rerun") or {}).get("div_index") is not None
        for d in detail
    )


def eval_p3(
    fork: list[dict], upstream: list[dict], nomtp: list[dict] | None = None
) -> dict[str, Any]:
    """Like-for-like: B=3 restored rows vs the same rows alone, restored."""
    if not fork:
        return _inconclusive("no fork runs")
    fk = _p3_summary(fork)
    all_ok = all(
        row[k].get("ok")
        for r in fork
        for row in r["scenario"]["rows"]
        for k in ("alone_restored", "batched")
    )
    n = len(fk["detail"])
    judged = bool(fk["detail"]) and all(d["classification_available"] for d in fk["detail"])
    nm = _p3_summary(nomtp) if nomtp else None
    nm_judged = bool(nm and nm["detail"]) and all(
        d["classification_available"] for d in nm["detail"]
    )
    nm_ok = nm_judged and all(
        row[k].get("ok")
        for r in nomtp
        for row in r["scenario"]["rows"]
        for k in ("alone_restored", "batched")
    )
    nm_check = (
        _noise_check(nm["detail"], nm_ok, label="B=3 vs B=1, MTP off (batching only)")
        if nm_judged
        else None
    )
    waive = (
        "the B=1 MTP reference is not reproducible: B=1 rerun diverges; "
        "batching-only comparison with MTP off is noise"
        if nm_check is not None
        and nm_check["ok"]
        and judged
        and _reference_irreproducible(fk["detail"])
        else None
    )
    checks = [
        _check(
            "all requests completed",
            all_ok,
            "every restored B=1 and B=3 request ok" if all_ok else "a request failed",
        ),
        _noise_check(fk["detail"], all_ok, waive=waive)
        if judged
        else _check(
            f"B=3 == B=1 (restored) for the first {P3_PREFIX_TOKENS} tokens, every row",
            all_ok and fk["diverged_in_prefix_rows"] == 0,
            f"{fk['diverged_in_prefix_rows']}/{n} rows diverge inside the prefix; "
            + (
                ", ".join(
                    f"{d['name']}: "
                    + (
                        "identical"
                        if d["first_divergence_chars"] is None
                        else f"char {d['first_divergence_chars']}"
                    )
                    for d in fk["detail"]
                )
                or "-"
            ),
        ),
        _check(
            "no degeneration in any output (distinct-4gram)",
            all_ok and fk["degenerate_rows"] == 0,
            f"{fk['degenerate_rows']}/{n} rows degenerate; "
            + (
                ", ".join(
                    f"{d['name']}: B1={_fmt(d['d4_b1'])} B3={_fmt(d['d4_b3'])}"
                    for d in fk["detail"]
                )
                or "-"
            ),
        ),
        _check(
            "3-row batch exercised (log)",
            fk["batch_max_b"] >= 3,
            f"max B seen in log: {fk['batch_max_b']}",
        ),
    ]
    if nm_check is not None:
        checks += [
            nm_check,
            _check(
                "3-row batch exercised with MTP off (log)",
                nm["batch_max_b"] >= 3,
                f"max B seen in log: {nm['batch_max_b']}",
            ),
        ]
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "nomtp": nm,
        "fork": fk,
        "upstream": _p3_summary(upstream),
    }


# --- P4 -------------------------------------------------------------------


def aggregate_decode_rate(rows: list[dict]) -> dict[str, Any]:
    """Sum of per-row rates over the window where ALL rows are decoding.

    Window = [latest first token, earliest last token], from per-chunk
    timestamps. Each streamed chunk stands for completion_tokens / n_chunks
    tokens (MTP may emit several tokens per frame).
    """
    none = {"rate": None, "window_s": None}
    if not rows or any(not r.get("chunk_times") for r in rows):
        return none
    start = max(r["chunk_times"][0] for r in rows)
    end = min(r["chunk_times"][-1] for r in rows)
    if end <= start:
        return none
    total = 0.0
    for r in rows:
        ts = r["chunk_times"]
        tpc = (r.get("completion_tokens") or len(ts)) / len(ts)
        total += sum(1 for t in ts if start < t <= end) * tpc
    return {"rate": total / (end - start), "window_s": end - start}


def pl_row(run: dict[str, Any]) -> dict[str, Any]:
    """One PL (batched-decode limit probe) run reduced to the table columns."""
    reqs = run["scenario"]["requests"]
    log = run.get("log", {})
    ok = bool(run.get("ok")) and all(r.get("ok") for r in reqs)
    good = [r for r in reqs if r.get("ok")]
    ratios = [distinct_ngram_ratio(r.get("text") or "") for r in good]
    degenerate = any(x is not None and x < P3_DEGEN_FLOOR for x in ratios)
    return {
        "ok": ok,
        "wall_s": run.get("wall_s"),
        "req_end_s": [r.get("end_s") for r in reqs],
        "req_decode_tps": [r.get("decode_tps") for r in reqs],
        "aggregate_tps": aggregate_decode_rate(good)["rate"] if len(good) == len(reqs) else None,
        "holds": log.get("holds", 0),
        "backfills": log.get("backfills", 0),
        "agings": log.get("agings", 0),
        "max_b": log.get("batch_max_b", 0),
        "peak_gb": run.get("samples", {}).get("footprint_peak_gb"),
        "aborts": log.get("aborts_total", 0),
        "distinct4": ratios,
        "degenerate": degenerate,
    }


def _p4_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"serial_wall_s": None, "concurrent_wall_s": None}
    aggs = [
        aggregate_decode_rate(r["scenario"]["concurrent"]["requests"]) for r in runs
    ]
    serial_rates = [
        q.get("decode_tps") for r in runs for q in r["scenario"]["serial"]["requests"]
    ]
    return {
        "serial_wall_s": median([r["scenario"]["serial"]["wall_s"] for r in runs]),
        "concurrent_wall_s": median(
            [r["scenario"]["concurrent"]["wall_s"] for r in runs]
        ),
        "aborts_total": sum(r["log"].get("aborts_total", 0) for r in runs),
        "completed": sum(
            1
            for r in runs
            for mode in ("serial", "concurrent")
            for q in r["scenario"][mode]["requests"]
            if q.get("ok")
        ),
        "total": sum(
            len(r["scenario"][m]["requests"])
            for r in runs
            for m in ("serial", "concurrent")
        ),
        "serial_decode_tps": [
            median(
                [r["scenario"]["serial"]["requests"][i].get("decode_tps") for r in runs]
            )
            for i in range(len(runs[0]["scenario"]["serial"]["requests"]))
        ],
        "concurrent_decode_tps": [
            median(
                [
                    r["scenario"]["concurrent"]["requests"][i].get("decode_tps")
                    for r in runs
                ]
            )
            for i in range(len(runs[0]["scenario"]["concurrent"]["requests"]))
        ],
        "serial_single_row_median_tps": median(serial_rates),
        "aggregate_decode_tps": median([a["rate"] for a in aggs]),
        "aggregate_decode_tps_all": [a["rate"] for a in aggs],
        "aggregate_window_s": median([a["window_s"] for a in aggs]),
    }


def eval_p4(fork: list[dict], upstream: list[dict]) -> dict[str, Any]:
    """P4 criterion: no waste. With B=3 the non-attention compute sets a
    ~49 ms step (~20 tok/s per row), so 'per-row >= 50% of single-row' is
    unreachable; the premise is about not wasting memory and time."""
    if not fork:
        return _inconclusive("no fork runs")
    fk = _p4_summary(fork)
    ratios = [
        (c / s) if (c is not None and s) else None
        for s, c in zip(fk["serial_decode_tps"], fk["concurrent_decode_tps"])
    ]
    agg, single = fk["aggregate_decode_tps"], fk["serial_single_row_median_tps"]
    if agg is None:
        agg_detail = "no window where all three rows decode (or no timestamps)"
    else:
        agg_detail = (
            f"aggregate {agg:.2f} tok/s over a {_fmt(fk['aggregate_window_s'], 1)} s "
            f"window vs median single-row {_fmt(single)} tok/s"
        )
    checks = [
        _check(
            "all requests completed",
            fk["completed"] == fk["total"],
            f"{fk['completed']}/{fk['total']} ok",
        ),
        _check("0 aborts", fk["aborts_total"] == 0, f"{fk['aborts_total']} aborts"),
        _check(
            "concurrent wall <= serial wall",
            fk["concurrent_wall_s"] is not None
            and fk["concurrent_wall_s"] <= fk["serial_wall_s"],
            f"concurrent {_fmt(fk['concurrent_wall_s'], 1)} s vs serial {_fmt(fk['serial_wall_s'], 1)} s",
        ),
        _check(
            "aggregate steady decode >= median single-row serial decode",
            agg is not None and single is not None and agg >= single,
            agg_detail,
        ),
    ]
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "fork": fk,
        "upstream": _p4_summary(upstream),
        # Information only (no longer a criterion).
        "row_ratios": ratios,
    }


# ---------------------------------------------------------------------------
# P5 - pi-driven unattended batch
# ---------------------------------------------------------------------------

# oMLX error signatures. Plain words such as "memory" or "aborted" are NOT
# signatures: the sessions summarize oMLX source and would match them.
P5_MEMORY_SIGNATURE = re.compile(
    r"oMLX memory guard"
    r"|process memory limit exceeded"
    r"|prefill_memory_exceeded"
    r"|prefill_memory_aborted"
    r"|memory_admission_stalled"
    r"|Model '[^']*' is busy"
)
P5_MIN_OUTPUT_CHARS = 200


def classify_p5_session(out: str, err: str, exit_code: int | None) -> dict[str, Any]:
    """Classify one pi session.

    A memory signature only counts in stderr or on a stdout line that starts
    with "Error:". A session that saw a signature but still exited 0 with a
    full answer was recovered (extension or auto-continue) and is a success.
    """
    err_hit = bool(P5_MEMORY_SIGNATURE.search(err or ""))
    out_hit = any(
        line.lstrip().startswith("Error:") and P5_MEMORY_SIGNATURE.search(line)
        for line in (out or "").splitlines()
    )
    signature = err_hit or out_hit
    complete = exit_code == 0 and len((out or "").strip()) > P5_MIN_OUTPUT_CHARS
    recovered = signature and complete
    memory_failure = signature and not complete
    return {
        "memory_signature": signature,
        "memory_failure": memory_failure,
        "recovered": recovered,
        "ok": complete and not memory_failure,
    }


def summarize_p5(
    d: Path,
    variant: str,
    mode: str,
    n: int,
    wall_s: int | None,
    log_text: str | None = None,
) -> dict[str, Any]:
    """Build the P5 summary for one variant/mode directory of session files."""

    def read(p: Path) -> str:
        return p.read_text(errors="replace") if p.exists() else ""

    rows = []
    for i in range(n):
        out, err = read(d / f"session{i}.out"), read(d / f"session{i}.err")
        ex, wl = read(d / f"session{i}.exit").strip(), read(d / f"session{i}.wall").strip()
        code = int(ex) if ex else None
        row = {
            "session": i,
            "exit": code,
            "out_chars": len(out),
            "wall_s": int(wl) if wl.isdigit() else None,
        }
        row.update(classify_p5_session(out, err, code))
        rows.append(row)
    aborts = (
        None
        if log_text is None
        else len(re.findall(r"[Aa]borted \d+ request", log_text))
    )
    return {
        "variant": variant,
        "mode": mode,
        "sessions": n,
        "completed": sum(r["ok"] for r in rows),
        "manual_interventions": sum(not r["ok"] for r in rows),
        "memory_failures": sum(r["memory_failure"] for r in rows),
        "recovered": sum(r["recovered"] for r in rows),
        "wall_s": wall_s,
        "server_aborts": aborts,
        "rows": rows,
    }


def eval_p5(
    fork_with: dict[str, Any] | None, baseline: dict[str, Any] | None
) -> dict[str, Any]:
    """P5: with the extension the batch needs no human; the baseline does."""
    if not fork_with or not baseline:
        return _inconclusive("need fork-with and upstream-without P5 summaries")
    n = fork_with.get("sessions")
    fw_ok = (
        fork_with.get("completed") == n and fork_with.get("manual_interventions") == 0
    )
    base_manual = baseline.get("manual_interventions") or 0
    checks = [
        _check(
            "fork+extension completes unattended",
            fw_ok,
            f"completed {fork_with.get('completed')}/{n}, "
            f"manual {fork_with.get('manual_interventions')}",
        ),
        _check(
            "baseline shows the failure mode",
            base_manual > 0,
            f"upstream without extension: manual {base_manual}/"
            f"{baseline.get('sessions')}",
        ),
    ]
    if fw_ok and base_manual == 0:
        v = _inconclusive("baseline never failed; the run did not create pressure")
        v["checks"] = checks
        return v
    return {"verdict": _verdict(checks), "checks": checks}
