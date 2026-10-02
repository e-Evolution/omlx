# SPDX-License-Identifier: Apache-2.0
"""Pure logic for the unattended-scheduling premise benchmark.

Everything here is side-effect free so it can be unit tested with fixtures:
server-log parsing, SSE stream accounting, rate computation, `footprint`
output parsing, prompt sizing helpers, safety guards and the PASS/FAIL
evaluation of each premise.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
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
_RE_RESTORE = re.compile(r"Prefix cache restore for \S+: source=\S+ cached=(\d+)")
_RE_CACHE_LIMIT = re.compile(r"MLX buffer cache limit set to ([\d.]+) GB")
_RE_TQ_FUSED = re.compile(
    r"TurboQuant batched attention: per-row fused path \(B=(\d+)\)"
)
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
    restores: list[int] = []
    cache_limit: float | None = None
    tq_rows: list[int] = []
    tq_fallback = False
    ssd_init = False
    ssd_disabled = False

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
        m = _RE_RESTORE.search(line)
        if m:
            restores.append(int(m.group(1)))
        m = _RE_CACHE_LIMIT.search(line)
        if m:
            cache_limit = float(m.group(1))
        m = _RE_TQ_FUSED.search(line)
        if m:
            tq_rows.append(int(m.group(1)))
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
        "restores": restores,
        "restore_cached_last": restores[-1] if restores else None,
        "restore_cached_max": max(restores) if restores else None,
        "mlx_cache_limit_gb": cache_limit,
        "tq_batched_fused_rows": tq_rows,
        "tq_batched_fused_max_b": max(tq_rows) if tq_rows else 0,
        "tq_batched_fallback": tq_fallback,
        "ssd_initialized": ssd_init and not ssd_disabled,
        "ssd_disabled": ssd_disabled,
    }


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
        for choice in (obj.get("choices") or []) if isinstance(obj, dict) else []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                piece = delta.get(key)
                if piece:
                    self.parts.append(piece)
                    got_text = True
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


def eval_p6(
    upstream: list[dict], fork: list[dict], tolerance: float = 0.05
) -> dict[str, Any]:
    if not fork:
        return _inconclusive("no fork runs")
    if not upstream:
        return _inconclusive("no upstream runs to compare against")

    def meds(runs: list[dict]) -> dict[str, float | None]:
        reqs = [
            r["scenario"]["request"] for r in runs if r["scenario"]["request"].get("ok")
        ]
        return {
            "prefill_tps": median([q["prefill_tps"] for q in reqs]),
            "decode_tps": median([q["decode_tps"] for q in reqs]),
            "ttft_s": median([q["ttft_s"] for q in reqs]),
            "n_ok": len(reqs),
        }

    up, fk = meds(upstream), meds(fork)
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
    return {"verdict": _verdict(checks), "checks": checks, "upstream": up, "fork": fk}


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


def _p3_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"rows": None, "mismatched_rows": None}
    rows = [row for r in runs for row in r["scenario"]["rows"]]
    mismatched = [
        row
        for row in rows
        if row["alone"].get("ok")
        and row["batched"].get("ok")
        and row["alone"]["text"] != row["batched"]["text"]
    ]
    cache_mismatch = [
        row
        for row in rows
        if row.get("alone_cached", {}).get("ok")
        and row["alone"].get("ok")
        and row["alone"]["text"] != row["alone_cached"]["text"]
    ]
    return {
        "rows": len(rows),
        "mismatched_rows": len(mismatched),
        # Informational: B=1 cold vs B=1 restored from the prefix cache.
        "cache_restore_mismatched_rows": len(cache_mismatch),
        "mismatch_detail": [
            {
                "name": row["name"],
                "common_prefix_chars": common_prefix_len(
                    row["alone"]["text"], row["batched"]["text"]
                ),
                "alone_tokens": row["alone"].get("completion_tokens"),
                "batched_tokens": row["batched"].get("completion_tokens"),
            }
            for row in mismatched
        ],
        "tq_batched_fused_max_b": max(
            (r["log"].get("tq_batched_fused_max_b", 0) for r in runs), default=0
        ),
    }


def eval_p3(fork: list[dict], upstream: list[dict]) -> dict[str, Any]:
    if not fork:
        return _inconclusive("no fork runs")
    fk = _p3_summary(fork)
    all_ok = all(
        row[k].get("ok")
        for r in fork
        for row in r["scenario"]["rows"]
        for k in ("alone", "batched")
    )
    exercised = all(r["log"].get("tq_batched_fused_max_b", 0) >= 3 for r in fork)
    checks = [
        _check(
            "all requests completed",
            all_ok,
            "every B=1 and B=3 request ok" if all_ok else "a request failed",
        ),
        _check(
            "greedy B=3 == B=1 for every row",
            fk["mismatched_rows"] == 0,
            f"{fk['mismatched_rows']}/{fk['rows']} rows differ",
        ),
        _check(
            "batch exercised (log: per-row fused path B=3)",
            exercised,
            f"max B seen in log: {fk['tq_batched_fused_max_b']}",
        ),
    ]
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "fork": fk,
        "upstream": _p3_summary(upstream),
    }


def _p4_summary(runs: list[dict]) -> dict[str, Any]:
    if not runs:
        return {"serial_wall_s": None, "concurrent_wall_s": None}
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
    }


def eval_p4(
    fork: list[dict], upstream: list[dict], min_row_ratio: float = 0.5
) -> dict[str, Any]:
    if not fork:
        return _inconclusive("no fork runs")
    fk = _p4_summary(fork)
    ratios = [
        (c / s) if (c is not None and s) else None
        for s, c in zip(fk["serial_decode_tps"], fk["concurrent_decode_tps"])
    ]
    checks = [
        _check(
            "all requests completed",
            fk["completed"] == fk["total"],
            f"{fk['completed']}/{fk['total']} ok",
        ),
        _check("0 aborts", fk["aborts_total"] == 0, f"{fk['aborts_total']} aborts"),
        _check(
            "concurrent wall < serial wall",
            fk["concurrent_wall_s"] is not None
            and fk["concurrent_wall_s"] < fk["serial_wall_s"],
            f"concurrent {_fmt(fk['concurrent_wall_s'], 1)} s vs serial {_fmt(fk['serial_wall_s'], 1)} s",
        ),
        _check(
            f"per-row concurrent decode >= {int(min_row_ratio * 100)}% of serial",
            all(x is not None and x >= min_row_ratio for x in ratios),
            "ratios " + ", ".join(_fmt(x) for x in ratios),
        ),
    ]
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "fork": fk,
        "upstream": _p4_summary(upstream),
        "row_ratios": ratios,
    }
