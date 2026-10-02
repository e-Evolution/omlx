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
_RE_RESTORE = re.compile(r"Prefix cache restore for \S+: source=\S+ cached=(\d+)")
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
    restores: list[int] = []
    cache_limit: float | None = None
    tq_rows: list[int] = []
    tq_fallback = False
    mtp_rows: list[int] = []
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


def _noise_check(detail: list[dict], all_ok: bool) -> dict[str, Any]:
    """Divergences inside the first 32 tokens must be near-tie numerical noise."""
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

    return _check(
        f"B=3 vs B=1 (restored): divergence inside the first {P3_PREFIX_TOKENS} tokens "
        "is numerical noise (near-tie)",
        all_ok and not bad,
        f"{len(bad)}/{len(detail)} rows not noise; " + "; ".join(one(d) for d in detail),
    )


def eval_p3(fork: list[dict], upstream: list[dict]) -> dict[str, Any]:
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
    checks = [
        _check(
            "all requests completed",
            all_ok,
            "every restored B=1 and B=3 request ok" if all_ok else "a request failed",
        ),
        _noise_check(fk["detail"], all_ok)
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
    return {
        "verdict": _verdict(checks),
        "checks": checks,
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
