#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Premise benchmark for the unattended-scheduling work (upstream vs fork).

Starts its OWN oMLX server per (variant, premise) on an isolated base path,
drives it with streaming chat requests, samples process memory and /health,
parses the server log, and writes one JSON file per variant/premise/run.
Results and logs stay outside git (default: ``--out`` below).

Safety rules enforced here:
  * only PIDs of servers started by this process are ever signalled (taken
    from our own subprocess handles); no pkill/killall/pgrep-by-name;
  * port 8001 (the user's app) is refused unless ``--allow-port 8001``;
  * a port that already accepts connections is refused;
  * the real ~/.omlx is read-only (settings are copied, never written);
  * deletions only happen strictly inside the bench roots.

Premises: P1 no lost work, P2 no aborts at 2x130k, P3 batch correctness,
P4 concurrency pays, P6 no single-request regression. (P5 is pi-driven:
see p5_pi_unattended.sh.) P7 (opt-in, never part of "all") forces hard memory
pressure with a low custom ceiling and records whether the "park, don't
abort" path fires for text-only vs image-bearing long prefills.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import copy
import hashlib
import json
import os
import platform
import plistlib
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_metrics as bm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
REAL_BASE = HOME / ".omlx"
BENCH_BASE_ROOT = HOME / ".omlx-bench"
# The volume root is not writable from a terminal session (macOS privacy
# protection), so the bench cache lives under the volume's old `.omlx` dir,
# NOT under the live `oLMXCache` dir. Override with --cache-root.
CACHE_ROOT = "/Volumes/ExternalSSD/.omlx/oLMXCache-bench"
DEFAULT_OUT = "/tmp/omlx-unattended-bench"
MODEL = "Qwen3.8-27B-oQ4e-mtp"
UPSTREAM_APP = Path("/Applications/oMLX.app")
UPSTREAM_CLI = UPSTREAM_APP / "Contents/MacOS/omlx-cli"
FORK_BIN = REPO / ".venv/bin/omlx"
# The dev checkout has no compiled Metal kernels (omlx/custom_kernels/*/ *.so,
# *.metallib; the Qwen3.5 GDN prefill kernels among them), so `.venv/bin/omlx`
# prefills ~20% slower than the shipped app for ANY omlx source. Fork variants
# therefore run the app's own Python runtime with the fork's `omlx/` sources
# overlaid on the app's `omlx/` -- exactly what packaging/build_fork_app.py
# ships -- unless --fork-runtime venv is requested.
APP_RESOURCES = UPSTREAM_APP / "Contents/Resources"
OVERLAY_ROOT = BENCH_BASE_ROOT / "fork-overlay"
UPSTREAM_TAG = "v0.7.0"

FORK_ON_ENV = {
    "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "4",
    "OMLX_MEMORY_PRESSURE_PARK": "true",
    "OMLX_MLX_CACHE_LIMIT_GB": "16",
    "OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS": "200000",
}
FORK_OFF_ENV = {
    "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "0",
    "OMLX_MEMORY_PRESSURE_PARK": "false",
    "OMLX_MLX_CACHE_LIMIT_GB": "0",
    "OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS": "0",
}
# Fork code with exactly one feature on (P6 prefill-delta attribution).
FORK_ONLY_ON = {
    "fork-d1only": {"OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "4"},
    "fork-cachelimit-only": {"OMLX_MLX_CACHE_LIMIT_GB": "16"},
}
# Fork with the model's MTP speculative decode off: plain decode at B=1 and
# B=3, so a B=1 vs B=3 comparison isolates batching from MTP verify numerics.
VARIANTS = ["upstream", "fork", "fork-off", "fork-nomtp", *FORK_ONLY_ON]
# P3 flow: cold B=1 (populates the prefix cache), B=1 restored, B=3 restored.
P3_PHASES = ("cold", "alone_restored", "batched")
# Controls run after the comparison: B=1 restored again (run-to-run
# determinism) and B=1 forced through the ragged B>1 attention kernel.
P3_CONTROL_PHASES = ("alone_restored2", "ragged_b1")
# settings.json keys that only exist in the fork (stripped for upstream).
FORK_ONLY_SETTINGS = {
    "scheduler": [
        "prefill_progress_commit_blocks",
        "memory_pressure_park",
        "max_parks_per_request",
        "park_resume_ok_polls",
        "max_batched_decode_context_tokens",
    ],
    "memory": ["mlx_cache_limit_gb"],
}

SAMPLING_DEFAULT = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
SAMPLING_GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}

PREMISE_CONCURRENCY = {"P6": 1, "P1": 1, "P2": 2, "P3": 3, "P4": 3, "PL": 2, "P7": 3}
ALL_PREMISES = ["P6", "P1", "P2", "P3", "P4"]

_LIVE_SERVERS: list[Server] = []


# --------------------------------------------------------------------------
# Settings / environment per variant
# --------------------------------------------------------------------------


def variant_env(
    variant: str, premise: str, batched_limit: int | None = None
) -> dict[str, str]:
    if variant == "upstream":
        return {}
    env = dict(
        FORK_ON_ENV if variant in ("fork", "fork-nomtp") else FORK_OFF_ENV
    )
    env.update(FORK_ONLY_ON.get(variant, {}))
    if premise == "P3":
        # Disable the T5b gate so the three rows really batch.
        env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] = "0"
    if batched_limit is not None:
        env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"] = str(batched_limit)
    return env


def bench_extra_env(environ: Any = None) -> dict[str, str]:
    """Extra OMLX_* server env from ``OMLX_BENCH_EXTRA_ENV="K=V,K2=V2"``.

    Lets a bench run enable features the fork app turns on (for example
    OMLX_MID_SYSTEM_POLICY) without hard-coding them for every variant.
    """
    raw = (environ if environ is not None else os.environ).get("OMLX_BENCH_EXTRA_ENV", "")
    out: dict[str, str] = {}
    for item in (p.strip() for p in raw.split(",")):
        if not item:
            continue
        key, sep, val = item.partition("=")
        if not sep or not key.startswith("OMLX_"):
            raise ValueError(f"OMLX_BENCH_EXTRA_ENV entry must be OMLX_*=value: {item!r}")
        out[key] = val
    return out


def bench_fork_repo(environ: Any = None) -> Path:
    """Source tree overlaid on the app runtime: ``OMLX_BENCH_FORK_REPO`` or the harness tree."""
    raw = (environ if environ is not None else os.environ).get("OMLX_BENCH_FORK_REPO")
    if not raw:
        return REPO
    path = Path(raw).expanduser().resolve()
    if not (path / "omlx").is_dir():
        raise ValueError(f"OMLX_BENCH_FORK_REPO has no omlx/ package: {path}")
    return path


def variant_model_settings(ms: dict[str, Any], variant: str) -> dict[str, Any]:
    """Per-variant model_settings.json tweaks (fork-nomtp: MTP off)."""
    out = copy.deepcopy(ms)
    if variant == "fork-nomtp":
        out.setdefault("models", {}).setdefault(MODEL, {})["mtp_enabled"] = False
    return out


def debug_env(variant: str, premise: str, base: Path) -> dict[str, str]:
    """Opt-in fork debug switches for P3 (logprob dump + forced ragged B=1)."""
    if variant == "upstream" or premise != "P3":
        return {}
    return {
        "OMLX_DEBUG_LOGPROB_DUMP": str(base / "logprob_dump"),
        "OMLX_DEBUG_FORCE_RAGGED_B1": str(base / "force_ragged_b1.flag"),
    }


def build_settings(
    real: dict[str, Any],
    variant: str,
    port: int,
    cache_dir: str,
    max_concurrent: int,
    env: dict[str, str],
) -> dict[str, Any]:
    """Derive an isolated settings.json from a copy of the real one."""
    s = copy.deepcopy(real)
    s.setdefault("server", {})["port"] = port
    s["server"]["auto_start_on_launch"] = True
    s.setdefault("cache", {})["ssd_cache_dir"] = cache_dir
    s["cache"]["enabled"] = True
    s.setdefault("memory", {})["memory_guard_tier"] = "aggressive"
    s["memory"]["prefill_memory_guard"] = True
    s.setdefault("scheduler", {})["max_concurrent_requests"] = max_concurrent
    if variant == "upstream":
        for section, keys in FORK_ONLY_SETTINGS.items():
            for key in keys:
                s.get(section, {}).pop(key, None)
    else:
        sch, mem = s["scheduler"], s["memory"]
        sch["prefill_progress_commit_blocks"] = int(
            env["OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS"]
        )
        sch["memory_pressure_park"] = env["OMLX_MEMORY_PRESSURE_PARK"] == "true"
        sch["max_batched_decode_context_tokens"] = int(
            env["OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS"]
        )
        mem["mlx_cache_limit_gb"] = float(env["OMLX_MLX_CACHE_LIMIT_GB"])
    return s


# --------------------------------------------------------------------------
# Server management (own PIDs only)
# --------------------------------------------------------------------------


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _lstart(pid: int) -> str:
    return subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()


def descendants(pid: int) -> list[int]:
    """PIDs below `pid` in the process tree (children we spawned transitively)."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True
    ).stdout
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        a, b = line.split()
        kids.setdefault(int(b), []).append(int(a))
    found: list[int] = []
    stack = [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    return found


def build_fork_overlay(
    app_resources: Path = APP_RESOURCES,
    fork_repo: Path = REPO,
    root: Path = OVERLAY_ROOT,
) -> Path:
    """Copy the app's ``omlx/`` (compiled kernels included), overlay the fork's
    sources and return the launcher path. Safe to call repeatedly."""
    marker = root / ".bench-overlay"
    if root.exists():
        if not marker.exists():
            raise RuntimeError(f"refusing to delete a non-overlay dir: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    marker.write_text("owned by bench_premises.build_fork_overlay\n")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(app_resources / "omlx", root / "omlx", ignore=ignore)
    shutil.copytree(fork_repo / "omlx", root / "omlx", ignore=ignore, dirs_exist_ok=True)
    py_root = app_resources / "Python"
    launcher = root / "omlx-fork-cli"
    launcher.write_text(
        "#!/bin/sh\nset -eu\n"
        f'PYROOT="{py_root}"\n'
        'export PYTHONHOME="$PYROOT/cpython-3.11"\n'
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f'export PYTHONPATH="{root}:$PYROOT/framework-mlx-base/lib/python3.11/site-packages"\n'
        'exec "$PYTHONHOME/bin/python3" -m omlx.cli "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


class Server:
    def __init__(
        self,
        variant: str,
        premise: str,
        port: int,
        max_concurrent: int,
        out: Path,
        keep_cache: bool,
        batched_limit: int | None = None,
        settings_override: dict[str, Any] | None = None,
    ):
        self.variant, self.premise, self.port = variant, premise, port
        # Deep-merged over the derived settings.json (never the real one).
        self.settings_override = settings_override or {}
        self.max_concurrent = max_concurrent
        self.out = out
        self.keep_cache = keep_cache
        tag = f"{variant}-{premise}"
        self.base = BENCH_BASE_ROOT / tag
        self.cache_dir = f"{CACHE_ROOT}/{tag}"
        self.env = variant_env(variant, premise, batched_limit)
        if variant != "upstream":
            self.env.update(bench_extra_env())
        self.env.update(debug_env(variant, premise, self.base))
        dump = self.env.get("OMLX_DEBUG_LOGPROB_DUMP")
        self.dump_dir = Path(dump) if dump else None
        flag = self.env.get("OMLX_DEBUG_FORCE_RAGGED_B1")
        self.ragged_flag = Path(flag) if flag else None
        self.proc: subprocess.Popen | None = None
        self.known_pids: dict[int, str] = {}  # pid -> process start time
        self.stdout_path = out / "logs" / f"{tag}.stdout.log"
        self.api_key = ""
        self.settings: dict[str, Any] = {}

    # -- preparation -------------------------------------------------------
    def _rmtree_guarded(self, path: str, roots: list[str]) -> None:
        if not bm.is_safe_bench_path(path, roots):
            raise RuntimeError(f"refusing to delete outside bench roots: {path}")
        shutil.rmtree(path, ignore_errors=True)

    def prepare(self) -> None:
        real = json.loads((REAL_BASE / "settings.json").read_text())
        self._rmtree_guarded(str(self.base), [str(BENCH_BASE_ROOT)])
        (self.base / "logs").mkdir(parents=True)
        self._rmtree_guarded(self.cache_dir, [CACHE_ROOT])
        Path(self.cache_dir).mkdir(parents=True)
        self.settings = build_settings(
            real, self.variant, self.port, self.cache_dir, self.max_concurrent, self.env
        )
        if self.settings_override:
            self.settings = bm.deep_merge(self.settings, self.settings_override)
        sp = self.base / "settings.json"
        sp.write_text(json.dumps(self.settings, indent=2))
        sp.chmod(0o600)
        ms = REAL_BASE / "model_settings.json"
        if ms.exists():
            (self.base / "model_settings.json").write_text(
                json.dumps(
                    variant_model_settings(json.loads(ms.read_text()), self.variant),
                    indent=2,
                )
            )
        self.api_key = self.settings.get("auth", {}).get("api_key") or ""

    # -- lifecycle ---------------------------------------------------------
    def command(self) -> list[str]:
        if self.variant == "upstream":
            exe = str(UPSTREAM_CLI)
        elif os.environ.get("OMLX_BENCH_FORK_RUNTIME", "app") == "venv":
            exe = str(FORK_BIN)
        else:
            exe = str(build_fork_overlay(fork_repo=bench_fork_repo()))
        return [
            exe,
            "serve",
            "--base-path",
            str(self.base),
            "--port",
            str(self.port),
            "--model-dir",
            str(REAL_BASE / "models"),
        ]

    def start(self) -> None:
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("OMLX_")}
        env.update(self.env)
        fh = open(self.stdout_path, "wb")  # noqa: SIM115 - handed to Popen
        self.proc = subprocess.Popen(
            self.command(),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(self.base),
        )
        _LIVE_SERVERS.append(self)
        print(
            f"[server] {self.variant}/{self.premise} pid={self.proc.pid} port={self.port}",
            flush=True,
        )

    def pids(self) -> list[int]:
        if self.proc is None:
            return []
        pids = [self.proc.pid, *descendants(self.proc.pid)]
        for pid in pids:
            if pid not in self.known_pids:
                self.known_pids[pid] = _lstart(pid)
        return pids

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> Path:
        return self.base / "logs" / "server.log"

    def log_size(self) -> int:
        return self.log_path.stat().st_size if self.log_path.exists() else 0

    def log_since(self, offset: int) -> str:
        if not self.log_path.exists():
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        import httpx

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {self.proc.returncode}); see {self.stdout_path}"
                )
            try:
                r = httpx.get(
                    f"{self.base_url}/health", headers=self.headers(), timeout=3.0
                )
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError("server did not become healthy")

    def stop(self, grace_s: float = 120.0) -> None:
        """Stop ONLY the processes this object spawned (recorded PIDs)."""
        proc = self.proc
        if proc is None:
            return
        self.pids()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        for pid, started in sorted(self.known_pids.items()):
            if pid == proc.pid:
                continue
            # Only descendants recorded from our own process tree, and only
            # if the PID still belongs to the same process (start time).
            if not started or _lstart(pid) != started:
                continue
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        self.proc = None
        if self in _LIVE_SERVERS:
            _LIVE_SERVERS.remove(self)
        print(
            f"[server] stopped {self.variant}/{self.premise} (port {self.port} "
            f"{'still busy!' if port_in_use(self.port) else 'free'})",
            flush=True,
        )

    def archive_and_clean(self) -> None:
        dst = self.out / "logs" / f"{self.variant}-{self.premise}.server.log"
        if self.log_path.exists():
            shutil.copy2(self.log_path, dst)
        if not self.keep_cache:
            self._rmtree_guarded(self.cache_dir, [CACHE_ROOT])


def _cleanup_all() -> None:
    for srv in list(_LIVE_SERVERS):
        try:
            srv.stop(grace_s=60)
        except Exception as exc:  # noqa: BLE001
            print(f"[cleanup] failed to stop server: {exc}", file=sys.stderr)


atexit.register(_cleanup_all)


def _signal_exit(signum, _frame):
    _cleanup_all()
    sys.exit(128 + signum)


# --------------------------------------------------------------------------
# Sampler (footprint + /health every second)
# --------------------------------------------------------------------------


class Sampler:
    def __init__(self, server: Server, interval_s: float = 1.0):
        self.server, self.interval = server, interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _loop(self) -> None:
        import httpx

        while not self._stop.is_set():
            tick = time.time()
            fp = None
            pids = self.server.pids()
            if pids:
                try:
                    out = subprocess.run(
                        ["footprint", *map(str, pids)],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    ).stdout
                    fp = bm.parse_footprint_gb(out)
                except Exception:
                    fp = None
            h_ok, h_s = True, None
            try:
                t = time.time()
                r = httpx.get(
                    f"{self.server.base_url}/health",
                    headers=self.server.headers(),
                    timeout=5.0,
                )
                h_s = time.time() - t
                h_ok = r.status_code == 200
            except Exception:
                h_ok = False
            self.samples.append(
                {
                    "t": round(tick - self._t0, 2),
                    "footprint_gb": fp,
                    "health_ok": h_ok,
                    "health_s": h_s,
                }
            )
            self._stop.wait(max(0.0, self.interval - (time.time() - tick)))

    def start(self) -> None:
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        return self.samples


# --------------------------------------------------------------------------
# Corpus / prompts
# --------------------------------------------------------------------------


class Corpus:
    """Deterministic token corpus built from the repo's own source files."""

    def __init__(self, model_path: Path, cache_dir: Path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_path))
        files = sorted((REPO / "omlx").rglob("*.py"))
        parts, size = [], 0
        for f in files:
            if f.stat().st_size > 2_000_000:
                continue
            body = f.read_text(errors="replace")
            parts.append(f"### {f.relative_to(REPO)}\n{body}\n")
            size += len(body)
            if size > 3_000_000:
                break
        text = "".join(parts)
        key = hashlib.sha1((text + str(model_path)).encode()).hexdigest()[:12]
        cache = cache_dir / f"corpus_ids_{key}.json"
        if cache.exists():
            self.ids = json.loads(cache.read_text())
        else:
            self.ids = self.tok(text, add_special_tokens=False)["input_ids"]
            cache.write_text(json.dumps(self.ids))
        self.suffix = "\n\nTask: summarize the code above in about 150 words."
        self.overhead = (
            len(
                self.tok(self.suffix + bm.make_nonce("x"), add_special_tokens=False)[
                    "input_ids"
                ]
            )
            + 24
        )

    def prompt(self, seed: str, n_tokens: int) -> str:
        """Prompt of ~n_tokens with a unique nonce prefix (cold prefix cache)."""
        start = int(hashlib.sha1(seed.encode()).hexdigest(), 16) % len(self.ids)
        body = bm.build_prompt_text(
            bm.make_nonce(seed),
            self.ids,
            start,
            max(16, n_tokens - self.overhead),
            lambda xs: self.tok.decode(xs, skip_special_tokens=False),
        )
        return body + self.suffix


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


async def stream_chat(
    client,
    server: Server,
    prompt: Any,
    max_tokens: int,
    sampling: dict[str, Any],
    t_zero: float,
    name: str = "",
    cancel_after_s: float | None = None,
    keep_text: bool = False,
    keep_chunks: bool = False,
) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling,
    }
    acc = bm.StreamAccumulator()
    t_start = time.time()
    result: dict[str, Any] = {"name": name, "start_s": round(t_start - t_zero, 2)}

    async def consume() -> None:
        async with client.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            json=body,
            headers=server.headers(),
        ) as resp:
            if resp.status_code != 200:
                payload = (await resp.aread()).decode("utf-8", "replace")[:300]
                acc.errors.append(f"HTTP {resp.status_code}: {payload}")
                return
            async for line in resp.aiter_lines():
                acc.feed(line, now=time.time())

    try:
        if cancel_after_s is not None:
            try:
                await asyncio.wait_for(consume(), timeout=cancel_after_s)
            except TimeoutError:
                result.update(
                    ok=False,
                    cancelled=True,
                    error="client-cancelled",
                    cancel_after_s=cancel_after_s,
                    first_token_seen=acc.first_token_at is not None,
                )
                return result
        else:
            await consume()
    except Exception as exc:  # noqa: BLE001
        acc.errors.append(f"{type(exc).__name__}: {exc}")
    t_end = time.time()
    fin = acc.finalize()
    m = bm.request_metrics(
        t_start,
        fin["first_token_at"],
        t_end,
        fin["prompt_tokens"],
        fin["completion_tokens"],
    )
    result.update(
        ok=fin["ok"],
        error=fin["error"],
        prompt_tokens=fin["prompt_tokens"],
        completion_tokens=fin["completion_tokens"],
        cached_tokens=fin["cached_tokens"],
        end_s=round(t_end - t_zero, 2),
        text_len=len(fin["text"]),
        **{k: (round(v, 3) if v is not None else None) for k, v in m.items()},
    )
    if keep_text:
        result["text"] = fin["text"]
    if keep_chunks:
        # Per-chunk arrival times (relative to the run start) and sizes: the
        # raw material for the steady-decode window and the 32-token prefix.
        result["chunk_times"] = [round(t - t_zero, 3) for t in fin["chunk_times"]]
        result["chunk_chars"] = fin["chunk_chars"]
    return result


def make_client():
    import httpx

    return httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=60.0),
        limits=httpx.Limits(max_connections=16),
    )


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


class Ctx:
    def __init__(
        self, server: Server, corpus: Corpus, client, args: argparse.Namespace, run: int
    ):
        self.server, self.corpus, self.client, self.args, self.run = (
            server,
            corpus,
            client,
            args,
            run,
        )
        self.t0 = time.time()

    def n(self, nominal: int) -> int:
        return bm.scale_tokens(nominal, self.args.smoke)

    def mt(self, nominal: int) -> int:
        return min(nominal, 16) if self.args.smoke else nominal

    def seed(self, *parts: Any) -> str:
        # Independent of the variant, so every variant gets identical prompts.
        return "|".join(["bench", *map(str, parts), f"run{self.run}"])

    async def chat(
        self, prompt: str, max_tokens: int, sampling=SAMPLING_DEFAULT, **kw
    ) -> dict[str, Any]:
        return await stream_chat(
            self.client, self.server, prompt, max_tokens, sampling, self.t0, **kw
        )


async def scenario_p6(ctx: Ctx) -> dict[str, Any]:
    prompt = ctx.corpus.prompt(ctx.seed("P6"), ctx.n(110_000))
    return {"request": await ctx.chat(prompt, ctx.mt(256), name="110k")}


async def scenario_p1(ctx: Ctx) -> dict[str, Any]:
    frac = ctx.args.p1_cancel_fraction
    n = ctx.n(110_000)
    calib = await ctx.chat(
        ctx.corpus.prompt(ctx.seed("P1", "calibration"), n), 1, name="calibration"
    )
    if not calib.get("ok") or not calib.get("ttft_s"):
        return {
            "error": "calibration failed",
            "calibration": calib,
            "reissue": {"ok": False, "error": "calibration failed"},
        }
    cancel_after = frac * calib["ttft_s"]
    prompt = ctx.corpus.prompt(ctx.seed("P1", "main"), n)
    cancelled = await ctx.chat(
        prompt, ctx.mt(256), name="cancelled", cancel_after_s=cancel_after
    )
    # Let the server notice the disconnect and settle before the re-issue.
    await asyncio.sleep(ctx.args.settle_s)
    off = ctx.server.log_size()
    reissue = await ctx.chat(prompt, 16, name="reissue")
    window = bm.parse_server_log(ctx.server.log_since(off))
    prompt_tokens = reissue.get("prompt_tokens") or calib.get("prompt_tokens") or 0
    cached = window["restore_cached_last"]
    if cached is None:
        cached = reissue.get("cached_tokens") or 0
    return {
        "calibration": calib,
        "cancel_fraction": frac,
        "cancel_after_s": round(cancel_after, 2),
        "cancelled": cancelled,
        "prompt_tokens": prompt_tokens,
        "progress_tokens_est": int(frac * prompt_tokens),
        "reissue": reissue,
        "reissue_cached": cached,
        "reissue_reprefilled": max(0, prompt_tokens - cached),
        "reissue_log_window": {
            "restores": window["restores"],
            "aborts": window["aborts_total"],
        },
    }


async def scenario_p2(ctx: Ctx) -> dict[str, Any]:
    prompts = [ctx.corpus.prompt(ctx.seed("P2", i), ctx.n(130_000)) for i in range(2)]
    t = time.time()
    results = await asyncio.gather(
        *[ctx.chat(p, ctx.mt(1024), name=f"130k-{i}") for i, p in enumerate(prompts)]
    )
    return {"requests": list(results), "wall_s": round(time.time() - t, 2)}


async def scenario_p3(ctx: Ctx) -> dict[str, Any]:
    """Like-for-like batch correctness (all arms read from the prefix cache).

    1. cold B=1 per prompt: populates the prefix cache only.
    2. B=1 restored per prompt: the reference output.
    3. B=3 restored, started together: the short suffix prefills finish
       quickly, so all three rows DECODE in one batch.
    4. Controls (fork only): B=1 restored again (run-to-run determinism) and
       B=1 restored through the ragged B>1 kernel (kernel numerics vs
       batching). With the fork's logprob dump every token carries its top-5
       logprobs, so the first divergence is judged by its near-tie margin.
    """
    lens = [("20k", 20_000), ("45k", 45_000), ("135k", 135_000)]
    mt = ctx.mt(ctx.args.p3_max_tokens)
    prompts = {
        name: ctx.corpus.prompt(ctx.seed("P3", name), ctx.n(nom)) for name, nom in lens
    }
    rows: dict[str, dict[str, Any]] = {name: {"name": name} for name, _ in lens}
    dump_dir = ctx.server.dump_dir

    def attach_lp(name: str, phase: str, res: dict[str, Any], paths: list[Path]):
        if dump_dir is None:
            return
        path = bm.pick_dump(paths, res.get("prompt_tokens"))
        if path is None and len(paths) == 1:
            path = paths[0]
        if path is not None:
            rows[name].setdefault("logprobs", {})[phase] = bm.load_logprob_dump(path)

    async def one(name: str, tag: str) -> dict[str, Any]:
        return await ctx.chat(
            prompts[name],
            mt,
            SAMPLING_GREEDY,
            name=f"{name}-{tag}",
            keep_text=True,
            keep_chunks=True,
        )

    async def alone(name: str, key: str, tag: str) -> None:
        before = bm.dump_listing(dump_dir) if dump_dir else set()
        rows[name][key] = await one(name, tag)
        if dump_dir:
            attach_lp(name, key, rows[name][key], bm.new_dumps(dump_dir, before))

    for name, _ in lens:  # cold: populate the prefix cache
        rows[name]["alone"] = await one(name, "cold")
    for name, _ in lens:  # B=1, restored (reference)
        await alone(name, "alone_restored", "alone-restored")
    off = ctx.server.log_size()
    before = bm.dump_listing(dump_dir) if dump_dir else set()
    t = time.time()
    order = [name for name, _ in reversed(lens)]  # longest first
    batched = await asyncio.gather(*[one(nm, "batched") for nm in order])
    batched_wall = round(time.time() - t, 2)
    new_paths = bm.new_dumps(dump_dir, before) if dump_dir else []
    for nm, res in zip(order, batched):
        rows[nm]["batched"] = res
        attach_lp(nm, "batched", res, new_paths)
    batch_log = bm.parse_server_log(ctx.server.log_since(off))
    if dump_dir is not None:  # controls (fork only: they need the debug hooks)
        for name, _ in lens:
            await alone(name, "alone_restored2", "alone-restored2")
        ctl_off = ctx.server.log_size()
        flag = ctx.server.ragged_flag
        flag.write_text("1")
        try:
            await asyncio.sleep(1.0)  # the server polls the flag file
            for name, _ in lens:
                await alone(name, "ragged_b1", "ragged-b1")
        finally:
            flag.unlink(missing_ok=True)
    forced_seen = (
        "L=1, forced)" in ctx.server.log_since(ctl_off) if dump_dir is not None else None
    )
    return {
        "rows": [rows[name] for name, _ in lens],
        "ragged_b1_forced_logged": forced_seen,
        "batched_wall_s": batched_wall,
        "batched_log_window": {
            "batch_max_b": batch_log["batch_max_b"],
            "tq_batched_fused_rows": batch_log["tq_batched_fused_rows"],
            "mtp_batch_rows": batch_log["mtp_batch_rows"],
            "restores": batch_log["restores"],
        },
        "phases": list(P3_PHASES),
        "control_phases": list(P3_CONTROL_PHASES) if dump_dir is not None else [],
        "note": "cold B=1 only populates the cache; B=1 restored vs B=3 restored is the comparison",
    }


async def scenario_p4(ctx: Ctx) -> dict[str, Any]:
    lens = [("20k", 20_000), ("45k", 45_000), ("90k", 90_000)]
    mt = ctx.mt(ctx.args.p4_max_tokens)

    def prompts(arm: str) -> list[str]:
        return [
            ctx.corpus.prompt(ctx.seed("P4", arm, nm), ctx.n(nom)) for nm, nom in lens
        ]

    serial_prompts, conc_prompts = prompts("serial"), prompts("concurrent")
    t = time.time()
    serial = [
        await ctx.chat(p, mt, name=f"{nm}-serial")
        for (nm, _), p in zip(lens, serial_prompts)
    ]
    serial_wall = round(time.time() - t, 2)
    await asyncio.sleep(ctx.args.settle_s)
    t = time.time()
    order = list(reversed(range(len(lens))))
    res = await asyncio.gather(
        *[
            ctx.chat(
                conc_prompts[i],
                mt,
                name=f"{lens[i][0]}-concurrent",
                keep_chunks=True,
            )
            for i in order
        ]
    )
    conc_wall = round(time.time() - t, 2)
    conc = [None] * len(lens)
    for i, r in zip(order, res):
        conc[i] = r
    return {
        "serial": {"wall_s": serial_wall, "requests": serial},
        "concurrent": {"wall_s": conc_wall, "requests": conc},
    }


async def scenario_pl(ctx: Ctx) -> dict[str, Any]:
    """Limit probe: ~165k + ~95k distinct prompts at once, default sampling."""
    lens = [("165k", 165_000), ("95k", 95_000)]
    mt = ctx.mt(ctx.args.pl_max_tokens)
    prompts = [ctx.corpus.prompt(ctx.seed("PL", nm), ctx.n(nom)) for nm, nom in lens]
    res = await asyncio.gather(
        *[
            ctx.chat(p, mt, name=nm, keep_text=True, keep_chunks=True)
            for (nm, _), p in zip(lens, prompts)
        ]
    )
    return {"requests": list(res)}


P7_NOMINAL_TOKENS = (130_000, 115_000, 100_000)


async def scenario_p7(ctx: Ctx) -> dict[str, Any]:
    """Forced hard pressure: does a prefill get parked instead of aborted?

    2-3 distinct long prompts are started a few seconds apart so they prefill
    concurrently under a low custom memory ceiling (set on the server by
    ``--p7-ceiling-gb``). ``--p7-mode image`` puts an image at the start of
    every prompt (``vlm_inputs_embeds`` set: not parkable). Requests that fail
    are re-issued once, as the pi extension would, to show how much of the
    prompt the progress commit kept warm.
    """
    a = ctx.args
    mode = a.p7_mode
    count = max(1, min(a.p7_requests, len(P7_NOMINAL_TOKENS)))
    prompts: list[Any] = []
    for i in range(count):
        text = ctx.corpus.prompt(ctx.seed("P7", i), ctx.n(P7_NOMINAL_TOKENS[i]))
        prompts.append(bm.p7_content(text, mode, ctx.seed("P7", i, "img")))
    mt = ctx.mt(a.p7_max_tokens)
    off = ctx.server.log_size()

    async def one(i: int) -> dict[str, Any]:
        await asyncio.sleep(i * a.p7_stagger_s)
        return await ctx.chat(prompts[i], mt, name=f"req{i}", keep_text=True)

    t = time.time()
    results = list(await asyncio.gather(*[one(i) for i in range(count)]))
    wall = round(time.time() - t, 2)
    reissues: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(results):
        if r.get("ok"):
            continue
        await asyncio.sleep(a.settle_s)
        reissues[r["name"]] = await ctx.chat(
            prompts[i], min(mt, 64), name=f"{r['name']}-reissue", keep_text=True
        )
    events = bm.parse_park_events(ctx.server.log_since(off))
    summary = bm.summarize_p7(results, reissues, events)
    for r in [*results, *reissues.values()]:
        r.pop("text", None)
    return {
        "mode": mode,
        "ceiling_gb": a.p7_ceiling_gb,
        "stagger_s": a.p7_stagger_s,
        "requests": results,
        "reissues": reissues,
        "wall_s": wall,
        "summary": summary,
    }


SCENARIOS = {
    "P7": scenario_p7,
    "PL": scenario_pl,
    "P6": scenario_p6,
    "P1": scenario_p1,
    "P2": scenario_p2,
    "P3": scenario_p3,
    "P4": scenario_p4,
}


def scenario_ok(premise: str, sc: dict[str, Any]) -> bool:
    def okreq(r):
        return bool(r and r.get("ok"))

    if premise == "P6":
        return okreq(sc["request"])
    if premise == "P1":
        return okreq(sc.get("reissue"))
    if premise in ("P2", "PL"):
        return all(okreq(r) for r in sc["requests"])
    if premise == "P3":
        return all(
            okreq(row[k])
            for row in sc["rows"]
            for k in ("alone", "alone_restored", "batched")
        )
    if premise == "P4":
        return all(
            okreq(r) for m in ("serial", "concurrent") for r in sc[m]["requests"]
        )
    if premise == "P7":
        return bool(sc["summary"]["all_final_ok"])
    return False


# --------------------------------------------------------------------------
# Setup block / orchestration
# --------------------------------------------------------------------------


def _sh(*cmd: str, cwd: str | None = None) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=cwd
        ).stdout.strip()
    except Exception:
        return ""


def collect_setup(args: argparse.Namespace, server: Server) -> dict[str, Any]:
    app = {}
    try:
        with open(UPSTREAM_APP / "Contents/Info.plist", "rb") as f:
            info = plistlib.load(f)
        app = {
            "version": info.get("CFBundleShortVersionString"),
            "build": info.get("CFBundleVersion"),
        }
    except Exception:
        pass
    ms = {}
    try:
        allm = json.loads((REAL_BASE / "model_settings.json").read_text())
        ms = (allm.get("models") or allm).get(MODEL, {})
    except Exception:
        pass
    st = server.settings
    return {
        "variant": server.variant,
        "premise": server.premise,
        "smoke": args.smoke,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fork_commit": _sh("git", "rev-parse", "HEAD", cwd=str(REPO)),
        "fork_describe": _sh("git", "describe", "--tags", "--always", cwd=str(REPO)),
        "fork_branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=str(REPO)),
        "upstream_tag": UPSTREAM_TAG,
        "upstream_commit": _sh(
            "git", "rev-parse", f"{UPSTREAM_TAG}^{{commit}}", cwd=str(REPO)
        ),
        "upstream_app": app,
        "chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_gb": round(int(_sh("sysctl", "-n", "hw.memsize") or 0) / 1024**3),
        "macos": _sh("sw_vers", "-productVersion"),
        "python": platform.python_version(),
        "model": MODEL,
        "model_settings": {
            k: ms.get(k)
            for k in (
                "turboquant_kv_enabled",
                "turboquant_kv_bits",
                "mtp_enabled",
                "enable_thinking",
                "thinking_budget_enabled",
                "thinking_budget_tokens",
            )
        },
        "server_command": [Path(server.command()[0]).name, *server.command()[1:]],
        "feature_env": server.env,
        "max_concurrent_requests": server.max_concurrent,
        "cache": {
            "ssd_cache_dir": server.cache_dir,
            "ssd_cache_max_size": st.get("cache", {}).get("ssd_cache_max_size"),
            "hot_cache_max_size": st.get("cache", {}).get("hot_cache_max_size"),
        },
        "memory": {
            "memory_guard_tier": st.get("memory", {}).get("memory_guard_tier"),
            "mlx_cache_limit_gb": st.get("memory", {}).get("mlx_cache_limit_gb"),
        },
        "sampling": {"default": SAMPLING_DEFAULT, "greedy": SAMPLING_GREEDY},
        "warmup": "1 request of ~2k tokens, max_tokens 16, before the first measured run",
        "cache_condition": "cold: unique nonce prefix per prompt, fresh SSD cache dir per server start",
        "p1_cancel_fraction": args.p1_cancel_fraction,
        "p3_max_tokens": args.p3_max_tokens,
        "p4_max_tokens": args.p4_max_tokens,
        "settings_override": server.settings_override,
        "p7": {
            "mode": getattr(args, "p7_mode", None),
            "ceiling_gb": getattr(args, "p7_ceiling_gb", None),
            "hard_threshold": getattr(args, "p7_hard_threshold", None),
            "soft_threshold": getattr(args, "p7_soft_threshold", None),
            "requests": getattr(args, "p7_requests", None),
            "stagger_s": getattr(args, "p7_stagger_s", None),
            "max_tokens": getattr(args, "p7_max_tokens", None),
        }
        if server.premise == "P7"
        else None,
        "p3_flow": "cold B=1 (cache populate) -> B=1 restored -> B=3 restored (started together, "
        "OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS=0 in fork variants, max_concurrent_requests=3)",
    }


async def run_premise_async(
    args: argparse.Namespace, variant: str, premise: str, out: Path, corpus: Corpus
) -> int:
    srv = Server(
        variant,
        premise,
        args.port,
        args.max_concurrent or PREMISE_CONCURRENCY[premise],
        out,
        args.keep_cache,
        args.batched_decode_limit,
        settings_override=(
            bm.p7_settings_override(
                args.p7_ceiling_gb, args.p7_hard_threshold, args.p7_soft_threshold
            )
            if premise == "P7"
            else None
        ),
    )
    srv.prepare()
    srv.start()
    code = 0
    try:
        srv.wait_ready()
        setup = collect_setup(args, srv)
        async with make_client() as client:
            # The model (and its PagedSSDCacheManager) loads lazily on the
            # first request, so the warm-up doubles as the load trigger.
            warm = Ctx(srv, corpus, client, args, run=0)
            w = await warm.chat(
                corpus.prompt(warm.seed("warmup"), 2000), 16, name="warmup"
            )
            print(
                f"[warmup] ok={w.get('ok')} ttft={w.get('ttft_s')} err={w.get('error')}",
                flush=True,
            )
            startup = bm.parse_server_log(srv.log_since(0))
            if not startup["ssd_initialized"] or startup["ssd_disabled"]:
                print(
                    f"[FATAL] SSD cache not initialized for {variant}/{premise} "
                    f"(disabled={startup['ssd_disabled']}); see {srv.log_path}",
                    file=sys.stderr,
                )
                return 3
            print(
                f"[ok] PagedSSDCacheManager initialized ({variant}/{premise})",
                flush=True,
            )
            for k in range(1, args.runs + 1):
                ctx = Ctx(srv, corpus, client, args, run=k)
                off = srv.log_size()
                sampler = Sampler(srv)
                sampler.start()
                t_run = time.time()
                try:
                    scenario = await SCENARIOS[premise](ctx)
                finally:
                    samples = sampler.stop()
                log = bm.parse_server_log(srv.log_since(off))
                ok = scenario_ok(premise, scenario)
                run = {
                    "schema": 1,
                    "variant": variant,
                    "premise": premise,
                    "run": k,
                    "runs_requested": args.runs,
                    "ok": ok,
                    "setup": setup,
                    "scenario": scenario,
                    "log": log,
                    "samples": bm.summarize_samples(samples),
                    "wall_s": round(time.time() - t_run, 2),
                }
                d = out / variant / premise
                d.mkdir(parents=True, exist_ok=True)
                (d / f"run{k}.json").write_text(json.dumps(run, indent=2))
                (d / f"run{k}.samples.json").write_text(json.dumps(samples))
                print(
                    f"[run] {variant}/{premise} run {k}/{args.runs} ok={ok} wall={run['wall_s']}s "
                    f"peak={run['samples']['footprint_peak_gb']} GB aborts={log['aborts_total']}",
                    flush=True,
                )
                if not ok:
                    code = 1
                await asyncio.sleep(args.settle_s)
    finally:
        srv.stop()
        srv.archive_and_clean()
    return code


async def _load_model(srv: Server) -> None:
    """Tiny request so the model (and its SSD cache) is loaded before READY."""
    async with make_client() as client:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        }
        await client.post(
            f"{srv.base_url}/v1/chat/completions", json=body, headers=srv.headers()
        )


def serve_only(args: argparse.Namespace, out: Path) -> int:
    premise = "P6"
    srv = Server(
        args.variant, premise, args.port, args.max_concurrent or 3, out, args.keep_cache
    )
    srv.prepare()
    srv.start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        srv.wait_ready()
        asyncio.run(_load_model(srv))
        startup = bm.parse_server_log(srv.log_since(0))
        if not startup["ssd_initialized"]:
            print("[FATAL] SSD cache not initialized", file=sys.stderr)
            return 3
        print(
            f"READY variant={args.variant} port={args.port} base={srv.base} log={srv.log_path}",
            flush=True,
        )
        stop.wait(args.serve_seconds or None)
    finally:
        srv.stop()
        srv.archive_and_clean()
    return 0


def main(argv: list[str] | None = None) -> int:
    global CACHE_ROOT
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument(
        "--premise", choices=[*ALL_PREMISES, "PL", "P7", "all"], default="all"
    )
    ap.add_argument(
        "--p7-mode",
        choices=["text", "image"],
        default="text",
        help="P7: text-only prompts, or an image at the start of each prompt",
    )
    ap.add_argument(
        "--p7-ceiling-gb",
        type=float,
        default=None,
        help="P7: custom memory-guard ceiling in GB (default: between the model "
        "and the unpressured peak, see bench_metrics.p7_default_ceiling_gb)",
    )
    ap.add_argument(
        "--p7-hard-threshold",
        type=float,
        default=None,
        help="P7: memory.hard_threshold (fraction of the ceiling); lower widens "
        "the band where a park can fire before the emergency brake",
    )
    ap.add_argument(
        "--p7-soft-threshold",
        type=float,
        default=None,
        help="P7: memory.soft_threshold (default: hard - 0.1 when hard is set)",
    )
    ap.add_argument("--p7-requests", type=int, default=3, help="P7: 1-3 prompts")
    ap.add_argument(
        "--p7-stagger-s", type=float, default=5.0, help="P7: seconds between starts"
    )
    ap.add_argument("--p7-max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument(
        "--cache-root",
        default=CACHE_ROOT,
        help="parent of the per-variant bench SSD cache dirs",
    )
    ap.add_argument(
        "--allow-port",
        type=int,
        action="append",
        default=[],
        help="explicitly allow a protected port (8001)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny prompts (nominal/200) and max_tokens<=16; results flagged smoke",
    )
    ap.add_argument(
        "--max-concurrent", type=int, default=0, help="override max_concurrent_requests"
    )
    ap.add_argument("--p1-cancel-fraction", type=float, default=0.6)
    ap.add_argument("--p3-max-tokens", type=int, default=512)
    ap.add_argument(
        "--p4-max-tokens",
        type=int,
        default=512,
        help="tokens per P4 request (raise it to widen the all-three-decoding window)",
    )
    ap.add_argument(
        "--pl-max-tokens", type=int, default=1500, help="tokens per PL request"
    )
    ap.add_argument(
        "--batched-decode-limit",
        type=int,
        default=None,
        help="override scheduler.max_batched_decode_context_tokens (0 = no limit)",
    )
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep the bench SSD cache dir after the run",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="only start the isolated server (for P5) and wait",
    )
    ap.add_argument(
        "--serve-seconds",
        type=float,
        default=0.0,
        help="with --serve: stop after N seconds (0 = until SIGINT/SIGTERM)",
    )
    args = ap.parse_args(argv)

    if not bm.port_allowed(args.port, args.allow_port):
        print(
            f"refusing protected port {args.port} (pass --allow-port {args.port} to override)",
            file=sys.stderr,
        )
        return 2
    if port_in_use(args.port):
        print(
            f"port {args.port} already accepts connections; refusing to start a server there",
            file=sys.stderr,
        )
        return 2
    if args.variant == "upstream" and not UPSTREAM_CLI.exists():
        print(f"missing {UPSTREAM_CLI}", file=sys.stderr)
        return 2
    if args.variant != "upstream" and not FORK_BIN.exists():
        print(f"missing {FORK_BIN}", file=sys.stderr)
        return 2
    CACHE_ROOT = args.cache_root
    if args.p7_ceiling_gb is None:
        args.p7_ceiling_gb = bm.p7_default_ceiling_gb()
    if args.premise == "P7":
        print(f"[P7] custom memory-guard ceiling = {args.p7_ceiling_gb} GB", flush=True)
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    real_settings = REAL_BASE / "settings.json"
    before = hashlib.sha256(real_settings.read_bytes()).hexdigest()

    signal.signal(signal.SIGTERM, _signal_exit)
    if args.serve:
        code = serve_only(args, out)
    else:
        premises = ALL_PREMISES if args.premise == "all" else [args.premise]
        corpus = Corpus(REAL_BASE / "models" / "Jundot" / MODEL, out)
        code = 0
        for p in premises:
            rc = asyncio.run(run_premise_async(args, args.variant, p, out, corpus))
            if rc == 3:
                return 3
            code = max(code, rc)
    after = hashlib.sha256(real_settings.read_bytes()).hexdigest()
    print(
        f"[check] real ~/.omlx/settings.json unchanged: {before == after}", flush=True
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
