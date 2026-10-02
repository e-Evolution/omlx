# SPDX-License-Identifier: Apache-2.0
"""Opt-in debug switches for numerics investigations (never on by default).

``OMLX_DEBUG_LOGPROB_DUMP=<dir>``
    Every emitted token appends ``{"pt", "t", "lp", "top"}`` (prompt tokens,
    token id, its log-probability, top-k ``[id, logprob]`` pairs) to
    ``<dir>/<request_id>.jsonl``. The chat API does not expose logprobs, so
    batching-vs-single-row comparisons read this file.

``OMLX_DEBUG_FORCE_RAGGED_B1=<flag file>``
    While the flag file exists, single-row TurboQuant batches go through the
    ragged multi-row kernel used for B>1, to separate "kernel numerics" from
    "batching/padding" effects. The file is polled (not the env) so a harness
    can toggle it between requests without restarting the server.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

ENV_LOGPROB_DUMP = "OMLX_DEBUG_LOGPROB_DUMP"
ENV_FORCE_RAGGED_B1 = "OMLX_DEBUG_FORCE_RAGGED_B1"

_FLAG_POLL_S = 0.1
_flag_cache: dict[str, tuple[float, bool]] = {}


def dump_enabled() -> bool:
    return bool(os.environ.get(ENV_LOGPROB_DUMP))


def record_logprobs(
    request_id: str,
    num_prompt_tokens: int,
    token: int,
    logprobs: Any,
    top_k: int = 5,
) -> None:
    """Append one token record; no-op unless the dump env var is set."""
    directory = os.environ.get(ENV_LOGPROB_DUMP)
    if not directory or logprobs is None:
        return
    import mlx.core as mx
    import numpy as np

    row = np.asarray(mx.array(logprobs).astype(mx.float32)).reshape(-1)
    k = min(top_k, row.shape[0])
    idx = np.argpartition(-row, k - 1)[:k]
    idx = idx[np.argsort(-row[idx], kind="stable")]
    record = {
        "pt": int(num_prompt_tokens),
        "t": int(token),
        "lp": float(row[int(token)]),
        "top": [[int(i), float(row[i])] for i in idx],
    }
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(request_id)).lstrip(".") or "req"
    path = Path(directory) / f"{safe}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def force_ragged_b1() -> bool:
    """True while the ragged-B1 flag file exists (cheap, polled)."""
    flag = os.environ.get(ENV_FORCE_RAGGED_B1)
    if not flag:
        return False
    now = time.monotonic()
    cached = _flag_cache.get(flag)
    if cached is not None and now - cached[0] < _FLAG_POLL_S:
        return cached[1]
    value = os.path.exists(flag)
    _flag_cache[flag] = (now, value)
    return value
