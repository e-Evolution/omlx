# SPDX-License-Identifier: Apache-2.0
"""Per-request prefix-cache restore outcome classification and counters."""

from __future__ import annotations

import threading

OUTCOMES = (
    "hit_tail",
    "block_aligned_divergence",
    "cold",
    "media_keyed",
    "restore_short",
)


def classify_restore_outcome(
    *,
    cached: int,
    common_prefix: int | None,
    stored_len: int | None,
    block: int,
    has_media_keys: bool,
) -> str:
    """Classify why a request restored ``cached`` tokens from the prefix cache.

    ``common_prefix``/``stored_len`` describe the closest recently stored
    sequence (None when no probe is available).

    - ``media_keyed``: nothing restored and the request's cache keys include
      media, so a changed or new image/video yields no match.
    - ``cold``: nothing restored and no stored sequence shares a full block.
    - ``restore_short``: a stored sequence shares at least one more block than
      was served (blocks evicted or not restorable).
    - ``block_aligned_divergence``: the prompt diverges inside the stored
      sequence; the restore stops at the last full block before it.
    - ``hit_tail``: the prompt extends the stored sequence and everything up to
      its (block-aligned) tail was served.
    """
    block = max(1, block)
    if cached <= 0 and has_media_keys:
        return "media_keyed"
    shared = common_prefix if common_prefix is not None else 0
    if cached <= 0 and shared < block:
        return "cold"
    if common_prefix is not None and shared - cached >= block:
        return "restore_short"
    if common_prefix is not None and stored_len is not None and shared < stored_len:
        return "block_aligned_divergence"
    return "hit_tail"


class RestoreOutcomeCounter:
    """Running per-outcome request counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(OUTCOMES, 0)

    def record(self, outcome: str) -> None:
        if outcome not in self._counts:
            raise ValueError(f"unknown restore outcome: {outcome}")
        with self._lock:
            self._counts[outcome] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)
