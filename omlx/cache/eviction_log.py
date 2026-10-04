# SPDX-License-Identifier: Apache-2.0
"""Rate-limited summary of SSD cache evictions.

Inline eviction runs on every save once the cache is full, so a per-event INFO
line would flood the log. ``EvictionSummary`` accumulates events and returns at
most one parseable summary line per interval.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class EvictionSummary:
    """Accumulates eviction events; ``record`` returns a line when one is due."""

    def __init__(
        self,
        interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval_s = interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._last_emit: float | None = None
        self._window = [0, 0, 0]  # blocks, bytes, sidecars since last line
        self._total = [0, 0, 0]

    @property
    def total_blocks(self) -> int:
        return self._total[0]

    def record(self, blocks: int = 0, nbytes: int = 0, sidecars: int = 0) -> str | None:
        """Account an eviction; return a summary line when the interval elapsed."""
        with self._lock:
            for acc in (self._window, self._total):
                acc[0] += blocks
                acc[1] += nbytes
                acc[2] += sidecars
            now = self._clock()
            if self._last_emit is not None and now - self._last_emit < self._interval_s:
                return None
            return self._format_locked(now)

    def flush(self) -> str | None:
        """Return the pending window as a line (e.g. at shutdown), if any."""
        with self._lock:
            if not any(self._window):
                return None
            return self._format_locked(self._clock())

    def _format_locked(self, now: float) -> str:
        window_s = 0.0 if self._last_emit is None else now - self._last_emit
        blocks, nbytes, sidecars = self._window
        line = (
            f"SSD cache evicted: blocks={blocks} bytes={nbytes} "
            f"sidecars={sidecars} window_s={window_s:.1f} "
            f"total_blocks={self._total[0]} total_bytes={self._total[1]} "
            f"total_sidecars={self._total[2]}"
        )
        self._window = [0, 0, 0]
        self._last_emit = now
        return line
