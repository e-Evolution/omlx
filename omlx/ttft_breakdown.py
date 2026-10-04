# SPDX-License-Identifier: Apache-2.0
"""Per-request time-to-first-token breakdown, logged as one INFO line.

Stages (all from timestamps the scheduler already takes, monotonic seconds):

- ``hold_s``: arrival until the prefix-cache restore began (queue + gate wait)
- ``restore_s``: prefix-cache lookup and reconstruction
- ``prefill_s``: end of restore until the request entered decode
  (wall time; includes decode steps interleaved with chunked prefill)
- ``first_decode_s``: entering decode until the first response
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class TtftTracker:
    def __init__(self, max_entries: int = 1024) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, float, float, str]] = (
            OrderedDict()
        )

    def mark_restore(
        self,
        request_id: str,
        arrival: float,
        restore_start: float,
        restore_end: float,
        outcome: str,
    ) -> None:
        with self._lock:
            self._entries[request_id] = (arrival, restore_start, restore_end, outcome)
            self._entries.move_to_end(request_id)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def discard(self, request_id: str) -> None:
        with self._lock:
            self._entries.pop(request_id, None)

    def first_token_line(
        self,
        request_id: str,
        generation_started_at: float,
        now: float,
        prompt_tokens: int,
        cached_tokens: int,
    ) -> str | None:
        """Pop the request's entry and format its line (once per request)."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
        if entry is None:
            return None
        arrival, restore_start, restore_end, outcome = entry
        hold = max(0.0, restore_start - arrival)
        restore = max(0.0, restore_end - restore_start)
        prefill = max(0.0, generation_started_at - restore_end)
        first_decode = max(0.0, now - generation_started_at)
        ttft = hold + restore + prefill + first_decode
        return (
            f"TTFT breakdown: request={request_id} prompt_tokens={prompt_tokens} "
            f"cached_tokens={cached_tokens} cache={outcome} "
            f"hold_s={hold:.3f} restore_s={restore:.3f} prefill_s={prefill:.3f} "
            f"first_decode_s={first_decode:.3f} ttft_s={ttft:.3f}"
        )
