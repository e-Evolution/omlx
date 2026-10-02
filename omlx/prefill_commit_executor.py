# SPDX-License-Identifier: Apache-2.0
"""Single daemon-thread executor for incremental prefill commits.

Kept out of ``scheduler.py`` so the scheduler module has exactly one
``shutdown`` definition (``Scheduler.shutdown``).
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from collections.abc import Callable
from typing import Any


class DaemonCommitExecutor:
    """Single daemon worker thread with the ``submit``/``shutdown`` surface.

    ``ThreadPoolExecutor`` workers are non-daemon and joined at interpreter
    exit, so a commit stuck in a store call would hang process exit. This
    worker is a daemon: an abandoned commit cannot block exit, and shutdown
    only waits for it through the cancel flag and the store lock.
    """

    def __init__(self, thread_name_prefix: str = "omlx-prefill-commit") -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name=f"{thread_name_prefix}_0", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 - delivered via the future
                future.set_exception(e)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = False) -> None:
        self._queue.put(None)
        if wait:
            self._thread.join()
