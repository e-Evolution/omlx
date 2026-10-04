# SPDX-License-Identifier: Apache-2.0
"""Rate-limited SSD eviction summary (A5)."""

import logging
from pathlib import Path

import pytest

from omlx.cache.eviction_log import EvictionSummary


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_first_event_emits_immediately():
    clock = _Clock()
    s = EvictionSummary(interval_s=60.0, clock=clock)
    line = s.record(blocks=1, nbytes=100)
    assert line == (
        "SSD cache evicted: blocks=1 bytes=100 sidecars=0 window_s=0.0 "
        "total_blocks=1 total_bytes=100 total_sidecars=0"
    )


def test_events_within_interval_are_batched_then_summed():
    clock = _Clock()
    s = EvictionSummary(interval_s=60.0, clock=clock)
    s.record(blocks=1, nbytes=100)
    for _ in range(5):
        clock.now += 10
        assert s.record(blocks=2, nbytes=50) is None
    clock.now += 10  # 60 s since the first line
    line = s.record(blocks=1, nbytes=10, sidecars=1)
    assert line == (
        "SSD cache evicted: blocks=11 bytes=260 sidecars=1 window_s=60.0 "
        "total_blocks=12 total_bytes=360 total_sidecars=1"
    )


def test_at_most_one_line_per_interval():
    clock = _Clock()
    s = EvictionSummary(interval_s=60.0, clock=clock)
    lines = []
    for _ in range(300):
        clock.now += 1
        line = s.record(blocks=1, nbytes=1)
        if line:
            lines.append(line)
    assert len(lines) <= 300 // 60 + 1
    assert s.total_blocks == 300


def test_window_resets_after_emit():
    clock = _Clock()
    s = EvictionSummary(interval_s=60.0, clock=clock)
    s.record(blocks=3, nbytes=30)
    clock.now += 61
    line = s.record(blocks=1, nbytes=5)
    assert "blocks=1 bytes=5" in line and "total_blocks=4" in line


def test_flush_returns_pending_remainder_once():
    clock = _Clock()
    s = EvictionSummary(interval_s=60.0, clock=clock)
    s.record(blocks=1, nbytes=1)
    s.record(blocks=2, nbytes=2)
    assert "blocks=2 bytes=2" in s.flush()
    assert s.flush() is None


def test_line_is_parseable_key_value():
    s = EvictionSummary(interval_s=60.0, clock=_Clock())
    line = s.record(blocks=4, nbytes=4096, sidecars=2)
    fields = dict(p.split("=") for p in line.split(": ", 1)[1].split())
    assert fields["blocks"] == "4" and fields["sidecars"] == "2"
    assert int(fields["bytes"]) == 4096


class _Meta:
    def __init__(self, path, size):
        self.file_path = path
        self.file_size = size


def test_manager_unlink_logs_info_summary_and_debug_event(tmp_path, caplog):
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    mgr = PagedSSDCacheManager(cache_dir=tmp_path, max_size_bytes=10**9)
    try:
        f = tmp_path / "blk.safetensors"
        f.write_bytes(b"x" * 10)
        with caplog.at_level(logging.DEBUG, logger="omlx.cache.paged_ssd_cache"):
            mgr._unlink_evicted(_Meta(f, 10))
        msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
        assert any(
            lvl == logging.INFO and m.startswith("SSD cache evicted: blocks=1 bytes=10")
            for lvl, m in msgs
        )
        assert any(lvl == logging.DEBUG and "Evicted SSD block" in m for lvl, m in msgs)
        assert not Path(f).exists()
    finally:
        mgr.close()


def test_manager_close_flushes_pending_summary(tmp_path, caplog):
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    mgr = PagedSSDCacheManager(cache_dir=tmp_path, max_size_bytes=10**9)
    mgr._eviction_summary = EvictionSummary(interval_s=3600.0, clock=_Clock())
    for i in range(3):
        f = tmp_path / f"b{i}.safetensors"
        f.write_bytes(b"x" * 10)
        mgr._unlink_evicted(_Meta(f, 10))  # 1st emits, 2nd/3rd stay pending
    with caplog.at_level(logging.INFO, logger="omlx.cache.paged_ssd_cache"):
        mgr.close()
    finals = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("SSD cache evicted: blocks=2 bytes=20")
    ]
    assert len(finals) == 1
    assert "total_blocks=3" in finals[0]
