# SPDX-License-Identifier: Apache-2.0
"""Cache-miss reason classification and TTFT breakdown (A5)."""

import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.cache.restore_outcome import (
    OUTCOMES,
    RestoreOutcomeCounter,
    classify_restore_outcome,
)
from omlx.scheduler import Scheduler
from omlx.ttft_breakdown import TtftTracker

BLOCK = 2048


def classify(**kw):
    base = dict(
        cached=0, common_prefix=None, stored_len=None, block=BLOCK, has_media_keys=False
    )
    base.update(kw)
    return classify_restore_outcome(**base)


class TestClassify:
    def test_cold_without_any_probe(self):
        assert classify() == "cold"

    def test_cold_when_shared_prefix_below_one_block(self):
        assert classify(common_prefix=100, stored_len=9000) == "cold"

    def test_media_keyed_when_nothing_restored_and_media_keys_present(self):
        assert classify(has_media_keys=True) == "media_keyed"
        assert (
            classify(common_prefix=9000, stored_len=9000, has_media_keys=True)
            == "media_keyed"
        )

    def test_hit_tail_when_prompt_extends_stored_sequence(self):
        assert (
            classify(cached=8192, common_prefix=9000, stored_len=9000) == "hit_tail"
        )

    def test_hit_tail_without_probe_when_something_restored(self):
        assert classify(cached=4096) == "hit_tail"

    def test_block_aligned_divergence_inside_stored_sequence(self):
        # Diverges at 8000 of 10000; restore stops at the last full block.
        assert (
            classify(cached=6144, common_prefix=8000, stored_len=10000)
            == "block_aligned_divergence"
        )

    def test_restore_short_when_shared_prefix_exceeds_served(self):
        # Shared 9000 tokens but only 2048 served: blocks lost or not restored.
        assert (
            classify(cached=2048, common_prefix=9000, stored_len=9000)
            == "restore_short"
        )
        assert classify(cached=0, common_prefix=9000, stored_len=9000) == "restore_short"

    def test_outcome_names_are_stable(self):
        assert OUTCOMES == (
            "hit_tail",
            "block_aligned_divergence",
            "cold",
            "media_keyed",
            "restore_short",
        )


class TestCounter:
    def test_counts_per_outcome_with_zero_defaults(self):
        c = RestoreOutcomeCounter()
        c.record("cold")
        c.record("cold")
        c.record("hit_tail")
        snap = c.snapshot()
        assert snap["cold"] == 2 and snap["hit_tail"] == 1
        assert snap["media_keyed"] == 0
        assert set(snap) == set(OUTCOMES)

    def test_unknown_outcome_is_rejected(self):
        with pytest.raises(ValueError):
            RestoreOutcomeCounter().record("nope")

    def test_snapshot_is_a_copy(self):
        c = RestoreOutcomeCounter()
        c.snapshot()["cold"] = 99
        assert c.snapshot()["cold"] == 0


class TestTtftTracker:
    def test_breakdown_line_format_and_values(self):
        t = TtftTracker()
        t.mark_restore(
            "r1", arrival=100.0, restore_start=102.0, restore_end=102.5, outcome="cold"
        )
        line = t.first_token_line(
            "r1",
            generation_started_at=112.5,
            now=112.75,
            prompt_tokens=30000,
            cached_tokens=0,
        )
        assert line == (
            "TTFT breakdown: request=r1 prompt_tokens=30000 cached_tokens=0 "
            "cache=cold hold_s=2.000 restore_s=0.500 prefill_s=10.000 "
            "first_decode_s=0.250 ttft_s=12.750"
        )

    def test_line_is_parseable_and_sums_to_ttft(self):
        t = TtftTracker()
        t.mark_restore("r", 0.0, 1.0, 3.0, "hit_tail")
        line = t.first_token_line("r", 8.0, 8.5, 10, 5)
        f = dict(p.split("=") for p in line.split(": ", 1)[1].split())
        total = sum(float(f[k]) for k in ("hold_s", "restore_s", "prefill_s", "first_decode_s"))
        assert total == pytest.approx(float(f["ttft_s"]))

    def test_one_line_per_request(self):
        t = TtftTracker()
        t.mark_restore("r", 0.0, 0.0, 0.1, "cold")
        assert t.first_token_line("r", 1.0, 1.1, 1, 0) is not None
        assert t.first_token_line("r", 1.0, 2.0, 1, 0) is None

    def test_unknown_request_gives_no_line(self):
        assert TtftTracker().first_token_line("x", 1.0, 1.1, 1, 0) is None

    def test_negative_gaps_clamp_to_zero(self):
        t = TtftTracker()
        t.mark_restore("r", 5.0, 4.0, 3.0, "cold")
        line = t.first_token_line("r", 2.0, 1.0, 1, 0)
        assert "-" not in line.split(": ", 1)[1].replace("request=r", "")

    def test_bounded_memory(self):
        t = TtftTracker(max_entries=3)
        for i in range(10):
            t.mark_restore(f"r{i}", 0.0, 0.0, 0.0, "cold")
        assert t.first_token_line("r0", 1.0, 1.0, 1, 0) is None
        assert t.first_token_line("r9", 1.0, 1.0, 1, 0) is not None

    def test_discard(self):
        t = TtftTracker()
        t.mark_restore("r", 0.0, 0.0, 0.0, "cold")
        t.discard("r")
        assert t.first_token_line("r", 1.0, 1.0, 1, 0) is None


def _sched(block=BLOCK):
    s = object.__new__(Scheduler)
    s._cache_probe_seqs = deque(maxlen=4)
    s.tokenizer = MagicMock()
    s.config = MagicMock(spec=[])
    s.config.paged_cache_block_size = block
    return s


def _req(prompt, cached=0, rid="new", keys=None):
    return SimpleNamespace(
        request_id=rid,
        prompt_token_ids=prompt,
        cached_tokens=cached,
        vlm_extra_keys_for_cache=keys,
    )


class TestSchedulerIntegration:
    def test_divergence_updates_counter_and_lookup(self):
        s = _sched()
        s._cache_probe_seqs.append(("old", list(range(10000))))
        prompt = list(range(8000)) + [999999] * 4143
        s._log_prefix_divergence(_req(prompt, cached=6144))
        s.waiting, s.prefilling, s.running = deque(), deque(), {}
        s.num_requests_processed = s.total_prompt_tokens = 0
        s.total_completion_tokens = 0
        s.block_aware_cache = None
        s._packed_prefill_stats = {}
        assert s.get_stats()["cache_outcomes"]["block_aligned_divergence"] == 1
        assert s._last_prefix_cache_lookup["outcome"] == "block_aligned_divergence"

    def test_cold_and_media_counted_without_probe(self):
        s = _sched()
        s._log_prefix_divergence(_req([1, 2, 3]))
        s._log_prefix_divergence(_req([1, 2, 3], keys=("img",)))
        counts = s._restore_outcomes.snapshot()
        assert counts["cold"] == 1 and counts["media_keyed"] == 1

    def test_info_line_gains_outcome_suffix(self, caplog):
        s = _sched()
        s._cache_probe_seqs.append(("old", list(range(10000))))
        prompt = list(range(8000)) + [999999] * 4143
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            s._log_prefix_divergence(_req(prompt, cached=6144))
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert infos[0].endswith("outcome=block_aligned_divergence")

    def test_small_reprefill_still_silent(self, caplog):
        s = _sched()
        s._cache_probe_seqs.append(("old", list(range(9000))))
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            s._log_prefix_divergence(_req(list(range(9000)) + [7] * 100, cached=7200))
        assert caplog.text == ""


class TestSchedulerTtftHook:
    def test_logs_one_line_at_first_response_only(self, caplog):
        s = _sched()
        req = SimpleNamespace(
            request_id="r1",
            arrival_time=10.0,
            num_prompt_tokens=500,
            cached_tokens=100,
            generation_started_at=14.0,
        )
        s._get_ttft_tracker().mark_restore("r1", 10.0, 11.0, 11.5, "hit_tail")
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            s._log_ttft_breakdown(req, 14.25)
            s._log_ttft_breakdown(req, 15.0)
        lines = [r.getMessage() for r in caplog.records]
        assert lines == [
            "TTFT breakdown: request=r1 prompt_tokens=500 cached_tokens=100 "
            "cache=hit_tail hold_s=1.000 restore_s=0.500 prefill_s=2.500 "
            "first_decode_s=0.250 ttft_s=4.250"
        ]


class _FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestTtftClockDomain:
    """All TTFT stamps share the ``time.monotonic`` domain (the Request
    default, the restore stamps and the step clock)."""

    def test_request_arrival_uses_monotonic_domain(self):
        import time

        from omlx.request import Request

        factory = Request.__dataclass_fields__["arrival_time"].default_factory
        assert factory is time.monotonic

    def test_stages_non_negative_and_sum_to_ttft(self, caplog):
        import re

        from omlx.request import Request, SamplingParams

        clock = _FakeClock(500.0)
        arrival = clock()
        req = Request(
            request_id="r1",
            prompt=[1],
            sampling_params=SamplingParams(),
            arrival_time=arrival,
        )
        req.num_prompt_tokens = 10

        s = _sched()
        tracker = s._get_ttft_tracker()
        clock.now += 0.7  # queue + gate hold
        restore_start = clock()
        clock.now += 0.2  # restore
        tracker.mark_restore("r1", arrival, restore_start, clock(), "cold")
        clock.now += 1.5  # prefill
        req.generation_started_at = clock()
        clock.now += 0.05  # first decode
        step_now = clock()

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            s._log_ttft_breakdown(req, step_now)
        line = caplog.records[0].getMessage()
        v = {k: float(x) for k, x in re.findall(r"(\w+_s)=([0-9.]+)", line)}
        parts = ("hold_s", "restore_s", "prefill_s", "first_decode_s")
        assert all(v[p] >= 0 for p in parts)
        assert sum(v[p] for p in parts) == pytest.approx(v["ttft_s"], abs=0.005)
        assert v["ttft_s"] == pytest.approx(step_now - arrival, abs=0.005)
