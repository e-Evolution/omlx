# SPDX-License-Identifier: Apache-2.0
"""Opt-in debug switches used by the P3 numerics investigation."""

import json

import mlx.core as mx

from omlx import debug_numerics as dn


def _lp(values):
    return mx.log(mx.softmax(mx.array(values, dtype=mx.float32)))


class TestLogprobDump:
    def test_disabled_without_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv(dn.ENV_LOGPROB_DUMP, raising=False)
        assert dn.dump_enabled() is False
        dn.record_logprobs("r1", 10, 3, _lp([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert list(tmp_path.iterdir()) == []

    def test_writes_chosen_and_top_k(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dn.ENV_LOGPROB_DUMP, str(tmp_path))
        lp = _lp([0.0, 5.0, 1.0, 4.0, 2.0, 3.0, -1.0])
        dn.record_logprobs("req-a", 123, 3, lp, top_k=3)
        dn.record_logprobs("req-a", 123, 1, lp, top_k=3)
        lines = (tmp_path / "req-a.jsonl").read_text().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["pt"] == 123 and rec["t"] == 3
        assert [i for i, _ in rec["top"]] == [1, 3, 5]  # sorted by logprob desc
        assert abs(rec["lp"] - float(lp[3])) < 1e-5
        assert abs(rec["top"][0][1] - float(lp[1])) < 1e-5

    def test_unsafe_request_id_stays_inside_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dn.ENV_LOGPROB_DUMP, str(tmp_path))
        dn.record_logprobs("../evil/x", 1, 0, _lp([1.0, 2.0, 3.0]), top_k=2)
        files = list(tmp_path.iterdir())
        assert len(files) == 1 and files[0].parent == tmp_path

    def test_none_logprobs_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dn.ENV_LOGPROB_DUMP, str(tmp_path))
        dn.record_logprobs("r", 1, 0, None)
        assert list(tmp_path.iterdir()) == []


class TestForceRaggedB1:
    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(dn.ENV_FORCE_RAGGED_B1, raising=False)
        assert dn.force_ragged_b1() is False

    def test_follows_flag_file(self, monkeypatch, tmp_path):
        flag = tmp_path / "ragged.flag"
        monkeypatch.setenv(dn.ENV_FORCE_RAGGED_B1, str(flag))
        dn._flag_cache.clear()
        assert dn.force_ragged_b1() is False
        flag.write_text("1")
        dn._flag_cache.clear()
        assert dn.force_ragged_b1() is True
