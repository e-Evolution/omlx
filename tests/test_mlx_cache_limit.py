# SPDX-License-Identifier: Apache-2.0
"""Bounded MLX buffer cache: setting, env override, startup hook, real MLX."""

import logging
import os
from unittest.mock import patch

import mlx.core as mx
import pytest

import omlx.process_memory_enforcer as pme
from omlx.settings import GlobalSettings, MemorySettings

GB = 1024**3


class TestSetting:
    def test_default_is_unbounded(self):
        assert MemorySettings().mlx_cache_limit_gb == 0.0

    def test_round_trip(self):
        d = MemorySettings(mlx_cache_limit_gb=16.0).to_dict()
        assert d["mlx_cache_limit_gb"] == 16.0
        assert MemorySettings.from_dict(d).mlx_cache_limit_gb == 16.0

    def test_from_dict_missing_key_is_zero(self):
        assert MemorySettings.from_dict({}).mlx_cache_limit_gb == 0.0

    @pytest.mark.parametrize("bad", [True, False, "x", None])
    def test_from_dict_ignores_non_numeric_and_bool(self, bad):
        got = MemorySettings.from_dict({"mlx_cache_limit_gb": bad})
        assert got.mlx_cache_limit_gb == 0.0

    def test_validate_rejects_negative(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        s.memory.mlx_cache_limit_gb = -1.0
        assert any("mlx_cache_limit_gb" in e for e in s.validate())

    def test_validate_rejects_bool(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        s.memory.mlx_cache_limit_gb = True  # type: ignore[assignment]
        assert any("mlx_cache_limit_gb" in e for e in s.validate())

    def test_validate_accepts_zero_and_positive(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        for v in (0.0, 16.0):
            s.memory.mlx_cache_limit_gb = v
            assert not any("mlx_cache_limit_gb" in e for e in s.validate())

    def test_env_override(self, tmp_path):
        with patch.dict(os.environ, {"OMLX_MLX_CACHE_LIMIT_GB": "12.5"}):
            s = GlobalSettings.load(base_path=tmp_path)
        assert s.memory.mlx_cache_limit_gb == 12.5

    def test_env_invalid_is_ignored(self, tmp_path):
        with patch.dict(os.environ, {"OMLX_MLX_CACHE_LIMIT_GB": "lots"}):
            s = GlobalSettings.load(base_path=tmp_path)
        assert s.memory.mlx_cache_limit_gb == 0.0


class TestStartupHook:
    def test_sets_limit_in_bytes_and_logs(self, caplog):
        with patch.object(pme.mx, "set_cache_limit", return_value=64 * GB) as m:
            with caplog.at_level(logging.INFO, logger=pme.logger.name):
                applied = pme.apply_mlx_cache_limit(16.0)
        m.assert_called_once_with(16 * GB)
        assert applied == 16 * GB
        assert "MLX buffer cache limit set to 16.0 GB (was 64.0 GB)" in caplog.text

    @pytest.mark.parametrize("value", [0, 0.0, -1.0, True])
    def test_noop_when_unset_or_invalid(self, value):
        with patch.object(pme.mx, "set_cache_limit") as m:
            assert pme.apply_mlx_cache_limit(value) == 0
        m.assert_not_called()

    def test_failure_is_swallowed(self):
        with patch.object(pme.mx, "set_cache_limit", side_effect=RuntimeError("x")):
            assert pme.apply_mlx_cache_limit(4.0) == 0


class TestRealMlx:
    def test_cache_stays_within_limit(self):
        previous = mx.set_cache_limit(0)  # remember the old limit
        try:
            limit = 8 * 1024 * 1024
            assert pme.apply_mlx_cache_limit(limit / GB) == limit
            mx.clear_cache()
            for _ in range(8):
                a = mx.random.normal((1024, 1024))  # 4 MiB each
                mx.eval(a * 2)
                del a
            mx.synchronize()
            # MLX trims on the next allocation, so allow one small slack.
            assert mx.get_cache_memory() <= limit + 1024 * 1024
            assert mx.get_cache_memory() < 8 * 4 * 1024 * 1024  # not unbounded
        finally:
            mx.set_cache_limit(previous)
            mx.clear_cache()
