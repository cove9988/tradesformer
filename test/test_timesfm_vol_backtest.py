"""Torch-free tests for src/timesfm_vol_backtest.py.

Run: PYTHONPATH=. .venv/bin/python3.12 test/test_timesfm_vol_backtest.py
"""
import unittest

import numpy as np

from src.timesfm_vol_backtest import (
    ENTRY_RULES,
    backtest_positions,
    build_backtest_windows,
    make_vol_target_size,
)

DATA_DIR = "data/split/EURUSD/weekly/train"


class TestSizing(unittest.TestCase):
    def test_constant_forecast_gives_unit_size(self):
        s = make_vol_target_size(np.array([2.0, 2.0, 2.0, 2.0]))
        np.testing.assert_allclose(s, np.ones(4), atol=1e-12)

    def test_lower_forecast_vol_gets_larger_size(self):
        fv = np.array([1.0, 2.0, 4.0])
        s = make_vol_target_size(fv, max_leverage=10.0)
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[1], s[2])
        self.assertAlmostEqual(s.mean(), 1.0, places=9)

    def test_size_is_clipped(self):
        fv = np.array([1e-6, 1.0, 1.0])
        s = make_vol_target_size(fv, max_leverage=2.0)
        self.assertLessEqual(s.max(), 2.0)
        self.assertGreaterEqual(s.min(), 0.0)


class TestBacktestAccounting(unittest.TestCase):
    def test_zero_direction_means_zero_pnl(self):
        r = backtest_positions(np.zeros(5), np.ones(5), np.array([1, -1, 1, -1, 1]) * 1e-3, horizon=48, spread_pips=0.5)
        self.assertEqual(r["total_logret"], 0.0)
        self.assertEqual(r["participation_rate"], 0.0)
        self.assertEqual(r["avg_gross_leverage"], 0.0)

    def test_profitable_series_has_positive_total_return(self):
        direction = np.ones(4)
        size = np.ones(4)
        actual = np.array([2e-3, 1e-3, 1.5e-3, 0.5e-3])
        r = backtest_positions(direction, size, actual, horizon=48, spread_pips=0.0)
        self.assertGreater(r["total_logret"], 0.0)
        self.assertEqual(r["hit_rate"], 1.0)

    def test_spread_reduces_return(self):
        direction = np.ones(4)
        size = np.ones(4)
        actual = np.array([2e-3, 1e-3, 1.5e-3, 0.5e-3])
        gross = backtest_positions(direction, size, actual, horizon=48, spread_pips=0.0)
        net = backtest_positions(direction, size, actual, horizon=48, spread_pips=1.0)
        self.assertLess(net["total_logret"], gross["total_logret"])


class TestWindowBuilder(unittest.TestCase):
    def test_build_backtest_windows_shapes_align(self):
        d = build_backtest_windows(DATA_DIR, context_len=128, horizon=48, window=48, stride=192, max_files=3)
        n = len(d["actual_cumret"])
        self.assertGreater(n, 0)
        self.assertEqual(d["rv_ctx"].shape, (n, 128))
        for k in ("persistence", "ewma", "climatology"):
            self.assertEqual(len(d[k]), n)
            self.assertTrue(np.isfinite(d[k]).all())
            self.assertTrue(np.all(d[k] > 0))
        for k in ENTRY_RULES:
            self.assertEqual(len(d["entries"][k]), n)
            self.assertTrue(np.isfinite(d["entries"][k]).all())
        self.assertTrue(np.isfinite(d["actual_cumret"]).all())

    def test_default_stride_is_horizon(self):
        d = build_backtest_windows(DATA_DIR, context_len=128, horizon=48, window=48, stride=None, max_files=1)
        self.assertEqual(d["stride"], 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
