"""Torch-free tests for src/timesfm_indicator_eval.py.

Run: PYTHONPATH=. .venv/bin/python3.12 test/test_timesfm_indicator_eval.py
"""
import unittest

import numpy as np

from src.timesfm_indicator_eval import (
    SERIES_TYPES,
    _compute_series,
    _fake_forecast,
    _score_direction,
    _signal_from_forecast,
    build_indicator_windows,
)

import pandas as pd

DATA_DIR = "data/split/EURUSD/weekly/train"


class TestComputeSeries(unittest.TestCase):
    def _df(self, n=400):
        rng = np.random.default_rng(0)
        close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
        return pd.DataFrame({
            "close": close,
            "log_return_1": np.concatenate([[0.0], np.diff(np.log(close))]),
            "sma_288": pd.Series(close).rolling(288, min_periods=1).mean(),
            "close_vs_sma_288": close / pd.Series(close).rolling(288, min_periods=1).mean() - 1,
        })

    def test_all_series_present_and_finite(self):
        s = _compute_series(self._df())
        self.assertEqual(set(s.keys()), set(SERIES_TYPES.keys()))
        for k, v in s.items():
            self.assertTrue(np.isfinite(v).all(), f"{k} has non-finite values")

    def test_ema_is_smoother_than_close(self):
        s = _compute_series(self._df())
        # EMA must have smaller bar-to-bar variation than raw close
        self.assertLess(np.abs(np.diff(s["ema_48"])).mean(),
                        np.abs(np.diff(s["close"])).mean())
        # longer span = smoother
        self.assertLess(np.abs(np.diff(s["ema_96"])).mean(),
                        np.abs(np.diff(s["ema_12"])).mean())

    def test_macd_is_ema12_minus_ema48(self):
        s = _compute_series(self._df())
        np.testing.assert_allclose(s["macd_12_48"], s["ema_12"] - s["ema_48"], atol=1e-12)


class TestSignalDerivation(unittest.TestCase):
    def test_level_signal_is_forecast_slope(self):
        pred = np.array([[1.0, 2.0, 3.0], [5.0, 4.0, 3.0]])  # rising, falling
        last = np.array([1.5, 4.5])
        sig = _signal_from_forecast("ema_48", pred, last)  # level type
        self.assertGreater(sig[0], 0)   # 3.0 - 1.5 > 0  rising
        self.assertLess(sig[1], 0)      # 3.0 - 4.5 < 0  falling

    def test_return_signal_is_sum(self):
        pred = np.array([[1.0, -0.5, 0.2], [-1.0, -1.0, 0.5]])
        sig = _signal_from_forecast("log_return_1", pred, np.zeros(2))
        self.assertAlmostEqual(sig[0], 0.7)
        self.assertAlmostEqual(sig[1], -1.5)

    def test_oscillator_signal_is_final_level(self):
        pred = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.2]])
        sig = _signal_from_forecast("macd_12_48", pred, np.zeros(2))
        self.assertAlmostEqual(sig[0], 0.3)
        self.assertAlmostEqual(sig[1], -0.2)


class TestScoreDirection(unittest.TestCase):
    def test_perfect_signal_scores_one(self):
        actual = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
        signal = actual.copy()
        r = _score_direction("perfect", signal, actual)
        self.assertEqual(r["dir_acc"], 1.0)
        self.assertEqual(r["n"], 5)

    def test_inverted_signal_scores_zero(self):
        actual = np.array([1.0, -2.0, 3.0, -1.0])
        r = _score_direction("inverted", -actual, actual)
        self.assertEqual(r["dir_acc"], 0.0)

    def test_zero_signals_excluded(self):
        actual = np.array([1.0, -1.0, 1.0, -1.0])
        signal = np.array([0.0, 0.0, 1.0, -1.0])  # first two have no opinion
        r = _score_direction("partial", signal, actual)
        self.assertEqual(r["n"], 2)


class TestFakeForecastAndWindows(unittest.TestCase):
    def test_fake_forecast_is_persistence(self):
        ctx = np.array([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]])
        f = _fake_forecast(ctx, 4)
        self.assertEqual(f.shape, (2, 4))
        np.testing.assert_array_equal(f[0], [3.0, 3.0, 3.0, 3.0])
        np.testing.assert_array_equal(f[1], [7.0, 7.0, 7.0, 7.0])

    def test_build_windows_alignment(self):
        d = build_indicator_windows(DATA_DIR, context_len=128, horizon=48,
                                    stride=192, max_files=3)
        n = len(d["actual"])
        self.assertGreater(n, 0)
        for k in SERIES_TYPES:
            self.assertEqual(d["ctx"][k].shape, (n, 128))
            self.assertEqual(len(d["last"][k]), n)
            self.assertTrue(np.isfinite(d["ctx"][k]).all())
        self.assertEqual(len(d["momentum"]), n)
        self.assertTrue(np.isfinite(d["actual"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
