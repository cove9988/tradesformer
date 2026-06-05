"""Torch-free tests for the volatility-forecast eval (src/timesfm_vol_eval.py).

Run: PYTHONPATH=. .venv/bin/python3.12 test/test_timesfm_vol_eval.py
"""
import math
import unittest

import numpy as np

from src.timesfm_vol_eval import (
    _forward_rv,
    _qlike,
    _score,
    _spearman,
    _trailing_rv,
    build_vol_windows,
)

DATA_DIR = "data/split/EURUSD/weekly/train"


class TestVolMath(unittest.TestCase):
    def test_trailing_rv_matches_manual(self):
        r = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        rv = _trailing_rv(r, 2)
        # index 1: rms(0,1)=sqrt(0.5); index 4: rms(2,-2)=2
        self.assertTrue(math.isnan(rv[0]))
        self.assertAlmostEqual(rv[1], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(rv[4], 2.0, places=12)

    def test_forward_rv_matches_manual(self):
        r = np.array([1.0, -1.0, 2.0, -2.0])
        fwd = _forward_rv(r, 2)
        # t=0: rms(1,-1)=1 ; t=2: rms(2,-2)=2 ; t=3: NaN (not enough future)
        self.assertAlmostEqual(fwd[0], 1.0, places=12)
        self.assertAlmostEqual(fwd[2], 2.0, places=12)
        self.assertTrue(math.isnan(fwd[3]))

    def test_trailing_rv_nonnegative(self):
        rng = np.random.default_rng(0)
        rv = _trailing_rv(rng.normal(size=200) * 1e-4, 48)
        self.assertTrue(np.all(rv[~np.isnan(rv)] >= 0.0))


class TestMetrics(unittest.TestCase):
    def test_spearman_perfect_monotone(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a ** 3  # strictly increasing -> rank corr 1
        self.assertAlmostEqual(_spearman(a, b), 1.0, places=9)

    def test_spearman_perfect_inverse(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(_spearman(a, -a), -1.0, places=9)

    def test_qlike_zero_for_perfect_forecast(self):
        actual = np.array([1e-3, 2e-3, 3e-3, 4e-3])
        self.assertAlmostEqual(_qlike(actual, actual.copy()), 0.0, places=9)

    def test_qlike_penalises_bias(self):
        actual = np.full(100, 2e-4)
        good = actual.copy()
        bad = actual * 2.0
        self.assertGreater(_qlike(actual, bad), _qlike(actual, good))

    def test_score_keys_and_calibration(self):
        rng = np.random.default_rng(1)
        actual = np.abs(rng.normal(size=500)) * 2e-4 + 1e-4
        pred = actual * 3.0  # heavy positive level bias, perfect ranking
        s = _score("biased", pred, actual)
        for k in ("spearman_ic", "pearson_r", "rmse", "mae",
                  "qlike_raw", "cal_scale", "qlike_cal"):
            self.assertIn(k, s)
        # perfect ranking despite bias
        self.assertGreater(s["spearman_ic"], 0.99)
        # recalibration scale ~1/3 recovers the level
        self.assertAlmostEqual(s["cal_scale"], 1 / 3, places=1)
        # calibrated QLIKE strictly better than raw under pure level bias
        self.assertLess(s["qlike_cal"], s["qlike_raw"])


class TestWindowBuilder(unittest.TestCase):
    def test_build_vol_windows_shapes_align(self):
        d = build_vol_windows(DATA_DIR, context_len=128, horizon=48,
                              window=48, stride=96, max_files=3)
        n = len(d["actual"])
        self.assertGreater(n, 0)
        self.assertEqual(d["rv_ctx"].shape, (n, 128))
        for k in ("persistence", "ewma", "clim"):
            self.assertEqual(len(d[k]), n)
        # all predictor inputs + targets finite and positive
        for k in ("persistence", "ewma", "clim", "actual"):
            self.assertTrue(np.all(np.isfinite(d[k])))
            self.assertTrue(np.all(d[k] > 0))
        self.assertTrue(np.all(np.isfinite(d["rv_ctx"])))

    def test_context_is_trailing_vol_not_returns(self):
        # trailing vol context must be strictly positive (returns would be signed)
        d = build_vol_windows(DATA_DIR, context_len=128, horizon=48,
                              window=48, stride=192, max_files=2)
        self.assertGreater(len(d["actual"]), 0)
        self.assertTrue(np.all(d["rv_ctx"] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
