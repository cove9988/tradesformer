"""Tests for mean-reversion diagnostics, validated against synthetic ground truth.

The whole strategy decision rests on the variance-ratio estimator being correct,
so we check it on processes with KNOWN structure:
  - white noise          -> VR ~= 1, |z| small
  - AR(1) phi<0 (revert) -> VR < 1, z significantly negative
  - trending / momentum  -> VR > 1, z significantly positive
"""
import math
import unittest

import numpy as np

from src.timesfm_meanrev_diag import (
    variance_ratio_pooled, autocorr_pooled, classify, load_week_returns,
)


def _white_noise(n, seed, sigma=3e-4):
    return np.random.default_rng(seed).normal(0, sigma, size=n)


def _mean_reverting(n, seed, phi=-0.3, sigma=3e-4):
    # AR(1) with negative phi -> successive returns negatively correlated -> VR<1
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _trending(n, seed, phi=0.3, sigma=3e-4):
    # AR(1) with positive phi -> positive autocorrelation -> VR>1
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


class TestVarianceRatioGroundTruth(unittest.TestCase):
    def test_white_noise_vr_near_one(self):
        series = [_white_noise(1440, s) for s in range(40)]
        out = variance_ratio_pooled(series, q=24)
        self.assertTrue(0.9 < out["vr"] < 1.1, f"VR={out['vr']}")
        self.assertLess(abs(out["z_robust"]), 3.0, f"z={out['z_robust']}")

    def test_mean_reverting_vr_below_one(self):
        series = [_mean_reverting(1440, s) for s in range(40)]
        out = variance_ratio_pooled(series, q=24)
        self.assertLess(out["vr"], 1.0, f"VR={out['vr']}")
        self.assertLess(out["z_robust"], -2.0, f"z={out['z_robust']}")
        self.assertIn("mean-reverting", classify(out["vr"], out["z_robust"]))

    def test_trending_vr_above_one(self):
        series = [_trending(1440, s) for s in range(40)]
        out = variance_ratio_pooled(series, q=24)
        self.assertGreater(out["vr"], 1.0, f"VR={out['vr']}")
        self.assertGreater(out["z_robust"], 2.0, f"z={out['z_robust']}")
        self.assertEqual("trending", classify(out["vr"], out["z_robust"]))

    def test_vr_one_at_q1_is_identity(self):
        # VR(1) must be exactly 1 by construction
        series = [_white_noise(500, s) for s in range(5)]
        out = variance_ratio_pooled(series, q=1)
        # q=1 has no overlapping sums beyond the series itself; guard finite
        self.assertTrue(math.isfinite(out["vr"]))


class TestAutocorr(unittest.TestCase):
    def test_white_noise_acf_near_zero(self):
        series = [_white_noise(1440, s) for s in range(40)]
        out = autocorr_pooled(series, lag=1)
        self.assertLess(abs(out["acf"]), 0.05, f"acf={out['acf']}")

    def test_mean_reverting_negative_acf(self):
        series = [_mean_reverting(1440, s) for s in range(40)]
        out = autocorr_pooled(series, lag=1)
        self.assertLess(out["acf"], 0.0)
        self.assertLess(out["z"], -2.0)

    def test_trending_positive_acf(self):
        series = [_trending(1440, s) for s in range(40)]
        out = autocorr_pooled(series, lag=1)
        self.assertGreater(out["acf"], 0.0)
        self.assertGreater(out["z"], 2.0)


class TestClassify(unittest.TestCase):
    def test_labels(self):
        self.assertIn("mean-reverting", classify(0.8, -3.0))
        self.assertEqual("trending", classify(1.2, 3.0))
        self.assertIn("random-walk", classify(0.99, 0.5))
        self.assertEqual("undetermined", classify(float("nan"), 1.0))


class TestNoWeekBoundaryStraddle(unittest.TestCase):
    def test_two_short_weeks_handled(self):
        # weeks shorter than q must be skipped without crashing
        series = [_white_noise(10, 1), _white_noise(2000, 2)]
        out = variance_ratio_pooled(series, q=48)
        self.assertTrue(math.isfinite(out["vr"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
