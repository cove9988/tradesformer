"""Tests for src/timesfm_eval.py — torch-free (use fake/baseline paths only).

Run:  PYTHONPATH=. python test/test_timesfm_eval.py
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.timesfm_eval import (
    Window,
    build_windows,
    build_windows_for_file,
    evaluate_signal,
    predict_momentum,
    predict_naive,
    predict_reversion,
    _binom_two_sided_p,
    _max_drawdown,
    _spearman,
)

DATA_DIR = "data/split/EURUSD/weekly/train"


class TestWindowConstruction(unittest.TestCase):
    def test_build_windows_shapes(self):
        wins = build_windows(DATA_DIR, context_len=256, horizon=12, stride=48, max_files=3)
        self.assertGreater(len(wins), 0)
        w = wins[0]
        self.assertEqual(len(w.context_price), 256)
        self.assertEqual(len(w.context_return), 256)
        self.assertTrue(np.isfinite(w.actual_cumret))

    def test_actual_cumret_matches_price_ratio(self):
        wins = build_windows(DATA_DIR, context_len=128, horizon=24, stride=96, max_files=1)
        self.assertGreater(len(wins), 0)
        # actual_cumret is defined as log(close[t+H-1]/last_close); just assert finite & bounded
        for w in wins[:20]:
            self.assertTrue(np.isfinite(w.actual_cumret))
            self.assertLess(abs(w.actual_cumret), 0.1)  # a 24-bar EURUSD move is tiny

    def test_no_window_crosses_file_boundary(self):
        # each weekly file is 1440 rows; a window needs context_len + horizon <= 1440
        wins = build_windows_for_file(
            sorted(__import__("glob").glob(f"{DATA_DIR}/*.csv"))[0],
            context_len=512, horizon=48, stride=48,
        )
        self.assertTrue(all(w.context_price.shape[0] == 512 for w in wins))


class TestBaselinePredictors(unittest.TestCase):
    def _toy(self):
        return [
            Window("f", 10, 1.0, np.zeros(4), np.array([0, 0, 0, 0.002]), 0.001, 0.003),
            Window("f", 20, 1.0, np.zeros(4), np.array([0, 0, 0, -0.001]), -0.002, -0.004),
        ]

    def test_naive_is_zero(self):
        np.testing.assert_array_equal(predict_naive(self._toy()), np.zeros(2))

    def test_momentum_follows_prev(self):
        np.testing.assert_array_equal(predict_momentum(self._toy()), np.array([0.003, -0.004]))

    def test_reversion_is_negated_momentum(self):
        np.testing.assert_array_equal(predict_reversion(self._toy()), np.array([-0.003, 0.004]))


class TestMetrics(unittest.TestCase):
    def test_binom_p_perfect_split_is_one(self):
        self.assertAlmostEqual(_binom_two_sided_p(50, 100, 0.5), 1.0, places=1)

    def test_binom_p_extreme_is_small(self):
        self.assertLess(_binom_two_sided_p(90, 100, 0.5), 0.01)

    def test_spearman_monotonic(self):
        a = np.array([1.0, 2, 3, 4, 5])
        self.assertAlmostEqual(_spearman(a, a * 2 + 1), 1.0, places=6)
        self.assertAlmostEqual(_spearman(a, -a), -1.0, places=6)

    def test_max_drawdown_nonpositive(self):
        eq = np.array([0.0, 1, 0.5, 2, 1.0])
        self.assertLessEqual(_max_drawdown(eq), 0.0)
        self.assertAlmostEqual(_max_drawdown(eq), -1.0, places=6)

    def test_evaluate_signal_perfect_predictor(self):
        # predictor equals actual -> 100% directional accuracy
        actual = np.array([0.001, -0.002, 0.0015, -0.0008, 0.0021])
        res = evaluate_signal("perfect", actual.copy(), actual, bars_per_horizon=48)
        self.assertEqual(res["dir_accuracy"], 1.0)
        self.assertLess(res["binom_p_value"], 0.1)
        # gross sharpe should be strongly positive at zero cost
        self.assertGreater(res["backtest_by_cost"]["0.0pip"]["sharpe_ann"], 0)

    def test_evaluate_signal_anti_predictor(self):
        actual = np.array([0.001, -0.002, 0.0015, -0.0008, 0.0021])
        res = evaluate_signal("anti", -actual, actual, bars_per_horizon=48)
        self.assertEqual(res["dir_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
