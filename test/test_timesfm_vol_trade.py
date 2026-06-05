"""Torch-free tests for volatility-direct trading eval (src/timesfm_vol_trade.py).

Run: PYTHONPATH=. .venv/bin/python3.12 test/test_timesfm_vol_trade.py
"""
import math
import unittest

import numpy as np

from src.timesfm_vol_trade import (
    _economic_ic,
    _fake_pred,
    _implied_proxy,
    _trade_pnl,
    evaluate_strategy,
    run,
)

DATA_DIR = "data/split/EURUSD/weekly/train"
MANIFEST = "data/split/EURUSD/weekly/manifest.json"


class TestVolTradeMath(unittest.TestCase):
    def test_implied_proxy_scales_ewma(self):
        ewma = np.array([1.0, 2.0, 3.0])
        got = _implied_proxy(ewma, vrp_mult=1.25)
        np.testing.assert_allclose(got, np.array([1.25, 2.5, 3.75]))

    def test_economic_ic_is_perfect_when_surprises_align(self):
        implied = np.array([10.0, 10.0, 10.0, 10.0])
        pred = np.array([11.0, 8.0, 12.0, 7.0])
        # Pearson corr of (pred-implied, actual-implied) should be exactly 1
        # when actual surprises are an affine transform of predicted surprises.
        actual = np.array([12.0, 6.0, 14.0, 4.0])
        self.assertAlmostEqual(_economic_ic(pred, actual, implied), 1.0, places=9)

    def test_trade_pnl_accounts_for_side_and_cost(self):
        pred = np.array([12.0, 8.0, 10.0])
        actual = np.array([13.0, 7.0, 12.0])
        implied = np.array([10.0, 10.0, 10.0])
        pnl = _trade_pnl(pred, actual, implied, cost=0.5, threshold=0.0)
        # long vol: +(13-10)-0.5 = 2.5
        # short vol: +(10-7)-0.5 = 2.5
        # flat edge:  sign(0) => no trade, no cost
        np.testing.assert_allclose(pnl, np.array([2.5, 2.5, 0.0]))

    def test_evaluate_strategy_matches_carry_when_always_below_implied(self):
        pred = np.array([9.0, 9.0, 9.0, 9.0])
        actual = np.array([8.0, 12.0, 7.0, 11.0])
        implied = np.array([10.0, 10.0, 10.0, 10.0])
        s = evaluate_strategy("model", pred, actual, implied, cost=0.25)
        carry = (implied - actual) - 0.25
        np.testing.assert_allclose(s["pnl"], carry)
        self.assertEqual(s["n_trades"], 4)
        self.assertAlmostEqual(s["trade_rate"], 1.0, places=12)

    def test_fake_pred_uses_last_48_mean(self):
        ctx = np.arange(100, dtype=float).reshape(2, 50)
        got = _fake_pred(ctx)
        want = ctx[:, -48:].mean(axis=1) * 1.05
        np.testing.assert_allclose(got, want)


class TestVolTradeRun(unittest.TestCase):
    def test_run_fake_backend_returns_expected_structure(self):
        rep = run(
            data_dir=DATA_DIR,
            manifest_path=MANIFEST,
            split="test",
            backend="fake",
            context_len=128,
            horizon=48,
            window=48,
            vrp_mults=(1.0, 1.1),
            cost_points=(0.0, 1e-5),
            max_files=2,
        )
        self.assertIn("config", rep)
        self.assertIn("scenarios", rep)
        self.assertEqual(rep["config"]["backend"], "fake")
        self.assertGreater(rep["config"]["n_obs"], 0)
        self.assertEqual(
            set(rep["scenarios"].keys()),
            {
                "vrp1.00_cost0.000000",
                "vrp1.00_cost0.000010",
                "vrp1.10_cost0.000000",
                "vrp1.10_cost0.000010",
            },
        )
        one = rep["scenarios"]["vrp1.00_cost0.000000"]
        for name in ("carry", "ewma", "persistence", "timesfm_zeroshot"):
            self.assertIn(name, one)
            self.assertIn("economic_ic", one[name])
            self.assertIn("mean_pnl", one[name])
            self.assertIn("n_trades", one[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
