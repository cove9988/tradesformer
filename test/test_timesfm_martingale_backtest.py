"""Tests for the martingale-family backtest simulator on controlled paths."""
import math
import unittest

import numpy as np

from src.timesfm_martingale_backtest import (
    simulate_week, summarize, _skew, run, MODES,
)


def _const_vol(n, v=2e-4):
    return np.full(n, v)


class TestMartingaleMechanics(unittest.TestCase):
    def test_doubling_against_trend_hits_ruin(self):
        # A relentless adverse trend must force the add-cap stop -> ruin event,
        # and the loss must be large (geometric lots all underwater).
        n = 600
        # price marches steadily UP; fade entry will be short -> always wrong
        close = 1.0 + np.linspace(0, 0.02, n)        # +200 pips over the week
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        cyc, ru = simulate_week(close, logret, _const_vol(n), mode="martingale",
                                step_pips=20, target_pips=20, max_adds=4,
                                spread_pips=0.5, lookback=10)
        self.assertTrue(any(ru), "expected at least one ruin event on a pure trend")
        # worst cycle must be a large loss (much bigger than one base target)
        self.assertLess(min(cyc), -20 * 1e-4, f"ruin loss too small: {min(cyc)}")

    def test_lots_double_geometrically(self):
        # Instrument the leg construction indirectly: on a monotone adverse path
        # with max_adds=k, the realized loss magnitude should scale ~ (2^k) target.
        n = 800
        close = 1.0 + np.linspace(0, 0.05, n)
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        loss2 = min(simulate_week(close, logret, _const_vol(n), mode="martingale",
                                  step_pips=20, target_pips=20, max_adds=2,
                                  spread_pips=0.0, lookback=10)[0])
        loss5 = min(simulate_week(close, logret, _const_vol(n), mode="martingale",
                                  step_pips=20, target_pips=20, max_adds=5,
                                  spread_pips=0.0, lookback=10)[0])
        # deeper cap -> much bigger blow-up loss
        self.assertLess(loss5, loss2 * 2.0, f"loss5={loss5} not >> loss2={loss2}")

    def test_profit_target_closes_small_win(self):
        # Mean-reverting saw-tooth: price dips then returns -> fade should win small
        n = 400
        base = np.concatenate([np.linspace(0, -0.003, 50), np.linspace(-0.003, 0.001, 50)])
        path = np.tile(base, n // len(base) + 1)[:n]
        close = 1.0 + path
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        cyc, ru = simulate_week(close, logret, _const_vol(n), mode="martingale",
                                step_pips=10, target_pips=10, max_adds=6,
                                spread_pips=0.1, lookback=10)
        self.assertGreater(len(cyc), 0)
        # at least some cycles should close positive on the reversion
        self.assertTrue(any(p > 0 for p in cyc))

    def test_vol_gate_reduces_cycle_count(self):
        # Bimodal vol: calm first half, stormy second half. A median gate should
        # block cycle-starts in the stormy half -> meaningfully fewer cycles.
        # Use a fast mean-reverting saw + tiny target so many short cycles form.
        n = 1440
        base = np.concatenate([np.linspace(0, -0.0008, 20), np.linspace(-0.0008, 0.0004, 20)])
        path = np.tile(base, n // len(base) + 1)[:n]
        close = 1.0 + path
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        tv = np.empty(n)
        tv[: n // 2] = 1e-4          # calm
        tv[n // 2 :] = 5e-4          # stormy
        ungated = simulate_week(close, logret, tv, mode="martingale", vol_gate=None,
                                step_pips=4, target_pips=4, max_adds=6, spread_pips=0.0,
                                lookback=10)[0]
        gated = simulate_week(close, logret, tv, mode="martingale", vol_gate=50.0,
                              step_pips=4, target_pips=4, max_adds=6, spread_pips=0.0,
                              lookback=10)[0]
        self.assertLess(len(gated), len(ungated),
                        f"gate did not cut cycles: gated={len(gated)} ungated={len(ungated)}")


class TestSummarize(unittest.TestCase):
    def test_skew_negative_for_martingale_shape(self):
        # many small wins, one huge loss -> strongly negative skew
        cycles = [0.001] * 99 + [-0.5]
        ruins = [False] * 99 + [True]
        s = summarize(cycles, ruins)
        self.assertLess(s["skew"], -3.0)
        self.assertEqual(s["ruin_events"], 1)
        self.assertLess(s["worst_cycle"], -0.4)

    def test_empty(self):
        self.assertEqual(summarize([], [])["n_cycles"], 0)

    def test_skew_helper_symmetric_near_zero(self):
        x = list(np.random.default_rng(1).normal(0, 1, 5000))
        self.assertLess(abs(_skew(x)), 0.2)


class TestRunSmoke(unittest.TestCase):
    def test_run_on_synthetic_dir(self):
        import tempfile, os, pandas as pd
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        for wk in range(3):
            lr = rng.normal(0, 2e-4, 1440)
            close = 1.1 * np.exp(np.cumsum(lr))
            pd.DataFrame({"close": close, "log_return_1": lr}).to_csv(
                os.path.join(d, f"EURUSD_2099_{wk:02d}.csv"), index=False)
        rep = run(data_dir=d, manifest_path=None, split="all",
                  spread_pips=0.5, max_files=None)
        self.assertEqual(rep["config"]["n_weeks"], 3)
        for label in ("flat_baseline", "martingale_naive", "anti_martingale"):
            self.assertIn(label, rep["variants"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
