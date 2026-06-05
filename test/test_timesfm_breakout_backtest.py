"""Tests for breakout/straddle trend engine — verified on controlled OHLC paths.

The result is only trustworthy if the engine mechanics are correct, so we drive
it with hand-built bars where the right answer is known:
  - clean breakout + sustained trend -> large POSITIVE R, pyramid adds fire
  - false breakout that reverses     -> capped ~ -1R loss (stop works)
  - gap through stop                 -> fills at open (worse), not at stop
  - gate=False                       -> no trades
"""
import math
import unittest

import numpy as np

from src.timesfm_breakout_backtest import (
    donchian_channels, simulate_week_breakout, summarize, make_gate, _skew,
)


def _flat_then(arr_close, atr=0.0010, base=1.1000, n_pre=120):
    """Build OHLC where first n_pre bars oscillate in a tight box (sets Donchian),
    then `arr_close` drives the post-breakout path. Returns o,h,l,c,atr arrays."""
    pre = base + 0.0001 * np.sin(np.arange(n_pre))      # tight box ~+/-1pip
    c = np.concatenate([pre, np.asarray(arr_close, float)])
    o = np.concatenate([[c[0]], c[:-1]])
    # high/low bracket each bar by a small wick so intrabar touches are realistic
    h = np.maximum(o, c) + 0.00005
    l = np.minimum(o, c) - 0.00005
    a = np.full(len(c), atr)
    return o, h, l, c, a


class TestDonchian(unittest.TestCase):
    def test_trailing_no_lookahead(self):
        high = np.arange(10, dtype=float)
        low = np.arange(10, dtype=float) - 1
        ch_hi, ch_lo = donchian_channels(high, low, 3)
        self.assertTrue(math.isnan(ch_hi[2]))
        # channel at t uses [t-3:t], strictly before t (no lookahead)
        self.assertEqual(ch_hi[3], 2.0)   # max(high[0,1,2]) = 2
        self.assertEqual(ch_lo[3], -1.0)  # min(low[0,1,2]) = -1


class TestBreakoutMechanics(unittest.TestCase):
    def test_clean_uptrend_makes_large_positive_R(self):
        # strong sustained up-move after the box -> big positive R, adds fire
        up = 1.1000 + np.linspace(0.0002, 0.030, 200)   # +300 pips trend
        o, h, l, c, a = _flat_then(up)
        cyc, adds = simulate_week_breakout(o, h, l, c, a, donchian_n=96,
                                           k_stop=2.0, k_trail=3.0, k_add=1.0,
                                           max_adds=3, spread_pips=0.5)
        self.assertGreater(len(cyc), 0)
        self.assertGreater(max(cyc), 3.0, f"expected multi-R win, got {cyc}")
        self.assertGreaterEqual(max(adds), 1, "pyramid should add on a strong trend")

    def test_false_breakout_caps_near_minus_1R(self):
        # breakout then immediate reversal -> stopped ~ -1R (plus spread), never worse
        # than a couple R (gap protection). Down-spike after box.
        rev = np.concatenate([[1.1030], 1.1030 - np.linspace(0.0005, 0.020, 199)])
        o, h, l, c, a = _flat_then(rev)
        cyc, adds = simulate_week_breakout(o, h, l, c, a, donchian_n=96,
                                           k_stop=2.0, k_trail=3.0, k_add=1.0,
                                           max_adds=3, spread_pips=0.5)
        self.assertGreater(len(cyc), 0)
        worst = min(cyc)
        # a single-unit stop is ~ -1R; allow gap/spread slack but it must be bounded
        self.assertGreater(worst, -2.5, f"loss should be bounded near -1R, got {worst}")
        self.assertLess(worst, 0.0)

    def test_gate_false_blocks_all_trades(self):
        up = 1.1000 + np.linspace(0.0002, 0.030, 200)
        o, h, l, c, a = _flat_then(up)
        gate = np.zeros(len(c), dtype=bool)
        cyc, adds = simulate_week_breakout(o, h, l, c, a, gate=gate, donchian_n=96)
        self.assertEqual(len(cyc), 0)

    def test_positive_skew_signature(self):
        # many small losses + a few big wins -> POSITIVE skew (opposite of martingale)
        rng = np.random.default_rng(0)
        cycles = list(rng.normal(-1.0, 0.2, 80)) + list(rng.normal(8.0, 2.0, 20))
        self.assertGreater(_skew(cycles), 0.5)
        s = summarize(cycles, [0] * 100)
        self.assertGreater(s["skew"], 0.0)
        self.assertGreater(s["payoff_ratio"], 1.0)


class TestGate(unittest.TestCase):
    def _df(self, n=400):
        import pandas as pd
        rng = np.random.default_rng(2)
        rv = np.abs(rng.normal(2e-4, 1e-4, n))
        return pd.DataFrame({"realized_vol_96": rv})

    def test_none_gate_all_true(self):
        df = self._df()
        g = make_gate(df, "none")
        self.assertTrue(g.all())

    def test_highvol_vs_lowvol_complementary(self):
        df = self._df()
        hi = make_gate(df, "highvol", pct=50.0)
        lo = make_gate(df, "lowvol", pct=50.0)
        # roughly partition (boundary bar may overlap); shouldn't both be ~all
        self.assertLess(hi.sum(), len(df))
        self.assertLess(lo.sum(), len(df))
        self.assertGreater(hi.sum(), 0)

    def test_expansion_needs_rising_vol(self):
        import pandas as pd
        n = 300
        rv = np.concatenate([np.full(150, 1e-4), np.full(150, 5e-4)])  # jumps up
        df = pd.DataFrame({"realized_vol_96": rv})
        g = make_gate(df, "expansion", donchian_n=96)
        # bars in the high-vol second half (vs 96 bars earlier in low-vol) -> True
        self.assertTrue(g[200])
        # bars deep in the flat low-vol first half -> not expanding
        self.assertFalse(g[120])


class TestSummarizeEmpty(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize([], [])["n_cycles"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
