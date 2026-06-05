"""Tests for Stage-2 TimesFM-gated breakout plumbing (torch-free, fake backend)."""
import unittest

import numpy as np
import pandas as pd

from src.timesfm_breakout_vol_gate import (
    find_breakout_bars, collect_contexts, build_model_gate, _fake_pred, run,
)


def _box_then_trend(n_pre=200, n_post=300, base=1.10, atr=0.0010):
    pre = base + 0.0001 * np.sin(np.arange(n_pre))
    post = base + np.linspace(0.0002, 0.030, n_post)
    close = np.concatenate([pre, post])
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + 5e-5
    l = np.minimum(o, close) - 5e-5
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "atr": np.full(len(close), atr), "log_return_1": r,
                         "realized_vol_48": pd.Series(r).rolling(48, min_periods=48).std()})


class TestBreakoutBars(unittest.TestCase):
    def test_finds_breakouts_after_box(self):
        df = _box_then_trend()
        bars, ch_hi, ch_lo = find_breakout_bars(df["high"].to_numpy(),
                                                df["low"].to_numpy(), donchian_n=96)
        self.assertGreater(len(bars), 0)
        self.assertTrue(all(t >= 96 for t in bars))


class TestContexts(unittest.TestCase):
    def test_contexts_are_causal_and_shaped(self):
        df = _box_then_trend()
        ctx, index, rv_prev = collect_contexts([df], context_len=128, window=48, donchian_n=96)
        if len(index):
            self.assertEqual(ctx.shape[1], 128)
            self.assertEqual(len(rv_prev), len(index))
            self.assertTrue(np.all(np.isfinite(ctx)))
            # rv_prev must be finite (causal trailing vol known at entry)
            self.assertTrue(np.all(np.isfinite(rv_prev)))


class TestModelGate(unittest.TestCase):
    def test_expansion_gate_true_when_pred_exceeds_rvprev(self):
        df = _box_then_trend()
        paths = [df]
        # two fake breakout bars in week 0
        index = [(0, 250), (0, 300)]
        pred = np.array([0.002, 0.0005])   # first > rv_prev, second < rv_prev
        rv_prev = np.array([0.001, 0.001])
        gates = build_model_gate(paths, index, pred, rv_prev, kind="expansion")
        self.assertTrue(gates[0][250])     # pred 0.002 > 0.001 -> allow
        self.assertFalse(gates[0][300])    # pred 0.0005 < 0.001 -> block

    def test_highvol_gate_uses_causal_median(self):
        paths = [_box_then_trend()]
        index = [(0, 210), (0, 220), (0, 230), (0, 240)]
        pred = np.array([1.0, 2.0, 3.0, 4.0])    # rising -> later bars beat running median
        rv_prev = np.zeros(4)
        gates = build_model_gate(paths, index, pred, rv_prev, kind="highvol")
        # first bar has no prior -> thr=inf -> False; last bar above median -> True
        self.assertFalse(gates[0][210])
        self.assertTrue(gates[0][240])


class TestFakeRunEndToEnd(unittest.TestCase):
    def test_run_with_fake_backend_produces_all_comparators(self):
        df = _box_then_trend()
        import src.timesfm_breakout_vol_gate as m
        orig = m.__dict__["run"].__globals__["build_paths"] if False else None
        # monkeypatch build_paths via the imported symbol
        import src.timesfm_breakout_backtest as bb
        saved = bb.build_paths
        bb.build_paths = lambda *a, **k: [df, df]
        try:
            rep = run("x", "y", "all", backend="fake", context_len=128, horizon=48,
                      window=48, donchian_n=96, spread_pips=0.5, cache_path=None)
        finally:
            bb.build_paths = saved
        self.assertIn("none", rep["results"])
        self.assertIn("naive_expansion", rep["results"])
        self.assertTrue(any("fake_model" in k for k in rep["results"]))

    def test_fake_pred_shape(self):
        ctx = np.random.rand(5, 128).astype(np.float32)
        p = _fake_pred(ctx)
        self.assertEqual(p.shape, (5,))
        self.assertTrue(np.all(p > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
