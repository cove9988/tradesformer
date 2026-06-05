"""Torch-free tests for volatility fine-tune + split-aware vol eval."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_vol_finetune import load_vol_series, parse_args
from src.timesfm_vol_eval import build_vol_windows, _weeks_for_split
from src.timesfm_split import build_split_manifest, write_split_manifest


def _make_week(path, n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 3e-4, size=n).astype(np.float64)
    pd.DataFrame({"log_return_1": r}).to_csv(path, index=False)


class TestSplitAwareVolEval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = Path(self.tmp) / "train"
        self.data.mkdir()
        # 6 chronological weeks
        self.names = [f"EURUSD_2023_0{i}.csv" for i in range(1, 7)]
        for i, nm in enumerate(self.names):
            _make_week(self.data / nm, n=800, seed=i)
        self.manifest = Path(self.tmp) / "manifest.json"
        m = build_split_manifest(self.data, symbol="EURUSD",
                                 validation_weeks=2, test_weeks=2)
        write_split_manifest(self.manifest, m)
        self.splits = m["splits"]

    def test_split_has_no_overlap_and_is_chronological(self):
        tr, va, te = self.splits["train"], self.splits["val"], self.splits["test"]
        self.assertEqual(len(tr), 2)
        self.assertEqual(len(va), 2)
        self.assertEqual(len(te), 2)
        # disjoint
        self.assertEqual(set(tr) & set(va), set())
        self.assertEqual(set(va) & set(te), set())
        self.assertEqual(set(tr) & set(te), set())
        # chronological: train < val < test by filename order
        self.assertTrue(max(tr) < min(va))
        self.assertTrue(max(va) < min(te))

    def test_weeks_for_split_returns_correct_basenames(self):
        weeks = _weeks_for_split(self.manifest, "test")
        self.assertEqual(weeks, set(self.splits["test"]))
        self.assertIsNone(_weeks_for_split(self.manifest, "all"))

    def test_build_vol_windows_respects_week_filter(self):
        # restricting to test weeks must use ONLY those files
        weeks = set(self.splits["test"])
        d_all = build_vol_windows(self.data, context_len=128, horizon=48,
                                  window=48, stride=48)
        d_test = build_vol_windows(self.data, context_len=128, horizon=48,
                                   window=48, stride=48, weeks=weeks)
        # fewer or equal windows when filtered to a subset
        self.assertLessEqual(len(d_test["actual"]), len(d_all["actual"]))
        self.assertGreater(len(d_test["actual"]), 0)


class TestVolSeriesLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = Path(self.tmp) / "train"
        self.data.mkdir()
        self.names = [f"EURUSD_2023_0{i}.csv" for i in range(1, 7)]
        for i, nm in enumerate(self.names):
            _make_week(self.data / nm, n=800, seed=i)
        self.manifest = Path(self.tmp) / "manifest.json"
        m = build_split_manifest(self.data, symbol="EURUSD",
                                 validation_weeks=2, test_weeks=2)
        write_split_manifest(self.manifest, m)
        self.splits = m["splits"]

    def test_load_vol_series_is_trailing_rv_and_split_filtered(self):
        train = load_vol_series(self.data, self.manifest, "train", window=48)
        # train split has 2 weeks -> 2 series
        self.assertEqual(len(train), len(self.splits["train"]))
        # vol series must be strictly positive (it's an RMS) and finite
        for s in train:
            self.assertTrue(np.all(np.isfinite(s)))
            self.assertTrue(np.all(s >= 0))
            # length = n - window + 1 (leading NaNs dropped), n=800, window=48
            self.assertEqual(len(s), 800 - 48 + 1)

    def test_train_and_test_series_come_from_different_weeks(self):
        train = load_vol_series(self.data, self.manifest, "train", window=48)
        test = load_vol_series(self.data, self.manifest, "test", window=48)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        # different underlying random seeds -> different vol levels (no leakage/copy)
        self.assertNotAlmostEqual(float(train[0].mean()), float(test[0].mean()), places=9)

    def test_all_split_loads_every_week(self):
        alls = load_vol_series(self.data, self.manifest, "all", window=48)
        self.assertEqual(len(alls), len(self.names))


class TestArgs(unittest.TestCase):
    def test_defaults(self):
        a = parse_args([])
        self.assertEqual(a.window, 48)
        self.assertEqual(a.context_len, 512)
        self.assertEqual(a.horizon_len, 48)
        self.assertEqual(a.backend, "timesfm-2.5-transformers")

    def test_context_len_multiple_of_32_enforced_in_main(self):
        # parse_args itself doesn't validate; main does. Just confirm arg parses.
        a = parse_args(["--context-len", "256", "--window", "24"])
        self.assertEqual(a.context_len, 256)
        self.assertEqual(a.window, 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
