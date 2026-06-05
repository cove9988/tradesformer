"""Torch-free tests for multi-pair volatility fine-tune data pipeline.

The Iron Law: these are written BEFORE src/timesfm_vol_finetune_multipair.py and
must fail (RED) until the loader exists, then pass (GREEN). They validate the
leakage-critical behaviour: pooling rv series across pairs with a per-pair
CHRONOLOGICAL val holdout (last N weeks of each pair -> early-stopping only),
so no forward-period week can ever enter training.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_vol_finetune_multipair import (
    list_pair_weeks,
    chrono_train_val_split,
    load_multipair_vol_series,
)


def _write_week(path: Path, n_rows: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 3e-4, size=n_rows).astype(np.float64)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "log_return_1": r})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestChronoSplit(unittest.TestCase):
    def test_last_n_go_to_val(self):
        files = [f"PAIRA_2022_{w:02d}.csv" for w in range(13, 23)]  # 10 files
        tr, va = chrono_train_val_split(files, val_weeks=2)
        self.assertEqual(len(tr), 8)
        self.assertEqual(len(va), 2)
        # val must be the chronologically LAST weeks
        self.assertEqual(va, files[-2:])
        self.assertEqual(tr, files[:-2])

    def test_no_overlap(self):
        files = [f"f{i}" for i in range(20)]
        tr, va = chrono_train_val_split(files, val_weeks=5)
        self.assertEqual(set(tr) & set(va), set())
        self.assertEqual(len(tr) + len(va), 20)

    def test_rejects_bad_val_weeks(self):
        files = [f"f{i}" for i in range(5)]
        with self.assertRaises(ValueError):
            chrono_train_val_split(files, val_weeks=0)
        with self.assertRaises(ValueError):
            chrono_train_val_split(files, val_weeks=5)


class TestMultipairLoader(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pairs = ["PAIRA", "PAIRB"]
        # 6 weeks per pair, chronological filenames
        for pi, pair in enumerate(self.pairs):
            for wi, w in enumerate(range(13, 19)):
                _write_week(self.root / pair / "weekly" / "train" / f"{pair}_2022_{w:02d}.csv",
                            n_rows=200, seed=pi * 100 + wi)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_list_pair_weeks_sorted(self):
        files = list_pair_weeks("PAIRA", self.root)
        self.assertEqual(len(files), 6)
        names = [Path(f).name for f in files]
        self.assertEqual(names, sorted(names))  # chronological

    def test_pools_across_pairs_with_chrono_val(self):
        train, val, meta = load_multipair_vol_series(
            self.pairs, self.root, window=48, val_weeks_per_pair=2)
        # 2 pairs * (6-2) train weeks = 8 ; 2 pairs * 2 val weeks = 4
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 4)
        # every series is a non-empty float32 vol array
        for s in train + val:
            self.assertEqual(s.dtype, np.float32)
            self.assertGreater(len(s), 0)
        # meta records per-pair chronological holdout (last 2 weeks -> val)
        for pair in self.pairs:
            self.assertEqual(meta[pair]["n_train"], 4)
            self.assertEqual(meta[pair]["n_val"], 2)
            self.assertEqual(meta[pair]["val_range"][1], f"{pair}_2022_18")  # last week
            self.assertEqual(meta[pair]["train_range"][0], f"{pair}_2022_13")  # first week

    def test_missing_pair_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_multipair_vol_series(["NOPE"], self.root, window=48, val_weeks_per_pair=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
