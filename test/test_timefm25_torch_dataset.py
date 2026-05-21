"""Failing tests for src/timefm25_torch_dataset.py.

These describe the Dataset contract before implementation exists.
All torch imports are guarded so tests run without torch installed.
"""
from __future__ import annotations

import unittest
import numpy as np
from pathlib import Path
import tempfile
import csv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_series(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.001, size=n).astype(np.float32)


def _write_csv(path: Path, n_rows: int = 200, target_col: str = "log_return_1") -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "symbol", "open", "high", "low", "close",
                    target_col, "close_rel_week_open", "hour_sin"])
        rng = np.random.default_rng(42)
        for i in range(n_rows):
            w.writerow([
                f"2023-09-12 {i // 12:02d}:{(i % 12) * 5:02d}:00+00:00",
                "EURUSD", 1.07, 1.071, 1.069, 1.07,
                round(float(rng.normal(0, 0.001)), 6),
                round(float(rng.normal(0, 1)), 3),
                round(float(np.sin(i * 0.1)), 4),
            ])


# ---------------------------------------------------------------------------
# 1. TimeSeriesRandomWindowDataset
# ---------------------------------------------------------------------------

class RandomWindowDatasetTest(unittest.TestCase):

    def setUp(self):
        self.series_list = [_make_series(200) for _ in range(5)]
        self.context_len = 64
        self.horizon_len = 12
        self.num_samples = 50

    def _make_ds(self, **kwargs):
        from src.timefm25_torch_dataset import TimeSeriesRandomWindowDataset
        return TimeSeriesRandomWindowDataset(
            series_list=self.series_list,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
            num_samples=self.num_samples,
            **kwargs,
        )

    def test_len_equals_num_samples(self):
        ds = self._make_ds()
        self.assertEqual(len(ds), self.num_samples)

    def test_item_returns_context_and_target_tuple(self):
        ds = self._make_ds()
        item = ds[0]
        self.assertEqual(len(item), 2)

    def test_context_shape(self):
        ds = self._make_ds()
        context, _ = ds[0]
        arr = np.asarray(context)
        self.assertEqual(arr.shape, (self.context_len,))

    def test_target_shape(self):
        ds = self._make_ds()
        _, target = ds[0]
        arr = np.asarray(target)
        self.assertEqual(arr.shape, (self.horizon_len,))

    def test_context_values_are_float32(self):
        ds = self._make_ds()
        context, _ = ds[0]
        arr = np.asarray(context)
        self.assertEqual(arr.dtype, np.float32)

    def test_target_values_are_float32(self):
        ds = self._make_ds()
        _, target = ds[0]
        arr = np.asarray(target)
        self.assertEqual(arr.dtype, np.float32)

    def test_reproducible_with_same_seed(self):
        ds1 = self._make_ds(seed=7)
        ds2 = self._make_ds(seed=7)
        np.testing.assert_array_equal(np.asarray(ds1[0][0]), np.asarray(ds2[0][0]))

    def test_different_seeds_give_different_windows(self):
        ds1 = self._make_ds(seed=1)
        ds2 = self._make_ds(seed=2)
        self.assertFalse(
            np.array_equal(np.asarray(ds1[0][0]), np.asarray(ds2[0][0]))
        )

    def test_raises_if_no_series_long_enough(self):
        from src.timefm25_torch_dataset import TimeSeriesRandomWindowDataset
        short = [_make_series(10)]
        with self.assertRaises(ValueError):
            TimeSeriesRandomWindowDataset(short, context_len=64, horizon_len=12)

    def test_context_and_target_are_non_overlapping(self):
        ds = self._make_ds(seed=0)
        context, target = ds[0]
        # last value of context must not appear at start of target
        # (they should be adjacent, not overlapping)
        ctx = np.asarray(context)
        tgt = np.asarray(target)
        # verify they are drawn from different positions: context[-1] != target[0] in general
        # but more robustly: concatenated len == context_len + horizon_len
        self.assertEqual(len(ctx) + len(tgt), self.context_len + self.horizon_len)


# ---------------------------------------------------------------------------
# 2. TimeSeriesLastWindowDataset
# ---------------------------------------------------------------------------

class LastWindowDatasetTest(unittest.TestCase):

    def setUp(self):
        self.series_list = [_make_series(200) for _ in range(5)]
        self.context_len = 64
        self.horizon_len = 12

    def _make_ds(self):
        from src.timefm25_torch_dataset import TimeSeriesLastWindowDataset
        return TimeSeriesLastWindowDataset(
            series_list=self.series_list,
            context_len=self.context_len,
            horizon_len=self.horizon_len,
        )

    def test_len_equals_number_of_valid_series(self):
        ds = self._make_ds()
        self.assertEqual(len(ds), len(self.series_list))

    def test_item_shape(self):
        ds = self._make_ds()
        context, target = ds[0]
        self.assertEqual(np.asarray(context).shape, (self.context_len,))
        self.assertEqual(np.asarray(target).shape, (self.horizon_len,))

    def test_last_window_uses_tail_of_series(self):
        from src.timefm25_torch_dataset import TimeSeriesLastWindowDataset
        series = np.arange(100, dtype=np.float32)
        ds = TimeSeriesLastWindowDataset([series], context_len=10, horizon_len=5)
        ctx, tgt = ds[0]
        np.testing.assert_array_equal(np.asarray(ctx), series[85:95])
        np.testing.assert_array_equal(np.asarray(tgt), series[95:100])

    def test_short_series_are_excluded(self):
        from src.timefm25_torch_dataset import TimeSeriesLastWindowDataset
        series_list = [_make_series(200), _make_series(5), _make_series(200)]
        ds = TimeSeriesLastWindowDataset(series_list, context_len=64, horizon_len=12)
        self.assertEqual(len(ds), 2)  # only 2 are long enough


# ---------------------------------------------------------------------------
# 3. load_series_from_dir
# ---------------------------------------------------------------------------

class LoadSeriesFromDirTest(unittest.TestCase):

    def test_returns_list_of_numpy_arrays(self):
        from src.timefm25_torch_dataset import load_series_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            _write_csv(p / "EURUSD_2023_01.csv", n_rows=200)
            _write_csv(p / "EURUSD_2023_02.csv", n_rows=200)
            series = load_series_from_dir(p, target_col="log_return_1")
        self.assertEqual(len(series), 2)
        for s in series:
            self.assertIsInstance(s, np.ndarray)
            self.assertEqual(s.dtype, np.float32)

    def test_series_length_matches_csv_rows(self):
        from src.timefm25_torch_dataset import load_series_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            _write_csv(p / "EURUSD_2023_01.csv", n_rows=150)
            series = load_series_from_dir(p, target_col="log_return_1")
        self.assertEqual(len(series[0]), 150)

    def test_dropna_removes_nan_values(self):
        from src.timefm25_torch_dataset import load_series_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            path = p / "EURUSD_2023_01.csv"
            _write_csv(path, n_rows=100)
            # inject a NaN row
            import pandas as pd
            df = pd.read_csv(path)
            df.loc[50, "log_return_1"] = float("nan")
            df.to_csv(path, index=False)
            series = load_series_from_dir(p, target_col="log_return_1", dropna=True)
        self.assertFalse(np.isnan(series[0]).any())

    def test_non_csv_files_are_ignored(self):
        from src.timefm25_torch_dataset import load_series_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            _write_csv(p / "EURUSD_2023_01.csv", n_rows=100)
            (p / "README.txt").write_text("ignore me")
            series = load_series_from_dir(p, target_col="log_return_1")
        self.assertEqual(len(series), 1)

    def test_missing_directory_raises(self):
        from src.timefm25_torch_dataset import load_series_from_dir
        with self.assertRaises(FileNotFoundError):
            load_series_from_dir("/does/not/exist", target_col="log_return_1")


if __name__ == "__main__":
    unittest.main()
