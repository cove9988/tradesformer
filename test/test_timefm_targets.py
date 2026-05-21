import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.timefm_dataset import build_timefm_dataset_grid, build_timefm_samples_for_split
from src.timefm_split import build_split_manifest, write_split_manifest
from src.timefm_targets import summarize_future_targets


class TimeFMTargetsTest(unittest.TestCase):
    def test_summarize_future_targets_derives_sum_direction_and_abs_move(self):
        summary = summarize_future_targets([0.01, -0.02, 0.03], positive_threshold=0.01, negative_threshold=-0.01)

        self.assertAlmostEqual(summary["future_return_sum"], 0.02)
        self.assertEqual(summary["future_direction"], "up")
        self.assertAlmostEqual(summary["future_abs_move"], 0.06)
        self.assertEqual(summary["future_direction_class"], 1)

    def test_summarize_future_targets_marks_flat_when_between_thresholds(self):
        summary = summarize_future_targets([0.001, -0.001], positive_threshold=0.01, negative_threshold=-0.01)

        self.assertEqual(summary["future_direction"], "flat")
        self.assertEqual(summary["future_direction_class"], 0)


class TimeFMDatasetSplitLoadingTest(unittest.TestCase):
    def _frame(self, bars=8, symbol="EURUSD"):
        start = pd.Timestamp("2024-01-01 00:00:00+00:00")
        times = [start + pd.Timedelta(minutes=5 * i) for i in range(bars)]
        close = [1.1000 + 0.0001 * i for i in range(bars)]
        return pd.DataFrame(
            {
                "time": times,
                "symbol": [symbol] * bars,
                "open": [value - 0.00005 for value in close],
                "high": [value + 0.00015 for value in close],
                "low": [value - 0.00015 for value in close],
                "close": close,
                "log_return_1": [0.0] + [0.001 * i for i in range(1, bars)],
                "close_rel_week_open": [i * 10.0 for i in range(bars)],
                "hour_sin": [0.15 * i for i in range(bars)],
            }
        )

    def test_build_timefm_samples_for_split_only_reads_requested_manifest_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "train"
            data_dir.mkdir()

            for week in range(1, 6):
                frame = self._frame()
                frame["time"] = pd.date_range(f"2024-01-{week:02d}", periods=8, freq="5min", tz="UTC")
                frame.to_csv(data_dir / f"EURUSD_2024_{week:02d}.csv", index=False)

            manifest = build_split_manifest(data_dir, symbol="EURUSD", validation_weeks=1, test_weeks=1)
            manifest_path = base / "manifest.json"
            write_split_manifest(manifest_path, manifest)

            samples = build_timefm_samples_for_split(
                data_directory=data_dir,
                manifest_path=manifest_path,
                split_name="val",
                context_window=3,
                forecast_horizon=2,
                target_col="log_return_1",
                feature_cols=["close_rel_week_open", "hour_sin"],
            )

            self.assertEqual(len(samples), 4)
            self.assertTrue(all(sample.source_file.endswith("EURUSD_2024_04.csv") for sample in samples))
            self.assertEqual(samples[0].iso_year, 2024)
            self.assertEqual(samples[0].iso_week, 4)
            self.assertAlmostEqual(samples[0].derived_targets["future_return_sum"], 0.007)

    def test_dataset_grid_sorts_single_digit_and_double_digit_weeks_chronologically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            week_2 = self._frame()
            week_10 = self._frame()
            week_2["time"] = pd.date_range("2024-01-08", periods=8, freq="5min", tz="UTC")
            week_10["time"] = pd.date_range("2024-03-04", periods=8, freq="5min", tz="UTC")
            week_10.to_csv(data_dir / "EURUSD_2024_10.csv", index=False)
            week_2.to_csv(data_dir / "EURUSD_2024_2.csv", index=False)

            grid = build_timefm_dataset_grid(
                data_directory=data_dir,
                context_windows=[3],
                forecast_horizons=[2],
                target_col="log_return_1",
                feature_cols=["close_rel_week_open", "hour_sin"],
            )

            self.assertEqual(grid[(3, 2)][0].iso_week, 2)
            self.assertEqual(grid[(3, 2)][-1].iso_week, 10)


if __name__ == "__main__":
    unittest.main()
