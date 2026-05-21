import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.timesfm_features import TIMESFM_OBSERVATION_COLUMNS
from src.timesfm_dataset import (
    TimeFMSample,
    build_timesfm_dataset_grid,
    build_timesfm_samples,
    load_timesfm_weekly_csv,
)


class TimeFMDatasetAdapterTest(unittest.TestCase):
    def _frame(self, bars=8, symbol="EURUSD"):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        times = [start + timedelta(minutes=5 * i) for i in range(bars)]
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
                "high_rel_week_open": [i * 10.0 + 2.0 for i in range(bars)],
                "low_rel_week_open": [i * 10.0 - 2.0 for i in range(bars)],
                "open_rel_week_open": [i * 10.0 - 1.0 for i in range(bars)],
                "returns_12": [0.01 * i for i in range(bars)],
                "returns_48": [0.02 * i for i in range(bars)],
                "returns_96": [0.03 * i for i in range(bars)],
                "returns_288": [0.04 * i for i in range(bars)],
                "realized_vol_48": [0.05 * i for i in range(bars)],
                "realized_vol_96": [0.06 * i for i in range(bars)],
                "realized_vol_288": [0.07 * i for i in range(bars)],
                "atr_rel": [0.08 * i for i in range(bars)],
                "close_vs_sma_288": [0.09 * i for i in range(bars)],
                "close_vs_sma_576": [0.10 * i for i in range(bars)],
                "sma_288_vs_sma_576": [0.11 * i for i in range(bars)],
                "bar_range": [0.12 * i for i in range(bars)],
                "bar_body": [0.13 * i for i in range(bars)],
                "close_loc_in_bar": [0.14 * i for i in range(bars)],
                "hour_sin": [0.15 * i for i in range(bars)],
                "hour_cos": [0.16 * i for i in range(bars)],
                "minute_sin": [0.17 * i for i in range(bars)],
                "minute_cos": [0.18 * i for i in range(bars)],
                "london_session": [i % 2 for i in range(bars)],
                "ny_session": [(i + 1) % 2 for i in range(bars)],
                "overlap_session": [1 if 2 <= i <= 4 else 0 for i in range(bars)],
                "week_progress": [i / (bars - 1) for i in range(bars)],
            }
        )

    def test_build_timefm_samples_creates_expected_window_count_and_shapes(self):
        frame = self._frame(bars=8)

        samples = build_timesfm_samples(
            frame,
            context_window=3,
            forecast_horizon=2,
            target_col="log_return_1",
            feature_cols=TIMESFM_OBSERVATION_COLUMNS,
        )

        self.assertEqual(len(samples), 4)
        first = samples[0]
        self.assertIsInstance(first, TimeFMSample)
        self.assertEqual(first.target_context.shape, (3,))
        self.assertEqual(first.dynamic_covariates.shape, (3, len(TIMESFM_OBSERVATION_COLUMNS)))
        self.assertEqual(first.target_future.shape, (2,))

    def test_build_timefm_samples_is_leakage_safe_on_timestamps_and_targets(self):
        frame = self._frame(bars=8)

        samples = build_timesfm_samples(
            frame,
            context_window=3,
            forecast_horizon=2,
            target_col="log_return_1",
            feature_cols=TIMESFM_OBSERVATION_COLUMNS,
        )

        first = samples[0]
        self.assertEqual(first.history_end_time, frame.loc[2, "time"])
        self.assertEqual(first.forecast_start_time, frame.loc[3, "time"])
        self.assertEqual(first.forecast_end_time, frame.loc[4, "time"])
        self.assertListEqual(first.target_context.tolist(), frame.loc[0:2, "log_return_1"].tolist())
        self.assertListEqual(first.target_future.tolist(), frame.loc[3:4, "log_return_1"].tolist())

    def test_build_timefm_samples_skips_invalid_windows_when_dropna_enabled(self):
        frame = self._frame(bars=8)
        frame.loc[2, "returns_12"] = pd.NA

        samples = build_timesfm_samples(
            frame,
            context_window=3,
            forecast_horizon=2,
            target_col="log_return_1",
            feature_cols=TIMESFM_OBSERVATION_COLUMNS,
            dropna=True,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].forecast_start_time, frame.loc[6, "time"])

    def test_directory_grid_loading_sorts_files_and_expands_window_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            early = self._frame(bars=8, symbol="EURUSD")
            late = self._frame(bars=9, symbol="EURUSD")
            early["time"] = pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC")
            late["time"] = pd.date_range("2024-01-08", periods=9, freq="5min", tz="UTC")
            early_path = tmpdir_path / "EURUSD_2024_01.csv"
            late_path = tmpdir_path / "EURUSD_2024_02.csv"
            late.to_csv(late_path, index=False)
            early.to_csv(early_path, index=False)

            loaded = load_timesfm_weekly_csv(early_path, target_col="log_return_1", feature_cols=TIMESFM_OBSERVATION_COLUMNS)
            self.assertIn("UTC", str(loaded["time"].dtype))

            grid = build_timesfm_dataset_grid(
                data_directory=tmpdir,
                context_windows=[3],
                forecast_horizons=[2, 3],
                target_col="log_return_1",
                feature_cols=TIMESFM_OBSERVATION_COLUMNS,
            )

            self.assertEqual(set(grid.keys()), {(3, 2), (3, 3)})
            self.assertEqual(len(grid[(3, 2)]), 9)
            self.assertEqual(len(grid[(3, 3)]), 7)
            self.assertLessEqual(grid[(3, 2)][0].history_start_time, grid[(3, 2)][-1].history_start_time)


if __name__ == "__main__":
    unittest.main()
