import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.timesfm_data_audit import audit_timesfm_dataset
from src.timesfm_split import build_split_manifest, write_split_manifest


class TimeFMDataAuditTest(unittest.TestCase):
    def _write_week(self, directory: Path, year: int, week: int, bars: int = 12):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=bars, freq="5min", tz="UTC"),
                "symbol": ["EURUSD"] * bars,
                "open": [1.1 + i * 0.0001 for i in range(bars)],
                "high": [1.1002 + i * 0.0001 for i in range(bars)],
                "low": [1.0998 + i * 0.0001 for i in range(bars)],
                "close": [1.1 + i * 0.0001 for i in range(bars)],
                "log_return_1": [0.0] + [0.0001] * (bars - 1),
                "close_rel_week_open": list(range(bars)),
                "hour_sin": [0.0] * bars,
            }
        )
        path = directory / f"EURUSD_{year}_{week:02d}.csv"
        frame.to_csv(path, index=False)
        return path

    def test_audit_reports_split_counts_date_ranges_and_window_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            train_dir = base / "train"
            train_dir.mkdir()
            for week in range(1, 6):
                self._write_week(train_dir, 2024, week)

            manifest = build_split_manifest(train_dir, symbol="EURUSD", validation_weeks=1, test_weeks=1)
            manifest_path = base / "manifest.json"
            write_split_manifest(manifest_path, manifest)

            report = audit_timesfm_dataset(
                data_directory=train_dir,
                manifest_path=manifest_path,
                target_col="log_return_1",
                feature_cols=["close_rel_week_open", "hour_sin"],
                context_windows=[3],
                forecast_horizons=[2],
            )

            self.assertEqual(report["symbol"], "EURUSD")
            self.assertEqual(report["splits"]["train"]["file_count"], 3)
            self.assertEqual(report["splits"]["val"]["file_count"], 1)
            self.assertEqual(report["splits"]["test"]["file_count"], 1)
            self.assertEqual(report["splits"]["train"]["sample_counts"]["3x2"], 24)
            self.assertTrue(report["splits"]["train"]["start_time"].endswith("+00:00"))
            self.assertTrue(report["splits"]["train"]["end_time"].endswith("+00:00"))
            self.assertEqual(report["splits"]["train"]["nan_rates"]["log_return_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
