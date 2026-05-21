import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.timesfm_split import (
    build_split_manifest,
    load_split_manifest,
    parse_weekly_csv_path,
    write_split_manifest,
)


class TimeFMSplitTest(unittest.TestCase):
    def _write_week(self, directory: Path, year: int, week: int, bars: int = 8):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=bars, freq="5min", tz="UTC"),
                "symbol": ["EURUSD"] * bars,
                "open": [1.1] * bars,
                "high": [1.11] * bars,
                "low": [1.09] * bars,
                "close": [1.1] * bars,
                "log_return_1": [0.0] * bars,
                "close_rel_week_open": [0.0] * bars,
            }
        )
        path = directory / f"EURUSD_{year}_{week:02d}.csv"
        frame.to_csv(path, index=False)
        return path

    def test_parse_weekly_csv_path_extracts_symbol_year_week(self):
        parsed = parse_weekly_csv_path(Path("EURUSD_2024_35.csv"))

        self.assertEqual(parsed["symbol"], "EURUSD")
        self.assertEqual(parsed["iso_year"], 2024)
        self.assertEqual(parsed["iso_week"], 35)
        self.assertEqual(parsed["file_name"], "EURUSD_2024_35.csv")

    def test_parse_weekly_csv_path_accepts_single_digit_week(self):
        parsed = parse_weekly_csv_path(Path("AUDUSD_2022_1.csv"))

        self.assertEqual(parsed["symbol"], "AUDUSD")
        self.assertEqual(parsed["iso_year"], 2022)
        self.assertEqual(parsed["iso_week"], 1)
        self.assertEqual(parsed["file_name"], "AUDUSD_2022_1.csv")

    def test_build_split_manifest_uses_chronological_tail_for_val_and_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            for week in range(1, 11):
                self._write_week(data_dir, 2024, week)

            manifest = build_split_manifest(
                data_dir,
                symbol="EURUSD",
                validation_weeks=2,
                test_weeks=2,
            )

            self.assertEqual(len(manifest["all_weeks"]), 10)
            self.assertEqual(len(manifest["splits"]["train"]), 6)
            self.assertEqual(len(manifest["splits"]["val"]), 2)
            self.assertEqual(len(manifest["splits"]["test"]), 2)
            self.assertEqual(manifest["splits"]["val"][0], "EURUSD_2024_07.csv")
            self.assertEqual(manifest["splits"]["test"][0], "EURUSD_2024_09.csv")
            self.assertEqual(manifest["splits"]["test"][1], "EURUSD_2024_10.csv")

    def test_write_and_load_split_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "train"
            data_dir.mkdir()
            for week in range(1, 5):
                self._write_week(data_dir, 2024, week)

            manifest = build_split_manifest(data_dir, symbol="EURUSD", validation_weeks=1, test_weeks=1)
            manifest_path = base / "manifest.json"
            write_split_manifest(manifest_path, manifest)

            self.assertTrue(manifest_path.exists())
            loaded = load_split_manifest(manifest_path)
            self.assertEqual(loaded["symbol"], "EURUSD")
            self.assertEqual(loaded["splits"], manifest["splits"])

            raw = json.loads(manifest_path.read_text())
            self.assertIn("generated_at", raw)


if __name__ == "__main__":
    unittest.main()
