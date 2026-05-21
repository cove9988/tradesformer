import math
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.timefm_features import (
    TIMEFM_OBSERVATION_COLUMNS,
    add_causal_timefm_features,
    add_time_context_features,
    validate_timefm_week,
)


class TimeFMFeaturesTest(unittest.TestCase):
    def _sample_week(self, bars=600):
        start = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        times = [start + timedelta(minutes=5 * i) for i in range(bars)]
        close = [1.1000 + 0.0001 * i for i in range(bars)]
        open_ = [price - 0.00005 for price in close]
        high = [price + 0.00015 for price in close]
        low = [price - 0.00015 for price in close]
        frame = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }, index=pd.DatetimeIndex(times, tz="UTC"))
        return frame

    def test_week_open_relative_features_anchor_to_first_open(self):
        df = self._sample_week(bars=600)
        df = add_time_context_features(df, symbol="EURUSD")
        result = add_causal_timefm_features(df, point=100000)

        first = result.iloc[0]
        self.assertAlmostEqual(first["open_rel_week_open"], 0.0, places=6)
        self.assertAlmostEqual(first["close_rel_week_open"], 5.0, places=6)

        second = result.iloc[1]
        self.assertAlmostEqual(second["close_rel_week_open"], 15.0, places=6)

    def test_returns_and_week_progress_are_causal(self):
        df = self._sample_week(bars=600)
        df = add_time_context_features(df, symbol="EURUSD")
        result = add_causal_timefm_features(df, point=100000)

        self.assertTrue(math.isnan(result.iloc[0]["returns_12"]))
        expected_return_12 = math.log(result.iloc[20]["close"] / result.iloc[8]["close"])
        self.assertAlmostEqual(result.iloc[20]["returns_12"], round(expected_return_12, 6), places=6)
        self.assertAlmostEqual(result.iloc[0]["week_progress"], 0.0, places=6)
        self.assertAlmostEqual(result.iloc[-1]["week_progress"], 1.0, places=6)

    def test_validation_requires_full_week_and_complete_feature_columns(self):
        df = self._sample_week(bars=600)
        df = add_time_context_features(df, symbol="EURUSD")
        result = add_causal_timefm_features(df, point=100000)

        is_valid, reason = validate_timefm_week(result, bars_per_week=1440, feature_cols=TIMEFM_OBSERVATION_COLUMNS)
        self.assertFalse(is_valid)
        self.assertIn("expected 1440 bars", reason)


if __name__ == "__main__":
    unittest.main()
