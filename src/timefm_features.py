import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFM_OBSERVATION_COLUMNS = [
    "close_rel_week_open",
    "high_rel_week_open",
    "low_rel_week_open",
    "open_rel_week_open",
    "returns_12",
    "returns_48",
    "returns_96",
    "returns_288",
    "realized_vol_48",
    "realized_vol_96",
    "realized_vol_288",
    "atr_rel",
    "close_vs_sma_288",
    "close_vs_sma_576",
    "sma_288_vs_sma_576",
    "bar_range",
    "bar_body",
    "close_loc_in_bar",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "london_session",
    "ny_session",
    "overlap_session",
    "week_progress",
]

TIMEFM_HELPER_COLUMNS = [
    "log_return_1",
    "true_range",
    "atr",
    "sma_288",
    "sma_576",
    "iso_year",
    "iso_week",
]

EPSILON = 1e-9


def _safe_log_ratio(current: pd.Series, prior: pd.Series) -> pd.Series:
    prior = prior.replace(0, np.nan)
    ratio = current / prior
    ratio = ratio.where((current > 0) & (prior > 0))
    return np.log(ratio)


def add_time_context_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex before adding TimeFM time features")

    result = df.copy()
    result.index = pd.to_datetime(result.index, utc=True)

    minute_of_day = result.index.hour * 60 + result.index.minute
    result["hour_sin"] = np.sin(2 * np.pi * result.index.hour / 24).round(6)
    result["hour_cos"] = np.cos(2 * np.pi * result.index.hour / 24).round(6)
    result["minute_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60)).round(6)
    result["minute_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60)).round(6)

    result["london_session"] = ((result.index.hour >= 8) & (result.index.hour < 16)).astype(int)
    result["ny_session"] = ((result.index.hour >= 13) & (result.index.hour < 21)).astype(int)
    result["overlap_session"] = ((result.index.hour >= 13) & (result.index.hour < 16)).astype(int)
    result["symbol"] = symbol
    return result


def add_causal_timefm_features(
    df: pd.DataFrame,
    point: float,
    returns_windows: Iterable[int] = (12, 48, 96, 288),
    realized_vol_windows: Iterable[int] = (48, 96, 288),
    sma_windows: Iterable[int] = (288, 576),
    atr_window: int = 288,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex before adding causal TimeFM features")

    result = df.copy()
    result.index = pd.to_datetime(result.index, utc=True)

    open_ = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    close = result["close"].astype(float)

    iso_calendar = result.index.isocalendar()
    iso_year = iso_calendar.year.astype(int)
    iso_week = iso_calendar.week.astype(int)
    week_group = [iso_year, iso_week]

    first_week_open = open_.groupby(week_group).transform("first")
    result["open_rel_week_open"] = ((open_ - first_week_open) * point).round(6)
    result["high_rel_week_open"] = ((high - first_week_open) * point).round(6)
    result["low_rel_week_open"] = ((low - first_week_open) * point).round(6)
    result["close_rel_week_open"] = ((close - first_week_open) * point).round(6)

    prev_close = close.shift(1)
    log_return_1 = _safe_log_ratio(close, prev_close)
    result["log_return_1"] = log_return_1.round(6)

    for window in returns_windows:
        result[f"returns_{window}"] = _safe_log_ratio(close, close.shift(window)).round(6)

    for window in realized_vol_windows:
        result[f"realized_vol_{window}"] = (
            log_return_1.rolling(window, min_periods=window).std().round(6)
        )

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["true_range"] = true_range.round(6)
    result["atr"] = true_range.rolling(atr_window, min_periods=atr_window).mean().round(6)
    result["atr_rel"] = (result["atr"] / close.replace(0, np.nan)).round(6)

    sma_windows = tuple(int(window) for window in sma_windows)
    for window in sma_windows:
        result[f"sma_{window}"] = close.rolling(window, min_periods=window).mean().round(6)

    short_window, long_window = sma_windows
    result[f"close_vs_sma_{short_window}"] = (
        close / result[f"sma_{short_window}"].replace(0, np.nan) - 1
    ).round(6)
    result[f"close_vs_sma_{long_window}"] = (
        close / result[f"sma_{long_window}"].replace(0, np.nan) - 1
    ).round(6)
    result[f"sma_{short_window}_vs_sma_{long_window}"] = (
        result[f"sma_{short_window}"] / result[f"sma_{long_window}"].replace(0, np.nan) - 1
    ).round(6)

    spread = (high - low).replace(0, np.nan)
    result["bar_range"] = ((high - low) / close.replace(0, np.nan)).round(6)
    result["bar_body"] = ((close - open_) / open_.replace(0, np.nan)).round(6)
    result["close_loc_in_bar"] = (((close - low) / spread).clip(0, 1).fillna(0.5)).round(6)

    bar_index_in_week = open_.groupby(week_group).cumcount().astype(float)
    bars_in_week = open_.groupby(week_group).transform("size").astype(float)
    result["week_progress"] = np.where(
        bars_in_week > 1,
        bar_index_in_week / (bars_in_week - 1),
        0.0,
    ).round(6)

    result["iso_year"] = iso_year
    result["iso_week"] = iso_week

    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return result


def validate_timefm_week(
    week_df: pd.DataFrame,
    bars_per_week: int,
    feature_cols: Iterable[str],
) -> tuple[bool, str]:
    if len(week_df) != bars_per_week:
        return False, f"expected {bars_per_week} bars, found {len(week_df)}"

    feature_frame = week_df[list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    missing_cols = [column for column in feature_frame.columns if feature_frame[column].isna().any()]
    if missing_cols:
        preview = ", ".join(missing_cols[:5])
        suffix = "..." if len(missing_cols) > 5 else ""
        return False, f"missing feature values in {preview}{suffix}"

    return True, "ok"
