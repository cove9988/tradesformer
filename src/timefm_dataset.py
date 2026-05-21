from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.timefm_features import TIMEFM_OBSERVATION_COLUMNS
from src.timefm_split import load_split_manifest, parse_weekly_csv_path
from src.timefm_targets import summarize_future_targets
from src.util.read_config import EnvConfig


@dataclass(frozen=True)
class TimeFMSample:
    symbol: str
    source_file: str
    iso_year: int
    iso_week: int
    context_window: int
    forecast_horizon: int
    target_col: str
    feature_cols: tuple[str, ...]
    history_start_time: pd.Timestamp
    history_end_time: pd.Timestamp
    forecast_start_time: pd.Timestamp
    forecast_end_time: pd.Timestamp
    target_context: np.ndarray
    dynamic_covariates: np.ndarray
    target_future: np.ndarray
    derived_targets: dict[str, float | int | str]


REQUIRED_BASE_COLUMNS = ("time", "symbol", "open", "high", "low", "close")


def _normalize_feature_cols(feature_cols: Iterable[str] | None) -> list[str]:
    if feature_cols is None:
        return list(TIMEFM_OBSERVATION_COLUMNS)
    return [str(column) for column in feature_cols]


def load_timefm_weekly_csv(
    csv_path: str | Path,
    target_col: str,
    feature_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing required column 'time' in {csv_path}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    required_columns: set[str] = set(REQUIRED_BASE_COLUMNS)
    required_columns.add(str(target_col))
    required_columns.update(_normalize_feature_cols(feature_cols))
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")

    return df


def _window_has_missing_values(
    history_frame: pd.DataFrame,
    future_target: pd.Series,
    target_history: pd.Series,
) -> bool:
    return (
        history_frame.isna().values.any()
        or future_target.isna().values.any()
        or target_history.isna().values.any()
    )


def build_timefm_samples(
    frame: pd.DataFrame,
    context_window: int,
    forecast_horizon: int,
    target_col: str,
    feature_cols: Iterable[str] | None = None,
    dropna: bool = True,
    source_file: str = "",
) -> list[TimeFMSample]:
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    if "time" not in frame.columns:
        raise ValueError("frame must include a 'time' column")

    feature_cols = _normalize_feature_cols(feature_cols)
    required_columns = ["time", "symbol", target_col, *feature_cols]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"frame is missing required columns: {missing_columns}")

    parsed_source = None
    if source_file:
        try:
            parsed_source = parse_weekly_csv_path(Path(source_file).name)
        except ValueError:
            parsed_source = None

    frame = frame.sort_values("time").reset_index(drop=True)
    total_rows = len(frame)
    sample_count = total_rows - context_window - forecast_horizon + 1
    if sample_count <= 0:
        return []

    samples: list[TimeFMSample] = []
    for start in range(sample_count):
        history_start = start
        history_end = start + context_window
        forecast_end = history_end + forecast_horizon

        history_frame = frame.iloc[history_start:history_end]
        target_history = history_frame[target_col]
        future_target = frame.iloc[history_end:forecast_end][target_col]

        if dropna and _window_has_missing_values(history_frame[feature_cols], future_target, target_history):
            continue

        symbol = str(history_frame.iloc[0]["symbol"])
        if parsed_source:
            iso_year = int(parsed_source["iso_year"])
            iso_week = int(parsed_source["iso_week"])
        else:
            time_parts = history_frame.iloc[-1]["time"].isocalendar()
            iso_year = int(time_parts.year)
            iso_week = int(time_parts.week)

        samples.append(
            TimeFMSample(
                symbol=symbol,
                source_file=source_file,
                iso_year=iso_year,
                iso_week=iso_week,
                context_window=context_window,
                forecast_horizon=forecast_horizon,
                target_col=target_col,
                feature_cols=tuple(feature_cols),
                history_start_time=history_frame.iloc[0]["time"],
                history_end_time=history_frame.iloc[-1]["time"],
                forecast_start_time=frame.iloc[history_end]["time"],
                forecast_end_time=frame.iloc[forecast_end - 1]["time"],
                target_context=target_history.to_numpy(dtype=float),
                dynamic_covariates=history_frame[feature_cols].to_numpy(dtype=float),
                target_future=future_target.to_numpy(dtype=float),
                derived_targets=summarize_future_targets(future_target.to_numpy(dtype=float)),
            )
        )

    return samples


def build_timefm_samples_from_file(
    csv_path: str | Path,
    context_window: int,
    forecast_horizon: int,
    target_col: str,
    feature_cols: Iterable[str] | None = None,
    dropna: bool = True,
) -> list[TimeFMSample]:
    frame = load_timefm_weekly_csv(csv_path, target_col=target_col, feature_cols=feature_cols)
    return build_timefm_samples(
        frame,
        context_window=context_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
        feature_cols=feature_cols,
        dropna=dropna,
        source_file=str(Path(csv_path)),
    )


def build_timefm_dataset_grid(
    data_directory: str | Path,
    context_windows: Iterable[int],
    forecast_horizons: Iterable[int],
    target_col: str,
    feature_cols: Iterable[str] | None = None,
    dropna: bool = True,
) -> dict[tuple[int, int], list[TimeFMSample]]:
    data_directory = Path(data_directory)
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")

    csv_files = [
        path
        for _, _, path in sorted(
            (
                int(parse_weekly_csv_path(path)["iso_year"]),
                int(parse_weekly_csv_path(path)["iso_week"]),
                path,
            )
            for path in data_directory.glob("*.csv")
        )
    ]
    grid: dict[tuple[int, int], list[TimeFMSample]] = {
        (int(context_window), int(forecast_horizon)): []
        for context_window in context_windows
        for forecast_horizon in forecast_horizons
    }

    feature_cols = _normalize_feature_cols(feature_cols)
    for csv_file in csv_files:
        frame = load_timefm_weekly_csv(csv_file, target_col=target_col, feature_cols=feature_cols)
        for context_window, forecast_horizon in grid:
            grid[(context_window, forecast_horizon)].extend(
                build_timefm_samples(
                    frame,
                    context_window=context_window,
                    forecast_horizon=forecast_horizon,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    dropna=dropna,
                    source_file=str(csv_file),
                )
            )

    return grid


def build_timefm_dataset_from_config(
    data_directory: str | Path,
    env_config_file: str = "./src/configure.json",
    feature_cols: Iterable[str] | None = None,
    dropna: bool = True,
) -> dict[tuple[int, int], list[TimeFMSample]]:
    cf = EnvConfig(env_config_file)
    config_feature_cols = feature_cols or cf.timefm_observation_list()
    return build_timefm_dataset_grid(
        data_directory=data_directory,
        context_windows=cf.env_parameters("context_windows"),
        forecast_horizons=cf.env_parameters("forecast_horizons"),
        target_col=cf.env_parameters("target_series"),
        feature_cols=config_feature_cols,
        dropna=dropna,
    )


def build_timefm_samples_for_split(
    data_directory: str | Path,
    manifest_path: str | Path,
    split_name: str,
    context_window: int,
    forecast_horizon: int,
    target_col: str,
    feature_cols: Iterable[str] | None = None,
    dropna: bool = True,
) -> list[TimeFMSample]:
    manifest = load_split_manifest(manifest_path)
    split_files = manifest["splits"][split_name]
    data_directory = Path(data_directory)

    samples: list[TimeFMSample] = []
    for file_name in split_files:
        samples.extend(
            build_timefm_samples_from_file(
                data_directory / file_name,
                context_window=context_window,
                forecast_horizon=forecast_horizon,
                target_col=target_col,
                feature_cols=feature_cols,
                dropna=dropna,
            )
        )
    return samples
