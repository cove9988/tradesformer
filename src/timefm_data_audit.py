from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.timefm_dataset import load_timefm_weekly_csv
from src.timefm_split import load_split_manifest


def _sample_count_for_file(row_count: int, context_window: int, forecast_horizon: int) -> int:
    return max(0, row_count - context_window - forecast_horizon + 1)


def _audit_files(
    data_directory: Path,
    file_names: list[str],
    target_col: str,
    feature_cols: Iterable[str],
    context_windows: Iterable[int],
    forecast_horizons: Iterable[int],
) -> dict[str, object]:
    feature_cols = list(feature_cols)
    frames = []
    row_counts: list[int] = []
    sample_counts = {f"{context}x{horizon}": 0 for context in context_windows for horizon in forecast_horizons}

    for file_name in file_names:
        frame = load_timefm_weekly_csv(data_directory / file_name, target_col=target_col, feature_cols=feature_cols)
        frames.append(frame)
        row_count = len(frame)
        row_counts.append(row_count)
        for context in context_windows:
            for horizon in forecast_horizons:
                sample_counts[f"{context}x{horizon}"] += _sample_count_for_file(row_count, context, horizon)

    if not frames:
        return {
            "file_count": 0,
            "row_count": 0,
            "sample_counts": sample_counts,
            "start_time": None,
            "end_time": None,
            "nan_rates": {column: None for column in [target_col, *feature_cols]},
        }

    combined = pd.concat(frames, ignore_index=True)
    audit_columns = [target_col, *feature_cols]
    nan_rates = {column: float(combined[column].isna().mean()) for column in audit_columns}

    return {
        "file_count": len(file_names),
        "row_count": int(sum(row_counts)),
        "sample_counts": sample_counts,
        "start_time": str(combined["time"].min()),
        "end_time": str(combined["time"].max()),
        "nan_rates": nan_rates,
    }


def audit_timefm_dataset(
    data_directory: str | Path,
    manifest_path: str | Path,
    target_col: str,
    feature_cols: Iterable[str],
    context_windows: Iterable[int],
    forecast_horizons: Iterable[int],
) -> dict[str, object]:
    data_directory = Path(data_directory)
    manifest = load_split_manifest(manifest_path)
    splits = manifest["splits"]

    return {
        "symbol": manifest["symbol"],
        "manifest_path": str(Path(manifest_path)),
        "splits": {
            split_name: _audit_files(
                data_directory=data_directory,
                file_names=list(file_names),
                target_col=target_col,
                feature_cols=feature_cols,
                context_windows=context_windows,
                forecast_horizons=forecast_horizons,
            )
            for split_name, file_names in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TimeFM dataset splits")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-directory", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--target-col", default="log_return_1")
    parser.add_argument("--feature-cols", nargs="*", default=["close_rel_week_open", "hour_sin"])
    parser.add_argument("--context-windows", nargs="*", type=int, default=[96, 192, 288])
    parser.add_argument("--forecast-horizons", nargs="*", type=int, default=[12, 24])
    args = parser.parse_args()

    data_directory = Path(args.data_directory) if args.data_directory else Path(f"./data/split/{args.symbol}/weekly/train")
    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(f"./data/split/{args.symbol}/weekly/manifest.json")
    report = audit_timefm_dataset(
        data_directory=data_directory,
        manifest_path=manifest_path,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        context_windows=args.context_windows,
        forecast_horizons=args.forecast_horizons,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
