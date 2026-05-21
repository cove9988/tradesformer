from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from src.timefm_dataset import build_timefm_samples_for_split
from src.timefm_forecaster import TimeFMForecastResult, TimeFMModelHandle, forecast_batch, load_model


def _result_to_record(result: TimeFMForecastResult) -> dict[str, object]:
    return {
        "model_id": result.model_id,
        "backend_name": result.backend_name,
        "symbol": result.symbol,
        "source_file": result.source_file,
        "context_window": result.context_window,
        "forecast_horizon": result.forecast_horizon,
        "forecast_start_time": result.forecast_start_time,
        "forecast_end_time": result.forecast_end_time,
        "actual_future": result.actual_future.tolist(),
        "predicted_mean": result.predicted_mean.tolist(),
        "lower_quantile": result.lower_quantile.tolist() if result.lower_quantile is not None else None,
        "upper_quantile": result.upper_quantile.tolist() if result.upper_quantile is not None else None,
    }


def _output_path(output_directory: Path, symbol: str, split_name: str, context_window: int, forecast_horizon: int) -> Path:
    return output_directory / symbol / split_name / f"context_{context_window}_horizon_{forecast_horizon}.jsonl"


def run_forecasts_for_split(
    data_directory: str | Path,
    manifest_path: str | Path,
    split_name: str,
    output_directory: str | Path,
    model: TimeFMModelHandle,
    target_col: str,
    feature_cols: Iterable[str],
    context_windows: Iterable[int],
    forecast_horizons: Iterable[int],
    symbol: str = "EURUSD",
) -> list[Path]:
    data_directory = Path(data_directory)
    output_directory = Path(output_directory)
    feature_cols = list(feature_cols)

    written_paths: list[Path] = []
    for context_window in context_windows:
        for forecast_horizon in forecast_horizons:
            samples = build_timefm_samples_for_split(
                data_directory=data_directory,
                manifest_path=manifest_path,
                split_name=split_name,
                context_window=int(context_window),
                forecast_horizon=int(forecast_horizon),
                target_col=target_col,
                feature_cols=feature_cols,
            )
            results = forecast_batch(model, samples)
            output_path = _output_path(output_directory, symbol, split_name, int(context_window), int(forecast_horizon))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(json.dumps(_result_to_record(result), sort_keys=True) for result in results) + ("\n" if results else ""))
            written_paths.append(output_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TimeFM forecasts for one manifest-defined split")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--data-directory", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-directory", default="./data/predictions/timefm")
    parser.add_argument("--target-col", default="log_return_1")
    parser.add_argument("--feature-cols", nargs="*", default=["close_rel_week_open", "hour_sin"])
    parser.add_argument("--context-windows", nargs="*", type=int, default=[96, 192, 288])
    parser.add_argument("--forecast-horizons", nargs="*", type=int, default=[12, 24])
    parser.add_argument("--backend-name", default="fake-timesfm")
    parser.add_argument("--model-id", default="google/timesfm-1.0-200m")
    args = parser.parse_args()

    data_directory = Path(args.data_directory) if args.data_directory else Path(f"./data/split/{args.symbol}/weekly/train")
    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(f"./data/split/{args.symbol}/weekly/manifest.json")
    model = load_model(model_id=args.model_id, backend_name=args.backend_name)
    written = run_forecasts_for_split(
        data_directory=data_directory,
        manifest_path=manifest_path,
        split_name=args.split,
        output_directory=Path(args.output_directory),
        model=model,
        target_col=args.target_col,
        feature_cols=list(args.feature_cols),
        context_windows=list(args.context_windows),
        forecast_horizons=list(args.forecast_horizons),
        symbol=args.symbol,
    )
    print(json.dumps([str(path) for path in written], indent=2))


if __name__ == "__main__":
    main()
