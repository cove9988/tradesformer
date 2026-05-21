from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.timefm_dataset import build_timefm_samples_from_file
from src.timefm_forecaster import forecast_sample, load_model


def build_spike_report(
    csv_path: str | Path,
    target_col: str,
    feature_cols: list[str],
    context_window: int,
    forecast_horizon: int,
    backend_name: str = "fake-timesfm",
    model_id: str = "google/timesfm-2.5-200m-transformers",
) -> dict[str, object]:
    samples = build_timefm_samples_from_file(
        csv_path,
        context_window=context_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    if not samples:
        raise ValueError(f"No TimeFM samples could be built from {csv_path}")

    model = load_model(model_id=model_id, backend_name=backend_name)
    result = forecast_sample(model, samples[0])
    return {
        "backend_name": result.backend_name,
        "model_id": result.model_id,
        "source_file": result.source_file,
        "context_window": result.context_window,
        "forecast_horizon": result.forecast_horizon,
        "predicted_mean_shape": list(result.predicted_mean.shape),
        "forecast_start_time": result.forecast_start_time,
        "forecast_end_time": result.forecast_end_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal TimeFM loader spike")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--target-col", default="log_return_1")
    parser.add_argument("--feature-cols", nargs="*", default=["close_rel_week_open", "hour_sin"])
    parser.add_argument("--context-window", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=12)
    parser.add_argument("--backend-name", default="fake-timesfm")
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    args = parser.parse_args()

    report = build_spike_report(
        csv_path=Path(args.csv_path),
        target_col=args.target_col,
        feature_cols=list(args.feature_cols),
        context_window=args.context_window,
        forecast_horizon=args.forecast_horizon,
        backend_name=args.backend_name,
        model_id=args.model_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
