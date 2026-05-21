"""Run forecasts using a fine-tuned TimesFM 2.5 model with all covariates.

Loads the LoRA/DoRA adapter from mt5/checkpoints/ and forecasts the
next HORIZON bars for each instrument using all covariate channels.

Run:
    python mt5/scripts/forecast.py
    python mt5/scripts/forecast.py --instrument XAUUSD --horizon 48
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    CHECKPOINT_DIR,
    CONTEXT_LEN,
    COV_DIR,
    HORIZON_LEN,
    INSTRUMENTS,
    LORA_ALPHA,
    LORA_RANK,
    NUM_ADAPTER_LAYERS,
    TIMEFRAME,
)

import timesfm
from peft import PEFTConfig, inject_adapters, load_adapter_weights


def parse_args():
    p = argparse.ArgumentParser(description="Forecast with fine-tuned TimesFM 2.5")
    p.add_argument("--instrument", default=None, help="Instrument name (default: all)")
    p.add_argument("--horizon", type=int, default=HORIZON_LEN)
    p.add_argument("--context", type=int, default=CONTEXT_LEN)
    p.add_argument(
        "--adapter_path",
        default=str(CHECKPOINT_DIR / "best_adapter.safetensors"),
    )
    p.add_argument("--no_adapter", action="store_true", help="Run base model only")
    p.add_argument("--no_covariates", action="store_true", help="Forecast without covariates")
    return p.parse_args()


def get_covariate_columns(df: pd.DataFrame) -> dict:
    """Extract dynamic numerical covariate arrays from a covariate DataFrame."""
    # All columns that are covariates (not OHLCV or metadata)
    skip_cols = {
        "datetime", "open", "high", "low", "close",
        "tick_volume", "spread", "asset_class", "instrument",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cov_cols = [c for c in numeric_cols if c not in skip_cols]
    return cov_cols


def forecast_instrument(model, name: str, args) -> dict | None:
    """Forecast one instrument with all covariates."""
    tf_label = TIMEFRAME.replace("TIMEFRAME_", "")
    path = COV_DIR / f"{name}_{tf_label}_covariates.csv"

    if not path.exists():
        print(f"  No covariate data found for {name}")
        return None

    df = pd.read_csv(path, parse_dates=["datetime"])
    print(f"  {name}: {len(df):,} bars available")

    # Use last `context + horizon` rows for context + known future covariates
    context = args.context
    horizon = args.horizon

    if len(df) < context:
        print(f"  WARNING: Only {len(df)} bars, need at least {context}. Skipping.")
        return None

    # Close prices — last `context` bars as input
    close_context = df["close"].values[-context:].astype(np.float32)

    if args.no_covariates:
        # Simple forecast without covariates
        point, quantiles = model.forecast(
            horizon=horizon,
            inputs=[close_context],
        )
    else:
        # Build covariate arrays — each needs length context + horizon
        # For the future horizon portion, we extend time features forward
        cov_cols = get_covariate_columns(df)

        # Take last context rows for the known part
        # For future covariates, we need to project them forward
        last_dt = pd.Timestamp(df["datetime"].iloc[-1])
        future_dts = pd.date_range(
            start=last_dt + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )

        # Build dynamic numerical covariates
        dynamic_numerical = {}
        for col in cov_cols:
            # Context portion (known)
            ctx_values = df[col].values[-context:].astype(np.float32)

            # Future portion — for time features we can compute exactly,
            # for others we repeat the last known value
            if col.startswith(("hour_", "dow_", "woy_", "moy_", "dom_")):
                # Recompute cyclical features for future dates
                future_vals = _compute_time_feature(col, future_dts)
            elif col.startswith("session_"):
                future_vals = _compute_session_feature(col, future_dts)
            elif col.startswith("phase_") or col.startswith("is_") or col.startswith("days_to"):
                # Moon phases are slow-moving — forward fill from last known
                future_vals = np.full(horizon, ctx_values[-1], dtype=np.float32)
            elif col.startswith("econ_"):
                # Economic events — set to 0 for unknown future
                future_vals = np.zeros(horizon, dtype=np.float32)
            elif col == "is_weekend":
                future_vals = (future_dts.dayofweek >= 5).astype(np.float32)
            else:
                # Technical features (ATR, volume, returns) — forward fill
                future_vals = np.full(horizon, ctx_values[-1], dtype=np.float32)

            full_array = np.concatenate([ctx_values, future_vals])
            dynamic_numerical[col] = [full_array]

        # Static categorical covariates
        asset_class = INSTRUMENTS[name]["asset_class"]

        point, quantiles = model.forecast_with_covariates(
            inputs=[close_context],
            dynamic_numerical_covariates=dynamic_numerical,
            static_categorical_covariates={
                "asset_class": [asset_class],
                "instrument": [name],
            },
            xreg_mode="xreg + timesfm",
        )

    # Convert list outputs to ndarrays for easy indexing
    point = np.array(point)
    quantiles = np.array(quantiles)

    result = {
        "instrument": name,
        "last_close": float(close_context[-1]),
        "last_datetime": str(df["datetime"].iloc[-1]),
        "horizon": horizon,
        "point_forecast": point[0].tolist(),
        "quantile_10": quantiles[0, :, 1].tolist(),  # 10th percentile
        "quantile_50": quantiles[0, :, 0].tolist(),   # median / mean
        "quantile_90": quantiles[0, :, -1].tolist(),  # 90th percentile
    }

    # Print summary
    pf = point[0]
    direction = "UP" if pf[-1] > close_context[-1] else "DOWN"
    pct_change = (pf[-1] / close_context[-1] - 1) * 100
    print(f"  Forecast: {direction} {abs(pct_change):.3f}% over {horizon} bars")
    print(f"  Last close: {close_context[-1]:.5f}")
    print(f"  End forecast: {pf[-1]:.5f}")
    print(f"  90% CI: [{quantiles[0, -1, 1]:.5f}, {quantiles[0, -1, -1]:.5f}]")

    return result


def _compute_time_feature(col: str, dts: pd.DatetimeIndex) -> np.ndarray:
    """Compute a cyclical time feature for future dates."""
    mapping = {
        "hour_sin": lambda: np.sin(2 * np.pi * dts.hour / 24),
        "hour_cos": lambda: np.cos(2 * np.pi * dts.hour / 24),
        "dow_sin": lambda: np.sin(2 * np.pi * dts.dayofweek / 7),
        "dow_cos": lambda: np.cos(2 * np.pi * dts.dayofweek / 7),
        "woy_sin": lambda: np.sin(2 * np.pi * dts.isocalendar().week.values.astype(float) / 52),
        "woy_cos": lambda: np.cos(2 * np.pi * dts.isocalendar().week.values.astype(float) / 52),
        "moy_sin": lambda: np.sin(2 * np.pi * dts.month / 12),
        "moy_cos": lambda: np.cos(2 * np.pi * dts.month / 12),
        "dom_sin": lambda: np.sin(2 * np.pi * dts.day / 31),
        "dom_cos": lambda: np.cos(2 * np.pi * dts.day / 31),
    }
    fn = mapping.get(col)
    if fn is not None:
        return fn().astype(np.float32)
    return np.zeros(len(dts), dtype=np.float32)


def _compute_session_feature(col: str, dts: pd.DatetimeIndex) -> np.ndarray:
    """Compute session flags for future dates."""
    from config import SESSIONS

    hour = dts.hour
    if col == "session_overlap":
        london = ((hour >= 7) & (hour < 16)).astype(np.float32)
        ny = ((hour >= 12) & (hour < 21)).astype(np.float32)
        return london * ny

    session_name = col.replace("session_", "")
    if session_name in SESSIONS:
        start_h, end_h = SESSIONS[session_name]
        if start_h < end_h:
            return ((hour >= start_h) & (hour < end_h)).astype(np.float32)
        else:
            return ((hour >= start_h) | (hour < end_h)).astype(np.float32)

    return np.zeros(len(dts), dtype=np.float32)


def main():
    args = parse_args()

    print("=" * 60)
    print("TimesFM 2.5 Forecast — MT5 Financial Instruments")
    print("=" * 60)

    # Detect and select best device
    print("\n--- Hardware Detection ---")
    if torch.cuda.is_available():
        best_idx = 0
        best_vram = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  CUDA:{i}  {props.name}  ({vram_gb:.1f} GB VRAM)")
            if props.total_memory > best_vram:
                best_vram = props.total_memory
                best_idx = i
        device = torch.device(f"cuda:{best_idx}")
        torch.cuda.set_device(device)
        print(f"  >> Inference device: {device}  ({torch.cuda.get_device_properties(best_idx).name})")
    else:
        device = torch.device("cpu")
        print(f"  >> Inference device: CPU")
    print("---\n")

    # Load model
    print("Loading TimesFM 2.5...")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    # Load fine-tuned adapter
    if not args.no_adapter and Path(args.adapter_path).exists():
        print(f"Loading adapter from {args.adapter_path}...")
        # Inject adapters matching the training config, then load weights
        peft_config = PEFTConfig(
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            num_adapter_layers=NUM_ADAPTER_LAYERS,
        )
        inject_adapters(model.model, peft_config)
        load_adapter_weights(model.model, args.adapter_path)
        # Move adapter weights to the same device as the model
        model.model.to(device)
        print("Adapter loaded.")
    elif not args.no_adapter:
        print(f"No adapter found at {args.adapter_path} — using base model")

    # Compile the model for inference
    model.compile(
        timesfm.ForecastConfig(
            max_context=args.context,
            max_horizon=args.horizon,
            per_core_batch_size=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )

    # Forecast
    instruments = (
        {args.instrument: INSTRUMENTS[args.instrument]}
        if args.instrument and args.instrument in INSTRUMENTS
        else INSTRUMENTS
    )

    results = []
    for name, info in instruments.items():
        print(f"\n--- {name} ({info['mt5_symbol']}) ---")
        result = forecast_instrument(model, name, args)
        if result:
            results.append(result)

    # Save results
    if results:
        out_path = COV_DIR.parent / "forecast_results.json"
        import json
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
