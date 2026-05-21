"""Build the full covariate dataset for each instrument.

Merges MT5 OHLCV data with:
  - Moon phases (phase_pct, phase_sin/cos, is_new/full/quarter, days_to)
  - Economic calendar (per-event binary flags, composite flags, days_to)
  - Time features (hour, dow, woy, moy — all sin/cos cyclical)
  - Market session flags (Tokyo, London, NY, overlap)
  - Technical proxies (ATR normalized, volume normalized, spread)

Outputs one CSV per instrument in mt5/data/covariates/:
    DJ30_H1_covariates.csv

Run:
    python mt5/scripts/build_covariates.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COV_DIR,
    ECON_DIR,
    INSTRUMENTS,
    MOON_DIR,
    MT5_DIR,
    SESSIONS,
    TIMEFRAME,
)


def load_moon_data() -> pd.DataFrame:
    """Load precomputed hourly moon phase data."""
    path = MOON_DIR / "moon_phases_hourly.csv"
    if not path.exists():
        print(f"Moon data not found at {path}")
        print("Run: python mt5/scripts/generate_moon_phases.py")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def load_econ_data() -> pd.DataFrame:
    """Load precomputed hourly economic calendar data."""
    path = ECON_DIR / "economic_calendar_hourly.csv"
    if not path.exists():
        print(f"Economic calendar not found at {path}")
        print("Run: python mt5/scripts/generate_economic_calendar.py")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-based features."""
    dt = df["datetime"]

    # Hour of day (0–23)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24).astype(np.float32)

    # Day of week (0=Mon, 6=Sun)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7).astype(np.float32)

    # Week of year (1–52)
    woy = dt.dt.isocalendar().week.values.astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52).astype(np.float32)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52).astype(np.float32)

    # Month of year (1–12)
    df["moy_sin"] = np.sin(2 * np.pi * dt.dt.month / 12).astype(np.float32)
    df["moy_cos"] = np.cos(2 * np.pi * dt.dt.month / 12).astype(np.float32)

    # Day of month (1–31)
    df["dom_sin"] = np.sin(2 * np.pi * dt.dt.day / 31).astype(np.float32)
    df["dom_cos"] = np.cos(2 * np.pi * dt.dt.day / 31).astype(np.float32)

    # Is weekend
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)

    return df


def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add forex market session binary flags."""
    hour = df["datetime"].dt.hour

    for session_name, (start_h, end_h) in SESSIONS.items():
        if start_h < end_h:
            df[f"session_{session_name}"] = (
                (hour >= start_h) & (hour < end_h)
            ).astype(np.float32)
        else:
            # Wraps around midnight
            df[f"session_{session_name}"] = (
                (hour >= start_h) | (hour < end_h)
            ).astype(np.float32)

    # London/NY overlap
    df["session_overlap"] = (
        df["session_london"] * df["session_ny"]
    ).astype(np.float32)

    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized technical features from OHLCV data."""
    # ATR proxy (high - low), normalized
    if "high" in df.columns and "low" in df.columns:
        atr = (df["high"] - df["low"]).astype(np.float32)
        atr_mean = atr.rolling(window=24, min_periods=1).mean()
        df["atr_norm"] = (atr / (atr_mean + 1e-8)).astype(np.float32)

    # Volume normalized (rolling mean)
    if "tick_volume" in df.columns:
        vol = df["tick_volume"].astype(np.float32)
        vol_mean = vol.rolling(window=24, min_periods=1).mean()
        df["volume_norm"] = (vol / (vol_mean + 1e-8)).astype(np.float32)

    # Spread normalized
    if "spread" in df.columns:
        spread = df["spread"].astype(np.float32)
        spread_mean = spread.rolling(window=24, min_periods=1).mean()
        df["spread_norm"] = (spread / (spread_mean + 1e-8)).astype(np.float32)

    # Returns (pct change)
    if "close" in df.columns:
        df["returns_1h"] = df["close"].pct_change(1).fillna(0).astype(np.float32)
        df["returns_4h"] = df["close"].pct_change(4).fillna(0).astype(np.float32)
        df["returns_24h"] = df["close"].pct_change(24).fillna(0).astype(np.float32)

    return df


def build_covariates_for_instrument(name: str, info: dict) -> pd.DataFrame | None:
    """Build the full covariate DataFrame for one instrument."""
    tf_label = TIMEFRAME.replace("TIMEFRAME_", "")
    mt5_path = MT5_DIR / f"{name}_{tf_label}.csv"

    if not mt5_path.exists():
        print(f"  MT5 data not found: {mt5_path}")
        print(f"  Run: python mt5/scripts/export_mt5_data.py")
        return None

    # Load OHLCV
    df = pd.read_csv(mt5_path, parse_dates=["datetime"])
    print(f"  Loaded {len(df):,} bars from {mt5_path.name}")

    # Load static datasets
    moon = load_moon_data()
    econ = load_econ_data()

    # Round datetimes to nearest hour for merge
    df["datetime"] = df["datetime"].dt.floor("h")
    moon["datetime"] = moon["datetime"].dt.floor("h")
    econ["datetime"] = econ["datetime"].dt.floor("h")

    # Merge moon phases
    df = df.merge(moon, on="datetime", how="left")

    # Merge economic calendar
    df = df.merge(econ, on="datetime", how="left")

    # Add computed features
    df = add_time_features(df)
    df = add_session_flags(df)
    df = add_technical_features(df)

    # Add static metadata
    df["asset_class"] = info["asset_class"]
    df["instrument"] = name

    # Fill any NaN from merge gaps
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def main():
    print("Building covariate datasets...\n")

    moon = load_moon_data()
    econ = load_econ_data()
    print(f"Moon data: {len(moon):,} hours")
    print(f"Econ data: {len(econ):,} hours\n")

    for name, info in INSTRUMENTS.items():
        print(f"Processing {name}...")
        df = build_covariates_for_instrument(name, info)

        if df is not None:
            tf_label = TIMEFRAME.replace("TIMEFRAME_", "")
            out_path = COV_DIR / f"{name}_{tf_label}_covariates.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df):,} rows, {len(df.columns)} columns → {out_path}")

            # Show columns
            cov_cols = [c for c in df.columns if c not in [
                "datetime", "open", "high", "low", "close",
                "tick_volume", "spread", "asset_class", "instrument"
            ]]
            print(f"  Covariate columns ({len(cov_cols)}): {cov_cols[:10]}...")
        print()

    print("Done. Covariate datasets ready in", COV_DIR)


if __name__ == "__main__":
    main()
