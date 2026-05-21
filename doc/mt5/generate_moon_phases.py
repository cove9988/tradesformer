"""Generate moon phase data from 2000-01-01 to 2026-04-08.

Outputs:
  - mt5/data/moon_phases/moon_phases_hourly.csv
    Columns: datetime, phase_pct, phase_sin, phase_cos,
             is_new_moon, is_first_quarter, is_full_moon, is_last_quarter,
             days_to_new, days_to_full

Uses the `ephem` library for precise astronomical calculations.

Run:
    python mt5/scripts/generate_moon_phases.py
"""

import sys
from pathlib import Path

import ephem
import numpy as np
import pandas as pd

# Allow running from repo root or from mt5/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_END, DATA_START, MOON_DIR


def compute_major_phases(start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame of all new/first-quarter/full/last-quarter moon dates."""
    phases = []
    date = ephem.Date(start)
    end_date = ephem.Date(end)

    while date < end_date:
        for phase_fn, name in [
            (ephem.next_new_moon, "new_moon"),
            (ephem.next_first_quarter_moon, "first_quarter"),
            (ephem.next_full_moon, "full_moon"),
            (ephem.next_last_quarter_moon, "last_quarter"),
        ]:
            d = phase_fn(date)
            if d < end_date:
                phases.append({
                    "datetime": ephem.Date(d).datetime(),
                    "phase_name": name,
                })
        date += 30  # advance ~1 month

    df = pd.DataFrame(phases).drop_duplicates(subset=["datetime", "phase_name"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_hourly_moon_data(start: str, end: str) -> pd.DataFrame:
    """Build an hourly DataFrame with continuous moon phase metrics."""
    print(f"Generating hourly moon data from {start} to {end}...")

    # Hourly date range
    dt_range = pd.date_range(start=start, end=end, freq="h")
    records = []

    for dt in dt_range:
        m = ephem.Moon()
        m.compute(dt.strftime("%Y/%m/%d %H:%M:%S"))
        phase_pct = float(m.phase) / 100.0  # 0–1 (0=new, ~1=full, back to 0)
        records.append({
            "datetime": dt,
            "phase_pct": phase_pct,
        })

    df = pd.DataFrame(records)

    # Cyclical encoding
    df["phase_sin"] = np.sin(2 * np.pi * df["phase_pct"])
    df["phase_cos"] = np.cos(2 * np.pi * df["phase_pct"])

    # Get major phase dates
    major = compute_major_phases(start, end)

    # For each major phase, find nearest hour
    for phase_name in ["new_moon", "first_quarter", "full_moon", "last_quarter"]:
        col = f"is_{phase_name}"
        df[col] = 0.0
        phase_dates = major.loc[major["phase_name"] == phase_name, "datetime"]
        for pd_dt in phase_dates:
            # Mark the nearest hour row
            idx = (df["datetime"] - pd_dt).abs().idxmin()
            df.loc[idx, col] = 1.0

    # Days to next new moon and full moon
    new_moon_dates = major.loc[major["phase_name"] == "new_moon", "datetime"].values
    full_moon_dates = major.loc[major["phase_name"] == "full_moon", "datetime"].values

    def days_to_next(dt, event_dates):
        future = event_dates[event_dates > np.datetime64(dt)]
        if len(future) == 0:
            return np.nan
        return (future[0] - np.datetime64(dt)) / np.timedelta64(1, "D")

    df["days_to_new"] = df["datetime"].apply(lambda x: days_to_next(x, new_moon_dates))
    df["days_to_full"] = df["datetime"].apply(lambda x: days_to_next(x, full_moon_dates))

    # Normalize days_to columns to 0–1 range (max ~29.5 days lunar cycle)
    df["days_to_new"] = df["days_to_new"] / 29.53
    df["days_to_full"] = df["days_to_full"] / 29.53

    return df


def main():
    df = build_hourly_moon_data(DATA_START, DATA_END)

    out_path = MOON_DIR / "moon_phases_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")

    # Also save phase event dates for reference
    major = compute_major_phases(DATA_START, DATA_END)
    events_path = MOON_DIR / "moon_phase_events.csv"
    major.to_csv(events_path, index=False)
    print(f"Saved {len(major):,} phase events to {events_path}")

    # Quick stats
    print(f"\nDate range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Full moons:      {int(df['is_full_moon'].sum())}")
    print(f"New moons:       {int(df['is_new_moon'].sum())}")
    print(f"First quarters:  {int(df['is_first_quarter'].sum())}")
    print(f"Last quarters:   {int(df['is_last_quarter'].sum())}")


if __name__ == "__main__":
    main()
