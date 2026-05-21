"""Download economic calendar events and build hourly feature files.

Fetches actual release dates for major US economic indicators from the
Federal Reserve Economic Data (FRED) REST API. FOMC meeting dates are sourced
from the Federal Reserve website. ECB and European events use approximate
schedules (not available through FRED).

Setup:
  1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
  2. Create  mt5/.env  with:  FRED_API_KEY=your_key_here
     (or set the environment variable directly)

Outputs:
  - mt5/data/economic_calendar/economic_calendar_hourly.csv
  - mt5/data/economic_calendar/event_dates.csv

Run:
    python mt5/scripts/generate_economic_calendar.py
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_END, DATA_START, ECON_DIR

# ── FRED Release IDs for US economic indicators ─────────────────────────
# Actual publication dates downloaded from https://fred.stlouisfed.org/releases
FRED_RELEASES = {
    "CPI_US":          {"release_id": 10,  "impact": "high"},
    "PPI_US":          {"release_id": 11,  "impact": "medium"},
    "GDP_US":          {"release_id": 53,  "impact": "high"},
    "NFP":             {"release_id": 50,  "impact": "high"},
    "Retail_Sales_US": {"release_id": 235, "impact": "medium"},
}

# ── FOMC meeting dates (verified from Federal Reserve website) ───────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# fmt: off
FOMC_DATES = [
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-03", "2001-01-31", "2001-03-20", "2001-04-18", "2001-05-15", "2001-06-27", "2001-08-21", "2001-09-17", "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18",
]
# fmt: on


# ── FRED download ────────────────────────────────────────────────────────

FRED_API_BASE = "https://api.stlouisfed.org/fred"


def _load_api_key() -> str:
    """Load FRED API key from environment or mt5/.env file."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: FRED_API_KEY not found.")
    print(f"  Create {env_path} with:  FRED_API_KEY=your_key_here")
    print("  Or set the environment variable directly.")
    sys.exit(1)


def _fred_get_release_dates(api_key: str, release_id: int) -> list[str]:
    """Fetch all release dates for a FRED release ID via the REST API."""
    all_dates: list[str] = []
    offset = 0
    limit = 1000
    while True:
        params = urllib.parse.urlencode({
            "release_id": release_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": limit,
            "offset": offset,
            "include_release_dates_with_no_data": "false",
        })
        url = f"{FRED_API_BASE}/release/dates?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        dates = data.get("release_dates", [])
        for entry in dates:
            d = entry if isinstance(entry, str) else entry.get("date", "")
            if d:
                all_dates.append(d)
        if len(dates) < limit:
            break
        offset += limit
    return all_dates


def download_fred_events(start: str, end: str) -> list[dict]:
    """Download actual release dates from FRED REST API for US indicators."""
    api_key = _load_api_key()
    events = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    for event_name, info in FRED_RELEASES.items():
        try:
            dates = _fred_get_release_dates(api_key, info["release_id"])
            seen = set()
            count = 0
            for d_str in dates:
                if d_str in seen:
                    continue
                seen.add(d_str)
                if start_dt <= pd.Timestamp(d_str) <= end_dt:
                    events.append({
                        "date": d_str,
                        "event": event_name,
                        "impact": info["impact"],
                    })
                    count += 1
            print(f"  Downloaded {event_name}: {count} dates (FRED release {info['release_id']})")
        except Exception as e:
            print(f"  WARNING: Failed to download {event_name}: {e}")

    return events


# ── Approximate schedule generators (events not on FRED) ─────────────────

def _generate_monthly_event(start: str, end: str, day_of_month: int) -> list[str]:
    """Monthly event on a fixed day, shifted off weekends. (Approximate.)"""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    dates = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month in range(1, 13):
            try:
                d = datetime(year, month, day_of_month)
            except ValueError:
                d = datetime(year, month, min(day_of_month, 28))
            while d.weekday() >= 5:
                d += timedelta(days=1)
            if start_dt <= pd.Timestamp(d) <= end_dt:
                dates.append(d.strftime("%Y-%m-%d"))
    return dates


def _generate_quarterly_event(start: str, end: str, months: list[int], day: int) -> list[str]:
    """Quarterly event on a fixed day, shifted off weekends. (Approximate.)"""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    dates = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month in months:
            try:
                d = datetime(year, month, day)
            except ValueError:
                d = datetime(year, month, min(day, 28))
            while d.weekday() >= 5:
                d += timedelta(days=1)
            if start_dt <= pd.Timestamp(d) <= end_dt:
                dates.append(d.strftime("%Y-%m-%d"))
    return dates


def _generate_weekly_event(start: str, end: str, weekday: int = 3) -> list[str]:
    """Weekly event on a fixed weekday. (Approximate.)"""
    dates = []
    d = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while d <= end_dt:
        if d.weekday() == weekday:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _generate_ecb_dates(start: str, end: str) -> list[str]:
    """ECB decisions: ~first Thursday of month in Jan/Mar/Apr/Jun/Jul/Sep/Oct/Dec. (Approximate.)"""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    ecb_months = [1, 3, 4, 6, 7, 9, 10, 12]
    dates = []
    for year in range(start_dt.year, end_dt.year + 1):
        for month in ecb_months:
            d = datetime(year, month, 1)
            while d.weekday() != 3:  # Thursday
                d += timedelta(days=1)
            if start_dt <= pd.Timestamp(d) <= end_dt:
                dates.append(d.strftime("%Y-%m-%d"))
    return dates


# ── Combine all sources ──────────────────────────────────────────────────

def build_all_events(start: str, end: str) -> pd.DataFrame:
    """Build DataFrame of all economic events from all sources."""
    events = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # ── Downloaded from FRED (accurate) ──────────────────────────────────
    print("Downloading US economic events from FRED...")
    events.extend(download_fred_events(start, end))

    # ── FOMC dates (verified from Federal Reserve website) ───────────────
    fomc_count = 0
    for d in FOMC_DATES:
        if start_dt <= pd.Timestamp(d) <= end_dt:
            events.append({"date": d, "event": "FOMC_Decision", "impact": "high"})
            fomc_count += 1
    print(f"  FOMC_Decision: {fomc_count} dates (hardcoded from Fed website)")

    # ── Approximate events (not available through FRED) ──────────────────
    print("Adding approximate events (ISM, Unemployment Claims, ECB, EU)...")

    for d in _generate_monthly_event(start, end, 1):
        events.append({"date": d, "event": "ISM_Manufacturing", "impact": "medium"})

    for d in _generate_monthly_event(start, end, 3):
        events.append({"date": d, "event": "ISM_Services", "impact": "medium"})

    for d in _generate_weekly_event(start, end, weekday=3):
        events.append({"date": d, "event": "Unemployment_Claims", "impact": "medium"})

    for d in _generate_ecb_dates(start, end):
        events.append({"date": d, "event": "ECB_Decision", "impact": "high"})

    for d in _generate_monthly_event(start, end, 17):
        events.append({"date": d, "event": "CPI_EU", "impact": "medium"})

    for d in _generate_quarterly_event(start, end, [2, 5, 8, 11], 14):
        events.append({"date": d, "event": "GDP_EU", "impact": "medium"})

    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_hourly_calendar(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an hourly DataFrame with binary event flags."""
    print(f"\nBuilding economic calendar from {start} to {end}...\n")

    events_df = build_all_events(start, end)
    dt_range = pd.date_range(start=start, end=end, freq="h")
    hourly = pd.DataFrame({"datetime": dt_range})

    event_types = sorted(events_df["event"].unique())
    for evt in event_types:
        hourly[f"econ_{evt}"] = 0.0

    hourly["econ_any_high"] = 0.0
    hourly["econ_any_medium"] = 0.0
    hourly["econ_any_event"] = 0.0

    for _, row in events_df.iterrows():
        date = row["date"].normalize()
        mask = hourly["datetime"].dt.normalize() == date
        hourly.loc[mask, f"econ_{row['event']}"] = 1.0

        if row["impact"] == "high":
            hourly.loc[mask, "econ_any_high"] = 1.0
        hourly.loc[mask, "econ_any_medium"] = 1.0
        hourly.loc[mask, "econ_any_event"] = 1.0

    events_df_high = events_df[events_df["impact"] == "high"]["date"].drop_duplicates().sort_values().values

    def days_to_next_event(dt, event_dates):
        future = event_dates[event_dates > np.datetime64(dt)]
        if len(future) == 0:
            return np.nan
        return (future[0] - np.datetime64(dt)) / np.timedelta64(1, "D")

    hourly["econ_days_to_high_event"] = hourly["datetime"].apply(
        lambda x: days_to_next_event(x, events_df_high)
    )
    hourly["econ_days_to_high_event"] = hourly["econ_days_to_high_event"] / 30.0

    return hourly, events_df


def main():
    hourly, events_df = build_hourly_calendar(DATA_START, DATA_END)

    out_path = ECON_DIR / "economic_calendar_hourly.csv"
    hourly.to_csv(out_path, index=False)
    print(f"\nSaved {len(hourly):,} rows to {out_path}")

    events_path = ECON_DIR / "event_dates.csv"
    events_df.to_csv(events_path, index=False)
    print(f"Saved {len(events_df):,} events to {events_path}")

    print(f"\nEvent counts:")
    print(events_df["event"].value_counts().to_string())

    fred_events = set(FRED_RELEASES.keys()) | {"FOMC_Decision"}
    approx_events = set(events_df["event"].unique()) - fred_events
    print(f"\nAccurate (FRED/Fed):  {sorted(fred_events)}")
    print(f"Approximate:          {sorted(approx_events)}")


if __name__ == "__main__":
    main()
