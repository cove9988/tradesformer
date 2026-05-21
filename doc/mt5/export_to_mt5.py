"""Export forecast results as CSV files into MetaTrader 5's MQL5/Files/ directory.

This writes one CSV per instrument so the TimesFM_Forecast.mq5 indicator
can read and draw the forecast + confidence bands on the chart.

Run:
    python mt5/scripts/export_to_mt5.py
    python mt5/scripts/export_to_mt5.py --input mt5/data/forecast_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, INSTRUMENTS

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def find_mt5_files_dir() -> Path:
    """Auto-detect MT5 terminal's MQL5/Files directory."""
    if mt5 is not None:
        if mt5.initialize():
            info = mt5.terminal_info()
            if info is not None:
                files_dir = Path(info.data_path) / "MQL5" / "Files"
                mt5.shutdown()
                return files_dir
            mt5.shutdown()

    # Fallback: scan common locations
    appdata = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    if appdata.exists():
        for terminal_dir in appdata.iterdir():
            candidate = terminal_dir / "MQL5" / "Files"
            if candidate.exists():
                return candidate

    return None


def export_forecasts(results: list[dict], output_dir: Path) -> None:
    """Write per-instrument forecast CSVs."""
    tf_subdir = output_dir / "TimesFM"
    tf_subdir.mkdir(parents=True, exist_ok=True)

    for r in results:
        name = r["instrument"]
        mt5_symbol = INSTRUMENTS.get(name, {}).get("mt5_symbol", name)
        horizon = r["horizon"]
        last_dt = pd.Timestamp(r["last_datetime"])

        # Build future datetimes
        dts = pd.date_range(
            start=last_dt + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )

        df = pd.DataFrame({
            "datetime": dts.strftime("%Y.%m.%d %H:%M"),
            "point": r["point_forecast"],
            "q10": r["quantile_10"],
            "q50": r["quantile_50"],
            "q90": r["quantile_90"],
        })

        csv_path = tf_subdir / f"{mt5_symbol}_forecast.csv"
        df.to_csv(csv_path, index=False)
        print(f"  {mt5_symbol}: {csv_path}")

    # Write a metadata file with last update time
    meta_path = tf_subdir / "last_update.txt"
    meta_path.write_text(pd.Timestamp.now().strftime("%Y.%m.%d %H:%M:%S"))


def main():
    p = argparse.ArgumentParser(description="Export forecasts to MT5 Files directory")
    p.add_argument("--input", default=str(DATA_DIR / "forecast_results.json"))
    p.add_argument("--mt5_dir", default=None, help="Override MT5 MQL5/Files path")
    args = p.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} instrument forecasts")

    if args.mt5_dir:
        output_dir = Path(args.mt5_dir)
    else:
        output_dir = find_mt5_files_dir()
        if output_dir is None:
            print("ERROR: Could not find MT5 MQL5/Files directory.")
            print("  Make sure MetaTrader 5 is installed, or use --mt5_dir <path>")
            sys.exit(1)

    print(f"Writing to: {output_dir / 'TimesFM'}")
    export_forecasts(results, output_dir)
    print("\nDone! Attach the TimesFM_Forecast indicator to your charts in MT5.")


if __name__ == "__main__":
    main()
