"""Export OHLCV data from MetaTrader 5 for all configured instruments.

Outputs one CSV per instrument in mt5/data/mt5_exports/:
    DJ30_H1.csv, NAS100_H1.csv, XAUUSD_H1.csv, ...

Columns: datetime, open, high, low, close, tick_volume, spread

Requirements:
    - MetaTrader 5 terminal must be running and logged in
    - pip install MetaTrader5

Run:
    python mt5/scripts/export_mt5_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BARS_TO_PULL, INSTRUMENTS, MT5_DIR, TIMEFRAME

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)


def export_instrument(name: str, info: dict) -> pd.DataFrame | None:
    """Pull OHLCV data for one instrument from MT5."""
    symbol = info["mt5_symbol"]

    # Ensure symbol is visible in MarketWatch
    if not mt5.symbol_select(symbol, True):
        print(f"  WARNING: Symbol {symbol} not available in MT5. Skipping.")
        return None

    # Map timeframe string to MT5 constant
    tf_const = getattr(mt5, TIMEFRAME, None)
    if tf_const is None:
        print(f"  ERROR: Unknown timeframe {TIMEFRAME}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, BARS_TO_PULL)
    if rates is None or len(rates) == 0:
        print(f"  WARNING: No data returned for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df[["datetime", "open", "high", "low", "close", "tick_volume", "spread"]]
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def main():
    print("Initializing MetaTrader 5...")
    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        print("Make sure the MetaTrader 5 terminal is running and logged in.")
        sys.exit(1)

    print(f"MT5 connected: {mt5.terminal_info().name}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Max bars: {BARS_TO_PULL:,}\n")

    for name, info in INSTRUMENTS.items():
        print(f"Exporting {name} ({info['mt5_symbol']})...")
        df = export_instrument(name, info)

        if df is not None:
            tf_label = TIMEFRAME.replace("TIMEFRAME_", "")
            out_path = MT5_DIR / f"{name}_{tf_label}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df):,} bars → {out_path}")
            print(f"  Range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        print()

    mt5.shutdown()
    print("Done. MT5 connection closed.")


if __name__ == "__main__":
    main()
