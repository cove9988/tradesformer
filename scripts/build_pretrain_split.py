#!/usr/bin/env python3
"""Build a pre-2025 weekly TRAIN split from a headerless MT5 5-min raw export.

This is the *train-side* twin of build_forward_holdout.py. It reuses the EXACT
production feature pipeline (patch_missing_data -> build_timesfm_dataset ->
split_timesfm_weeks) so the engineered 39-column weekly files are byte-for-byte
schema-identical to the existing committed train splits (EURUSD/GBPUSD/USDCAD).

Only two pairs (USDCHF, USDJPY) are missing their pre-2025 train split; the raw
pre-2025 M5 history for them lives at data/raw/<SYMBOL>_M5.csv. Output goes to
the standard location data/split/<SYMBOL>/weekly/train/ (the configure.json
default), so no config override is needed for the writer.

Usage:
  PYTHONPATH=. .venv/bin/python3.12 scripts/build_pretrain_split.py --symbols USDCHF USDJPY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data_processor import (
    patch_missing_data,
    build_timesfm_dataset,
    split_timesfm_weeks,
)
from src.util.read_config import EnvConfig


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a headerless MT5 export, auto-detecting tab vs comma delimiter."""
    sample = path.read_text().splitlines()[0]
    sep = "\t" if "\t" in sample else ","
    return pd.read_csv(path, sep=sep, header=None)


def build_one(symbol: str, raw_path: Path, config_path: str) -> dict:
    cf = EnvConfig(config_path)
    if symbol not in cf.config["symbol"]:
        raise KeyError(f"{symbol} not in configure.json symbol table")

    df = _read_raw(raw_path)
    n_raw = len(df)
    df = patch_missing_data(df, cf=cf)            # reindex 5-min grid, ffill, drop weekends
    df = build_timesfm_dataset(df, symbol=symbol, cf=cf)  # causal 39-col features

    # writer uses configure.json default: base_path=./data/split/{symbol}/weekly, train_dir=train
    out_dir = Path("data/split") / symbol / "weekly" / "train"
    before = {p.name for p in out_dir.glob("*.csv")} if out_dir.exists() else set()
    split_timesfm_weeks(df, symbol=symbol, cf=cf)
    after = sorted(out_dir.glob("*.csv"))
    written = [p for p in after if p.name not in before]
    weeks = sorted(p.name for p in after)
    return {
        "symbol": symbol,
        "raw_rows": n_raw,
        "weeks_written": len(written),
        "weeks_total": len(after),
        "first_week": weeks[0] if weeks else None,
        "last_week": weeks[-1] if weeks else None,
        "out_dir": str(out_dir),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", required=True, help="e.g. USDCHF USDJPY")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--config", default="./src/configure.json")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    ok = 0
    for sym in (s.upper() for s in args.symbols):
        raw_path = raw_dir / f"{sym}_M5.csv"
        if not raw_path.exists():
            print(f"[FAIL] {sym}: raw file not found at {raw_path}", file=sys.stderr)
            continue
        try:
            res = build_one(sym, raw_path, args.config)
            ok += 1
            print(f"[ok] {res['symbol']:8s} raw={res['raw_rows']:>7d}  "
                  f"weeks_written={res['weeks_written']:>3d}  total={res['weeks_total']:>3d}  "
                  f"range={res['first_week']} .. {res['last_week']}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {sym}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    print(f"\nDone. {ok}/{len(args.symbols)} pairs processed.")
    return 0 if ok == len(args.symbols) else 2


if __name__ == "__main__":
    raise SystemExit(main())
