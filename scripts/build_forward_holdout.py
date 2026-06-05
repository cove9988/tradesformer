#!/usr/bin/env python3
"""Build an out-of-time forward-holdout set from freshly downloaded MT5 5-min data.

Reuses the EXACT production feature pipeline (src.timesfm_features /
src.data_processor) so the engineered 39-column weekly files are byte-for-byte
schema-identical to the training splits -- no reimplementation, no leakage risk.

The new raw files (tab-separated, no header: datetime, O, H, L, C, volume) live in
data/raw/20250203-20260605/ and cover 2025-02-03 .. 2026-06-05, which is strictly
AFTER every existing training file (latest 2025-01-13). Output goes to a SEPARATE
directory so the existing clean splits in data/split/<SYM>/weekly are never touched.

Output: data/split/forward/<SYMBOL>/weekly/test/<SYMBOL>_<isoyear>_<isoweek>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_processor import (
    patch_missing_data,
    build_timesfm_dataset,
    split_timesfm_weeks,
)
from src.util.read_config import EnvConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("build_forward_holdout")


# map raw filename stem (e.g. EURUSD5) -> config symbol
def _symbol_from_stem(stem: str) -> str:
    return stem[:-1] if stem.endswith("5") else stem


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a headerless MT5 export, auto-detecting tab vs comma delimiter."""
    sample = path.read_text().splitlines()[0]
    sep = "\t" if "\t" in sample else ","
    df = pd.read_csv(path, sep=sep, header=None)
    return df


def build_one(raw_path: Path, cf: EnvConfig, out_root: Path) -> dict:
    symbol = _symbol_from_stem(raw_path.stem)
    if symbol not in cf.config["symbol"]:
        raise KeyError(f"{symbol} not in configure.json symbol table")

    df = _read_raw(raw_path)
    n_raw = len(df)

    # patch_missing_data assigns required_cols(+vol) by position, reindexes to a
    # continuous 5-min UTC grid, ffills OHLC, and drops weekend bars.
    df = patch_missing_data(df, cf=cf)

    # Engineer the causal 39-col feature set (identical fns used for training).
    df = build_timesfm_dataset(df, symbol=symbol, cf=cf)

    # Redirect the weekly writer to the forward dir WITHOUT mutating configure.json
    # on disk: split_timesfm_weeks reads base_path/train_dir from the in-memory cfg.
    base_path = str(out_root / symbol / "weekly")
    cf.config["data_processing"]["train_eval_split"]["base_path"] = base_path
    cf.config["data_processing"]["train_eval_split"]["train_dir"] = "test"

    out_dir = Path(base_path) / "test"
    before = set(p.name for p in out_dir.glob("*.csv")) if out_dir.exists() else set()
    split_timesfm_weeks(df, symbol=symbol, cf=cf)
    after = sorted(p for p in out_dir.glob("*.csv"))
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
    p.add_argument("--raw-dir", default="data/raw/20250203-20260605")
    p.add_argument("--out-root", default="data/split/forward")
    p.add_argument("--config", default="./src/configure.json")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="restrict to these symbols (e.g. EURUSD). default: all in raw-dir")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_root)
    raw_files = sorted(raw_dir.glob("*.csv"))
    if args.symbols:
        want = set(s.upper() for s in args.symbols)
        raw_files = [f for f in raw_files if _symbol_from_stem(f.stem) in want]
    if not raw_files:
        print(f"No raw csv files found in {raw_dir}", file=sys.stderr)
        return 1

    cf = EnvConfig(args.config)
    rows = []
    for rf in raw_files:
        # fresh config object per pair so base_path overrides don't bleed
        cf_i = EnvConfig(args.config)
        try:
            res = build_one(rf, cf_i, out_root)
            rows.append(res)
            print(f"[ok] {res['symbol']:8s} raw={res['raw_rows']:>7d}  "
                  f"weeks_written={res['weeks_written']:>3d}  "
                  f"range={res['first_week']} .. {res['last_week']}")
        except Exception as e:
            print(f"[FAIL] {rf.name}: {e}", file=sys.stderr)
            log.exception("failed on %s", rf)
    print(f"\nDone. {len(rows)}/{len(raw_files)} pairs processed into {out_root}/<SYM>/weekly/test/")
    return 0 if len(rows) == len(raw_files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
