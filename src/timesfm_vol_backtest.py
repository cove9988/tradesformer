"""Vol-targeted backtest on EURUSD using TimesFM volatility forecasts.

This answers the only tradeable question left after the direction evals:
not "can TimesFM predict up/down?" (it can't), but "does its volatility edge
improve risk-adjusted P&L when used for position sizing?"

Backtest structure
------------------
Each decision time t opens ONE trade for the next H bars (default non-overlap:
stride=H). The trade direction comes from a simple, transparent entry rule:
  - momentum:  sign(sum of last-H returns)
  - sma_trend: sign(close_vs_sma_288 at t-1)
  - macd:      sign(EMA12-EMA48 at t-1)

The notional size is set by one of these sizing models:
  - flat        : constant size = 1
  - persistence : inverse trailing realized vol
  - ewma        : inverse EWMA volatility
  - climatology : inverse rolling mean trailing vol
  - timesfm     : inverse TimesFM-predicted forward vol

Size formula (classic vol targeting):
    raw_size = 1 / forecast_vol
    size = clip(raw_size / mean(raw_size), 0, max_leverage)

The mean-normalisation keeps average gross exposure ~1x for fair comparison
against flat sizing. This isolates whether BETTER vol forecasts translate into
better Sharpe / drawdown, rather than simply using less leverage.

P&L per trade:
    pnl = direction * size * actual_cumret - spread_cost * abs(size)
where actual_cumret is the H-bar cumulative log return and spread_cost is in
log-return terms (~pips * 1e-4 for EURUSD).

Run:
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m src.timesfm_vol_backtest --horizons 48 96 --window 48 --spread-pips 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_eval import _max_drawdown, PIP
from src.timesfm_vol_eval import _load_timesfm, _timesfm_vol_pred, _trailing_rv, DEFAULT_DATA_DIR


ENTRY_RULES = ("momentum", "sma_trend", "macd")
SIZING_MODELS = ("flat", "persistence", "ewma", "climatology", "timesfm")


def _compute_macd(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema48 = close.ewm(span=48, adjust=False).mean()
    return ema12 - ema48


def build_backtest_windows(data_dir, context_len, horizon, window, stride=None, max_files=None):
    """Aligned windows with entry signals, actual forward returns, and vol inputs."""
    stride = horizon if stride is None else stride
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if max_files:
        files = files[:max_files]

    rv_ctx, persistence, ewma, clim = [], [], [], []
    actual_cumret = []
    entries = {k: [] for k in ENTRY_RULES}

    for f in files:
        df = pd.read_csv(f).dropna(subset=["close", "log_return_1"]).reset_index(drop=True)
        close = df["close"].to_numpy(dtype=np.float64)
        logret = df["log_return_1"].to_numpy(dtype=np.float64)
        close_vs_sma = (
            df["close_vs_sma_288"].to_numpy(dtype=np.float64)
            if "close_vs_sma_288" in df.columns
            else (close / pd.Series(close).rolling(288, min_periods=1).mean().to_numpy(dtype=np.float64) - 1.0)
        )
        macd = _compute_macd(df["close"].astype(np.float64)).to_numpy(dtype=np.float64)

        rv = _trailing_rv(logret, window)
        lam = 0.94
        ew = np.full(len(logret), np.nan)
        var = None
        for t in range(len(logret)):
            var = logret[t] ** 2 if var is None else lam * var + (1 - lam) * logret[t] ** 2
            ew[t] = math.sqrt(max(var, 0.0))

        start = max(context_len, horizon, window)
        for t in range(start, len(close) - horizon + 1, stride):
            ctx = rv[t - context_len:t]
            if np.isnan(ctx).any() or not np.isfinite(close_vs_sma[t - 1]) or not np.isfinite(macd[t - 1]):
                continue

            rv_ctx.append(ctx.copy())
            persistence.append(rv[t - 1])
            ewma.append(ew[t - 1])
            clim.append(np.nanmean(rv[max(0, t - context_len):t]))
            actual_cumret.append(math.log(close[t + horizon - 1] / close[t - 1]))

            entries["momentum"].append(np.sign(logret[t - horizon:t].sum()))
            entries["sma_trend"].append(np.sign(close_vs_sma[t - 1]))
            entries["macd"].append(np.sign(macd[t - 1]))

    return {
        "rv_ctx": np.stack(rv_ctx) if rv_ctx else np.empty((0, context_len)),
        "persistence": np.array(persistence),
        "ewma": np.array(ewma),
        "climatology": np.array(clim),
        "actual_cumret": np.array(actual_cumret),
        "entries": {k: np.array(v) for k, v in entries.items()},
        "stride": stride,
    }


def make_vol_target_size(forecast_vol: np.ndarray, max_leverage: float = 3.0, floor_quantile: float = 5.0) -> np.ndarray:
    """Inverse-vol size with robust floor and mean-normalization (~1x average gross)."""
    fv = np.asarray(forecast_vol, dtype=np.float64)
    if len(fv) == 0:
        return fv.copy()
    floor = max(np.percentile(np.clip(fv[np.isfinite(fv)], 0, None), floor_quantile), 1e-12)
    raw = 1.0 / np.clip(fv, floor, None)
    size = raw / raw.mean()
    return np.clip(size, 0.0, max_leverage)


def backtest_positions(direction: np.ndarray, size: np.ndarray, actual_cumret: np.ndarray,
                       horizon: int, spread_pips: float = 0.5, bars_per_year: int = 1440 * 52) -> dict:
    direction = np.asarray(direction, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    actual_cumret = np.asarray(actual_cumret, dtype=np.float64)
    pos = direction * size
    net = pos * actual_cumret - abs(pos) * (spread_pips * PIP)
    mean = net.mean()
    sd = net.std()
    periods_per_year = bars_per_year / horizon
    sharpe = (mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
    equity = np.cumsum(net)
    return {
        "n_trades": int(len(net)),
        "participation_rate": round(float((direction != 0).mean()), 4),
        "avg_gross_leverage": round(float(np.mean(abs(pos))), 4),
        "mean_trade": float(f"{mean:.3e}"),
        "total_logret": round(float(net.sum()), 5),
        "sharpe_ann": round(float(sharpe), 3),
        "hit_rate": round(float((net > 0).mean()), 4),
        "max_drawdown": round(_max_drawdown(equity), 5),
    }


def run_backtest(data_dir=DEFAULT_DATA_DIR, context_len=512, horizons=(48, 96), window=48,
                 stride=None, spread_pips=0.5, backend="timesfm", max_files=None,
                 batch_size=64, device="cpu", max_leverage=3.0,
                 out_dir="data/predictions/eval"):
    report = {
        "config": {
            "data_dir": data_dir,
            "context_len": context_len,
            "horizons": list(horizons),
            "window": window,
            "stride": stride,
            "spread_pips": spread_pips,
            "backend": backend,
            "max_files": max_files,
            "max_leverage": max_leverage,
        },
        "results": {},
    }

    model = _load_timesfm(device) if backend == "timesfm" else None

    for H in horizons:
        d = build_backtest_windows(data_dir, context_len, H, window, stride, max_files)
        n = len(d["actual_cumret"])
        if n == 0:
            report["results"][f"H{H}"] = {"error": "no windows"}
            continue

        vol_preds = {
            "flat": np.ones(n, dtype=np.float64),
            "persistence": d["persistence"],
            "ewma": d["ewma"],
            "climatology": d["climatology"],
        }
        if backend == "timesfm":
            vol_preds["timesfm"] = _timesfm_vol_pred(model, d["rv_ctx"], H, batch_size)
        else:
            vol_preds["timesfm"] = d["persistence"].copy()

        size_by_model = {
            "flat": np.ones(n, dtype=np.float64),
            "persistence": make_vol_target_size(vol_preds["persistence"], max_leverage=max_leverage),
            "ewma": make_vol_target_size(vol_preds["ewma"], max_leverage=max_leverage),
            "climatology": make_vol_target_size(vol_preds["climatology"], max_leverage=max_leverage),
            "timesfm": make_vol_target_size(vol_preds["timesfm"], max_leverage=max_leverage),
        }

        entry_results = {}
        for entry_name, direction in d["entries"].items():
            rows = []
            for sizing_name in SIZING_MODELS:
                rows.append({
                    "entry": entry_name,
                    "sizing": sizing_name,
                    **backtest_positions(direction, size_by_model[sizing_name], d["actual_cumret"], H, spread_pips),
                })
            entry_results[entry_name] = rows

        report["results"][f"H{H}"] = {
            "n_windows": n,
            "actual_cumret_std": float(f"{d['actual_cumret'].std():.3e}"),
            "actual_vol_mean": float(f"{d['persistence'].mean():.3e}"),
            "stride_used": d["stride"],
            "entries": entry_results,
        }

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"vol_backtest_c{context_len}_sp{spread_pips}_{backend}.json"
        p.write_text(json.dumps(report, indent=2))
        report["_saved_to"] = str(p)
    return report


def print_summary(report):
    c = report["config"]
    stride_label = c["stride"] if c["stride"] is not None else "horizon(non-overlap)"
    print("=" * 104)
    print(f"TimesFM VOL-TARGETED BACKTEST | backend={c['backend']} context={c['context_len']} window={c['window']} "
          f"spread={c['spread_pips']} pip stride={stride_label} maxLev={c['max_leverage']}")
    print("Entry gives direction; sizing controls notional. Goal: better Sharpe / drawdown than flat sizing.")
    print("=" * 104)
    for hk, hr in report["results"].items():
        if "entries" not in hr:
            print(f"\n[{hk}] {hr.get('error')}")
            continue
        H = int(hk[1:])
        print(f"\n[{hk}] horizon={H} bars (~{H*5}min) n={hr['n_windows']} stride={hr['stride_used']} "
              f"actual_cumret_std={hr['actual_cumret_std']}")
        for entry_name, rows in hr["entries"].items():
            print(f"\n  Entry: {entry_name}")
            print(f"  {'sizing':<12}{'sharpe':>9}{'totRet':>10}{'maxDD':>9}{'hit':>8}{'avgLev':>9}{'partic':>8}")
            print("  " + "-" * 72)
            for r in sorted(rows, key=lambda x: (x['sharpe_ann'] if x['sharpe_ann'] == x['sharpe_ann'] else -999), reverse=True):
                print(f"  {r['sizing']:<12}{r['sharpe_ann']:>9}{r['total_logret']:>10}{r['max_drawdown']:>9}"
                      f"{r['hit_rate']:>8}{r['avg_gross_leverage']:>9}{r['participation_rate']:>8}")
    if report.get('_saved_to'):
        print(f"\nSaved -> {report['_saved_to']}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Vol-targeted backtest using TimesFM vol forecasts")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--horizons", type=int, nargs="+", default=[48, 96])
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--stride", type=int, default=None, help="default=None => use horizon (non-overlap)")
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--backend", choices=["timesfm", "fake"], default="timesfm")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    report = run_backtest(a.data_dir, a.context_len, tuple(a.horizons), a.window,
                          a.stride, a.spread_pips, a.backend, a.max_files,
                          a.batch_size, a.device, a.max_leverage, a.out_dir)
    print_summary(report)


if __name__ == "__main__":
    main()
