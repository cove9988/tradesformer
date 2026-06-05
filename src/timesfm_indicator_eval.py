"""Do trend indicators (EMA / SMA / MACD) help TimesFM predict DIRECTION?

User hypothesis: a smoothed, trend-carrying indicator series may forecast price
direction better than raw price/returns (which are ~white noise).

TimesFM 2.5 is strictly UNIVARIATE (forward() takes only past_values) — there is
no covariate channel, so we cannot fuse close+EMA+ATR as parallel inputs.
Instead we test each indicator AS the input series: feed TimesFM the indicator's
recent history, forecast it H steps ahead, derive a directional vote, and score
it against the actual future price move. This is the correct univariate test of
"does carrying the historical trend forward help."

Why ATR/realized-vol are NOT here: those are VOLATILITY (magnitude), not
direction. They are handled by src/timesfm_vol_eval.py, where TimesFM already
beats the baselines. ATR's role in a strategy is sizing/stops, not up/down.

Input series tested (all causal — computed only from past closes):
  close            raw price (baseline)                       [level]
  log_return_1     raw per-bar return (baseline)              [return]
  ema_12/48/96     exponential MAs (~1h/4h/8h) — trend carriers [level]
  sma_288          24h simple MA (already in CSV)             [level]
  close_vs_sma_288 price-minus-MA trend oscillator (in CSV)   [oscillator]
  macd_12_48       ema_12 - ema_48 trend-momentum             [oscillator]

Signal -> direction:
  level series      : sign(pred_final - series[t-1])   (is the smoothed line rising?)
  return series     : sign(sum(pred))                  (net forecast drift)
  oscillator series : sign(pred_final)                 (trend forecast positive?)
Continuous signal (for Spearman IC vs actual cumret) = the pre-sign quantity.

Run:
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m src.timesfm_indicator_eval --context-len 512 --horizons 48 96 --stride 48
  PYTHONPATH=. python -m src.timesfm_indicator_eval --backend fake   # baselines only
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_eval import _binom_two_sided_p, _spearman, _timesfm_forecast, _load_timesfm

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"

# series name -> ("level" | "return" | "oscillator")
SERIES_TYPES = {
    "close": "level",
    "log_return_1": "return",
    "ema_12": "level",
    "ema_48": "level",
    "ema_96": "level",
    "sma_288": "level",
    "close_vs_sma_288": "oscillator",
    "macd_12_48": "oscillator",
}


def _compute_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build every candidate input series for one weekly file (all causal)."""
    close = df["close"].astype(np.float64)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema48 = close.ewm(span=48, adjust=False).mean()
    ema96 = close.ewm(span=96, adjust=False).mean()
    sma288 = df["sma_288"].astype(np.float64) if "sma_288" in df.columns \
        else close.rolling(288, min_periods=1).mean()
    cvs = df["close_vs_sma_288"].astype(np.float64) if "close_vs_sma_288" in df.columns \
        else (close / sma288 - 1.0)
    macd = ema12 - ema48
    return {
        "close": close.to_numpy(),
        "log_return_1": df["log_return_1"].astype(np.float64).to_numpy(),
        "ema_12": ema12.to_numpy(),
        "ema_48": ema48.to_numpy(),
        "ema_96": ema96.to_numpy(),
        "sma_288": sma288.to_numpy(),
        "close_vs_sma_288": cvs.to_numpy(),
        "macd_12_48": macd.to_numpy(),
    }


def build_indicator_windows(data_dir, context_len, horizon, stride, max_files=None):
    """Aligned per-window arrays: for each series a context stack + last value,
    plus the actual forward cumulative return and a momentum reference."""
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if max_files:
        files = files[:max_files]
    ctx = {k: [] for k in SERIES_TYPES}
    last = {k: [] for k in SERIES_TYPES}
    actual, momentum = [], []
    for f in files:
        df = pd.read_csv(f).dropna(subset=["close", "log_return_1"]).reset_index(drop=True)
        if len(df) < context_len + horizon:
            continue
        series = _compute_series(df)
        close = series["close"]
        logret = series["log_return_1"]
        n = len(close)
        start = max(context_len, horizon)
        for t in range(start, n - horizon + 1, stride):
            ok = True
            for k, s in series.items():
                seg = s[t - context_len:t]
                if not np.isfinite(seg).all():
                    ok = False
                    break
            if not ok:
                continue
            for k, s in series.items():
                ctx[k].append(s[t - context_len:t].astype(np.float64))
                last[k].append(float(s[t - 1]))
            actual.append(math.log(close[t + horizon - 1] / close[t - 1]))
            momentum.append(float(logret[t - horizon:t].sum()))
    out = {
        "ctx": {k: (np.stack(v) if v else np.empty((0, context_len))) for k, v in ctx.items()},
        "last": {k: np.array(v) for k, v in last.items()},
        "actual": np.array(actual),
        "momentum": np.array(momentum),
    }
    return out


def _signal_from_forecast(name: str, pred: np.ndarray, last_val: np.ndarray) -> np.ndarray:
    """Continuous directional signal for one series (sign of this = the trade)."""
    typ = SERIES_TYPES[name]
    if typ == "return":
        return pred.sum(axis=1)
    if typ == "oscillator":
        return pred[:, -1]                       # forecast trend level
    return pred[:, -1] - last_val                # level: forecast slope


def _fake_forecast(ctx: np.ndarray, horizon: int) -> np.ndarray:
    """Last-value persistence forecast (pipeline check, no model)."""
    return np.repeat(ctx[:, -1:], horizon, axis=1)


def _score_direction(name, signal, actual):
    mask = (signal != 0) & np.isfinite(signal) & np.isfinite(actual)
    s, a = signal[mask], actual[mask]
    n = len(s)
    if n == 0:
        return {"name": name, "n": 0}
    correct = int(np.sum(np.sign(s) == np.sign(a)))
    acc = correct / n
    return {
        "name": name,
        "n": n,
        "dir_acc": round(acc, 4),
        "p_val": round(_binom_two_sided_p(correct, n), 4),
        "ic": round(_spearman(s, a), 4),
    }


def run(data_dir=DEFAULT_DATA_DIR, context_len=512, horizons=(48, 96), stride=48,
        backend="timesfm", max_files=None, batch_size=64, device="cpu",
        out_dir="data/predictions/eval"):
    report = {"config": dict(data_dir=data_dir, context_len=context_len,
              horizons=list(horizons), stride=stride, backend=backend,
              max_files=max_files), "results": {}}
    model = _load_timesfm(device) if backend == "timesfm" else None
    for H in horizons:
        d = build_indicator_windows(data_dir, context_len, H, stride, max_files)
        actual = d["actual"]
        if len(actual) == 0:
            report["results"][f"H{H}"] = {"error": "no windows"}
            continue
        rows = []
        # momentum reference (sign of last-H realised return)
        rows.append(_score_direction("momentum_ref", d["momentum"], actual))
        for name in SERIES_TYPES:
            cstack = d["ctx"][name]
            if backend == "timesfm":
                pred = _timesfm_forecast(model, cstack, H, batch_size)
            else:
                pred = _fake_forecast(cstack, H)
            sig = _signal_from_forecast(name, pred, d["last"][name])
            rows.append(_score_direction(f"tf_{name}", sig, actual))
        report["results"][f"H{H}"] = {
            "n_windows": int(len(actual)),
            "actual_cumret_std": float(f"{actual.std():.3e}"),
            "up_rate": round(float((actual > 0).mean()), 4),
            "scores": rows,
        }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"indicator_eval_c{context_len}_{backend}.json"
        p.write_text(json.dumps(report, indent=2))
        report["_saved_to"] = str(p)
    return report


def print_summary(report):
    c = report["config"]
    print("=" * 78)
    print(f"TimesFM DIRECTION-via-INDICATORS | backend={c['backend']} "
          f"context={c['context_len']} stride={c['stride']}")
    print("dir_acc>0.5 & low p = directional skill; |IC|>0 = rank signal. 50% = coin-flip.")
    print("=" * 78)
    for hk, hr in report["results"].items():
        if "scores" not in hr:
            print(f"\n[{hk}] {hr.get('error')}")
            continue
        H = int(hk[1:])
        print(f"\n[{hk}] forward {H} bars (~{H*5}min)  n={hr['n_windows']}  "
              f"up_rate={hr['up_rate']}  cumret_std={hr['actual_cumret_std']}")
        print(f"{'input series':<22}{'n':>7}{'dir_acc':>9}{'p_val':>8}{'IC':>8}")
        print("-" * 78)
        for s in hr["scores"]:
            if s.get("n", 0) == 0:
                print(f"{s['name']:<22}{'0':>7}  (no signal)")
                continue
            print(f"{s['name']:<22}{s['n']:>7}{s['dir_acc']:>9.4f}{s['p_val']:>8.3f}{s['ic']:>8.4f}")
    if report.get("_saved_to"):
        print(f"\nSaved -> {report['_saved_to']}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TimesFM direction-via-indicators eval (EURUSD)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--horizons", type=int, nargs="+", default=[48, 96])
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--backend", choices=["timesfm", "fake"], default="timesfm")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    print_summary(run(a.data_dir, a.context_len, tuple(a.horizons), a.stride,
                      a.backend, a.max_files, a.batch_size, a.device, a.out_dir))


if __name__ == "__main__":
    main()
