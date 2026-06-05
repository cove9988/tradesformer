"""Does TimesFM 2.5 forecast EURUSD *volatility*? (the forecastable part of FX)

Direction is ~unpredictable (see timesfm_eval.py). Volatility is persistent and
clustered, so it's the natural place a foundation model could add tradeable value
(position sizing, vol-targeting, regime filters). This script tests whether
zero-shot TimesFM beats the standard hard-to-beat baselines on forward realized
vol.

Setup
-----
trailing realized vol:  rv[t]    = sqrt(mean(r[t-W:t]^2))      (W-bar trailing)
forward  realized vol:  fwd[t]   = sqrt(mean(r[t:t+H]^2))      (target, H-bar ahead)

Predictors of fwd[t]:
  persistence   -> rv[t]                       (last trailing vol; very strong)
  ewma          -> EWMA(r^2) sqrt              (RiskMetrics-style)
  climatology   -> rolling mean of rv          (unconditional level)
  timesfm       -> feed rv[t-C:t] series, forecast H steps, take RMS of forecast
                   (TimesFM predicts the future trailing-vol trajectory)

Metrics (higher rank-corr / lower QLIKE = better):
  spearman IC   -> rank corr(pred, actual fwd vol)   -- usable for vol-targeting
  pearson r
  rmse, mae
  QLIKE         -> mean( actual^2/pred^2 - log(actual^2/pred^2) - 1 )  (vol loss)

Run:
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m src.timesfm_vol_eval --context-len 512 --horizons 48 96 --window 48
  PYTHONPATH=. python -m src.timesfm_vol_eval --backend fake   # baselines only
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

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
MODEL_ID = "google/timesfm-2.5-200m-transformers"
EPS = 1e-12


def _trailing_rv(r: np.ndarray, w: int) -> np.ndarray:
    """sqrt(mean(r^2)) over trailing w bars, per index (NaN until enough history)."""
    sq = r * r
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    rv = np.full(len(r), np.nan)
    for t in range(w, len(r) + 1):
        rv[t - 1] = math.sqrt(max((csum[t] - csum[t - w]) / w, 0.0))
    return rv


def _forward_rv(r: np.ndarray, h: int) -> np.ndarray:
    """sqrt(mean(r[t:t+h]^2)) at each t (NaN when not enough future)."""
    sq = r * r
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    fwd = np.full(len(r), np.nan)
    for t in range(0, len(r) - h + 1):
        fwd[t] = math.sqrt(max((csum[t + h] - csum[t]) / h, 0.0))
    return fwd


def build_vol_windows(data_dir, context_len, horizon, window, stride, max_files=None):
    """Return dict of aligned arrays for every decision point across weekly files."""
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if max_files:
        files = files[:max_files]
    rv_ctx, persistence, ewma, clim, actual_fwd = [], [], [], [], []
    for f in files:
        df = pd.read_csv(f).dropna(subset=["log_return_1"]).reset_index(drop=True)
        r = df["log_return_1"].to_numpy(dtype=np.float64)
        n = len(r)
        rv = _trailing_rv(r, window)
        fwd = _forward_rv(r, horizon)
        # EWMA of squared returns (lambda=0.94 RiskMetrics)
        lam = 0.94
        ew = np.full(n, np.nan)
        var = None
        for t in range(n):
            var = r[t] ** 2 if var is None else lam * var + (1 - lam) * r[t] ** 2
            ew[t] = math.sqrt(max(var, 0.0))
        start = max(context_len, window)
        for t in range(start, n - horizon + 1, stride):
            ctx = rv[t - context_len:t]
            if np.isnan(ctx).any() or np.isnan(fwd[t]) or np.isnan(rv[t - 1]):
                continue
            rv_ctx.append(ctx.copy())
            persistence.append(rv[t - 1])
            ewma.append(ew[t - 1])
            clim.append(np.nanmean(rv[max(0, t - context_len):t]))
            actual_fwd.append(fwd[t])
    return {
        "rv_ctx": np.array(rv_ctx) if rv_ctx else np.empty((0, context_len)),
        "persistence": np.array(persistence),
        "ewma": np.array(ewma),
        "clim": np.array(clim),
        "actual": np.array(actual_fwd),
    }


def _load_timesfm(device="cpu"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    m = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, device_map=device)
    m.eval()
    return m


def _timesfm_vol_pred(model, rv_ctx, horizon, batch_size=64):
    """Forecast the trailing-vol series H steps; predicted fwd vol = RMS of forecast path."""
    import torch
    C = rv_ctx.shape[1]
    out = np.empty(len(rv_ctx))
    with torch.no_grad():
        for i in range(0, len(rv_ctx), batch_size):
            chunk = rv_ctx[i:i + batch_size].astype(np.float32)
            filled = 0
            cur = chunk
            blocks = []
            while filled < horizon:
                x = torch.tensor(cur, dtype=torch.float32)
                o = model(past_values=x, forecast_context_len=cur.shape[1])
                blk = o.mean_predictions.float().cpu().numpy()
                take = min(blk.shape[1], horizon - filled)
                blocks.append(blk[:, :take])
                filled += take
                if filled < horizon:
                    cur = np.concatenate([cur, blk[:, :take]], axis=1)
            path = np.concatenate(blocks, axis=1)[:, :horizon]
            path = np.clip(path, EPS, None)  # vol must be positive
            # predicted forward vol = RMS of the predicted trailing-vol path
            out[i:i + len(chunk)] = np.sqrt((path ** 2).mean(axis=1))
    return out


def _spearman(a, b):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _qlike(actual, pred):
    # QLIKE is unstable when vol -> 0 (quiet FX micro-windows). Floor both at the
    # 5th-percentile of actual vol so the loss reflects real forecast quality,
    # not division by a near-zero denominator.
    floor = max(np.percentile(np.clip(actual, 0, None), 5), EPS)
    a2 = np.clip(actual, floor, None) ** 2
    p2 = np.clip(pred, floor, None) ** 2
    ratio = a2 / p2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _score(name, pred, actual):
    pred = np.asarray(pred, dtype=np.float64)
    # in-sample affine recalibration (pred -> a*pred) isolates RANKING skill from
    # level bias: a = median(actual/pred). Illustrative (single scalar, labelled).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = actual / np.clip(pred, EPS, None)
    scale = float(np.median(ratios[np.isfinite(ratios)])) if np.isfinite(ratios).any() else 1.0
    pred_cal = pred * scale
    return {
        "name": name,
        "spearman_ic": round(_spearman(pred, actual), 4),
        "pearson_r": round(float(np.corrcoef(pred, actual)[0, 1]), 4) if pred.std() > 0 else float("nan"),
        "rmse": math.sqrt(((pred - actual) ** 2).mean()),
        "mae": float(np.abs(pred - actual).mean()),
        "qlike_raw": round(_qlike(actual, pred), 3),
        "cal_scale": round(scale, 3),
        "qlike_cal": round(_qlike(actual, pred_cal), 4),
    }


def run(data_dir=DEFAULT_DATA_DIR, context_len=512, horizons=(48, 96), window=48,
        stride=48, backend="timesfm", max_files=None, batch_size=64, device="cpu",
        out_dir="data/predictions/eval"):
    report = {"config": dict(data_dir=data_dir, context_len=context_len,
              horizons=list(horizons), window=window, stride=stride, backend=backend,
              max_files=max_files), "results": {}}
    model = _load_timesfm(device) if backend == "timesfm" else None
    for H in horizons:
        d = build_vol_windows(data_dir, context_len, H, window, stride, max_files)
        if len(d["actual"]) == 0:
            report["results"][f"H{H}"] = {"error": "no windows"}; continue
        actual = d["actual"]
        scores = [
            _score("persistence", d["persistence"], actual),
            _score("ewma", d["ewma"], actual),
            _score("climatology", d["clim"], actual),
        ]
        if backend == "timesfm":
            tf = _timesfm_vol_pred(model, d["rv_ctx"], H, batch_size)
            scores.append(_score("timesfm", tf, actual))
        report["results"][f"H{H}"] = {
            "n_windows": int(len(actual)),
            "actual_vol_mean": float(f"{actual.mean():.3e}"),
            "scores": scores,
        }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"vol_eval_c{context_len}_w{window}_{backend}.json"
        p.write_text(json.dumps(report, indent=2)); report["_saved_to"] = str(p)
    return report


def print_summary(report):
    c = report["config"]
    print("=" * 84)
    print(f"TimesFM VOLATILITY eval | backend={c['backend']} context={c['context_len']} "
          f"trailing_window={c['window']} stride={c['stride']}")
    print("(lower QLIKE/RMSE = better; higher spearman IC = better; persistence is the baseline to beat)")
    print("=" * 84)
    for hk, hr in report["results"].items():
        if "scores" not in hr:
            print(f"\n[{hk}] {hr.get('error')}"); continue
        H = int(hk[1:])
        print(f"\n[{hk}] forward vol over {H} bars (~{H*5}min)  n={hr['n_windows']}  "
              f"actual_vol_mean={hr['actual_vol_mean']}")
        print(f"{'predictor':<13}{'spearmanIC':>11}{'pearson':>9}{'rmse':>11}{'mae':>11}"
              f"{'qlikeRaw':>10}{'scale':>8}{'qlikeCal':>10}")
        print("-" * 84)
        for s in hr["scores"]:
            pr = s["pearson_r"] if s["pearson_r"] == s["pearson_r"] else float("nan")  # NaN-safe
            print(f"{s['name']:<13}{s['spearman_ic']:>11.4f}{pr:>9.3f}"
                  f"{s['rmse']:>11.3e}{s['mae']:>11.3e}"
                  f"{s['qlike_raw']:>10.2f}{s['cal_scale']:>8.2f}{s['qlike_cal']:>10.4f}")
    if report.get("_saved_to"):
        print(f"\nSaved -> {report['_saved_to']}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TimesFM 2.5 volatility-forecast eval on EURUSD")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--horizons", type=int, nargs="+", default=[48, 96])
    p.add_argument("--window", type=int, default=48, help="trailing-vol window in bars")
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--backend", choices=["timesfm", "fake"], default="timesfm")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    print_summary(run(a.data_dir, a.context_len, tuple(a.horizons), a.window, a.stride,
                      a.backend, a.max_files, a.batch_size, a.device, a.out_dir))


if __name__ == "__main__":
    main()
