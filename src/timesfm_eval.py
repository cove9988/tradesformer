"""Zero-shot TimesFM 2.5 evaluation harness for EURUSD weekly forex strategy.

Answers the question: does TimesFM have *tradeable* skill on EURUSD?

We do NOT try to predict individual 5-min returns (those are ~white noise,
std 2.8e-4, mean 0 -- an MSE-minimiser just predicts 0). Instead we evaluate
DIRECTIONAL skill over a horizon (e.g. will price be higher 1h / 4h from now),
which is what a weekly strategy actually trades on.

Two ways TimesFM can be asked:
  * "price"  framing -- feed the close-price level, forecast future prices,
                        direction = sign(pred_final - last_close).
                        (TimesFM is trained on levels and does RevIN internally.)
  * "return" framing -- feed log_return_1, forecast future returns,
                        direction = sign(sum(pred_returns)).

Baselines for honest comparison:
  * naive      -- always flat-ish: predict 0 move (coin-flip direction).
  * momentum   -- direction = sign of last-H realised return (trend continuation).
  * reversion  -- direction = -sign of last-H realised return.

For each (context_len, horizon) we report, per strategy:
  * directional accuracy (hit rate) + binomial p-value vs 50%
  * information coefficient (Spearman corr of predicted vs actual cumulative return)
  * a long/short backtest: gross & net Sharpe at several spread-cost levels,
    hit rate, mean trade, total log return, max drawdown.

Windows never cross a week boundary (each weekly CSV is processed independently),
matching the project's weekly-split design and avoiding weekend gaps.

CLI:
    python -m src.timesfm_eval --context-len 512 --horizons 12 48 \
        --stride 48 --backend timesfm --out data/predictions/eval

    # fast dry run without the model:
    python -m src.timesfm_eval --backend fake --max-files 10
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
MODEL_ID = "google/timesfm-2.5-200m-transformers"
PIP = 1e-4  # EURUSD 1 pip in price terms


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

@dataclass
class Window:
    file: str
    t: int                 # decision index within the week's series
    last_close: float      # close[t-1], the price we trade from
    context_price: np.ndarray   # close[t-C:t]              (len C)
    context_return: np.ndarray  # log_return_1[t-C:t]       (len C)
    actual_cumret: float        # log(close[t+H-1]/close[t-1]) == sum(logret[t:t+H])
    realized_prev: float        # sum(logret[t-H:t]) -- last-H realised return (for momentum)


def build_windows_for_file(
    csv_path: str | Path,
    context_len: int,
    horizon: int,
    stride: int,
    price_col: str = "close",
    return_col: str = "log_return_1",
) -> list[Window]:
    df = pd.read_csv(csv_path)
    if price_col not in df.columns or return_col not in df.columns:
        return []
    df = df.dropna(subset=[price_col, return_col]).reset_index(drop=True)
    close = df[price_col].to_numpy(dtype=np.float64)
    logret = df[return_col].to_numpy(dtype=np.float64)
    n = len(close)
    windows: list[Window] = []
    # need context_len history AND horizon future; also horizon history for momentum
    start = max(context_len, horizon)
    for t in range(start, n - horizon + 1, stride):
        last_close = close[t - 1]
        actual_cumret = math.log(close[t + horizon - 1] / last_close)
        realized_prev = float(logret[t - horizon:t].sum())
        windows.append(
            Window(
                file=Path(csv_path).name,
                t=t,
                last_close=last_close,
                context_price=close[t - context_len:t].copy(),
                context_return=logret[t - context_len:t].copy(),
                actual_cumret=actual_cumret,
                realized_prev=realized_prev,
            )
        )
    return windows


def build_windows(
    data_dir: str | Path,
    context_len: int,
    horizon: int,
    stride: int,
    max_files: int | None = None,
) -> list[Window]:
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if max_files:
        files = files[:max_files]
    out: list[Window] = []
    for f in files:
        out.extend(build_windows_for_file(f, context_len, horizon, stride))
    return out


# ---------------------------------------------------------------------------
# Predictors -> each returns predicted cumulative log-return per window
# ---------------------------------------------------------------------------

def predict_naive(windows: list[Window]) -> np.ndarray:
    return np.zeros(len(windows))


def predict_momentum(windows: list[Window]) -> np.ndarray:
    # signal proportional to last-H realised return (trend continuation)
    return np.array([w.realized_prev for w in windows])


def predict_reversion(windows: list[Window]) -> np.ndarray:
    return np.array([-w.realized_prev for w in windows])


def _load_timesfm(device: str = "cpu"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, device_map=device)
    model.eval()
    return model


def _timesfm_forecast(model, contexts: np.ndarray, horizon: int, batch_size: int = 64) -> np.ndarray:
    """Return predicted future series [N, horizon] for a stack of contexts [N, C].

    NOTE: TimesFM 2.5 emits a fixed max of 128 forecast steps per call
    (mean_predictions is [B, 128]). For horizon > 128 we autoregressively roll
    the forecast forward, appending each 128-step block back onto the context.
    """
    import torch
    context_len = contexts.shape[1]
    n = contexts.shape[0]
    preds = np.empty((n, horizon), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = contexts[i:i + batch_size].astype(np.float32)
            filled = 0
            cur = chunk
            while filled < horizon:
                x = torch.tensor(cur, dtype=torch.float32)
                out = model(past_values=x, forecast_context_len=cur.shape[1])
                block = out.mean_predictions.float().cpu().numpy()   # [b, 128]
                take = min(block.shape[1], horizon - filled)
                preds[i:i + len(chunk), filled:filled + take] = block[:, :take]
                filled += take
                if filled < horizon:
                    # extend context with the just-predicted block for AR rollout
                    cur = np.concatenate([cur, block[:, :take]], axis=1)
    return preds


def predict_timesfm_price(model, windows: list[Window], horizon: int, batch_size: int = 64) -> np.ndarray:
    """Price framing: forecast future prices, predicted cumret = log(pred_final/last_close)."""
    contexts = np.stack([w.context_price for w in windows])
    fut = _timesfm_forecast(model, contexts, horizon, batch_size)
    last = np.array([w.last_close for w in windows])
    final = fut[:, -1]
    # guard against non-positive forecasts
    final = np.where(final <= 0, last, final)
    return np.log(final / last)


def predict_timesfm_return(model, windows: list[Window], horizon: int, batch_size: int = 64) -> np.ndarray:
    """Return framing: forecast future log returns, predicted cumret = sum."""
    contexts = np.stack([w.context_return for w in windows])
    fut = _timesfm_forecast(model, contexts, horizon, batch_size)
    return fut.sum(axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial test p-value (normal approx for large n, exact-ish small)."""
    if n == 0:
        return 1.0
    # normal approximation with continuity correction
    mu = n * p
    sd = math.sqrt(n * p * (1 - p))
    if sd == 0:
        return 1.0
    z = (abs(k - mu) - 0.5) / sd
    # two-sided; clamp to [0,1] (continuity correction can push erfc slightly >1
    # when k sits exactly at the mean and the -0.5 makes the argument negative)
    return min(1.0, math.erfc(z / math.sqrt(2)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = -math.inf
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return float(mdd)


def evaluate_signal(
    name: str,
    pred_cumret: np.ndarray,
    actual_cumret: np.ndarray,
    bars_per_horizon: int,
    bars_per_year: int = 1440 * 52,
    cost_levels_pips: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> dict:
    n = len(actual_cumret)
    pred_dir = np.sign(pred_cumret)
    actual_dir = np.sign(actual_cumret)

    # directional accuracy over windows where we take a position (pred_dir != 0)
    traded = pred_dir != 0
    n_traded = int(traded.sum())
    if n_traded > 0:
        hits = int((pred_dir[traded] == actual_dir[traded]).sum())
        # treat actual_dir==0 (flat) as a miss; rare with continuous returns
        acc = hits / n_traded
        p_val = _binom_two_sided_p(hits, n_traded, 0.5)
    else:
        acc, p_val, hits = float("nan"), 1.0, 0

    ic = _spearman(pred_cumret, actual_cumret)
    corr = float(np.corrcoef(pred_cumret, actual_cumret)[0, 1]) if n > 2 and pred_cumret.std() > 0 else float("nan")
    mae = float(np.abs(pred_cumret - actual_cumret).mean())
    rmse = float(np.sqrt(((pred_cumret - actual_cumret) ** 2).mean()))

    # backtest: position = sign(pred), pnl = pos*actual_cumret - cost when position changes / held
    pos = pred_dir
    gross_pnl = pos * actual_cumret  # per-window log return contribution
    periods_per_year = bars_per_year / bars_per_horizon  # number of horizons per year

    cost_results = {}
    for pips in cost_levels_pips:
        # round-trip cost per traded window in log-return terms (~pips * PIP)
        cost = pips * PIP * (pos != 0)
        net = gross_pnl - cost
        mean = net.mean()
        sd = net.std()
        sharpe = (mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
        equity = np.cumsum(net)
        cost_results[f"{pips}pip"] = {
            "sharpe_ann": round(sharpe, 3),
            "mean_trade": float(f"{mean:.3e}"),
            "total_logret": round(float(net.sum()), 5),
            "hit_rate": round(float((net > 0).mean()), 4),
            "max_drawdown": round(_max_drawdown(equity), 5),
        }

    return {
        "name": name,
        "n_windows": n,
        "n_traded": n_traded,
        "dir_accuracy": None if math.isnan(acc) else round(acc, 4),
        "binom_p_value": round(p_val, 4),
        "information_coef_spearman": None if math.isnan(ic) else round(ic, 4),
        "pearson_corr": None if (isinstance(corr, float) and math.isnan(corr)) else round(corr, 4),
        "mae": float(f"{mae:.3e}"),
        "rmse": float(f"{rmse:.3e}"),
        "backtest_by_cost": cost_results,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir: str = DEFAULT_DATA_DIR,
    context_len: int = 512,
    horizons: tuple[int, ...] = (12, 48),
    stride: int = 48,
    backend: str = "timesfm",
    max_files: int | None = None,
    batch_size: int = 64,
    device: str = "cpu",
    out_dir: str | None = "data/predictions/eval",
) -> dict:
    report: dict = {
        "config": {
            "data_dir": data_dir, "context_len": context_len, "horizons": list(horizons),
            "stride": stride, "backend": backend, "max_files": max_files,
            "model_id": MODEL_ID if backend == "timesfm" else None,
        },
        "results": {},
    }

    model = None
    if backend == "timesfm":
        t0 = time.time()
        model = _load_timesfm(device)
        report["config"]["model_load_s"] = round(time.time() - t0, 1)

    for H in horizons:
        windows = build_windows(data_dir, context_len, H, stride, max_files)
        if not windows:
            report["results"][f"H{H}"] = {"error": "no windows"}
            continue
        actual = np.array([w.actual_cumret for w in windows])

        signals: dict[str, np.ndarray] = {
            "naive_flat": predict_naive(windows),
            "momentum": predict_momentum(windows),
            "reversion": predict_reversion(windows),
        }
        timings: dict[str, float] = {}
        if backend == "timesfm":
            t0 = time.time(); signals["timesfm_price"] = predict_timesfm_price(model, windows, H, batch_size); timings["timesfm_price_s"] = round(time.time() - t0, 1)
            t0 = time.time(); signals["timesfm_return"] = predict_timesfm_return(model, windows, H, batch_size); timings["timesfm_return_s"] = round(time.time() - t0, 1)

        strat_results = [
            evaluate_signal(name, pred, actual, bars_per_horizon=H)
            for name, pred in signals.items()
        ]
        report["results"][f"H{H}"] = {
            "n_windows": len(windows),
            "actual_cumret_std": float(f"{actual.std():.3e}"),
            "timings": timings,
            "strategies": strat_results,
        }

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"eval_c{context_len}_{backend}.json"
        out_path.write_text(json.dumps(report, indent=2))
        report["_saved_to"] = str(out_path)
    return report


def print_summary(report: dict) -> None:
    cfg = report["config"]
    print("=" * 92)
    print(f"TimesFM zero-shot eval | backend={cfg['backend']} context_len={cfg['context_len']} stride={cfg['stride']}")
    print("=" * 92)
    for hkey, hres in report["results"].items():
        if "strategies" not in hres:
            print(f"\n[{hkey}] {hres.get('error')}")
            continue
        H = int(hkey[1:])
        print(f"\n[{hkey}]  horizon={H} bars (~{H*5}min)  n_windows={hres['n_windows']}  actual_cumret_std={hres['actual_cumret_std']}")
        print(f"{'strategy':<16}{'dir_acc':>9}{'p_val':>8}{'IC':>8}{'sharpe@0':>10}{'sharpe@.5p':>11}{'sharpe@1p':>10}{'totRet@.5p':>12}")
        print("-" * 92)
        for s in hres["strategies"]:
            bc = s["backtest_by_cost"]
            acc = s["dir_accuracy"]
            ic = s["information_coef_spearman"]
            print(f"{s['name']:<16}"
                  f"{(f'{acc:.3f}' if acc is not None else 'n/a'):>9}"
                  f"{s['binom_p_value']:>8.3f}"
                  f"{(f'{ic:.3f}' if ic is not None else 'n/a'):>8}"
                  f"{bc['0.0pip']['sharpe_ann']:>10}"
                  f"{bc['0.5pip']['sharpe_ann']:>11}"
                  f"{bc['1.0pip']['sharpe_ann']:>10}"
                  f"{bc['0.5pip']['total_logret']:>12}")
    if report.get("_saved_to"):
        print(f"\nSaved full report -> {report['_saved_to']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot TimesFM 2.5 eval on EURUSD weekly data")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--horizons", type=int, nargs="+", default=[12, 48])
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--backend", choices=["timesfm", "fake"], default="timesfm",
                   help="'fake' skips the model and only runs baselines (pipeline check)")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_evaluation(
        data_dir=args.data_dir, context_len=args.context_len, horizons=tuple(args.horizons),
        stride=args.stride, backend=args.backend, max_files=args.max_files,
        batch_size=args.batch_size, device=args.device, out_dir=args.out_dir,
    )
    print_summary(report)


if __name__ == "__main__":
    main()
