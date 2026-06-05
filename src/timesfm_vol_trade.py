"""Trade forward volatility directly against an implied-vol proxy.

The honest benchmark for a vol forecast is not persistence — it is the market's
implied vol, which already prices persistence and usually embeds a variance risk
premium (VRP). With only spot OHLC data available, this module uses a deliberately
simple proxy:

  implied_proxy[t] = vrp_mult * ewma_vol[t]

This is an optimistic upper bound because real FX implied vol is harder to beat
than an EWMA-based proxy. If a model cannot beat this proxy after costs, it is
unlikely to beat the live market.

At each non-overlapping decision point t:
  edge_pred[t]   = pred_fwd_vol[t]   - implied_proxy[t]
  edge_actual[t] = actual_fwd_vol[t] - implied_proxy[t]

Trading rule:
  edge_pred >  threshold  -> long vol
  edge_pred < -threshold  -> short vol
  otherwise               -> flat

Per-trade P&L in vol points:
  long : actual - implied - cost
  short: implied - actual - cost
  flat : 0

Key metric:
  economic IC = corr(pred - implied, actual - implied)

Run:
  PYTHONPATH=. .venv/bin/python3.12 -m src.timesfm_vol_trade --backend fake --split test
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    .venv/bin/python3.12 -m src.timesfm_vol_trade --split test --device mps \
    --adapter-path data/adapters/eurusd_vol_h96
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.timesfm_eval import _max_drawdown, PIP
from src.timesfm_vol_eval import (
    _load_timesfm,
    _spearman,
    _timesfm_vol_pred,
    build_vol_windows,
)

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
DEFAULT_MANIFEST = "data/split/EURUSD/weekly/manifest.json"
DEFAULT_OUT = "data/predictions/eval"


def _fake_pred(contexts: np.ndarray) -> np.ndarray:
    if len(contexts) == 0:
        return np.empty(0, dtype=np.float64)
    tail = contexts[:, -min(48, contexts.shape[1]):]
    return tail.mean(axis=1).astype(np.float64) * 1.05


def _implied_proxy(ewma: np.ndarray, vrp_mult: float) -> np.ndarray:
    return np.asarray(ewma, dtype=np.float64) * float(vrp_mult)


def _economic_ic(pred: np.ndarray, actual: np.ndarray, implied: np.ndarray) -> float:
    a = np.asarray(pred, dtype=np.float64) - np.asarray(implied, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64) - np.asarray(implied, dtype=np.float64)
    if len(a) == 0:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _trade_pnl(pred: np.ndarray, actual: np.ndarray, implied: np.ndarray, *,
               cost: float, threshold: float = 0.0) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    implied = np.asarray(implied, dtype=np.float64)
    edge = pred - implied
    out = np.zeros(len(pred), dtype=np.float64)
    long_mask = edge > threshold
    short_mask = edge < -threshold
    out[long_mask] = (actual[long_mask] - implied[long_mask]) - cost
    out[short_mask] = (implied[short_mask] - actual[short_mask]) - cost
    return out


def evaluate_strategy(name: str, pred: np.ndarray, actual: np.ndarray, implied: np.ndarray, *,
                      cost: float, threshold: float = 0.0) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    implied = np.asarray(implied, dtype=np.float64)
    pnl = _trade_pnl(pred, actual, implied, cost=cost, threshold=threshold)
    edge = pred - implied
    traded = np.abs(edge) > threshold
    eq = np.cumsum(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    mean = float(np.mean(pnl)) if len(pnl) else float("nan")
    sd = float(np.std(pnl)) if len(pnl) else float("nan")
    sharpe = (mean / sd * math.sqrt(52.0)) if sd > 0 else float("nan")
    return {
        "name": name,
        "economic_ic": round(_economic_ic(pred, actual, implied), 4),
        "spearman_edge": round(_spearman(pred - implied, actual - implied), 4),
        "mean_pnl": round(mean, 8),
        "total_pnl": round(float(np.sum(pnl)), 8),
        "n_trades": int(traded.sum()),
        "trade_rate": round(float(traded.mean()), 4) if len(traded) else 0.0,
        "win_rate": round(float((pnl > 0).mean()), 4) if len(pnl) else float("nan"),
        "payoff_ratio": round(float(payoff), 4) if np.isfinite(payoff) else float("nan"),
        "sharpe_naive": round(float(sharpe), 4) if np.isfinite(sharpe) else float("nan"),
        "max_drawdown": round(float(_max_drawdown(eq)), 8) if len(eq) else 0.0,
        "best": round(float(np.max(pnl)), 8) if len(pnl) else 0.0,
        "worst": round(float(np.min(pnl)), 8) if len(pnl) else 0.0,
        "pnl": pnl,
    }


def _predictions(rv_ctx: np.ndarray, *, backend: str, device: str, adapter_path: str | None,
                 horizon: int, batch_size: int = 64) -> tuple[str, np.ndarray]:
    if backend == "fake":
        return "timesfm_zeroshot", _fake_pred(rv_ctx)
    model = _load_timesfm(device=device, adapter_path=adapter_path)
    tag = "timesfm_finetuned" if adapter_path else "timesfm_zeroshot"
    return tag, _timesfm_vol_pred(model, rv_ctx, horizon, batch_size=batch_size)


def run(data_dir=DEFAULT_DATA_DIR, manifest_path=DEFAULT_MANIFEST, split="test", *,
        backend="timesfm", device="cpu", adapter_path=None, context_len=512, horizon=96,
        window=48, max_files=None, vrp_mults=(1.0, 1.1, 1.2),
        cost_points=(0.0, 1e-5, 2e-5), threshold=0.0, cache_path=None):
    d = build_vol_windows(
        data_dir,
        context_len=context_len,
        horizon=horizon,
        window=window,
        stride=horizon,
        max_files=max_files,
        weeks=None if split == "all" else __import__("src.timesfm_vol_eval", fromlist=["_weeks_for_split"])._weeks_for_split(manifest_path, split),
    )
    rv_ctx = d["rv_ctx"]
    actual = d["actual"]
    persistence = d["persistence"]
    ewma = d["ewma"]
    if len(actual) == 0:
        raise ValueError("no observations found")

    pred_tag = "timesfm_zeroshot"
    model_pred = None
    if cache_path and Path(cache_path).exists() and backend != "fake":
        z = np.load(cache_path, allow_pickle=True)
        if z["pred"].shape[0] == len(actual):
            model_pred = z["pred"]
            pred_tag = "timesfm_finetuned" if adapter_path else "timesfm_zeroshot"
    if model_pred is None:
        pred_tag, model_pred = _predictions(
            rv_ctx,
            backend=backend,
            device=device,
            adapter_path=adapter_path,
            horizon=horizon,
        )
        if cache_path and backend != "fake":
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, pred=model_pred)

    scenarios = {}
    for vrp in vrp_mults:
        implied = _implied_proxy(ewma, vrp)
        for cost in cost_points:
            tag = f"vrp{vrp:.2f}_cost{cost:.6f}"
            scenarios[tag] = {
                "carry": evaluate_strategy("carry", implied * 0.0, actual, implied, cost=cost, threshold=-1.0),
                "ewma": evaluate_strategy("ewma", ewma, actual, implied, cost=cost, threshold=threshold),
                "persistence": evaluate_strategy("persistence", persistence, actual, implied, cost=cost, threshold=threshold),
                pred_tag: evaluate_strategy(pred_tag, model_pred, actual, implied, cost=cost, threshold=threshold),
            }
    return {
        "config": {
            "split": split,
            "backend": backend,
            "adapter_path": adapter_path,
            "context_len": context_len,
            "horizon": horizon,
            "window": window,
            "n_obs": int(len(actual)),
            "vrp_mults": list(vrp_mults),
            "cost_points": list(cost_points),
            "threshold": threshold,
        },
        "scenarios": scenarios,
    }


def _format_stat(v):
    if isinstance(v, float) and not np.isfinite(v):
        return "nan"
    return f"{v}"


def print_report(rep: dict) -> None:
    c = rep["config"]
    print("=" * 118)
    print(
        f"Vol-direct trade test | split={c['split']} obs={c['n_obs']} backend={c['backend']} "
        f"adapter={'yes' if c['adapter_path'] else 'no'} horizon={c['horizon']} window={c['window']}"
    )
    print("WANT: positive economic IC and P&L that beats the carry baseline under VRP and cost sweeps")
    print("=" * 118)
    hdr = f"{'scenario':<23}{'model':<18}{'econIC':>8}{'spIC':>8}{'meanPnL':>12}{'totPnL':>12}{'trades':>8}{'trade%':>8}{'maxDD':>12}"
    print(hdr)
    print("-" * 118)
    for scenario, rows in rep["scenarios"].items():
        first = True
        for model, s in rows.items():
            prefix = scenario if first else ""
            first = False
            print(
                f"{prefix:<23}{model:<18}{_format_stat(s['economic_ic']):>8}{_format_stat(s['spearman_edge']):>8}"
                f"{_format_stat(s['mean_pnl']):>12}{_format_stat(s['total_pnl']):>12}{s['n_trades']:>8}{s['trade_rate']*100:>7.1f}%"
                f"{_format_stat(s['max_drawdown']):>12}"
            )
        print("-" * 118)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Trade forward volatility directly vs implied-vol proxy")
    p.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest-path", dest="manifest_path", default=DEFAULT_MANIFEST)
    p.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--backend", default="timesfm", choices=["timesfm", "fake"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--adapter-path", dest="adapter_path", default=None)
    p.add_argument("--context-len", dest="context_len", type=int, default=512)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--max-files", dest="max_files", type=int, default=None)
    p.add_argument("--vrp-mults", dest="vrp_mults", nargs="+", type=float, default=(1.0, 1.1, 1.2))
    p.add_argument("--cost-points", dest="cost_points", nargs="+", type=float, default=(0.0, 1e-5, 2e-5))
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--cache-path", dest="cache_path", default=None)
    p.add_argument("--out", dest="out_dir", default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cache = a.cache_path
    if cache is None and a.backend != "fake":
        atag = "ft" if a.adapter_path else "zs"
        cache = f"{a.out_dir}/vol_trade_pred_{a.split}_c{a.context_len}_h{a.horizon}_w{a.window}_{atag}.npz"
    rep = run(
        data_dir=a.data_dir,
        manifest_path=a.manifest_path,
        split=a.split,
        backend=a.backend,
        device=a.device,
        adapter_path=a.adapter_path,
        context_len=a.context_len,
        horizon=a.horizon,
        window=a.window,
        max_files=a.max_files,
        vrp_mults=tuple(a.vrp_mults),
        cost_points=tuple(a.cost_points),
        threshold=a.threshold,
        cache_path=cache,
    )
    print_report(rep)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        atag = "ft" if a.adapter_path else ("fake" if a.backend == "fake" else "zs")
        path = Path(a.out_dir) / f"vol_trade_{a.split}_h{a.horizon}_sp{atag}.json"
        serializable = {
            "config": rep["config"],
            "scenarios": {
                scenario: {
                    model: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()}
                    for model, row in rows.items()
                }
                for scenario, rows in rep["scenarios"].items()
            },
        }
        path.write_text(json.dumps(serializable, indent=2))
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
