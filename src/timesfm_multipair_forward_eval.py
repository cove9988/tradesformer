"""TEST 2 forward evaluation: does pooled multi-pair vol fine-tuning generalize?

Runs a 3 x 5 matrix on the 2025-26 forward sets (data/split/forward/<PAIR>/weekly/test):

  adapters (rows):
    zs        -> zero-shot TimesFM (no adapter)            = the floor
    eurusd_ft -> EURUSD-only adapter (TEST 1)              = cross-pair control
    multi_ft  -> pooled 5-pair adapter (TEST 2)            = the thing under test

  pairs (cols): EURUSD GBPUSD USDCAD USDCHF USDJPY

For each cell we compute the honest economic metric: economic IC =
corr(pred - implied_proxy, actual - implied_proxy), plus net P&L after a
realistic cost, vs the EWMA implied-vol proxy (timesfm_vol_trade's benchmark).

Efficiency: each adapter is loaded ONCE and reused across all 5 pairs (3 model
loads, not 15). Forward windows are built once per pair and cached in-process.

The verdict logic: multi_ft only "wins" if it (a) beats the zero-shot floor on
economic IC for the pairs it was trained on, AND (b) does so out-of-sample on
the 2025-26 forward set without the EURUSD-FT control already explaining the lift.

Run:
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    .venv/bin/python3.12 -m src.timesfm_multipair_forward_eval --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.timesfm_vol_eval import build_vol_windows, _load_timesfm, _timesfm_vol_pred
from src.timesfm_vol_trade import _implied_proxy, evaluate_strategy

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]
FORWARD_TMPL = "data/split/forward/{pair}/weekly/test"

ADAPTERS = {
    "zs": None,
    "eurusd_ft": "data/adapters/eurusd_vol_h96",
    "multi_ft": "data/adapters/multipair_vol_h96",
}


def build_all_windows(pairs, context_len, horizon, window):
    """Build forward vol windows once per pair. Returns {pair: dict}."""
    out = {}
    for pair in pairs:
        d = build_vol_windows(
            FORWARD_TMPL.format(pair=pair),
            context_len=context_len, horizon=horizon, window=window,
            stride=horizon, max_files=None, weeks=None)
        if len(d["actual"]) == 0:
            raise ValueError(f"no forward observations for {pair}")
        out[pair] = d
    return out


def eval_cell(d, pred, vrp_mult, cost, threshold):
    """One adapter x pair cell: economic metrics vs implied proxy."""
    actual = d["actual"]
    ewma = d["ewma"]
    implied = _implied_proxy(ewma, vrp_mult)
    model_row = evaluate_strategy("model", pred, actual, implied,
                                  cost=cost, threshold=threshold)
    carry = evaluate_strategy("carry", implied * 0.0, actual, implied,
                              cost=cost, threshold=-1.0)
    return model_row, carry


def main(argv=None):
    p = argparse.ArgumentParser(description="TEST 2 multi-pair forward eval matrix")
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--device", default="mps")
    p.add_argument("--context-len", dest="context_len", type=int, default=512)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--vrp-mult", dest="vrp_mult", type=float, default=1.0)
    p.add_argument("--cost", type=float, default=1e-5)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--out", default="data/predictions/forward/multipair_test2_matrix.json")
    a = p.parse_args(argv)

    print(f"Building forward windows for {len(a.pairs)} pairs ...")
    win = build_all_windows(a.pairs, a.context_len, a.horizon, a.window)
    for pair in a.pairs:
        print(f"  {pair}: n_obs={len(win[pair]['actual'])}")

    # adapter -> pair -> metrics
    results = {}
    for aname, apath in ADAPTERS.items():
        if apath is not None and not Path(apath).exists():
            print(f"[skip] adapter {aname} missing at {apath}")
            continue
        print(f"\n=== Loading model: {aname} (adapter={apath}) ===")
        model = _load_timesfm(device=a.device, adapter_path=apath)
        results[aname] = {}
        for pair in a.pairs:
            d = win[pair]
            pred = _timesfm_vol_pred(model, d["rv_ctx"], a.horizon)
            model_row, carry = eval_cell(d, pred, a.vrp_mult, a.cost, a.threshold)
            results[aname][pair] = {
                "economic_ic": model_row["economic_ic"],
                "spearman_edge": model_row["spearman_edge"],
                "mean_pnl": model_row["mean_pnl"],
                "total_pnl": model_row["total_pnl"],
                "sharpe_naive": model_row["sharpe_naive"],
                "n_trades": model_row["n_trades"],
                "trade_rate": model_row["trade_rate"],
                "max_drawdown": model_row["max_drawdown"],
                "carry_total_pnl": carry["total_pnl"],
                "n_obs": int(len(d["actual"])),
            }
            print(f"  {aname:>10} {pair}: econIC={model_row['economic_ic']:+.4f} "
                  f"spIC={model_row['spearman_edge']:+.4f} "
                  f"netPnL={model_row['total_pnl']:+.6f} "
                  f"sharpe={model_row['sharpe_naive']} "
                  f"trades={model_row['n_trades']}")
        del model

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "config": {"pairs": a.pairs, "horizon": a.horizon, "window": a.window,
                   "context_len": a.context_len, "vrp_mult": a.vrp_mult,
                   "cost": a.cost, "threshold": a.threshold},
        "results": results,
    }, indent=2))
    print(f"\nSaved -> {a.out}")

    # ---- verdict table: econIC lift vs zero-shot floor ----
    if "zs" in results and "multi_ft" in results:
        print("\n" + "=" * 78)
        print("ECONOMIC IC: zero-shot floor vs EURUSD-FT vs MULTI-FT (forward 2025-26)")
        print("=" * 78)
        print(f"{'pair':<9}{'zs':>10}{'eurusd_ft':>12}{'multi_ft':>11}"
              f"{'multi-zs':>11}{'multi-eur':>11}")
        print("-" * 78)
        wins = 0
        for pair in a.pairs:
            zs = results["zs"][pair]["economic_ic"]
            eur = results.get("eurusd_ft", {}).get(pair, {}).get("economic_ic", float("nan"))
            mlt = results["multi_ft"][pair]["economic_ic"]
            d_zs = mlt - zs
            d_eur = mlt - eur
            if d_zs > 0:
                wins += 1
            print(f"{pair:<9}{zs:>+10.4f}{eur:>+12.4f}{mlt:>+11.4f}"
                  f"{d_zs:>+11.4f}{d_eur:>+11.4f}")
        print("-" * 78)
        print(f"multi_ft beats zero-shot floor on econIC for {wins}/{len(a.pairs)} pairs")


if __name__ == "__main__":
    main()
