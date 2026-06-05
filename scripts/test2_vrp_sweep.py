"""TEST 2 VRP-sensitivity: does the vol edge survive a realistic variance risk premium?

At vrp_mult=1.0 the implied proxy = raw EWMA (no variance risk premium) -- the
optimistic upper bound. Real FX implied vol typically sits 5-15% ABOVE realized,
i.e. vrp_mult ~ 1.05-1.15. A vol forecast edge that only exists at vrp=1.0 is an
artifact of an unrealistically cheap short-vol benchmark. This sweeps vrp for the
zero-shot floor and the pooled multi_ft adapter, scoring net PnL after cost.

Loads each model ONCE, caches predictions, scores across the vrp grid.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")
from src.timesfm_multipair_forward_eval import build_all_windows
from src.timesfm_vol_eval import _load_timesfm, _timesfm_vol_pred
from src.timesfm_vol_trade import _implied_proxy, evaluate_strategy

PAIRS = ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]
VRP_GRID = [1.0, 1.05, 1.10, 1.15, 1.20]
COST = 1e-5
ADAPTERS = {"zs": None, "multi_ft": "data/adapters/multipair_vol_h96"}


def main():
    win = build_all_windows(PAIRS, 512, 96, 48)
    preds = {}
    for name, apath in ADAPTERS.items():
        print(f"Loading {name} ...", flush=True)
        model = _load_timesfm(device="mps", adapter_path=apath)
        preds[name] = {p: _timesfm_vol_pred(model, win[p]["rv_ctx"], 96) for p in PAIRS}
        del model

    print("\n=== NET PnL after cost across VRP (avg over 5 pairs) ===")
    print(f"{'model':<10}" + "".join(f"vrp={v:<6}" for v in VRP_GRID))
    for name in ADAPTERS:
        row = []
        for vrp in VRP_GRID:
            tot = 0.0
            for p in PAIRS:
                d = win[p]
                implied = _implied_proxy(d["ewma"], vrp)
                s = evaluate_strategy(name, preds[name][p], d["actual"], implied,
                                      cost=COST, threshold=0.0)
                tot += s["total_pnl"]
            row.append(tot / len(PAIRS))
        print(f"{name:<10}" + "".join(f"{x:>10.5f}" for x in row))

    print("\n=== ECONOMIC IC across VRP (avg over 5 pairs) -- IC is vrp-invariant in sign but shown for completeness ===")
    print(f"{'model':<10}" + "".join(f"vrp={v:<6}" for v in VRP_GRID))
    for name in ADAPTERS:
        row = []
        for vrp in VRP_GRID:
            ics = []
            for p in PAIRS:
                d = win[p]
                implied = _implied_proxy(d["ewma"], vrp)
                s = evaluate_strategy(name, preds[name][p], d["actual"], implied,
                                      cost=COST, threshold=0.0)
                ics.append(s["economic_ic"])
            row.append(float(np.nanmean(ics)))
        print(f"{name:<10}" + "".join(f"{x:>10.4f}" for x in row))

    # Per-pair multi_ft net PnL at the realistic vrp=1.10
    print("\n=== multi_ft per-pair NET PnL at realistic vrp=1.10 (vs carry) ===")
    for p in PAIRS:
        d = win[p]
        implied = _implied_proxy(d["ewma"], 1.10)
        s = evaluate_strategy("multi_ft", preds["multi_ft"][p], d["actual"], implied,
                              cost=COST, threshold=0.0)
        carry = evaluate_strategy("carry", implied * 0.0, d["actual"], implied,
                                  cost=COST, threshold=-1.0)
        print(f"  {p}: net={s['total_pnl']:+.5f} sharpe={s['sharpe_naive']} "
              f"trades={s['n_trades']} carry={carry['total_pnl']:+.5f}")


if __name__ == "__main__":
    main()
