"""Martingale-family backtest on real EURUSD paths — does vol-aware averaging help?

Motivated by the question: "can a smart / volatility-improved martingale beat the
straight strategy?" We test the WHOLE family on actual EURUSD weekly price paths
and report the metrics that expose martingale risk (skew, max drawdown, ruin
events, worst-cycle loss) — NOT just Sharpe, which is blind to negative skew and
looks great right up to the blow-up.

Strategy cycles (path-based, within each week; baskets never straddle weeks)
---------------------------------------------------------------------------
A "cycle" opens a position and manages it until it closes in profit, hits the
add-cap stop, or the week ends. Entry direction is the MEAN-REVERSION (fade) bet
for martingale modes — the only thesis under which averaging-down is coherent —
i.e. d = -sign(last `lookback` returns). Anti-martingale uses the momentum bet.

Modes
-----
  flat        : one fixed-size fade trade per cycle, no adds (baseline)
  martingale  : DOUBLE the lot each time price moves `step` against the basket;
                close the whole basket when unrealized >= base*target (a small
                net profit). `max_adds` caps the doublings; naive = large cap.
  anti        : momentum entry; ADD (pyramid) as price moves IN FAVOR; a single
                adverse step closes the basket. This is what a (faintly) trending
                random walk would favor, if anything.

Volatility controls (the "smart / improved" part of the question)
-----------------------------------------------------------------
  --vol-gate PCT : only START a cycle when trailing realized vol is BELOW its
                   PCT-th percentile (stand aside in high-vol/trend regimes that
                   blow up grids).
  --vol-scale    : grid step & target scale with trailing vol / median vol, so
                   spacing widens in volatile regimes.

P&L accounting (base-notional return units; spread is a tunable cost)
---------------------------------------------------------------------
  - each leg pays spread_pips * PIP on its lot at round trip
  - basket unrealized = sum_i lot_i * d * (price - entry_i)
  - profit close when unrealized >= base_lot * target_pips * PIP
  - forced stop at add-cap: close at current (loss) -> counts as a "ruin event"
  - the trap: target is fixed (~base*target) but spread cost grows with TOTAL lot
    (doubles each add), so deep cycles lose even when they "win", and the capped
    tail loss is enormous.

Run:
  PYTHONPATH=. python -m src.timesfm_martingale_backtest --split all --spread-pips 0.5
  PYTHONPATH=. python -m src.timesfm_martingale_backtest --split test --spread-pips 0.5
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
from src.timesfm_vol_eval import _trailing_rv

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
DEFAULT_MANIFEST = "data/split/EURUSD/weekly/manifest.json"
MODES = ("flat", "martingale", "anti")


def _weeks_for_split(manifest_path, split):
    if not split or split == "all":
        return None
    from src.timesfm_split import load_split_manifest
    m = load_split_manifest(manifest_path)
    weeks = set(m.get("splits", {}).get(split, []))
    if not weeks:
        raise ValueError(f"split '{split}' empty/missing in {manifest_path}")
    return weeks


def simulate_week(close, logret, trailing_vol, *, mode="martingale", step_pips=20.0,
                  target_pips=20.0, max_adds=4, spread_pips=0.5, lookback=24,
                  vol_gate=None, vol_scale=False, leverage_cap=64.0):
    """Run cycles over one week's price path. Returns a list of per-cycle P&Ls
    (base-notional return units) and a parallel list of 'ruin' flags."""
    n = len(close)
    cycles = []
    ruins = []
    vol_med = float(np.nanmedian(trailing_vol[np.isfinite(trailing_vol)])) if np.isfinite(trailing_vol).any() else 0.0
    gate_thresh = None
    if vol_gate is not None:
        finite = trailing_vol[np.isfinite(trailing_vol)]
        if len(finite):
            gate_thresh = float(np.percentile(finite, vol_gate))

    t = max(lookback, 1)
    while t < n - 1:
        v = trailing_vol[t]
        if not np.isfinite(v) or v <= 0:
            t += 1
            continue
        # volatility gate: only start in calm regimes
        if gate_thresh is not None and v > gate_thresh:
            t += 1
            continue
        scale = (v / vol_med) if (vol_scale and vol_med > 0) else 1.0
        step = step_pips * PIP * scale
        target = target_pips * PIP * scale
        recent = logret[t - lookback:t].sum()
        if mode == "anti":
            d = np.sign(recent) or 1.0          # momentum entry
        else:
            d = -(np.sign(recent) or 1.0)       # fade / mean-reversion entry

        legs = [(close[t], 1.0)]                 # (entry_price, lot)
        total_lot = 1.0
        next_trigger = close[t] - d * step       # adverse level for martingale
        fav_trigger = close[t] + d * step        # favorable level for anti
        adds = 0
        cycle_pnl = None
        is_ruin = False
        tt = t + 1
        while tt < n:
            p = close[tt]
            unreal = sum(lot * d * (p - e) for e, lot in legs)
            cost = total_lot * spread_pips * PIP
            if mode == "flat":
                if unreal >= 1.0 * target:
                    cycle_pnl = unreal - cost; break
                if unreal <= -1.0 * target * max(max_adds, 1):
                    cycle_pnl = unreal - cost; is_ruin = True; break
            elif mode == "martingale":
                if unreal >= 1.0 * target:                  # basket recovered
                    cycle_pnl = unreal - cost; break
                if d * (p - next_trigger) <= 0:             # moved another step against
                    if adds >= max_adds or total_lot * 2 > leverage_cap:
                        cycle_pnl = unreal - cost; is_ruin = True; break  # forced stop = blow-up
                    new_lot = total_lot                      # double total
                    legs.append((p, new_lot))
                    total_lot += new_lot
                    adds += 1
                    next_trigger = p - d * step
            elif mode == "anti":
                if d * (p - fav_trigger) >= 0:              # moved in favor -> pyramid
                    if adds < max_adds and total_lot * 2 <= leverage_cap:
                        new_lot = total_lot
                        legs.append((p, new_lot))
                        total_lot += new_lot
                        adds += 1
                        fav_trigger = p + d * step
                if unreal <= -1.0 * target:                 # one adverse target -> bail
                    cycle_pnl = unreal - cost; break
                if unreal >= 1.0 * target * (adds + 1):      # let winners run with adds
                    cycle_pnl = unreal - cost; break
            tt += 1
        if cycle_pnl is None:                                # week ended mid-cycle
            p = close[min(tt, n - 1)]
            unreal = sum(lot * d * (p - e) for e, lot in legs)
            cycle_pnl = unreal - total_lot * spread_pips * PIP
        cycles.append(float(cycle_pnl))
        ruins.append(bool(is_ruin))
        t = tt + 1                                           # next cycle after close
    return cycles, ruins


def _skew(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else float("nan")


def summarize(cycles, ruins, periods_per_year=52.0):
    c = np.asarray(cycles, dtype=np.float64)
    if len(c) == 0:
        return {"n_cycles": 0}
    equity = np.cumsum(c)
    mean, sd = c.mean(), c.std()
    sharpe = (mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
    return {
        "n_cycles": int(len(c)),
        "total_return": round(float(c.sum()), 5),
        "mean_cycle": float(f"{mean:.3e}"),
        "win_rate": round(float((c > 0).mean()), 4),
        "sharpe_naive": round(float(sharpe), 3),
        "skew": round(_skew(c), 3),
        "max_drawdown": round(_max_drawdown(equity), 5),
        "worst_cycle": round(float(c.min()), 5),
        "best_cycle": round(float(c.max()), 5),
        "ruin_events": int(np.sum(ruins)),
        "ruin_rate": round(float(np.mean(ruins)), 4),
    }


def build_paths(data_dir, manifest_path=None, split="all", window=48, max_files=None):
    weeks = _weeks_for_split(manifest_path, split)
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if weeks is not None:
        files = [f for f in files if Path(f).name in weeks]
    if max_files:
        files = files[:max_files]
    out = []
    for f in files:
        df = pd.read_csv(f).dropna(subset=["close", "log_return_1"]).reset_index(drop=True)
        close = df["close"].to_numpy(dtype=np.float64)
        logret = df["log_return_1"].to_numpy(dtype=np.float64)
        tv = _trailing_rv(logret, window)
        out.append((close, logret, tv))
    return out


def run(data_dir=DEFAULT_DATA_DIR, manifest_path=DEFAULT_MANIFEST, split="all",
        modes=MODES, step_pips=20.0, target_pips=20.0, spread_pips=0.5,
        window=48, max_files=None, variants=None):
    paths = build_paths(data_dir, manifest_path, split, window, max_files)
    # (label, mode, kwargs) — the family on trial
    if variants is None:
        variants = [
            ("flat_baseline",        "flat",       dict(max_adds=4)),
            ("martingale_naive",     "martingale", dict(max_adds=12)),
            ("martingale_capped4",   "martingale", dict(max_adds=4)),
            ("martingale_volgated",  "martingale", dict(max_adds=4, vol_gate=50.0)),
            ("martingale_volscaled", "martingale", dict(max_adds=4, vol_gate=50.0, vol_scale=True)),
            ("anti_martingale",      "anti",       dict(max_adds=4)),
        ]
    report = {"config": {"split": split, "n_weeks": len(paths), "spread_pips": spread_pips,
                         "step_pips": step_pips, "target_pips": target_pips, "window": window},
              "variants": {}}
    for label, mode, kw in variants:
        all_cycles, all_ruins = [], []
        for close, logret, tv in paths:
            cyc, ru = simulate_week(close, logret, tv, mode=mode, step_pips=step_pips,
                                    target_pips=target_pips, spread_pips=spread_pips, **kw)
            all_cycles.extend(cyc); all_ruins.extend(ru)
        report["variants"][label] = summarize(all_cycles, all_ruins)
    return report


def print_report(report):
    c = report["config"]
    print("=" * 100)
    print(f"Martingale-family backtest | split={c['split']} weeks={c['n_weeks']} "
          f"spread={c['spread_pips']}pip step={c['step_pips']} target={c['target_pips']}")
    print("Sharpe_naive LIES for these strategies — watch skew (very negative = blow-up risk), "
          "ruin_events, worst_cycle")
    print("=" * 100)
    hdr = (f"{'variant':<22}{'cycles':>7}{'totRet':>9}{'win%':>7}{'Sharpe':>8}"
           f"{'skew':>8}{'maxDD':>9}{'worst':>9}{'ruin#':>7}{'ruin%':>7}")
    print(hdr)
    print("-" * 100)
    for label, s in report["variants"].items():
        if s.get("n_cycles", 0) == 0:
            print(f"{label:<22}  (no cycles)"); continue
        print(f"{label:<22}{s['n_cycles']:>7}{s['total_return']:>9.4f}"
              f"{s['win_rate']*100:>6.1f}%{s['sharpe_naive']:>8.2f}{s['skew']:>8.2f}"
              f"{s['max_drawdown']:>9.4f}{s['worst_cycle']:>9.4f}"
              f"{s['ruin_events']:>7}{s['ruin_rate']*100:>6.1f}%")
    print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Martingale-family backtest on EURUSD")
    p.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest-path", dest="manifest_path", default=DEFAULT_MANIFEST)
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--step-pips", type=float, default=20.0)
    p.add_argument("--target-pips", type=float, default=20.0)
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    report = run(a.data_dir, a.manifest_path, a.split, step_pips=a.step_pips,
                 target_pips=a.target_pips, spread_pips=a.spread_pips,
                 window=a.window, max_files=a.max_files)
    print_report(report)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(a.out_dir) / f"martingale_bt_{a.split}_sp{a.spread_pips}.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
