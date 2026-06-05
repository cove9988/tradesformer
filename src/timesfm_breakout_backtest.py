"""Breakout / straddle trend engine on real EURUSD paths — the POSITIVE-skew bet.

The structural opposite of the martingale. Motivated by the variance-ratio finding
that EURUSD has NO mean reversion and a faint TREND at long horizons (H96 VR>1):
the only edge-shaped structure in the data is momentum, so we try to harvest it
the way trend-followers do — and let VOLATILITY decide WHEN to take the breakout.

Thesis (and why skew flips positive)
------------------------------------
  - Donchian breakout: go with the break of the N-bar high/low (straddle = stop
    orders both sides, whichever triggers first sets direction). Harvests the
    faint momentum the variance-ratio test found.
  - ATR chandelier trailing stop: losers are cut fast at ~-1R; the trailing stop
    lets winners run.
  - Pyramid trend-add: press winners (add units as the move extends) so the rare
    sustained trend pays multiples of the initial risk.
  => most cycles are small -1R losses; a few are big +manyR wins. POSITIVE skew —
     the exact opposite of the martingale's "many small wins, one ruin".

Volatility's job here (the actual question)
-------------------------------------------
False breakouts in quiet/contracting regimes are what bleed a breakout system.
The gate decides WHEN a breakout is worth taking. The engine takes a precomputed
per-bar boolean `gate` array so the SAME engine can be driven by:
  Stage 1: realized-vol gates (price-only, free) — does breakout even have a pulse?
  Stage 2: the fine-tuned TimesFM forward-vol-EXPANSION forecast — does the model's
           vol edge improve breakout selection over the naive realized-vol gate?

Intrabar realism / anti-optimism
--------------------------------
  - Entry fills at the channel level, or at the bar OPEN if price gapped through it.
  - Stop fills at the stop level, or at the bar OPEN if it gapped through (worse fill).
  - Same-bar ambiguity: if a bar touches BOTH the stop and a pyramid/extension level,
    the STOP is assumed to fill first (conservative — never optimistic on path).
  - Spread charged on every unit at entry AND exit (and every pyramid add).
  - P&L in R-multiples: R = initial_stop_distance on the first unit. Makes the skew
    and the win/loss asymmetry directly comparable across regimes.

Run:
  PYTHONPATH=. python -m src.timesfm_breakout_backtest --split all --spread-pips 0.5
  PYTHONPATH=. python -m src.timesfm_breakout_backtest --split test --spread-pips 0.5
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

DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
DEFAULT_MANIFEST = "data/split/EURUSD/weekly/manifest.json"


def _weeks_for_split(manifest_path, split):
    if not split or split == "all":
        return None
    from src.timesfm_split import load_split_manifest
    m = load_split_manifest(manifest_path)
    weeks = set(m.get("splits", {}).get(split, []))
    if not weeks:
        raise ValueError(f"split '{split}' empty/missing in {manifest_path}")
    return weeks


def donchian_channels(high, low, n):
    """Trailing N-bar highest-high / lowest-low BELOW the current bar (no lookahead).

    channel_high[t] = max(high[t-n:t]); channel_low[t] = min(low[t-n:t]).
    First n bars are NaN (insufficient history)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    n_bars = len(h)
    ch_hi = np.full(n_bars, np.nan)
    ch_lo = np.full(n_bars, np.nan)
    for t in range(n, n_bars):
        ch_hi[t] = h[t - n:t].max()
        ch_lo[t] = l[t - n:t].min()
    return ch_hi, ch_lo


def simulate_week_breakout(open_, high, low, close, atr, *, gate=None,
                           donchian_n=96, k_stop=2.0, k_trail=3.0, k_add=1.0,
                           max_adds=3, spread_pips=0.5, cooldown=12,
                           min_atr=1e-5):
    """One week's price path -> list of per-cycle R-multiples and metadata.

    gate: optional per-bar bool array; entries only allowed where gate[t] is True.
    Returns (cycle_R_list, n_adds_list)."""
    o = np.asarray(open_, float); h = np.asarray(high, float)
    l = np.asarray(low, float); c = np.asarray(close, float)
    a = np.asarray(atr, float)
    n = len(c)
    ch_hi, ch_lo = donchian_channels(h, l, donchian_n)
    spread = spread_pips * PIP
    cycles = []
    adds_log = []

    t = donchian_n
    while t < n - 1:
        if not np.isfinite(ch_hi[t]) or not np.isfinite(a[t]) or a[t] < min_atr:
            t += 1; continue
        if gate is not None and not gate[t]:
            t += 1; continue

        long_trig = h[t] >= ch_hi[t]
        short_trig = l[t] <= ch_lo[t]
        if not (long_trig or short_trig):
            t += 1; continue
        # straddle: if both sides break in the same bar, take the one the open is
        # already beyond; default long (rare, ~0.1% of bars).
        direction = 1.0 if long_trig else -1.0
        if long_trig and short_trig:
            direction = 1.0 if o[t] >= ch_hi[t] else -1.0

        atr0 = a[t]
        risk = k_stop * atr0                         # initial R in price units
        level = ch_hi[t] if direction > 0 else ch_lo[t]
        # entry fill: at the breakout level, or the open if gapped through
        entry = o[t] if (direction > 0 and o[t] > level) or (direction < 0 and o[t] < level) else level
        legs = [entry]                               # one unit per leg, equal size
        stop = entry - direction * risk
        hw = entry                                   # high-water (favorable extreme)
        next_add = entry + direction * k_add * atr0
        n_units = 1
        exit_price = None

        tt = t + 1
        while tt < n:
            atr_t = a[tt] if (np.isfinite(a[tt]) and a[tt] >= min_atr) else atr0
            # 1) STOP first (conservative): did this bar trade through the stop?
            stop_hit = (l[tt] <= stop) if direction > 0 else (h[tt] >= stop)
            if stop_hit:
                gap = (o[tt] < stop) if direction > 0 else (o[tt] > stop)
                exit_price = o[tt] if gap else stop  # gap-through fills worse
                break
            # 2) favorable extension -> update trail + maybe pyramid
            hw = max(hw, h[tt]) if direction > 0 else min(hw, l[tt])
            trail = hw - direction * k_trail * atr_t
            stop = max(stop, trail) if direction > 0 else min(stop, trail)
            reached_add = (h[tt] >= next_add) if direction > 0 else (l[tt] <= next_add)
            if reached_add and n_units <= max_adds:
                add_fill = next_add
                legs.append(add_fill)
                n_units += 1
                next_add = add_fill + direction * k_add * atr0
            tt += 1
        if exit_price is None:                        # week ended in-position
            exit_price = c[min(tt, n - 1)]

        # R-multiple P&L across all units, spread charged each unit round-trip
        pnl_price = sum(direction * (exit_price - e) - 2 * spread for e in legs)
        cycle_R = pnl_price / risk
        cycles.append(float(cycle_R))
        adds_log.append(n_units - 1)
        t = tt + cooldown                             # cooldown before re-arming
    return cycles, adds_log


def _skew(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return float("nan")
    m, s = x.mean(), x.std()
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else float("nan")


def summarize(cycles, adds, periods_per_year=52.0):
    c = np.asarray(cycles, float)
    if len(c) == 0:
        return {"n_cycles": 0}
    eq = np.cumsum(c)
    mean, sd = c.mean(), c.std()
    wins = c[c > 0]; losses = c[c < 0]
    sharpe = (mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    exp_R = float(mean)                               # expectancy per trade in R
    return {
        "n_cycles": int(len(c)),
        "total_R": round(float(c.sum()), 3),
        "expectancy_R": round(exp_R, 4),              # >0 => positive edge per trade
        "win_rate": round(float((c > 0).mean()), 4),
        "payoff_ratio": round(float(payoff), 3),      # avg win / avg loss
        "skew": round(_skew(c), 3),                   # >0 desired (opposite of martingale)
        "sharpe_naive": round(float(sharpe), 3),
        "max_drawdown_R": round(_max_drawdown(eq), 3),
        "best_R": round(float(c.max()), 3),
        "worst_R": round(float(c.min()), 3),
        "avg_adds": round(float(np.mean(adds)), 3) if adds else 0.0,
    }


def build_paths(data_dir, manifest_path=None, split="all", max_files=None):
    weeks = _weeks_for_split(manifest_path, split)
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if weeks is not None:
        files = [f for f in files if Path(f).name in weeks]
    if max_files:
        files = files[:max_files]
    out = []
    for f in files:
        df = pd.read_csv(f).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        out.append(df)
    return out


def _causal_percentile_gate(rv, pct, donchian_n, above):
    """Causal expanding-percentile gate (NO lookahead).

    At each bar t, the threshold is the pct-th percentile of rv[:t] (strictly past
    bars only). Warmup: no entries until at least `donchian_n` finite past values
    exist (the engine doesn't trade before then anyway). `above=True` -> rv[t] >=
    threshold (highvol); `above=False` -> rv[t] <= threshold (lowvol)."""
    n = len(rv)
    g = np.zeros(n, dtype=bool)
    finite_mask = np.isfinite(rv)
    for t in range(donchian_n, n):
        past = rv[:t][finite_mask[:t]]
        if len(past) < donchian_n or not finite_mask[t]:
            continue
        thr = np.percentile(past, pct)
        g[t] = (rv[t] >= thr) if above else (rv[t] <= thr)
    return g


def make_gate(df, kind, donchian_n=96, pct=50.0, vol_col="realized_vol_96", lag=1):
    """Per-bar entry gate from realized-vol columns (Stage 1, price-only). CAUSAL.

    none      : always allow.
    highvol   : allow when trailing realized vol is ABOVE its causal-expanding
                pct-th percentile (active regimes; avoid dead chop). No lookahead.
    expansion : allow when realized vol is RISING vs `donchian_n` bars ago
                (volatility EXPANSION — the classic breakout precondition). Causal.
    lowvol    : allow when BELOW causal pct-th percentile (control; expect worse).

    lag: shift the gate FORWARD by `lag` bars. realized_vol_96[t] needs close[t],
    but a breakout triggers intrabar during bar t (before close[t] is known), so
    gating entry at t on vol[t] is a mild lookahead. lag=1 gates entry at t on
    vol[t-1] (strictly known) — the honest default."""
    n = len(df)
    if kind == "none":
        return np.ones(n, dtype=bool)
    rv = df[vol_col].to_numpy(float) if vol_col in df.columns else np.full(n, np.nan)
    if kind == "highvol":
        g = _causal_percentile_gate(rv, pct, donchian_n, above=True)
    elif kind == "lowvol":
        g = _causal_percentile_gate(rv, pct, donchian_n, above=False)
    elif kind == "expansion":
        g = np.zeros(n, dtype=bool)
        g[donchian_n:] = (np.isfinite(rv[donchian_n:]) & np.isfinite(rv[:-donchian_n])
                          & (rv[donchian_n:] > rv[:-donchian_n]))
    else:
        return np.zeros(n, dtype=bool)
    if lag > 0:                                   # entry at t uses vol known by t-lag
        g = np.concatenate([np.zeros(lag, dtype=bool), g[:-lag]])
    return g


def run(data_dir=DEFAULT_DATA_DIR, manifest_path=DEFAULT_MANIFEST, split="all",
        donchian_n=96, k_stop=2.0, k_trail=3.0, k_add=1.0, max_adds=3,
        spread_pips=0.5, max_files=None, gates=("none", "highvol", "expansion", "lowvol"),
        vol_col="realized_vol_96", gate_pct=50.0):
    paths = build_paths(data_dir, manifest_path, split, max_files)
    report = {"config": {"split": split, "n_weeks": len(paths), "spread_pips": spread_pips,
                         "donchian_n": donchian_n, "k_stop": k_stop, "k_trail": k_trail,
                         "k_add": k_add, "max_adds": max_adds, "vol_col": vol_col,
                         "gate_pct": gate_pct},
              "gates": {}}
    for gkind in gates:
        all_c, all_a = [], []
        for df in paths:
            gate = make_gate(df, gkind, donchian_n, gate_pct, vol_col)
            c, ad = simulate_week_breakout(
                df["open"].to_numpy(float), df["high"].to_numpy(float),
                df["low"].to_numpy(float), df["close"].to_numpy(float),
                df["atr"].to_numpy(float), gate=gate, donchian_n=donchian_n,
                k_stop=k_stop, k_trail=k_trail, k_add=k_add, max_adds=max_adds,
                spread_pips=spread_pips)
            all_c.extend(c); all_a.extend(ad)
        report["gates"][gkind] = summarize(all_c, all_a)
    return report


def print_report(report):
    c = report["config"]
    print("=" * 104)
    print(f"Breakout/straddle + pyramid + ATR-trail | split={c['split']} weeks={c['n_weeks']} "
          f"spread={c['spread_pips']}pip donchian={c['donchian_n']} stop={c['k_stop']}ATR "
          f"trail={c['k_trail']}ATR adds<={c['max_adds']}")
    print("WANT: expectancy_R>0 AND positive skew (opposite of martingale). vol_col="
          f"{c['vol_col']} gate_pct={c['gate_pct']}")
    print("=" * 104)
    hdr = (f"{'gate':<12}{'cycles':>7}{'totR':>9}{'expR':>9}{'win%':>7}{'payoff':>8}"
           f"{'skew':>8}{'Sharpe':>8}{'maxDD_R':>9}{'best':>7}{'worst':>7}{'adds':>6}")
    print(hdr)
    print("-" * 104)
    for g, s in report["gates"].items():
        if s.get("n_cycles", 0) == 0:
            print(f"{g:<12}  (no cycles)"); continue
        print(f"{g:<12}{s['n_cycles']:>7}{s['total_R']:>9.2f}{s['expectancy_R']:>9.4f}"
              f"{s['win_rate']*100:>6.1f}%{s['payoff_ratio']:>8.2f}{s['skew']:>8.2f}"
              f"{s['sharpe_naive']:>8.2f}{s['max_drawdown_R']:>9.2f}{s['best_R']:>7.1f}"
              f"{s['worst_R']:>7.1f}{s['avg_adds']:>6.2f}")
    print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Breakout/straddle trend backtest on EURUSD")
    p.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest-path", dest="manifest_path", default=DEFAULT_MANIFEST)
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--donchian-n", type=int, default=96)
    p.add_argument("--k-stop", type=float, default=2.0)
    p.add_argument("--k-trail", type=float, default=3.0)
    p.add_argument("--k-add", type=float, default=1.0)
    p.add_argument("--max-adds", type=int, default=3)
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--vol-col", default="realized_vol_96")
    p.add_argument("--gate-pct", type=float, default=50.0)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    report = run(a.data_dir, a.manifest_path, a.split, donchian_n=a.donchian_n,
                 k_stop=a.k_stop, k_trail=a.k_trail, k_add=a.k_add, max_adds=a.max_adds,
                 spread_pips=a.spread_pips, max_files=a.max_files, vol_col=a.vol_col,
                 gate_pct=a.gate_pct)
    print_report(report)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(a.out_dir) / f"breakout_bt_{a.split}_sp{a.spread_pips}_n{a.donchian_n}.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
