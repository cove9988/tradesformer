"""Stage 2: gate breakouts with the FINE-TUNED TimesFM forward-vol forecast.

Stage 1 (timesfm_breakout_backtest.py) established: EURUSD breakout has the right
SHAPE (positive skew, ~4:1 payoff) and a faint edge that vol-EXPANSION gating
improves (test-set expansion gate +0.13R/trade at 0.5pip) but which DIES by 1.0pip
spread. The naive expansion gate looks BACKWARD (did realized vol rise over the last
N bars). This module asks the honest follow-up:

  Does the model's FORWARD vol forecast select breakouts better than the naive
  backward-looking realized-vol rule?

Mechanism (strictly causal — no lookahead)
------------------------------------------
At each breakout-trigger bar t (h[t]>=channel_high or l[t]<=channel_low), build the
trailing-realized-vol context rv[t-C:t] (its last point rv[t-1] uses returns through
bar t-1 only), forecast the forward trailing-vol path with TimesFM, and reduce it to
a predicted forward vol (RMS of the path) — exactly as the validated vol eval does.
Then gate entry at t on:

  expansion-analog : pred_fwd_vol[t] >  rv[t-1]         (model predicts vol rises)
  highvol-analog   : pred_fwd_vol[t] >= causal median of pred_fwd so far

Comparisons (same validated engine, same weeks, same costs)
-----------------------------------------------------------
  none            : ungated breakout (reference)
  naive_expansion : realized-vol rose over last `donchian_n` bars (Stage 1 best)
  tsfm_zeroshot   : zero-shot TimesFM forward-vol-expansion gate
  tsfm_finetuned  : FINE-TUNED adapter forward-vol-expansion gate  <-- the question

If the fine-tuned gate doesn't beat naive_expansion AND clear spread, the model's
vol skill does not convert into breakout-selection alpha. Honest by construction.

Predictions are cached to data/predictions/eval/ so threshold tuning is free.

Run (heavy — background it):
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    python -m src.timesfm_breakout_vol_gate --split test --device mps \
      --adapter-path data/adapters/eurusd_vol_h96 --context-len 512 --horizon 96 --window 48
  PYTHONPATH=. python -m src.timesfm_breakout_vol_gate --split test --backend fake  # plumbing
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_breakout_backtest import (
    donchian_channels, simulate_week_breakout, summarize, make_gate, _weeks_for_split,
)
from src.timesfm_vol_eval import _trailing_rv


def find_breakout_bars(high, low, donchian_n):
    """Indices where a breakout triggers (channel break), excluding warmup."""
    h = np.asarray(high, float); l = np.asarray(low, float)
    ch_hi, ch_lo = donchian_channels(h, l, donchian_n)
    bars = []
    for t in range(donchian_n, len(h)):
        if np.isfinite(ch_hi[t]) and (h[t] >= ch_hi[t] or l[t] <= ch_lo[t]):
            bars.append(t)
    return bars, ch_hi, ch_lo


def collect_contexts(paths, context_len, window, donchian_n):
    """Gather causal trailing-vol contexts at every breakout-trigger bar.

    Returns (contexts[N,C], index[(week_i, bar_t)], rv_prev[N]) where rv_prev is
    rv[t-1] (trailing vol known at entry) for the expansion comparison."""
    ctxs, index, rv_prev = [], [], []
    for wi, df in enumerate(paths):
        r = df["log_return_1"].to_numpy(float)
        rv = _trailing_rv(r, window)
        bars, _, _ = find_breakout_bars(df["high"].to_numpy(float),
                                        df["low"].to_numpy(float), donchian_n)
        for t in bars:
            if t < context_len:
                continue
            ctx = rv[t - context_len:t]           # ends at rv[t-1] -> causal
            if np.isnan(ctx).any() or not np.isfinite(rv[t - 1]):
                continue
            ctxs.append(ctx.astype(np.float32))
            index.append((wi, t))
            rv_prev.append(float(rv[t - 1]))
    C = context_len
    return (np.array(ctxs) if ctxs else np.empty((0, C), np.float32),
            index, np.array(rv_prev))


def _fake_pred(contexts):
    """Deterministic stand-in for plumbing tests: predicted fwd vol = mean of last
    48 context points * 1.05 (a trivial 'slight expansion' predictor)."""
    if len(contexts) == 0:
        return np.empty(0)
    return contexts[:, -48:].mean(axis=1) * 1.05


def model_predictions(contexts, backend, device, adapter_path, horizon, batch_size=64):
    if backend == "fake":
        return _fake_pred(contexts)
    from src.timesfm_vol_eval import _load_timesfm, _timesfm_vol_pred
    model = _load_timesfm(device=device, adapter_path=adapter_path)
    return _timesfm_vol_pred(model, contexts, horizon, batch_size=batch_size)


def build_model_gate(paths, index, pred_fwd, rv_prev, kind="expansion"):
    """Per-week boolean gate arrays from model predictions at breakout bars.

    kind='expansion': gate[t]=True iff pred_fwd > rv[t-1] (model predicts vol rises).
    kind='highvol'  : gate[t]=True iff pred_fwd >= causal running median of pred_fwd.
    Bars without a prediction default to False (no entry)."""
    gates = [np.zeros(len(df), dtype=bool) for df in paths]
    if kind == "highvol":
        # causal running median of predictions in arrival (chronological) order
        running = []
        thr = np.full(len(pred_fwd), np.inf)
        for i, p in enumerate(pred_fwd):
            if running:
                thr[i] = np.median(running)
            running.append(p)
        decision = pred_fwd >= thr
    else:
        decision = pred_fwd > rv_prev
    for (wi, t), d in zip(index, decision):
        if d:
            gates[wi][t] = True
    return gates


def run(data_dir, manifest_path, split, *, backend="timesfm", device="cpu",
        adapter_path=None, context_len=512, horizon=96, window=48, donchian_n=96,
        k_stop=2.0, k_trail=3.0, k_add=1.0, max_adds=3, spread_pips=0.5,
        max_files=None, cache_path=None):
    from src.timesfm_breakout_backtest import build_paths
    paths = build_paths(data_dir, manifest_path, split, max_files)

    contexts, index, rv_prev = collect_contexts(paths, context_len, window, donchian_n)
    print(f"[stage2] {len(paths)} weeks, {len(index)} breakout-trigger contexts "
          f"(ctx={context_len}, window={window}, donchian={donchian_n})")

    # cache model predictions so threshold tuning is free
    pred = None
    if cache_path and Path(cache_path).exists() and backend != "fake":
        z = np.load(cache_path, allow_pickle=True)
        if z["pred"].shape[0] == len(index):
            pred = z["pred"]
            print(f"[stage2] loaded cached predictions <- {cache_path}")
    if pred is None:
        pred = model_predictions(contexts, backend, device, adapter_path, horizon)
        if cache_path and backend != "fake":
            np.savez(cache_path, pred=pred)
            print(f"[stage2] cached predictions -> {cache_path}")

    def _engine(gate_for_week):
        allc, alla = [], []
        for df, g in zip(paths, gate_for_week):
            c, a = simulate_week_breakout(
                df["open"].to_numpy(float), df["high"].to_numpy(float),
                df["low"].to_numpy(float), df["close"].to_numpy(float),
                df["atr"].to_numpy(float), gate=g, donchian_n=donchian_n,
                k_stop=k_stop, k_trail=k_trail, k_add=k_add, max_adds=max_adds,
                spread_pips=spread_pips)
            allc.extend(c); alla.extend(a)
        return summarize(allc, alla)

    # comparators
    results = {}
    results["none"] = _engine([make_gate(df, "none") for df in paths])
    results["naive_expansion"] = _engine([make_gate(df, "expansion", donchian_n,
                                                     vol_col=f"realized_vol_{window}") for df in paths])
    tag = "tsfm_finetuned" if adapter_path else "tsfm_zeroshot"
    if backend == "fake":
        tag = "fake_model"
    results[f"{tag}_expansion"] = _engine(
        build_model_gate(paths, index, pred, rv_prev, kind="expansion"))
    results[f"{tag}_highvol"] = _engine(
        build_model_gate(paths, index, pred, rv_prev, kind="highvol"))

    return {"config": {"split": split, "n_weeks": len(paths), "n_contexts": len(index),
                       "backend": backend, "adapter_path": adapter_path,
                       "context_len": context_len, "horizon": horizon, "window": window,
                       "donchian_n": donchian_n, "spread_pips": spread_pips},
            "results": results}


def print_report(rep):
    c = rep["config"]
    print("=" * 104)
    print(f"Stage 2: TimesFM-gated breakout | split={c['split']} weeks={c['n_weeks']} "
          f"ctxs={c['n_contexts']} backend={c['backend']} adapter={'yes' if c['adapter_path'] else 'no'} "
          f"spread={c['spread_pips']}pip")
    print("WANT: a model gate that beats naive_expansion expectancy_R AND stays >0 at higher spread")
    print("=" * 104)
    print(f"{'gate':<26}{'cycles':>7}{'totR':>9}{'expR':>9}{'win%':>7}{'payoff':>8}"
          f"{'skew':>8}{'Sharpe':>8}{'maxDD_R':>9}{'best':>7}{'worst':>7}")
    print("-" * 104)
    for g, s in rep["results"].items():
        if s.get("n_cycles", 0) == 0:
            print(f"{g:<26}  (no cycles)"); continue
        print(f"{g:<26}{s['n_cycles']:>7}{s['total_R']:>9.2f}{s['expectancy_R']:>9.4f}"
              f"{s['win_rate']*100:>6.1f}%{s['payoff_ratio']:>8.2f}{s['skew']:>8.2f}"
              f"{s['sharpe_naive']:>8.2f}{s['max_drawdown_R']:>9.2f}{s['best_R']:>7.1f}{s['worst_R']:>7.1f}")
    print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 2: TimesFM-gated breakout backtest")
    p.add_argument("--data-dir", dest="data_dir", default="data/split/EURUSD/weekly/train")
    p.add_argument("--manifest-path", dest="manifest_path", default="data/split/EURUSD/weekly/manifest.json")
    p.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--backend", default="timesfm", choices=["timesfm", "fake"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--adapter-path", dest="adapter_path", default=None)
    p.add_argument("--context-len", dest="context_len", type=int, default=512)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--donchian-n", dest="donchian_n", type=int, default=96)
    p.add_argument("--spread-pips", dest="spread_pips", type=float, default=0.5)
    p.add_argument("--max-files", dest="max_files", type=int, default=None)
    p.add_argument("--cache-path", dest="cache_path", default=None)
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cache = a.cache_path
    if cache is None and a.backend != "fake":
        atag = "ft" if a.adapter_path else "zs"
        cache = f"{a.out_dir}/breakout_volgate_pred_{a.split}_c{a.context_len}_h{a.horizon}_w{a.window}_{atag}.npz"
    rep = run(a.data_dir, a.manifest_path, a.split, backend=a.backend, device=a.device,
              adapter_path=a.adapter_path, context_len=a.context_len, horizon=a.horizon,
              window=a.window, donchian_n=a.donchian_n, spread_pips=a.spread_pips,
              max_files=a.max_files, cache_path=cache)
    print_report(rep)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        atag = "ft" if a.adapter_path else ("fake" if a.backend == "fake" else "zs")
        p = Path(a.out_dir) / f"breakout_volgate_{a.split}_sp{a.spread_pips}_{atag}.json"
        p.write_text(json.dumps(rep, indent=2))
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
