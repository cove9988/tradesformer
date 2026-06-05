"""Market-structure diagnostics for EURUSD: does it mean-revert or trend?

This decides which strategy FAMILY can possibly work BEFORE we build any
averaging/martingale backtest:

    variance ratio VR(q) < 1  -> mean-reverting  -> capped vol-gated grid has a chance
    VR(q) ~= 1                -> random walk      -> no averaging edge; costs win
    VR(q) > 1                 -> trending         -> martingale fatal; anti-martingale instead

Tools
-----
1. Lo-MacKinlay variance ratio with heteroskedasticity-ROBUST z-stat (M2).
   VR(q) = Var(q-period return) / (q * Var(1-period return)).
   Robust z accounts for the fat tails / vol clustering in FX so we don't
   mistake heteroskedasticity for mean reversion.
2. Return autocorrelation at lags 1..q with Bartlett standard errors.

Both are computed WITHIN each weekly file (1440 bars) and POOLED across weeks,
so we never compute a return across the gap between two non-contiguous weeks.

Run:
  PYTHONPATH=. python -m src.timesfm_meanrev_diag --split all --qs 12 24 48 96
  PYTHONPATH=. python -m src.timesfm_meanrev_diag --split test --qs 12 24 48 96
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

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


def load_week_returns(data_dir, manifest_path=None, split="all", target_col="log_return_1"):
    """Return a list of 1-D log-return arrays, one per week in the split."""
    weeks = _weeks_for_split(manifest_path, split)
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if weeks is not None:
        files = [f for f in files if Path(f).name in weeks]
    out = []
    for f in files:
        df = pd.read_csv(f).dropna(subset=[target_col]).reset_index(drop=True)
        r = df[target_col].to_numpy(dtype=np.float64)
        if len(r) > 0:
            out.append(r)
    return out


def variance_ratio_pooled(series_list, q):
    """Lo-MacKinlay VR(q) pooled across independent series (weeks).

    Returns dict with vr, z_robust (heteroskedasticity-consistent), and n.

    Implementation pools the moment sums across weeks so q-period returns never
    straddle a week boundary. Uses overlapping q-returns (Lo-MacKinlay estimator)
    with the bias-corrected denominators and the M2 robust variance from
    Lo & MacKinlay (1988).
    """
    # 1-period variance (mean-adjusted), pooled
    n_total = 0
    mu_num = 0.0
    for r in series_list:
        mu_num += r.sum()
        n_total += len(r)
    if n_total == 0:
        return {"q": q, "vr": float("nan"), "z_robust": float("nan"), "n": 0}
    mu = mu_num / n_total

    # sigma_a^2 : 1-period variance (unbiased, mean-adjusted), pooled
    sa_num = 0.0
    n_a = 0
    for r in series_list:
        d = r - mu
        sa_num += np.sum(d * d)
        n_a += len(r)
    sigma_a2 = sa_num / (n_a - 1) if n_a > 1 else float("nan")

    # sigma_c^2 : variance of overlapping q-period returns, pooled with
    # Lo-MacKinlay bias correction m = q*(n-q+1)*(1 - q/n) accumulated per week.
    sc_num = 0.0
    m_denom = 0.0
    usable = False
    for r in series_list:
        n = len(r)
        if n < q + 1:
            continue
        usable = True
        d = r - mu
        csum = np.cumsum(d)
        # q-period overlapping sums: S_t = sum_{i=t-q+1..t} d_i = csum[t] - csum[t-q]
        qsum = csum[q - 1:] - np.concatenate(([0.0], csum[:-q]))
        sc_num += np.sum(qsum * qsum)
        m_denom += q * (n - q + 1) * (1.0 - q / n)
    if not usable or m_denom <= 0 or not math.isfinite(sigma_a2) or sigma_a2 <= 0:
        return {"q": q, "vr": float("nan"), "z_robust": float("nan"), "n": n_total}
    sigma_c2 = sc_num / m_denom
    vr = sigma_c2 / sigma_a2

    # Heteroskedasticity-robust variance of VR (Lo-MacKinlay M2):
    #   sum_{j=1..q-1} [2(q-j)/q]^2 * delta_j
    # delta_j = sum d_t^2 d_{t-j}^2 / (sum d_t^2)^2  (pooled across weeks)
    denom_sq = 0.0
    for r in series_list:
        d = r - mu
        denom_sq += np.sum(d * d)
    denom_sq = denom_sq ** 2
    theta = 0.0
    for j in range(1, q):
        delta_num = 0.0
        for r in series_list:
            d = r - mu
            if len(d) <= j:
                continue
            d2 = d * d
            delta_num += np.sum(d2[j:] * d2[:-j])
        delta_j = (delta_num * n_total) / denom_sq if denom_sq > 0 else 0.0
        w = (2.0 * (q - j) / q) ** 2
        theta += w * delta_j
    z_robust = (vr - 1.0) / math.sqrt(theta / n_total) if theta > 0 else float("nan")
    return {"q": q, "vr": vr, "z_robust": z_robust, "n": n_total}


def autocorr_pooled(series_list, lag):
    """Pooled return autocorrelation at `lag` with Bartlett SE and z-stat."""
    n_total = sum(len(r) for r in series_list)
    if n_total == 0:
        return {"lag": lag, "acf": float("nan"), "z": float("nan"), "n": 0}
    mu = sum(r.sum() for r in series_list) / n_total
    num = 0.0
    den = 0.0
    n_pairs = 0
    for r in series_list:
        d = r - mu
        den += np.sum(d * d)
        if len(d) > lag:
            num += np.sum(d[lag:] * d[:-lag])
            n_pairs += len(d) - lag
    acf = num / den if den > 0 else float("nan")
    se = 1.0 / math.sqrt(n_total)
    z = acf / se if se > 0 else float("nan")
    return {"lag": lag, "acf": acf, "z": z, "n_pairs": n_pairs}


def classify(vr, z_robust, z_thresh=2.0):
    """Map VR + significance to a regime label."""
    if not math.isfinite(vr) or not math.isfinite(z_robust):
        return "undetermined"
    if abs(z_robust) < z_thresh:
        return "random-walk (no significant structure)"
    return "mean-reverting" if vr < 1.0 else "trending"


def run(data_dir=DEFAULT_DATA_DIR, manifest_path=DEFAULT_MANIFEST, split="all",
        qs=(12, 24, 48, 96)):
    series = load_week_returns(data_dir, manifest_path, split)
    report = {"config": {"data_dir": data_dir, "split": split, "qs": list(qs),
                         "n_weeks": len(series),
                         "n_bars": int(sum(len(r) for r in series))},
              "variance_ratio": [], "autocorr": []}
    for q in qs:
        vr = variance_ratio_pooled(series, q)
        vr["regime"] = classify(vr["vr"], vr["z_robust"])
        report["variance_ratio"].append(vr)
    for lag in qs:
        report["autocorr"].append(autocorr_pooled(series, lag))
    return report


def print_report(report):
    c = report["config"]
    print("=" * 72)
    print(f"EURUSD mean-reversion diagnostics | split={c['split']} "
          f"weeks={c['n_weeks']} bars={c['n_bars']}")
    print("VR<1 mean-revert | VR~1 random walk | VR>1 trend  "
          "(|z|>2 => significant)")
    print("=" * 72)
    print(f"\n{'q (bars)':>9} {'~time':>7} {'VR':>8} {'z_robust':>10}  regime")
    print("-" * 72)
    mins = {12: "1h", 24: "2h", 48: "4h", 96: "8h", 288: "24h"}
    for v in report["variance_ratio"]:
        q = v["q"]
        print(f"{q:>9} {mins.get(q, str(q)):>7} {v['vr']:>8.4f} "
              f"{v['z_robust']:>10.2f}  {v['regime']}")
    print(f"\n{'lag':>9} {'ACF':>10} {'z':>10}")
    print("-" * 40)
    for a in report["autocorr"]:
        print(f"{a['lag']:>9} {a['acf']:>10.5f} {a['z']:>10.2f}")
    print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EURUSD mean-reversion / trend diagnostics")
    p.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest-path", dest="manifest_path", default=DEFAULT_MANIFEST)
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--qs", type=int, nargs="+", default=[12, 24, 48, 96])
    p.add_argument("--out", dest="out_dir", default="data/predictions/eval")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    report = run(a.data_dir, a.manifest_path, a.split, tuple(a.qs))
    print_report(report)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        qtag = "q" + "-".join(str(q) for q in a.qs)
        p = Path(a.out_dir) / f"meanrev_diag_{a.split}_{qtag}.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
