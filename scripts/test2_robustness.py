"""TEST 2 robustness checks: cross-pair independence + breakeven cost."""
import json
import sys
import numpy as np

sys.path.insert(0, ".")
from src.timesfm_multipair_forward_eval import build_all_windows

pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]
win = build_all_windows(pairs, 512, 96, 48)
A = np.vstack([win[p]["actual"] for p in pairs])
E = np.vstack([win[p]["ewma"] for p in pairs])

# 1) Cross-pair correlation of the ACTUAL forward-vol edge (actual - implied)
edge = A - E
C = np.corrcoef(edge)
print("=== Cross-pair corr of forward-vol EDGE (actual - implied) ===")
print("        " + "".join(f"{p:>9}" for p in pairs))
for i, p in enumerate(pairs):
    print(f"{p:<8}" + "".join(f"{C[i, j]:>9.3f}" for j in range(len(pairs))))
off = C[np.triu_indices(5, 1)]
ev = np.linalg.eigvalsh(C)[::-1]
n_eff = (ev.sum() ** 2) / (ev ** 2).sum()
print(f"\nmean off-diag corr = {off.mean():.3f}  (1=identical bet, 0=independent)")
print(f"eigenvalues = {np.round(ev, 2)}")
print(f"effective independent pairs (participation ratio) = {n_eff:.2f} of 5")

# Also: correlation of the multi_ft PREDICTION edge across pairs (are signals 1 bet?)
# reconstruct pred edge sign agreement isn't in json; use actual-edge corr as proxy for
# how correlated the *opportunity* is. High corr => 5/5 is ~n_eff independent confirmations.

# 2) Breakeven cost. threshold=0 -> every window trades (N=544), PnL(cost)=gross - cost*N.
d = json.load(open("data/predictions/forward/multipair_test2_matrix.json"))
res = d["results"]
N = 544
c0 = d["config"]["cost"]
PIP = 1e-4  # FX pip in price units; vol is in return units, so express breakeven in pips of vol
print("\n=== Breakeven cost (per trade, in return/vol-point units) ===")
print(f"used cost c0 = {c0:.0e} per trade")
print(f"{'pair':<9}{'model':<11}{'net_pnl':>11}{'gross/trd':>12}{'breakeven_c':>13}{'x over c0':>11}")
for p in pairs:
    for m in ["zs", "eurusd_ft", "multi_ft"]:
        tot = res[m][p]["total_pnl"]
        gross = tot + c0 * N
        gross_per = gross / N
        margin = gross_per / c0
        print(f"{p:<9}{m:<11}{tot:>11.5f}{gross_per:>12.2e}{gross_per:>13.2e}{margin:>10.1f}x")

# 3) Honest framing: convert a representative breakeven to "pips of vol" and to bp
print("\n=== Scale sanity: what is one 'vol point' here? ===")
# actual forward vol magnitude
print(f"median forward vol (actual) across pairs: {np.median(A):.2e}  (i.e. ~{np.median(A)*1e4:.1f} 'pips' of 5-min rv)")
print(f"a 1e-5 cost vs typical edge {np.median(np.abs(edge)):.2e}: cost/edge = {c0/np.median(np.abs(edge)):.2%}")
