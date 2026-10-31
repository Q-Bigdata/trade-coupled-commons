"""Hub-failure outcomes under alternative depletion thresholds and capacity weighting (observed kappa, five largest traders)."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
NUM = json.load(open(os.path.join(R, "numbers.json"))); KOBS = NUM["kobs"]
E1 = pd.read_csv(os.path.join(R, "E1.csv.gz"))
d = E1[(E1.seedset == "top5") & np.isclose(E1.kappa, E1.commodity.map(KOBS))]
cols = ["final_dep", "dep03", "dep07", "final_dep_cap", "cascade", "cascade_cap", "short", "short_base"]
out = {}
for inst, g in d.groupby("inst"):
    per = g.groupby("commodity")[cols].mean()
    out[inst] = {c: float(per[c].mean()) for c in cols}
    out[inst]["by_crop"] = {k: {c: float(v[c]) for c in cols} for k, v in per.iterrows()}
# Monte Carlo comparison of the observed-network rows of E1null with the Table 1 baseline (E7)
N = pd.read_csv(os.path.join(R, "E1null.csv.gz")); E7 = pd.read_csv(os.path.join(R, "E7.csv.gz"))
on = N[(N.null == "observed") & np.isclose(N.kappa, N.commodity.map(KOBS))]
b7 = E7[(E7.variant == "base") & np.isclose(E7.kappa, E7.commodity.map(KOBS))]
mc = {}
for inst in ["Equal", "Mixed", "Distilled"]:
    a = on[on.inst == inst].groupby("commodity").cascade; b = b7[b7.inst == inst].groupby("commodity").cascade
    se = lambda g: float(np.sqrt((g.var() / g.count()).sum()) / g.ngroups)
    mc[inst] = dict(null_observed=float(a.mean().mean()), null_se=se(a), base=float(b.mean().mean()), base_se=se(b))
out["mc_check"] = mc
json.dump(out, open(os.path.join(R, "extra_thresholds.json"), "w"), indent=1)
print(json.dumps({k: {c: round(v[c], 3) for c in cols} for k, v in out.items() if k != "mc_check"}))
print(json.dumps(mc))
