"""Summaries, headline numbers and source data for the revised manuscript (runs on the compute host).
Reads results/*.csv.gz; writes results/numbers.json and source_data/*.csv."""
import os, json, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.join(HERE, "results"); SD = os.path.join(HERE, "source_data"); os.makedirs(SD, exist_ok=True)
COMS = ["wheat", "maize", "rice", "soybean"]
AUD = json.load(open(os.path.join(R, "net_audit.json")))
KOBS = {c: round(AUD[f"{c}_2021_2023"]["kappa_unfiltered"], 3) for c in COMS}
NUM = {"kobs": KOBS}


SUF = os.environ.get("SUF", "")


def rd(name):
    return pd.read_csv(os.path.join(R, f"{name}{SUF}.csv.gz"))


def tci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def strat(df, col, by):
    """Equal-weight mean over commodities with stratified 95% CI (conditional on the four networks)."""
    out = []
    for key, d in df.groupby(by):
        per = d.groupby("commodity")[col].agg(["mean", "var", "count"])
        m = per["mean"].mean(); se = np.sqrt((per["var"] / per["count"]).sum()) / len(per)
        out.append(dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,)), mean=m, ci=1.96 * se, n=int(per["count"].sum()), ncom=len(per)))
    return pd.DataFrame(out)


def at_obs(df):
    return df[np.isclose(df.kappa, df.commodity.map(KOBS))]


# ------------------------------------------------------------------ E1
E1 = rd("E1")
assert "error" not in E1 or E1["error"].isna().all()
E1.to_csv(os.path.join(SD, "E1_hub_failures_runs.csv.gz"), index=False, compression="gzip")
g = E1.groupby(["commodity", "inst", "seedset", "kappa"])
agg = g[["final_dep", "base_dep", "cascade", "final_dep_cap", "cascade_cap", "short", "dep03", "dep07"]].agg(["mean", tci]).reset_index()
agg.columns = ["_".join([c for c in col if c]) for col in agg.columns]
agg.to_csv(os.path.join(SD, "Fig2_EDFig1_cascades_summary.csv"), index=False)
o5 = at_obs(E1[E1.seedset == "top5"])
NUM["E1_top5_obs"] = {f"{c}|{i}": dict(final=float(d.final_dep.mean()), final_ci=tci(d.final_dep), cascade=float(d.cascade.mean()), cascade_ci=tci(d.cascade),
                                         base=float(d.base_dep.mean()), cap=float(d.cascade_cap.mean()), n=len(d))
                      for (c, i), d in o5.groupby(["commodity", "inst"])}
s = strat(E1[(E1.seedset == "top5") & np.isclose(E1.kappa, 0.4)], "cascade", "inst")
NUM["E1_top5_k04_cascade_pooled"] = s.set_index("inst")[["mean", "ci"]].to_dict("index")
s = strat(E1[(E1.seedset == "top5") & np.isclose(E1.kappa, 0.4)], "final_dep", "inst")
NUM["E1_top5_k04_final_pooled"] = s.set_index("inst")[["mean", "ci"]].to_dict("index")
NUM["E1_rand5_cascade_max_mean"] = float(E1[E1.seedset == "rand5"].groupby(["commodity", "inst", "kappa"]).cascade.mean().max())
NUM["E1_rand5_cascade_mean"] = float(E1[E1.seedset == "rand5"].cascade.mean())
t1 = at_obs(E1[E1.seedset == "top1"])
NUM["E1_top1_obs_cascade"] = {f"{c}|{i}": float(d.cascade.mean()) for (c, i), d in t1.groupby(["commodity", "inst"])}
NUM["E1_margins"] = {i: dict(median=float(d.margin_median.mean()), frac_neg=float(d.frac_m_neg.mean()), frac_iso_unviable=float(d.frac_iso_unviable.mean()),
                              lam_full=float(d.lam_full_mean.mean()), lam_max=float(d.lam_max_mean.mean()))
                     for i, d in E1[E1.commodity == "wheat"].groupby("inst")}
NUM["E1_margins_all"] = {f"{c}|{i}": dict(median=float(d.margin_median.mean()), frac_neg=float(d.frac_m_neg.mean()), frac_iso_unviable=float(d.frac_iso_unviable.mean()))
                         for (c, i), d in E1.groupby(["commodity", "inst"])}
NUM["E1_seed_cap_share_top5"] = {c: float(d.seed_cap_share.mean()) for c, d in E1[E1.seedset == "top5"].groupby("commodity")}
NUM["E1_terminal_vs_window"] = dict(mean_abs_diff=float(np.mean(np.abs(E1.final_dep - E1.final_dep_terminal))), recovered=float(E1.recovered_terminal.mean()))
# reductions
red = {}
for tag in ["interp_conv", "exact_conv", "interp_h300", "exact_h300"]:
    a = E1[f"red_{tag}_acc"]; dep = E1[f"red_{tag}_dep"]
    r2 = float(np.corrcoef(E1.final_dep, dep)[0, 1] ** 2)
    fp = E1[f"red_{tag}_fp"].sum(); fn = E1[f"red_{tag}_fn"].sum(); pos = E1[f"red_{tag}_pos"].sum()
    Ntot = int(E1.commodity.map({c: AUD[f"{c}_2021_2023"]["countries_final"] for c in COMS}).sum())
    red[tag] = dict(acc=float(a.mean()), acc_p05=float(a.quantile(0.05)), share_runs_ge97=float((a >= 0.97).mean()), r2=r2,
                    fp_rate=float(fp / (Ntot - pos)), fn_rate=float(fn / pos), absx=float(E1[f"red_{tag}_absx"].mean()),
                    by_inst={i: float(d[f"red_{tag}_acc"].mean()) for i, d in E1.groupby("inst")},
                    fn_by_inst={i: float(d[f"red_{tag}_fn"].sum() / max(d[f"red_{tag}_pos"].sum(), 1)) for i, d in E1.groupby("inst")},
                    by_seed={i: float(d[f"red_{tag}_acc"].mean()) for i, d in E1.groupby("seedset")},
                    by_com={i: float(d[f"red_{tag}_acc"].mean()) for i, d in E1.groupby("commodity")},
                    min_kappa_acc=float(E1.groupby("kappa")[f"red_{tag}_acc"].mean().min()),
                    argmin_kappa=float(E1.groupby("kappa")[f"red_{tag}_acc"].mean().idxmin()))
    if tag.endswith("conv"):
        red[tag]["converged"] = float(E1[f"red_{tag}_converged"].mean()); red[tag]["cycle"] = float(E1[f"red_{tag}_cycle"].mean())
        red[tag]["cycle_by_inst"] = {i: float(d[f"red_{tag}_cycle"].mean()) for i, d in E1.groupby("inst")}
NUM["reduction"] = red
NUM["E1_n_runs"] = int(len(E1))
# first-order criteria
crit = {}
cc = E1.dropna(subset=["crit_deplete_new_tp"])
for nm in ["capkeep", "reserve", "deplete"]:
    for tg in ["new", "dep"]:
        for sub, d in [("all", cc), ("k_le_obs", cc[cc.kappa <= cc.commodity.map(KOBS) + 1e-9]), ("top1", cc[cc.seedset == "top1"])]:
            tp, fp, fn, tn = [d[f"crit_{nm}_{tg}_{k}"].sum() for k in ["tp", "fp", "fn", "tn"]]
            crit[f"{nm}|{tg}|{sub}"] = dict(acc=float((tp + tn) / (tp + fp + fn + tn)), precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                                           fpr=float(fp / max(fp + tn, 1)), n=int(tp + fp + fn + tn))
NUM["criteria"] = crit

# ------------------------------------------------------------------ E1alt / E1net / E1null
A = rd("E1alt"); A.to_csv(os.path.join(SD, "EDFig2_seed_semantics_runs.csv.gz"), index=False, compression="gzip")
NUM["E1alt_obs"] = {f"{k}|{i}": dict(cascade=float(d.cascade.mean()), final=float(d.final_dep.mean()), peak=float(d.peak_dep.mean()))
                    for (k, i), d in at_obs(A).groupby(["kind", "inst"])}
NUM["E1alt_obs_by_com"] = {f"{k}|{c}|{i}": float(d.cascade.mean()) for (k, c, i), d in at_obs(A).groupby(["kind", "commodity", "inst"])}
Nn = rd("E1net"); Nn.to_csv(os.path.join(SD, "EDFig3_coupling_variants_runs.csv.gz"), index=False, compression="gzip")
NUM["E1net"] = {f"{cpl}|{s}|{m}|{i}": dict(cascade=float(d.cascade.mean()), final=float(d.final_dep.mean()), latent=float(d.latent_dep.mean()),
                                          visible=float(d.visible_dep.mean()), eff=float(d.efficiency.mean()))
                for (cpl, s, m, i), d in Nn.groupby(["coupling", "seedset", "mult", "inst"])}
NUM["E1net_by_com"] = {f"{cpl}|{s}|{c}|{i}": float(d.cascade.mean()) for (cpl, s, c, i), d in Nn[Nn.mult == 1.0].groupby(["coupling", "seedset", "commodity", "inst"])}
Nl = rd("E1null"); Nl.to_csv(os.path.join(SD, "EDFig3_null_models_runs.csv.gz"), index=False, compression="gzip")
NUM["E1null"] = {f"{nl}|{sm}|{i}|{k}": dict(cascade=float(d.cascade.mean()), final=float(d.final_dep.mean()), seedcap=float(d.seed_cap_share.mean()), q=float(d.median_q.mean()))
                 for (nl, sm, i, k), d in Nl.assign(k=np.where(np.isclose(Nl.kappa, 0.4), "k0.4", "obs")).groupby(["null", "seedmode", "inst", "k"])}

# ------------------------------------------------------------------ E2
E2 = rd("E2"); E2.to_csv(os.path.join(SD, "Fig3_EDFig8_hidden_fragility_runs.csv.gz"), index=False, compression="gzip")
b2 = E2[E2.tag == "2021_2023"]
for col in ["visible_dep", "latent_dep_300", "latent_dep_600", "efficiency", "waste", "gini_cum", "gini_late", "excl5_cum", "excl5_late", "excl1_late", "excl10_late", "excl20_late", "wmean"]:
    s = strat(b2, col, ["inst", "kappa"]); s.to_csv(os.path.join(SD, f"E2_pooled_{col}.csv"), index=False)
NUM["E2_pooled"] = {}
for col in ["visible_dep", "latent_dep_600", "efficiency", "waste", "gini_late", "excl5_late", "excl5_cum", "gini_cum", "wmean"]:
    s = strat(b2[b2.kappa.isin([0.0, 0.25, 0.4, 0.7])], col, ["inst", "kappa"])
    NUM["E2_pooled"][col] = {f"{r.inst}|{r.kappa}": [float(r.mean), float(r.ci)] for r in s.itertuples()}
NUM["E2_pred_unviable"] = {i: float(d.pred_unviable.mean()) for i, d in b2.groupby("inst")}
NUM["E2_latent_agree"] = {i: float(d.latent_agree.mean()) for i, d in b2.groupby("inst")}
NUM["E2_latent300_vs_600"] = float(np.mean(np.abs(b2.latent_dep_300 - b2.latent_dep_600)))
NUM["E2_obs"] = {f"{c}|{i}": dict(visible=float(d.visible_dep.mean()), latent=float(d.latent_dep_600.mean()), eff=float(d.efficiency.mean()))
                 for (c, i), d in at_obs(b2).groupby(["commodity", "inst"])}
NUM["E2_period2017"] = {f"{i}|{k}": float(d.latent_dep_600.mean()) for (i, k), d in E2[E2.tag == "2017_2019"].groupby(["inst", "kappa"]) if k in (0.0, 0.4)}

# ------------------------------------------------------------------ E3 (hazards, export rules)
E3 = rd("E3"); E3.to_csv(os.path.join(SD, "Fig3c_Fig4_EDFig9_hazard_runs.csv.gz"), index=False, compression="gzip")
main = E3[E3.hazard == "both"]
rows = []
for (inst, rule, kap), d in main.groupby(["inst", "rule", "kappa"]):
    if d.commodity.nunique() < 4 and kap not in [KOBS[c] for c in COMS]:
        pass
    k = int(d.sys50_any.sum()); n = len(d); lo, hi = wilson(k, n)
    rows.append(dict(inst=inst, rule=rule, kappa=kap, n=n, events=k, p=k / n, lo=lo, hi=hi, occ=float(d.sys50_occ.mean()),
                     eff=float(d.efficiency.mean()), eff_ci=tci(d.efficiency), dep_mean=float(d.dep_mean.mean()), ncom=d.commodity.nunique(),
                     sys40=float(d.sys40_any.mean()), sys60=float(d.sys60_any.mean()), syscap=float(d.syscap50_any.mean()),
                     sys_d03=float(d.sys50_dep03_any.mean()), sys_d07=float(d.sys50_dep07_any.mean()), first=float(d.sys50_first.mean())))
H = pd.DataFrame(rows); H.to_csv(os.path.join(SD, "Fig3c_Fig4_systemic_summary.csv"), index=False)
NUM["E3_n_per_cell"] = int(main.groupby(["inst", "rule", "kappa", "commodity"]).size().median())
def hsel(inst, rule, kap):
    r = H[(H.inst == inst) & (H.rule == rule) & np.isclose(H.kappa, kap) & (H.ncom == 4)]
    return r.iloc[0].to_dict() if len(r) else None
NUM["E3_none"] = {f"{i}|{k}": hsel(i, "none", k) for i in ["Equal", "Mixed", "Distilled", "Proportional", "Step"] for k in [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.7]}
NUM["E3_rules"] = {f"{i}|{ru}|{k}": hsel(i, ru, k) for i in ["Equal", "Mixed", "Distilled"] for ru in ["none", "ban", "halve", "ban_hyst", "halve_hyst", "quarter", "ban_th0.6", "ban_th0.7", "ban_th0.9"] for k in [0.25, 0.4, 0.5, 0.6]}
rows = []
for (c, inst, rule), d in at_obs(main).groupby(["commodity", "inst", "rule"]):
    k = int(d.sys50_any.sum()); n = len(d); lo, hi = wilson(k, n)
    rows.append(dict(commodity=c, kappa=KOBS[c], inst=inst, rule=rule, n=n, events=k, p=k / n, lo=lo, hi=hi, eff=float(d.efficiency.mean()), dep_mean=float(d.dep_mean.mean()), occ=float(d.sys50_occ.mean())))
HO = pd.DataFrame(rows); HO.to_csv(os.path.join(SD, "Table_obs_kappa_systemic.csv"), index=False)
NUM["E3_obs"] = {f"{r.commodity}|{r.inst}|{r.rule}": dict(events=int(r.events), n=int(r.n), p=float(r.p), lo=float(r.lo), hi=float(r.hi), eff=float(r.eff), dep=float(r.dep_mean)) for r in HO.itertuples()}
sens = E3[E3.hazard != "both"]
NUM["E3_hazard_sens"] = {f"{hz}|{i}|{k}": dict(p=float(d.sys50_any.mean()), n=len(d), dep=float(d.dep_mean.mean()), eff=float(d.efficiency.mean()))
                         for (hz, i, k), d in sens.assign(k=np.where(np.isclose(sens.kappa, 0.4), "0.4", np.where(sens.kappa.round(3).isin(list(KOBS.values())), "obs", "other"))).groupby(["hazard", "inst", "k"]) if k != "other"}
E4b = rd("E4b"); E4b.to_csv(os.path.join(SD, "Fig4b_production_shock_runs.csv.gz"), index=False, compression="gzip")
NUM["E4b"] = {f"{i}|{ru}|{k}": dict(peak=float(d.peak_dep.mean()), loss=float(d.welfare_loss.mean()), final=float(d.final_dep.mean()))
              for (i, ru, k), d in E4b.groupby(["inst", "rule", "kappa"]) if d.commodity.nunique() == 4}
NUM["E4b_obs"] = {f"{c}|{i}|{ru}": dict(peak=float(d.peak_dep.mean()), loss=float(d.welfare_loss.mean())) for (c, i, ru), d in at_obs(E4b).groupby(["commodity", "inst", "rule"])}

# ------------------------------------------------------------------ E5
E5 = rd("E5"); E5.to_csv(os.path.join(SD, "Fig5_targeting_runs.csv.gz"), index=False, compression="gzip")
NUM["E5"] = {}
for (sh, pl, k, h), d in E5.assign(k=np.where(np.isclose(E5.kappa, 0.4), "0.4", "obs")).groupby(["shockset", "placement", "k", "h"]):
    if h in (0.0, 0.1, 0.2, 0.4, 1.0):
        NUM["E5"][f"{sh}|{pl}|{k}|{h}"] = dict(cascade=float(d.cascade.mean()), final=float(d.final_dep.mean()), latent=float(d.latent_dep.mean()),
                                              seeds_adopting=float(d.seeds_adopting.mean()), adopter_cap=float(d.adopter_cap_share.mean()), eff=float(d.efficiency.mean()))

# ------------------------------------------------------------------ E6
E6 = rd("E6"); E6.to_csv(os.path.join(SD, "EDFig10_mobility_runs.csv.gz"), index=False, compression="gzip")
w6 = E6[E6.commodity == "wheat"]
NUM["E6"] = {f"{who}|{f}|{h}|{m}": {c: float(d[c].mean()) for c in d.columns if d[c].dtype != object and c not in ("rep",)}
             for (who, f, h, m), d in w6.groupby(["who", "f_mean", "h_eq", "migration"])}
fr6 = w6[(w6.who == "freeriders") & (w6.h_eq > 0) & (w6.h_eq < 1) & (w6.migration > 0)]
NUM["E6_shift_eq_pooled_max"] = float((fr6.eq_fr_pooled - fr6.eq_fr0_pooled).groupby([fr6.f_mean, fr6.h_eq, fr6.migration]).mean().max())
NUM["E6_shift_d_pooled_min"] = float((fr6.d_fr_pooled - fr6.d_fr0_pooled).groupby([fr6.f_mean, fr6.h_eq, fr6.migration]).mean().min())
NUM["E6_shift_eq_country_max"] = float((fr6.eq_fr_country_mean - fr6.eq_fr0_country_mean).groupby([fr6.f_mean, fr6.h_eq, fr6.migration]).mean().max())
NUM["E6_shift_d_country_min"] = float((fr6.d_fr_country_mean - fr6.d_fr0_country_mean).groupby([fr6.f_mean, fr6.h_eq, fr6.migration]).mean().min())

# ------------------------------------------------------------------ E7 and Table 1
E7 = rd("E7"); E7.to_csv(os.path.join(SD, "Table1_EDFig5_EDFig6_variants_runs.csv.gz"), index=False, compression="gzip")
assert "error" not in E7 or E7["error"].isna().all()
ob = at_obs(E7)
def var_row(d):
    out = {}
    for i in ["Equal", "Mixed", "Distilled"]:
        di = d[d.inst == i]
        if len(di):
            per = di.groupby("commodity")
            out[f"casc_{i}"] = float(per.cascade.mean().mean()); out[f"final_{i}"] = float(per.final_dep.mean().mean())
            out[f"lat_{i}"] = float(per.latent_dep.mean().mean()); out[f"eff_{i}"] = float(per.efficiency.mean().mean())
            out[f"base_{i}"] = float(per.base_dep.mean().mean())
            out[f"casc_ci_{i}"] = float(1.96 * np.sqrt((per.cascade.var() / per.cascade.count()).sum()) / per.ngroups)
            out[f"lammax_{i}"] = float(per.lam_max_mean.mean().mean()); out[f"lamfull_{i}"] = float(per.lam_full_mean.mean().mean())
            out[f"unviable_{i}"] = float(per.frac_iso_unviable.mean().mean()); out[f"margin_{i}"] = float(per.margin_median.mean().mean())
    out["n"] = int(len(d[d.inst == "Equal"]) / max(d[d.inst == "Equal"].commodity.nunique(), 1))
    return out
T1 = {v: var_row(d) for v, d in ob.groupby("variant")}
# bootstrap summary
boots = [T1[v] for v in T1 if v.startswith("human_boot")]
if boots:
    T1["human_boot_summary"] = {k: float(np.mean([b[k] for b in boots if k in b])) for k in boots[0]}
    T1["human_boot_summary_sd"] = {k: float(np.std([b[k] for b in boots if k in b])) for k in boots[0]}
# network-coupling variants from E1net and nulls (observed kappa equivalents)
for cpl, s_ in [("supply", "top5"), ("directed", "top5"), ("directed", "exp5")]:
    d = Nn[(Nn.coupling == cpl) & (Nn.seedset == s_) & (Nn.mult == 1.0)]
    T1[f"coupling_{cpl}_{s_}"] = var_row(d.assign(lam_max_mean=np.nan, lam_full_mean=np.nan, frac_iso_unviable=np.nan, margin_median=np.nan))
for nl in ["weights_shuffled", "strength_preserving", "degree_preserving", "erdos_renyi"]:
    d = at_obs(Nl[(Nl.null == nl) & (Nl.seedmode == "rerank")])
    T1[f"null_{nl}"] = var_row(d.assign(latent_dep=np.nan, efficiency=np.nan, lam_max_mean=np.nan, lam_full_mean=np.nan, frac_iso_unviable=np.nan, margin_median=np.nan))
    d = at_obs(Nl[(Nl.null == nl) & (Nl.seedmode == "fixed")])
    T1[f"null_{nl}_fixedseeds"] = var_row(d.assign(latent_dep=np.nan, efficiency=np.nan, lam_max_mean=np.nan, lam_full_mean=np.nan, frac_iso_unviable=np.nan, margin_median=np.nan))
json.dump(T1, open(os.path.join(R, "table1.json"), "w"), indent=1)
pd.DataFrame(T1).T.to_csv(os.path.join(SD, "Table1_robustness_summary.csv"))
NUM["T1_ranking_holds"] = {v: bool(r.get("casc_Distilled", np.nan) < r.get("casc_Equal", np.nan)) for v, r in T1.items() if "casc_Equal" in r}
NUM["T1_ranking_holds_mixed"] = {v: bool(r.get("casc_Distilled", np.nan) <= r.get("casc_Mixed", np.nan)) for v, r in T1.items() if "casc_Mixed" in r}
b3 = E7[E7.behaviour == "B3"]
NUM["B3_evo"] = {v: {c: float(d[c].mean()) for c in d.columns if c.startswith("evo_")} for v, d in b3[b3.inst == "Distilled"].groupby("variant")}
NUM["B3_evo_equal"] = {v: {c: float(d[c].mean()) for c in d.columns if c.startswith("evo_")} for v, d in b3[b3.inst == "Equal"].groupby("variant")}

# ------------------------------------------------------------------ E8 ablation
E8 = rd("E8"); E8.to_csv(os.path.join(SD, "EDFig7_ablation_runs.csv.gz"), index=False, compression="gzip")
ab = {}
for (rn, k), d in E8.assign(k=np.where(np.isclose(E8.kappa, 0.4), "0.4", "obs")).groupby(["rule", "k"]):
    per = d.groupby("commodity")
    ab[f"{rn}|{k}"] = {c: float(per[c].mean().mean()) for c in ["cascade", "final_dep", "latent_dep", "efficiency", "waste", "wmean", "sys50_any", "gini_late", "excl5_late", "excl1_late", "margin_median"]}
NUM["E8"] = ab
json.dump(NUM, open(os.path.join(R, "numbers.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("numbers written", len(NUM))
