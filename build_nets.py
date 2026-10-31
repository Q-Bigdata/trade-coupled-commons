"""Build every network variant and the construction audit.  Inputs: tm_staples.csv.gz, qcl_staples.csv.gz.
Outputs: nets/<com>_<tag>.npz and results/net_audit.json, results/net_table.csv."""
import os, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks2 as nw

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "nets"), exist_ok=True); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
tm = pd.read_csv(os.path.join(HERE, "tm_staples.csv.gz"))
qcl = pd.read_csv(os.path.join(HERE, "qcl_staples.csv.gz"))
COMS = ["wheat", "maize", "rice", "soybean"]
PERIODS = {"2021_2023": [2021, 2022, 2023], "2017_2019": [2017, 2018, 2019], "2021": [2021], "2022": [2022], "2023": [2023]}
FILTERS = {"base": (1000.0, 1e4, False), "nofilter": (0.0, 0.0, False), "light": (100.0, 1e3, False),
           "heavy": (1e4, 1e5, False), "iterative": (1000.0, 1e4, True)}
audit = {}; rows = []
for com in COMS:
    for ptag, yrs in PERIODS.items():
        f, names, st0 = nw.reconcile(tm, com, yrs)
        prod = nw.production(qcl, com, yrs)
        wprod = float(prod.sum())
        for ftag, (me, mn, it) in FILTERS.items():
            if ptag != "2021_2023" and ftag != "base":
                continue
            net, st = nw.build(f, names, me, mn, it)
            tag = ptag if ftag == "base" else f"{ptag}_{ftag}"
            prod_t = prod.reindex(net["codes"]).fillna(0.0).values
            np.savez_compressed(os.path.join(HERE, "nets", f"{com}_{tag}.npz"), W=net["W"], F=net["F"], codes=net["codes"],
                                names=net["names"], prod=prod_t)
            st.update(st0)
            st["world_production_Mt"] = wprod / 1e6
            st["kappa_unfiltered"] = float(f.flow.sum() / wprod)
            st["kappa_network"] = float(net["F"].sum() / wprod)
            st["prod_coverage_of_world"] = float(prod_t.sum() / wprod)
            st["n_zero_production"] = int((prod_t <= 0).sum())
            sb = nw.supply_based(net["W"], net["F"], prod_t)
            st["kappa_supply_weighted"] = sb["kappa_bar"]; st["kappa_j_median"] = float(np.median(sb["kappa_j"]))
            st["kappa_j_p10"] = float(np.quantile(sb["kappa_j"], 0.1)); st["kappa_j_p90"] = float(np.quantile(sb["kappa_j"], 0.9))
            dr = nw.directed(net["W"], net["F"], prod_t)
            st["d_median"] = float(np.median(dr["d"])); st["d_share_gt_half"] = float(np.mean(dr["d"] > 0.5))
            exp_rank = np.argsort(-dr["exports"])[:5]; str_rank = np.argsort(-net["W"].sum(1))[:5]
            st["top5_strength"] = [str(net["names"][i]) for i in str_rank]
            st["top5_exporters"] = [str(net["names"][i]) for i in exp_rank]
            st["top5_strength_net_export_Mt"] = [float((dr["exports"][i] - dr["imports"][i]) / 1e6) for i in str_rank]
            audit[f"{com}_{tag}"] = st
            if ftag == "base":
                W = net["W"]; P = W / W.sum(1, keepdims=True); q = P.max(1)
                rows.append(dict(commodity=com, period=ptag, countries=W.shape[0], links=st["links_final"],
                                 trade_Mt=st["mass_final_Mt"], production_Mt=wprod / 1e6, kappa=st["kappa_unfiltered"],
                                 kappa_network=st["kappa_network"], median_q=float(np.median(q)), share_q_gt_half=float(np.mean(q > 0.5)),
                                 mass_retained=st["mass_final"], top5=", ".join(st["top5_strength"]),
                                 top5_exporters=", ".join(st["top5_exporters"])))
        # item breakdown for rice
        if com == "rice" and ptag == "2021_2023":
            br = {}
            for it, fac in nw.ITEMS["rice"].items():
                sub = tm[(tm["Item Code"] == it) & tm.Year.isin(yrs) & (tm["Element Code"] == 5610) & (tm["Reporter Country Code"] < 5000)]
                br[str(it)] = dict(factor=fac, import_reports_Mt_raw=float(sub.Value.sum() / 3 / 1e6), import_reports_Mt_milled_eq=float(sub.Value.sum() * fac / 3 / 1e6))
            audit["rice_item_breakdown"] = br
json.dump(audit, open(os.path.join(HERE, "results", "net_audit.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "net_table.csv"), index=False)
print("networks built", len(audit))
