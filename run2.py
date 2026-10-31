"""Revision-2 experiment suite.  Usage: python3 run2.py EXP [EXP ...] [--procs 150] [--quick]
Writes results/<EXP>.csv (one row per run or per run x coupling) and results/<EXP>_extra.npz where needed."""
import os, sys, zlib, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ccsim2 as cs, theory2 as th, networks2 as nw

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
COMS = ["wheat", "maize", "rice", "soybean"]
AUD = json.load(open(os.path.join(OUT, "net_audit.json")))
KOBS = {c: round(AUD[f"{c}_2021_2023"]["kappa_unfiltered"], 3) for c in COMS}
BASEGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
def kgrid(c):
    return sorted(set(BASEGRID) | {KOBS[c]})
INSTS = [cs.EQUAL, cs.MIXED, cs.PROP, cs.DISTILLED, cs.STEP]
MAIN3 = [cs.EQUAL, cs.MIXED, cs.DISTILLED]
DEF = dict(n=8, f_mean=0.05, f_conc=5.0, phi_lo=0.6, phi_hi=1.0, r=0.4, k=22.0, exec_sd=0.03, prod_sd=0.05, b_recip=0.2)
HAZ = dict(crisis_p=0.001, crisis_dur=20, harvest_p=0.02, harvest_loss=0.5, regen=0.005)
_NET = {}; _PHI = {}


def seed_of(*key):
    return zlib.crc32(repr(key).encode()) & 0x7FFFFFFF


def load(com, tag="2021_2023", kind="balanced", null=None, null_seed=0):
    key = (com, tag, kind, null, null_seed)
    if key not in _NET:
        z = np.load(os.path.join(HERE, "nets", f"{com}_{tag}.npz"))
        W = z["W"]; F = z["F"]; prod = z["prod"]
        if null is not None:
            W = nw.null_model(W, null, np.random.default_rng(null_seed))
        if kind == "balanced":
            net = nw.balanced(W)
        elif kind == "supply":
            net = nw.supply_based(W, F, prod)
        else:
            net = nw.directed(W, F, prod)
        net["N"] = W.shape[0]; net["names"] = list(z["names"]); net["W"] = W
        _NET[key] = net
    return _NET[key]


def phi_sample(name):
    if name not in _PHI:
        _PHI[name] = np.load(os.path.join(HERE, "nets", f"{name}.npy"))
    return _PHI[name]


def agents(net, rng, p, n=None):
    emp = None; groups = None
    src = p.get("phi_src")
    if src == "games":
        g = phi_sample("phi_games"); groups = [row[~np.isnan(row)] for row in g]
    elif src and src.startswith("phi_boot"):
        emp = phi_sample("phi_boot")[int(src.split(":")[1])]
    elif src:
        emp = phi_sample(src)
    return cs.make_agents(net["N"], p["n"] if n is None else n, rng, f_mean=p["f_mean"], f_conc=p["f_conc"],
                          phi_lo=p["phi_lo"], phi_hi=p["phi_hi"], phi_emp=emp, phi_groups=groups)


def seeds_for(net, name, rng):
    order = np.argsort(-net["s"])
    if name.startswith("top"):
        return order[: int(name[3:])]
    if name.startswith("exp"):   # largest directed exporters
        return np.argsort(-net["exports"])[: int(name[3:])]
    if name == "rand5":
        return rng.choice(order[5:], 5, replace=False)
    raise ValueError(name)


def skw(p):
    return dict(r=p["r"], k=p["k"], exec_sd=p["exec_sd"], prod_sd=p["prod_sd"], b_recip=p.get("b_recip", 0.2))


def cmp_counts(pred, obs, mask):
    pred = pred[mask]; obs = obs[mask]
    return dict(tp=int((pred & obs).sum()), fp=int((pred & ~obs).sum()), fn=int((~pred & obs).sum()), tn=int((~pred & ~obs).sum()))


def twin_pair(net, instv, node, phi, fr, kap, shock, T, sd, p, **kw):
    """Shocked run and its unshocked twin with identical random numbers."""
    a = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=T, shock=shock, rng=np.random.default_rng(sd), **skw(p), **kw)
    b = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=T, shock=None, rng=np.random.default_rng(sd), **skw(p), **kw)
    return a, b


def cascade_metrics(a, b, seeds, N, K):
    ns = np.ones(N, bool); ns[np.asarray(seeds, int)] = False
    da = a["x_window"] < 0.5; db = b["x_window"] < 0.5
    new = da & ~db & ns
    return dict(final_dep=float(da.mean()), base_dep=float(db.mean()), cascade=float(new.sum() / ns.sum()),
                final_dep_cap=a["depcap_window"], cascade_cap=float(K[new].sum() / K[ns].sum()),
                short=a["short_window"], short_base=b["short_window"], dep03=a["dep03_window"], dep07=a["dep07_window"],
                final_dep_terminal=a["dep_final"], recovered_terminal=float(np.mean(da & ~(a["x_final"] < 0.5)))), da, db, ns


# ---------------------------------------------------------------- E1: hub failures with twins, reductions, criteria
def task_E1(args):
    com, inst, sset, rep = args
    p = DEF; net = load(com); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E1", com, sset, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, sset, rng)
    instv = np.full(N, inst)
    gf = th.growth_factors(node, phi, instv, r=p["r"], k=p["k"])
    iso = th.criteria(net, 0.0, instv, gf, seeds=[], k=p["k"])  # isolated viability (kappa = 0)
    rows = []
    for kap in kgrid(com):
        sd = seed_of("E1s", com, inst, sset, rep, kap)
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="institutional", nodes=seeds, t0=100), 400, sd, p)
        m, da, db, ns = cascade_metrics(a, b, seeds, N, K)
        row = dict(commodity=com, inst=cs.INST_NAMES[inst], seedset=sset, rep=rep, kappa=kap, efficiency=b["efficiency"],
                   seed_cap_share=float(K[seeds].sum() / K.sum()), margin_median=float(np.median(gf["margin"][np.isfinite(gf["margin"])])),
                   frac_m_neg=float(np.mean(gf["margin"] < 0)), frac_iso_unviable=float(np.mean(iso["deplete"])),
                   lam_full_mean=float(gf["lam_full"].mean()), lam_max_mean=float(gf["lam_max"].mean()), **m)
        for mode in ("interp", "exact"):
            for hz in (None, 300):
                red = th.reduced(net, kap, instv, gf, seeds=seeds, mode=mode, horizon=hz, k=p["k"])
                rd = red["x_win"] < 0.5
                tag = f"{mode}_{'conv' if hz is None else 'h300'}"
                c = cmp_counts(rd, da, np.ones(N, bool))
                row[f"red_{tag}_acc"] = (c["tp"] + c["tn"]) / N; row[f"red_{tag}_dep"] = float(rd.mean())
                row[f"red_{tag}_fp"] = c["fp"]; row[f"red_{tag}_fn"] = c["fn"]; row[f"red_{tag}_pos"] = int(da.sum())
                row[f"red_{tag}_absx"] = float(np.mean(np.abs(red["x_win"] - a["x_window"])))
                if hz is None:
                    row[f"red_{tag}_converged"] = red["converged"]; row[f"red_{tag}_cycle"] = red["cycle"]
        if sset in ("top1", "top5"):
            cr = th.criteria(net, kap, instv, gf, seeds, k=p["k"])
            newobs = da & ~db
            for nm, pred in (("capkeep", ~cr["cap_keep"]), ("reserve", cr["reserve_fail"]), ("deplete", cr["deplete"])):
                for tgt_nm, tgt in (("new", newobs), ("dep", da)):
                    c = cmp_counts(pred & ~db if tgt_nm == "new" else pred, tgt, ns)
                    for kk, vv in c.items():
                        row[f"crit_{nm}_{tgt_nm}_{kk}"] = vv
        rows.append(row)
    return rows


def jobs_E1(reps=20):
    return [(c, i, s, r) for c in COMS for i in INSTS for s in ["top1", "top3", "top5", "rand5"] for r in range(reps)]


# ---------------------------------------------------------------- E1alt: alternative seed semantics
def task_E1alt(args):
    com, inst, kind, rep = args
    p = DEF; net = load(com); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E1", com, "top5", rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, "top5", rng); instv = np.full(N, inst)
    shock = {"inst_perm": dict(kind="institutional", nodes=seeds, t0=100),
             "inst_temp20": dict(kind="institutional", nodes=seeds, t0=100, dur=20),
             "prod_temp20": dict(kind="production", nodes=seeds, t0=100, dur=20, loss=1.0),
             "prod_perm": dict(kind="production", nodes=seeds, t0=100, dur=10 ** 9, loss=1.0),
             "export_halt": dict(kind="export_halt", nodes=seeds, t0=100),
             "isolation": dict(kind="isolation", nodes=seeds, t0=100)}[kind]
    rows = []
    for kap in sorted({0.1, 0.25, 0.4, KOBS[com]}):
        a, b = twin_pair(net, instv, node, phi, fr, kap, shock, 400, seed_of("E1alt", com, inst, kind, rep, kap), p)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], kind=kind, rep=rep, kappa=kap, peak_dep=float(a["ts"]["dep"][100:].max()), **m))
    return rows


def jobs_E1alt(reps=20):
    return [(c, i, k, r) for c in COMS for i in INSTS for k in ["inst_perm", "inst_temp20", "prod_temp20", "prod_perm", "export_halt", "isolation"] for r in range(reps)]


# ---------------------------------------------------------------- E1net: supply-based and directed couplings
def task_E1net(args):
    com, inst, kind, sset, rep = args
    p = DEF; net = load(com, kind=kind); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E1net", com, kind, sset, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, sset, rng); instv = np.full(N, inst)
    kn = net.get("kappa_j") if kind == "supply" else None
    rows = []
    for mult in (0.5, 1.0):
        sd = seed_of("E1nets", com, inst, kind, sset, rep, mult)
        a, b = twin_pair(net, instv, node, phi, fr, mult, dict(kind="institutional", nodes=seeds, t0=100), 400, sd, p, kappa_node=kn)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        ks = np.full(600, mult); ks[300:] = 0.0
        h = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=600, rng=np.random.default_rng(sd + 1), kappa_node=kn, **skw(p))
        vis = float(np.mean(h["ts"]["dep"][250:300])); lat = h["dep_window"]
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], coupling=kind, seedset=sset, rep=rep, mult=mult,
                         seed_cap_share=float(K[seeds].sum() / K.sum()), visible_dep=vis, latent_dep=lat, efficiency=b["efficiency"], **m))
    return rows


def jobs_E1net(reps=20):
    J = [(c, i, "supply", "top5", r) for c in COMS for i in INSTS for r in range(reps)]
    J += [(c, i, "directed", s, r) for c in COMS for i in INSTS for s in ["top5", "exp5"] for r in range(reps)]
    return J


# ---------------------------------------------------------------- E1null: null networks, one realisation per replicate
def task_E1null(args):
    com, inst, null, seedmode, rep = args
    p = DEF; base = load(com)
    net = load(com, null=null, null_seed=seed_of("null", com, null, rep)) if null != "observed" else base
    N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E1null", com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net if seedmode == "rerank" else base, "top5", rng)
    instv = np.full(N, inst); rows = []
    P = net["P"]; q = P.max(1)
    for kap in sorted({KOBS[com], 0.4}):
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="institutional", nodes=seeds, t0=100), 400, seed_of("E1nulls", com, inst, null, seedmode, rep, kap), p)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], null=null, seedmode=seedmode, rep=rep, kappa=kap,
                         seed_cap_share=float(K[seeds].sum() / K.sum()), median_q=float(np.median(q)), **m))
    return rows


def jobs_E1null(reps=20):
    J = []
    for c in COMS:
        for i in [cs.EQUAL, cs.MIXED, cs.DISTILLED]:
            for r in range(reps):
                J.append((c, i, "observed", "rerank", r))
                for nl in ["weights_shuffled", "strength_preserving", "degree_preserving", "erdos_renyi"]:
                    for sm in ["rerank", "fixed"]:
                        J.append((c, i, nl, sm, r))
    return J


# ---------------------------------------------------------------- E2: hidden fragility and welfare decomposition
def task_E2(args):
    com, inst, rep, tag = args
    p = DEF; net = load(com, tag=tag); N = net["N"]
    rng = np.random.default_rng(seed_of("E2", com, rep, tag))
    node, phi, fr, _ = agents(net, rng, p)
    instv = np.full(N, inst)
    gf = th.growth_factors(node, phi, instv, r=p["r"], k=p["k"])
    iso = th.criteria(net, 0.0, instv, gf, seeds=[], k=p["k"])["deplete"]
    rows = []
    for kap in kgrid(com):
        ks = np.full(900, kap); ks[300:] = 0.0
        sd = seed_of("E2s", com, inst, rep, kap, tag)
        h = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=900, rng=np.random.default_rng(sd), window=50, **skw(p))
        o = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=400, rng=np.random.default_rng(sd), **skw(p))
        d = h["ts"]["dep"]
        lat = h["x_window"] < 0.5
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], rep=rep, kappa=kap, tag=tag,
                         visible_dep=float(d[250:300].mean()), latent_dep_300=float(d[550:600].mean()), latent_dep_600=float(d[850:900].mean()),
                         pred_unviable=float(iso.mean()), latent_agree=float(np.mean(lat == iso)),
                         efficiency=o["efficiency"], waste=o["waste"], wmean=o["wmean"], dep_mean=float(o["ts"]["dep"][100:].mean()),
                         **{kk: o[kk] for kk in o if kk.startswith(("gini_", "excl"))}))
    return rows


def jobs_E2(reps=20):
    J = [(c, i, r, "2021_2023") for c in COMS for i in INSTS for r in range(reps)]
    J += [(c, i, r, "2017_2019") for c in COMS for i in MAIN3 for r in range(reps)]
    return J


# ---------------------------------------------------------------- E3/E4: stochastic hazards and export rules
RULES = {"none": None, "ban": dict(theta=0.8, rho=0.0), "halve": dict(theta=0.8, rho=0.5),
         "ban_th0.6": dict(theta=0.6, rho=0.0), "ban_th0.7": dict(theta=0.7, rho=0.0), "ban_th0.9": dict(theta=0.9, rho=0.0),
         "ban_hyst": dict(theta=0.8, rho=0.0, theta_open=0.95), "halve_hyst": dict(theta=0.8, rho=0.5, theta_open=0.95),
         "quarter": dict(theta=0.8, rho=0.75)}


def systemic_stats(ts, burn=100):
    out = {}
    d = ts["dep"][burn:]; dc = ts["depcap"][burn:]
    for thr in (0.4, 0.5, 0.6):
        ev = d > thr
        out[f"sys{int(thr * 100)}_any"] = float(ev.any()); out[f"sys{int(thr * 100)}_occ"] = float(ev.mean())
        out[f"sys{int(thr * 100)}_first"] = float(np.argmax(ev)) if ev.any() else np.nan
    out["syscap50_any"] = float((dc > 0.5).any()); out["syscap50_occ"] = float((dc > 0.5).mean())
    for nm in ("dep03", "dep07"):
        out[f"sys50_{nm}_any"] = float((ts[nm][burn:] > 0.5).any())
    ev = d > 0.5
    starts = np.flatnonzero(ev[1:] & ~ev[:-1]).size + int(ev[0])
    out["sys50_episodes"] = starts
    out["dep_mean"] = float(d.mean()); out["dep_p95"] = float(np.quantile(d, 0.95)); out["dep_max"] = float(d.max())
    out["depcap_mean"] = float(dc.mean()); out["short_mean"] = float(ts["short"][burn:].mean())
    return out


def task_E3(args):
    com, inst, rule, haz, rep = args
    p = DEF; net = load(com); N = net["N"]
    rng = np.random.default_rng(seed_of("E3", com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    instv = np.full(N, inst)
    hz = dict(HAZ)
    if haz == "harvest_only":
        hz["crisis_p"] = 0.0
    elif haz == "crisis_only":
        hz["harvest_p"] = 0.0
    elif haz != "both":
        key, val = haz.split("=")
        hz[key] = float(val)
    rows = []
    kk = kgrid(com) if (rule in ("none", "ban", "halve") and haz == "both") else sorted({0.25, 0.4, 0.5, 0.6, KOBS[com]})
    for kap in kk:
        o = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=1100, burn=100, export_rule=RULES[rule],
                        rng=np.random.default_rng(seed_of("E3s", com, inst, rule, haz, rep, kap)), **skw(p), **hz)
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], rule=rule, hazard=haz, rep=rep, kappa=kap,
                         efficiency=o["efficiency"], waste=o["waste"], **systemic_stats(o["ts"])))
    return rows


def jobs_E3(reps=100):
    J = [(c, i, ru, "both", r) for c in COMS for i in INSTS for ru in ["none", "ban", "halve"] for r in range(reps)]
    J += [(c, i, ru, "both", r) for c in COMS for i in INSTS for ru in ["ban_th0.6", "ban_th0.7", "ban_th0.9", "ban_hyst", "halve_hyst", "quarter"] for r in range(40)]
    J += [(c, i, "none", hz, r) for c in COMS for i in MAIN3 for hz in ["harvest_only", "crisis_only",
          "harvest_p=0.01", "harvest_p=0.04", "harvest_loss=0.3", "harvest_loss=0.7", "crisis_p=0.0005", "crisis_p=0.002",
          "crisis_dur=10", "crisis_dur=40", "regen=0.0", "regen=0.0025", "regen=0.01"] for r in range(40)]
    return J


def task_E4b(args):
    com, inst, rule, sset, rep = args
    p = DEF; net = load(com); N = net["N"]
    rng = np.random.default_rng(seed_of("E4b", com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = np.argsort(-net["s"])[:2]; instv = np.full(N, inst); rows = []
    for kap in sorted({0.1, 0.17, 0.25, 0.35, 0.45, 0.6, KOBS[com]}):
        sd = seed_of("E4bs", com, inst, rule, rep, kap)
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="production", nodes=seeds, t0=100, dur=20, loss=0.5), 400, sd, p,
                         export_rule=RULES[rule], regen=HAZ["regen"])
        loss = 1 - a["ts"]["pay"][100:].sum() / max(b["ts"]["pay"][100:].sum(), 1e-12)
        da = a["ts"]["dep"][100:]
        rows.append(dict(commodity=com, inst=cs.INST_NAMES[inst], rule=rule, rep=rep, kappa=kap, peak_dep=float(da.max()),
                         peak_round=int(np.argmax(da)) + 100, final_dep=a["dep_window"], base_dep=b["dep_window"], welfare_loss=float(loss),
                         recovery_rounds=float(np.argmax(da[np.argmax(da):] <= b["ts"]["dep"][100:][np.argmax(da):] + 1e-9)) if da.max() > 0 else 0.0))
    return rows


def jobs_E4b(reps=20):
    return [(c, i, ru, "top2", r) for c in COMS for i in INSTS for ru in ["none", "ban", "halve", "ban_hyst"] for r in range(reps)]


# ---------------------------------------------------------------- E5: adoption targeting
def task_E5(args):
    com, kap, place, shockset, rep = args
    p = DEF; net = load(com); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E5", com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, "top5", rng) if shockset == "top5" else seeds_for(net, "rand5", np.random.default_rng(seed_of("E5r", com, rep)))
    gf_eq = th.growth_factors(node, phi, np.full(N, cs.EQUAL), r=p["r"])
    m_eq = gf_eq["margin"].copy(); m_eq[~np.isfinite(m_eq)] = -1.0
    erng = np.random.default_rng(seed_of("E5e", com, rep, place))
    if place == "random":
        order = erng.permutation(N)
    elif place == "strength":
        order = np.argsort(-net["s"])
    elif place == "strength_noseed":
        o_ = np.argsort(-net["s"]); order = np.concatenate([o_[~np.isin(o_, seeds)], o_[np.isin(o_, seeds)]])
    elif place == "concentration":
        order = np.argsort(-net["P"].max(1))
    elif place.startswith("fragility"):
        sig = float(place.split(":")[1]) if ":" in place else 0.0
        order = np.argsort(m_eq + erng.normal(0, sig, N))
    rows = []
    for h in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        adopt = order[: int(round(h * N))]
        instv = np.full(N, cs.EQUAL); instv[adopt] = cs.DISTILLED
        sd = seed_of("E5s", com, kap, place, shockset, rep, h)
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="institutional", nodes=seeds, t0=100), 400, sd, p)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        ks = np.full(900, kap); ks[300:] = 0.0
        hh = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=900, rng=np.random.default_rng(sd + 7), **skw(p))
        rows.append(dict(commodity=com, kappa=kap, placement=place, shockset=shockset, rep=rep, h=h,
                         seeds_adopting=int(np.isin(seeds, adopt).sum()), adopter_cap_share=float(K[adopt].sum() / K.sum()),
                         latent_dep=hh["dep_window"], visible_dep=float(hh["ts"]["dep"][250:300].mean()), efficiency=b["efficiency"], **m))
    return rows


def jobs_E5(reps=20):
    J = []
    for c in COMS:
        for kap in sorted({KOBS[c], 0.4}):
            for pl in ["random", "strength", "strength_noseed", "concentration", "fragility", "fragility:0.02", "fragility:0.05", "fragility:0.1"]:
                for r in range(reps):
                    J.append((c, kap, pl, "top5", r))
            for pl in ["random", "strength", "fragility"]:
                for r in range(reps):
                    J.append((c, kap, pl, "rand5", r))
    return J


# ---------------------------------------------------------------- E6: mobility
def task_E6(args):
    com, h_eq, mig, who, f_mean, rep = args
    p = dict(DEF, f_mean=f_mean); net = load(com); N = net["N"]; K = net["K"]
    kap = KOBS[com]
    rng = np.random.default_rng(seed_of("E6", com, f_mean, rep))
    nvec = np.maximum(4, np.round(8 * K / np.median(K))).astype(int)
    node, phi, fr, _ = agents(net, rng, p, n=nvec)
    eq = rng.random(N) < h_eq
    instv = np.where(eq, cs.EQUAL, cs.DISTILLED)
    o = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=2000, burn=1000, record=True, epoch=10, migration=mig, mig_who=who,
                    rng=np.random.default_rng(seed_of("E6s", com, h_eq, mig, who, f_mean, rep)), **skw(p))
    nf, frf = o["node_final"], o["fr_final"]
    cnt = np.bincount(nf, minlength=N).astype(float); nfr = np.bincount(nf, weights=frf.astype(float), minlength=N)
    cnt0 = np.bincount(node, minlength=N).astype(float); nfr0 = np.bincount(node, weights=fr.astype(float), minlength=N)
    occ = cnt > 0; xw = o["x_window"]
    row = dict(commodity=com, h_eq=h_eq, migration=mig, who=who, f_mean=f_mean, rep=rep, kappa=kap, efficiency=o["efficiency"],
               dep_all=o["dep_window"], dep_occupied=o["dep_window_occupied"], depcap=o["depcap_window"], empty=o["empty_nodes"],
               empty_cap_share=float(K[~occ].sum() / K.sum()))
    for nm, mask in (("eq", eq), ("d", ~eq)):
        if mask.any():
            row[f"{nm}_fr0_pooled"] = nfr0[mask].sum() / max(cnt0[mask].sum(), 1); row[f"{nm}_fr_pooled"] = nfr[mask].sum() / max(cnt[mask].sum(), 1)
            mo = mask & occ
            row[f"{nm}_fr_country_mean"] = float(np.mean(nfr[mo] / cnt[mo])) if mo.any() else np.nan
            row[f"{nm}_fr0_country_mean"] = float(np.mean(nfr0[mask] / cnt0[mask]))
            row[f"{nm}_dep"] = float(np.mean(xw[mask] < 0.5)); row[f"{nm}_dep_occupied"] = float(np.mean(xw[mo] < 0.5)) if mo.any() else np.nan
            row[f"{nm}_pop"] = cnt[mask].sum() / cnt.sum()
    return [row]


def jobs_E6(reps=20):
    J = [("wheat", h, m, who, f, r) for h in [0.0, 0.25, 0.5, 0.75, 1.0] for m in [0.0, 0.05, 0.2] for who in ["freeriders", "all"] for f in [0.05, 0.1] for r in range(reps)]
    J += [(c, h, m, "freeriders", f, r) for c in ["maize", "rice", "soybean"] for h in [0.0, 0.5, 1.0] for m in [0.0, 0.05] for f in [0.05, 0.1] for r in range(reps)]
    return J


# ---------------------------------------------------------------- E7: behavioural and parameter variants
def variants():
    V = [("base", DEF, "2021_2023", "B1", {})]
    V.append(("period_2017_2019", DEF, "2017_2019", "B1", {}))
    for nm in ["2021", "2022", "2023"]:
        V.append((f"year_{nm}", DEF, nm, "B1", {}))
    for nm in ["nofilter", "light", "heavy", "iterative"]:
        V.append((f"filter_{nm}", DEF, f"2021_2023_{nm}", "B1", {}))
    for b in [0.0057, 0.012, 0.05, 0.2]:
        V.append((f"B2_b{b}", dict(DEF, b_recip=b), "2021_2023", "B2", {}))
    V.append(("B3_base", DEF, "2021_2023", "B3", {}))
    V.append(("B3_burn3000", DEF, "2021_2023", "B3", dict(pre=3000)))
    for ib in [2.0, 10.0]:
        V.append((f"B3_beta{ib}", DEF, "2021_2023", "B3", dict(imit_beta=ib)))
    for ms in [0.005, 0.02]:
        V.append((f"B3_mut{ms}", DEF, "2021_2023", "B3", dict(mut_sd=ms)))
    for ep in [10, 40]:
        V.append((f"B3_epoch{ep}", DEF, "2021_2023", "B3", dict(epoch=ep)))
    V.append(("B3_normpay", DEF, "2021_2023", "B3", dict(imit_norm=True)))
    for src, nm in [("phi_emp", "human_raw"), ("phi_shrunk", "human_shrunk"), ("phi_equalcond", "human_equalcond"), ("phi_rlcond", "human_rlcond"), ("games", "human_games")]:
        for n in [4, 8, 16]:
            if nm != "human_raw" and n != 8:
                continue
            V.append((f"{nm}_n{n}", dict(DEF, f_mean=0.0, phi_src=src, n=n), "2021_2023", "B1", {}))
    V.append(("human_raw_noexec", dict(DEF, f_mean=0.0, phi_src="phi_emp", exec_sd=0.0), "2021_2023", "B1", {}))
    for bi in range(20):
        V.append((f"human_boot{bi}", dict(DEF, f_mean=0.0, phi_src=f"phi_boot:{bi}"), "2021_2023", "B1", dict(reps=5)))
    for r_ in [0.2, 0.3, 0.6]:
        V.append((f"r_{r_}", dict(DEF, r=r_), "2021_2023", "B1", {}))
    for k_ in [5.0, 10.0, 50.0]:
        V.append((f"k_{int(k_)}", dict(DEF, k=k_), "2021_2023", "B1", {}))
    for n in [4, 16]:
        V.append((f"n_{n}", dict(DEF, n=n), "2021_2023", "B1", {}))
    for f in [0.0, 0.025, 0.1]:
        V.append((f"f_{f}", dict(DEF, f_mean=f), "2021_2023", "B1", {}))
    V.append(("phi_0.72_0.92", dict(DEF, phi_lo=0.72, phi_hi=0.92), "2021_2023", "B1", {}))
    V.append(("phi_0.5_1.0", dict(DEF, phi_lo=0.5, phi_hi=1.0), "2021_2023", "B1", {}))
    for sd_ in [0.02, 0.10]:
        V.append((f"prod_sd_{sd_}", dict(DEF, prod_sd=sd_), "2021_2023", "B1", {}))
    for es in [0.0, 0.06]:
        V.append((f"exec_sd_{es}", dict(DEF, exec_sd=es), "2021_2023", "B1", {}))
    return V


VDICT = {v[0]: v for v in variants()}


def task_E7(args):
    vname, com, inst, rep = args
    _, p, tag, beh, extra = VDICT[vname]
    net = load(com, tag=tag); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E7", vname, com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    instv = np.full(N, inst); seeds = seeds_for(net, "top5", rng)
    pre = extra.get("pre", 1500) if beh == "B3" else 0
    bkw = {kk: extra[kk] for kk in ("imit_beta", "mut_sd", "epoch", "imit_norm") if kk in extra}
    rows = []
    for kap in sorted({KOBS[com], 0.4}):
        sd = seed_of("E7s", vname, com, inst, rep, kap)
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="institutional", nodes=seeds, t0=pre + 100), pre + 400, sd, p,
                         behaviour=beh, burn=pre + 100, record_evo=(beh == "B3"), **bkw)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        ks = np.full(pre + 900, kap); ks[pre + 300:] = 0.0
        hh = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=pre + 900, burn=pre + 100, behaviour=beh, rng=np.random.default_rng(sd + 3), **skw(p), **bkw)
        phi_now = b["phi_final"] if beh == "B3" else phi
        node_now = b["node_final"] if beh == "B3" else node
        gf = th.growth_factors(node_now, phi_now, instv, r=p["r"], k=p["k"])
        iso = th.criteria(net, 0.0, instv, gf, seeds=[], k=p["k"])["deplete"]
        mm = gf["margin"][np.isfinite(gf["margin"])]
        row = dict(variant=vname, behaviour=beh, commodity=com, inst=cs.INST_NAMES[inst], rep=rep, kappa=kap,
                   visible_dep=float(hh["ts"]["dep"][pre + 250:pre + 300].mean()), latent_dep=hh["dep_window"],
                   efficiency=b["efficiency"], lam_full_mean=float(gf["lam_full"].mean()), lam_max_mean=float(gf["lam_max"].mean()),
                   margin_median=float(np.median(mm)) if mm.size else np.nan, frac_m_neg=float(np.mean(gf["margin"] < 0)),
                   frac_iso_unviable=float(iso.mean()), phi_coop_mean=float(phi_now[~fr].mean()), phi_max_mean=float(np.mean([phi_now[node_now == j].max() for j in range(N)])), **m)
        if beh == "B3":
            ev = b["evo"]
            for t_ in (100, 500, 1000, 1500):
                ii = np.argmin(np.abs(ev[:, 0] - t_))
                row[f"evo_phimean_{t_}"] = ev[ii, 1]; row[f"evo_phimax_{t_}"] = ev[ii, 2]; row[f"evo_dep_{t_}"] = ev[ii, 3]
            row["evo_dphimean_last500"] = float(ev[ev[:, 0] >= pre - 500, 1][-1] - ev[ev[:, 0] >= pre - 500, 1][0])
        rows.append(row)
    return rows


def jobs_E7(reps=20):
    J = []
    for vname, p, tag, beh, extra in variants():
        coms = COMS
        nrep = extra.get("reps", reps)
        insts = INSTS if not vname.startswith(("human_boot", "B3_", "year_", "filter_")) or vname == "B3_base" else MAIN3
        for c in coms:
            for i in insts:
                for r in range(nrep):
                    J.append((vname, c, i, r))
    return J


# ---------------------------------------------------------------- E8: institutional ablation
ABL = {"distilled": None}
for th_ in [0.8, 0.9, 0.95, 0.969, 0.99]:
    ABL[f"step_th{th_}"] = dict(shape="step", theta=th_)
for k_ in [2, 5, 10, 22, 50, 100]:
    ABL[f"power_k{k_}"] = dict(shape="power", k=k_)
for tau in [0.002, 0.01, 0.03]:
    ABL[f"logistic_tau{tau}"] = dict(shape="logistic", theta=0.969, tau=tau)
for wl in [0.1, 0.25, 0.5]:
    ABL[f"wlo{wl}"] = dict(shape="power", k=22, w_lo=wl)
for wh in [0.5, 0.75]:
    ABL[f"whi{wh}"] = dict(shape="power", k=22, w_hi=wh)
for wf in [0.05, 0.1, 0.25]:
    ABL[f"fixed{wf}"] = dict(shape="power", k=22, w_lo=wf, w_hi=wf)
for fl in [0.05, 0.1, 0.2]:
    ABL[f"floor{fl}"] = dict(shape="power", k=22, floor=fl)


def task_E8(args):
    rname, com, rep = args
    p = DEF; net = load(com); N = net["N"]; K = net["K"]
    rng = np.random.default_rng(seed_of("E8", com, rep))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, "top5", rng)
    rule = ABL[rname]
    instv = np.full(N, cs.DISTILLED if rule is None else cs.GENERIC)
    gf = th.growth_factors(node, phi, instv, r=p["r"], k=p["k"], rule=rule)
    rows = []
    for kap in sorted({KOBS[com], 0.4}):
        sd = seed_of("E8s", rname, com, rep, kap)
        a, b = twin_pair(net, instv, node, phi, fr, kap, dict(kind="institutional", nodes=seeds, t0=100), 400, sd, p, rule=rule)
        m, *_ = cascade_metrics(a, b, seeds, N, K)
        ks = np.full(900, kap); ks[300:] = 0.0
        hh = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=900, rule=rule, rng=np.random.default_rng(sd + 5), **skw(p))
        st = cs.simulate(net, instv, node, phi, fr, kappa=kap, T=1100, burn=100, rule=rule, rng=np.random.default_rng(sd + 9), **skw(p), **HAZ)
        rows.append(dict(rule=rname, commodity=com, rep=rep, kappa=kap, efficiency=b["efficiency"], waste=b["waste"], wmean=b["wmean"],
                         latent_dep=hh["dep_window"], margin_median=float(np.median(gf["margin"][np.isfinite(gf["margin"])])),
                         sys50_any=float((st["ts"]["dep"][100:] > 0.5).any()), haz_dep_mean=float(st["ts"]["dep"][100:].mean()),
                         **{kk: b[kk] for kk in b if kk.startswith(("gini_", "excl"))}, **m))
    return rows


def jobs_E8(reps=20):
    return [(rn, c, r) for rn in ABL for c in COMS for r in range(reps)]


# ---------------------------------------------------------------- examples for figures
def examples():
    p = DEF; ex = {}
    net = load("wheat"); N = net["N"]
    rng = np.random.default_rng(seed_of("E1", "wheat", "top5", 0))
    node, phi, fr, _ = agents(net, rng, p)
    seeds = seeds_for(net, "top5", rng)
    for inst in MAIN3:
        instv = np.full(N, inst)
        o = cs.simulate(net, instv, node, phi, fr, kappa=KOBS["wheat"], T=400, record=True, shock=dict(kind="institutional", nodes=seeds, t0=100),
                        rng=np.random.default_rng(1), **skw(p))
        ex[f"casc_{inst}"] = o["ts"]["dep"]
        ks = np.full(900, 0.4); ks[300:] = 0.0
        o = cs.simulate(net, instv, node, phi, fr, kappa=ks, T=900, record=True, rng=np.random.default_rng(2), **skw(p))
        ex[f"halt_{inst}"] = o["ts"]["dep"]
        gf = th.growth_factors(node, phi, instv, r=p["r"])
        ex[f"margin_{inst}"] = gf["margin"]; ex[f"lamfull_{inst}"] = gf["lam_full"]; ex[f"lammax_{inst}"] = gf["lam_max"]
        ex[f"curves_{inst}"] = gf["curves"]
        cr = th.criteria(net, KOBS["wheat"], instv, gf, seeds, k=p["k"])
        ex[f"crit_q_{inst}"] = cr["q"]; ex[f"crit_xend_{inst}"] = cr["x_end"]
    ex["seeds"] = seeds; ex["fr_node"] = np.bincount(node, weights=fr.astype(float), minlength=N); ex["wgrid"] = th.WGRID
    # export-ban instability trace (wheat, kappa 0.6, equal sharing, stochastic hazards)
    o = cs.simulate(net, np.full(N, cs.EQUAL), node, phi, fr, kappa=0.6, T=600, burn=100, record=True, export_rule=RULES["ban"],
                    rng=np.random.default_rng(3), **skw(p), **HAZ)
    ex["ban_R"] = o["R_ts"]; ex["ban_dep"] = o["ts"]["dep"]
    o = cs.simulate(net, np.full(N, cs.EQUAL), node, phi, fr, kappa=0.6, T=600, burn=100, record=True, export_rule=None,
                    rng=np.random.default_rng(3), **skw(p), **HAZ)
    ex["open_R"] = o["R_ts"]; ex["open_dep"] = o["ts"]["dep"]
    # worked trajectory of one isolated community (8 members, one free rider)
    one = dict(mode="balanced", P=np.ones((1, 1)), K=np.array([200.0]))
    nd = np.zeros(8, int); ph1 = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.0]); fr1 = ph1 == 0
    for inst in (cs.EQUAL, cs.DISTILLED, cs.STEP):
        o = cs.simulate(one, np.array([inst]), nd, ph1, fr1, kappa=0.0, T=120, exec_sd=0.0, prod_sd=0.0, record=True, record_agents=True,
                        shock=dict(kind="production", nodes=[0], t0=30, dur=10, loss=0.6), rng=np.random.default_rng(0))
        ex[f"worked_{inst}_agents"] = o["agents"]; ex[f"worked_{inst}_R"] = o["R_ts"][:, 0]
    np.savez_compressed(os.path.join(OUT, "examples2.npz"), **{k: np.asarray(v) for k, v in ex.items()})


TASKS = dict(E1=(task_E1, jobs_E1), E1alt=(task_E1alt, jobs_E1alt), E1net=(task_E1net, jobs_E1net), E1null=(task_E1null, jobs_E1null),
             E2=(task_E2, jobs_E2), E3=(task_E3, jobs_E3), E4b=(task_E4b, jobs_E4b), E5=(task_E5, jobs_E5), E6=(task_E6, jobs_E6),
             E7=(task_E7, jobs_E7), E8=(task_E8, jobs_E8))


def _dispatch(item):
    name, a = item
    try:
        return TASKS[name][0](a)
    except Exception as ex:
        import traceback
        return [dict(error=repr(ex) + " | " + traceback.format_exc().splitlines()[-3], args=repr(a))]


if __name__ == "__main__":
    argv = sys.argv[1:]; procs = 150
    if "--procs" in argv:
        i = argv.index("--procs"); procs = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    quick = "--quick" in argv; argv = [a for a in argv if a != "--quick"]
    for e in argv:
        t0 = time.time()
        if e == "EX":
            examples(); print("EX", round(time.time() - t0, 1), flush=True); continue
        fn, jf = TASKS[e]
        jobs = jf()
        if quick:
            rs = np.random.default_rng(0); jobs = [jobs[i] for i in rs.choice(len(jobs), min(len(jobs), procs), replace=False)]
        rows = []
        with Pool(procs) as pool:
            for r in pool.imap_unordered(_dispatch, [(e, j) for j in jobs], chunksize=1):
                rows.extend(r)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"{e}{'_quick' if quick else ''}.csv.gz"), index=False, compression="gzip")
        nerr = int(df["error"].notna().sum()) if "error" in df else 0
        print(e, len(jobs), "tasks", len(rows), "rows", nerr, "errors", round(time.time() - t0, 1), "s", flush=True)
        if nerr:
            print(df[df["error"].notna()].head(3).to_string(), flush=True)
