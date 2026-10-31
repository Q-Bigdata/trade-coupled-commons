"""Staple trade networks from FAOSTAT (revision 2), with a construction audit and network variants.

Pipeline per commodity and period (all quantities in tonnes per year):
  1. item harmonisation: rice items are converted to milled equivalent BEFORE aggregation
     (paddy 27 x 0.667, husked 28 x 0.667/0.80, milled 31 x 1, broken 32 x 1); other crops use one item.
  2. mirror reconciliation per directed pair (exporter e, importer i) and year: mean of the importer-reported
     import quantity (element 5610) and the exporter-reported export quantity (element 5910) when both
     exist, the single report otherwise.  Pair-years without any report count as zero in the period mean.
  3. period mean over the years (sum / number of years).
  4. filters: directed flows below min_edge are dropped; then countries whose total (exports + imports)
     is below min_node are dropped (once, or iteratively until stable); then the largest connected
     component of the symmetrised graph is kept.
  5. W = F + F^T (symmetrised trade intensity), strength s_j = sum_k W_jk, P_jk = W_jk / s_j.
Countries: FAOSTAT area codes < 5000, excluding the aggregate 'China' (351); 'China, mainland' (41),
Taiwan, Hong Kong SAR and Macao SAR are separate entities.
"""
import json
import numpy as np
import pandas as pd
import networkx as nx

ITEMS = {"wheat": {15: 1.0}, "maize": {56: 1.0}, "rice": {27: 0.667, 28: 0.667 / 0.80, 31: 1.0, 32: 1.0}, "soybean": {236: 1.0}}
PROD_ITEMS = {"wheat": {15: 1.0}, "maize": {56: 1.0}, "rice": {27: 0.667}, "soybean": {236: 1.0}}
EXCLUDE = {351}


def reconcile(tm, com, years):
    fac = ITEMS[com]
    d = tm[tm["Item Code"].isin(fac) & tm["Year"].isin(years)].copy()
    d = d[(d["Reporter Country Code"] < 5000) & (d["Partner Country Code"] < 5000)]
    d = d[~d["Reporter Country Code"].isin(EXCLUDE) & ~d["Partner Country Code"].isin(EXCLUDE)]
    d["q"] = d["Value"] * d["Item Code"].map(fac)
    imp = d[d["Element Code"] == 5610].rename(columns={"Reporter Country Code": "i", "Partner Country Code": "e"})
    exp = d[d["Element Code"] == 5910].rename(columns={"Reporter Country Code": "e", "Partner Country Code": "i"})
    imp = imp.groupby(["e", "i", "Year"])["q"].sum().rename("imp")
    exp = exp.groupby(["e", "i", "Year"])["q"].sum().rename("exp")
    m = pd.concat([imp, exp], axis=1)
    m = m[m.index.get_level_values("e") != m.index.get_level_values("i")]
    m["both"] = m.imp.notna() & m.exp.notna()
    m["flow"] = m[["imp", "exp"]].mean(axis=1, skipna=True)
    m["disc"] = np.where(m.both, (m.imp - m.exp).abs() / m[["imp", "exp"]].mean(axis=1), np.nan)
    f = (m.groupby(["e", "i"])["flow"].sum() / len(years)).reset_index()
    f = f[f.flow > 0]
    names = {}
    for a, b in [("Reporter Country Code", "Reporter Countries"), ("Partner Country Code", "Partner Countries")]:
        names.update(dict(zip(d[a], d[b])))
    stats = dict(pair_years=int(len(m)), share_pair_years_single=float(1 - m.both.mean()),
                 mass_share_single=float(m.loc[~m.both, "flow"].sum() / m.flow.sum()),
                 median_rel_discrepancy_dual=float(np.nanmedian(m.disc)),
                 mass_raw_Mt=float(f.flow.sum() / 1e6), pairs_raw=int(len(f)))
    return f, names, stats


def build(f, names, min_edge=1000.0, min_node=10000.0, iterative=False):
    st = {}
    tot0 = pd.concat([f.groupby("e").flow.sum(), f.groupby("i").flow.sum()], axis=1).fillna(0).sum(1)
    st["countries_raw"] = int(len(tot0))
    g = f[f.flow >= min_edge]
    st["pairs_after_edge"] = int(len(g)); st["mass_after_edge"] = float(g.flow.sum() / f.flow.sum())
    while True:
        tot = pd.concat([g.groupby("e").flow.sum(), g.groupby("i").flow.sum()], axis=1).fillna(0).sum(1)
        keep = set(tot[tot >= min_node].index)
        g2 = g[g.e.isin(keep) & g.i.isin(keep)]
        if not iterative or len(g2) == len(g):
            g = g2; break
        g = g2
    st["countries_after_node"] = int(len(keep)); st["mass_after_node"] = float(g.flow.sum() / f.flow.sum())
    G = nx.Graph()
    for e, i, w in g[["e", "i", "flow"]].itertuples(index=False):
        G.add_edge(e, i, w=G[e][i]["w"] + w if G.has_edge(e, i) else w)
    cc = max(nx.connected_components(G), key=len)
    codes = sorted(cc); idx = {c: n for n, c in enumerate(codes)}; N = len(codes)
    F = np.zeros((N, N))
    for e, i, w in g[["e", "i", "flow"]].itertuples(index=False):
        if e in idx and i in idx:
            F[idx[e], idx[i]] += w
    W = F + F.T
    st["countries_final"] = N; st["links_final"] = int((np.triu(W, 1) > 0).sum())
    st["mass_final"] = float(F.sum() / f.flow.sum()); st["mass_final_Mt"] = float(F.sum() / 1e6)
    # partner concentration in the final network versus the same countries before edge pruning
    sub = f[f.e.isin(codes) & f.i.isin(codes)]
    F0 = np.zeros((N, N))
    for e, i, w in sub[["e", "i", "flow"]].itertuples(index=False):
        F0[idx[e], idx[i]] += w
    W0 = F0 + F0.T
    q = (W / W.sum(1, keepdims=True)).max(1); q0 = (W0 / W0.sum(1, keepdims=True)).max(1)
    st["median_q_final"] = float(np.median(q)); st["median_q_unpruned_same_countries"] = float(np.median(q0))
    return dict(codes=np.array(codes), names=np.array([names.get(c, str(c)) for c in codes], dtype="U80"), W=W, F=F), st


def production(qcl, com, years):
    fac = PROD_ITEMS[com]
    q = qcl[qcl["Item Code"].isin(fac) & qcl.Year.isin(years) & (qcl["Element Code"] == 5510)]
    q = q[(q["Area Code"] < 5000) & ~q["Area Code"].isin(EXCLUDE)].copy()
    q["v"] = q.Value * q["Item Code"].map(fac)
    return q.groupby("Area Code").v.sum() / len(years)


def null_model(W, kind, rng, nbins=20):
    """Null weighted networks.  'weights_shuffled' keeps adjacency, permutes weights over links;
    'degree_preserving' rewires by connected double-edge swaps and permutes weights; 'strength_preserving'
    swaps only links of similar weight (same log-weight bin) and moves each weight with its link-end, so
    degrees are exact and strengths are approximately preserved; 'erdos_renyi' same density, permuted
    weights; 'complete' uniform weights."""
    N = W.shape[0]
    iu = np.triu_indices(N, 1); wts = W[iu]; mask = wts > 0; ew = wts[mask]
    if kind == "weights_shuffled":
        out = np.zeros_like(W); vals = wts.copy(); vals[mask] = rng.permutation(ew); out[iu] = vals
        return out + out.T
    if kind in ("degree_preserving", "strength_preserving"):
        edges = [(int(a), int(b)) for a, b in zip(iu[0][mask], iu[1][mask])]
        wd = {e: w for e, w in zip(edges, ew)}
        if kind == "degree_preserving":
            G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges)
            nx.connected_double_edge_swap(G, nswap=5 * len(edges), seed=int(rng.integers(1 << 30)))
            out = np.zeros_like(W); E = np.array(G.edges())
            out[E[:, 0], E[:, 1]] = rng.permutation(ew); return out + out.T
        lw = np.log(ew); bins = np.quantile(lw, np.linspace(0, 1, nbins + 1)); bidx = np.clip(np.searchsorted(bins, lw, side="right") - 1, 0, nbins - 1)
        E = np.array(edges); wv = ew.copy()
        adj = {i: set() for i in range(N)}
        for a, b in edges:
            adj[a].add(b); adj[b].add(a)
        groups = [np.where(bidx == g)[0] for g in range(nbins)]
        for _ in range(10 * len(edges)):
            g = groups[rng.integers(nbins)]
            if g.size < 2:
                continue
            e1, e2 = rng.choice(g, 2, replace=False)
            a, b = E[e1]; c, d = E[e2]
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4 or d in adj[a] or b in adj[c]:
                continue
            adj[a].discard(b); adj[b].discard(a); adj[c].discard(d); adj[d].discard(c)
            adj[a].add(d); adj[d].add(a); adj[c].add(b); adj[b].add(c)
            E[e1] = (a, d); E[e2] = (c, b)
        out = np.zeros_like(W); out[E[:, 0], E[:, 1]] = wv; out = out + out.T
        Gc = nx.from_numpy_array((out > 0).astype(int))
        if not nx.is_connected(Gc):
            comps = sorted(nx.connected_components(Gc), key=len, reverse=True)
            for comp in comps[1:]:
                u = next(iter(comp)); v = next(iter(comps[0]))
                out[u, v] = out[v, u] = np.median(ew)
        return out
    if kind == "erdos_renyi":
        E_n = int(mask.sum())
        while True:
            G = nx.gnm_random_graph(N, E_n, seed=int(rng.integers(1 << 30)))
            if nx.is_connected(G):
                break
        out = np.zeros_like(W); E = np.array(G.edges()); out[E[:, 0], E[:, 1]] = rng.permutation(ew)
        return out + out.T
    if kind == "complete":
        out = np.ones_like(W); np.fill_diagonal(out, 0); return out
    raise ValueError(kind)


def balanced(W, Kscale=200.0):
    s = W.sum(1); P = W / s[:, None]; K = Kscale * s / s.mean()
    return dict(mode="balanced", P=P, K=K, s=s)


def supply_based(W, F, prod_t, kmax=0.9, Kscale=200.0):
    """Balanced coupling with capacities from national availability and country-specific traded shares.
    Availability S_j = production + imports - exports (t/yr), floored at s_j / (2 kmax); traded share
    kappa_j = (s_j / 2) / S_j.  Balance holds because kappa_j K_j = s_j / 2 for every j."""
    s = W.sum(1); imp = F.sum(0); exp = F.sum(1)
    S = np.maximum(prod_t + imp - exp, s / (2 * kmax))
    kap = (s / 2) / S
    P = W / s[:, None]
    K = Kscale * S / S.mean()
    return dict(mode="balanced", P=P, K=K, s=s, kappa_j=kap, kappa_bar=float((s / 2).sum() / S.sum()))


def directed(W, F, prod_t, dmax=0.95, Kscale=200.0):
    """Directed supply-share coupling: d_j = imports / (production + imports), M_jk = F_kj / imports_j."""
    imp = F.sum(0)
    d = np.where(imp > 0, imp / np.maximum(prod_t + imp, 1e-9), 0.0); d = np.minimum(d, dmax)
    M = np.where(imp[:, None] > 0, F.T / np.maximum(imp[:, None], 1e-9), 0.0)
    s = W.sum(1); P = W / s[:, None]
    avail = prod_t + imp
    K = Kscale * np.maximum(avail, 1.0) / np.maximum(avail, 1.0).mean()
    return dict(mode="directed", P=P, K=K, s=s, d=d, M=M, exports=F.sum(1), imports=imp)
