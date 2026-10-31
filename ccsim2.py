"""Coupled commons, revision 2.

Each country j holds a resource pool R_j (capacity K_j) shared among n_j members by an allocation rule
(the common-pool trust game of Koster et al. 2025, Nat. Commun. 16, 2824).  Countries are coupled by trade.

Within-round order (all countries update synchronously):
  1. institutional state: seed shocks and stochastic crises switch members to zero return; a crisis that
     arrives while another is active is ignored (no restart, no extension); original return fractions
     resume when the crisis ends
  2. allocation: offers o_i = R_j [w/n_j + (1-w) c_i,t-1 / sum_l c_l,t-1], w = W(R_j/K_j); equal shares in
     round 1 and whenever the previous returns of the community sum to zero
  3. returns: c_i = phi_i,t o_i with phi_i,t = clip(phi_i [+ b(o_i n_j/R_j - 1)] + execution noise, 0, 1),
     forced to 0 for committed free riders and members in crisis (zero status overrides noise)
  4. regeneration: G_j = (1+r) eps_j sum_i c_i + g0 K_j; eps_j = lognormal noise x (1 - harvest loss) x
     (1 - production shock)
  5. trade (balanced mode): exports X_j = kappa_j,t G_j, kappa_j,t = kappa_t a_j rho_j,t, where a_j is the
     node-specific exposure multiplier (1 in the baseline) and rho_j,t the export-restriction factor
     evaluated on the pool at the START of the round (rho = rho_restr while R_j < theta K_j; with
     hysteresis a restricted country reopens only once R_j >= theta_open K_j).  Withheld output stays at
     home: R_j' = min(K_j, G_j - X_j + sum_k P_kj X_k).
     directed mode: R_j' = min(K_j, (1 - s_t d_j) G_j + s_t d_j K_j sum_k M_jk G_k / K_k), where d_j is
     the import share of national availability and M_jk the share of j's imports from k (supply-share
     coupling; exporters are not drawn down).
  6. measurement: depletion indicators at the end of the round.
"""
import numpy as np

EQUAL, MIXED, PROP, DISTILLED, STEP, GENERIC = 0, 1, 2, 3, 4, 5
INST_NAMES = {EQUAL: "Equal", MIXED: "Mixed", PROP: "Proportional", DISTILLED: "Distilled", STEP: "Step", GENERIC: "Generic"}


def weight(inst, x, k=22.0, step_thr=0.969, rule=None):
    """Equal-sharing weight w = W(x) for every country; x = R/K.
    GENERIC rule dict: w = max(floor, w_lo + (w_hi - w_lo) S(x)), with S(x) = x**k_g (shape 'power'),
    1[x >= theta] (shape 'step') or a logistic of width tau around theta (shape 'logistic')."""
    x = np.clip(x, 0, 1)
    w = np.ones_like(x)
    w = np.where(inst == MIXED, 0.5, w)
    w = np.where(inst == PROP, 0.0, w)
    w = np.where(inst == DISTILLED, x ** k, w)
    w = np.where(inst == STEP, (x >= step_thr).astype(float), w)
    if rule is not None and np.any(inst == GENERIC):
        shp = rule.get("shape", "power")
        if shp == "power":
            S = x ** rule.get("k", 22.0)
        elif shp == "step":
            S = (x >= rule.get("theta", 0.969)).astype(float)
        else:
            S = 1.0 / (1.0 + np.exp(-(x - rule.get("theta", 0.969)) / rule.get("tau", 0.01)))
        wg = rule.get("w_lo", 0.0) + (rule.get("w_hi", 1.0) - rule.get("w_lo", 0.0)) * S
        wg = np.maximum(wg, rule.get("floor", 0.0))
        w = np.where(inst == GENERIC, wg, w)
    return w


def make_agents(N, n, rng, f_mean=0.05, f_conc=5.0, phi_lo=0.6, phi_hi=1.0, f_node=None, phi_emp=None, phi_groups=None):
    """Members of every country.  Baseline: committed free riders with country propensity
    f_j ~ Beta(f_mean nu, (1-f_mean) nu); other members phi ~ U(lo, hi).  phi_emp: draw phi i.i.d. from an
    empirical sample.  phi_groups: list of arrays (e.g. the four players of one human game); communities
    are then filled with whole groups, preserving within-game dependence."""
    nvec = np.full(N, int(n)) if np.ndim(n) == 0 else np.asarray(n, int)
    node = np.repeat(np.arange(N), nvec)
    if f_node is None:
        f_node = np.zeros(N) if f_mean <= 0 else rng.beta(f_mean * f_conc, (1 - f_mean) * f_conc, N)
    fr = rng.random(node.size) < f_node[node]
    if phi_groups is not None:
        phi = np.empty(node.size)
        pos = 0
        for j in range(N):
            vals = []
            while len(vals) < nvec[j]:
                vals.extend(phi_groups[rng.integers(len(phi_groups))])
            phi[pos:pos + nvec[j]] = np.asarray(vals[:nvec[j]]); pos += nvec[j]
    elif phi_emp is not None:
        phi = rng.choice(phi_emp, node.size, replace=True)
    else:
        phi = rng.uniform(phi_lo, phi_hi, node.size)
    phi[fr] = 0.0
    return node, phi, fr, f_node


def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2 * (idx * x).sum() / (n * x.sum()) - (n + 1) / n)


def simulate(net, inst, node, phi, fr, *, T=400, kappa=0.3, r=0.4, k=22.0, rule=None,
             behaviour="B1", b_recip=0.2, exec_sd=0.03, prod_sd=0.05,
             shock=None, crisis_p=0.0, crisis_dur=20, harvest_p=0.0, harvest_loss=0.5, regen=0.0,
             export_rule=None, burn=100, window=50, late=100,
             epoch=20, imit_beta=5.0, mut_sd=0.01, imit_norm=False,
             migration=0.0, mig_beta=2.0, mig_who="all",
             record=False, record_evo=False, record_agents=False, rng=None, kappa_node=None):
    """One trajectory.  net: dict with P (N x N), K (N,), and for directed mode d (N,), M (N x N).
    kappa: scalar, or a length-T schedule (balanced: traded share; directed: multiplier s_t of d).
    shock kinds: 'institutional' (seed members free ride from t0, for dur rounds or permanently),
      'production' (seed regeneration x (1-loss) for dur rounds), 'export_halt' (seeds stop exporting from
      t0, keep producing and importing), 'isolation' (from t0 seeds neither export nor import; partners
      keep the output they would have sent to the seeds).
    export_rule: dict(theta, rho, theta_open=None)."""
    rng = np.random.default_rng() if rng is None else rng
    P = net["P"]; N = P.shape[0]
    K = np.asarray(net["K"], float).copy()
    directed = net.get("mode", "balanced") == "directed"
    node = node.copy(); phi = phi.copy(); fr = fr.copy()
    A = node.size
    R = K.copy()
    c_prev = np.zeros(A); first = True
    PT = np.ascontiguousarray(P.T)
    Pcum = np.cumsum(P, axis=1)
    if directed:
        dvec = np.asarray(net["d"], float); Mm = net["M"]
    a_node = np.ones(N) if kappa_node is None else np.asarray(kappa_node, float)
    pay_epoch = np.zeros(A)
    ceiling = K * r / (1 + r)
    Ctot = ceiling.sum()
    crisis_until = np.full(N, -1.0)
    restricted = np.zeros(N, bool)
    shock_nodes = np.array([], int) if shock is None else np.asarray(shock["nodes"], int)
    is_seed = np.zeros(N, bool); is_seed[shock_nodes] = True
    if shock is not None and shock["kind"] == "isolation":
        Piso = P.copy(); Piso[:, is_seed] = 0.0; Piso[is_seed, :] = 0.0
        PisoT = np.ascontiguousarray(Piso.T); keep_share = P[:, is_seed].sum(1)
    kap_sched = np.broadcast_to(np.asarray(kappa, float), (T,)) if np.ndim(kappa) == 0 else np.asarray(kappa, float)
    ts = {nm: np.zeros(T) for nm in ("dep", "dep03", "dep07", "depcap", "short", "pay", "waste", "wmean")}
    agent_rec = [] if record_agents else None
    xsum = np.zeros(N); nwin = 0
    agent_pay_total = np.zeros(A); agent_pay_late = np.zeros(A)
    evo = [] if record_evo else None
    R_ts = np.zeros((T, N), np.float32) if record else None
    nfr_ts = np.zeros((T, N), np.float32) if record else None
    n_ts = np.zeros((T, N), np.float32) if record else None
    x = R / K
    for t in range(T):
        kt = kap_sched[t]
        if shock is not None and shock["kind"] == "institutional" and t == shock["t0"]:
            dur = shock.get("dur")
            crisis_until[shock_nodes] = np.inf if dur is None else t + dur
        if crisis_p > 0:
            new = (rng.random(N) < crisis_p) & (crisis_until <= t)
            crisis_until[new] = t + crisis_dur
        in_crisis = crisis_until > t
        eff_fr = fr | in_crisis[node]
        cnt = np.bincount(node, minlength=N)
        n_j = cnt.astype(float)
        x = R / K
        w = weight(inst, x, k, rule=rule)
        nn = np.maximum(n_j[node], 1)
        if first:
            share = 1.0 / nn
        else:
            Cp = np.bincount(node, weights=c_prev, minlength=N)[node]
            prop = np.where(Cp > 0, c_prev / np.maximum(Cp, 1e-12), 1.0 / nn)
            share = w[node] / nn + (1 - w[node]) * prop
        o = R[node] * share
        if behaviour == "B2":
            eq = R[node] / nn
            rel = np.where(eq > 1e-9, o / np.maximum(eq, 1e-9) - 1.0, 0.0)
            ph = phi + b_recip * rel
        else:
            ph = phi
        if exec_sd > 0:
            ph = ph + rng.normal(0, exec_sd, A)
        ph = np.where(eff_fr, 0.0, np.clip(ph, 0, 1))
        c = o * ph
        pay = o - c
        ts["wmean"][t] = float(np.mean(w))
        if record_agents:
            agent_rec.append(np.stack([o, c, pay]).astype(np.float32))
        pay_epoch += pay; agent_pay_total += pay
        if t >= T - late:
            agent_pay_late += pay
        C = np.bincount(node, weights=c, minlength=N)
        eps = np.exp(rng.normal(0, prod_sd, N) - prod_sd ** 2 / 2) if prod_sd > 0 else np.ones(N)
        if harvest_p > 0:
            hit = rng.random(N) < harvest_p
            eps = np.where(hit, eps * (1 - harvest_loss), eps)
        if shock is not None and shock["kind"] == "production" and shock["t0"] <= t < shock["t0"] + shock.get("dur", 20):
            eps[shock_nodes] *= (1 - shock.get("loss", 1.0))
        G = (1 + r) * C * eps + regen * K
        trade_shock = shock is not None and shock["kind"] in ("export_halt", "isolation") and t >= shock["t0"] and \
            (shock.get("dur") is None or t < shock["t0"] + shock["dur"])
        if directed:
            dt = kt * dvec
            Gx = np.where(is_seed, 0.0, G) if trade_shock else G
            imp = K * (Mm @ (Gx / K))
            Rn = (1 - dt) * G + dt * imp
        else:
            if export_rule is not None:
                th = export_rule.get("theta", 0.8); tho = export_rule.get("theta_open")
                restricted = (x < th) if tho is None else np.where(restricted, x < tho, x < th)
                rho = np.where(restricted, export_rule.get("rho", 0.0), 1.0)
            else:
                rho = 1.0
            X = kt * a_node * rho * G
            if trade_shock and shock["kind"] == "export_halt":
                X = np.where(is_seed, 0.0, X)
                Rn = G - X + PT @ X
            elif trade_shock and shock["kind"] == "isolation":
                X = np.where(is_seed, 0.0, X)
                Rn = G - X + PisoT @ X + X * keep_share
            else:
                Rn = G - X + PT @ X
        ts["waste"][t] = np.maximum(Rn - K, 0).sum() / Ctot
        R = np.minimum(Rn, K)
        c_prev = c; first = False
        x = R / K
        dep = x < 0.5
        ts["dep"][t] = dep.mean(); ts["dep03"][t] = np.mean(x < 0.3); ts["dep07"][t] = np.mean(x < 0.7)
        ts["depcap"][t] = K[dep].sum() / K.sum(); ts["short"][t] = np.mean(1 - x)
        ts["pay"][t] = pay.sum() / Ctot
        if t >= T - window:
            xsum += x; nwin += 1
        if record:
            R_ts[t] = x
            nfr_ts[t] = np.bincount(node, weights=fr.astype(float), minlength=N); n_ts[t] = n_j
        if (t + 1) % epoch == 0:
            if behaviour == "B3":
                if record_evo:
                    nco_ = np.bincount(node, weights=(~fr).astype(float), minlength=N)
                    mphi = np.bincount(node, weights=np.where(fr, 0, phi), minlength=N) / np.maximum(nco_, 1)
                    xphi = np.zeros(N); np.maximum.at(xphi, node, phi)
                    evo.append((t + 1, float(np.mean(mphi)), float(np.mean(xphi)), float(np.mean(x < 0.5)), float(ts["pay"][t])))
                order = np.argsort(node, kind="stable")
                starts = np.searchsorted(node[order], np.arange(N))
                u = rng.random(A)
                model = order[np.minimum(starts[node] + np.floor(u * cnt[node]).astype(int), A - 1)]
                scale = imit_beta * epoch
                if imit_norm:
                    pc = K / np.maximum(n_j, 1); scale = scale * pc[node] / np.mean(pc)
                pr = 1.0 / (1.0 + np.exp(-(pay_epoch[model] - pay_epoch) / scale))
                adopt = (rng.random(A) < pr) & (~fr) & (~fr[model])
                phi = np.where(adopt, phi[model], phi)
                phi = np.where(fr, 0.0, np.clip(phi + rng.normal(0, mut_sd, A), 0, 1))
            if migration > 0:
                mover = rng.random(A) < migration
                if mig_who == "freeriders":
                    mover &= fr
                idx = np.where(mover)[0]
                if idx.size:
                    u = rng.random(idx.size)
                    dest = np.minimum((Pcum[node[idx]] < u[:, None]).sum(1), N - 1)
                    frf = fr.astype(float)
                    nfr = np.bincount(node, weights=frf, minlength=N); nco = cnt - nfr
                    fr_pay = np.bincount(node, weights=pay_epoch * frf, minlength=N) / np.maximum(nfr, 1e-9)
                    co_pay = np.bincount(node, weights=pay_epoch * (1 - frf), minlength=N) / np.maximum(nco, 1e-9)
                    npc = np.bincount(node, weights=pay_epoch, minlength=N) / np.maximum(cnt, 1)
                    fallback = np.where(cnt > 0, npc, R * epoch)
                    exp_fr = np.where(nfr > 0, fr_pay, fallback); exp_co = np.where(nco > 0, co_pay, fallback)
                    tgt = np.where(fr[idx], exp_fr[dest], exp_co[dest])
                    pm = 1.0 / (1.0 + np.exp(-(tgt - pay_epoch[idx]) / (mig_beta * epoch)))
                    go = (rng.random(idx.size) < pm) & (dest != node[idx])
                    node[idx[go]] = dest[go]; c_prev[idx[go]] = 0.0
            pay_epoch[:] = 0.0
    post = slice(burn, T)
    xbar = xsum / max(nwin, 1)
    out = dict(efficiency=float(ts["pay"][post].mean()), waste=float(ts["waste"][post].mean()),
               dep_final=float(np.mean(x < 0.5)), dep_window=float(np.mean(xbar < 0.5)),
               depcap_window=float(K[xbar < 0.5].sum() / K.sum()), short_window=float(np.mean(1 - xbar)),
               dep03_window=float(np.mean(xbar < 0.3)), dep07_window=float(np.mean(xbar < 0.7)),
               empty_nodes=int((np.bincount(node, minlength=N) == 0).sum()))
    occ = np.bincount(node, minlength=N) > 0
    out["dep_window_occupied"] = float(np.mean(xbar[occ] < 0.5)) if occ.any() else np.nan
    for tag, ap in (("cum", agent_pay_total), ("late", agent_pay_late)):
        gw, wts, exm = [], [], {f: [] for f in (0.01, 0.05, 0.1, 0.2)}
        for j in np.where(occ)[0]:
            pj = ap[node == j]
            if pj.size >= 2 and pj.sum() > 0:
                gw.append(gini(pj)); wts.append(K[j])
                for f in exm:
                    exm[f].append(float(np.mean(pj < f * pj.sum() / pj.size)))
        wts = np.asarray(wts)
        out[f"gini_{tag}"] = float(np.dot(gw, wts) / wts.sum()) if wts.size else np.nan
        for f in exm:
            out[f"excl{int(round(f * 100))}_{tag}"] = float(np.dot(exm[f], wts) / wts.sum()) if wts.size else np.nan
    out["x_window"] = xbar; out["x_final"] = x
    out["ts"] = ts
    if record:
        out.update(R_ts=R_ts, node_final=node, fr_final=fr, phi_final=phi, nfr_ts=nfr_ts, n_ts=n_ts)
    if record_evo:
        out["evo"] = np.array(evo); out["phi_final"] = phi; out["node_final"] = node; out["fr_final"] = fr
    if record_agents:
        out["agents"] = np.array(agent_rec)
    out["wmean"] = float(ts["wmean"][post].mean())
    return out
