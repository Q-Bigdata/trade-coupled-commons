"""Analytical quantities for the coupled-commons model (revision 2).

Stationary allocation.  For a community with fixed return fractions phi_i and a constant equal-sharing
weight 0 < w <= 1 the share map is s_i' = w/n + (1-w) s_i phi_i / Phi(s), Phi(s) = sum_i s_i phi_i.
Any fixed point has s_i >= w/n > 0 and satisfies s_i = w Phi / (n [Phi - (1-w) phi_i]); summing over i,
Phi solves g(Phi) = (w/n) sum_i Phi / (Phi - (1-w) phi_i) = 1 on Phi > (1-w) phi_max.  g is continuous and
strictly decreasing there (each term with phi_i > 0 decreases, the others are constant), g -> +inf as
Phi -> (1-w) phi_max from above (if phi_max > 0) and g -> w < 1 as Phi -> inf, so the fixed point exists
and is unique.  (If all phi_i = 0 the equal-share fallback applies and Phi = 0.)  We compute it by
bisection.  For w = 0 the map is s_i' proportional to s_i phi_i, so shares concentrate on the initially
represented members with the largest phi (all members are represented in round 1, and ties keep several
survivors); Phi -> phi_max.
"""
import numpy as np
from ccsim2 import weight, EQUAL, MIXED, PROP, DISTILLED, STEP, GENERIC


def stationary_phi(phi, w, tol=1e-13):
    """Exact stationary retention Phi(w) of one community (phi: array of return fractions)."""
    phi = np.asarray(phi, float); n = phi.size
    if n == 0 or phi.max() <= 0:
        return 0.0
    if w >= 1:
        return float(phi.mean())
    if w <= 0:
        return float(phi.max())
    lo = (1 - w) * phi.max() * (1 + 1e-15) + 1e-300; hi = max(phi.max(), 1e-12) * 2 + 1.0
    def g(P):
        return (w / n) * np.sum(P / (P - (1 - w) * phi)) - 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


WGRID = np.concatenate([[0.0], np.logspace(-4, 0, 60)])


def phi_curves(node, phi, N):
    """Phi_j(w) on WGRID for every community (array N x len(WGRID))."""
    out = np.zeros((N, WGRID.size))
    cnt = np.bincount(node, minlength=N).astype(float)
    pmax = np.zeros(N); np.maximum.at(pmax, node, phi)
    pmean = np.bincount(node, weights=phi, minlength=N) / np.maximum(cnt, 1)
    for iw, w in enumerate(WGRID):
        if w <= 0:
            out[:, iw] = pmax; continue
        if w >= 1:
            out[:, iw] = pmean; continue
        lo = (1 - w) * pmax + 1e-15; hi = pmax + 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = (w / np.maximum(cnt, 1)) * np.bincount(node, weights=mid[node] / (mid[node] - (1 - w) * phi), minlength=N) - 1.0
            lo = np.where(g > 0, mid, lo); hi = np.where(g > 0, hi, mid)
        out[:, iw] = np.where(pmax > 0, 0.5 * (lo + hi), 0.0)
    return out


def growth_factors(node, phi, inst, r=0.4, k=22.0, rule=None):
    """lambda_full (at the institution's weight with a full pool), lambda_max (upper envelope: weight at
    its lowest reachable value; for the distilled rule the weight tends to 0 but is never exactly 0),
    lambda_mean (mean-return growth) and the resilience reserve m = 1 - 1/lambda_max."""
    N = inst.size
    cnt = np.bincount(node, minlength=N).astype(float)
    curves = phi_curves(node, phi, N)
    w_full = weight(inst, np.ones(N), k, rule=rule)
    w_low = weight(inst, np.zeros(N) + 1e-9, k, rule=rule)
    lf = np.array([(1 + r) * np.interp(w_full[j], WGRID, curves[j]) for j in range(N)])
    lm = np.array([(1 + r) * np.interp(w_low[j], WGRID, curves[j]) for j in range(N)])
    m = np.where((lm > 0) & (cnt > 0), 1 - 1 / np.maximum(lm, 1e-12), -np.inf)
    return dict(lam_full=lf, lam_max=lm, margin=m, curves=curves, lam_mean=(1 + r) * curves[:, -1])


def lam_of_x(inst, x, gf, mode="exact", k=22.0, r=0.4, rule=None):
    """Effective growth factor at relative pool x.  'interp': w lam_full + (1-w) lam_max (the original
    two-factor closure, an upper bound on the stationary value for threshold rules);
    'exact': (1+r) Phi_j(W(x)) from the exact stationary curve."""
    w = weight(inst, x, k, rule=rule)
    if mode == "interp":
        lam = w * gf["lam_full"] + (1 - w) * gf["lam_max"]
        return np.where(np.isin(inst, [EQUAL, MIXED, PROP]), gf["lam_full"], lam)
    cur = gf["curves"]
    # vectorised interpolation of each row at its own w
    idx = np.clip(np.searchsorted(WGRID, w) - 1, 0, WGRID.size - 2)
    w0 = WGRID[idx]; w1 = WGRID[idx + 1]
    a = np.clip((w - w0) / np.maximum(w1 - w0, 1e-300), 0, 1)
    rows = np.arange(inst.size)
    return (1 + r) * ((1 - a) * cur[rows, idx] + a * cur[rows, idx + 1])


def reduced(net, kappa, inst, gf, seeds=(), iters=3000, mode="exact", horizon=None, k=22.0, rule=None, kappa_node=None):
    """Node-level reduction iterated from capacity.  Returns dict(x=final relative pools, x_win=mean over
    the last 50 iterations, converged, cycle (legacy flag: non-converged terminal difference > 1e-6), iterations).  horizon: if given, iterate
    exactly that many rounds (matched to the agent model's post-shock horizon) and report the window mean."""
    P = net["P"]; K = np.asarray(net["K"], float); N = P.shape[0]
    PT = np.ascontiguousarray(P.T)
    directed = net.get("mode", "balanced") == "directed"
    seeds = np.asarray(seeds, int)
    a = np.ones(N) if kappa_node is None else np.asarray(kappa_node, float)
    x = np.ones(N); hist = []
    n_it = horizon if horizon is not None else iters
    conv = False
    for it in range(n_it):
        lam = lam_of_x(inst, x, gf, mode=mode, k=k, rule=rule)
        lam[seeds] = 0.0
        G = lam * x * K
        if directed:
            dt = kappa * net["d"]
            xn = np.minimum(1.0, (1 - dt) * lam * x + dt * (net["M"] @ (lam * x)))
        else:
            X = kappa * a * G
            xn = np.minimum(1.0, (G - X + PT @ X) / K)
        if horizon is None and np.abs(xn - x).max() < 1e-10:
            x = xn; conv = True; break
        x = xn
        if it >= n_it - 50:
            hist.append(x.copy())
    xw = np.mean(hist, axis=0) if hist else x
    cyc = (not conv) and horizon is None and np.abs(hist[-1] - hist[-2]).max() > 1e-6 if len(hist) > 1 else False
    return dict(x=x, x_win=xw, converged=conv, cycle=bool(cyc), iterations=it + 1)


def one_d_orbit(lam_fun, kappa, inflow, x0=1.0, T=2000):
    """Orbit of the isolated-country map x' = min(1, (1-kappa) lam(x) x + kappa * inflow) from x0.
    inflow: P-weighted growth-times-pool of surviving partners (partners at capacity: sum_k P_jk lam_k)."""
    x = x0; xs_min = 1.0
    for _ in range(T):
        x = min(1.0, (1 - kappa) * lam_fun(x) * x + kappa * inflow)
        xs_min = min(xs_min, x)
    return x, xs_min


def criteria(net, kappa, inst, gf, seeds, k=22.0, r=0.4, rule=None, T=600):
    """First-order (no-cascade) predictions for every country when the seeds fail and every other partner
    stays at capacity with its full-pool growth factor.
      cap_keep : one-round capacity retention  (1-kappa) lam_full_j + kappa I_j >= 1
      reserve  : the original margin rule kappa q_j <= m_j
      deplete  : the 1-D orbit from capacity ends below 1/2 (long-run depletion; exact stationary closure)
    where I_j = sum_{k not seed} P_jk lam_full_k and q_j = sum_{k in seeds} P_jk."""
    P = net["P"]; N = P.shape[0]
    seeds = np.asarray(seeds, int); is_seed = np.zeros(N, bool); is_seed[seeds] = True
    lf = gf["lam_full"]
    q = P[:, is_seed].sum(1)
    I = P[:, ~is_seed] @ lf[~is_seed]
    cap_keep = (1 - kappa) * lf + kappa * I >= 1 - 1e-12
    reserve_fail = kappa * q > gf["margin"]
    x = np.ones(N); xs = []
    for it in range(T):
        lam = lam_of_x(inst, x, gf, mode="exact", k=k, r=r, rule=rule)
        x = np.minimum(1.0, (1 - kappa) * lam * x + kappa * I)
        if it >= T - 50:
            xs.append(x)
    xend = np.mean(xs, axis=0)
    dep = (xend < 0.5) | is_seed
    return dict(q=q, I=I, cap_keep=cap_keep, reserve_fail=reserve_fail, deplete=dep, x_end=xend)


def tolerance_constant(lam, lam_h, kappa):
    """Closed forms for a constant growth factor lam with surviving partners at capacity growing lam_h.
    capacity retention: q <= 1 - (1 - (1-kappa) lam) / (kappa lam_h)
    no long-run depletion: q <= 1 - (1 - (1-kappa) lam) / (2 kappa lam_h)   (valid when (1-kappa) lam < 1)"""
    a = 1 - (1 - kappa) * lam
    qcap = 1 - a / (kappa * lam_h)
    qdep = np.where((1 - kappa) * lam < 1, 1 - a / (2 * kappa * lam_h), 1.0)
    return qcap, qdep
