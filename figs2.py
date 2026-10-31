"""Main and Extended Data figures for the revised manuscript (runs on the compute host after report2.py)."""
import os, sys, json, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import apply_figure_style, panel_letter, plt, mpl
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from scipy import stats
import ccsim2 as cs, theory2 as th

apply_figure_style()
R = os.path.join(HERE, "results"); SD = os.path.join(HERE, "source_data"); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
SUF = os.environ.get("SUF", "")
MINCOM = 1 if SUF else 4
NUM = json.load(open(os.path.join(R, "numbers.json"))); KOBS = NUM["kobs"]
HUM = json.load(open(os.path.join(R, "human_summary.json"))); AUD = json.load(open(os.path.join(R, "net_audit.json")))
T1 = json.load(open(os.path.join(R, "table1.json")))
EX = np.load(os.path.join(R, "examples2.npz"))
COMS = ["wheat", "maize", "rice", "soybean"]
COL = {"Equal": "#E69F00", "Mixed": "#56B4E9", "Proportional": "#7F7F7F", "Distilled": "#0072B2", "Step": "#009E73"}
SHORT = {"Equal": "Equal sharing", "Mixed": "Mixed (w = 0.5)", "Proportional": "Proportional", "Distilled": "Resource-contingent", "Step": "Collective step"}
MAIN3 = ["Equal", "Mixed", "Distilled"]; ALL5 = ["Equal", "Mixed", "Proportional", "Distilled", "Step"]
CODE = {"Equal": 0, "Mixed": 1, "Distilled": 3}
W2 = 7.09


def rd(name):
    return pd.read_csv(os.path.join(R, f"{name}{SUF}.csv.gz"))


def tci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def strat(df, col, by):
    out = []
    for key, d in df.groupby(by):
        per = d.groupby("commodity")[col].agg(["mean", "var", "count"])
        m = per["mean"].mean(); se = np.sqrt((per["var"].fillna(0) / per["count"]).sum()) / len(per)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,))); rec.update(mean=m, ci=1.96 * se, ncom=len(per)); out.append(rec)
    o = pd.DataFrame(out)
    return o[o.ncom >= MINCOM] if len(o) else o


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def band(ax, x, m, lo, hi, color, label=None, ls="-", ms=2.4, lw=1.3, alpha=0.16, marker="o"):
    ax.plot(x, m, ls, color=color, lw=lw, marker=marker, ms=ms, label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=2)


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf")); fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300); plt.close(fig)


nets = {c: np.load(os.path.join(HERE, "nets", f"{c}_2021_2023.npz")) for c in COMS}
E1 = rd("E1"); E2 = rd("E2"); E2 = E2[E2.tag == "2021_2023"]; E3 = rd("E3"); E4b = rd("E4b"); E5 = rd("E5")

# ============================================================ Figure 1
fig = plt.figure(figsize=(W2, 5.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], wspace=0.5, hspace=0.62, left=0.05, right=0.985, top=0.93, bottom=0.1)
ax = fig.add_subplot(gs[:, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 21); ax.set_aspect("equal", adjustable="box", anchor="N"); ax.axis("off")
pos = {"A": (2.1, 16.6), "B": (7.9, 16.6), "C": (5.0, 10.4)}; fill = {"A": 0.95, "B": 0.55, "C": 0.2}
for k_, (x0, y0) in pos.items():
    ax.add_patch(Circle((x0, y0), 1.75, fc="white", ec="0.3", lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x0 - 0.65, y0 - 0.45), 1.3, 1.6 * fill[k_], fc="#9ecae1", ec="none", zorder=3))
    ax.add_patch(Rectangle((x0 - 0.65, y0 - 0.45), 1.3, 1.6, fc="none", ec="0.2", lw=0.8, zorder=4))
    for i, dx in enumerate([-0.9, -0.3, 0.3, 0.9]):
        ax.add_patch(Circle((x0 + dx, y0 - 0.95), 0.22, fc="0.35" if not (k_ == "C" and i == 0) else "#D55E00", ec="none", zorder=5))
    ax.text(x0 + (2.3 if k_ == "C" else 0), y0 + (0 if k_ == "C" else 1.95), {"A": "country j", "B": "country k", "C": "country l"}[k_], ha="left" if k_ == "C" else "center", va="center" if k_ == "C" else "bottom", fontsize=7)
for a_, b_ in [("A", "B"), ("B", "C"), ("A", "C")]:
    ax.add_patch(FancyArrowPatch(pos[a_], pos[b_], arrowstyle="<|-|>", mutation_scale=7, lw=0.9, color="0.45", shrinkA=16, shrinkB=16, zorder=1))
ax.text(5.0, 19.9, "balanced trade: a share $\\kappa$ of output,\nsplit by gross-trade shares $P_{jk}$", ha="center", fontsize=6, color="0.3")
ax.annotate("free rider", xy=(4.1, 9.45), xytext=(1.2, 8.2), fontsize=6, color="#D55E00", arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.6))
ax.text(0.0, 6.4, "Within each country, every round:", fontsize=7, fontweight="bold")
ax.text(0.0, 5.9, "1. the pool $R_j$ is allocated with weight\n    $w(R_j/K_j)$ on equal shares and $1-w$\n    on shares proportional to last returns\n2. members return fractions $\\phi_i$, which\n    grow by the factor $1+r$\n3. a share $\\kappa$ is exported; the pool is\n    capped at capacity $K_j$", fontsize=6, va="top", linespacing=1.35)
ax.set_title("Coupled commons", fontsize=8)
panel_letter(ax, "a", dx=-0.06, dy=1.0)
# b: exact stationary growth versus the interpolation
ax = fig.add_subplot(gs[0, 1])
wg = EX["wgrid"]; nofr = EX["fr_node"] == 0
lm = EX[f"lammax_3"]; j = np.where(nofr)[0][np.argmin(np.abs(lm[nofr] - np.median(lm[nofr])))]
x = np.linspace(0.3, 1.0, 400)
cur = EX["curves_3"][j]
lam_ex = 1.4 * np.interp(x ** 22, wg, cur); lf, lmx = 1.4 * cur[-1], 1.4 * cur[0]
ax.plot(x, np.full_like(x, 1.4 * EX["curves_0"][j][-1]), color=COL["Equal"], label=SHORT["Equal"])
ax.plot(x, np.full_like(x, 1.4 * np.interp(0.5, wg, EX["curves_1"][j])), color=COL["Mixed"], label=SHORT["Mixed"])
ax.plot(x, lam_ex, color=COL["Distilled"], label=SHORT["Distilled"] + " (exact)")
ax.plot(x, x ** 22 * lf + (1 - x ** 22) * lmx, color=COL["Distilled"], ls="--", lw=0.9, label="two-factor interpolation")
ax.axhline(1, color="0.5", lw=0.7, ls=":")
ax.annotate("", xy=(0.36, 1.0), xytext=(0.36, lmx), arrowprops=dict(arrowstyle="<->", color=COL["Distilled"], lw=0.8))
ax.text(0.375, 1.02, "growth reserve $\\lambda^{\\max}-1$", color=COL["Distilled"], fontsize=6, va="bottom")
ax.set_xlabel("Pool relative to capacity, $R_j/K_j$"); ax.set_ylabel(r"Stationary growth factor $\lambda_j$")
ax.set_ylim(0.95, max(1.42, lmx + 0.04)); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.66), fontsize=5.6, handlelength=1.6)
ax.set_title("Growth under scarcity", fontsize=8); panel_letter(ax, "b")
# c: margins
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-0.4, 0.32, 37)
for nm in MAIN3:
    m = EX[f"margin_{CODE[nm]}"]; m = np.clip(m[np.isfinite(m)], -0.4, 0.32)
    ax.hist(m, bins=bins, histtype="step", color=COL[nm], lw=1.3, label=SHORT[nm])
ax.axvline(0, color="0.5", lw=0.7, ls=":")
ax.text(-0.39, ax.get_ylim()[1] * 0.95, "not viable\nwithout imports", fontsize=6, va="top")
ax.set_xlabel(r"Reserve $m_j = 1-1/\lambda_j^{\max}$"); ax.set_ylabel("Countries (wheat)")
ax.set_title("Reserves (wheat)", fontsize=8); panel_letter(ax, "c")
# d: concentration
ax = fig.add_subplot(gs[1, 1])
shade = {"wheat": "0.1", "maize": "0.35", "rice": "0.55", "soybean": "0.75"}; mk = {"wheat": "o", "maize": "s", "rice": "^", "soybean": "D"}
for c in COMS:
    W = nets[c]["W"]; P = W / W.sum(1, keepdims=True); q = np.sort(P.max(1))
    ax.plot(q, 1 - np.arange(q.size) / q.size, color=shade[c], lw=1.3, label=c)
ax.set_xlabel("Largest single-partner share of gross trade, $q_j$"); ax.set_ylabel(r"Fraction of countries $\geq q_j$")
ax.legend(fontsize=6, loc="lower left"); ax.set_title("Exposure is concentrated", fontsize=8); panel_letter(ax, "d")
# e: capacity-retention versus depletion boundaries in the kappa-q plane (median community, healthy partners)
ax = fig.add_subplot(gs[1, 2])
kk = np.linspace(0.02, 0.8, 80)
def boundary(inst, crit):
    code = CODE[inst]; jj = j
    curj = EX[f"curves_{code}"][jj]; lf_ = 1.4 * curj[-1] if inst != "Mixed" else 1.4 * np.interp(0.5, wg, curj)
    lam_h = lf_
    out = []
    for kap in kk:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            q = 0.5 * (lo + hi)
            xx = 1.0; xmin = 1.0
            for _t in range(800):
                w = {"Equal": 1.0, "Mixed": 0.5, "Distilled": min(1, xx) ** 22}[inst]
                lam = 1.4 * np.interp(w, wg, curj)
                xx = min(1.0, (1 - kap) * lam * xx + kap * lam_h * (1 - q)); xmin = min(xmin, xx)
                if crit == "cap" and _t == 0:
                    break
            ok = (xx >= 1 - 1e-9) if crit == "cap" else (xx >= 0.5)
            lo, hi = (q, hi) if ok else (lo, q)
        out.append(lo)
    return np.array(out)
for nm in MAIN3:
    ax.plot(kk, boundary(nm, "dep"), color=COL[nm], lw=1.4)
    ax.plot(kk, boundary(nm, "cap"), color=COL[nm], lw=0.9, ls="--")
for c in COMS:
    W = nets[c]["W"]; P = W / W.sum(1, keepdims=True)
    ax.plot([KOBS[c]], [np.median(P.max(1))], marker=mk[c], color=shade[c], ms=4.5, ls="none", zorder=5, label=c)
ax.set_xlim(0, 0.8); ax.set_ylim(0, 1.02)
ax.set_xlabel(r"Traded share of production, $\kappa$"); ax.set_ylabel("Exposure to failed partner(s), $q$")
h1 = [Line2D([], [], color="0.3", lw=1.4, label="long-run depletion"), Line2D([], [], color="0.3", lw=0.9, ls="--", label="capacity loss\n(equal = res.-cont.)")]
leg = ax.legend(handles=h1, fontsize=5.4, loc="upper right", bbox_to_anchor=(1.03, 1.03)); ax.add_artist(leg)
ax.legend(fontsize=5.4, loc="lower left", handlelength=0.8, title="median country", title_fontsize=5.4, ncol=2)
ax.set_title("Capacity vs depletion", fontsize=8); panel_letter(ax, "e")
save(fig, "Fig1")

# ============================================================ Figure 2
fig = plt.figure(figsize=(W2, 4.7))
gs = fig.add_gridspec(2, 4, hspace=0.65, wspace=0.12, left=0.085, right=0.95, top=0.93, bottom=0.1)
gtop = gs[0, :].subgridspec(1, 2, wspace=0.33)
ax = fig.add_subplot(gtop[0, 0])
for nm in MAIN3:
    ax.plot(np.arange(400), EX[f"casc_{CODE[nm]}"], color=COL[nm], lw=1.4, label=SHORT[nm])
ax.axvline(100, color="#D55E00", lw=0.8, ls="--"); ax.text(96, 0.5, "five largest wheat\ntraders fail", color="#D55E00", fontsize=6, ha="right")
ax.set_xlim(0, 400); ax.set_ylim(-0.02, 0.62); ax.set_xlabel("Round"); ax.set_ylabel("Fraction of countries depleted")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.25), fontsize=6)
ax.set_title(f"Wheat, observed $\\kappa$ = {KOBS['wheat']:.2f}", fontsize=8); panel_letter(ax, "a", dx=-0.11)
ax = fig.add_subplot(gtop[0, 1])
hb = ax.hexbin(E1.final_dep, E1.red_exact_h300_dep, gridsize=35, bins="log", cmap="Greys", mincnt=1, linewidths=0)
ax.plot([0, 1], [0, 1], color=COL["Distilled"], lw=0.8, ls="--")
rr = NUM["reduction"]["exact_h300"]
ax.text(0.03, 0.97, f"country-level agreement {100 * rr['acc']:.1f}%\nfalse positives {100 * rr['fp_rate']:.1f}%, false negatives {100 * rr['fn_rate']:.1f}%\n$R^2$ = {rr['r2']:.3f}, n = {len(E1):,} runs", fontsize=6, transform=ax.transAxes, va="top")
cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Runs", fontsize=6); cb.ax.tick_params(labelsize=6)
ax.set_xlabel("Depleted fraction, agent-based model"); ax.set_ylabel("Node-level reduction")
ax.set_title("Node-level reduction reproduces cascades", fontsize=8); panel_letter(ax, "b", dx=-0.14)
d5 = E1[(E1.seedset == "top5") & E1.inst.isin(MAIN3)]
for i, c in enumerate(COMS):
    ax = fig.add_subplot(gs[1, i])
    for nm in MAIN3:
        g = d5[(d5.commodity == c) & (d5.inst == nm)].groupby("kappa")
        m = g.cascade.mean(); e = g.cascade.agg(tci)
        band(ax, m.index, m, m - e, m + e, COL[nm], label=SHORT[nm] if i == 0 else None)
        ax.plot(m.index, g.final_dep.mean(), color=COL[nm], lw=0.7, ls=":")
    ax.plot([KOBS[c]], [-0.03], marker="^", color="0.25", ms=4.5, clip_on=False, zorder=6)
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.02, 0.72); ax.set_title(c.capitalize(), fontsize=7); ax.set_xlabel(r"$\kappa$")
    if i == 0:
        ax.set_ylabel("Newly depleted non-seed countries"); panel_letter(ax, "c", dx=-0.32)
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    else:
        ax.tick_params(labelleft=False)
fig.axes[-1].plot([], [], color="0.3", lw=0.7, ls=":", label="all depleted"); fig.axes[-1].plot([], [], marker="^", color="0.25", ls="none", ms=4, label="observed $\\kappa$"); fig.axes[-1].legend(fontsize=5.3, loc="lower right")
save(fig, "Fig2")

# ============================================================ Figure 3
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.8)); fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1, wspace=0.3, hspace=0.55)
ax = axs[0, 0]
for nm in MAIN3:
    for col, ls, lab in [("visible_dep", "-", True), ("latent_dep_600", "--", False)]:
        s = strat(E2[E2.inst == nm], col, ["kappa"]).sort_values("kappa")
        band(ax, s.kappa, s["mean"], s["mean"] - s.ci, s["mean"] + s.ci, COL[nm], label=SHORT[nm] if lab else None, ls=ls, ms=2 if lab else 0)
    ax.plot([0.72], [NUM["E2_pred_unviable"][nm]], marker="<", color=COL[nm], ms=4, clip_on=False)
ax.text(0.32, 0.215, "after trade halts (dashed)", fontsize=6); ax.text(0.33, 0.014, "while trading (solid)", fontsize=6)
ax.set_xlabel(r"Traded share, $\kappa$"); ax.set_ylabel("Fraction of countries depleted"); ax.set_ylim(-0.01, 0.27); ax.set_xlim(-0.02, 0.74)
ax.legend(fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.55)); ax.set_title("Trade hides locally unviable institutions", fontsize=8); panel_letter(ax, "a", dx=-0.2)
ax = axs[0, 1]
for nm in MAIN3:
    s = strat(E2[E2.inst == nm], "efficiency", ["kappa"]).sort_values("kappa")
    band(ax, s.kappa, s["mean"], s["mean"] - s.ci, s["mean"] + s.ci, COL[nm])
ax.set_xlabel(r"Traded share, $\kappa$"); ax.set_ylabel("Surplus (fraction of ceiling)")
ax.set_title("Coupling raises the surplus of weak institutions", fontsize=8); panel_letter(ax, "b", dx=-0.2)
main = E3[(E3.hazard == "both") & (E3.rule == "none")]
ax = axs[1, 0]
for nm in MAIN3:
    rows = []
    for kap, d in main[main.inst == nm].groupby("kappa"):
        if d.commodity.nunique() < MINCOM:
            continue
        k_ = d.sys50_any.sum(); n_ = len(d); lo, hi = wilson(k_, n_); rows.append((kap, k_ / n_, lo, hi))
    a = np.array(rows); band(ax, a[:, 0], a[:, 1], a[:, 2], a[:, 3], COL[nm], label=SHORT[nm])
ax.set_xlabel(r"Traded share, $\kappa$"); ax.set_ylabel("Runs with a systemic event")
ax.legend(fontsize=6, loc="upper left"); ax.set_title("Coupling adds tail risk under equal sharing", fontsize=8); panel_letter(ax, "c", dx=-0.2)
ax = axs[1, 1]
for nm in MAIN3:
    rows = []
    for kap, d in main[main.inst == nm].groupby("kappa"):
        if d.commodity.nunique() < MINCOM:
            continue
        rows.append((kap, d.sys50_any.mean(), d.efficiency.mean()))
    a = np.array(rows); ax.plot(a[:, 1], a[:, 2], "-o", color=COL[nm], ms=2.4, lw=1.2)
    for kv in (0.7,):
        r_ = a[np.isclose(a[:, 0], kv)]
        if len(r_):
            ax.annotate(f"{kv:g}", (r_[0, 1], r_[0, 2]), fontsize=5.5, color=COL[nm], xytext=(3, -7), textcoords="offset points")
ax.set_xlabel("Runs with a systemic event"); ax.set_ylabel("Modelled surplus"); ax.set_title(r"Surplus versus tail risk ($\kappa$ = 0 to 0.7)", fontsize=8); panel_letter(ax, "d", dx=-0.2)
save(fig, "Fig3")

# ============================================================ Figure 4
ARMS = [("Equal", "none"), ("Equal", "halve"), ("Equal", "ban"), ("Equal", "ban_hyst"), ("Distilled", "none"), ("Distilled", "ban")]
ACOL = {("Equal", "none"): COL["Equal"], ("Equal", "halve"): "#C77C00", ("Equal", "ban"): "#7A4B00", ("Equal", "ban_hyst"): "#B8860B",
        ("Distilled", "none"): COL["Distilled"], ("Distilled", "ban"): "#00365A"}
ALS = {"none": "-", "halve": "--", "ban": ":", "ban_hyst": "-."}
ALAB = {"none": "open trade", "halve": "exports halved below 80%", "ban": "export ban below 80%", "ban_hyst": "ban, reopen at 95%"}
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9)); fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1, wspace=0.3, hspace=0.55)
hz = E3[E3.hazard == "both"]
for ax, ycol, ylab, ttl, let in [(axs[0, 0], "sys50_any", "Runs with a systemic event", "Bans help at moderate coupling, destabilise at high", "a"),
                                  (axs[1, 0], "efficiency", "Modelled surplus", "Open trade with an internal rule keeps surplus", "c")]:
    for arm in ARMS:
        d = hz[(hz.inst == arm[0]) & (hz.rule == arm[1])]
        rows = []
        for kap, dd in d.groupby("kappa"):
            if dd.commodity.nunique() < MINCOM:
                continue
            if ycol == "sys50_any":
                k_ = dd[ycol].sum(); n_ = len(dd); lo, hi = wilson(k_, n_); rows.append((kap, k_ / n_, lo, hi))
            else:
                m = dd[ycol].mean(); e = tci(dd[ycol]); rows.append((kap, m, m - e, m + e))
        a = np.array(rows)
        band(ax, a[:, 0], a[:, 1], a[:, 2], a[:, 3], ACOL[arm], label=f"{SHORT[arm[0]]}, {ALAB[arm[1]]}", ls=ALS[arm[1]], ms=1.8, alpha=0.1)
    ax.set_xlabel(r"Traded share, $\kappa$"); ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=8); panel_letter(ax, let, dx=-0.2)
axs[0, 0].legend(fontsize=5.2, loc="upper left", handlelength=2.4)
ax = axs[0, 1]
for arm in [("Equal", "none"), ("Equal", "halve"), ("Equal", "ban"), ("Distilled", "none"), ("Distilled", "ban")]:
    d = E4b[(E4b.inst == arm[0]) & (E4b.rule == arm[1])]
    g = d.groupby("kappa"); m = g.peak_dep.mean(); e = g.peak_dep.agg(tci); keep = g.commodity.nunique() >= MINCOM
    band(ax, m.index[keep], m[keep], (m - e)[keep], (m + e)[keep], ACOL[arm], ls=ALS[arm[1]], ms=1.8, alpha=0.1)
ax.set_xlabel(r"Traded share, $\kappa$"); ax.set_ylabel("Peak depleted fraction")
ax.set_title("Two largest traders lose half their output", fontsize=8); panel_letter(ax, "b", dx=-0.2)
ax = axs[1, 1]
t = np.arange(EX["ban_dep"].size)
sm = lambda v: pd.Series(v).rolling(10, min_periods=1, center=True).mean().values
ax.plot(t, EX["ban_dep"], color="#7A4B00", lw=0.3, alpha=0.35)
ax.plot(t, sm(EX["open_dep"]), color=COL["Equal"], lw=1.1, label="open trade: depleted")
ax.plot(t, sm(EX["ban_dep"]), color="#7A4B00", lw=1.1, label="export ban: depleted")
ax.plot(t, sm((EX["ban_R"] < 0.8).mean(1)), color="#7A4B00", lw=0.9, ls="--", label="export ban: share restricting")
ax.set_xlabel("Round"); ax.set_ylabel("Fraction of countries"); ax.set_xlim(100, t[-1])
ax.legend(fontsize=5.6, loc="upper left", ncol=1, framealpha=0.9); ax.set_ylim(0, 1.0); ax.set_title(r"Ban cycles (wheat, equal sharing, $\kappa$ = 0.6)", fontsize=8); panel_letter(ax, "d", dx=-0.2)
save(fig, "Fig4")

# ============================================================ Figure 5
PCOL = {"strength": "#0072B2", "strength_noseed": "#56B4E9", "fragility": "#CC79A7", "concentration": "#999999", "random": "0.15"}
PLAB = {"strength": "largest traders first", "strength_noseed": "largest non-seed traders first", "fragility": "lowest reserve first",
        "concentration": "most concentrated exposure first", "random": "random"}
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.8)); fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1, wspace=0.3, hspace=0.55)
d = E5[(E5.shockset == "top5") & np.isclose(E5.kappa, 0.4)]
for ax, ycol, ylab, ttl, let in [(axs[0, 0], "cascade", "Newly depleted non-seed countries", "Reform large traders to stop cascades", "a"),
                                  (axs[0, 1], "latent_dep", "Depleted after trade halts", "Reform fragile countries to remove hidden risk", "b")]:
    for pl in ["strength", "strength_noseed", "fragility", "concentration", "random"]:
        s = strat(d[d.placement == pl], ycol, ["h"]).sort_values("h")
        band(ax, s.h, s["mean"], s["mean"] - s.ci, s["mean"] + s.ci, PCOL[pl], label=PLAB[pl], ms=2.2, alpha=0.1)
    ax.set_xlabel("Share of countries adopting the\nresource-contingent rule"); ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=8); panel_letter(ax, let, dx=-0.2)
axs[0, 0].legend(fontsize=5.4, loc="upper right")
ax = axs[1, 0]
for hh, ls in [(0.2, "-"), (0.4, "--")]:
    rows = []
    for sig in [0.0, 0.02, 0.05, 0.1]:
        pl = "fragility" if sig == 0 else f"fragility:{sig}"
        s = strat(d[(d.placement == pl) & np.isclose(d.h, hh)], "latent_dep", ["h"]); rows.append((sig, s["mean"].iloc[0], s.ci.iloc[0]))
    a = np.array(rows); band(ax, a[:, 0], a[:, 1], a[:, 1] - a[:, 2], a[:, 1] + a[:, 2], PCOL["fragility"], label=f"{int(hh * 100)}% adopt, targeted", ls=ls)
    s = strat(d[(d.placement == "random") & np.isclose(d.h, hh)], "latent_dep", ["h"])
    ax.axhline(s["mean"].iloc[0], color="0.15", lw=0.8, ls=ls, label=f"{int(hh * 100)}% adopt, random")
ax.set_xlabel("Error (s.d.) in estimated reserves"); ax.set_ylabel("Depleted after trade halts")
ax.legend(fontsize=5.6, loc="center right"); ax.set_title("Targeting tolerates moderate estimation error", fontsize=8); panel_letter(ax, "c", dx=-0.2)
ax = axs[1, 1]
dr = E5[(E5.shockset == "rand5") & np.isclose(E5.kappa, 0.4)]
for pl in ["strength", "fragility", "random"]:
    s = strat(dr[dr.placement == pl], "final_dep", ["h"]).sort_values("h")
    band(ax, s.h, s["mean"], s["mean"] - s.ci, s["mean"] + s.ci, PCOL[pl], label=PLAB[pl], ms=2.2, alpha=0.1)
ax.set_xlabel("Share of countries adopting"); ax.set_ylabel("All depleted countries")
ax.legend(fontsize=5.6, loc="upper right"); ax.set_title("Five random countries fail instead", fontsize=8); panel_letter(ax, "d", dx=-0.2)
save(fig, "Fig5")
print("main figures done", flush=True)

# ============================================================ Extended Data Figure 1: all institutions, all seed sets
fig, axs = plt.subplots(4, 4, figsize=(W2, 7.2), sharex=True, sharey=True); fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.07, wspace=0.08, hspace=0.32)
for r_, (ss, lab) in enumerate([("top1", "largest trader"), ("top3", "three largest"), ("top5", "five largest"), ("rand5", "five random")]):
    for c_, c in enumerate(COMS):
        ax = axs[r_, c_]
        for nm in ["Equal", "Mixed", "Proportional", "Step", "Distilled"]:
            g = E1[(E1.seedset == ss) & (E1.commodity == c) & (E1.inst == nm)].groupby("kappa")
            m = g.cascade.mean(); e = g.cascade.agg(tci)
            band(ax, m.index, m, m - e, m + e, COL[nm], label=SHORT[nm] if (r_ == 0 and c_ == 0) else None, ms=1.5 if nm != "Step" else 0, lw=1.0 if nm != "Distilled" else 1.3, alpha=0.12, ls="--" if nm == "Step" else "-")
            ax.plot(m.index, g.final_dep.mean(), color=COL[nm], lw=0.6, ls=":")
        ax.axvline(KOBS[c], color="0.4", lw=0.6, ls=":")
        if r_ == 0: ax.set_title(c.capitalize(), fontsize=7)
        if c_ == 0: ax.set_ylabel(f"{lab} fail\nnewly depleted", fontsize=7)
        if r_ == 3: ax.set_xlabel(r"$\kappa$")
axs[0, 0].legend(fontsize=5.2, loc="upper left")
for r_, l in enumerate("abcd"): panel_letter(axs[r_, 0], l, dx=-0.42)
save(fig, "ED_Fig1")

# ============================================================ ED Fig 2: seed semantics
A = rd("E1alt"); A = A[np.isclose(A.kappa, A.commodity.map(KOBS))]
KINDS = [("inst_perm", "members free ride\n(permanent)"), ("inst_temp20", "members free ride\n(20 rounds)"), ("prod_temp20", "output lost\n(20 rounds)"),
         ("prod_perm", "output lost\n(permanent)"), ("export_halt", "exports halted\n(permanent)"), ("isolation", "leave the network\n(permanent)")]
fig, axs = plt.subplots(2, 4, figsize=(W2, 4.9), sharey="row"); fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.2, wspace=0.08, hspace=0.25)
for row, (col, ylab) in enumerate([("cascade", "Newly depleted non-seed\n(end of run)"), ("peak_dep", "Peak depleted fraction")]):
    for ax, c in zip(axs[row], COMS):
        for ii, nm in enumerate(ALL5):
            sub = [A[(A.commodity == c) & (A.kind == k_) & (A.inst == nm)][col] for k_, _ in KINDS]
            ax.bar(np.arange(len(KINDS)) + (ii - 2) * 0.16, [s.mean() for s in sub], 0.16, yerr=[tci(s) for s in sub], color=COL[nm], label=SHORT[nm], error_kw=dict(lw=0.5, capsize=0))
        ax.set_xticks(range(len(KINDS)))
        ax.set_xticklabels([l for _, l in KINDS] if row == 1 else [], rotation=90, fontsize=5.5)
        if row == 0: ax.set_title(f"{c.capitalize()}, $\\kappa$ = {KOBS[c]:.2f}", fontsize=7)
    axs[row, 0].set_ylabel(ylab)
axs[0, 0].legend(fontsize=5.2, loc="upper right")
for r_ in range(2):
    for c_, l in enumerate("abcd" if r_ == 0 else "efgh"): panel_letter(axs[r_, c_], l, dx=-0.3 if c_ == 0 else -0.08)
save(fig, "ED_Fig2")

# ============================================================ ED Fig 3: null models and coupling variants
Nl = rd("E1null"); Nn = rd("E1net")
fig = plt.figure(figsize=(W2, 4.9)); gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], left=0.22, right=0.98, top=0.94, bottom=0.1, wspace=0.12, hspace=0.45)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
NL = [("observed", "rerank", "observed network"), ("weights_shuffled", "rerank", "weights shuffled"), ("strength_preserving", "rerank", "strength-preserving swaps"),
      ("degree_preserving", "rerank", "degree-preserving"), ("erdos_renyi", "rerank", "random graph"), ("degree_preserving", "fixed", "degree-pres., same seeds"),
      ("erdos_renyi", "fixed", "random graph, same seeds")]
ax = axs[0]; dd = Nl[np.isclose(Nl.kappa, Nl.commodity.map(KOBS))]
for yi, (nl, sm, lab) in enumerate(NL):
    for jj, nm in enumerate(MAIN3):
        s = strat(dd[(dd.null == nl) & (dd.seedmode == sm) & (dd.inst == nm)], "cascade", ["inst"])
        if len(s): ax.errorbar(s["mean"].iloc[0], yi + (jj - 1) * 0.22, xerr=s.ci.iloc[0], color=COL[nm], marker="o", ms=3, lw=0.8, label=SHORT[nm] if yi == 0 else None)
ax.set_yticks(range(len(NL))); ax.set_yticklabels([l for *_, l in NL], fontsize=6); ax.invert_yaxis()
ax.set_xlabel("Newly depleted non-seed countries"); ax.legend(fontsize=5.4, loc="lower right"); ax.set_title("Null networks, observed $\\kappa$", fontsize=8); panel_letter(ax, "a", dx=-0.45)
ax = axs[1]
for yi, (nl, sm, lab) in enumerate(NL):
    v = dd[(dd.null == nl) & (dd.seedmode == sm)].seed_cap_share.mean(); ax.barh(yi, v, color="0.6")
ax.set_yticks(range(len(NL))); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel("Capacity share of the five seeds"); ax.set_title("Seed weight", fontsize=8); panel_letter(ax, "b", dx=-0.06)
ax = axs[2]
CV = [("supply", "top5", "supply-based $K_j$, $\\kappa_j$"), ("directed", "top5", "directed, largest traders"), ("directed", "exp5", "directed, largest exporters")]
for yi, (cpl, s_, lab) in enumerate(CV):
    for jj, nm in enumerate(MAIN3):
        s = strat(Nn[(Nn.coupling == cpl) & (Nn.seedset == s_) & (Nn.mult == 1.0) & (Nn.inst == nm)], "cascade", ["inst"])
        s2 = strat(Nn[(Nn.coupling == cpl) & (Nn.seedset == s_) & (Nn.mult == 1.0) & (Nn.inst == nm)], "latent_dep", ["inst"])
        ax.errorbar(s["mean"].iloc[0], yi + (jj - 1) * 0.22, xerr=s.ci.iloc[0], color=COL[nm], marker="o", ms=3, lw=0.8)
        ax.plot(s2["mean"].iloc[0], yi + (jj - 1) * 0.22, marker="x", color=COL[nm], ms=3.5, ls="none")
ax.set_yticks(range(len(CV))); ax.set_yticklabels([l for *_, l in CV], fontsize=6); ax.invert_yaxis()
ax.set_xlabel("Newly depleted non-seed (circles); depleted after trade halts (crosses)"); ax.set_title("Alternative couplings, observed $\\kappa$", fontsize=8); panel_letter(ax, "c", dx=-0.45)
save(fig, "ED_Fig3")

# ============================================================ ED Fig 4: reduction and criteria diagnostics
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.12, wspace=0.3, hspace=0.6); axs = axs.ravel()
ax = axs[0]
phi_ex = np.array([1.0] + [0.6] * 7); ws = np.linspace(0, 1, 101)
ex_curve = [1.4 * th.stationary_phi(phi_ex, w) for w in ws]
ax.plot(ws, ex_curve, color=COL["Distilled"], label="exact stationary")
ax.plot(ws, 1.4 * (ws * phi_ex.mean() + (1 - ws) * phi_ex.max()), color=COL["Distilled"], ls="--", lw=0.9, label="linear interpolation")
ax.axhline(1, color="0.5", lw=0.6, ls=":"); ax.set_xlabel("Equal-sharing weight $w$"); ax.set_ylabel(r"Growth factor $(1+r)\Phi(w)$")
ax.legend(fontsize=5.6); ax.set_title("Eight members: one at 1, seven at 0.6", fontsize=7); panel_letter(ax, "a", dx=-0.2)
ax = axs[1]
tags = [("interp_conv", "interp., 3,000 max."), ("exact_conv", "exact, 3,000 max."), ("interp_h300", "interp., 300 rounds"), ("exact_h300", "exact, 300 rounds")]
for ii, nm in enumerate(ALL5):
    ax.bar(np.arange(4) + (ii - 2) * 0.16, [100 * (1 - NUM["reduction"][t_]["by_inst"][nm]) for t_, _ in tags], 0.16, color=COL[nm], label=SHORT[nm])
ax.set_xticks(range(4)); ax.set_xticklabels([l.replace(", ", "\n") for _, l in tags], fontsize=5.8); ax.set_ylabel("Misclassified countries (%)")
ax.legend(fontsize=5.2, ncol=2, loc="upper left"); ax.set_ylim(0, 1.1); ax.set_title("Reduction error by institution", fontsize=7); panel_letter(ax, "b", dx=-0.15)
ax = axs[2]
for t_, ls in [("interp_h300", "--"), ("exact_h300", "-")]:
    g = E1.groupby("kappa")[f"red_{t_}_acc"].mean(); g = g[E1.groupby("kappa").commodity.nunique() >= MINCOM]; ax.plot(g.index, 100 * (1 - g), ls, color="0.2", marker="o", ms=2, label=t_.split("_")[0])
ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("Misclassified countries (%)"); ax.legend(fontsize=5.6); ax.set_title("Error by coupling", fontsize=7); panel_letter(ax, "c", dx=-0.2)
ax = axs[3]
cr = NUM["criteria"]; names = [("capkeep", "capacity\nretention"), ("reserve", "reserve rule\n$\\kappa q>m$"), ("deplete", "long-run\ndepletion")]
for ii, (met, colr) in enumerate([("precision", "0.25"), ("recall", "0.65")]):
    ax.bar(np.arange(3) + (ii - 0.5) * 0.35, [cr[f"{n}|new|k_le_obs"][met] for n, _ in names], 0.35, color=colr, label=met)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in names], fontsize=5.5); ax.set_ylim(0, 1.05)
ax.set_ylabel("Score against new depletions"); ax.legend(fontsize=5.6, loc="upper left"); ax.set_ylim(0, 1.15); ax.set_title(r"First-order criteria ($\kappa \leq$ observed)", fontsize=7); panel_letter(ax, "d", dx=-0.15)
save(fig, "ED_Fig4")

# ============================================================ ED Fig 5: behavioural calibration
E7 = rd("E7"); ob = E7[np.isclose(E7.kappa, E7.commodity.map(KOBS))]
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.1, wspace=0.45, hspace=0.55); axs = axs.ravel()
ax = axs[0]
for nm, colr, lab in [("phi_emp", "0.2", "all games"), ("phi_equalcond", COL["Equal"], "equal-sharing games"), ("phi_rlcond", COL["Distilled"], "RL-planner games")]:
    v = np.sort(np.load(os.path.join(HERE, "nets", f"{nm}.npy"))); ax.plot(v, np.arange(v.size) / v.size, color=colr, lw=1.1, label=lab)
v = np.sort(np.load(os.path.join(HERE, "nets", "phi_shrunk.npy"))); ax.plot(v, np.arange(v.size) / v.size, color="0.2", lw=0.8, ls="--", label="reliability-shrunk")
ax.axvline(1 / 1.4, color="#D55E00", lw=0.8, ls=":"); ax.set_xlabel(r"Return fraction $\phi$ (player-game)"); ax.set_ylabel("Cumulative share")
ax.legend(fontsize=5.2); ax.set_title(f"Human play, n = {HUM['n_player_games']:,}", fontsize=7); panel_letter(ax, "a", dx=-0.2)
ax = axs[1]
br = HUM["by_round"]; ax.plot(range(len(br)), list(br.values()), "-o", color="0.2", ms=2.5)
ax.set_xticks(range(len(br))); ax.set_xticklabels([k_.replace("[", "").replace(")", "").replace(", ", "-") for k_ in br], rotation=90, fontsize=5.5)
ax.set_ylabel("Pooled return fraction"); ax.set_xlabel("Rounds"); ax.set_title("Stable within games", fontsize=7); panel_letter(ax, "b")
ax = axs[2]
VV = [v_ for v_ in T1 if v_.startswith(("base", "human_", "B2_", "B3_", "phi_", "f_", "n_", "r_")) and "boot" not in v_ and "lammax_Distilled" in T1[v_]]
for v_ in VV:
    for nm, mk_ in [("Equal", "o"), ("Distilled", "s")]:
        r_ = T1[v_]
        if f"lammax_{nm}" in r_ and np.isfinite(r_.get(f"lammax_{nm}", np.nan)):
            sc = ax.scatter(r_[f"lamfull_{nm}"], r_[f"lammax_{nm}"], c=[r_[f"casc_{nm}"]], cmap="viridis", vmin=0, vmax=1, marker=mk_, s=14, edgecolors="none")
ax.plot([0.7, 1.45], [0.7, 1.45], color="0.6", lw=0.5); ax.axhline(1, color="0.6", lw=0.5, ls=":"); ax.axvline(1, color="0.6", lw=0.5, ls=":")
cbar = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.02); cbar.set_label("Newly depleted", fontsize=6); cbar.ax.tick_params(labelsize=5.5)
ax.set_xlabel(r"Mean full-pool factor $\lambda^{\rm full}$"); ax.set_ylabel(r"Mean scarcity factor $\lambda^{\max}$"); ax.set_title("Behaviour variants (circles equal, squares res.-cont.)", fontsize=7); panel_letter(ax, "c", dx=-0.2)
ax = axs[3]
HV = [("human_raw_n8", "human, n=8"), ("human_raw_n4", "human, n=4"), ("human_raw_n16", "human, n=16"), ("human_shrunk_n8", "shrunk"),
      ("human_equalcond_n8", "equal-cond."), ("human_rlcond_n8", "RL-cond."), ("human_games_n8", "intact games"), ("human_raw_noexec", "no exec. noise"), ("human_boot_summary", "game bootstrap")]
for yi, (v_, lab) in enumerate(HV):
    r_ = T1.get(v_, {})
    for nm, dy in [("Equal", -0.15), ("Mixed", 0.0), ("Distilled", 0.15)]:
        if f"casc_{nm}" in r_:
            xe = T1["human_boot_summary_sd"][f"casc_{nm}"] if v_ == "human_boot_summary" else 0
            ax.errorbar(r_[f"casc_{nm}"], yi + dy, xerr=xe, color=COL[nm], marker="o", ms=2.5, lw=0.7)
            ax.plot(r_[f"final_{nm}"], yi + dy, marker="o", mfc="white", mec=COL[nm], ms=3, ls="none")
ax.set_yticks(range(len(HV))); ax.set_yticklabels([l for _, l in HV], fontsize=5.5); ax.invert_yaxis(); ax.set_xlabel("Newly depleted non-seed (filled); all depleted (open)")
ax.set_title("Human-calibrated variants, observed $\\kappa$", fontsize=7); panel_letter(ax, "d", dx=-0.35)
save(fig, "ED_Fig5")

# ============================================================ ED Fig 6: adaptation
b3 = ob[ob.behaviour == "B3"]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7)); fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.38, wspace=0.45)
ax = axs[0]
for nm in ["Equal", "Distilled"]:
    d = b3[(b3.variant == "B3_base") & (b3.inst == nm)]
    ts_ = [100, 500, 1000, 1500]
    ax.plot(ts_, [d[f"evo_phimean_{t_}"].mean() for t_ in ts_], "-o", color=COL[nm], ms=2.5, label=f"{SHORT[nm]}: mean $\\phi$")
    ax.plot(ts_, [d[f"evo_phimax_{t_}"].mean() for t_ in ts_], "--s", color=COL[nm], ms=2.5, label=f"{SHORT[nm]}: max $\\phi$")
ax.axhline(1 / 1.4, color="#D55E00", lw=0.7, ls=":"); ax.set_xlabel("Round of imitation"); ax.set_ylabel("Return fraction"); ax.legend(fontsize=5)
ax.set_title("Imitation erodes the cooperative tail", fontsize=7); panel_letter(ax, "a", dx=-0.25)
BV = [("base", "fixed $\\phi$"), ("B3_base", "imitation"), ("B3_burn3000", "3,000 rounds"), ("B3_beta2.0", "strong selection"), ("B3_beta10.0", "weak selection"),
      ("B3_mut0.005", "mutation 0.005"), ("B3_mut0.02", "mutation 0.02"), ("B3_epoch10", "update every 10"), ("B3_epoch40", "update every 40"), ("B3_normpay", "normalised payoffs")]
for ax, key, ttl, let in [(axs[1], "final", "All depleted after hub failure", "b"), (axs[2], "lat", "Depleted after trade halts", "c")]:
    for ii, nm in enumerate(["Equal", "Distilled"]):
        ax.bar(np.arange(len(BV)) + (ii - 0.5) * 0.38, [T1.get(v_, {}).get(f"{key}_{nm}", np.nan) for v_, _ in BV], 0.38, color=COL[nm], label=SHORT[nm])
    ax.set_xticks(range(len(BV))); ax.set_xticklabels([l for _, l in BV], rotation=90, fontsize=5.5); ax.set_ylabel(ttl); ax.set_title("Observed $\\kappa$, four networks", fontsize=7); panel_letter(ax, let)
axs[1].legend(fontsize=5.4)
save(fig, "ED_Fig6")

# ============================================================ ED Fig 7: institutional ablation and worked trajectory
E8 = rd("E8"); e8 = E8[np.isclose(E8.kappa, E8.commodity.map(KOBS))]
fig = plt.figure(figsize=(W2, 4.8)); gs = fig.add_gridspec(2, 3, left=0.2, right=0.99, top=0.94, bottom=0.08, wspace=0.5, hspace=0.5)
def _rk(r_):
    fam = ["distilled", "power", "step", "logistic", "fixed", "wlo", "whi", "floor"]
    i = next((n for n, p_ in enumerate(fam) if r_.startswith(p_)), 9)
    num = re.findall(r"[0-9.]+", r_); return (i, float(num[-1].rstrip(".")) if num else 0)
RULES = sorted(dict.fromkeys(E8.rule), key=_rk)
lab = {r_: r_.replace("step_th", "step, threshold ").replace("power_k", "power, k = ").replace("logistic_tau", "logistic, width ").replace("wlo", "low-pool equal weight ").replace("whi", "full-pool equal weight ").replace("fixed", "fixed weight ").replace("floor", "equal-share floor ").replace("distilled", "distilled rule (k = 22)") for r_ in RULES}
for col_, (ycol, ttl) in enumerate([("cascade", "Newly depleted"), ("latent_dep", "Depleted after halt"), ("efficiency", "Modelled surplus")]):
    ax = fig.add_subplot(gs[0, col_])
    for yi, r_ in enumerate(RULES):
        s = strat(e8[e8.rule == r_], ycol, ["rule"])
        ax.errorbar(s["mean"].iloc[0], yi, xerr=s.ci.iloc[0], color=COL["Distilled"] if r_ == "distilled" else "0.3", marker="o", ms=2.5, lw=0.7)
    ax.set_yticks(range(len(RULES))); ax.set_yticklabels([lab[r_] for r_ in RULES] if col_ == 0 else [], fontsize=5.2); ax.invert_yaxis()
    ax.set_xlabel(ttl); ax.set_title("Observed $\\kappa$" if col_ else "Rule ablation", fontsize=7); panel_letter(ax, "abc"[col_], dx=-0.8 if col_ == 0 else -0.18)
for ii, (inst, nm) in enumerate([(0, "Equal"), (3, "Distilled"), (4, "Step")]):
    ax = fig.add_subplot(gs[1, ii]); ag = EX[f"worked_{inst}_agents"]
    ax.plot(ag[:, 0, 0], color="#0072B2", lw=1.0, label="allocation, top contributor ($\\phi$=0.95)")
    ax.plot(ag[:, 0, 7], color="#D55E00", lw=1.0, label="allocation, free rider")
    ax.plot(ag[:, 2, 7], color="#D55E00", lw=0.7, ls=":", label="kept, free rider")
    ax.plot(EX[f"worked_{inst}_R"] * 200 / 8, color="0.5", lw=0.7, ls="--", label="equal share of pool")
    ax.axvspan(30, 40, color="0.9", lw=0); ax.set_xlabel("Round"); ax.set_ylabel("Units per member" if ii == 0 else "")
    ax.set_title(SHORT[nm], fontsize=7); panel_letter(ax, "def"[ii])
    if ii == 0: ax.legend(fontsize=5, loc="upper right")
save(fig, "ED_Fig7")

# ============================================================ ED Fig 8: welfare decomposition
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6)); fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.33, wspace=0.45)
k25 = E2[np.isclose(E2.kappa, E2.commodity.map(KOBS))]
ax = axs[0]
kept = [k25[k25.inst == nm].groupby("commodity").efficiency.mean().mean() for nm in ALL5]
waste = [k25[k25.inst == nm].groupby("commodity").waste.mean().mean() for nm in ALL5]
ax.bar(np.arange(5) - 0.19, kept, 0.38, color=[COL[n] for n in ALL5], label="kept by members")
ax.bar(np.arange(5) + 0.19, waste, 0.38, color="none", edgecolor=[COL[n] for n in ALL5], hatch="///", lw=0.6, label="lost to overflow at capacity")
ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[n] for n in ALL5], rotation=90, fontsize=5.5); ax.set_ylabel("Relative to sustainable ceiling")
ax.legend(fontsize=5.4, loc="upper left"); ax.set_ylim(0, 1.45); ax.set_title("Surplus kept and overflow", fontsize=7); panel_letter(ax, "a", dx=-0.3)
ax = axs[1]
for ii, (col, hatch) in enumerate([("gini_cum", None), ("gini_late", "///")]):
    ax.bar(np.arange(5) + (ii - 0.5) * 0.38, [k25[k25.inst == nm].groupby("commodity")[col].mean().mean() for nm in ALL5], 0.38,
           color=[COL[n] for n in ALL5] if hatch is None else "none", edgecolor=[COL[n] for n in ALL5], hatch=hatch, lw=0.6, label="cumulative" if ii == 0 else "last 100 rounds")
ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[n] for n in ALL5], rotation=90, fontsize=5.5); ax.set_ylabel("Within-country Gini of payoffs"); ax.set_ylim(0, 1.0); ax.legend(fontsize=5.4, loc="upper left")
ax.set_title("Inequality", fontsize=7); panel_letter(ax, "b")
ax = axs[2]
for nm in ALL5:
    ax.plot([1, 5, 10, 20], [k25[k25.inst == nm].groupby("commodity")[f"excl{t_}_late"].mean().mean() for t_ in (1, 5, 10, 20)], "-o", color=COL[nm], ms=2.5, label=SHORT[nm])
ax.set_xlabel("Threshold (% of equal share)"); ax.set_ylabel("Share of members below threshold"); ax.legend(fontsize=5); ax.set_title("Exclusion", fontsize=7); panel_letter(ax, "c")
save(fig, "ED_Fig8")

# ============================================================ ED Fig 9: tail-risk sensitivity
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.14, wspace=0.3, hspace=0.75); axs = axs.ravel()
k4 = E3[(E3.hazard == "both") & (E3.rule == "none") & np.isclose(E3.kappa, 0.4)]
ax = axs[0]
MET = [("sys40_any", ">40% countries"), ("sys50_any", ">50% countries"), ("sys60_any", ">60% countries"), ("syscap50_any", ">50% of capacity"),
       ("sys50_dep03_any", "pool < 0.3 K"), ("sys50_dep07_any", "pool < 0.7 K")]
for ii, nm in enumerate(MAIN3):
    ax.bar(np.arange(len(MET)) + (ii - 1) * 0.27, [k4[k4.inst == nm][m_].mean() for m_, _ in MET], 0.27, color=COL[nm], label=SHORT[nm])
ax.set_xticks(range(len(MET))); ax.set_xticklabels([l for _, l in MET], rotation=90, fontsize=5.5); ax.set_ylabel("Runs with an event")
ax.legend(fontsize=5.4); ax.set_title("Event definitions, $\\kappa$ = 0.4", fontsize=7); panel_letter(ax, "a", dx=-0.2)
ax = axs[1]
sens = E3[(E3.rule == "none") & np.isclose(E3.kappa, 0.4)]
HZ = ["both", "harvest_only", "crisis_only", "harvest_p=0.01", "harvest_p=0.04", "harvest_loss=0.3", "harvest_loss=0.7", "crisis_p=0.0005", "crisis_p=0.002", "crisis_dur=10", "crisis_dur=40", "regen=0.0", "regen=0.0025", "regen=0.01"]
for ii, nm in enumerate(MAIN3):
    ax.bar(np.arange(len(HZ)) + (ii - 1) * 0.27, [sens[(sens.inst == nm) & (sens.hazard == h_)].sys50_any.mean() for h_ in HZ], 0.27, color=COL[nm])
ax.set_xticks(range(len(HZ))); ax.set_xticklabels([h_.replace("_", " ").replace("both", "baseline") for h_ in HZ], rotation=90, fontsize=5.5); ax.set_ylabel("Runs with a systemic event")
ax.set_title("Hazard process, $\\kappa$ = 0.4", fontsize=7); panel_letter(ax, "b", dx=-0.2)
ax = axs[2]
obs = E3[(E3.hazard == "both") & (E3.rule == "none") & np.isclose(E3.kappa, E3.commodity.map(KOBS))]
for ii, nm in enumerate(ALL5):
    ps = []; los = []; his = []
    for c in COMS:
        d = obs[(obs.inst == nm) & (obs.commodity == c)]; k_ = d.sys50_any.sum(); n_ = max(len(d), 1); lo, hi = wilson(k_, n_); ps.append(k_ / n_); los.append(k_ / n_ - lo); his.append(hi - k_ / n_)
    ax.errorbar(np.arange(4) + (ii - 2) * 0.15, ps, yerr=[los, his], color=COL[nm], marker="o", ms=2.5, lw=0.7, ls="none", label=SHORT[nm])
ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\n$\\kappa$={KOBS[c]:.2f}" for c in COMS], fontsize=5.5); ax.set_ylabel("Runs with a systemic event")
ax.legend(fontsize=5.2, loc="upper left", ncol=2); ax.set_title("Observed coupling (Wilson 95% intervals)", fontsize=7); panel_letter(ax, "c", dx=-0.2)
ax = axs[3]
RU = ["none", "ban", "ban_th0.6", "ban_th0.7", "ban_th0.9", "ban_hyst", "halve", "halve_hyst", "quarter"]
rv = E3[(E3.hazard == "both") & np.isclose(E3.kappa, 0.6)]
for ii, nm in enumerate(MAIN3):
    ax.bar(np.arange(len(RU)) + (ii - 1) * 0.27, [rv[(rv.inst == nm) & (rv.rule == r_)].sys50_any.mean() for r_ in RU], 0.27, color=COL[nm])
ax.set_xticks(range(len(RU))); ax.set_xticklabels([r_.replace("_", " ") for r_ in RU], rotation=90, fontsize=5.5); ax.set_ylabel("Runs with a systemic event")
ax.set_title("Restriction designs, $\\kappa$ = 0.6", fontsize=7); panel_letter(ax, "d", dx=-0.2)
save(fig, "ED_Fig9")

# ============================================================ ED Fig 10: mobility
E6 = rd("E6"); w6 = E6[E6.commodity == "wheat"]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4)); fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.2, wspace=0.5)
ax = axs[0]
fr6 = w6[(w6.who == "freeriders") & (w6.f_mean == 0.1) & (w6.h_eq > 0) & (w6.h_eq < 1)]
for m_, ls in [(0.05, "-"), (0.2, "--")]:
    g = fr6[fr6.migration == m_].groupby("h_eq")
    ax.plot(g.eq_fr_pooled.mean().index, 100 * (g.eq_fr_pooled.mean() - g.eq_fr0_pooled.mean()), ls, marker="o", ms=2.5, color=COL["Equal"], label=f"equal-sharing, m = {m_}")
    ax.plot(g.d_fr_pooled.mean().index, 100 * (g.d_fr_pooled.mean() - g.d_fr0_pooled.mean()), ls, marker="o", ms=2.5, color=COL["Distilled"], label=f"resource-contingent, m = {m_}")
ax.axhline(0, color="0.5", lw=0.6, ls=":"); ax.set_xlabel("Share of equal-sharing countries"); ax.set_ylabel("Change in free-rider share (points)"); ax.legend(fontsize=5)
ax.set_title("Weak sorting of free riders", fontsize=7); panel_letter(ax, "a", dx=-0.25)
ax = axs[1]
for m_, colr in [(0.0, "0.6"), (0.05, "0.2")]:
    g = w6[(w6.who == "freeriders") & (w6.f_mean == 0.1) & (w6.migration == m_) & (w6.h_eq > 0)].groupby("h_eq").eq_dep
    ax.errorbar(g.mean().index, g.mean(), yerr=g.agg(tci), color=colr, marker="o", ms=2.5, lw=0.9, capsize=1.5, label="no mobility" if m_ == 0 else "free riders mobile")
ax.set_xlabel("Share of equal-sharing countries"); ax.set_ylabel("Depleted equal-sharing countries"); ax.legend(fontsize=5.4); ax.set_title("Mobility dilutes free riding", fontsize=7); panel_letter(ax, "b")
ax = axs[2]
a6 = w6[w6.f_mean == 0.05]
for who, colr in [("freeriders", "0.2"), ("all", "#CC79A7")]:
    g = a6[a6.who == who].groupby("migration")
    ax.plot(g.dep_all.mean().index, g.dep_all.mean(), "-o", color=colr, ms=2.5, label={"freeriders": "free riders move", "all": "everyone moves"}[who] + ": all")
    ax.plot(g.dep_occupied.mean().index, g.dep_occupied.mean(), "--s", color=colr, ms=2.5, label={"freeriders": "free riders move", "all": "everyone moves"}[who] + ": occupied")
ax.set_xlabel("Mobility per 10 rounds, m"); ax.set_ylabel("Depleted countries"); ax.legend(fontsize=5); ax.set_title("Depopulation drives depletion", fontsize=7); panel_letter(ax, "c")
save(fig, "ED_Fig10")
print("ED figures done", flush=True)
