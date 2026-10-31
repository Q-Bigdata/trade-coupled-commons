"""Behavioural calibration on the human games of Koster et al. (2025).  Input: human_rounds.csv.gz, one row
per player-round of the 680 human games (columns game, launch, cond, round, pool, player, offer, action).
Outputs: results/human_summary.json, nets/phi_*.npy."""
import os, json, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); os.makedirs(os.path.join(HERE, "nets"), exist_ok=True)
L = pd.read_csv(os.path.join(HERE, "human_rounds.csv.gz"))
THR = 1 / 1.4
S = {}
L = L[(L.offer >= 0.5) & L.action.notna()].copy()
L["frac"] = np.clip(L.action / L.offer, 0, 1)
L["exp"] = L.cond.str.extract(r"Exp (\d)").astype(int)
L["pg"] = L.game.astype(str) + "_" + L.player.astype(str)
E = L[L["round"] < 30]
S["n_games"] = int(E.game.nunique()); S["n_player_games"] = int(E.pg.nunique())
S["n_player_rounds"] = int(len(E))
S["n_launch_player"] = int((E.launch.astype(str) + "_" + E.player.astype(str)).nunique())
S["games_by_exp"] = E.groupby("exp").game.nunique().to_dict()


def pgphi(d):
    g = d.groupby(["pg", "game", "launch", "cond", "exp", "player"]).agg(a=("action", "sum"), o=("offer", "sum"), n=("action", "size")).reset_index()
    g["phi"] = np.clip(g.a / g.o, 0, 1)
    return g


G = pgphi(E)
S["phi_mean"] = float(G.phi.mean()); S["phi_median"] = float(G.phi.median())
S["share_above_sustainability"] = float(np.mean(G.phi > THR)); S["share_below_0.1"] = float(np.mean(G.phi < 0.1))
S["by_condition"] = {c: dict(n=int(len(d)), mean=float(d.phi.mean()), median=float(d.phi.median()), sd=float(d.phi.std()),
                           share_above_sust=float(np.mean(d.phi > THR)), share_below_01=float(np.mean(d.phi < 0.1)))
                     for c, d in G.groupby("cond")}
# split-half reliability: rounds 0-14 versus 15-29
e1 = pgphi(E[E["round"] < 15]).set_index("pg").phi; e2 = pgphi(E[E["round"] >= 15]).set_index("pg").phi
j = pd.concat([e1, e2], axis=1, keys=["a", "b"]).dropna()
r_half = float(np.corrcoef(j.a, j.b)[0, 1]); rel = 2 * r_half / (1 + r_half)
S["split_half_r"] = r_half; S["spearman_brown_reliability"] = rel; S["split_half_n"] = int(len(j))
S["early_mean"] = float(j.a.mean()); S["late_mean"] = float(j.b.mean())
# game-to-game stability in Experiment 4 (same group plays three consecutive games)
g4 = G[G.exp == 4].copy()
g4["gnum"] = g4.cond.str.extract(r"Game (\d)").astype(float)
piv = g4.pivot_table(index=["launch", "player"], columns="gnum", values="phi")
S["exp4_retest"] = {f"{a:.0f}-{b:.0f}": dict(r=float(piv[[a, b]].dropna().corr().iloc[0, 1]), n=int(len(piv[[a, b]].dropna())))
                    for a, b in [(1, 2), (2, 3), (1, 3)] if a in piv and b in piv}
# per-round response to the relative offer, player-game demeaned, cluster-robust by game
M = E[(~E.cond.str.contains("Equal")) & (E.pool > 20)].copy()
M["rel"] = (M.offer / (M.pool / 4) - 1).clip(-1, 3)


def fe_ols(d, cl):
    y = d.frac - d.groupby("pg").frac.transform("mean"); x = d.rel - d.groupby("pg").rel.transform("mean")
    b = float((x * y).sum() / (x * x).sum()); u = y - b * x
    sc = (x * u).groupby(d[cl]).sum(); V = float((sc ** 2).sum() / (x * x).sum() ** 2)
    G_ = d[cl].nunique(); V *= G_ / (G_ - 1)
    return dict(b=b, se=float(np.sqrt(V)), n=int(len(d)), clusters=int(G_))


S["reciprocity"] = dict(all_game=fe_ols(M, "game"), all_launch=fe_ols(M, "launch"),
                        early=fe_ols(M[M["round"] < 15], "game"), late=fe_ols(M[M["round"] >= 15], "game"),
                        low_pool=fe_ols(M[M.pool <= 100], "game"),
                        by_exp={int(k): fe_ols(d, "game") for k, d in M.groupby("exp") if d.game.nunique() > 5})
# round and pool-state profiles
E2 = E.copy(); E2["rbin"] = pd.cut(E2["round"], [0, 5, 10, 15, 20, 25, 30], right=False)
E2["pbin"] = pd.cut(E2.pool, [0, 50, 100, 150, 190, 201])
S["by_round"] = {str(k): float(v) for k, v in E2.groupby("rbin", observed=True).apply(lambda d: d.action.sum() / d.offer.sum()).items()}
S["by_pool"] = {str(k): float(v) for k, v in E2.groupby("pbin", observed=True).apply(lambda d: d.action.sum() / d.offer.sum()).items()}
# calibration samples
phi = G.phi.values
mu = phi.mean()
shr = np.clip(mu + np.sqrt(max(rel, 0)) * (phi - mu), 0, 1)
np.save(os.path.join(HERE, "nets", "phi_emp.npy"), phi)
np.save(os.path.join(HERE, "nets", "phi_shrunk.npy"), shr)
np.save(os.path.join(HERE, "nets", "phi_equalcond.npy"), G[G.cond.str.contains("Equal")].phi.values)
np.save(os.path.join(HERE, "nets", "phi_rlcond.npy"), G[G.cond.str.contains("RL")].phi.values)
groups = [d.phi.values for _, d in G.groupby("game")]
np.save(os.path.join(HERE, "nets", "phi_games.npy"), np.array([np.pad(gp, (0, 4 - len(gp)), constant_values=np.nan) if len(gp) < 4 else gp[:4] for gp in groups]))
rng = np.random.default_rng(7)
games = G.game.unique()
boots = []
for b in range(20):
    pick = rng.choice(games, games.size, replace=True)
    boots.append(np.concatenate([G[G.game == g_].phi.values for g_ in pick]))
np.save(os.path.join(HERE, "nets", "phi_boot.npy"), np.array([np.resize(bb, phi.size) for bb in boots]))
S["shrunk_sd"] = float(shr.std()); S["raw_sd"] = float(phi.std())
S["shrunk_share_above_sust"] = float(np.mean(shr > THR))
# community-level composition for n = 4, 8, 16 random draws
for n in (4, 8, 16):
    draws = rng.choice(phi, (20000, n))
    S[f"community_n{n}"] = dict(mean_of_means=float(draws.mean(1).mean()), share_mean_above_sust=float(np.mean(draws.mean(1) > THR)),
                                 mean_max=float(draws.max(1).mean()), share_max_above_sust=float(np.mean(draws.max(1) > THR)))
json.dump(S, open(os.path.join(HERE, "results", "human_summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: S[k] for k in ["n_games", "n_player_games", "phi_mean", "split_half_r", "spearman_brown_reliability"]}))
