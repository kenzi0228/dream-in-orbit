"""Analyse des deux circuits dominants (NPWD2551, NPWD2451) : distribution, niveaux, features correlees,
commandes DMOP (toutes, y compris rares) associees aux transitions, regime des mauvaises semaines."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.features import split_stem
H = 3600000
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
print("vocab : cmds", len(v["cmds"]), "stems", len(v.get("stems", [])), "switches", len(v.get("switches", [])), "prefixes", len(v["prefixes"]))
for circ in ["NPWD2551", "NPWD2451", "NPWD2532", "NPWD2851"]:
    print(f"\n######## {circ}")
    for name, Y in [("annee1", Y1), ("annee2", Y2)]:
        y = Y[circ].dropna()
        q = np.quantile(y, [0, .05, .25, .5, .75, .95, 1])
        print(f" {name}: moy {y.mean():.3f} std {y.std():.3f} quantiles {np.round(q,3)}  autocorr lag1 {y.autocorr(1):.3f} lag24 {y.autocorr(24):.3f}")
        d = y.diff().abs(); print(f"   sauts horaires |d|>0.3 : {int((d>0.3).sum())} ({100*(d>0.3).mean():.2f}%)  |d|>0.1 : {int((d>0.1).sum())}")
    # niveaux bruts (sous-horaires) sur annee1
    raw = pd.read_csv(kit / "annee1" / "power.csv", usecols=["ut_ms", circ])
    r = raw[circ].to_numpy(); hist, edges = np.histogram(r, bins=20)
    top = np.argsort(hist)[::-1][:6]
    print("  niveaux bruts (annee1) : bins les plus peuples :", [(round(float(edges[i]),3), round(float(edges[i+1]),3), round(100*hist[i]/len(r),1)) for i in sorted(top)])
    # features les plus correlees (annee1 et annee2 separement, pour voir la stabilite)
    for name, X, Y in [("annee1", X1, Y1), ("annee2", X2, Y2)]:
        y = Y[circ]; m = y.notna()
        Xs = X.loc[m]; ys = y[m]
        sd = Xs.std(); keep = sd[sd > 0].index
        c = Xs[keep].apply(lambda col: np.corrcoef(col, ys)[0, 1]) if False else ((Xs[keep] - Xs[keep].mean()) * (ys.values[:, None] - ys.mean())).mean() / (sd[keep] * ys.std())
        c = c.dropna().sort_values(key=np.abs, ascending=False)
        print(f"  top corr {name}:", [(k, round(float(val), 2)) for k, val in c.head(14).items()])
    # commandes DMOP (toutes) associees aux grandes transitions (annee1 + annee2)
    stats = {}
    for name, Y in [("annee1", Y1), ("annee2", Y2)]:
        dm = pd.read_csv(kit / name / "dmop.csv"); dm["h"] = dm["ut_ms"] // H * H
        y = Y[circ]; d = y.diff()
        up = set(y.index[d > 0.25]); dn = set(y.index[d < -0.25])
        hours_all = set(y.index)
        for cmd, g in dm.groupby("subsystem"):
            hs = set(g["h"]) | set(g["h"] - H) ; n = len(hs & hours_all)
            if n < 5: continue
            s = stats.setdefault(cmd, [0, 0, 0, 0, 0, 0])
            s[0] += n; s[1] += len(hs & up); s[2] += len(hs & dn); s[3] += len(up); s[4] += len(dn)
    rows = []
    for cmd, (n, nu, nd, Nu, Nd, _) in stats.items():
        rows.append((cmd, n, nu, nd, round(nu / max(Nu, 1), 2), round(nd / max(Nd, 1), 2), cmd in set(v["cmds"]), split_stem(cmd)[0] in set(v.get("stems", []))))
    df = pd.DataFrame(rows, columns=["cmd", "n_h", "n_up", "n_down", "couv_up", "couv_down", "in_vocab_cmd", "in_vocab_stem"])
    df["score"] = (df.n_up + df.n_down) / (df.n_h + 20) * np.log1p(df.n_up + df.n_down)
    print("  commandes liees aux transitions (±0.25) :"); print(df.sort_values("score", ascending=False).head(14).to_string(index=False))
# regime des semaines : erreur du melange par semaine vs ev_umbra, ftl, flux, dm_total (strict)
meta = json.load(open("weights_holdout/meta.json")); w = meta["gbm_weight"]
blend = pd.read_csv("weights_holdout/val_pred.csv", index_col="ut_ms")
T = Y2.loc[blend.index]; e2 = ((blend - T) ** 2)
wk = (np.arange(len(blend)) // 168)
df = pd.DataFrame({"sem": wk + 1, "mse": e2.mean(axis=1).values, "mse_2551": e2["NPWD2551"].values, "mse_2451": e2["NPWD2451"].values,
                   "umbra": X2.loc[blend.index, "ev_umbra"].values if "ev_umbra" in X2 else 0, "comms": X2.loc[blend.index, "ftl_comms"].values if "ftl_comms" in X2 else 0,
                   "fx": X2.loc[blend.index, "fx_total"].values, "dm": X2.loc[blend.index, "dm_total"].values}).groupby("sem").mean()
df["rmse"] = np.sqrt(df.mse); df["part_2551+2451_%"] = 100 * (df.mse_2551 + df.mse_2451) / 33 / df.mse
print("\nsemaines (strict) triees par erreur :"); print(df.sort_values("rmse", ascending=False).head(12).round(3).to_string())
print(df.sort_values("rmse").head(6).round(3).to_string())
