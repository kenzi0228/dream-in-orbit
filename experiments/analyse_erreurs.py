"""Analyse d'erreurs hors ligne (aucun entrainement) : GRU vs GBM par semaine et par circuit, deux protocoles,
melange pondere par l'incertitude, projection de rang faible. Utilise les predictions deja sur disque."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
k = len(X2) // 2
protos = {"strict": (Y1, X2, Y2, "weights_holdout"), "split": (pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:], "weights_holdout_split")}
out = {}
for proto, (Ytr, Xte, Yte, wdir) in protos.items():
    meta = json.load(open(f"{wdir}/meta.json")); w = meta["gbm_weight"]
    blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms").to_numpy()
    std = pd.read_csv(f"{wdir}/val_std.csv", index_col="ut_ms").to_numpy()
    g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
    gru = (blend - w * pg) / (1 - w)
    T = Yte.to_numpy(); m = ~np.isnan(T).any(axis=1); m[:N] = False
    gru, pg, blend, std, T = gru[m], pg[m], blend[m], std[m], T[m]
    print(f"\n===== {proto} : gru {rmse(gru,T):.5f} gbm {rmse(pg,T):.5f} blend {rmse(blend,T):.5f} (n={len(T)})")
    # par semaine (horizon)
    nw = len(T) // 168
    rows = []
    for i in range(nw):
        s = slice(i * 168, (i + 1) * 168)
        rows.append((i + 1, rmse(gru[s], T[s]), rmse(pg[s], T[s]), rmse(blend[s], T[s]), T[s].mean()))
    df = pd.DataFrame(rows, columns=["semaine", "gru", "gbm", "blend", "y_moy"])
    print("par semaine (toutes les 4) :"); print(df.iloc[::4].round(4).to_string(index=False))
    print("corr(semaine, err gru) =", round(np.corrcoef(df.semaine, df.gru)[0, 1], 3), " corr(semaine, err gbm) =", round(np.corrcoef(df.semaine, df.gbm)[0, 1], 3))
    # par circuit : part de la MSE
    mse_c = ((blend - T) ** 2).mean(axis=0); part = mse_c / mse_c.sum()
    pc = pd.DataFrame({"circuit": Yte.columns, "gru": np.sqrt(((gru - T) ** 2).mean(0)), "gbm": np.sqrt(((pg - T) ** 2).mean(0)),
                       "blend": np.sqrt(mse_c), "part_mse_%": 100 * part, "y_moy": T.mean(0), "y_std": T.std(0)}).sort_values("part_mse_%", ascending=False)
    print("par circuit (top 12, part de la MSE du melange) :"); print(pc.head(12).round(4).to_string(index=False))
    # poids optimal par circuit (info)
    # melange pondere par l'incertitude : w_gbm = clip(a + b * z), z = std / mediane(std) par circuit
    z = std / np.median(std, axis=0, keepdims=True)
    best = None
    for a in np.arange(0.2, 0.8, 0.05):
        for b in np.arange(-0.2, 0.41, 0.05):
            wg = np.clip(a + b * (z - 1), 0, 1)
            r = rmse((1 - wg) * gru + wg * pg, T)
            if best is None or r < best[0]: best = (r, round(a, 2), round(b, 2))
    print("gating incertitude : meilleur (rmse, a, b) =", (round(best[0], 5), best[1], best[2]), " vs constant 0.45 :", round(rmse(0.55 * gru + 0.45 * pg, T), 5))
    out.setdefault("gating", {})[proto] = best
    # biais global par circuit (le melange sur/sous-estime-t-il ?)
    bias = (blend - T).mean(0); print("biais moyen |b| moyen =", round(np.abs(bias).mean(), 4), " RMSE si biais par circuit retire (oracle) =", round(rmse(blend - bias, T), 5))
    # projection de rang faible : composantes principales de Ytr
    mu = np.nanmean(Ytr.to_numpy(), axis=0); Yc = Ytr.to_numpy() - mu; Yc = Yc[~np.isnan(Yc).any(axis=1)]
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    for kk in [8, 12, 16, 20, 24, 28]:
        P = Vt[:kk].T @ Vt[:kk]
        proj = (blend - mu) @ P + mu
        print(f"  rang {kk}: blend projete {rmse(proj, T):.5f}  | 50/50 {rmse(0.5*proj+0.5*blend, T):.5f}")
    var_expl = np.cumsum(S ** 2) / np.sum(S ** 2); print("  variance expliquee rang 8/16/24 :", np.round(var_expl[[7, 15, 23]], 4))
    # erreurs : part des heures 'extremes' (|err| > 0.3)
    e = np.abs(blend - T); print("  part de la MSE due aux |err|>0.3 :", round(100 * (e[e > 0.3] ** 2).sum() / (e ** 2).sum(), 1), "% sur", round(100 * (e > 0.3).mean(), 2), "% des cases")
    # oracle de melange temporel : poids optimal par semaine
    ws = []
    for i in range(nw):
        s = slice(i * 168, (i + 1) * 168); bestw = min(np.arange(0, 1.01, 0.05), key=lambda ww: rmse((1 - ww) * gru[s] + ww * pg[s], T[s])); ws.append(bestw)
    print("  poids gbm optimal par semaine : moy", round(np.mean(ws), 2), " std", round(np.std(ws), 2), " oracle rmse", round(np.sqrt(np.mean([rmse((1-ws[i])*gru[i*168:(i+1)*168]+ws[i]*pg[i*168:(i+1)*168], T[i*168:(i+1)*168])**2 for i in range(nw)])), 5))
    out.setdefault("per_circuit", {})[proto] = pc.round(5).to_dict("records")
Path("reports/analyse_erreurs.json").write_text(json.dumps(out, indent=1, default=float))
