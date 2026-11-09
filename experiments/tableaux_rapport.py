"""Chiffres des sections 4 du rapport a partir de weights_holdout (protocole strict) : RMSE par circuit des deux
composantes et du melange, derive par semaine, dispersion de l'incertitude, pire semaine. Ecrit reports/tableaux_rapport.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"]); X2, Y2 = load_period(kit / "annee2", v)
wdir = "weights_holdout"; meta = json.load(open(f"{wdir}/meta.json")); w0 = meta["gbm_weight"]
blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms"); std = pd.read_csv(f"{wdir}/val_std.csv", index_col="ut_ms").to_numpy()
g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(X2[g.features].to_numpy())
B = blend.to_numpy(); gru = (B - w0 * pg) / (1 - w0); T = Y2.to_numpy(); m = ~np.isnan(T).any(axis=1); m[:N] = False
cols = list(blend.columns)
out = {"rmse_blend": rmse(B[m], T[m]), "rmse_gru": rmse(gru[m], T[m]), "rmse_gbm": rmse(pg[m], T[m])}
# par circuit
per = []
for j, c in enumerate(cols):
    per.append({"circuit": c, "blend": rmse(B[m, j], T[m, j]), "gru": rmse(gru[m, j], T[m, j]), "gbm": rmse(pg[m, j], T[m, j]),
                "std_reel": float(np.std(T[m, j])), "part_mse_pct": 100 * np.sum((B[m, j] - T[m, j]) ** 2) / np.sum((B[m] - T[m]) ** 2)})
per.sort(key=lambda r: -r["blend"]); out["par_circuit"] = per[:8]
# derive par semaine (semaines depuis le debut du reve)
h = np.arange(len(T)) - N; wk = h // 168
weeks = sorted(set(wk[m])); by_week = [(int(w), rmse(B[m & (wk == w)], T[m & (wk == w)])) for w in weeks if w >= 0]
r = np.array([x[1] for x in by_week]); out["derive"] = {"semaines_1_4": float(r[:4].mean()), "semaines_5_50": float(r[4:50].mean()), "semaines_51_98": float(r[50:].mean()),
                                                      "pire_semaine": int(np.argmax(r)) + 1, "pire_rmse": float(r.max()), "n_semaines": len(r)}
# incertitude : ecart-type de l'ensemble contre erreur absolue
err = np.abs(B[m] - T[m]); s = std[m]
out["incertitude"] = {"spearman": float(spearmanr(s.ravel(), err.ravel()).correlation), "rmse_sur_std_moyen": float(np.sqrt(np.mean(err ** 2)) / s.mean()),
                      "std_moyen": float(s.mean()), "rmse": float(np.sqrt(np.mean(err ** 2)))}
# correlation des erreurs GRU / GBM
e1 = (gru[m] - T[m]).ravel(); e2 = (pg[m] - T[m]).ravel(); out["corr_erreurs_gru_gbm"] = float(np.corrcoef(e1, e2)[0, 1])
Path("reports/tableaux_rapport.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "par_circuit"}, indent=1))
print(pd.DataFrame(out["par_circuit"]).round(3).to_string(index=False))
