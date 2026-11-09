"""Chiffres de la section Resultats du rapport, pour un dossier de validation (strict par defaut) :
RMSE par circuit du systeme livre, du gradient boosting seul et du GRU seul (dossier d'ablation `GRU seul`
s'il existe), derive par semaine, dispersion de l'incertitude, pire semaine, correlation des erreurs.

    python experiments/tableaux_rapport.py [--weights weights_holdout] [--gru weights_holdout_abl_GRU]
Ecrit reports/tableaux_rapport[_<suffixe>].json."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse

PERIOD = {"strict": "annee2", "reverse": "annee1", "split": "annee2"}
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights_holdout")
ap.add_argument("--gru", default="weights_holdout_abl_GRU", help="dossier de validation du GRU seul (ablation)")
ap.add_argument("--kit", default="kit_etudiants")
a = ap.parse_args()
w = Path(a.weights); meta = json.load(open(w / "meta.json")); N = meta["n_init"]; kit = Path(a.kit)
X, Y = load_period(kit / PERIOD[meta["protocol"]], meta["vocab"])
final = pd.read_csv(w / "val_pred.csv", index_col="ut_ms"); std = pd.read_csv(w / "val_std.csv", index_col="ut_ms")
X, Y = X.reindex(final.index), Y.reindex(final.index)[final.columns]
g = GBMComponent.load(w / "gbm.pkl"); pg = g.predict(X[g.features].to_numpy())
gru = None
if Path(a.gru, "val_pred.csv").exists():
    gru = pd.read_csv(Path(a.gru) / "val_pred.csv", index_col="ut_ms").reindex(final.index).to_numpy()
elif meta.get("combine") == "blend":
    w0 = meta["gbm_weight"]; gru = (final.to_numpy() - w0 * pg) / (1 - w0)
B, T, S = final.to_numpy(), Y.to_numpy(), std.to_numpy(); m = ~np.isnan(T).any(axis=1); m[:N] = False
cols = list(final.columns)
out = {"weights": a.weights, "protocol": meta["protocol"], "combine": meta.get("combine"), "rmse_final": rmse(B[m], T[m]), "rmse_gbm": rmse(pg[m], T[m]),
       "rmse_gru_seul": None if gru is None else rmse(gru[m], T[m])}
per = []
for j, c in enumerate(cols):
    per.append({"circuit": c, "final": rmse(B[m, j], T[m, j]), "gru_seul": None if gru is None else rmse(gru[m, j], T[m, j]), "gbm": rmse(pg[m, j], T[m, j]),
                "std_reel": float(np.std(T[m, j])), "part_mse_pct": 100 * np.sum((B[m, j] - T[m, j]) ** 2) / np.sum((B[m] - T[m]) ** 2)})
per.sort(key=lambda r: -r["final"]); out["par_circuit"] = per[:8]
wk = (np.arange(len(T)) - N) // 168
by_week = [rmse(B[m & (wk == k)], T[m & (wk == k)]) for k in sorted(set(wk[m])) if k >= 0]
r = np.array(by_week)
out["derive"] = {"semaines_1_4": float(r[:4].mean()), "semaines_5_50": float(r[4:50].mean()), "semaines_51_fin": float(r[50:].mean()) if len(r) > 50 else None,
                 "pire_semaine": int(np.argmax(r)) + 1, "pire_rmse": float(r.max()), "n_semaines": len(r)}
err = np.abs(B[m] - T[m]); s = S[m]
out["incertitude_brute"] = {"spearman": float(pd.Series(s.ravel()).corr(pd.Series(err.ravel()), method="spearman")),
                            "rmse_sur_rms_std": float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(s ** 2)))}
if gru is not None:
    out["corr_erreurs_gru_gbm"] = float(np.corrcoef((gru[m] - T[m]).ravel(), (pg[m] - T[m]).ravel())[0, 1])
suffix = "" if a.weights == "weights_holdout" else "_" + meta["protocol"]
Path(f"reports/tableaux_rapport{suffix}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "par_circuit"}, indent=1))
print(pd.DataFrame(out["par_circuit"]).round(3).to_string(index=False))
