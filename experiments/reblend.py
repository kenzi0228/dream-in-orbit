"""Change le poids du composant GBM d'un dossier de poids sans ré-entraîner (le mélange est fait à la
prédiction) et recalcule val_pred.csv / val_report.json si une validation existe.

    python experiments/reblend.py --weights weights_holdout --gbm-weight 0.45 [--split 0.5]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse, rmse_by_horizon, rmse_per_circuit
N = 168
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--gbm-weight", type=float, required=True)
ap.add_argument("--split", type=float, default=0.0, help="fraction de annee2 dans le train si validation chronologique")
a = ap.parse_args()
d = Path(a.weights); meta = json.loads((d / "meta.json").read_text())
old = meta.get("gbm_weight", 0.0); meta["gbm_weight"] = a.gbm_weight; meta["args"]["gbm_weight"] = a.gbm_weight
(d / "meta.json").write_text(json.dumps(meta))
print(f"{d}: gbm_weight {old} -> {a.gbm_weight}")
if (d / "val_pred.csv").exists() and old > 0:
    kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
    X2, Y2 = load_period(kit / "annee2", v)
    k = int(len(X2) * a.split) if a.split > 0 else 0
    Xte, Yte = X2.iloc[k:], Y2.iloc[k:]
    blend = pd.read_csv(d / "val_pred.csv", index_col="ut_ms").reindex(Yte.index).to_numpy()
    g = GBMComponent.load(d / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
    gru = (blend - old * pg) / (1 - old)
    new = np.clip((1 - a.gbm_weight) * gru + a.gbm_weight * pg, 0, None)
    T = Yte.to_numpy(); targets = list(Yte.columns)
    rep = json.loads((d / "val_report.json").read_text())
    rep.update({"val_rmse_gru": rmse(gru[N:], T[N:]), "val_rmse_gbm": rmse(pg[N:], T[N:]), "val_rmse": rmse(new[N:], T[N:]),
                "val_rmse_per_circuit": dict(zip(targets, rmse_per_circuit(new[N:], T[N:]).round(5).tolist())),
                "val_rmse_by_week": rmse_by_horizon(new[N:], T[N:]), "gbm_weight": a.gbm_weight})
    (d / "val_report.json").write_text(json.dumps(rep, indent=1))
    pd.DataFrame(new, index=Yte.index, columns=targets).to_csv(d / "val_pred.csv")
    print(json.dumps({k: round(v, 5) for k, v in rep.items() if k.startswith("val_rmse") and isinstance(v, float)}))
