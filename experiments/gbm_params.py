"""
Reglage du composant gradient boosting (features : toutes sauf calendrier), seul et en melange
avec le GRU, dans les deux protocoles.

    python experiments/gbm_params.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period  # noqa: E402
from dio.metrics import rmse  # noqa: E402
from experiments.gbm_blend import fit_predict  # noqa: E402

N = 168
PARAMS = {
    "p200_lr06_l15_m50": {"max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 50},
    "p400_lr03_l15_m50": {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 50},
    "p300_lr05_l31_m100": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0, "min_samples_leaf": 100},
    "p300_lr05_l7_m50": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0, "min_samples_leaf": 50},
    "p500_lr03_l15_m200_l2x5": {"max_iter": 500, "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 5.0, "min_samples_leaf": 200},
}


def main():
    kit = Path("kit_etudiants")
    v = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", v)
    X2, Y2 = load_period(kit / "annee2", v)
    cols = [c for c in X1.columns if not re.search("lt_earthmars|lt_sunmarsearth", c)]
    k = len(X2) // 2
    protos = {
        "strict": (X1, Y1, X2, Y2, pd.read_csv("weights_holdout/val_pred.csv", index_col="ut_ms")),
        "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:],
                  pd.read_csv("weights_holdout_split_phys/val_pred.csv", index_col="ut_ms")),
    }
    res = {}
    for name, params in PARAMS.items():
        res[name] = {}
        for proto, (Xtr, Ytr, Xte, Yte, gru) in protos.items():
            t0 = time.time()
            P = fit_predict(Xtr[cols].to_numpy(), Ytr.to_numpy(), Xte[cols].to_numpy()[N:], params)
            T = Yte.to_numpy()[N:]
            Gp = gru.reindex(Yte.index[N:]).to_numpy()
            r = {"gbm": rmse(P, T)}
            for w in [0.5, 0.6, 0.65, 0.7]:
                r[f"blend_{w}"] = rmse(w * Gp + (1 - w) * P, T)
            r["t_s"] = round(time.time() - t0)
            res[name][proto] = {kk: round(vv, 5) for kk, vv in r.items()}
            print(json.dumps({name: {proto: res[name][proto]}}), flush=True)
        Path("reports/gbm_params.json").write_text(json.dumps(res, indent=1))
    print("GBM PARAMS DONE")


if __name__ == "__main__":
    main()
