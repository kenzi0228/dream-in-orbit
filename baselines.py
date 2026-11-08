"""
Baselines classiques, evaluees exactement comme le world model (train annee1, test annee2
a partir du 8e jour) :
    - constante : moyenne de chaque circuit sur le train
    - constante + niveau initial : moyenne des 7 premiers jours de la periode de test
    - gradient boosting heure par heure (HistGradientBoostingRegressor, un modele par circuit)
      sur les features horaires, sans memoire du passe.

    python baselines.py --kit kit_etudiants --out reports/baselines.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dio.data import get_vocab, load_period
from dio.metrics import rmse, rmse_per_circuit

N_INIT = 24 * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--out", default="reports/baselines.json")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--split", type=float, default=0.0, help="validation chronologique : train = annee1 + fraction de annee2")
    a = ap.parse_args()
    kit = Path(a.kit)
    vocab = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", vocab)
    X2, Y2 = load_period(kit / "annee2", vocab)
    targets = list(Y1.columns)
    if a.split > 0:
        k = int(len(X2) * a.split)
        X1, Y1 = pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]])
        X2, Y2 = X2.iloc[k:], Y2.iloc[k:]
    Ytr, Yte = Y1.to_numpy(), Y2.to_numpy()
    Yte_eval = Yte[N_INIT:]
    res = {}

    const = np.tile(np.nanmean(Ytr, axis=0), (len(Yte_eval), 1))
    res["constante_train"] = rmse(const, Yte_eval)
    const_init = np.tile(np.nanmean(Yte[:N_INIT], axis=0), (len(Yte_eval), 1))
    res["constante_7jours"] = rmse(const_init, Yte_eval)
    persist = np.tile(pd.DataFrame(Yte[:N_INIT]).ffill().to_numpy()[-1], (len(Yte_eval), 1))
    res["persistance_derniere_heure"] = rmse(persist, Yte_eval)
    print(json.dumps(res, indent=1), flush=True)

    t0 = time.time()
    mtr = ~np.isnan(Ytr).any(axis=1)
    Xtr, Xte = X1.to_numpy(), X2.to_numpy()
    pred = np.zeros_like(Yte_eval)
    for j, c in enumerate(targets):
        m = HistGradientBoostingRegressor(max_iter=a.max_iter, learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=50, l2_regularization=1.0, random_state=0)
        m.fit(Xtr[mtr], Ytr[mtr, j])
        pred[:, j] = m.predict(Xte[N_INIT:])
        print(f"  {c}: rmse {np.sqrt(np.nanmean((pred[:, j] - Yte_eval[:, j]) ** 2)):.5f}", flush=True)
    pred = np.clip(pred, 0, None)
    res["gradient_boosting"] = rmse(pred, Yte_eval)
    res["gradient_boosting_per_circuit"] = dict(zip(targets, rmse_per_circuit(pred, Yte_eval).round(5).tolist()))
    res["gradient_boosting_time_s"] = round(time.time() - t0)
    # variante : correction de niveau a partir des 7 premiers jours (residu moyen)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    pd.DataFrame(pred, index=X2.index[N_INIT:], columns=targets).to_csv(Path(a.out).with_suffix(".pred.csv"))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
