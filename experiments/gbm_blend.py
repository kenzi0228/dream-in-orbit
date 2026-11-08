"""
Etude de melange : un gradient boosting (sklearn) entraine sur differents jeux de features,
melange avec les predictions du world model GRU, dans les deux protocoles.

    python experiments/gbm_blend.py --gru-strict weights_holdout/val_pred.csv --gru-split weights_holdout_split_phys/val_pred.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period  # noqa: E402
from dio.metrics import rmse  # noqa: E402

N = 168


def fit_predict(Xtr, Ytr, Xte, params):
    mtr = ~np.isnan(Ytr).any(axis=1)
    P = np.zeros((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(random_state=0, **params)
        m.fit(Xtr[mtr], Ytr[mtr, j])
        P[:, j] = m.predict(Xte)
    return np.clip(P, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-strict", default="weights_holdout/val_pred.csv")
    ap.add_argument("--gru-split", default="weights_holdout_split_phys/val_pred.csv")
    ap.add_argument("--out", default="reports/gbm_blend.json")
    a = ap.parse_args()
    kit = Path("kit_etudiants")
    v = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", v)
    X2, Y2 = load_period(kit / "annee2", v)
    G = feature_groups(X1.columns)
    nocal = lambda cols: [c for c in cols if not re.search("lt_earthmars|lt_sunmarsearth", c)]
    phys = nocal(G["saaf"] + G["ltdata"] + G["evtf"] + G["ftl"])
    sets = {
        "tout": list(X1.columns),
        "tout_sans_calendrier": nocal(list(X1.columns)),
        "phys": phys,
        "phys+dmop_cmd": phys + G["dmop_cmd"],
        "phys+dmop_base+switch": phys + G["dmop_base"] + G["dmop_switch"],
        "phys+dmop_tout": phys + G["dmop_base"] + G["dmop_switch"] + G["dmop_cmd"] + G["dmop_prefix_rec"] + G["dmop_stem_rec"],
        "phys+flux+hist": phys + G["flux"] + G["hist"],
    }
    params = {"max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 50}
    k = len(X2) // 2
    protos = {
        "strict": (X1, Y1, X2, Y2, pd.read_csv(a.gru_strict, index_col="ut_ms")),
        "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:], pd.read_csv(a.gru_split, index_col="ut_ms")),
    }
    res = {}
    for name, cols in sets.items():
        res[name] = {}
        for proto, (Xtr, Ytr, Xte, Yte, gru) in protos.items():
            t0 = time.time()
            P = fit_predict(Xtr[cols].to_numpy(), Ytr.to_numpy(), Xte[cols].to_numpy()[N:], params)
            T = Yte.to_numpy()[N:]
            Gp = gru.reindex(Yte.index[N:]).to_numpy()
            r = {"gbm": rmse(P, T), "gru": rmse(Gp, T)}
            for w in [0.9, 0.8, 0.7, 0.6]:
                r[f"blend_{w}"] = rmse(w * Gp + (1 - w) * P, T)
            r["t_s"] = round(time.time() - t0)
            res[name][proto] = {kk: round(vv, 5) for kk, vv in r.items()}
            pd.DataFrame(P, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/gbm_{name}_{proto}.pred.csv")
            print(json.dumps({name: {proto: res[name][proto]}}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("GBM BLEND DONE")


if __name__ == "__main__":
    main()
