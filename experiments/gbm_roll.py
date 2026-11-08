"""GBM seul : toutes features sauf calendrier, avec / sans le groupe roll (statistiques glissantes)."""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period, feature_groups
from dio.gbm import GBM_PARAMS
from dio.metrics import rmse
from experiments.gbm_blend import fit_predict
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
G = feature_groups(X1.columns)
nocal = [c for c in X1.columns if not re.search("lt_earthmars|lt_sunmarsearth", c)]
sets = {"tout_sans_roll": [c for c in nocal if c not in G["roll"]], "tout_avec_roll": nocal}
k = len(X2) // 2
protos = {"strict": (X1, Y1, X2, Y2), "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:])}
res = {}
for name, cols in sets.items():
    res[name] = {}
    for proto, (Xtr, Ytr, Xte, Yte) in protos.items():
        t0 = time.time()
        P = fit_predict(Xtr[cols].to_numpy(), Ytr.to_numpy(), Xte[cols].to_numpy()[N:], GBM_PARAMS)
        res[name][proto] = round(rmse(P, Yte.to_numpy()[N:]), 5)
        pd.DataFrame(P, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/gbm_{name}_{proto}.pred.csv")
        print(json.dumps({name: {proto: res[name][proto], "n_feat": len(cols), "t_s": round(time.time() - t0)}}), flush=True)
Path("reports/gbm_roll.json").write_text(json.dumps(res, indent=1))
print("GBM ROLL DONE")
