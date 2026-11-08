"""Troisieme famille : forets (RandomForest / ExtraTrees, multi-sorties) sur toutes les features sauf calendrier.
Sauvegarde les predictions des deux protocoles pour l'etude de melange."""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
cols = [c for c in X1.columns if not re.search("lt_earthmars|lt_sunmarsearth", c)]
k = len(X2) // 2
protos = {"strict": (X1, Y1, X2, Y2), "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:])}
models = {
    "et": lambda: ExtraTreesRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=10, n_jobs=4, random_state=0),
    "rf": lambda: RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=10, n_jobs=4, random_state=0),
}
res = {}
for name, mk in models.items():
    res[name] = {}
    for proto, (Xtr, Ytr, Xte, Yte) in protos.items():
        t0 = time.time()
        Ytr_np = Ytr.to_numpy(); m = ~np.isnan(Ytr_np).any(axis=1)
        model = mk().fit(Xtr[cols].to_numpy()[m], Ytr_np[m])
        P = np.clip(model.predict(Xte[cols].to_numpy()[N:]), 0, None)
        res[name][proto] = round(rmse(P, Yte.to_numpy()[N:]), 5)
        pd.DataFrame(P, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/{name}_{proto}.pred.csv")
        print(json.dumps({name: {proto: res[name][proto], "t_s": round(time.time() - t0)}}), flush=True)
Path("reports/forests.json").write_text(json.dumps(res, indent=1))
print("FORESTS DONE")
