"""Ponderation des donnees recentes (protocole chrono seulement) : poids w0 sur annee1, 1 sur annee2."""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period
from dio.gbm import GBM_PARAMS, GBMComponent
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
G = feature_groups(X1.columns)
cols = [c for c in X1.columns if not re.search("lt_earthmars|lt_sunmarsearth", c) and c not in G["roll"] and c not in G["ewm"]]
k = len(X2) // 2
Xtr = pd.concat([X1, X2.iloc[:k]])[cols].to_numpy(); Ytr = pd.concat([Y1, Y2.iloc[:k]]).to_numpy()
Xte = X2.iloc[k:][cols].to_numpy()[N:]; T = Y2.iloc[k:].to_numpy()[N:]
meta = json.load(open("weights_holdout_split/meta.json")); w = meta["gbm_weight"]
blend = pd.read_csv("weights_holdout_split/val_pred.csv", index_col="ut_ms").to_numpy()
g0 = GBMComponent.load(Path("weights_holdout_split/gbm.pkl")); pg0 = g0.predict(X2.iloc[k:][g0.features].to_numpy())
gru = ((blend - w * pg0) / (1 - w))[N:]
mask = ~np.isnan(Ytr).any(axis=1)
res = {}
for w0 in [1.0, 0.5, 0.25]:
    t0 = time.time(); sw = np.where(np.arange(len(Xtr)) < len(X1), w0, 1.0)
    P = np.zeros_like(T)
    for j in range(T.shape[1]):
        m = HistGradientBoostingRegressor(random_state=0, **GBM_PARAMS).fit(Xtr[mask], Ytr[mask, j], sample_weight=sw[mask])
        P[:, j] = m.predict(Xte)
    P = np.clip(P, 0, None)
    res[f"w_annee1={w0}"] = {"gbm": round(rmse(P, T), 5), "blend": round(rmse(0.55 * gru + 0.45 * P, T), 5), "t_s": round(time.time() - t0)}
    print(json.dumps({f"w_annee1={w0}": res[f"w_annee1={w0}"]}), flush=True)
Path("reports/gbm_recency.json").write_text(json.dumps(res, indent=1)); print("GBM RECENCY DONE")
