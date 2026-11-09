"""Poids de melange GRU / GBM : RMSE par poids sur les deux protocoles a partir des dossiers de validation
(val_pred.csv = melange au poids de meta.json, gbm.pkl pour reconstruire les deux composantes)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"]); X2, Y2 = load_period(kit / "annee2", v); k = len(X2) // 2
res = {}
for proto, wdir, Xte, Yte in [("strict", "weights_holdout", X2, Y2), ("split", "weights_holdout_split", X2.iloc[k:], Y2.iloc[k:])]:
    meta = json.load(open(f"{wdir}/meta.json")); w0 = meta["gbm_weight"]
    blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms").to_numpy()
    g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
    gru = (blend - w0 * pg) / (1 - w0); T = Yte.to_numpy(); m = ~np.isnan(T).any(axis=1); m[:N] = False
    res[proto] = {w: rmse(np.clip((1 - w) * gru[m] + w * pg[m], 0, None), T[m]) for w in np.round(np.arange(0, 1.01, 0.05), 2)}
    print(proto, "gru", round(res[proto][0.0], 5), "gbm", round(res[proto][1.0], 5))
rows = [(w, res["strict"][w], res["split"][w], (res["strict"][w] + res["split"][w]) / 2) for w in res["strict"]]
df = pd.DataFrame(rows, columns=["poids_gbm", "strict", "split", "moyenne"]); print(df.round(5).to_string(index=False))
Path("reports/poids_commun.json").write_text(df.to_json(orient="records", indent=1))
