"""Trous dans les fichiers de planification, saison d'eclipses, comportement de NPWD2551 par semaine (annee1 vs annee2)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
H = 3600000
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
for name in ["annee1", "annee2"]:
    print(f"\n=== trous (> 24 h) dans {name}")
    for f in ["dmop", "ftl", "evtf", "saaf", "ltdata", "power"]:
        df = pd.read_csv(kit / name / f"{f}.csv", usecols=["ut_ms"] if f != "ftl" else None)
        t = np.sort(df["ut_ms"].to_numpy() if "ut_ms" in df else df["utb_ms"].to_numpy())
        t0 = Y1.index[0] if name == "annee1" else Y2.index[0]
        d = np.diff(t); big = np.where(d > 24 * H)[0]
        gaps = [(round((t[i] - t0) / H / 24, 1), round(d[i] / H / 24, 1)) for i in big]
        print(f"  {f}: debut jour {round((t[0]-t0)/H/24,1)} fin jour {round((t[-1]-t0)/H/24,1)} ; trous (jour, duree j) : {gaps[:12]}")
    print("  heures de power manquantes (NaN) :", int(Y1.isna().any(axis=1).sum() if name == "annee1" else Y2.isna().any(axis=1).sum()))
# semaine par semaine : NPWD2551, umbra, dm_total, comms, etats AXXX301/305 ; annee1 et annee2
blend = pd.read_csv("weights_holdout/val_pred.csv", index_col="ut_ms")
meta = json.load(open("weights_holdout/meta.json")); w = meta["gbm_weight"]
from dio.gbm import GBMComponent
g = GBMComponent.load(Path("weights_holdout/gbm.pkl")); pg = pd.DataFrame(g.predict(X2.loc[blend.index, g.features].to_numpy()), index=blend.index, columns=blend.columns)
gru = (blend - w * pg) / (1 - w)
sw_cols = [c for c in X1.columns if c.startswith("dm_s_AXXX30")]
print("\ninterrupteurs AXXX30x :", sw_cols)
for name, X, Y in [("annee1", X1, Y1), ("annee2", X2, Y2)]:
    wk = np.arange(len(X)) // 168
    d = pd.DataFrame({"sem": wk + 1, "y2551": Y["NPWD2551"].values, "y2451": Y["NPWD2451"].values, "umbra": X["ev_umbra"].values, "dm": X["dm_total"].values,
                      "comms": X["ftl_comms"].values, **{c[5:]: X[c].values for c in sw_cols}})
    if name == "annee2":
        d["gru2551"] = gru["NPWD2551"].reindex(X.index).values; d["gbm2551"] = pg["NPWD2551"].reindex(X.index).values
        d["gru2451"] = gru["NPWD2451"].reindex(X.index).values; d["gbm2451"] = pg["NPWD2451"].reindex(X.index).values
    agg = d.groupby("sem").mean()
    print(f"\n=== {name} semaines 50-75 et 92-99")
    print(agg.loc[list(range(50, 76)) + list(range(92, 99))].round(3).to_string())
