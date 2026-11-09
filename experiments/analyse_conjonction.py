"""Decomposition de l'erreur de NPWD2551/2451 par bande de semaines (strict) et test d'intervention sur le GBM :
pendant la conjonction (annee2 sem 64-70), quelles features forcees a leur valeur des semaines 52-58 font retomber la prediction ?"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
blend = pd.read_csv("weights_holdout/val_pred.csv", index_col="ut_ms"); meta = json.load(open("weights_holdout/meta.json")); w = meta["gbm_weight"]
g = GBMComponent.load(Path("weights_holdout/gbm.pkl"))
pg = pd.DataFrame(g.predict(X2.loc[blend.index, g.features].to_numpy()), index=blend.index, columns=blend.columns)
gru = (blend - w * pg) / (1 - w); T = Y2.loc[blend.index]
wk = pd.Series(np.arange(len(blend)) // 168 + 1, index=blend.index)
tot = ((blend - T) ** 2).to_numpy(); tot = np.nansum(tot)
print("part de la SSE totale (melange, strict) par bande de semaines et circuit :")
bands = {"sem 1-10": (1, 10), "sem 11-34 (eclipses 1)": (11, 34), "sem 35-50": (35, 50), "sem 51-62 (eclipses 2 avant conj.)": (51, 62), "sem 63-70 (conjonction)": (63, 70), "sem 71-91": (71, 91), "sem 92-99": (92, 99)}
rows = []
for name, (a, b) in bands.items():
    m = (wk >= a) & (wk <= b)
    e = (blend[m] - T[m]) ** 2
    rows.append((name, int(m.sum()), 100 * np.nansum(e.to_numpy()) / tot, 100 * np.nansum(e["NPWD2551"]) / tot, 100 * np.nansum(e["NPWD2451"]) / tot,
                 float(np.nanmean(blend[m]["NPWD2551"] - T[m]["NPWD2551"])), float(np.nanmean(gru[m]["NPWD2551"] - T[m]["NPWD2551"])), float(np.nanmean(pg[m]["NPWD2551"] - T[m]["NPWD2551"]))))
print(pd.DataFrame(rows, columns=["bande", "h", "SSE_%", "SSE_2551_%", "SSE_2451_%", "biais_2551_blend", "biais_2551_gru", "biais_2551_gbm"]).round(3).to_string(index=False))
# si on retirait le biais hebdomadaire de 2551 et 2451 (oracle) :
for c in ["NPWD2551", "NPWD2451"]:
    b = (blend[c] - T[c]).groupby(wk).transform("mean"); corr = blend.copy(); corr[c] = blend[c] - b
    print(f"  oracle biais hebdo {c} : rmse {rmse(corr.to_numpy(), T.to_numpy()):.5f} (vs {rmse(blend.to_numpy(), T.to_numpy()):.5f})")
# intervention : semaines 64-70 ; remplacer un groupe de features par sa moyenne des semaines 52-58, voir la prediction GBM de 2551/2451
G = feature_groups(X2.columns); Xc = X2.loc[blend.index]
m_conj = ((wk >= 64) & (wk <= 70)).to_numpy(); m_ref = ((wk >= 52) & (wk <= 58)).to_numpy()
base = g.predict(Xc[g.features].to_numpy()[m_conj])
j51 = list(blend.columns).index("NPWD2551"); j45 = list(blend.columns).index("NPWD2451")
print(f"\nconjonction sem 64-70 : verite 2551 {np.nanmean(T['NPWD2551'][m_conj]):.3f} 2451 {np.nanmean(T['NPWD2451'][m_conj]):.3f} | GBM 2551 {base[:, j51].mean():.3f} 2451 {base[:, j45].mean():.3f}")
ref_vals = Xc[m_ref].mean()
tests = {grp: cols for grp, cols in G.items() if cols}
tests.update({"ftl_comms": ["ftl_comms"], "ftl_EARTH": ["ftl_EARTH"], "ftl_WARMUP": ["ftl_WARMUP"], "ftl_D4PNPO": ["ftl_D4PNPO"], "ftl_SLEW": ["ftl_SLEW"],
              "dm_total+dm_p": [c for c in Xc.columns if c.startswith("dm_p_") or c == "dm_total"], "dm_r (recence cmd)": [c for c in Xc.columns if c.startswith("dm_r_")],
              "dm_st (recence stem)": [c for c in Xc.columns if c.startswith("dm_st_")], "dm_pr (recence prefixe)": [c for c in Xc.columns if c.startswith("dm_pr_")],
              "hist": G["hist"], "evtf": G["evtf"]})
rows = []
for name, cols in tests.items():
    cols = [c for c in cols if c in g.features]
    if not cols: continue
    Xi = Xc[m_conj].copy(); Xi[cols] = ref_vals[cols].to_numpy()
    p = g.predict(Xi[g.features].to_numpy())
    rows.append((name, len(cols), p[:, j51].mean(), p[:, j45].mean(), np.sqrt(np.nanmean((p - T[m_conj].to_numpy()) ** 2))))
print("\nintervention (groupe force a sa valeur moyenne des sem 52-58) -> GBM moyen 2551 / 2451 sur sem 64-70, rmse toutes voies :")
print(pd.DataFrame(rows, columns=["groupe", "n", "gbm_2551", "gbm_2451", "rmse_33"]).sort_values("gbm_2551").round(3).to_string(index=False))
print("rmse_33 de base sur sem 64-70 :", round(np.sqrt(np.nanmean((base - T[m_conj].to_numpy()) ** 2)), 4))
