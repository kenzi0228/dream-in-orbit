"""Tests hors ligne sur les predictions sauvegardees (GRU reconstruit, GBM de plusieurs vues) :
moyenne de vues GBM, poids dependant de l'horizon, calibration par les 7 jours, lissage."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.gbm import GBMComponent
from dio.metrics import rmse
N = 168
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X2, Y2 = load_period(kit / "annee2", v); k = len(X2) // 2
D = {}
for proto, Xte, Yte, wdir in [("strict", X2, Y2, "weights_holdout"), ("split", X2.iloc[k:], Y2.iloc[k:], "weights_holdout_split")]:
    meta = json.load(open(f"{wdir}/meta.json")); w = meta["gbm_weight"]
    blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms").to_numpy()
    g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
    gru = (blend - w * pg) / (1 - w)
    idx = Yte.index[N:]
    views = {}
    for name in ["gbm_tout_sans_roll", "gbm_tout_avec_roll", "gbm_phys+dmop_tout", "gbm_phys+flux+hist", "gbm_phys+dmop_cmd", "gbm_tout_sans_calendrier", "gbm_phys"]:
        f = Path(f"reports/{name}_{proto}.pred.csv")
        if f.exists():
            views[name] = pd.read_csv(f, index_col="ut_ms").reindex(idx).to_numpy()
    D[proto] = {"T": Yte.to_numpy(), "gru": gru, "gbm": pg, "views": views}

def ev(proto, pred_after):  # pred_after : predictions a partir de N
    return round(rmse(pred_after, D[proto]["T"][N:]), 5)

print("=== reference 0,55/0,45")
for p in D:
    d = D[p]; print(p, ev(p, (0.55 * d["gru"] + 0.45 * d["gbm"])[N:]))

print("=== moyenne de plusieurs vues GBM (au lieu du seul GBM 'tout')")
for p in D:
    d = D[p]; vs = d["views"]
    for combo in [["gbm_tout_sans_roll", "gbm_tout_avec_roll"], ["gbm_tout_sans_roll", "gbm_phys+dmop_tout"], ["gbm_tout_sans_roll", "gbm_phys+flux+hist"],
                  ["gbm_tout_sans_roll", "gbm_phys+dmop_tout", "gbm_phys+flux+hist"], ["gbm_tout_sans_roll", "gbm_tout_sans_calendrier", "gbm_phys+dmop_cmd", "gbm_phys+flux+hist"]]:
        if all(c in vs for c in combo):
            m = np.mean([vs[c] for c in combo], axis=0)
            print(p, "+".join(c.replace("gbm_", "") for c in combo), "gbm_moy", ev(p, m), "blend", ev(p, 0.55 * d["gru"][N:] + 0.45 * m))

print("=== poids dependant de l'horizon : w_gru = a pendant les h premieres heures, b ensuite")
for p in D:
    d = D[p]; T = len(d["gru"]) - N
    best = []
    for h in [24 * 7, 24 * 14, 24 * 28, 24 * 56]:
        for a in [0.55, 0.65, 0.75, 0.85]:
            for b in [0.45, 0.5, 0.55]:
                wg = np.where(np.arange(T) < h, a, b)[:, None]
                pr = wg * d["gru"][N:] + (1 - wg) * d["gbm"][N:]
                best.append((ev(p, pr), h // 24, a, b))
    best.sort(); print(p, best[:4])

print("=== calibration par les 7 jours (residu moyen par circuit sur l'init ajoute a la suite)")
for p in D:
    d = D[p]; bl = 0.55 * d["gru"] + 0.45 * d["gbm"]
    res = np.nan_to_num(np.nanmean(d["T"][:N] - bl[:N], axis=0))
    print(p, {a: ev(p, bl[N:] + a * res) for a in [0.0, 0.25, 0.5]}, " (gbm seul calibre :", {a: ev(p, d["gbm"][N:] + a * np.nan_to_num(np.nanmean(d["T"][:N] - d["gbm"][:N], axis=0))) for a in [0.25, 0.5]}, ")")

print("=== lissage du GBM (moyenne mobile centree 3 h) avant melange")
for p in D:
    d = D[p]
    sm = pd.DataFrame(d["gbm"]).rolling(3, center=True, min_periods=1).mean().to_numpy()
    print(p, "gbm lisse", ev(p, sm[N:]), "blend", ev(p, (0.55 * d["gru"] + 0.45 * sm)[N:]))

print("=== clip aux bornes du train par circuit")
X1, Y1 = load_period(kit / "annee1", v)
lo, hi = np.nanmin(Y1.to_numpy(), axis=0), np.nanmax(Y1.to_numpy(), axis=0)
for p in D:
    d = D[p]; bl = 0.55 * d["gru"] + 0.45 * d["gbm"]
    print(p, "sans clip", ev(p, bl[N:]), "clip", ev(p, np.clip(bl[N:], lo, hi * 1.05)))
