"""Bagging temporel du GBM : K modeles, chacun entraine sans un bloc contigu de 1/K du train ; moyenne.
Compare GBM seul et melange 0,55/0,45 avec le GRU (reconstruit), deux protocoles."""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period
from dio.gbm import GBM_PARAMS, GBMComponent
from dio.metrics import rmse
N = 168; K = 5
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
G = feature_groups(X1.columns)
from experiments.gbm_sets import resolve
SET = sys.argv[1] if len(sys.argv) > 1 else "all"   # ex. "all-dmop_cmd-dmop_switch+dmop_sw+dmop_stem_cnt"
cols = resolve(SET, X1.columns, G)
k = len(X2) // 2
protos = {"strict": (X1, Y1, X2, Y2, "weights_holdout"), "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:], "weights_holdout_split")}
res = {}
for proto, (Xtr, Ytr, Xte, Yte, wdir) in protos.items():
    meta = json.load(open(f"{wdir}/meta.json")); w = meta["gbm_weight"]
    blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms").to_numpy()
    g0 = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg0 = g0.predict(Xte[g0.features].to_numpy())
    gru = ((blend - w * pg0) / (1 - w))[N:]
    Xa, Ya, Xt, T = Xtr[cols].to_numpy(), Ytr.to_numpy(), Xte[cols].to_numpy()[N:], Yte.to_numpy()[N:]
    t0 = time.time(); bounds = np.linspace(0, len(Xa), K + 1).astype(int); preds = []
    for i in range(K):
        tr = np.r_[0:bounds[i], bounds[i + 1]:len(Xa)]
        preds.append(GBMComponent(cols).fit(Xa[tr], Ya[tr]).predict(Xt))
        print(proto, "sac", i, round(rmse(preds[-1], T), 5), flush=True)
    bag = np.mean(preds, axis=0)
    res[proto] = {"gbm_plein": round(rmse(pg0[N:], T), 5), "gbm_bagging": round(rmse(bag, T), 5),
                  "blend_plein": round(rmse(0.55 * gru + 0.45 * pg0[N:], T), 5), "blend_bagging": round(rmse(0.55 * gru + 0.45 * bag, T), 5),
                  "blend_plein+bagging": round(rmse(0.55 * gru + 0.45 * (0.5 * pg0[N:] + 0.5 * bag), T), 5), "t_s": round(time.time() - t0)}
    pd.DataFrame(bag, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/gbm_bagging_{SET}_{proto}.pred.csv")
    print(json.dumps({proto: res[proto]}), flush=True)
out = Path("reports/gbm_bagging.json"); allres = json.loads(out.read_text()) if out.exists() else {}; allres[SET] = res
out.write_text(json.dumps(allres, indent=1)); print("GBM BAGGING DONE", SET)
