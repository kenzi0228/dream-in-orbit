"""Re-melange hors ligne : pour chaque jeu de predictions GBM sauvegarde (reports/gbm_{tag}_{set}_{proto}.pred.csv),
RMSE du GBM seul, du melange a poids fixe, poids optimal par protocole et poids croise (choisi sur l'autre protocole).
Les predictions du GRU sont reconstruites une fois et mises en cache (reports/gru_{proto}.pred.csv).

    python experiments/reblend_sets.py --tag lettres [--sets a,b] [--weights 0.3,0.4,...]
"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.metrics import rmse
N = 168
WD = {"strict": "weights_holdout", "split": "weights_holdout_split"}


def truth_and_gru(proto):
    f = Path(f"reports/gru_{proto}.pred.csv"); t = Path(f"reports/truth_{proto}.csv")
    if f.exists() and t.exists():
        return pd.read_csv(t, index_col="ut_ms"), pd.read_csv(f, index_col="ut_ms")
    from dio.data import get_vocab, load_period
    from dio.gbm import GBMComponent
    kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
    X2, Y2 = load_period(kit / "annee2", v); k = len(X2) // 2
    Xte, Yte = (X2, Y2) if proto == "strict" else (X2.iloc[k:], Y2.iloc[k:])
    wdir = WD[proto]; meta = json.load(open(f"{wdir}/meta.json")); w = meta["gbm_weight"]
    blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms")
    g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
    gru = ((blend - w * pg) / (1 - w)).iloc[N:]; T = Yte.iloc[N:]
    gru.to_csv(f); T.to_csv(t)
    return T, gru


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--sets", default="")
    ap.add_argument("--weights", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    a = ap.parse_args(); ws = [float(x) for x in a.weights.split(",")]
    data = {p: truth_and_gru(p) for p in WD}
    sets = a.sets.split(",") if a.sets else sorted({re.sub(r"_(strict|split)\.pred\.csv$", "", Path(f).name[len(f"gbm_{a.tag}_"):]) for f in glob.glob(f"reports/gbm_{a.tag}_*_*.pred.csv")})
    rows = []
    for s in sets:
        res = {}
        for p, (T, gru) in data.items():
            f = Path(f"reports/gbm_{a.tag}_{s}_{p}.pred.csv")
            if not f.exists(): continue
            P = pd.read_csv(f, index_col="ut_ms").reindex(T.index).to_numpy(); Tn = T.to_numpy(); G = gru.to_numpy()
            m = ~np.isnan(Tn).any(axis=1)
            res[p] = {"gbm": rmse(P[m], Tn[m]), **{w: rmse((1 - w) * G[m] + w * P[m], Tn[m]) for w in ws}}
        if not res: continue
        row = {"set": s}
        for p in res:
            best = min(ws, key=lambda w: res[p][w]); row[f"{p}_gbm"] = round(res[p]["gbm"], 5); row[f"{p}_b45"] = round(res[p][0.45], 5)
            row[f"{p}_wopt"] = best; row[f"{p}_bopt"] = round(res[p][best], 5)
        if len(res) == 2:
            row["strict@w_split"] = round(res["strict"][row["split_wopt"]], 5); row["split@w_strict"] = round(res["split"][row["strict_wopt"]], 5)
            row["moy_b45"] = round((res["strict"][0.45] + res["split"][0.45]) / 2, 5)
            wm = min(ws, key=lambda w: res["strict"][w] + res["split"][w]); row["w_commun"] = wm; row["moy_bcommun"] = round((res["strict"][wm] + res["split"][wm]) / 2, 5)
        rows.append(row)
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    Path(f"reports/reblend_{a.tag}.json").write_text(df.to_json(orient="records", indent=1))


if __name__ == "__main__":
    main()
