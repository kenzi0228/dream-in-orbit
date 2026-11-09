"""
GBM seul et en melange 0,55 / 0,45 avec le GRU (reconstruit), pour des jeux de features definis
par groupes : "all-dmop_cmd-leadlag" = toutes les colonnes (sans calendrier ni roll) moins ces groupes,
"all+roll" = en ajoutant un groupe.

    python experiments/gbm_sets.py --sets all,all-dmop_cmd,all-leadlag --tag groupes
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period
from dio.gbm import GBM_PARAMS, GBMComponent
from dio.metrics import rmse
from experiments.gbm_blend import fit_predict
N = 168
CAL = "lt_earthmars|lt_sunmarsearth"


def resolve(spec, columns, G):
    excl = set(G["roll"]) | set(G["ewm"]) | set(G["dmop_sw"]) | set(G["dmop_stem_cnt"]) | set(G["dmop_stem_next"])
    cols = [c for c in columns if not re.search(CAL, c) and c not in excl]
    for tok in re.findall(r"[+-][a-z_]+", spec[3:]):
        grp = G[tok[1:]]
        cols = [c for c in cols if c not in grp] if tok[0] == "-" else cols + [c for c in grp if c not in cols]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--tag", default="sets")
    ap.add_argument("--params", default="", help="json : surcharge des hyperparametres du GBM")
    a = ap.parse_args()
    params = dict(GBM_PARAMS, **(json.loads(a.params) if a.params else {}))
    kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
    G = feature_groups(X1.columns); k = len(X2) // 2
    protos = {"strict": (X1, Y1, X2, Y2, "weights_holdout"), "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:], "weights_holdout_split")}
    gru = {}
    for proto, (Xtr, Ytr, Xte, Yte, wdir) in protos.items():
        meta = json.load(open(f"{wdir}/meta.json")); w = meta["gbm_weight"]
        blend = pd.read_csv(f"{wdir}/val_pred.csv", index_col="ut_ms").to_numpy()
        g = GBMComponent.load(Path(wdir) / "gbm.pkl"); pg = g.predict(Xte[g.features].to_numpy())
        gru[proto] = ((blend - w * pg) / (1 - w))[N:]
    out = Path(f"reports/gbm_{a.tag}.json"); res = json.loads(out.read_text()) if out.exists() else {}
    for spec in a.sets.split(","):
        cols = resolve(spec, X1.columns, G); res[spec] = {"n_feat": len(cols)}
        for proto, (Xtr, Ytr, Xte, Yte, _) in protos.items():
            t0 = time.time()
            P = fit_predict(Xtr[cols].to_numpy(), Ytr.to_numpy(), Xte[cols].to_numpy()[N:], params)
            T = Yte.to_numpy()[N:]
            res[spec][proto] = {"gbm": round(rmse(P, T), 5), "blend": round(rmse(0.55 * gru[proto] + 0.45 * P, T), 5), "t_s": round(time.time() - t0)}
            pd.DataFrame(P, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/gbm_{a.tag}_{spec}_{proto}.pred.csv")
            print(json.dumps({spec: {proto: res[spec][proto]}}), flush=True)
        out.write_text(json.dumps(res, indent=1))
    print("GBM SETS DONE", a.tag)


if __name__ == "__main__":
    main()
