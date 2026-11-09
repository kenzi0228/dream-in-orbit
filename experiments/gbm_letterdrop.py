"""Augmentation "oubli des lettres" pour le GBM : on ajoute K copies du train ou, pour une partie des stems tires au
hasard, les features specifiques a la lettre (dm_c_, dm_r_, dm_s_) sont mises a la valeur "jamais vue" (compte 0,
recence plafonnee, interrupteur 0). Le modele apprend a se rabattre sur les features sans lettre quand une commande
change de lettre d'une annee a l'autre (ce qui arrive pour 20 des 29 interrupteurs entre annee1 et annee2).

    python experiments/gbm_letterdrop.py --set "all+dmop_sw+dmop_stem_cnt" --copies 2 --p-stem 0.5 --tag ldrop
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import feature_groups, get_vocab, load_period
from dio.features import RECENCY_CAP_H, split_stem
from dio.gbm import GBM_PARAMS
from dio.metrics import rmse
from experiments.gbm_sets import resolve
N = 168


def letter_columns(cols):
    """stem -> {colonne: valeur 'jamais vue'} pour les colonnes specifiques a une lettre."""
    by_stem = {}
    for c in cols:
        if c.startswith("dm_c_") or c.startswith("dm_r_"):
            st = split_stem(c[5:])[0]; by_stem.setdefault(st, {})[c] = 0.0 if c.startswith("dm_c_") else float(np.log1p(RECENCY_CAP_H))
        elif c.startswith("dm_s_"):
            by_stem.setdefault(c[5:-1], {})[c] = 0.0
    return by_stem


def augment(X, cols, copies, p_stem, rng):
    by_stem = letter_columns(cols); stems = list(by_stem); idx = {c: i for i, c in enumerate(cols)}
    out = [X]
    for _ in range(copies):
        Xc = X.copy()
        for st in stems:
            if rng.random() < p_stem:
                for c, val in by_stem[st].items():
                    Xc[:, idx[c]] = val
        out.append(Xc)
    return np.vstack(out), len(by_stem)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="all"); ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--p-stem", type=float, default=0.5); ap.add_argument("--tag", default="ldrop"); ap.add_argument("--params", default="")
    a = ap.parse_args(); params = dict(GBM_PARAMS, **(json.loads(a.params) if a.params else {}))
    kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
    X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
    G = feature_groups(X1.columns); k = len(X2) // 2; cols = resolve(a.set, X1.columns, G)
    protos = {"strict": (X1, Y1, X2, Y2), "split": (pd.concat([X1, X2.iloc[:k]]), pd.concat([Y1, Y2.iloc[:k]]), X2.iloc[k:], Y2.iloc[k:])}
    name = f"{a.set}~c{a.copies}p{a.p_stem}"; out = Path(f"reports/gbm_{a.tag}.json"); res = json.loads(out.read_text()) if out.exists() else {}
    res[name] = {"n_feat": len(cols)}
    for proto, (Xtr, Ytr, Xte, Yte) in protos.items():
        t0 = time.time(); rng = np.random.default_rng(0)
        Xa, Ya = Xtr[cols].to_numpy(), Ytr.to_numpy(); m = ~np.isnan(Ya).any(axis=1); Xa, Ya = Xa[m], Ya[m]
        Xaug, n_st = augment(Xa, cols, a.copies, a.p_stem, rng); Yaug = np.vstack([Ya] * (a.copies + 1))
        Xt = Xte[cols].to_numpy()[N:]; T = Yte.to_numpy()[N:]; P = np.zeros_like(T)
        for j in range(T.shape[1]):
            P[:, j] = HistGradientBoostingRegressor(random_state=0, **params).fit(Xaug, Yaug[:, j]).predict(Xt)
        P = np.clip(P, 0, None)
        res[name][proto] = {"gbm": round(rmse(P, T), 5), "n_stems": n_st, "t_s": round(time.time() - t0)}
        pd.DataFrame(P, index=Yte.index[N:], columns=Yte.columns).to_csv(f"reports/gbm_{a.tag}_{name}_{proto}.pred.csv")
        print(json.dumps({name: {proto: res[name][proto]}}), flush=True); out.write_text(json.dumps(res, indent=1))
    print("LETTERDROP DONE", name)


if __name__ == "__main__":
    main()
