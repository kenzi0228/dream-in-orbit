"""
Etude de melange finale : GRU (ensemble), gradient boosting (avec / sans statistiques glissantes),
ExtraTrees, RandomForest. Poids globaux (grille sur le simplexe) et poids par circuit, avec
validation croisee entre les deux protocoles (poids choisis sur l'un, evalues sur l'autre).

    python experiments/blend_all.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period  # noqa: E402
from dio.gbm import GBMComponent  # noqa: E402
from dio.metrics import rmse, rmse_per_circuit  # noqa: E402

N = 168


def gru_only(weights_dir: Path, Xte: pd.DataFrame, idx) -> np.ndarray:
    """Recupere la prediction GRU seule a partir de val_pred.csv (melange) et du composant GBM."""
    meta = json.loads((weights_dir / "meta.json").read_text())
    blend = pd.read_csv(weights_dir / "val_pred.csv", index_col="ut_ms").reindex(idx).to_numpy()
    w = meta.get("gbm_weight", 0)
    if w and (weights_dir / "gbm.pkl").exists():
        g = GBMComponent.load(weights_dir / "gbm.pkl")
        pg = g.predict(Xte[g.features].to_numpy())
        return (blend - w * pg) / (1 - w)
    return blend


def main():
    kit = Path("kit_etudiants")
    v = get_vocab(kit, ["annee1", "annee2"])
    X2, Y2 = load_period(kit / "annee2", v)
    k = len(X2) // 2
    protos = {"strict": (X2, Y2, Path("weights_holdout")), "split": (X2.iloc[k:], Y2.iloc[k:], Path("weights_holdout_split"))}
    comps = {"gbm_roll": "gbm_tout_avec_roll", "gbm": "gbm_tout_sans_roll", "et": "et", "rf": "rf"}
    P, T = {}, {}
    for proto, (Xte, Yte, wdir) in protos.items():
        idx = Yte.index[N:]
        T[proto] = Yte.to_numpy()[N:]
        P[proto] = {"gru": gru_only(wdir, Xte.iloc[N:], idx)}
        for name, f in comps.items():
            fp = Path(f"reports/{f}_{proto}.pred.csv")
            if fp.exists():
                P[proto][name] = pd.read_csv(fp, index_col="ut_ms").reindex(idx).to_numpy()
    names = [n for n in ["gru", "gbm_roll", "gbm", "et", "rf"] if all(n in P[p] for p in protos)]
    print("composants :", names)
    for proto in protos:
        print(proto, {n: round(rmse(P[proto][n], T[proto]), 5) for n in names})

    # grille de poids globaux (pas 0,05) sur le simplexe
    step = 0.05
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    def score(proto, w):
        return rmse(sum(wi * P[proto][n] for wi, n in zip(w, names)), T[proto])
    res = {}
    for proto in protos:
        best = min(grid, key=lambda w: score(proto, w))
        other = [p for p in protos if p != proto][0]
        res[f"poids_choisis_sur_{proto}"] = {"poids": dict(zip(names, [round(x, 2) for x in best])),
                                            proto: round(score(proto, best), 5), other: round(score(other, best), 5)}
    best_mean = min(grid, key=lambda w: (score("strict", w) + score("split", w)) / 2)
    res["poids_choisis_sur_la_moyenne"] = {"poids": dict(zip(names, [round(x, 2) for x in best_mean])),
                                          "strict": round(score("strict", best_mean), 5), "split": round(score("split", best_mean), 5)}
    ref = {"gru_seul": {p: round(rmse(P[p]["gru"], T[p]), 5) for p in protos}}
    if "gbm" in names:
        ref["0.6gru+0.4gbm"] = {p: round(rmse(0.6 * P[p]["gru"] + 0.4 * P[p]["gbm"], T[p]), 5) for p in protos}
    res["references"] = ref
    print(json.dumps(res, indent=1))

    # poids par circuit : choisis sur un protocole, evalues sur l'autre
    grid2 = [w for w in itertools.product(np.arange(0, 1 + 1e-9, 0.1), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    def per_circuit_weights(proto):
        C = T[proto].shape[1]
        W = np.zeros((C, len(names)))
        for j in range(C):
            tj = T[proto][:, j]
            m = ~np.isnan(tj)
            best = min(grid2, key=lambda w: np.mean((sum(wi * P[proto][n][m, j] for wi, n in zip(w, names)) - tj[m]) ** 2))
            W[j] = best
        return W
    def apply(proto, W):
        return sum(W[:, i][None, :] * P[proto][n] for i, n in enumerate(names))
    cross = {}
    for proto in protos:
        other = [p for p in protos if p != proto][0]
        W = per_circuit_weights(proto)
        cross[f"par_circuit_choisis_sur_{proto}"] = {proto: round(rmse(apply(proto, W), T[proto]), 5),
                                                     other: round(rmse(apply(other, W), T[other]), 5)}
        pd.DataFrame(W, index=Y2.columns, columns=names).to_csv(f"reports/blend_weights_per_circuit_{proto}.csv")
    res["par_circuit"] = cross
    print(json.dumps(cross, indent=1))
    Path("reports/blend_all.json").write_text(json.dumps(res, indent=1))
    print("BLEND ALL DONE")


if __name__ == "__main__":
    main()
