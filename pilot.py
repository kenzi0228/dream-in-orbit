"""
Bonus "pilote" : le world model comme simulateur pour comparer des plannings, en comparaison appariee.

Chaque membre de l'ensemble (une graine) simule le planning reel et le planning modifie ; la
difference d'energie est calculee membre par membre. On rapporte la moyenne de ces differences, leur
dispersion entre membres et l'accord des membres sur le signe : c'est l'incertitude du modele sur la
*difference*, bien plus petite que celle sur le niveau. Rien ici n'est verifiable contre la realite
(ces plannings n'ont pas vole) : le pilote mesure ce que le modele croit, pas ce que le satellite ferait.

  1. scenarios "et si" sur une fenetre de la periode de validation : pointage hors communication
     remplace par EARTH (plus d'observation), par NADIR (observation continue), manoeuvres (SLEW)
     supprimees. Les fenetres de communication avec la Terre ne sont jamais touchees, et les features
     derivees (anticipation, inertie) sont recalculees apres chaque modification du planning.
  2. recherche locale : permutation, a l'interieur de chaque journee, des heures de pointage hors
     communication (meme volume d'observation par mode et par jour) ; une permutation est gardee si
     tous les membres de l'ensemble la jugent economique.

    python pilot.py --weights weights_holdout --start-day 100 --days 14 --iters 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dio.data import load_period
from dio.features import recompute_derived
from dio.simulate import load_ensemble, simulate_ensemble

N_INIT = 24 * 7
PERIOD = {"strict": "annee2", "reverse": "annee1", "split": "annee2", "final": "annee2"}


def compare(E0: np.ndarray, E1: np.ndarray) -> dict:
    """Comparaison appariee membre par membre : ecart moyen, dispersion entre membres, accord sur le signe."""
    d = E1 - E0
    base = float(E0.mean())
    return {"energie": float(E1.mean()), "ecart_pct": float(100 * d.mean() / base),
            "dispersion_pct": float(100 * d.std(ddof=1) / base) if len(d) > 1 else 0.0,
            "accord_membres": float(np.mean(np.sign(d) == np.sign(d.mean()))) if d.mean() != 0 else 0.0,
            "robuste": bool(np.all(d < 0) or np.all(d > 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_holdout")
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--period", default=None, help="annee simulee (defaut : la periode de validation des poids)")
    ap.add_argument("--start-day", type=int, default=100, help="jour de la periode ou commence la fenetre")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--out", default="reports/pilot.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta, models = load_ensemble(a.weights)
    kit = Path(a.kit)
    period = a.period or PERIOD[meta.get("protocol", "strict")]
    X, Y = load_period(kit / period, meta["vocab"])
    gbm = meta.get("gbm")
    a0 = a.start_day * 24
    sl = slice(a0, a0 + N_INIT + a.days * 24)
    Yw = Y.iloc[sl][meta["targets"]].to_numpy()

    def energies(Xfull: pd.DataFrame) -> np.ndarray:
        """Energie simulee par membre de l'ensemble sur la fenetre (somme horaire des 33 circuits)."""
        _, _, members = simulate_ensemble(models, Xfull.iloc[sl], Yw, N_INIT, gbm=gbm)
        return members[:, N_INIT:].sum(axis=(1, 2))

    E0 = energies(X)
    truth_e = float(np.nansum(Yw[N_INIT:]))
    res = {"fenetre": {"periode": period, "debut": str(pd.to_datetime(X.index[sl][N_INIT], unit="ms")), "jours": a.days},
           "membres": int(len(E0)), "energie_reelle_somme_horaire": truth_e,
           "planning_reel_simule": {"energie": float(E0.mean()), "dispersion_membres": float(E0.std(ddof=1)) if len(E0) > 1 else 0.0,
                                    "ecart_au_reel_pct": float(100 * (E0.mean() - truth_e) / truth_e)}}
    print(json.dumps(res, indent=1), flush=True)

    ftl_cols = [c for c in X.columns if c.startswith("ftl_") and c not in ("ftl_comms", "ftl_n_switch", "ftl_other")]
    comms = X["ftl_comms"].to_numpy()
    free = np.array([i for i in range(sl.start + N_INIT, sl.stop) if comms[i] == 0])  # heures sans communication : modifiables
    col = {c: X.columns.get_loc(c) for c in ftl_cols}

    def modified(fn) -> pd.DataFrame:
        Xs = X.copy()
        fn(Xs)
        return recompute_derived(Xs)

    def set_all(mode):
        def fn(Xs):
            for c in ftl_cols:
                Xs.iloc[free, col[c]] = 1.0 if c == f"ftl_{mode}" else 0.0
        return fn

    def no_slew(Xs):
        Xs.iloc[free, col["ftl_EARTH"]] = Xs.iloc[free, col["ftl_EARTH"]].to_numpy() + Xs.iloc[free, col["ftl_SLEW"]].to_numpy()
        Xs.iloc[free, col["ftl_SLEW"]] = 0.0

    scen = {}
    if "ftl_EARTH" in ftl_cols:
        scen["hors_comms_tout_EARTH"] = compare(E0, energies(modified(set_all("EARTH"))))
    if "ftl_NADIR" in ftl_cols:
        scen["hors_comms_tout_NADIR"] = compare(E0, energies(modified(set_all("NADIR"))))
    if "ftl_SLEW" in ftl_cols and "ftl_EARTH" in ftl_cols:
        scen["hors_comms_sans_manoeuvres"] = compare(E0, energies(modified(no_slew)))
    res["scenarios"] = scen
    print(json.dumps(scen, indent=1), flush=True)

    # --- recherche locale : permutation des heures hors communication a l'interieur d'une journee
    best, Eb = X.copy(), E0
    hist, accepted = [(0, float(E0.mean()))], 0
    cols_idx = [col[c] for c in ftl_cols]
    for it in range(1, a.iters + 1):
        day = int(rng.integers(0, a.days))
        lo, hi = sl.start + N_INIT + day * 24, sl.start + N_INIT + (day + 1) * 24
        hours = free[(free >= lo) & (free < hi)]
        if len(hours) < 2:
            continue
        i, j = rng.choice(hours, 2, replace=False)
        vi, vj = best.iloc[i, cols_idx].to_numpy().copy(), best.iloc[j, cols_idx].to_numpy().copy()
        if np.allclose(vi, vj):
            continue
        cand = best.copy()
        cand.iloc[i, cols_idx], cand.iloc[j, cols_idx] = vj, vi
        E = energies(recompute_derived(cand))
        if np.all(E < Eb):  # gardee seulement si tous les membres sont d'accord
            best, Eb, accepted = cand, E, accepted + 1
            hist.append((it, float(E.mean())))
            print(f"iter {it}: energie {E.mean():.3f} ({100 * (E.mean() - E0.mean()) / E0.mean():+.3f} %)", flush=True)
    res["recherche_locale"] = {"iterations": a.iters, "permutations_gardees": accepted, **compare(E0, Eb), "historique": hist}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    best.iloc[sl][["ftl_comms"] + ftl_cols].to_csv(Path(a.out).with_suffix(".planning.csv"))
    print(json.dumps({k: v for k, v in res["recherche_locale"].items() if k != "historique"}, indent=1))


if __name__ == "__main__":
    main()
