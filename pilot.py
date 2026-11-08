"""
Bonus "pilote" : utilise le world model comme simulateur pour comparer des plannings.

Deux usages :
  1. scénarios "et si" sur une fenêtre de la période de validation (pointage hors
     communications remplacé par EARTH, suppression des manœuvres, interrupteurs...) ;
  2. recherche locale : on permute, à l'intérieur de chaque journée, les heures de
     pointage hors communication (les fenêtres de communication avec la Terre et le
     volume d'observations par mode restent identiques), on garde les permutations qui
     font baisser l'énergie simulée.

    python pilot.py --weights weights_holdout --start-day 150 --days 14 --iters 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dio.data import get_vocab, load_period
from dio.simulate import load_ensemble, simulate_ensemble

N_INIT = 24 * 7


def energy(models, X, Y_init, n_init):
    mean, std, _ = simulate_ensemble(models, X, Y_init, n_init)
    return float(mean[n_init:].sum()), float(np.sqrt((std[n_init:] ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_holdout")
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--start-day", type=int, default=150, help="jour de annee2 ou commence la fenetre")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--out", default="reports/pilot.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta, models = load_ensemble(a.weights)
    vocab = get_vocab(Path(a.kit), ["annee1", "annee2"])
    X2, Y2 = load_period(Path(a.kit) / "annee2", vocab)
    feats = meta["features"]
    a0 = a.start_day * 24
    sl = slice(a0, a0 + N_INIT + a.days * 24)
    Xw = X2.iloc[sl][feats].copy()
    Yw = Y2.iloc[sl][meta["targets"]].to_numpy()
    base_e, base_u = energy(models, Xw.to_numpy(), Yw, N_INIT)
    truth_e = float(np.nansum(Yw[N_INIT:]))
    res = {"fenetre": {"debut": str(pd.to_datetime(Xw.index[N_INIT], unit="ms")), "jours": a.days},
           "energie_reelle_somme_horaire": truth_e, "planning_reel_simule": {"energie": base_e, "incertitude": base_u}}
    print(json.dumps(res, indent=1), flush=True)

    ftl_cols = [c for c in feats if c.startswith("ftl_") and c not in ("ftl_comms", "ftl_n_switch", "ftl_other")]
    comms = Xw["ftl_comms"].to_numpy()
    free = np.where(comms[N_INIT:] == 0)[0] + N_INIT  # heures sans communication : modifiables

    # --- scenarios "et si"
    scen = {}
    # 1. hors communication, tout le pointage devient EARTH (pas d'observation)
    Xs = Xw.copy()
    for c in ftl_cols:
        Xs.iloc[free, Xs.columns.get_loc(c)] = 1.0 if c == "ftl_EARTH" else 0.0
    scen["hors_comms_tout_EARTH"] = energy(models, Xs.to_numpy(), Yw, N_INIT)
    # 2. hors communication, tout le pointage devient NADIR (observation continue)
    if "ftl_NADIR" in ftl_cols:
        Xs = Xw.copy()
        for c in ftl_cols:
            Xs.iloc[free, Xs.columns.get_loc(c)] = 1.0 if c == "ftl_NADIR" else 0.0
        scen["hors_comms_tout_NADIR"] = energy(models, Xs.to_numpy(), Yw, N_INIT)
    # 3. aucune commande dmop hors communication (seulement si le modele lit les commandes)
    dm_cols = [c for c in feats if c.startswith("dm_c_") or c.startswith("dm_p_") or c == "dm_total"]
    if dm_cols:
        Xs = Xw.copy()
        for c in dm_cols:
            Xs.iloc[free, Xs.columns.get_loc(c)] = 0.0
        scen["hors_comms_sans_commandes"] = energy(models, Xs.to_numpy(), Yw, N_INIT)
    # 4. pas de manoeuvre (SLEW) hors communication : le temps de SLEW est rendu a EARTH
    if "ftl_SLEW" in ftl_cols:
        Xs = Xw.copy()
        sl_ = Xs.columns.get_loc("ftl_SLEW")
        e_ = Xs.columns.get_loc("ftl_EARTH")
        Xs.iloc[free, e_] = Xs.iloc[free, e_].to_numpy() + Xs.iloc[free, sl_].to_numpy()
        Xs.iloc[free, sl_] = 0.0
        scen["hors_comms_sans_manoeuvres"] = energy(models, Xs.to_numpy(), Yw, N_INIT)
    res["scenarios"] = {k: {"energie": e, "incertitude": u, "ecart_pct": 100 * (e - base_e) / base_e} for k, (e, u) in scen.items()}
    print(json.dumps(res["scenarios"], indent=1), flush=True)

    # --- recherche locale : permutation des heures hors communication a l'interieur d'une journee
    best = Xw.copy()
    best_e = base_e
    cols_idx = [best.columns.get_loc(c) for c in ftl_cols]
    hist = [(0, best_e)]
    for it in range(1, a.iters + 1):
        cand = best.copy()
        day = rng.integers(0, a.days)
        hours = [h for h in free if N_INIT + day * 24 <= h < N_INIT + (day + 1) * 24]
        if len(hours) < 2:
            continue
        i, j = rng.choice(hours, 2, replace=False)
        vi = cand.iloc[i, cols_idx].to_numpy().copy()
        vj = cand.iloc[j, cols_idx].to_numpy().copy()
        if np.allclose(vi, vj):
            continue
        cand.iloc[i, cols_idx] = vj
        cand.iloc[j, cols_idx] = vi
        e, _ = energy(models, cand.to_numpy(), Yw, N_INIT)
        if e < best_e:
            best, best_e = cand, e
            hist.append((it, e))
            print(f"iter {it}: energie {e:.3f} ({100 * (e - base_e) / base_e:+.2f} %)", flush=True)
    res["recherche_locale"] = {"iterations": a.iters, "energie_finale": best_e, "gain_pct": 100 * (best_e - base_e) / base_e,
                               "historique": hist}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    best.to_csv(Path(a.out).with_suffix(".planning.csv"))
    print(json.dumps(res["recherche_locale"] | {"historique": "..."}, indent=1))


if __name__ == "__main__":
    main()
