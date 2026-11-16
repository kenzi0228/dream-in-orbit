#!/usr/bin/env python
"""
Construit demo/data.js (« window.DEMO_DATA = {...}; ») : toutes les données de la démo pas à pas,
tirées des sources de vérité du dépôt (kit, validation stricte, poids weights_holdout, rapports JSON).

    .venv/bin/python demo/build_demo_data.py          # depuis la racine du dépôt, ~30 s

Protocole illustré : strict (entraînement annee1, test annee2), poids weights_holdout. Rien n'est
ré-entraîné : le composant gradient boosting est seulement rechargé (weights_holdout/gbm.pkl) pour
obtenir sa prédiction seule (dérive, erreur par circuit) ; les features horaires viennent du cache
(dio.data.load_period, vocabulaire de weights_holdout/meta.json). Les séries régulières sont décrites
par {t0_ms, step_ms, n} + tableaux de valeurs (null = aucune mesure cette heure-là).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                       # dio.data lit le cache en chemin relatif (cache/)
sys.path.insert(0, str(ROOT))

from dio.data import feature_groups, load_period  # noqa: E402
from dio.features import H, _read_dmop, split_stem  # noqa: E402
from dio.gbm import GBMComponent  # noqa: E402
from dio.metrics import rmse, rmse_by_horizon  # noqa: E402

KIT, W, REP = Path("kit_etudiants"), Path("weights_holdout"), Path("reports")
OUT = Path("demo/data.js")
N_INIT = 24 * 7
# RMSE du flux officiel make_validation.py -> predict.py -> evaluate.py (reports/run_all.log, README, RAPPORT § 4) :
# constante documentaire, vérifiée ci-dessous contre val_report.json (rmse_final arrondie à 6 décimales).
FLUX_OFFICIEL = 0.074968
SIX = ["NPWD2551", "NPWD2451", "NPWD2532", "NPWD2851", "NPWD2881", "NPWD2561", "NPWD2531", "NPWD2791"]  # les 6 circuits qui pèsent le plus, un changement de régime, une ligne à deux états


# ------------------------------------------------------------------ petits outils

def r(x, nd):
    """Arrondi ; None si la valeur n'est pas finie (NaN -> null en JSON)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def ser(a, nd):
    return [r(v, nd) for v in np.asarray(a, dtype=float)]


def sig(a, n=3):
    """Arrondi à n chiffres significatifs (pour des valeurs sur plusieurs décades, tracées en échelle log)."""
    return [float(f"{v:.{n}g}") if np.isfinite(v) else None for v in np.asarray(a, dtype=float)]


def ints(a):
    return [int(round(v)) if np.isfinite(v) else None for v in np.asarray(a, dtype=float)]


def ms(date: str) -> int:
    return int(pd.Timestamp(date).value // 10 ** 6)


def day(t_ms) -> str:
    return str(pd.to_datetime(int(t_ms), unit="ms").date())


def tr6(a):
    """Moyenne par tranche de 6 h (NaN ignorés ; tranche vide -> NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(a, dtype=float).reshape(-1, 6), axis=1)


def jload(p):
    return json.loads(Path(p).read_text())


def fr(s: str) -> str:
    """Typographie française : espace insécable avant : ; ? ! et %."""
    for p in (":", ";", "?", "!", "%"):
        s = s.replace(f" {p}", f" {p}")
    return s


def count_rows(csv: Path) -> int:
    """Nombre de lignes de données d'un gros CSV (sans le lire en entier dans pandas)."""
    n, last = 0, b"\n"
    with open(csv, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n - 1 + (0 if last == b"\n" else 1)


def trous_dmop(path: Path, t_debut: int | None = None, t_fin: int | None = None, seuil_jours: float = 3.0) -> list[dict]:
    """Écarts de plus de `seuil_jours` entre deux commandes consécutives de dmop.csv, bornes de la période comprises :
    un silence entre le début de la période (`t_debut`, ms) et la première commande, ou entre la dernière commande et
    la fin de la période (`t_fin`, ms, fin de la dernière heure) compte aussi (`borne` = "debut" / "fin")."""
    t = np.sort(pd.read_csv(path)["ut_ms"].to_numpy(dtype=np.int64))
    seuil = seuil_jours * 24 * H

    def trou(a, b, borne=None):
        return {"debut": day(a), "fin": day(b - 1 if borne == "fin" else b), "jours": round(float((b - a) / H / 24), 1),
                "debut_ms": int(a), "fin_ms": int(b), "borne": borne}

    out = []
    if t_debut is not None and t[0] - t_debut > seuil:
        out.append(trou(t_debut, t[0], "debut"))
    g = np.diff(t)
    out += [trou(t[i], t[i + 1]) for i in np.where(g > seuil)[0]]
    if t_fin is not None and t_fin - t[-1] > seuil:
        out.append(trou(t[-1], t_fin, "fin"))
    return out


def lettres_du_stem(prefix: str) -> dict:
    """Lettres de variante émises pour un stem (commande sans sa lettre), par année, avec leurs comptes."""
    out = {}
    for yr in ["annee1", "annee2"]:
        df = _read_dmop(KIT / yr / "dmop.csv")
        sub = df[df["cmd"].str.startswith(prefix) & ~df["cmd"].str.contains(r"\.", regex=True)]
        sv = pd.DataFrame([split_stem(c) for c in sub["cmd"]], columns=["stem", "var"])
        cnt = sv[sv["stem"] == prefix + "_"]["var"].value_counts()
        out[yr] = {k: int(v) for k, v in cnt.items()}
    return out


def clean(o):
    """Types numpy -> types JSON ; NaN -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


# ------------------------------------------------------------------ chargement

def main():
    t_start = dt.datetime.now()
    meta = jload(W / "meta.json")
    vocab, targets = meta["vocab"], meta["targets"]
    assert meta["protocol"] == "strict" and meta["n_init"] == N_INIT
    X2, Y2 = load_period(KIT / "annee2", vocab)
    X1, Y1 = load_period(KIT / "annee1", vocab)
    Y1, Y2 = Y1[targets], Y2[targets]
    pred = pd.read_csv(W / "val_pred.csv", index_col="ut_ms")
    std = pd.read_csv(W / "val_std.csv", index_col="ut_ms")
    assert (pred.index == X2.index).all() and list(pred.columns) == targets and (std.index == X2.index).all()
    assert np.all(np.diff(X2.index.values) == H), "grille horaire non contiguë"
    T, B, S = Y2.to_numpy(dtype=float), pred.to_numpy(dtype=float), std.to_numpy(dtype=float)
    n_grid, n_c = T.shape
    cells = np.isfinite(T)
    m = cells.all(axis=1)                      # heures notées (toutes les cellules ou aucune)
    assert (cells.any(axis=1) == m).all()
    m[:N_INIT] = False                         # les 7 jours réels ne comptent pas dans le score

    gbm = GBMComponent.load(W / "gbm.pkl")
    PG = gbm.predict(X2[gbm.features].to_numpy())            # GBM seul (5 sacs), déterministe
    GRU = 2.0 * B - PG                                        # ensemble de GRU seul (mélange 0,5 / 0,5 inversé)
    gru1_path = Path("weights_holdout_abl_GRU/val_pred.csv")  # GRU seul, 1 graine (référence des ablations)
    GRU1 = pd.read_csv(gru1_path, index_col="ut_ms").reindex(pred.index)[targets].to_numpy(dtype=float) if gru1_path.exists() else None
    cal = jload(W / "calibration.json")
    scale = np.array([cal["scale"][c] for c in targets])
    S_cal = S * scale[None, :]
    const = np.nanmean(Y1.to_numpy(dtype=float), axis=0)
    C = np.tile(const, (n_grid, 1))

    val = jload(W / "val_report.json")
    val_rev, val_split = jload("weights_holdout_reverse/val_report.json"), jload("weights_holdout_split/val_report.json")
    base, base_rev, base_split = jload(REP / "baselines.json"), jload(REP / "baselines_reverse.json"), jload(REP / "baselines_split.json")
    tab = jload(REP / "tableaux_rapport.json")
    abl = jload(REP / "ablations.json")
    pilot = jload(REP / "pilot.json")
    calrev = jload(REP / "calibration_reverse.json")

    # cohérence avec les chiffres publiés
    rmse_final, rmse_gbm, rmse_gru = rmse(B[m], T[m]), rmse(PG[m], T[m]), rmse(GRU[m], T[m])
    rmse_const = rmse(C[m], T[m])
    assert abs(rmse_final - val["val_rmse"]) < 1e-9, rmse_final
    assert abs(rmse_gbm - val["val_rmse_gbm"]) < 1e-9, rmse_gbm
    assert abs(rmse_gru - val["val_rmse_gru"]) < 1e-6, rmse_gru
    assert abs(rmse_const - base["constante_train"]) < 1e-6, rmse_const
    assert abs(round(rmse_final, 6) - FLUX_OFFICIEL) < 1e-9, (rmse_final, FLUX_OFFICIEL)   # le flux officiel reproduit val_rmse
    n_params_gru = int(sum(v.numel() for v in torch.load(W / "model_0.pt", map_location="cpu")["state_dict"].values()))
    print(f"RMSE strict : final {rmse_final:.6f}  gbm {rmse_gbm:.6f}  gru {rmse_gru:.6f}  constante {rmse_const:.6f}")

    data = {}

    # ------------------------------------------------------------------ meta
    n_notees = int(m.sum())
    verite = pd.read_csv("validation/verite_horaire.csv", index_col="ut_ms")
    n_cellules = int(verite.notna().sum().sum())
    data["meta"] = {
        "genere_le": dt.date.today().isoformat(), "protocole_demo": "strict", "poids": "weights_holdout",
        # 6 décimales : la page affiche 4 décimales et un arrondi intermédiaire à 5 ferait 0,1057 au lieu de 0,1058
        "rmse_final": r(rmse_final, 6), "rmse_gru": r(rmse_gru, 6), "rmse_gbm": r(rmse_gbm, 6),
        "rmse_constante": r(rmse_const, 6), "rmse_persistance": r(base["persistance_derniere_heure"], 6),
        "rmse_constante_7jours": r(base["constante_7jours"], 6),
        "flux_officiel": FLUX_OFFICIEL, "n_cellules_notees": 530409, "n_heures_test": 16242, "n_init_h": N_INIT,
        "n_circuits": n_c, "n_features_gru": len(meta["features"]), "n_features_gbm": len(gbm.features),
        "n_params_gru": n_params_gru, "membres": val["n_members"],
        # précisions sur la grille (voir notes du script)
        "n_heures_grille": int(n_grid), "n_heures_sans_mesure": int(n_grid - cells.any(axis=1).sum()),
        "n_heures_notees_hors_init": int(n_cellules // n_c), "n_heures_notees_masque_demo": n_notees, "n_heures_a_predire_flux_officiel": int(len(pd.read_csv("validation/test/heures_a_predire.csv"))),
        "n_cellules_notees_flux_officiel": n_cellules,
        "pred_init": fr("Les 168 premières heures de val_pred ne sont pas la vérité : ce sont les prédictions un pas en avant "
                        "de l'ensemble sous teacher forcing (la vérité de l'heure précédente est ré-injectée), mélangées 0,5 / 0,5 "
                        "avec le GBM ; le rêve libre commence à l'heure 168."),
    }
    assert n_cellules == 530409 and len(verite) * n_c == 530409

    # ------------------------------------------------------------------ périodes
    y1_cache, y2_cache = pd.read_parquet("cache/annee1_y.parquet"), pd.read_parquet("cache/annee2_y.parquet")
    data["periodes"] = {
        "annee1": {"debut": day(X1.index[0]), "fin": day(X1.index[-1]), "heures": int(len(y1_cache)), "heures_grille": int(len(X1)),
                   "mesures_brutes": count_rows(KIT / "annee1" / "power.csv"), "jours": int(len(X1) // 24)},
        "annee2": {"debut": day(X2.index[0]), "fin": day(X2.index[-1]), "heures": int(len(y2_cache)), "heures_grille": int(len(X2)),
                   "mesures_brutes": count_rows(KIT / "annee2" / "power.csv"), "jours": int(len(X2) // 24)},
    }
    assert data["periodes"]["annee1"]["heures"] == 16456 and data["periodes"]["annee2"]["heures"] == 16242

    # ------------------------------------------------------------------ circuits
    sq_const = np.nansum((C[m] - T[m]) ** 2, axis=0)
    sq_final = np.nansum((B[m] - T[m]) ** 2, axis=0)
    y1v, y2v = Y1.to_numpy(dtype=float), Y2.to_numpy(dtype=float)
    circuits = []
    for j, c in enumerate(targets):
        row = {"id": c,
               "moyenne_annee1": r(np.nanmean(y1v[:, j]), 4), "ecart_type_annee1": r(np.nanstd(y1v[:, j]), 4),
               "moyenne_annee2": r(np.nanmean(y2v[:, j]), 4), "ecart_type_annee2": r(np.nanstd(y2v[:, j]), 4),
               "part_const_mse_pct": r(100 * sq_const[j] / sq_const.sum(), 2),
               "rmse_final": r(rmse(B[m, j], T[m, j]), 5), "rmse_gru": r(rmse(GRU[m, j], T[m, j]), 5),
               "rmse_gru_1graine": None if GRU1 is None else r(rmse(GRU1[m, j], T[m, j]), 5),
               "rmse_gbm": r(rmse(PG[m, j], T[m, j]), 5), "rmse_constante": r(rmse(C[m, j], T[m, j]), 5),
               "part_mse_pct": r(100 * sq_final[j] / sq_final.sum(), 2),
               "quasi_constant": bool(np.nanstd(y1v[:, j]) < 0.001),
               "facteur_calibration": r(cal["scale"][c], 3)}
        circuits.append(row)
    circuits.sort(key=lambda x: -x["part_const_mse_pct"])
    data["circuits"] = circuits
    print("part MSE constante :", [(x["id"], x["part_const_mse_pct"]) for x in circuits[:6]])
    print("quasi constants annee1 :", sum(x["quasi_constant"] for x in circuits))

    # ------------------------------------------------------------------ brut : 48 h de mesures irrégulières
    pi = pd.read_csv("validation/test/power_init.csv").sort_values("ut_ms")
    hours = X2.index[:N_INIT]
    j51, j45 = targets.index("NPWD2551"), targets.index("NPWD2451")
    scores = [np.nanmean(T[k:k + 48, j45]) for k in range(0, N_INIT - 48 + 1)]
    k0 = int(np.argmax(scores))                         # les 48 h où NPWD2451 est le plus actif (NPWD2551 est presque éteint en juillet, hors saison d'éclipses)
    t0 = int(hours[k0])
    sel = pi[(pi["ut_ms"] >= t0) & (pi["ut_ms"] < t0 + 48 * H)]
    hourly = sel.assign(h=(sel["ut_ms"] // H) * H).groupby("h")[["NPWD2551", "NPWD2451"]].mean().reindex(hours[k0:k0 + 48])
    assert np.allclose(hourly["NPWD2451"].to_numpy(), T[k0:k0 + 48, j45], atol=1e-5)
    data["brut"] = {
        "circuit": "NPWD2451", "t0_ms": t0, "n": int(len(sel)), "debut": str(pd.to_datetime(t0, unit="ms")),
        "t_rel_s": ser((sel["ut_ms"].to_numpy() - t0) / 1000.0, 1), "y": ser(sel["NPWD2451"].to_numpy(), 4),
        "circuit_2": "NPWD2551", "y_2": ser(sel["NPWD2551"].to_numpy(), 4),
        "pas_median_s": r(np.median(np.diff(sel["ut_ms"].to_numpy())) / 1000.0, 1),
        "horaire": {"t0_ms": t0, "step_ms": H, "n": 48, "y": ser(hourly["NPWD2451"].to_numpy(), 4), "y_2": ser(hourly["NPWD2551"].to_numpy(), 4)},
    }

    # ------------------------------------------------------------------ fenêtre de 14 jours (saison d'éclipses)
    i0 = int(X2.index.get_loc(ms("2010-10-25")))
    sl = slice(i0, i0 + 336)
    w = X2.iloc[sl]
    data["fenetre14"] = {
        "t0_ms": int(X2.index[i0]), "step_ms": H, "n": 336, "debut": "2010-10-25", "fin": day(X2.index[i0 + 335]),
        "jour_du_reve": int((i0 - N_INIT) // 24),
        "verite": {c: ser(T[sl, targets.index(c)], 4) for c in ["NPWD2551", "NPWD2451", "NPWD2532"]},
        "pred": {c: ser(B[sl, targets.index(c)], 4) for c in ["NPWD2551", "NPWD2451", "NPWD2532"]},
        "std_cal": {c: ser(S_cal[sl, targets.index(c)], 4) for c in ["NPWD2551", "NPWD2451", "NPWD2532"]},
        "saaf_sa": ser(w["saaf_sa"], 3), "saaf_sx": ser(w["saaf_sx"], 3),
        "ev_umbra": ser(w["ev_umbra"], 4), "ev_penumbra": ser(w["ev_penumbra"], 4),
        "ftl_comms": ser(w["ftl_comms"], 4), "ftl_EARTH": ser(w["ftl_EARTH"], 4), "ftl_NADIR": ser(w["ftl_NADIR"], 4),
        "ftl_SLEW": ser(w["ftl_SLEW"], 4), "ftl_D4PNPO": ser(w["ftl_D4PNPO"], 4),
        "dmop_par_heure": ints(w["dm_total"]),
        "lt_eclipseduration_min": ser(w["lt_eclipseduration_min"], 2), "lt_solarconstantmars": ser(w["lt_solarconstantmars"], 2),
        "ev_log_alt": ser(w["ev_log_alt"], 4),
    }

    # ------------------------------------------------------------------ features
    g_gru, g_gbm = feature_groups(meta["features"]), feature_groups(gbm.features)
    descr = {
        "saaf": ("Angles du Soleil (saaf.csv) : moyenne horaire des angles sa, sx, sy, sz et de leurs cosinus, écart-type horaire de sa",
                 ["saaf_sa", "saaf_cos_sx", "saaf_sa_std"]),
        "ltdata": ("Tendances journalières (ltdata.csv) interpolées à l'heure : distance Soleil-Mars, constante solaire, durées d'éclipse et d'occultation ; les deux colonnes de calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) sont retirées",
                   ["lt_solarconstantmars", "lt_eclipseduration_min", "lt_sunmars_km"]),
        "evtf": ("Événements orbitaux (evtf.csv) : fraction d'heure en ombre (umbra, pénombre, ombre des lunes), en occultation, en contact avec une station sol ; temps depuis et jusqu'au péricentre, phase orbitale (cos, sin), période, log de l'altitude, nombre d'événements",
                 ["ev_umbra", "ev_orbit_phase_cos", "ev_log_alt"]),
        "ftl": ("Pointage (ftl.csv) : fraction d'heure dans chacun des 14 types appris sur annee1 (EARTH, SLEW, NADIR, D4PNPO…) et hors vocabulaire, fraction en communication, nombre de changements de pointage dans l'heure",
                ["ftl_EARTH", "ftl_comms", "ftl_n_switch"]),
        "dmop_stem_cnt": ("Commandes (dmop.csv) : comptes horaires par stem, c'est-à-dire par commande sans sa lettre de variante (117 stems émis au moins 50 fois dans annee1)",
                          ["dm_sc_AAAAF20_1", "dm_sc_ATTTF301_", "dm_sc_AXXX301_"]),
        "dmop_base": ("Commandes : comptes horaires par sous-système (26 préfixes), total de commandes dans l'heure, fraction d'heure dans les intervalles début/fin appariés (PENS/PENE, MOCS/MOCE…)",
                      ["dm_p_ATTT", "dm_total", "dm_act_PENS"]),
        "dmop_prefix_rec": ("Récence par sous-système : log(1 + heures depuis la dernière commande du préfixe), plafonnée à 30 jours",
                            ["dm_pr_AAAA", "dm_pr_ATTT", "dm_pr_AXXX"]),
        "dmop_stem_rec": ("Récence par stem : log(1 + heures depuis la dernière émission du stem, toutes lettres confondues), plafonnée à 30 jours",
                          ["dm_st_AAAAF20_1", "dm_st_ATTTF301_", "dm_st_AXXX301_"]),
        "dmop_sw": ("Interrupteurs sans lettre : pour chacun des 18 interrupteurs (stems à plusieurs variantes), heures depuis le dernier changement de variante et heures jusqu'au prochain (le planning est connu), en log1p",
                    ["dm_sw_AAAAF09_1_chg", "dm_sw_AAAAF09_1_next", "dm_sw_AXXX301__chg"]),
        "dmop_stem_next": ("Commandes à venir : compte de chaque stem dans l'heure suivante (dm_sn1_) et sur les 6 heures suivantes (dm_sn6_)",
                           ["dm_sn1_AAAAF20_1", "dm_sn6_AAAAF20_1", "dm_sn6_ATTTF301_"]),
        "flux": ("Flux solaire reçu par face : constante solaire × cos(angle) tronqué à 0, pour chaque face (±x, ±y, ±z et l'axe sa), corrigé de l'ombre, plus le flux total",
                 ["fx_total", "fx_sx_plus", "fx_sz_minus"]),
        "hist": ("Mémoire thermique explicite : heures depuis la dernière ombre, ombre cumulée sur 7 h, 24 h et 7 jours, flux moyen sur 6 h et 24 h, communication moyenne sur 24 h",
                 ["hist_h_since_umbra", "hist_umbra_24h", "hist_fx_24h"]),
        "leadlag": ("Anticipation et retard : valeurs aux heures t−3 … t+3 de dix colonnes clés (ombre, pénombre, cosinus solaires, communication, pointage EARTH et SLEW, altitude)",
                    ["ll_ev_umbra_+1", "ll_saaf_cos_sa_-3", "ll_ftl_comms_+3"]),
    }
    groupes = []
    for nom in ["saaf", "ltdata", "evtf", "ftl", "dmop_stem_cnt", "dmop_base", "dmop_prefix_rec", "dmop_stem_rec", "dmop_sw", "dmop_stem_next", "flux", "hist", "leadlag"]:
        n_gru, n_gbm = len(g_gru.get(nom, [])), len(g_gbm.get(nom, []))
        ex = [c for c in descr[nom][1] if c in set(g_gbm.get(nom, [])) | set(g_gru.get(nom, []))]
        assert len(ex) == 3, (nom, ex)
        groupes.append({"nom": nom, "n": n_gru if n_gru else n_gbm, "n_gru": n_gru, "n_gbm": n_gbm,
                        "dans_gru": n_gru > 0, "dans_gbm": n_gbm > 0, "description_fr": fr(descr[nom][0]), "exemples": ex})
    assert sum(g["n_gru"] for g in groupes) == len(meta["features"]) and sum(g["n_gbm"] for g in groupes) == len(gbm.features)
    data["features"] = {
        "total_gru": len(meta["features"]), "total_gbm": len(gbm.features), "total_cache": int(X2.shape[1]), "groupes": groupes,
        "exclus_gru": fr("calendrier (lt_earthmars_km, lt_sunmarsearthangle_deg), récences, interrupteurs, commandes à venir, flux par face, mémoire thermique explicite, anticipation, statistiques glissantes, toute feature encodée par lettre"),
        "exclus_gbm": fr("calendrier (lt_earthmars_km, lt_sunmarsearthangle_deg), statistiques glissantes (roll_, ewm_) et toute feature encodée par lettre (dm_s_ états, dm_c_ comptes, dm_r_ récences par commande)"),
        "groupes_gru": meta["args"]["keep_groups"].split(","),
    }

    # ------------------------------------------------------------------ entraînement (journal de la graine 0)
    log = [json.loads(l) for l in (W / "train_log.jsonl").read_text().splitlines() if l.strip()]
    ep0 = [l for l in log if l.get("seed") == 0 and "epoch" in l]
    args = meta["args"]
    hyper = {"epochs": args["epochs"], "tf_epochs": args["tf_epochs"], "hidden": args["hidden"], "dropout": args["dropout"],
             "feat_dropout": args["feat_dropout"], "lr": args["lr"], "weight_decay": args["weight_decay"], "streams": args["streams"],
             "seg": args["seg"], "noise": args["noise"], "seeds": args["seeds"]}
    assert hyper == {"epochs": 80, "tf_epochs": 15, "hidden": 128, "dropout": 0.3, "feat_dropout": 0.2, "lr": 0.001, "weight_decay": 0.001,
                     "streams": 16, "seg": 168, "noise": 0.05, "seeds": 5}
    finals = {l["seed"]: l["val_rmse"] for l in log if "epoch" in l and "val_rmse" in l and l["epoch"] == args["epochs"]}
    data["entrainement"] = {
        "graine": 0, "epoque": [l["epoch"] for l in ep0], "train_rmse": [r(l["train_rmse"], 5) for l in ep0], "p_tf": [r(l["p_tf"], 3) for l in ep0],
        "reve": [[l["epoch"], r(l["val_rmse"], 5)] for l in ep0 if "val_rmse" in l],
        "hyperparametres": hyper, "membres_rmse": [r(v, 5) for v in val["val_rmse_members"]],
        "membres_rmse_gru_seul": [r(finals[s], 5) for s in sorted(finals)],
        "temps_total_s": int(round(sum(l.get("time_s", 0) + l.get("val_time_s", 0) for l in log if "epoch" in l))),
        "temps_graine0_s": int(round(sum(l.get("time_s", 0) + l.get("val_time_s", 0) for l in ep0))),
        "gbm": {"features": next(l["gbm_features"] for l in log if "gbm_features" in l), "sacs": gbm.bags, "temps_s": next(l["time_s"] for l in log if "gbm_features" in l),
                "rmse": r(rmse_gbm, 5), "params": gbm.params},
        "note": fr("reve = RMSE de validation (annee2, GRU seul de la graine 0, sans GBM) mesurée toutes les 10 époques par une simulation libre complète ; "
                   "membres_rmse = RMSE des cinq membres du mélange final (0,5 GRU + 0,5 GBM)."),
    }

    # ------------------------------------------------------------------ rêve : moyennes par tranche de 6 h sur toute annee2
    assert n_grid % 6 == 0
    tot_T = np.where(m | (np.arange(n_grid) < N_INIT), np.nansum(T, axis=1), np.nan)
    tot_T[~cells.any(axis=1)] = np.nan
    reve = {"t0_ms": int(X2.index[0]), "step_ms": 6 * H, "n": int(n_grid // 6), "n_heures": int(n_grid), "init_h": N_INIT,
            "fin_init_ms": int(X2.index[N_INIT]), "tranche_fin_init": int(N_INIT // 6),
            "circuits": {}, "total": {"verite": ser(tr6(tot_T), 4), "pred": ser(tr6(B.sum(axis=1)), 4), "std_cal": None,
                                      "note": fr("pas de bande pour la somme des 33 circuits : la somme en quadrature des σ par circuit "
                                                 "supposerait des erreurs indépendantes entre circuits, alors que les membres se trompent "
                                                 "ensemble (RAPPORT § 4) ; il faudrait les prédictions par membre")},
            "note_std_cal": fr("std_cal = σ horaire calibré (facteur par circuit ajusté sur le protocole inversé) moyenné par tranche de 6 h : "
                               "un ordre de grandeur de l'incertitude, pas l'écart-type de la moyenne 6 h")}
    for c in SIX:
        j = targets.index(c)
        reve["circuits"][c] = {"verite": ser(tr6(T[:, j]), 4), "pred": ser(tr6(B[:, j]), 4), "std_cal": ser(tr6(S_cal[:, j]), 4),
                               "rmse": r(rmse(B[m, j], T[m, j]), 5), "facteur_calibration": r(cal["scale"][c], 3)}
    data["reve"] = reve

    # ------------------------------------------------------------------ dérive : RMSE par semaine depuis le début du rêve
    wk_final = rmse_by_horizon(B[N_INIT:], T[N_INIT:])
    wk_gbm = rmse_by_horizon(PG[N_INIT:], T[N_INIT:])
    wk_gru = rmse_by_horizon(GRU[N_INIT:], T[N_INIT:])
    wk_const = rmse_by_horizon(C[N_INIT:], T[N_INIT:])
    C7 = np.tile(np.nanmean(T[:N_INIT], axis=0), (n_grid, 1))      # constante calée sur les 7 jours réels (derive.png)
    wk_const7 = rmse_by_horizon(C7[N_INIT:], T[N_INIT:])
    assert abs(rmse(C7[m], T[m]) - base["constante_7jours"]) < 1e-6
    assert [k for k, _ in wk_final] == [k for k, _ in val["val_rmse_by_week"]]
    assert max(abs(a - b) for (_, a), (_, b) in zip(wk_final, val["val_rmse_by_week"])) < 1e-9
    data["derive"] = {
        "semaine": [k for k, _ in wk_final], "t0_ms": int(X2.index[N_INIT]), "step_ms": 7 * 24 * H,
        "rmse_final": [r(v, 5) for _, v in wk_final], "rmse_gbm_seul": [r(v, 5) for _, v in wk_gbm],
        "rmse_gru_seul": [r(v, 5) for _, v in wk_gru], "rmse_constante": [r(v, 5) for _, v in wk_const],
        "rmse_constante_7jours": [r(v, 5) for _, v in wk_const7],
        "note": fr("rmse_constante = moyenne de annee1 par circuit ; rmse_constante_7jours = moyenne des 7 jours réels (courbe grise de reports/figures/derive.png) ; "
                   "rmse_gbm_seul = composant gradient boosting livré (weights_holdout/gbm.pkl, 5 sacs), pas la référence de baselines.py"),
        "horizons": {"semaines_1_4": r(tab["derive"]["semaines_1_4"], 5), "semaines_5_50": r(tab["derive"]["semaines_5_50"], 5),
                     "semaines_51_fin": r(tab["derive"]["semaines_51_fin"], 5), "pire_semaine": tab["derive"]["pire_semaine"],
                     "pire_rmse": r(tab["derive"]["pire_rmse"], 5), "n_semaines": tab["derive"]["n_semaines"],
                     "numerotation": fr("pire_semaine est numérotée à partir de 1 (semaine 63 = indice 62 de derive.semaine)")},
    }
    assert data["derive"]["horizons"]["pire_semaine"] == int(np.argmax([v for _, v in wk_final])) + 1

    # ------------------------------------------------------------------ résultats des trois protocoles
    def proto(v, b, n_gru_seeds):
        final, cst = v["val_rmse"], b["constante_train"]
        return {"constante": r(cst, 6), "gbm": r(v["val_rmse_gbm"], 6), "gru": r(v["val_rmse_gru"], 6), "final": r(final, 6),
                "reduction_pct": int(round(100 * (1 - final / cst))), "membres": [r(x, 6) for x in v["val_rmse_members"]],
                "graines_gru": n_gru_seeds, "constante_7jours": r(b["constante_7jours"], 6), "persistance": r(b["persistance_derniere_heure"], 6)}

    order = np.argsort([-v for _, v in wk_final])[:5]
    data["resultats"] = {
        "protocoles": {"strict": proto(val, base, 5), "inverse": proto(val_rev, base_rev, 5), "chrono": proto(val_split, base_split, 3)},
        "poids_gbm": {k: {w_: r(v[f"val_rmse_blend_{w_}"], 5) for w_ in ["0.3", "0.4", "0.5", "0.6", "0.7"]}
                      for k, v in [("strict", val), ("inverse", val_rev), ("chrono", val_split)]},
        "corr_erreurs_gru_gbm": r(tab["corr_erreurs_gru_gbm"], 2), "corr_erreurs_gru_gbm_precis": r(tab["corr_erreurs_gru_gbm"], 5),
        "pires_semaines": [{"semaine": int(wk_final[i][0]) + 1, "indice": int(wk_final[i][0]), "rmse": r(wk_final[i][1], 5),
                            "debut": day(X2.index[N_INIT + 168 * int(wk_final[i][0])])} for i in order],
        "flux_officiel": FLUX_OFFICIEL,
        "libelles_fr": {"strict": fr("strict (annee1 → annee2)"), "inverse": fr("inversé (annee2 → annee1)"), "chrono": fr("chronologique (2 ans → ½ annee2)"),
                        "constante": fr("Constante (moyenne du train)"), "gbm": fr("Gradient boosting seul (5 sacs, sans lettre)"),
                        "gru": fr("Ensemble de GRU seul (5 graines ; 3 en chronologique)"), "final": fr("World model livré : 0,5 GRU + 0,5 GBM")},
    }
    assert [p["reduction_pct"] for p in data["resultats"]["protocoles"].values()] == [29, 28, 23]
    assert data["resultats"]["corr_erreurs_gru_gbm"] == 0.78, data["resultats"]["corr_erreurs_gru_gbm"]   # chiffre publié (RAPPORT § 4)

    # ------------------------------------------------------------------ incertitude
    chk, fit = calrev["check"], calrev["fit_check"]["weights_holdout_reverse"]
    lignes = []
    for nom, key in [("strict", "weights_holdout"), ("chrono", "weights_holdout_split")]:
        b_, c_ = chk[key]["brut"], chk[key]["calibre"]
        lignes.append({"protocole": nom, "cov1_brut": r(b_["couverture_1s"], 4), "cov2_brut": r(b_["couverture_2s"], 4),
                       "cov1_cal": r(c_["couverture_1s"], 4), "cov2_cal": r(c_["couverture_2s"], 4), "ratio_cal": r(c_["rmse_sur_rms_std"], 3),
                       "ratio_brut": r(b_["rmse_sur_rms_std"], 3), "spearman_brut": r(b_["spearman_std_vs_abs_err"], 4),
                       "spearman_cal": r(c_["spearman_std_vs_abs_err"], 4), "ajustement": False})
    lignes.append({"protocole": "inverse", "cov1_brut": None, "cov2_brut": None, "cov1_cal": r(fit["couverture_1s"], 4), "cov2_cal": r(fit["couverture_2s"], 4),
                   "ratio_cal": r(fit["rmse_sur_rms_std"], 3), "ratio_brut": None, "spearman_brut": None, "spearman_cal": r(fit["spearman_std_vs_abs_err"], 4), "ajustement": True})
    e_all, s_all = np.abs(B[N_INIT:] - T[N_INIT:]).ravel(), S[N_INIT:].ravel()
    ok = np.isfinite(e_all)
    e_all, s_all = e_all[ok], s_all[ok]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(e_all), size=3000, replace=False)
    actifs = [j for j in range(n_c) if np.nanstd(y1v[:, j]) >= 0.001]
    e_act, s_act = np.abs(B[N_INIT:][:, actifs] - T[N_INIT:][:, actifs]).ravel(), S[N_INIT:][:, actifs].ravel()
    ok_a = np.isfinite(e_act)
    e_act, s_act = e_act[ok_a], s_act[ok_a]
    pick_a = np.random.default_rng(0).choice(len(e_act), size=3000, replace=False)
    spearman = float(pd.Series(s_all).corr(pd.Series(e_all), method="spearman"))
    ratio = float(np.sqrt(np.mean(e_all ** 2)) / np.sqrt(np.mean(s_all ** 2)))
    assert abs(spearman - tab["incertitude_brute"]["spearman"]) < 1e-6 and abs(ratio - tab["incertitude_brute"]["rmse_sur_rms_std"]) < 1e-6
    data["incertitude"] = {
        "spearman_brut": r(spearman, 4), "ratio_brut": r(ratio, 3), "lignes": lignes,
        "facteurs": {c: r(cal["scale"][c], 3) for c in targets}, "facteur_global": r(cal["global"], 3), "ajuste_sur": cal["fit_on"],
        # 3 chiffres significatifs (et non des décimales fixes) : les valeurs sont tracées en échelle log sur plusieurs décades
        "nuage": {"std": sig(s_all[pick]), "abs_err": sig(e_all[pick]), "n": 3000, "graine": 0},
        "nuage_actifs": {"std": sig(s_act[pick_a]), "abs_err": sig(e_act[pick_a]), "n": 3000, "graine": 0,
                         "circuits": [targets[j] for j in actifs], "note": fr("même tirage, restreint aux circuits non quasi constants (écart-type annee1 ≥ 0,001)")},
        "unites": fr("couvertures en fraction de cellules (0,4476 = 45 %) ; attendu gaussien : 0,68 à ±1σ et 0,95 à ±2σ ; "
                     "nuage = écart-type brut de l'ensemble (avant calibration) contre erreur absolue, cellules du strict hors 168 h d'init"),
        "libelles_fr": {"strict": "strict (annee2)", "chrono": fr("chronologique (½ annee2)"), "inverse": fr("inversé (annee1, ajustement)")},
    }

    # ------------------------------------------------------------------ trous du planning et décrochage de 2011
    bornes = {"annee1": (int(X1.index[0]), int(X1.index[-1]) + H), "annee2": (int(X2.index[0]), int(X2.index[-1]) + H)}
    trous = {yr: trous_dmop(KIT / yr / "dmop.csv", *bornes[yr]) for yr in ["annee1", "annee2"]}
    # annee2 : la dernière commande date du 14 mai 2012, 12,6 jours avant la fin de la période (trou terminal, borne = "fin")
    assert len(trous["annee1"]) == 7 and len(trous["annee2"]) == 9 and trous["annee2"][-1]["borne"] == "fin", trous
    data["trous_dmop"] = trous
    # même règle de comptage que le rapport : trous de plus de 3 jours entre deux commandes, hors silence terminal
    jours2 = sum(t["jours"] for t in trous["annee2"] if t.get("borne") != "fin")
    jours1 = sum(t["jours"] for t in trous["annee1"] if t.get("borne") != "fin")
    assert abs(jours2 - 126) < 1 and abs(jours1 - 56) < 1, (jours1, jours2)
    data["trous_dmop"]["jours_rapport"] = {"annee1": r(jours1, 1), "annee2": r(jours2, 1), "regle": fr("trous de plus de 3 jours entre deux commandes, hors silence terminal")}

    # ------------------------------------------------------------------ NPWD2551 hors saison d'éclipses, par tranche d'angle d'aspect solaire
    hs = (X2["lt_eclipseduration_min"].to_numpy() == 0) & ~np.isnan(T[:, j51])
    sa = X2["saaf_sa"].to_numpy()
    tranches = [("≤ 2°", sa <= 2), ("2 à 10°", (sa > 2) & (sa <= 10)), ("10 à 20°", (sa > 10) & (sa <= 20)), ("> 20°", sa > 20)]
    data["sa_hors_saison"] = {"circuit": "NPWD2551", "periode": "annee2",
                              "definition": fr("heures des jours sans éclipse (lt_eclipseduration_min = 0) ; vérité horaire de NPWD2551 par tranche d'angle d'aspect solaire sa"),
                              "tranches": [{"tranche": fr(lab), "moyenne": r(float(np.nanmean(T[hs & m, j51])), 3), "n": int((hs & m).sum())} for lab, m in tranches if (hs & m).sum() > 0]}
    print("sa hors saison :", data["sa_hors_saison"]["tranches"])
    dm2 = _read_dmop(KIT / "annee2" / "dmop.csv")     # AACF n'existe pas dans annee1 : absent du vocabulaire strict, compté ici directement
    aacf = dm2[dm2["prefix"] == "AACF"].groupby("h").size().reindex(X2.index).fillna(0).to_numpy()
    assert int(aacf.sum()) == 5825, int(aacf.sum())
    d0, d1 = int(X2.index.get_loc(ms("2011-07-15"))), int(X2.index.get_loc(ms("2011-11-30 23:00"))) + 1
    sd = slice(d0, d1)
    data["decrochage"] = {
        "t0_ms": int(X2.index[d0]), "step_ms": H, "n": int(d1 - d0), "debut": "2011-07-15", "fin": "2011-11-30",
        "verite_2551": ser(T[sd, j51], 4), "pred_2551": ser(B[sd, j51], 4), "std_cal_2551": ser(S_cal[sd, j51], 4),
        "verite_2451": ser(T[sd, targets.index("NPWD2451")], 4), "pred_2451": ser(B[sd, targets.index("NPWD2451")], 4),
        "dmop_par_heure": ints(X2["dm_total"].iloc[sd]), "ev_umbra": ser(X2["ev_umbra"].iloc[sd], 4),
        "ftl_D4PNPO": ser(X2["ftl_D4PNPO"].iloc[sd], 4), "dmop_AACF_par_heure": ints(aacf[sd]),
        "trous": trous["annee2"], "date_rupture": "2011-08-30", "date_rupture_diag_2551": jload(REP / "diag_2551.json")["rupture"]["jour"],
        "note": fr("date_rupture = 30 août 2011, premier jour sans commande du trou de 14 jours (RAPPORT § 5) ; diag_2551.json date la rupture "
                   "au 2 septembre 2011 (premier jour où NPWD2551 en umbra passe sous la référence de annee1)."),
    }
    assert data["decrochage"]["n"] == 139 * 24

    # ------------------------------------------------------------------ lettres de variante
    ex = lettres_du_stem("AXXX301")
    data["lettres"] = {
        "interrupteurs": 29, "changent": 20, "sans_lettre_commune": 7, "commandes_seulement_annee2": 49, "source": fr("rapport, § 8 (journal des essais)"),
        "interrupteurs_vocab_strict": len({st for st, _ in vocab["switches"]}), "couples_stem_lettre_vocab_strict": len(vocab["switches"]),
        "stems_vocab_strict": len(vocab["stems"]), "commandes_vocab_strict": len(vocab["cmds"]),
        "exemple": {"stem": "AXXX301", "stem_interne": "AXXX301_",
                    "annee1": sorted(k for k, v in ex["annee1"].items() if v >= 50), "annee2": sorted(k for k, v in ex["annee2"].items() if v >= 50),
                    "comptes_annee1": ex["annee1"], "comptes_annee2": ex["annee2"], "seuil_vocabulaire": 50},
    }

    # ------------------------------------------------------------------ ablations
    lib = {
        "complet (GRU + gradient boosting)": ("système complet : GRU + composant gradient boosting (0,5)", "−0,005",
                                               "le GBM corrige les deux chauffages (0,28 → 0,26 ; 0,24 → 0,22) sans toucher au reste (pires : NPWD2551 0,26, NPWD2451 0,22)"),
        "GRU seul (sans composant gradient boosting)": ("GRU seul", "référence", "une graine ; la moyenne de 5 graines donne 0,0796 (pires : NPWD2551 0,28, NPWD2451 0,24)"),
        "sans_memoire (MLP heure par heure, avec re-injection)": ("sans mémoire (MLP heure par heure)", "+0,002",
                                                                 "la mémoire vaut 0,002, à peine plus que le bruit de graine (pires : NPWD2551 0,28, NPWD2451 0,25)"),
        "sans_reinjection (planning seul)": ("sans ré-injection (planning seul)", "−0,001",
                                             "relire y_{t-1} n'apporte rien : la boucle d'état passe par h_t (pires : NPWD2551 0,28, NPWD2451 0,24)"),
        "sans_scheduled_sampling (teacher forcing seul)": ("sans scheduled sampling (teacher forcing seul)", "+0,012",
                                                           "apprendre sur ses propres sorties est ce qui tient le rêve 16 000 pas (pires : NPWD2551 0,36, NPWD2451 0,27)"),
        "fenetres_independantes (pas d'etat transmis)": ("fenêtres indépendantes (état remis à zéro, 20 époques)", "+0,010",
                                                         "sans état transmis, pas de régimes longs ; confondu avec 20 époques au lieu de 80 (pires : NPWD2551 0,33, NPWD2451 0,26)"),
        "sans_eclipses_evenements (evtf)": ("sans éclipses ni événements (evtf)", "+0,001",
                                            "l'ombre est déjà dans les angles solaires et les durées d'éclipse de ltdata (pires : NPWD2551 0,29, NPWD2451 0,24)"),
        "sans_soleil (saaf)": ("sans Soleil (saaf)", "+0,020",
                               "l'ingrédient n° 1 : NPWD2532 passe de 0,14 à 0,24, NPWD2451 de 0,24 à 0,33 (pires : NPWD2451 0,33, NPWD2551 0,30)"),
        "sans_tendances (ltdata)": ("sans tendances (ltdata : constante solaire, durées d'éclipse et d'occultation)", "+0,008",
                                    "la saison (constante solaire, durées d'éclipse) fixe le niveau de l'année (pires : NPWD2551 0,31, NPWD2451 0,28)"),
        "sans_pointage (ftl)": ("sans pointage (ftl)", "+0,010",
                                "l'attitude (quelle face au Soleil) : NPWD2551 passe de 0,28 à 0,36 (pires : NPWD2551 0,36, NPWD2451 0,25)"),
        "sans_comptes_commandes (dmop_stem_cnt)": ("sans comptes de commandes (dmop_stem_cnt : 45 features physiques)", "+0,002",
                                                   "gain petit mais net sur les lignes à deux états : NPWD2881 entre dans les pires (pires : NPWD2551 0,29, NPWD2451 0,23)"),
        "avec_interrupteurs_par_lettre (dmop_switch)": ("avec interrupteurs dmop par lettre (dmop_switch)", "+0,005",
                                                        "les lettres changent d'année : ce qu'elles font apprendre est perdu au test (pires : NPWD2551 0,31, NPWD2451 0,25)"),
        "avec_toutes_commandes_dmop (toutes les colonnes)": ("avec toutes les colonnes dmop", "+0,006",
                                                             "des centaines de colonnes de plus : NPWD2531 (0,19) et NPWD2851 (0,17) décrochent (pires : NPWD2551 0,27, NPWD2451 0,23)"),
        "avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre)": ("avec calendrier (distance Terre-Mars, angle Soleil-Mars-Terre)", "+0,009",
                                                                           "une horloge : le réseau mémorise l'année d'entraînement au lieu de la physique (pires : NPWD2551 0,34, NPWD2451 0,25)"),
    }
    gru_seul = abl["GRU seul (sans composant gradient boosting)"]["val_rmse"]
    ablations = []
    for cle, v in abl.items():
        assert cle in lib, cle
        ablations.append({"cle": cle, "libelle_fr": fr(lib[cle][0]), "rmse": r(v["val_rmse"], 5), "ecart_vs_gru": r(v["val_rmse"] - gru_seul, 5),
                          "ecart_rapport": lib[cle][1], "lecture_fr": fr(lib[cle][2]), "pires": {c: r(x, 5) for c, x in v["worst_circuits"].items()},
                          "systeme_complet": cle.startswith("complet"), "reference": cle.startswith("GRU seul")})
    data["ablations"] = ablations
    data["ablations_meta"] = {"gru_seul": r(gru_seul, 4), "gru_seul_precis": r(gru_seul, 5), "bruit_graines": 0.001, "graines": 1, "epoques": 80, "protocole": "strict",
                              "note": fr("ablations du GRU mesurées sur le GRU seul (0,0802, 1 graine) ; la première ligne situe le système complet")}
    assert abs(gru_seul - 0.0802) < 5e-5, gru_seul                     # chiffre publié (RAPPORT § 6)
    data["gru_seul"] = r(gru_seul, 4)

    # ------------------------------------------------------------------ pilote
    pil = dict(pilot)
    hist = pil.get("recherche_locale", {}).get("historique")
    if isinstance(hist, list) and len(hist) > 50:
        # sous-échantillonnage (un point sur k, plus le dernier) plutôt que suppression : la page trace cet historique
        k = -(-len(hist) // 49)
        keep = hist[::k] + ([hist[-1]] if (len(hist) - 1) % k else [])
        pil["recherche_locale"] = dict(pil["recherche_locale"], historique=keep, historique_sous_echantillonne=k)
    pil["libelles_fr"] = {"planning_reel_simule": fr("Réel (mesuré : 1 726,2)"),
                          "hors_comms_tout_EARTH": fr("Hors communication, tout en pointage EARTH (plus d'observation)"),
                          "hors_comms_tout_NADIR": fr("Hors communication, tout en NADIR (observation continue)"),
                          "hors_comms_sans_manoeuvres": fr("Hors communication, sans manœuvres (SLEW rendu à EARTH)"),
                          "recherche_locale": fr("Recherche locale : permutations d'heures de pointage dans chaque journée, 120 essais, 15 gardées")}
    data["pilote"] = pil

    # ------------------------------------------------------------------ tableau par circuit (RAPPORT § 4)
    data["par_circuit_table"] = [{"circuit": x["circuit"], "final": r(x["final"], 6), "gru_seul": r(x["gru_seul"], 6), "gbm": r(x["gbm"], 6),
                                  "std_reel": r(x["std_reel"], 6), "part_mse_pct": r(x["part_mse_pct"], 2)} for x in tab["par_circuit"][:8]]

    # ------------------------------------------------------------------ reproduire
    etapes = [
        {"titre": "Installation", "commandes": ["python3 -m venv .venv", ".venv/bin/pip install -r requirements.txt"],
         "note": fr("Python 3.9, CPU seulement ; le kit doit être dans kit_etudiants/ (annee1/, annee2/, make_validation.py, evaluate.py)")},
        {"titre": fr("Modèle final (annee1 + annee2, ~70 min sur 6 cœurs)"), "commandes": [".venv/bin/python train.py"], "note": fr("écrit les poids dans weights/")},
        {"titre": fr("Simuler une période de test au format de l'évaluation"),
         "commandes": [".venv/bin/python predict.py --test_dir DOSSIER --out predictions.csv [--std_out incertitude.csv]"],
         "note": fr("--std_out : incertitude calibrée, optionnelle")},
        {"titre": fr("Validation stricte (annee1 → annee2), conditions exactes de l'évaluation"),
         "commandes": [".venv/bin/python train.py --holdout",
                       "cd kit_etudiants && ../.venv/bin/python make_validation.py --annee annee2 --out ../validation && cd ..",
                       ".venv/bin/python predict.py --test_dir validation/test --out pred.csv --weights weights_holdout",
                       ".venv/bin/python kit_etudiants/evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv"],
         "note": fr("RMSE = 0,074968 ; train.py --holdout-reverse et train.py --holdout-split 0.5 sont les deux autres protocoles")},
        {"titre": fr("Tout reproduire (~4 h sur 6 cœurs)"), "commandes": ["./run_all.sh 6"],
         "note": fr("les trois validations, le flux officiel, la calibration, le modèle final, la démo, le pilote, les ablations, les tableaux du rapport, le diagnostic de NPWD2551 et les tests")},
        {"titre": "Tests", "commandes": [".venv/bin/python tests/test_features.py", ".venv/bin/python tests/test_predict_smoke.py"], "note": ""},
        {"titre": fr("Données de cette démo"), "commandes": [".venv/bin/python demo/build_demo_data.py"], "note": fr("régénère demo/data.js en ~30 s à partir des sources du dépôt")},
    ]
    data["reproduire"] = {"commandes": [c for e in etapes for c in e["commandes"]], "etapes": etapes}

    # ------------------------------------------------------------------ écriture + vérifications
    data = clean(data)
    txt = "window.DEMO_DATA = " + json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + ";\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    size_ko = OUT.stat().st_size / 1024

    # invariants
    assert set(data) >= {"meta", "periodes", "circuits", "brut", "fenetre14", "features", "entrainement", "reve", "derive", "resultats",
                         "incertitude", "decrochage", "trous_dmop", "lettres", "ablations", "pilote", "par_circuit_table", "reproduire"}
    assert len(data["circuits"]) == 33 and abs(sum(x["part_mse_pct"] for x in data["circuits"]) - 100) < 0.05
    assert abs(sum(x["part_const_mse_pct"] for x in data["circuits"]) - 100) < 0.05
    assert data["fenetre14"]["n"] == 336 == len(data["fenetre14"]["ev_umbra"]) == len(data["fenetre14"]["verite"]["NPWD2551"])
    assert data["reve"]["n"] == len(data["reve"]["total"]["pred"]) == len(data["reve"]["circuits"]["NPWD2551"]["verite"]) == n_grid // 6
    assert data["brut"]["n"] == len(data["brut"]["y"]) == len(data["brut"]["t_rel_s"]) and data["brut"]["horaire"]["n"] == 48
    assert len(data["derive"]["semaine"]) == len(data["derive"]["rmse_final"]) == len(data["derive"]["rmse_gbm_seul"]) == len(data["derive"]["rmse_constante"]) == 98
    assert data["decrochage"]["n"] == len(data["decrochage"]["verite_2551"]) == len(data["decrochage"]["dmop_par_heure"])
    assert len(data["incertitude"]["nuage"]["std"]) == 3000 and len(data["ablations"]) == 14 and len(data["par_circuit_table"]) == 8
    assert size_ko < 2500, size_ko
    print(f"{OUT} : {size_ko:.0f} Ko, clés : {', '.join(data)}  ({(dt.datetime.now() - t_start).seconds} s)")


if __name__ == "__main__":
    main()
