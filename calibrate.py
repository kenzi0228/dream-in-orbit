"""
Calibration de l'incertitude. L'ecart-type de l'ensemble (Deep Ensembles) classe bien les heures incertaines
mais il est tres sous-disperse : il ne mesure que le desaccord entre graines, pas l'erreur systematique.
On ajuste donc, par circuit, un facteur d'echelle s_c = RMSE_c / RMS(ecart-type_c) sur un dossier de
validation, puis on verifie sur d'autres protocoles (autres annees de test) que les intervalles calibres
couvrent ce qu'ils annoncent (68 % a 1 ecart-type, 95 % a 2 sous l'hypothese gaussienne).

    python calibrate.py --fit weights_holdout_reverse --check weights_holdout,weights_holdout_split --out weights/calibration.json

predict.py --std_out applique les facteurs de weights/calibration.json s'il existe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dio.data import load_period

PERIOD = {"strict": "annee2", "reverse": "annee1", "split": "annee2"}


def load_val(wdir: Path, kit: Path):
    meta = json.loads((wdir / "meta.json").read_text())
    n_init = meta["n_init"]
    pred = pd.read_csv(wdir / "val_pred.csv", index_col="ut_ms")
    std = pd.read_csv(wdir / "val_std.csv", index_col="ut_ms")
    _, truth = load_period(kit / PERIOD[meta["protocol"]], meta["vocab"])
    truth = truth.reindex(pred.index)[pred.columns]
    return pred.iloc[n_init:], std.iloc[n_init:], truth.iloc[n_init:]


def fit_scales(pred, std, truth):
    err, s = (pred - truth).to_numpy(), std.to_numpy()
    m = np.isfinite(err)
    scale = {}
    for j, c in enumerate(pred.columns):
        mj = m[:, j]
        scale[c] = float(np.sqrt(np.mean(err[mj, j] ** 2)) / max(np.sqrt(np.mean(s[mj, j] ** 2)), 1e-9))
    g = float(np.sqrt(np.mean(err[m] ** 2)) / np.sqrt(np.mean(s[m] ** 2)))
    return scale, g


def check(pred, std, truth, scale):
    err = np.abs((pred - truth).to_numpy())
    s = std.to_numpy() * np.array([scale.get(c, 1.0) for c in pred.columns])[None]
    m = np.isfinite(err)
    e, s = err[m], s[m]
    return {"couverture_1s": float(np.mean(e <= s)), "couverture_2s": float(np.mean(e <= 2 * s)),
            "rmse_sur_rms_std": float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(s ** 2))),
            "spearman_std_vs_abs_err": float(pd.Series(s).corr(pd.Series(e), method="spearman"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="weights_holdout_reverse", help="dossier(s) de validation (virgules) sur lesquels ajuster les facteurs")
    ap.add_argument("--check", default="weights_holdout,weights_holdout_split", help="dossiers de validation ou verifier la couverture")
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--out", default="weights/calibration.json")
    ap.add_argument("--report", default="reports/calibration.json")
    a = ap.parse_args()
    kit = Path(a.kit)
    fits = [x for x in a.fit.split(",") if x]
    per_fit = []
    for d in fits:
        pred, std, truth = load_val(Path(d), kit)
        per_fit.append((d, pred, std, truth, *fit_scales(pred, std, truth)))
    # plusieurs dossiers : moyenne geometrique des facteurs (deux annees de test valent mieux qu'une)
    scale = {c: float(np.exp(np.mean([np.log(max(sc[c], 1e-9)) for _, _, _, _, sc, _ in per_fit]))) for c in per_fit[0][1].columns}
    g = float(np.exp(np.mean([np.log(gg) for *_, gg in per_fit])))
    res = {"fit_on": a.fit, "global": g, "scale": scale, "fit_check": {d: check(p_, s_, t_, scale) for d, p_, s_, t_, _, _ in per_fit}, "check": {}}
    for d in [x for x in a.check.split(",") if x and Path(x, "val_pred.csv").exists()]:
        p2, s2, t2 = load_val(Path(d), kit)
        res["check"][d] = {"brut": check(p2, s2, t2, {}), "calibre": check(p2, s2, t2, scale)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"fit_on": a.fit, "global": g, "scale": scale}, indent=1))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scale"}, indent=1))


if __name__ == "__main__":
    main()
