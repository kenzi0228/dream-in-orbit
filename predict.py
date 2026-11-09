"""
Simule la consommation horaire des 33 circuits sur une periode de test.

    python predict.py --test_dir DOSSIER --out predictions.csv [--weights weights] [--std_out incertitude.csv]

DOSSIER contient dmop.csv, evtf.csv, ftl.csv, saaf.csv, ltdata.csv, power_init.csv (7 premiers
jours) et heures_a_predire.csv. La sortie a une colonne ut_ms et les 33 colonnes NPWDxxxx,
une ligne par heure demandee, sans valeur manquante.

--std_out : ecart-type de l'ensemble, multiplie par les facteurs de calibration par circuit de
weights/calibration.json s'il existe (ajustes sur une validation, voir calibrate.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dio.features import build_features, hourly_grid, load_power_hourly
from dio.simulate import load_ensemble, simulate_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--std_out", default=None, help="ecart-type de l'ensemble (incertitude), meme format")
    ap.add_argument("--init-bias", type=float, default=None, help="surcharge la correction de niveau (defaut : celle de l'entrainement)")
    a = ap.parse_args()
    test = Path(a.test_dir)

    meta, models = load_ensemble(a.weights)
    heures = pd.read_csv(test / "heures_a_predire.csv")["ut_ms"].to_numpy(dtype=np.int64)
    init = load_power_hourly(test / "power_init.csv")
    grid = hourly_grid(min(init.index.min(), heures.min()), heures.max())
    n_init = int(np.searchsorted(grid, init.index.max(), side="right"))
    Xall = build_features(test, grid, meta["vocab"])
    Y_init = init.reindex(grid)[meta["targets"]].to_numpy()
    init_bias = a.init_bias if a.init_bias is not None else meta["args"].get("init_bias", 0.0)
    adapt_steps = meta["args"].get("adapt_steps", 0)
    adapt_lr = meta["args"].get("adapt_lr", 1e-4)

    mean, std, _ = simulate_ensemble(models, Xall, Y_init, n_init, init_bias=init_bias,
                                     adapt_steps=adapt_steps, adapt_lr=adapt_lr, gbm=meta.get("gbm"))
    pred = pd.DataFrame(mean, index=grid, columns=meta["targets"]).reindex(heures)
    pred = pred.ffill().bfill().fillna(0.0)
    pred.index.name = "ut_ms"
    pred.to_csv(a.out)
    if a.std_out:
        s = pd.DataFrame(std, index=grid, columns=meta["targets"]).reindex(heures).fillna(0.0)
        cal = Path(a.weights) / "calibration.json"
        if cal.exists():
            scale = json.loads(cal.read_text())["scale"]
            s = s * pd.Series(scale).reindex(s.columns).fillna(1.0)
        s.index.name = "ut_ms"
        s.to_csv(a.std_out)
    mode = {"stack": "GRU empile sur gradient boosting", "blend": "GRU + gradient boosting (melange)", "none": "GRU"}[meta.get("combine", "blend" if meta.get("gbm") else "none")]
    print(f"{len(pred)} heures predites avec {len(models)} {mode} -> {a.out}")


if __name__ == "__main__":
    main()
