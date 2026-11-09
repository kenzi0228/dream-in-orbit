"""Test de bout en bout : train.py en mode fumee (quelques epoques, GBM minuscule), puis predict.py sur
validation/test au format officiel ; verifie le format de la sortie et la calibration optionnelle.

    python tests/test_predict_smoke.py        (environ 3 minutes ; construit validation/ s'il manque)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*cmd):
    subprocess.run([PY, *cmd], check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def test_train_predict_smoke():
    out = ROOT / "weights_smoke"
    if not (ROOT / "validation" / "test" / "heures_a_predire.csv").exists():
        subprocess.run([PY, "make_validation.py", "--annee", "annee2", "--out", "../validation"], check=True, cwd=ROOT / "kit_etudiants")
    run("train.py", "--holdout", "--seeds", "2", "--epochs", "2", "--tf-epochs", "1", "--max-train-hours", "6000",
        "--gbm-bags", "2", "--gbm-params", '{"max_iter": 5}', "--no-gbm-cache", "--eval-every", "100", "--out", str(out))
    (out / "calibration.json").write_text(json.dumps({"scale": {"NPWD2551": 2.0}}))
    run("predict.py", "--test_dir", "validation/test", "--out", str(out / "pred.csv"), "--weights", str(out), "--std_out", str(out / "std.csv"))
    pred = pd.read_csv(out / "pred.csv")
    std = pd.read_csv(out / "std.csv")
    heures = pd.read_csv(ROOT / "validation" / "test" / "heures_a_predire.csv")
    assert list(pred["ut_ms"]) == list(heures["ut_ms"]), "une ligne par heure demandee, dans l'ordre"
    assert pred.shape[1] == 34 and all(c.startswith("NPWD") for c in pred.columns[1:]), "ut_ms + 33 circuits"
    assert not pred.isna().any().any() and (pred.iloc[:, 1:] >= 0).all().all(), "pas de NaN, pas de negatif"
    assert std.shape == pred.shape and (std.iloc[:, 1:] >= 0).all().all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["protocol"] == "strict" and meta["vocab_years"] == ["annee1"], "vocabulaire appris sur l'entrainement seul"
    print("ok test_train_predict_smoke", pred.shape)


if __name__ == "__main__":
    test_train_predict_smoke()
