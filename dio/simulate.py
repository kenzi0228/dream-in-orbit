"""
Simulation ("reve") d'une periode complete avec un ensemble de world models,
a partir des features horaires et des premieres heures reelles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import MLPModel, Normalizer, WorldModel


def load_ensemble(weights_dir: Path):
    weights_dir = Path(weights_dir)
    meta = json.loads((weights_dir / "meta.json").read_text())
    models = []
    for f in sorted(weights_dir.glob("model_*.pt")):
        ck = torch.load(f, map_location="cpu")
        cfg = ck["config"]
        cls = MLPModel if cfg.get("no_memory") else WorldModel
        m = cls(n_feat=len(meta["features"]), n_out=len(meta["targets"]), hidden=cfg["hidden"],
                       layers=cfg["layers"], feedback=cfg["feedback"], dropout=cfg["dropout"],
                       feat_dropout=cfg.get("feat_dropout", 0.0))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append((m, Normalizer.from_state_dict(ck["normalizer"])))
    return meta, models


@torch.no_grad()
def simulate_one(model: WorldModel, norm: Normalizer, X: np.ndarray, Y_init: np.ndarray | None,
                 n_init: int, init_bias: float = 0.0) -> np.ndarray:
    """
    X : (T, F) features brutes ; Y_init : (T, C) verite brute (NaN la ou inconnue, seules les
    n_init premieres lignes sont utilisees). Retourne (T, C) predictions brutes.
    init_bias : fraction du residu moyen observe sur la fenetre initiale ajoutee a la suite
    (correction de niveau par circuit a partir des 7 jours reels).
    """
    model.eval()
    xs = norm.x(X).unsqueeze(0)
    ys = None
    if Y_init is not None and n_init > 0:
        Yf = pd.DataFrame(Y_init).ffill().bfill().fillna(0.0).to_numpy()
        ys = norm.y(Yf).unsqueeze(0)
    pred = model.rollout(xs, ys, n_init=n_init)
    pred = norm.y_inv(pred)[0].numpy()
    if init_bias > 0 and Y_init is not None and n_init > 0:
        resid = np.nanmean(Y_init[:n_init] - pred[:n_init], axis=0)
        resid = np.nan_to_num(resid)
        pred[n_init:] += init_bias * resid
    return np.clip(pred, 0.0, None)


def simulate_ensemble(models, X: np.ndarray, Y_init, n_init: int, init_bias: float = 0.0):
    """Moyenne et ecart-type de l'ensemble : (T, C), (T, C)."""
    preds = np.stack([simulate_one(m, n, X, Y_init, n_init, init_bias) for m, n in models])
    return preds.mean(axis=0), preds.std(axis=0), preds
