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
    """Charge les GRU (model_*.pt) et, s'il existe, le composant gradient boosting (gbm.pkl).
    Retourne (meta, [(modele, normalizer, poids), ...]) ; meta["gbm"] = (composant, poids) ou None."""
    weights_dir = Path(weights_dir)
    meta = json.loads((weights_dir / "meta.json").read_text())
    models = []
    mw = meta.get("model_weights", {})
    for f in sorted(weights_dir.glob("model_*.pt")):
        ck = torch.load(f, map_location="cpu")
        cfg = ck["config"]
        feats = ck.get("features", meta["features"])  # chaque modele porte son jeu de features
        cls = MLPModel if cfg.get("no_memory") else WorldModel
        m = cls(n_feat=len(feats), n_out=len(meta["targets"]), hidden=cfg["hidden"],
                       layers=cfg["layers"], feedback=cfg["feedback"], dropout=cfg["dropout"],
                       feat_dropout=cfg.get("feat_dropout", 0.0), cell=cfg.get("cell", "gru"))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append((m, Normalizer.from_state_dict(ck["normalizer"]), float(mw.get(f.name, 1.0)), list(feats)))
    meta["gbm"] = None
    if meta.get("gbm_weight", 0) > 0 and (weights_dir / "gbm.pkl").exists():
        from .gbm import GBMComponent
        meta["gbm"] = (GBMComponent.load(weights_dir / "gbm.pkl"), float(meta["gbm_weight"]))
    return meta, models


def adapt_on_init(model: WorldModel, norm: Normalizer, xs: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor,
                  n_init: int, steps: int, lr: float) -> WorldModel:
    """
    Adaptation au moment du test : quelques pas de gradient (teacher forcing) sur les n_init
    heures reelles fournies, pour caler le modele sur le regime courant du satellite.
    Deterministe (graine fixe, pas de dropout) ; le modele d'origine n'est pas modifie.
    """
    import copy
    torch.manual_seed(0)
    m = copy.deepcopy(model)
    m.eval()  # pas de dropout : l'adaptation est deterministe
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xb, yb, mb = xs[:, :n_init], ys[:, :n_init], mask[:n_init]
    y_std = norm.y_std
    with torch.enable_grad():
        for _ in range(steps):
            pred = m.forward_tf(xb, yb)
            err = (((pred - yb) * y_std) ** 2) * mb.view(1, -1, 1)
            loss = err.sum() / (mb.sum() * yb.shape[-1] + 1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m


@torch.no_grad()
def simulate_one(model: WorldModel, norm: Normalizer, X: np.ndarray, Y_init: np.ndarray | None,
                 n_init: int, init_bias: float = 0.0, adapt_steps: int = 0, adapt_lr: float = 1e-4) -> np.ndarray:
    """
    X : (T, F) features brutes ; Y_init : (T, C) verite brute (NaN la ou inconnue, seules les
    n_init premieres lignes sont utilisees). Retourne (T, C) predictions brutes.
    init_bias : fraction du residu moyen observe sur la fenetre initiale ajoutee a la suite
    (correction de niveau par circuit a partir des 7 jours reels).
    adapt_steps : pas d'adaptation (fine-tuning) sur la fenetre initiale avant le reve.
    """
    model.eval()
    xs = norm.x(X).unsqueeze(0)
    ys = None
    if Y_init is not None and n_init > 0:
        Yf = pd.DataFrame(Y_init).ffill().bfill().fillna(0.0).to_numpy()
        ys = norm.y(Yf).unsqueeze(0)
        if adapt_steps > 0:
            mask = torch.as_tensor(~np.isnan(Y_init).any(axis=1), dtype=torch.float32)
            model = adapt_on_init(model, norm, xs, ys, mask, n_init, adapt_steps, adapt_lr)
    pred = model.rollout(xs, ys, n_init=n_init)
    pred = norm.y_inv(pred)[0].numpy()
    if init_bias > 0 and Y_init is not None and n_init > 0:
        resid = np.nanmean(Y_init[:n_init] - pred[:n_init], axis=0)
        resid = np.nan_to_num(resid)
        pred[n_init:] += init_bias * resid
    return np.clip(pred, 0.0, None)


def simulate_ensemble(models, Xdf: pd.DataFrame, Y_init, n_init: int, init_bias: float = 0.0,
                      adapt_steps: int = 0, adapt_lr: float = 1e-4, gbm=None):
    """
    Moyenne ponderee et ecart-type de l'ensemble de GRU : (T, C), (T, C), puis melange avec le
    composant gradient boosting : prediction = (1 - w) * GRU + w * GBM.
    Xdf : DataFrame de toutes les features horaires (chaque modele y prend ses colonnes).
    models : [(modele, normalizer, poids, features)] ; gbm : (GBMComponent, w) ou None.
    """
    preds = np.stack([simulate_one(m, n, Xdf[feats].to_numpy(), Y_init, n_init, init_bias, adapt_steps, adapt_lr)
                      for m, n, _, feats in models])
    w = np.array([wt for _, _, wt, _ in models], dtype=np.float64)
    w = w / w.sum()
    mean = np.tensordot(w, preds, axes=1)
    std = np.sqrt(np.tensordot(w, (preds - mean) ** 2, axes=1))
    if gbm is not None:
        comp, wg = gbm
        mean = (1 - wg) * mean + wg * comp.predict(Xdf[comp.features].to_numpy())
    return mean, std, preds
