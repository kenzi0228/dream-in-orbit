"""
Composant gradient boosting (sklearn, HistGradientBoostingRegressor, un modele par circuit).

Seul, il est moins bon que le GRU (il n'a pas de memoire), mais ses erreurs sont peu
correlees a celles du GRU (il lit les commandes dmop, que le GRU ne peut pas exploiter
sans sur-apprendre, et il a un biais inductif different) : melange avec un poids modere,
il reduit nettement la RMSE de simulation. Deterministe (random_state fixe).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

GBM_PARAMS = {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 50}


class GBMComponent:
    def __init__(self, features: list[str], params: dict | None = None):
        self.features = list(features)
        self.params = dict(params or GBM_PARAMS)
        self.models = []

    def fit(self, X: np.ndarray, Y: np.ndarray):
        mask = ~np.isnan(Y).any(axis=1)
        self.models = []
        for j in range(Y.shape[1]):
            m = HistGradientBoostingRegressor(random_state=0, **self.params)
            m.fit(X[mask], Y[mask, j])
            self.models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.stack([m.predict(X) for m in self.models], axis=1)
        return np.clip(P, 0.0, None)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"features": self.features, "params": self.params, "models": self.models}, f)

    @classmethod
    def load(cls, path: Path) -> "GBMComponent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        g = cls(d["features"], d["params"])
        g.models = d["models"]
        return g
