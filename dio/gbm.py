"""
Composant gradient boosting (sklearn, HistGradientBoostingRegressor, un modele par circuit).

Seul, il est moins bon que le GRU (il n'a pas de memoire), mais ses erreurs sont peu
correlees a celles du GRU (il lit les commandes dmop, que le GRU ne peut pas exploiter
sans sur-apprendre, et il a un biais inductif different) : melange avec un poids modere,
il reduit nettement la RMSE de simulation. Deterministe (random_state fixe).
"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# 600 arbres lents et plus regularises : 0,0820 / 0,0825 (strict / chrono) contre 0,0822 / 0,0829 pour
# 400 arbres a 0,03, l2 1,0, 50 par feuille (reports/gbm_p600.json). 7 feuilles generalisent mieux d'une annee
# a l'autre (0,0803) mais moins bien a lettres stables (0,0849) ; 31 feuilles l'inverse : 15 est le compromis.
GBM_PARAMS = {"max_iter": 600, "learning_rate": 0.02, "max_leaf_nodes": 15, "l2_regularization": 3.0, "min_samples_leaf": 100}


class GBMComponent:
    """Un HistGradientBoostingRegressor par circuit ; avec bags > 0, bagging temporel : `bags` jeux de modeles,
    chacun entraine sans un bloc contigu de 1/bags de la periode d'entrainement (une saison), moyennes.
    Chaque sac doit extrapoler la saison qu'il n'a pas vue ; la moyenne lisse ces extrapolations et
    sur-apprend moins l'annee d'entrainement (strict : 0,0796 contre 0,0822 pour un seul ajustement)."""

    def __init__(self, features: list[str], params: dict | None = None, bags: int = 0):
        self.features = list(features)
        self.params = dict(params or GBM_PARAMS)
        self.bags = int(bags)
        self.models = []        # liste de jeux de modeles (un jeu = un modele par circuit)

    def _fit_set(self, X: np.ndarray, Y: np.ndarray) -> list:
        mask = ~np.isnan(Y).any(axis=1)
        return [HistGradientBoostingRegressor(random_state=0, **self.params).fit(X[mask], Y[mask, j]) for j in range(Y.shape[1])]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if self.bags <= 1:
            self.models = [self._fit_set(X, Y)]
            return self
        n = len(X)
        bounds = np.linspace(0, n, self.bags + 1).astype(int)
        self.models = []
        for i in range(self.bags):
            keep = np.r_[0:bounds[i], bounds[i + 1]:n]
            self.models.append(self._fit_set(X[keep], Y[keep]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.mean([np.stack([m.predict(X) for m in ms], axis=1) for ms in self.models], axis=0)
        return np.clip(P, 0.0, None)

    def save(self, path: Path):
        # pickle compresse (zlib) : 5 sacs x 33 modeles pesent 206 Mo bruts, ~35 Mo compresses
        raw = pickle.dumps({"features": self.features, "params": self.params, "bags": self.bags, "models": self.models})
        Path(path).write_bytes(zlib.compress(raw, 6))

    @classmethod
    def load(cls, path: Path) -> "GBMComponent":
        raw = Path(path).read_bytes()
        if raw[:1] == b"\x78":          # en-tete zlib ; sinon ancien pickle non compresse
            raw = zlib.decompress(raw)
        d = pickle.loads(raw)
        g = cls(d["features"], d["params"], d.get("bags", 0))
        models = d["models"]
        # anciens fichiers : une liste plate de modeles (un par circuit)
        g.models = models if models and isinstance(models[0], list) else [models]
        return g
