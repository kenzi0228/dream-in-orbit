"""
Composant gradient boosting (sklearn, HistGradientBoostingRegressor, un modele par circuit).

Seul, il est moins bon que le GRU en strict (il n'a pas de memoire), mais ses erreurs sont peu
correlees a celles du GRU (biais inductif different, lecture des commandes en escalier). Il sert
de base a l'empilement : ses predictions hors-echantillon (par sac) sont donnees au GRU, qui
apprend a les corriger avec sa memoire. Deterministe (random_state fixe).
"""
from __future__ import annotations

import hashlib
import json
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
    """Un HistGradientBoostingRegressor par circuit ; avec bags > 1, bagging temporel : `bags` jeux de modeles,
    chacun entraine sans un bloc contigu de 1/bags de la periode d'entrainement (une saison), moyennes.
    Chaque sac doit extrapoler la saison qu'il n'a pas vue ; la moyenne lisse ces extrapolations et
    sur-apprend moins l'annee d'entrainement (strict : 0,0796 contre 0,0822 pour un seul ajustement).
    Les sacs donnent aussi, gratuitement, des predictions hors-echantillon sur la periode d'entrainement
    (le bloc i est predit par le sac i, qui ne l'a pas vu) : c'est la base de l'empilement avec le GRU."""

    def __init__(self, features: list[str], params: dict | None = None, bags: int = 0):
        self.features = list(features)
        self.params = dict(params or GBM_PARAMS)
        self.bags = int(bags)
        self.models = []        # liste de jeux de modeles (un jeu = un modele par circuit)
        self.bounds = None      # bornes des blocs temporels des sacs (indices de la matrice d'entrainement)

    def _fit_set(self, X: np.ndarray, Y: np.ndarray) -> list:
        mask = ~np.isnan(Y).any(axis=1)
        return [HistGradientBoostingRegressor(random_state=0, **self.params).fit(X[mask], Y[mask, j]) for j in range(Y.shape[1])]

    @staticmethod
    def _predict_set(models: list, X: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(X) for m in models], axis=1)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n = len(X)
        if self.bags <= 1:
            self.models, self.bounds = [self._fit_set(X, Y)], [0, n]
            return self
        self.bounds = np.linspace(0, n, self.bags + 1).astype(int).tolist()
        self.models = []
        for i in range(self.bags):
            keep = np.r_[0:self.bounds[i], self.bounds[i + 1]:n]
            self.models.append(self._fit_set(X[keep], Y[keep]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.mean([self._predict_set(ms, X) for ms in self.models], axis=0)
        return np.clip(P, 0.0, None)

    def predict_oof(self, X: np.ndarray) -> np.ndarray:
        """Predictions hors-echantillon sur la matrice d'entrainement elle-meme : le bloc i est predit par le
        sac i, qui ne l'a pas vu. Le GRU empile apprend ainsi a corriger un GBM qui n'a pas vu l'heure,
        comme au test."""
        if self.bags <= 1 or self.bounds is None or self.bounds[-1] != len(X):
            raise ValueError("predict_oof : il faut au moins 2 sacs et la matrice d'entrainement d'origine")
        P = np.empty((len(X), len(self.models[0])), dtype=np.float64)
        for i, ms in enumerate(self.models):
            a, b = self.bounds[i], self.bounds[i + 1]
            P[a:b] = self._predict_set(ms, X[a:b])
        return np.clip(P, 0.0, None)

    def save(self, path: Path):
        # pickle compresse (zlib) : 5 sacs x 33 modeles pesent 206 Mo bruts, ~45 Mo compresses
        raw = pickle.dumps({"features": self.features, "params": self.params, "bags": self.bags,
                            "bounds": self.bounds, "models": self.models})
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
        g.bounds = d.get("bounds")
        return g

    @classmethod
    def fit_cached(cls, features: list[str], X: np.ndarray, Y: np.ndarray, params: dict | None = None, bags: int = 0,
                   cache_dir: Path = Path("cache"), key_extra: str = "") -> "GBMComponent":
        """Ajuste, ou recharge depuis `cache_dir` un ajustement identique (memes features, hyperparametres,
        sacs et matrice d'entrainement : la cle hache les donnees). Les relances et les ablations du GRU
        ne reajustent pas le GBM."""
        h = hashlib.md5()
        h.update(json.dumps([list(features), dict(params or GBM_PARAMS), int(bags), key_extra], sort_keys=True).encode())
        h.update(str(X.shape).encode())
        h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(np.nan_to_num(Y, nan=-1.0), dtype=np.float32).tobytes())
        f = Path(cache_dir) / f"gbm_{h.hexdigest()[:12]}.pkl"
        if f.exists():
            return cls.load(f)
        g = cls(features, params, bags).fit(X, Y)
        Path(cache_dir).mkdir(exist_ok=True)
        g.save(f)
        return g
