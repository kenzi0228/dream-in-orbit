"""
Le world model : un GRU qui, a chaque heure, lit le planning / contexte de l'heure
(actions + contexte) et sa propre estimation de l'etat precedent (consommation de
l'heure d'avant), et predit la consommation de l'heure courante.

    etat_{t-1} + actions_t + contexte_t  --(GRU, memoire h_t)-->  etat_t

Deux modes :
    - teacher forcing : l'etat precedent injecte est le vrai (entrainement rapide,
      sequence complete en un appel GRU) ;
    - rollout / reve : apres `n_init` heures reelles, le modele re-injecte sa
      propre prediction (c'est la situation de l'evaluation : 7 jours reels puis
      simulation libre). Pendant l'entrainement on melange les deux avec une
      probabilite p_tf (scheduled sampling, Bengio et al. 2015).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, n_feat: int, n_out: int, hidden: int = 192, layers: int = 2,
                 feedback: bool = True, dropout: float = 0.1, feat_dropout: float = 0.05):
        super().__init__()
        self.n_feat, self.n_out, self.feedback = n_feat, n_out, feedback
        n_in = n_feat + (n_out if feedback else 0)
        self.feat_drop = nn.Dropout(feat_dropout)
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def _step_input(self, x_t: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        x_t = self.feat_drop(x_t)
        return torch.cat([x_t, y_prev], dim=-1) if self.feedback else x_t

    def forward_tf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Teacher forcing complet. x : (B, T, F), y : (B, T, C) standardises. Retourne (B, T, C)."""
        y_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        inp = self.inp(self._step_input(x, y_prev))
        out, _ = self.gru(inp)
        return self.head(out)

    def rollout(self, x: torch.Tensor, y: torch.Tensor | None, n_init: int,
                p_tf: float = 0.0, noise: float = 0.0, detach: bool = False,
                state=None, return_state: bool = False):
        """
        Simulation pas a pas. Les `n_init` premieres heures sont reelles (y connu) ;
        ensuite le modele reutilise sa prediction (sauf avec probabilite p_tf, ou la
        verite est re-injectee : scheduled sampling). `noise` ajoute un bruit gaussien
        (en unites standardisees) a l'etat re-injecte pour apprendre a corriger ses erreurs.
        `state` = (h, y_prev) permet de poursuivre une simulation (entrainement stateful,
        BPTT tronquee : l'etat est transmis d'un segment au suivant).
        """
        B, T, _ = x.shape
        if state is None:
            h, y_prev = None, torch.zeros(B, self.n_out, device=x.device, dtype=x.dtype)
        else:
            h, y_prev = state
        preds = []
        for t in range(T):
            inp = self.inp(self._step_input(x[:, t], y_prev)).unsqueeze(1)
            out, h = self.gru(inp, h)
            p = self.head(out[:, 0])
            preds.append(p)
            if y is not None and t < n_init:
                y_prev = y[:, t]
            else:
                y_prev = p.detach() if detach else p
                if y is not None and p_tf > 0:
                    use_truth = (torch.rand(B, 1, device=x.device) < p_tf).to(x.dtype)
                    y_prev = use_truth * y[:, t] + (1 - use_truth) * y_prev
            if noise > 0 and self.training:
                y_prev = y_prev + noise * torch.randn_like(y_prev)
        preds = torch.stack(preds, dim=1)
        if return_state:
            return preds, (h, y_prev)
        return preds


class Normalizer:
    """Standardisation des features et des cibles (statistiques du train, stockees dans le checkpoint)."""

    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean, self.x_std = torch.as_tensor(x_mean, dtype=torch.float32), torch.as_tensor(x_std, dtype=torch.float32)
        self.y_mean, self.y_std = torch.as_tensor(y_mean, dtype=torch.float32), torch.as_tensor(y_std, dtype=torch.float32)

    X_CLIP = 10.0  # une feature standardisee ne depasse jamais +/- 10 (robustesse aux valeurs jamais vues)

    @classmethod
    def fit(cls, X, Y):
        x_mean, x_std = X.mean(axis=0), X.std(axis=0)
        # une feature quasi constante dans le train (ex. mode de pointage absent) garde son echelle brute
        x_std = np.where(x_std < 1e-3, 1.0, x_std)
        y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-6
        return cls(x_mean, x_std, y_mean, y_std)

    def x(self, X):
        xs = (torch.as_tensor(X, dtype=torch.float32) - self.x_mean) / self.x_std
        return xs.clamp(-self.X_CLIP, self.X_CLIP)

    def y(self, Y):
        return (torch.as_tensor(Y, dtype=torch.float32) - self.y_mean) / self.y_std

    def y_inv(self, Ys):
        return Ys * self.y_std + self.y_mean

    def state_dict(self):
        return {"x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["x_mean"], d["x_std"], d["y_mean"], d["y_std"])


class MLPModel(WorldModel):
    """Ablation 'sans memoire' : meme entree, meme tete, mais pas de GRU (chaque heure est independante)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gru = None
        self.mlp = nn.Sequential(nn.Linear(k["hidden"], k["hidden"]), nn.GELU())

    def forward_tf(self, x, y):
        y_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        return self.head(self.mlp(self.inp(self._step_input(x, y_prev))))

    def rollout(self, x, y, n_init, p_tf=0.0, noise=0.0, detach=False, state=None, return_state=False):
        B, T, _ = x.shape
        y_prev = torch.zeros(B, self.n_out, device=x.device, dtype=x.dtype) if state is None else state[1]
        preds = []
        for t in range(T):
            p = self.head(self.mlp(self.inp(self._step_input(x[:, t], y_prev))))
            preds.append(p)
            if t + 1 >= T:
                break
            if y is not None and t < n_init:
                y_prev = y[:, t]
            else:
                y_prev = p.detach() if detach else p
                if y is not None and p_tf > 0:
                    use_truth = (torch.rand(B, 1, device=x.device) < p_tf).to(x.dtype)
                    y_prev = use_truth * y[:, t] + (1 - use_truth) * y_prev
            if noise > 0 and self.training:
                y_prev = y_prev + noise * torch.randn_like(y_prev)
        preds = torch.stack(preds, dim=1)
        if return_state:
            return preds, (None, y_prev)
        return preds
