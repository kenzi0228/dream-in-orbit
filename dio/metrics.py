"""RMSE identique a evaluate.py (toutes les cellules ou la verite existe)."""
import numpy as np


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    m = ~np.isnan(truth)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)))


def rmse_per_circuit(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = []
    for j in range(truth.shape[1]):
        m = ~np.isnan(truth[:, j])
        out.append(float(np.sqrt(np.mean((pred[m, j] - truth[m, j]) ** 2))) if m.any() else np.nan)
    return np.array(out)


def rmse_by_horizon(pred: np.ndarray, truth: np.ndarray, step: int = 24 * 7) -> list[tuple[int, float]]:
    """RMSE par tranche d'horizon (par defaut par semaine depuis le debut du reve) : courbe de derive."""
    out = []
    for a in range(0, len(truth), step):
        p, t = pred[a:a + step], truth[a:a + step]
        if np.isfinite(t).any():
            out.append((a // step, rmse(p, t)))
    return out
