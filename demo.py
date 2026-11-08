"""
Demo "reve contre realite" sur la periode de validation (annee2), avec l'incertitude de
l'ensemble, et courbe de derive (RMSE par semaine depuis le debut du reve).

    python demo.py --weights weights_holdout --out reports/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dio.data import get_vocab, load_period
from dio.metrics import rmse, rmse_by_horizon, rmse_per_circuit

N_INIT = 24 * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_holdout")
    ap.add_argument("--kit", default="kit_etudiants")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--baseline-pred", default="reports/baselines.pred.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Path(a.weights)
    pred = pd.read_csv(w / "val_pred.csv", index_col="ut_ms")
    std = pd.read_csv(w / "val_std.csv", index_col="ut_ms")
    vocab = get_vocab(Path(a.kit), ["annee1", "annee2"])
    _, truth = load_period(Path(a.kit) / "annee2", vocab)
    truth = truth.reindex(pred.index)
    t = pd.to_datetime(pred.index, unit="ms")
    targets = list(truth.columns)
    P, T, S = pred.to_numpy(), truth.to_numpy(), std.to_numpy()
    total = rmse(P[N_INIT:], T[N_INIT:])

    # 1) reve vs realite : total + les 4 circuits qui pesent le plus dans l'erreur
    per = rmse_per_circuit(P[N_INIT:], T[N_INIT:])
    top = [targets[i] for i in np.argsort(-per)[:4]]
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
    daily = lambda arr: pd.Series(arr, index=t).resample("D").mean()
    ax = axes[0]
    ax.plot(daily(np.nansum(T, axis=1)), color="black", lw=1, label="realite (somme des 33 circuits)")
    ax.plot(daily(P.sum(axis=1)), color="tab:red", lw=1, label="reve (ensemble)")
    s_tot = daily(np.sqrt((S ** 2).sum(axis=1)))
    ax.fill_between(s_tot.index, (daily(P.sum(axis=1)) - 2 * s_tot).values, (daily(P.sum(axis=1)) + 2 * s_tot).values, color="tab:red", alpha=0.2, label="±2 ecarts-types de l'ensemble")
    ax.axvline(t[N_INIT], color="gray", ls="--")
    ax.set_title(f"Reve contre realite (moyennes journalieres) — annee2, RMSE horaire = {total:.4f}")
    ax.legend(loc="upper right")
    for ax, c in zip(axes[1:], top):
        j = targets.index(c)
        ax.plot(daily(T[:, j]), color="black", lw=1, label="realite")
        ax.plot(daily(P[:, j]), color="tab:red", lw=1, label="reve")
        sj = daily(S[:, j])
        ax.fill_between(sj.index, (daily(P[:, j]) - 2 * sj).values, (daily(P[:, j]) + 2 * sj).values, color="tab:red", alpha=0.2)
        ax.axvline(t[N_INIT], color="gray", ls="--")
        ax.set_title(f"{c} — RMSE {per[j]:.4f}")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "reve_vs_realite.png", dpi=110)

    # 2) zoom horaire sur 2 semaines au milieu de la periode, circuit dominant
    j = targets.index(top[0])
    a0 = N_INIT + 24 * 95
    sl = slice(a0, a0 + 24 * 14)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t[sl], T[sl, j], color="black", lw=1, label="realite horaire")
    ax.plot(t[sl], P[sl, j], color="tab:red", lw=1, label="reve")
    ax.fill_between(t[sl], P[sl, j] - 2 * S[sl, j], P[sl, j] + 2 * S[sl, j], color="tab:red", alpha=0.2)
    ax.set_title(f"{top[0]} — zoom horaire sur 2 semaines (jour 95 du reve, saison d'eclipses)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "zoom_horaire.png", dpi=110)

    # 3) derive : RMSE par semaine depuis le debut du reve, contre la constante et le GBM
    weeks = rmse_by_horizon(P[N_INIT:], T[N_INIT:])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([w for w, _ in weeks], [r for _, r in weeks], marker="o", ms=3, label="world model (ensemble)")
    const = np.tile(np.nanmean(T[:N_INIT], axis=0), (len(T) - N_INIT, 1))
    wc = rmse_by_horizon(const, T[N_INIT:])
    ax.plot([w for w, _ in wc], [r for _, r in wc], color="gray", ls="--", label="constante (7 premiers jours)")
    if Path(a.baseline_pred).exists():
        gb = pd.read_csv(a.baseline_pred, index_col="ut_ms").reindex(pred.index[N_INIT:]).to_numpy()
        wg = rmse_by_horizon(gb, T[N_INIT:])
        ax.plot([w for w, _ in wg], [r for _, r in wg], color="tab:green", ls=":", label="gradient boosting (sans memoire)")
    ax.set_xlabel("semaines depuis le debut du reve")
    ax.set_ylabel("RMSE de la semaine")
    ax.set_title("Courbe de derive : l'erreur n'explose pas avec l'horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "derive.png", dpi=110)

    # 4) erreurs par circuit
    fig, ax = plt.subplots(figsize=(14, 4))
    order = np.argsort(-per)
    ax.bar([targets[i] for i in order], per[order], color="tab:red", label="world model")
    cper = rmse_per_circuit(const, T[N_INIT:])
    ax.bar([targets[i] for i in order], cper[order], fill=False, edgecolor="gray", label="constante (7 jours)")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([targets[i] for i in order], rotation=90)
    ax.set_ylabel("RMSE")
    ax.set_title("Erreur par circuit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "erreur_par_circuit.png", dpi=110)

    # 5) calibration de l'incertitude : erreur absolue vs ecart-type de l'ensemble
    e = np.abs(P[N_INIT:] - T[N_INIT:]).ravel()
    s = S[N_INIT:].ravel()
    m = np.isfinite(e)
    e, s = e[m], s[m]
    bins = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, s, side="right") - 1, 0, 9)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([s[idx == k].mean() for k in range(10)], [np.sqrt((e[idx == k] ** 2).mean()) for k in range(10)], marker="o")
    ax.plot([0, s.max()], [0, s.max()], color="gray", ls="--")
    ax.set_xlabel("ecart-type de l'ensemble (decile)")
    ax.set_ylabel("RMSE observe")
    ax.set_title("L'incertitude de l'ensemble suit-elle l'erreur ?")
    fig.tight_layout()
    fig.savefig(out / "calibration_incertitude.png", dpi=110)

    summary = {"val_rmse": total, "rmse_by_week": weeks, "top_circuits": top,
               "spearman_std_vs_abs_err": float(pd.Series(s).corr(pd.Series(e), method="spearman"))}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rmse_by_week"}, indent=1))


if __name__ == "__main__":
    main()
