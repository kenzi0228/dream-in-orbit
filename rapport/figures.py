"""
Figures du rapport LaTeX, écrites dans rapport/figures/ (PDF vectoriel + aperçu PNG).

Mêmes entrées que demo.py (prédictions et écarts-types de validation de weights_holdout, vérité du
kit, facteurs de calibration ajustés sur l'autre année de test, prédictions de la référence
gradient boosting de baselines.py), donc mêmes chiffres ; mais sans titre interne (les légendes
LaTeX les portent), avec des libellés accentués, la virgule décimale, des tailles de police
adaptées à la largeur du texte, les bandes d'incertitude tronquées à zéro, les semaines de la
courbe de dérive numérotées à partir de 1 (comme dans le texte) et les axes de la figure de
calibration limités à l'étendue des déciles.

    .venv/bin/python rapport/figures.py        # depuis la racine du dépôt (~10 s)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio.data import load_period  # noqa: E402
from dio.metrics import rmse, rmse_by_horizon, rmse_per_circuit  # noqa: E402

N_INIT = 24 * 7
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
ROUGE, NOIR, GRIS, VERT, BLEU = "#c0392b", "black", "#7f7f7f", "#2a9d5c", "#1f5f9f"


def virgule(x, _=None):
    """Nombre avec virgule décimale et vrai signe moins."""
    return f"{x:g}".replace("-", "−").replace(".", ",")


def mois_annee(x, _=None):
    d = mdates.num2date(x)
    return f"{MOIS[d.month - 1]}\n{d.year}"


def jour_mois(x, _=None):
    d = mdates.num2date(x)
    return f"{d.day} {MOIS[d.month - 1]}"


def dec(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 0.8, "axes.linewidth": 0.6,
        "legend.framealpha": 0.9, "legend.edgecolor": "0.8", "pdf.fonttype": 42, "savefig.dpi": 200,
    })


def sauver(fig, out: Path, nom: str):
    fig.savefig(out / f"{nom}.pdf")
    fig.savefig(out / f"{nom}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(RACINE / "weights_holdout"))
    ap.add_argument("--kit", default=str(RACINE / "kit_etudiants"))
    ap.add_argument("--calibration", default=str(RACINE / "reports" / "calibration_reverse.json"),
                    help="facteurs par circuit appliqués à l'écart-type (ajustés sur l'autre année de test)")
    ap.add_argument("--baseline-pred", default=str(RACINE / "reports" / "baselines.pred.csv"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    w = Path(a.weights)
    meta = json.loads((w / "meta.json").read_text())
    pred = pd.read_csv(w / "val_pred.csv", index_col="ut_ms")
    std = pd.read_csv(w / "val_std.csv", index_col="ut_ms")
    period = {"strict": "annee2", "reverse": "annee1", "split": "annee2"}[meta.get("protocol", "strict")]
    _, truth = load_period(Path(a.kit) / period, meta["vocab"])
    truth = truth.reindex(pred.index)[pred.columns]
    cal = json.loads(Path(a.calibration).read_text())
    std = std * pd.Series(cal["scale"]).reindex(std.columns).fillna(1.0)
    t = pd.to_datetime(pred.index, unit="ms")
    targets = list(truth.columns)
    P, T, S = pred.to_numpy(), truth.to_numpy(), std.to_numpy()
    total = rmse(P[N_INIT:], T[N_INIT:])
    per = rmse_per_circuit(P[N_INIT:], T[N_INIT:])
    top = [targets[i] for i in np.argsort(-per)[:4]]

    def daily(arr):
        s = pd.Series(arr, index=t).resample("D").mean()
        return s.index.to_pydatetime(), s.to_numpy()

    # 1) rêve contre réalité : somme des 33 circuits, puis les 4 circuits qui pèsent le plus
    fig, axes = plt.subplots(5, 1, figsize=(6.3, 8.2), sharex=True)
    ax = axes[0]
    dx, somme_reel = daily(np.nansum(T, axis=1))
    _, somme_reve = daily(P.sum(axis=1))
    _, s_tot = daily(np.sqrt((S ** 2).sum(axis=1)))
    ax.fill_between(dx, np.maximum(somme_reve - 2 * s_tot, 0), somme_reve + 2 * s_tot, color=ROUGE, alpha=0.18, lw=0,
                    label="± 2 écarts-types calibrés (tronqués à zéro)")
    ax.plot(dx, somme_reel, color=NOIR, label="réalité")
    ax.plot(dx, somme_reve, color=ROUGE, label="rêve (moyenne de l'ensemble)")
    ax.axvline(t[N_INIT], color=GRIS, ls="--", lw=0.8, label="fin des 7 jours réels")
    ax.set_title(f"somme des 33 circuits — RMSE horaire {dec(total, 4)}", loc="left")
    poignees, libelles = ax.get_legend_handles_labels()
    fig.legend(poignees, libelles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.998), frameon=False)
    for ax, c in zip(axes[1:], top):
        j = targets.index(c)
        _, reel = daily(T[:, j])
        _, reve = daily(P[:, j])
        _, sj = daily(S[:, j])
        ax.fill_between(dx, np.maximum(reve - 2 * sj, 0), reve + 2 * sj, color=ROUGE, alpha=0.18, lw=0)
        ax.plot(dx, reel, color=NOIR)
        ax.plot(dx, reve, color=ROUGE)
        ax.axvline(t[N_INIT], color=GRIS, ls="--", lw=0.8)
        ax.set_title(f"{c} — RMSE {dec(per[j])}", loc="left")
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(virgule))
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(FuncFormatter(mois_annee))
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.962))
    sauver(fig, out, "reve_vs_realite")

    # 2) zoom horaire : deux semaines à partir du jour 95 du rêve, circuit dominant
    j = targets.index(top[0])
    a0 = N_INIT + 24 * 95
    sl = slice(a0, a0 + 24 * 14)
    tz = t[sl].to_pydatetime()
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.fill_between(tz, np.maximum(P[sl, j] - 2 * S[sl, j], 0), P[sl, j] + 2 * S[sl, j], color=ROUGE, alpha=0.18, lw=0,
                    label="± 2 écarts-types calibrés (tronqués à zéro)")
    ax.plot(tz, T[sl, j], color=NOIR, label="réalité horaire")
    ax.plot(tz, P[sl, j], color=ROUGE, label="rêve (moyenne de l'ensemble)")
    ax.set_ylabel(top[0])
    ax.set_ylim(0, 1.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(FuncFormatter(jour_mois))
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    sauver(fig, out, "zoom_horaire")

    # 3) dérive : RMSE par semaine depuis le début du rêve (semaine 1 = première semaine simulée)
    weeks = rmse_by_horizon(P[N_INIT:], T[N_INIT:])
    const = np.tile(np.nanmean(T[:N_INIT], axis=0), (len(T) - N_INIT, 1))
    wc = rmse_by_horizon(const, T[N_INIT:])
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.plot([w_ + 1 for w_, _ in wc], [r for _, r in wc], color=GRIS, ls="--", label="constante calée sur les 7 premiers jours")
    if Path(a.baseline_pred).exists():
        gb = pd.read_csv(a.baseline_pred, index_col="ut_ms").reindex(pred.index[N_INIT:]).to_numpy()
        wg = rmse_by_horizon(gb, T[N_INIT:])
        ax.plot([w_ + 1 for w_, _ in wg], [r for _, r in wg], color=VERT, ls=":", label="gradient boosting heure par heure (sans mémoire)")
    xs = [w_ + 1 for w_, _ in weeks]
    ys = [r for _, r in weeks]
    ax.plot(xs, ys, color=BLEU, marker="o", ms=2.2, label="world model (ensemble)")
    k = int(np.argmax(ys))
    ax.annotate(f"semaine {xs[k]} : {dec(ys[k])}", (xs[k], ys[k]), xytext=(xs[k], 0.166), fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.6))
    ax.set_ylim(0.03, 0.175)
    ax.set_xlabel("semaine depuis le début du rêve (la semaine 1 est la première semaine simulée)")
    ax.set_ylabel("RMSE de la semaine")
    ax.set_xlim(0, xs[-1] + 1)
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    sauver(fig, out, "derive")

    # 4) erreur par circuit
    order = np.argsort(-per)
    cper = rmse_per_circuit(const, T[N_INIT:])
    noms = [targets[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.bar(range(len(noms)), cper[order], fill=False, edgecolor=GRIS, lw=0.7, label="constante calée sur les 7 premiers jours")
    ax.bar(range(len(noms)), per[order], color=ROUGE, width=0.6, label="world model")
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels(noms, rotation=90, fontsize=6.5)
    ax.set_xlim(-0.7, len(noms) - 0.3)
    ax.set_ylabel("RMSE du circuit")
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.legend(loc="upper right")
    fig.tight_layout()
    sauver(fig, out, "erreur_par_circuit")

    # 5) calibration : RMSE observée par décile d'écart-type calibré
    e = np.abs(P[N_INIT:] - T[N_INIT:]).ravel()
    s = S[N_INIT:].ravel()
    m = np.isfinite(e)
    e, s = e[m], s[m]
    bins = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, s, side="right") - 1, 0, 9)
    xs = [float(s[idx == k].mean()) for k in range(10)]
    ys = [float(np.sqrt((e[idx == k] ** 2).mean())) for k in range(10)]
    lim = 1.12 * max(max(xs), max(ys))
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    ax.plot([0, lim], [0, lim], color=GRIS, ls="--", label="RMSE = écart-type")
    ax.plot(xs, ys, color=BLEU, marker="o", ms=3.5, label="déciles d'écart-type")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("écart-type calibré de l'ensemble (moyenne du décile)")
    ax.set_ylabel("RMSE observée dans le décile")
    ax.xaxis.set_major_formatter(FuncFormatter(virgule))
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    sauver(fig, out, "calibration_incertitude")

    resume = {"periode": period, "rmse": total, "top_circuits": top, "pire_semaine": int(np.argmax([r for _, r in weeks])) + 1,
              "pire_rmse": float(max(r for _, r in weeks)), "deciles_std": xs, "deciles_rmse": ys,
              "zoom": [str(t[a0]), str(t[a0 + 24 * 14 - 1])], "calibration": a.calibration}
    (out / "figures_resume.json").write_text(json.dumps(resume, indent=1, ensure_ascii=False))
    print(json.dumps(resume, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
