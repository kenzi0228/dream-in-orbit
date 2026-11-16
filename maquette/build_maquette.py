"""Maquette de l'interface du pilote (console de planification), construite à partir des résultats
réels du projet : fenêtre du pilote (reports/pilot.json), prédictions et écarts-types du protocole
strict (weights_holdout/), vérité horaire (validation/), calibration et planning.

    .venv/bin/python maquette/build_maquette.py                 # -> maquette/index.html, maquette/ecrans/*.html
    .venv/bin/python maquette/build_maquette.py --canvas DOSSIER # + artboards .dc.html pour une toile de design

Trois écrans de 1440 x 1000 px : le rêve sur la fenêtre, la comparaison de plannings, la recherche
locale. Tout est statique (SVG calculé ici), sans bibliothèque ni script côté page.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "maquette"
N_INIT = 24 * 7
W_FRAME, H_FRAME = 1440, 1000

# --- palette (dataviz : catégories validées, texte en encre, statut réservé) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#6f6e69"
SURFACE, PANEL, BORDER, GRID = "#fcfcfb", "#ffffff", "#e4e2dc", "#ebe9e3"
PRED, MESURE, BAND = "#2a78d6", "#eb6834", "rgba(42,120,214,0.14)"
PRIMARY = "#1c5cab"
GOOD = "#0ca30c"
UMBRA = "#f1efe9"
MODES = [  # (colonne ftl, libellé, couleur)
    ("ftl_EARTH", "Terre (EARTH)", "#cfcdc6"),
    ("ftl_RADIO_SCIENCE", "Radio-science", "#e87ba4"),
    ("ftl_SLEW", "Manœuvre (SLEW)", "#eda100"),
    ("ftl_NADIR", "Nadir", "#1baf7a"),
    ("ftl_MAINTENANCE", "Maintenance", "#6250d6"),
    ("__autre", "Autres pointages", "#7d7c76"),
]
LIBELLES = {  # libellés longs des scénarios du pilote (reports/pilot.json n'en contient pas)
    "hors_comms_tout_EARTH": "Hors communication, tout en pointage EARTH (plus d’observation)",
    "hors_comms_tout_NADIR": "Hors communication, tout en NADIR (observation continue)",
    "hors_comms_sans_manoeuvres": "Hors communication, sans manœuvres (SLEW rendu à EARTH)",
    "recherche_locale": "Recherche locale : permutations d’heures de pointage dans chaque journée, 120 essais, 15 gardées",
}
NOMS = {
    "hors_comms_tout_EARTH": "Tout en EARTH hors communication",
    "hors_comms_tout_NADIR": "Tout en NADIR hors communication",
    "hors_comms_sans_manoeuvres": "Sans manœuvres (SLEW rendu à EARTH)",
    "recherche_locale": "Recherche locale (15 permutations gardées)",
}
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
NBSP, NNBSP = " ", " "


# ----------------------------------------------------------------------------- formats
def fr(x: float, d: int = 1, signe: bool = False) -> str:
    """Nombre à la française : virgule décimale, espace fine insécable pour les milliers."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{abs(x):,.{d}f}".replace(",", NNBSP).replace(".", ",")
    if x < 0:
        return "−" + s
    return ("+" if signe and x > 0 else "") + s


def pct(x: float, d: int = 2, signe: bool = True) -> str:
    return fr(x, d, signe) + NNBSP + "%"


def jour(ms: int, mois: bool = True) -> str:
    d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
    return f"{d.day}{NBSP}{MOIS[d.month - 1]}" if mois else str(d.day)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    a = math.floor(lo / step) * step
    b = math.ceil(hi / step) * step
    k = int(round((b - a) / step))
    return [round(a + i * step, 10) for i in range(k + 1)]


# ----------------------------------------------------------------------------- données
def charger() -> dict:
    pil = json.loads((ROOT / "reports" / "pilot.json").read_text())
    debut = pd.Timestamp(pil["fenetre"]["debut"], tz="UTC")
    jours = int(pil["fenetre"]["jours"])
    t0 = int(debut.value // 10**6)
    n = jours * 24
    t1 = t0 + n * 3_600_000

    P = pd.read_csv(ROOT / "weights_holdout" / "val_pred.csv")
    S = pd.read_csv(ROOT / "weights_holdout" / "val_std.csv")
    T = pd.read_csv(ROOT / "validation" / "verite_horaire.csv")
    cal = json.loads((ROOT / "weights_holdout" / "calibration.json").read_text())
    circuits = [c for c in P.columns if c != "ut_ms"]

    def fen(df: pd.DataFrame) -> pd.DataFrame:
        d = df[(df.ut_ms >= t0) & (df.ut_ms < t1)].set_index("ut_ms")
        return d.reindex(range(t0, t1, 3_600_000))

    Pw, Sw, Tw = fen(P)[circuits], fen(S)[circuits], fen(T)[circuits]
    k = np.array([cal["scale"][c] for c in circuits])
    sig = np.sqrt(((Sw.to_numpy() * k) ** 2).sum(axis=1))  # combinaison en quadrature des σ calibrés
    tot_pred = Pw.sum(axis=1).to_numpy()
    tot_truth = Tw.sum(axis=1, min_count=len(circuits)).to_numpy()
    err = (Pw.to_numpy() - Tw.to_numpy())
    rmse_fen = float(np.sqrt(np.nanmean(err**2)))
    couverture = float(np.nanmean(np.abs(err) <= 2 * Sw.to_numpy() * k))

    # planning : le réel si le chargeur du projet est disponible, sinon celui du pilote (15 permutations)
    plan, source_plan = None, ""
    try:
        sys.path.insert(0, str(ROOT))
        from dio.data import load_period  # type: ignore

        meta = json.loads((ROOT / "weights_holdout" / "meta.json").read_text())
        X, _ = load_period(ROOT / "kit_etudiants" / pil["fenetre"]["periode"], meta["vocab"])
        plan = X[(X.index >= t0) & (X.index < t1)]
        plan = plan[[c for c in plan.columns if c.startswith("ftl_") and c not in ("ftl_n_switch", "ftl_other")]]
        source_plan = "planning réel (ftl.csv du kit, aligné sur la grille horaire)"
    except Exception as e:  # noqa: BLE001
        pl = pd.read_csv(ROOT / "reports" / "pilot.planning.csv")
        plan = pl[(pl.ut_ms >= t0) & (pl.ut_ms < t1)].set_index("ut_ms")
        source_plan = f"planning du pilote après recherche locale (reports/pilot.planning.csv ; {type(e).__name__})"
    plan = plan.reindex(range(t0, t1, 3_600_000)).fillna(0.0)
    modes = [c for c in plan.columns if c != "ftl_comms"]
    comms = plan["ftl_comms"].to_numpy()
    libre = comms == 0
    heures = {m: float(plan[m].sum()) for m in modes}
    heures_comms = {m: float(plan.loc[~libre, m].sum()) for m in modes}
    n_libre = int(libre.sum())

    def scen_heures(nom: str) -> dict:
        if nom == "hors_comms_tout_EARTH":
            return {m: heures_comms[m] + (n_libre if m == "ftl_EARTH" else 0) for m in modes}
        if nom == "hors_comms_tout_NADIR":
            return {m: heures_comms[m] + (n_libre if m == "ftl_NADIR" else 0) for m in modes}
        if nom == "hors_comms_sans_manoeuvres":
            h = dict(heures)
            slew_libre = float(plan.loc[libre, "ftl_SLEW"].sum())
            h["ftl_EARTH"] += slew_libre
            h["ftl_SLEW"] -= slew_libre
            return h
        return dict(heures)

    # mode dominant par heure (hors communication) pour la frise
    dom = []
    for i in range(n):
        row = plan.iloc[i]
        best, val = "ftl_EARTH", -1.0
        for m in modes:
            if row[m] > val:
                best, val = m, float(row[m])
        dom.append(best if val > 0 else "ftl_EARTH")

    # éclipses (ombre) sur la fenêtre, depuis les données de la démo si présentes
    umbra = np.zeros(n)
    try:
        s = (ROOT / "demo" / "data.js").read_text(encoding="utf-8")
        d = json.loads(s[s.find("{"): s.rfind("}") + 1])
        f14 = d.get("fenetre14") or {}
        if int(f14.get("t0_ms", -1)) == t0 and len(f14.get("ev_umbra", [])) == n:
            umbra = np.array(f14["ev_umbra"], dtype=float)
    except Exception:  # noqa: BLE001
        pass

    e_pred = Pw.sum(axis=0)
    e_truth = Tw.sum(axis=0)
    rmse_c = np.sqrt(np.nanmean(err**2, axis=0))
    top = e_pred.sort_values(ascending=False).head(8).index.tolist()
    circuits_tab = [{"id": c, "pred": float(e_pred[c]), "mesure": float(e_truth[c]),
                     "ecart_pct": float(100 * (e_pred[c] - e_truth[c]) / e_truth[c]) if e_truth[c] else float("nan"),
                     "rmse": float(rmse_c[circuits.index(c)]), "part": float(e_pred[c] / e_pred.sum())} for c in top]

    return {"pil": pil, "t0": t0, "n": n, "jours": jours, "debut_ms": t0, "fin_ms": t1 - 3_600_000,
            "init_ms": t0 - N_INIT * 3_600_000,
            "tot_pred": tot_pred, "tot_truth": tot_truth, "sig": sig, "rmse_fen": rmse_fen, "couverture": couverture,
            "comms": comms, "dom": dom, "umbra": umbra, "heures": heures, "scen_heures": scen_heures, "modes": modes,
            "n_libre": n_libre, "source_plan": source_plan, "circuits_tab": circuits_tab,
            "chauffage": {"NPWD2551", "NPWD2451"}}


# ----------------------------------------------------------------------------- SVG
def svg_total(D: dict, W: int = 944, H: int = 420) -> str:
    ml, mr, mt, mb = 70, 70, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    n = D["n"]
    pred, truth, sig = D["tot_pred"], D["tot_truth"], D["sig"]
    lo = min(float(np.min(pred - 2 * sig)), float(np.nanmin(truth)))
    hi = max(float(np.max(pred + 2 * sig)), float(np.nanmax(truth)))
    ticks = nice_ticks(lo, hi, 6)
    y0, y1 = ticks[0], ticks[-1]
    xs = lambda i: ml + pw * i / (n - 1)  # noqa: E731
    ys = lambda v: mt + ph * (1 - (v - y0) / (y1 - y0))  # noqa: E731
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Puissance totale mesurée et prédite, heure par heure, avec sa bande d’incertitude">']
    # ombre (éclipses)
    i = 0
    while i < n:
        if D["umbra"][i] >= 0.2:
            j = i
            while j < n and D["umbra"][j] >= 0.2:
                j += 1
            x0, x1 = xs(i), xs(min(j, n - 1))
            out.append(f'<rect x="{x0:.1f}" y="{mt}" width="{max(x1 - x0, 1.5):.1f}" height="{ph}" fill="{UMBRA}"></rect>')
            i = j
        else:
            i += 1
    for t in ticks:
        y = ys(t)
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"></line>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{fr(t, 1)}</text>')
    for d in range(D["jours"]):
        x = xs(d * 24)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt + ph}" y2="{mt + ph + 4}" stroke="{BORDER}" stroke-width="1"></line>')
        out.append(f'<text x="{xs(d * 24 + 12):.1f}" y="{mt + ph + 18}" text-anchor="middle" class="axis">{jour(D["t0"] + d * 24 * 3_600_000, d == 0 or dt.datetime.fromtimestamp((D["t0"] + d * 86_400_000) / 1000, dt.timezone.utc).day == 1)}</text>')
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="{BORDER}" stroke-width="1"></line>')
    # bande ±2σ
    up = " ".join(f"{xs(i):.1f},{ys(pred[i] + 2 * sig[i]):.1f}" for i in range(n))
    dn = " ".join(f"{xs(i):.1f},{ys(pred[i] - 2 * sig[i]):.1f}" for i in range(n - 1, -1, -1))
    out.append(f'<polygon points="{up} {dn}" fill="{BAND}" stroke="none"></polygon>')
    # mesuré (avec trous)
    seg, started = [], False
    for i in range(n):
        v = truth[i]
        if v != v:
            started = False
            continue
        seg.append(("L" if started else "M") + f"{xs(i):.1f},{ys(v):.1f}")
        started = True
    out.append(f'<path d="{" ".join(seg)}" fill="none" stroke="{MESURE}" stroke-width="1.6" stroke-linejoin="round"></path>')
    out.append('<path d="' + " ".join(("M" if i == 0 else "L") + f"{xs(i):.1f},{ys(pred[i]):.1f}" for i in range(n)) +
               f'" fill="none" stroke="{PRED}" stroke-width="2" stroke-linejoin="round"></path>')
    # étiquettes directes en bout de courbe
    yp, yt = ys(pred[-1]), ys(np.nanmean(truth[-6:]))
    if abs(yp - yt) < 14:
        if yp < yt:
            yp, yt = (yp + yt) / 2 - 7, (yp + yt) / 2 + 7
        else:
            yp, yt = (yp + yt) / 2 + 7, (yp + yt) / 2 - 7
    out.append(f'<text x="{ml + pw + 8}" y="{yp + 4:.1f}" font-size="11.5" font-weight="600" fill="{INK}">Prédit</text>')
    out.append(f'<text x="{ml + pw + 8}" y="{yt + 4:.1f}" font-size="11.5" font-weight="600" fill="{INK}">Mesuré</text>')
    out.append("</svg>")
    return "".join(out)


def svg_planning(D: dict, W: int = 944) -> str:
    ml, mr = 70, 70
    pw = W - ml - mr
    n = D["n"]
    h_mode, h_comms, gap = 40, 12, 6
    top = 4
    H = top + h_mode + gap + h_comms + 22
    cw = pw / n
    col = {m: c for m, _, c in MODES}
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Planning des pointages et fenêtres de communication, heure par heure">']
    out.append(f'<text x="{ml - 8}" y="{top + h_mode / 2 + 4}" text-anchor="end" class="axis">Pointage</text>')
    out.append(f'<text x="{ml - 8}" y="{top + h_mode + gap + h_comms / 2 + 4}" text-anchor="end" class="axis">Comms</text>')
    # cellules de pointage : fusion des heures consécutives de même mode
    i = 0
    while i < n:
        m = D["dom"][i]
        j = i
        while j < n and D["dom"][j] == m:
            j += 1
        key = m if m in col else "__autre"
        out.append(f'<rect x="{ml + i * cw:.2f}" y="{top}" width="{(j - i) * cw - 0.6:.2f}" height="{h_mode}" fill="{col[key]}"></rect>')
        i = j
    i = 0
    while i < n:
        if D["comms"][i] > 0:
            j = i
            while j < n and D["comms"][j] > 0:
                j += 1
            out.append(f'<rect x="{ml + i * cw:.2f}" y="{top + h_mode + gap}" width="{(j - i) * cw - 0.6:.2f}" height="{h_comms}" fill="{INK2}"></rect>')
            i = j
        else:
            i += 1
    yb = top + h_mode + gap + h_comms
    for d in range(D["jours"]):
        x = ml + d * 24 * cw
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{yb + 3}" stroke="#ffffff" stroke-width="1.2"></line>')
        out.append(f'<text x="{x + 12 * cw:.1f}" y="{yb + 16}" text-anchor="middle" class="axis">{jour(D["t0"] + d * 86_400_000, d == 0 or dt.datetime.fromtimestamp((D["t0"] + d * 86_400_000) / 1000, dt.timezone.utc).day == 1)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_scenarios(D: dict, W: int = 560) -> str:
    pil = D["pil"]
    rows = [(k, NOMS[k], v) for k, v in pil["scenarios"].items()]
    rows.append(("recherche_locale", NOMS["recherche_locale"], pil["recherche_locale"]))
    ml, mr, rh, top = 300, 30, 46, 28
    pw = W - ml - mr
    H = top + rh * len(rows) + 30
    lim = max(0.8, max(abs(v["ecart_pct"]) + v["dispersion_pct"] for _, _, v in rows) * 1.15)
    ticks = nice_ticks(-lim, lim, 7)
    x = lambda v: ml + pw * (v + lim) / (2 * lim)  # noqa: E731
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Écart d’énergie de chaque scénario par rapport au planning réel, avec la dispersion des membres">']
    for t in ticks:
        if abs(t) > lim:
            continue
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}" stroke-width="1"></line>')
        out.append(f'<text x="{x(t):.1f}" y="{top + rh * len(rows) + 16}" text-anchor="middle" class="axis">{pct(t, 1 if abs(t) < 1 else 0, True) if t else "0"}</text>')
    out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 10}" y2="{top + rh * len(rows)}" stroke="{INK2}" stroke-width="1"></line>')
    out.append(f'<text x="{x(-lim):.1f}" y="{top - 12}" class="axis">← économie</text>')
    out.append(f'<text x="{x(lim):.1f}" y="{top - 12}" text-anchor="end" class="axis">surcoût →</text>')
    for r, (k, lib, v) in enumerate(rows):
        cy = top + rh * r + rh / 2
        e, d = v["ecart_pct"], v["dispersion_pct"]
        x0, x1 = x(0), x(e)
        color = PRED if e <= 0 else "#e34948"
        bx, bw = (min(x0, x1), abs(x1 - x0))
        rx = "4" if bw > 8 else "0"
        out.append(f'<rect x="{bx:.1f}" y="{cy - 7:.1f}" width="{max(bw, 1.5):.1f}" height="14" rx="{rx}" fill="{color}"></rect>')
        out.append(f'<line x1="{x(e - d):.1f}" x2="{x(e + d):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{INK}" stroke-width="1.4"></line>')
        out.append(f'<line x1="{x(e - d):.1f}" x2="{x(e - d):.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" stroke="{INK}" stroke-width="1.4"></line>')
        out.append(f'<line x1="{x(e + d):.1f}" x2="{x(e + d):.1f}" y1="{cy - 4:.1f}" y2="{cy + 4:.1f}" stroke="{INK}" stroke-width="1.4"></line>')
        label = lib if len(lib) <= 44 else lib[:42].rstrip() + "…"
        out.append(f'<text x="{ml - 12}" y="{cy + 4:.1f}" text-anchor="end" font-size="12" fill="{INK}">{esc(label)}</text>')
        tx = x(0) + 8 if e <= 0 else x(0) - 8
        anchor = "start" if e <= 0 else "end"
        out.append(f'<text x="{tx:.1f}" y="{cy + 4:.1f}" text-anchor="{anchor}" font-size="12" font-weight="600" fill="{INK}">{pct(e, 2)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_recherche(D: dict, W: int = 640, H: int = 430) -> str:
    rl = D["pil"]["recherche_locale"]
    hist = rl["historique"]
    iters = rl["iterations"]
    ml, mr, mt, mb = 60, 24, 18, 28
    pw, ph = W - ml - mr, H - mt - mb
    es = [e for _, e in hist]
    ticks = nice_ticks(min(es) - 0.15, max(es) + 0.15, 6)
    y0, y1 = ticks[0], ticks[-1]
    xs = lambda i: ml + pw * i / iters  # noqa: E731
    ys = lambda v: mt + ph * (1 - (v - y0) / (y1 - y0))  # noqa: E731
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Énergie simulée au fil des itérations de la recherche locale">']
    for t in ticks:
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{ys(t):.1f}" y2="{ys(t):.1f}" stroke="{GRID}" stroke-width="1"></line>')
        out.append(f'<text x="{ml - 8}" y="{ys(t) + 4:.1f}" text-anchor="end" class="axis">{fr(t, 1)}</text>')
    for t in range(0, iters + 1, 20):
        out.append(f'<text x="{xs(t):.1f}" y="{mt + ph + 18}" text-anchor="middle" class="axis">{t}</text>')
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="{BORDER}" stroke-width="1"></line>')
    # ligne en escalier : l'énergie ne change qu'aux itérations gardées
    d = []
    for k, (it, e) in enumerate(hist):
        if k == 0:
            d.append(f"M{xs(it):.1f},{ys(e):.1f}")
        else:
            d.append(f"L{xs(it):.1f},{ys(hist[k - 1][1]):.1f} L{xs(it):.1f},{ys(e):.1f}")
    d.append(f"L{xs(iters):.1f},{ys(hist[-1][1]):.1f}")
    out.append(f'<path d="{" ".join(d)}" fill="none" stroke="{PRED}" stroke-width="2" stroke-linejoin="round"></path>')
    for it, e in hist[1:]:
        out.append(f'<circle cx="{xs(it):.1f}" cy="{ys(e):.1f}" r="4" fill="{PRED}" stroke="#ffffff" stroke-width="2"></circle>')
    out.append(f'<text x="{xs(0) + 6:.1f}" y="{ys(hist[0][1]) - 8:.1f}" font-size="11.5" font-weight="600" fill="{INK}">Départ {fr(hist[0][1], 1)}</text>')
    out.append(f'<text x="{xs(iters):.1f}" y="{ys(hist[-1][1]) + 18:.1f}" text-anchor="end" font-size="11.5" font-weight="600" fill="{INK}">Arrivée {fr(hist[-1][1], 1)} ({pct(rl["ecart_pct"], 2)})</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- écrans
def icon(name: str) -> str:
    p = {"play": '<path d="M5 3l10 5-10 5z"></path>', "down": '<path d="M8 2v8M4 7l4 4 4-4M3 13h10"></path>',
         "plus": '<path d="M8 3v10M3 8h10"></path>', "check": '<path d="M3 8l3 3 7-7"></path>',
         "send": '<path d="M2 8l12-5-4 12-2-5z"></path>'}[name]
    fill = "currentColor" if name == "play" else "none"
    return f'<svg width="16" height="16" viewBox="0 0 16 16" fill="{fill}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{p}</svg>'


def topbar(actif: int) -> str:
    tabs = [("Rêve", "Main.dc.html"), ("Scénarios", "Scenarios.dc.html"), ("Optimiseur", "Optimiseur.dc.html")]
    t = "".join(f'<a class="tab{" on" if i + 1 == actif else ""}" href="{"#" if i + 1 == actif else href}">{lab}</a>' for i, (lab, href) in enumerate(tabs))
    return (f'<header class="topbar"><div class="brand"><span class="dot"></span>Dream in Orbit<span class="sub">Console de planification</span></div>'
            f'<nav class="tabs" aria-label="Écrans">{t}</nav>'
            f'<div class="topright"><span>Mars Express · annee2 (validation) · poids weights_holdout</span>'
            f'<span class="chip"><span class="sw" style="background:{PRED}"></span>World model 0,5{NBSP}GRU + 0,5{NBSP}GBM</span>'
            f'<span class="chip">RMSE strict 0,0750</span></div></header>')


def ecran_reve(D: dict) -> str:
    pil = D["pil"]
    prs = pil["planning_reel_simule"]
    e_pred, e_mes = prs["energie"], pil["energie_reelle_somme_horaire"]
    tiles = [
        ("Énergie prédite", fr(e_pred, 1), "somme horaire des 33 circuits sur 14 jours"),
        ("Énergie mesurée", fr(e_mes, 1), "vérité horaire du kit, même fenêtre"),
        ("Écart prédit − mesuré", pct(prs["ecart_au_reel_pct"], 2), "sur l’énergie de la fenêtre"),
        ("Dispersion des membres", "±" + NNBSP + fr(prs["dispersion_membres"], 1), f"écart-type des 5 rêves ({pct(100 * prs['dispersion_membres'] / e_pred, 1, False)})"),
        ("RMSE sur la fenêtre", fr(D["rmse_fen"], 4), "toutes les cellules heure × circuit"),
    ]
    tiles_html = "".join(f'<div class="tile"><span class="lab">{esc(a)}</span><span class="val">{b}</span><span class="hint">{esc(c)}</span></div>' for a, b, c in tiles)
    rows = []
    for r in D["circuits_tab"]:
        role = "chauffage" if r["id"] in D["chauffage"] else ""
        rows.append(f'<tr><td><strong>{r["id"]}</strong> <span class="muted">{role}</span></td>'
                    f'<td class="num"><span class="bar" style="width:{max(4, round(90 * r["part"] / D["circuits_tab"][0]["part"]))}px"></span> {fr(r["pred"], 1)}</td>'
                    f'<td class="num">{fr(r["mesure"], 1)}</td><td class="num">{pct(r["ecart_pct"], 1)}</td><td class="num">{fr(r["rmse"], 3)}</td></tr>')
    legend_modes = "".join(f'<span><i style="background:{c}"></i>{esc(l)}</span>' for _, l, c in MODES)
    return f'''<div class="frame" lang="fr" style="width:{W_FRAME}px;height:{H_FRAME}px;background:{SURFACE};color:{INK};font-family:Inter,system-ui,sans-serif">
{topbar(1)}
<main class="main">
<div class="titlerow"><div><h1>Rêve sur la fenêtre du {jour(D["debut_ms"])} au {jour(D["fin_ms"])} 2010</h1>
<div class="sub">7{NBSP}jours réels d’amorçage ({jour(D["init_ms"])} → {jour(D["debut_ms"] - 86_400_000)}), puis {D["jours"]}{NBSP}jours simulés en boucle ouverte · 5{NBSP}membres · planning réel</div></div>
<form class="controls" action="#"><div class="field"><label for="debut">Début de la fenêtre</label><input id="debut" type="date" value="2010-10-25"></div>
<div class="field"><label for="duree">Durée</label><select id="duree"><option>14 jours</option><option>7 jours</option><option>28 jours</option></select></div>
<div class="field"><label for="plan">Planning</label><select id="plan"><option>Réel (ftl.csv, dmop.csv)</option><option>Scénario enregistré…</option></select></div>
<button class="btn primary" type="button">{icon("play")}Relancer le rêve</button><button class="btn" type="button">{icon("down")}predictions.csv</button></form></div>
<div class="tiles">{tiles_html}</div>
<div class="cols">
<div class="col-l"><section class="panel"><div class="phead"><h2>Puissance totale, heure par heure</h2>
<div class="legend"><span><i class="line" style="background:{MESURE}"></i>Mesuré</span><span><i class="line" style="background:{PRED}"></i>Prédit (moyenne des 5 membres)</span><span><i style="background:{BAND}"></i>Bande ±2σ</span><span><i style="background:{UMBRA}"></i>Ombre (éclipse)</span></div></div>
{svg_total(D)}
<div class="note">La bande ±2σ combine en quadrature les écarts-types calibrés par circuit (facteurs ajustés sur le protocole inversé). Couverture observée sur la fenêtre : {pct(100 * D["couverture"], 0, False)} des cellules.</div></section>
<section class="panel"><div class="phead"><h2>Planning des pointages (ftl) et communications</h2><div class="legend">{legend_modes}<span><i style="background:{INK2}"></i>Communication avec la Terre</span></div></div>
{svg_planning(D)}</section></div>
<div class="col-r"><section class="panel"><h2>Circuits les plus consommateurs sur la fenêtre</h2>
<table><thead><tr><th>Circuit</th><th class="num">Énergie prédite</th><th class="num">Mesurée</th><th class="num">Écart</th><th class="num">RMSE</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<div class="note">Les deux circuits de chauffage font 63{NNBSP}% de l’erreur quadratique du protocole strict.</div></section>
<section class="panel"><h2>Ce que vaut ce rêve</h2>
<ul class="facts"><li>RMSE du protocole strict (annee1 → annee2) : <strong>0,0750</strong> ; attendez-vous à 0,08 – 0,09 sur une année nouvelle.</li>
<li>Après le 7<sup>e</sup> jour, le modèle ne revoit plus aucune mesure : l’écart peut dériver, surtout sur les circuits de chauffage.</li>
<li>Une bande qui s’élargit signale un planning que le modèle connaît mal (trou de commandes, mode rare).</li></ul></section></div>
</div></main></div>'''


def ecran_scenarios(D: dict) -> str:
    pil = D["pil"]
    prs = pil["planning_reel_simule"]
    scen = dict(pil["scenarios"])
    scen["recherche_locale"] = pil["recherche_locale"]
    libs = LIBELLES
    membres = pil["membres"]
    sel = "hors_comms_tout_EARTH"

    def row(k: str, v: dict, ref: bool = False, checked: bool = False) -> str:
        lib = "Planning réel (référence)" if ref else NOMS[k]
        if ref:
            right = f'<span class="val">{fr(v["energie"], 1)}</span><span class="muted">mesuré : {fr(pil["energie_reelle_somme_horaire"], 1)}</span>'
            dots = ""
        else:
            ok = int(round(v["accord_membres"] * membres))
            dots = "".join(f'<i class="{"on" if i < ok else ""}"></i>' for i in range(membres))
            right = (f'<span class="val">{pct(v["ecart_pct"], 2)}</span><span class="muted">{fr(v["energie"], 1)} · ±{NNBSP}{pct(v["dispersion_pct"], 2, False)}</span>')
            if v.get("robuste"):
                dots += f'<span class="chip ok"><span class="sw" style="background:{GOOD}"></span>{icon("check")}robuste</span>'
        chk = ' checked="checked"' if checked else ""
        return (f'<label class="scen{" sel" if checked else ""}"><input type="radio" name="scen" value="{k}"{chk}>'
                f'<span class="scen-l"><span class="nm">{esc(lib)}</span><span class="dots">{dots}</span></span><span class="scen-r">{right}</span></label>')

    liste = row("reel", prs, ref=True) + "".join(row(k, v, checked=(k == sel)) for k, v in scen.items())
    avant, apres = D["heures"], D["scen_heures"](sel)
    ordre = ["ftl_EARTH", "ftl_RADIO_SCIENCE", "ftl_SLEW", "ftl_NADIR", "ftl_MAINTENANCE", "ftl_D4PNPO"]
    autres = [m for m in D["modes"] if m not in ordre]
    lib_mode = {m: l for m, l, _ in MODES}
    lib_mode["ftl_D4PNPO"] = "D4PNPO"
    trs = []
    for m in ordre + ["__autres"]:
        if m == "__autres":
            a, b, lib = sum(avant[x] for x in autres), sum(apres[x] for x in autres), "Autres pointages"
        else:
            a, b, lib = avant[m], apres[m], lib_mode.get(m, m[4:])
        delta = b - a
        cls = "muted" if abs(delta) < 0.05 else ""
        trs.append(f'<tr><td>{esc(lib)}</td><td class="num">{fr(a, 0)}{NBSP}h</td><td class="num"><strong>{fr(b, 0)}{NBSP}h</strong></td><td class="num {cls}">{fr(delta, 0, True)}{NBSP}h</td></tr>')
    v = scen[sel]
    tab = [f'<tr><td>Planning réel (référence)</td><td class="num">{fr(prs["energie"], 1)}</td><td class="num">–</td><td class="num">±{NNBSP}{fr(prs["dispersion_membres"], 1)}</td><td class="num">–</td></tr>']
    for k2, v2 in scen.items():
        tab.append(f'<tr><td>{esc(NOMS[k2])}</td><td class="num">{fr(v2["energie"], 1)}</td><td class="num">{pct(v2["ecart_pct"], 2)}</td>'
                   f'<td class="num">±{NNBSP}{pct(v2["dispersion_pct"], 2, False)}</td><td class="num">{int(round(v2["accord_membres"] * membres))}{NBSP}/{NBSP}{membres}</td></tr>')
    tableau = f'<h3>Les mêmes chiffres en tableau</h3><table><thead><tr><th>Scénario</th><th class="num">Énergie</th><th class="num">Écart</th><th class="num">Dispersion</th><th class="num">Accord</th></tr></thead><tbody>{"".join(tab)}</tbody></table>'
    return f'''<div class="frame" lang="fr" style="width:{W_FRAME}px;height:{H_FRAME}px;background:{SURFACE};color:{INK};font-family:Inter,system-ui,sans-serif">
{topbar(2)}
<main class="main">
<div class="titlerow"><div><h1>Comparer des plannings sur la fenêtre du {jour(D["debut_ms"])} au {jour(D["fin_ms"])} 2010</h1>
<div class="sub">Comparaison appariée : chaque membre de l’ensemble simule le planning réel et le planning modifié ; l’incertitude porte sur la différence, pas sur le niveau.</div></div>
<div class="controls"><button class="btn" type="button">{icon("plus")}Nouveau scénario</button><button class="btn primary" type="button">{icon("play")}Simuler les 4 scénarios</button></div></div>
<div class="cols">
<section class="panel scen-list"><h2>Scénarios</h2><fieldset class="scens"><legend class="sr">Choisir un scénario</legend>{liste}</fieldset>
<div class="ghost">{icon("plus")}Créer un scénario à partir du planning réel</div>
<div class="note">Les fenêtres de communication avec la Terre ne sont jamais modifiées ; les features dérivées (anticipation, inertie) sont recalculées après chaque modification.</div></section>
<section class="panel"><div class="phead"><h2>Écart d’énergie par rapport au planning réel</h2><div class="legend"><span><i style="background:{PRED}"></i>Économie</span><span><i style="background:#e34948"></i>Surcoût</span><span><i class="line" style="background:{INK}"></i>± dispersion des membres</span></div></div>
{svg_scenarios(D)}
{tableau}
<div class="note">Énergie du planning réel simulé : {fr(prs["energie"], 1)} (mesurée : {fr(pil["energie_reelle_somme_horaire"], 1)}). Un scénario est « robuste » quand les {membres} membres sont d’accord sur le signe de la différence.</div></section>
<section class="panel detail"><h2>{esc(NOMS[sel])}</h2><div class="sub">{esc(libs[sel])}</div>
<div class="kpi"><span class="val">{pct(v["ecart_pct"], 2)}</span><span class="muted">{fr(v["energie"] - prs["energie"], 1, True)} sur 14{NBSP}jours · ±{NNBSP}{pct(v["dispersion_pct"], 2, False)} · {int(round(v["accord_membres"] * membres))}{NBSP}membres sur {membres}</span></div>
<h3>Heures par mode de pointage sur la fenêtre</h3>
<table><thead><tr><th>Mode</th><th class="num">Réel</th><th class="num">Scénario</th><th class="num">Δ</th></tr></thead><tbody>{"".join(trs)}</tbody></table>
<div class="note">{D["n_libre"]}{NBSP}heures hors communication sont modifiables sur {D["n"]}.</div>
<div class="actions"><button class="btn primary" type="button">{icon("play")}Rêver ce scénario en détail</button><button class="btn" type="button">{icon("down")}Exporter le planning</button></div>
<div class="warn">Rien ici n’est vérifiable contre la réalité : ces plannings n’ont pas volé. Le pilote mesure ce que le modèle croit, pas ce que le satellite ferait.</div></section>
</div></main></div>'''


def ecran_optimiseur(D: dict) -> str:
    pil = D["pil"]
    rl = pil["recherche_locale"]
    prs = pil["planning_reel_simule"]
    hist = rl["historique"]
    e0 = hist[0][1]
    trs = "".join(f'<tr><td class="num">{it}</td><td class="num">{fr(e, 2)}</td><td class="num">{pct(100 * (e - e0) / e0, 3)}</td></tr>' for it, e in hist[1:])
    return f'''<div class="frame" lang="fr" style="width:{W_FRAME}px;height:{H_FRAME}px;background:{SURFACE};color:{INK};font-family:Inter,system-ui,sans-serif">
{topbar(3)}
<main class="main">
<div class="titlerow"><div><h1>Optimiser le planning réel par recherche locale</h1>
<div class="sub">Permutation de deux heures de pointage hors communication dans une même journée : même volume d’observation par mode et par jour, fenêtres de communication intouchées.</div></div>
<div class="controls"><span class="chip"><span class="sw" style="background:{GOOD}"></span>Terminé · {rl["iterations"]}{NBSP}itérations · {rl["permutations_gardees"]}{NBSP}gardées</span></div></div>
<div class="cols">
<section class="panel params"><h2>Paramètres</h2>
<form action="#" class="vform"><div class="field"><label for="iters">Itérations</label><input id="iters" type="number" value="{rl["iterations"]}"></div>
<div class="field"><label for="graine">Graine aléatoire</label><input id="graine" type="number" value="0"></div>
<div class="field"><label for="jours">Journées concernées</label><select id="jours"><option>Les 14 jours de la fenêtre</option><option>Une journée</option></select></div>
<label class="check"><input type="checkbox" checked="checked"> Garder une permutation seulement si les 5 membres la jugent économique</label>
<label class="check"><input type="checkbox" checked="checked"> Recalculer les features dérivées après chaque permutation</label>
<button class="btn primary" type="button">{icon("play")}Lancer la recherche</button></form>
<div class="note">Chaque itération simule la fenêtre entière avec les 5 membres (environ 2{NBSP}s sur 6{NBSP}cœurs).</div>
<h3>Une itération</h3><ol class="steps"><li>Tirer une journée de la fenêtre et deux de ses heures hors communication aux pointages différents.</li><li>Échanger leurs pointages, recalculer les features dérivées (anticipation, inertie).</li><li>Rêver la fenêtre avec les 5 membres ; garder l’échange seulement si chacun prédit moins d’énergie.</li></ol></section>
<section class="panel"><div class="phead"><h2>Énergie simulée au fil des itérations</h2><div class="legend"><span><i class="line" style="background:{PRED}"></i>Meilleur planning courant</span><span><i style="background:{PRED};border-radius:50%"></i>Permutation gardée</span></div></div>
{svg_recherche(D)}
<div class="result"><div><span class="lab">Planning optimisé</span><span class="val">{fr(rl["energie"], 1)}</span><span class="muted">{pct(rl["ecart_pct"], 2)} par rapport au réel simulé ({fr(prs["energie"], 1)}) · ±{NNBSP}{pct(rl["dispersion_pct"], 3, False)} · {int(round(rl["accord_membres"] * pil["membres"]))}{NBSP}membres sur {pil["membres"]}</span></div>
<div class="actions"><button class="btn primary" type="button">{icon("send")}Proposer ce planning</button><button class="btn" type="button">{icon("down")}pilot.planning.csv</button></div></div>
<div class="warn">Un gain de 0,05{NNBSP}% est en dessous de ce que le modèle sait distinguer de la réalité (écart prédit − mesuré : {pct(prs["ecart_au_reel_pct"], 2)}). Il montre la mécanique de l’outil, pas une économie démontrée.</div></section>
<section class="panel kept"><h2>Permutations gardées</h2>
<div class="scroll"><table><thead><tr><th class="num">Itération</th><th class="num">Énergie</th><th class="num">Gain cumulé</th></tr></thead><tbody>{trs}</tbody></table></div>
<div class="note">Gain total : {fr(hist[-1][1] - e0, 2, True)} ({pct(100 * (hist[-1][1] - e0) / e0, 3)}) sur {fr(e0, 1)}.</div></section>
</div></main></div>'''


# ----------------------------------------------------------------------------- CSS et pages
CSS = f"""
*{{box-sizing:border-box}}
.frame{{position:relative;overflow:hidden;display:flex;flex-direction:column;font-size:13px;line-height:1.4;-webkit-font-smoothing:antialiased}}
.frame a{{color:{PRIMARY}}}
.topbar{{height:56px;flex:none;display:flex;align-items:center;gap:20px;padding:0 24px;border-bottom:1px solid {BORDER};background:{PANEL}}}
.brand{{display:flex;align-items:center;gap:10px;font-weight:600;font-size:14px;white-space:nowrap}}
.brand .dot{{width:10px;height:10px;border-radius:50%;background:{PRED}}}
.brand .sub{{color:{MUTED};font-weight:400;font-size:13px}}
.tabs{{display:flex;gap:4px;margin-left:12px}}
.tab{{padding:7px 12px;border-radius:6px;color:{INK2};text-decoration:none;font-weight:500}}
.tab.on{{background:#e9f1fb;color:{PRIMARY}}}
.topright{{margin-left:auto;display:flex;align-items:center;gap:10px;color:{INK2};font-size:12px;white-space:nowrap}}
.chip{{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:999px;border:1px solid {BORDER};background:{PANEL};font-size:11.5px;color:{INK};font-weight:500;white-space:nowrap}}
.chip .sw{{width:8px;height:8px;border-radius:50%;flex:none}}
.chip.ok{{margin-left:8px}}
.main{{flex:1;padding:18px 24px 22px;display:flex;flex-direction:column;gap:14px;min-height:0}}
.titlerow{{display:flex;align-items:flex-end;gap:16px;flex:none}}
h1{{margin:0;font-size:20px;font-weight:600;letter-spacing:-0.01em;line-height:1.25}}
h2{{margin:0;font-size:13px;font-weight:600}}
h3{{margin:6px 0 0;font-size:12px;font-weight:600;color:{INK2}}}
.sub{{color:{INK2};font-size:12.5px;margin-top:2px}}
.muted{{color:{MUTED}}}
.controls{{display:flex;align-items:flex-end;gap:10px;margin-left:auto;flex:none}}
.field{{display:flex;flex-direction:column;gap:4px;font-size:11px;color:{MUTED}}}
.field input,.field select{{height:34px;border:1px solid #d6d4cd;border-radius:6px;padding:0 10px;font:inherit;font-size:13px;color:{INK};background:{PANEL};min-width:150px}}
.btn{{height:36px;padding:0 14px;border-radius:6px;border:1px solid #d6d4cd;background:{PANEL};font:inherit;font-size:13px;font-weight:500;color:{INK};display:inline-flex;align-items:center;gap:8px;cursor:pointer;white-space:nowrap}}
.btn.primary{{background:{PRIMARY};border-color:{PRIMARY};color:#fff}}
.tiles{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;flex:none}}
.tile{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}}
.tile .lab{{font-size:11.5px;color:{INK2}}}
.tile .val{{font-size:26px;font-weight:600;letter-spacing:-0.02em;line-height:1.15}}
.tile .hint{{font-size:11px;color:{MUTED}}}
.cols{{display:flex;gap:14px;flex:1;min-height:0}}
.col-l{{flex:1;min-width:0;display:flex;flex-direction:column;gap:14px}}
.col-r{{width:396px;flex:none;display:flex;flex-direction:column;gap:14px}}
.panel{{background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:10px;min-height:0}}
.phead{{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}}
.phead .legend{{margin-left:auto}}
.note{{font-size:11.5px;color:{INK2};line-height:1.45}}
.warn{{font-size:11.5px;color:{INK};background:#fbf4e4;border:1px solid #f0dfb0;border-radius:6px;padding:8px 10px;line-height:1.45}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:11.5px;color:{INK2};align-items:center}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.legend i.line{{height:2px;border-radius:1px;vertical-align:3px;width:14px}}
table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-weight:500;color:{MUTED};font-size:11px;padding:4px 6px;border-bottom:1px solid {BORDER};white-space:nowrap}}
td{{padding:6px 6px;border-bottom:1px solid #f0eee9;white-space:nowrap}}
td.num,th.num{{text-align:right}}
.bar{{display:inline-block;height:6px;border-radius:3px;background:{PRED};vertical-align:middle;margin-right:6px}}
.facts{{margin:0;padding-left:18px;font-size:12px;color:{INK};line-height:1.5;display:flex;flex-direction:column;gap:6px}}
svg text{{font-family:inherit}}
.axis{{font-size:11px;fill:{MUTED}}}
.scen-list{{width:420px;flex:none}}
.scens{{border:0;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}}
.sr{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}}
.scen{{display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid {BORDER};border-radius:8px;cursor:pointer;background:{PANEL}}}
.scen.sel{{border-color:{PRIMARY};box-shadow:0 0 0 2px #e9f1fb}}
.scen input{{margin:0;width:16px;height:16px;flex:none}}
.scen-l{{display:flex;flex-direction:column;gap:4px;min-width:0;flex:1}}
.scen-l .nm{{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.dots{{display:flex;align-items:center;gap:3px;height:18px}}
.dots i{{width:8px;height:8px;border-radius:50%;background:{GRID};display:inline-block}}
.dots i.on{{background:{INK2}}}
.scen-r{{display:flex;flex-direction:column;align-items:flex-end;gap:2px;flex:none}}
.scen-r .val{{font-weight:600;font-size:15px;font-variant-numeric:tabular-nums}}
.scen-r .muted{{font-size:11px}}
.detail{{width:380px;flex:none}}
.kpi{{display:flex;flex-direction:column;gap:2px;margin-top:4px}}
.kpi .val{{font-size:30px;font-weight:600;letter-spacing:-0.02em;line-height:1.1}}
.kpi .muted{{font-size:11.5px}}
.actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:auto}}
.params{{width:380px;flex:none}}
.vform{{display:flex;flex-direction:column;gap:10px}}
.vform .field input,.vform .field select{{width:100%}}
.check{{display:flex;gap:8px;align-items:flex-start;font-size:12px;line-height:1.4;color:{INK}}}
.check input{{margin:2px 0 0;flex:none;width:15px;height:15px}}
.result{{display:flex;align-items:center;gap:16px;border:1px solid {BORDER};border-radius:8px;padding:10px 14px;background:{SURFACE}}}
.result > div:first-child{{display:flex;flex-direction:column;gap:2px;min-width:0}}
.result .lab{{font-size:11.5px;color:{INK2}}}
.result .val{{font-size:24px;font-weight:600;letter-spacing:-0.02em;line-height:1.15}}
.result .muted{{font-size:11.5px}}
.result .actions{{margin:0 0 0 auto;flex:none}}
.kept{{width:300px;flex:none}}
.ghost{{border:1px dashed #cfcdc6;border-radius:8px;padding:12px;color:{INK2};font-size:12.5px;display:flex;gap:8px;align-items:center;margin-top:6px}}
.steps{{margin:0;padding-left:18px;font-size:12px;line-height:1.5;display:flex;flex-direction:column;gap:6px;color:{INK}}}
.scroll{{overflow:auto;min-height:0;flex:1}}
"""

FONT_LINK = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;display=swap" rel="stylesheet">'


def page_ecran(titre: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1440">
<title>{esc(titre)}</title>
{FONT_LINK}
<style>body{{margin:0;background:{SURFACE}}}{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def page_index(ecrans: list[tuple[str, str, str]], D: dict) -> str:
    frames = "".join(f'<section class="wrap" id="{slug}"><div class="cap"><span class="num">Écran {i + 1}</span><strong>{esc(t)}</strong><span class="muted">{esc(d)}</span></div><div class="scaler"><div class="inner">{body}</div></div></section>'
                     for i, (slug, t, d, body) in enumerate(ecrans))
    nav = "".join(f'<a href="#{slug}">{i + 1}. {esc(t)}</a>' for i, (slug, t, _, _) in enumerate(ecrans))
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maquette du pilote</title>
<meta name="description" content="Maquette de l’interface du pilote de Dream in Orbit : rêver une fenêtre, comparer des plannings, optimiser par recherche locale.">
{FONT_LINK}
<style>
body{{margin:0;background:#edebe6;color:{INK};font-family:Inter,system-ui,sans-serif;font-size:14px;line-height:1.5}}
.chrome{{max-width:1440px;margin:0 auto;padding:28px 16px 8px}}
.chrome h1{{font-size:24px;margin:0 0 6px;letter-spacing:-0.01em}}
.chrome p{{margin:0 0 10px;color:{INK2};max-width:900px}}
.chrome nav{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px}}
.chrome nav a{{color:{PRIMARY};text-decoration:none}}
.wrap{{max-width:1440px;margin:18px auto 0;padding:0 16px}}
.cap{{display:flex;gap:12px;align-items:baseline;font-size:13px;margin-bottom:8px;flex-wrap:wrap}}
.cap .num{{color:{MUTED};font-variant-numeric:tabular-nums}}
.scaler{{width:100%;overflow:hidden;border:1px solid {BORDER};border-radius:10px;box-shadow:0 1px 2px rgba(0,0,0,0.04);background:{SURFACE}}}
.inner{{width:{W_FRAME}px;height:{H_FRAME}px;transform-origin:0 0}}
.foot{{max-width:1440px;margin:24px auto 40px;padding:0 16px;font-size:12.5px;color:{INK2}}}
{CSS}
@page{{size:{W_FRAME}px {H_FRAME}px;margin:0}}
@media print{{body{{background:#fff}}.chrome,.foot,.cap{{display:none}}.wrap{{margin:0;padding:0;max-width:none;page-break-after:always;break-after:page}}.scaler{{border:0;border-radius:0;box-shadow:none;height:{H_FRAME}px!important;overflow:visible}}.inner{{transform:none!important}}}}
</style>
</head>
<body>
<div class="chrome"><h1>Maquette — console de planification du pilote</h1>
<p>Trois écrans de l’outil que le world model rend possible : rêver la consommation d’une fenêtre sous un planning donné, comparer des plannings en comparaison appariée, puis optimiser le planning réel par recherche locale. Tous les chiffres sont les résultats réels du projet (fenêtre du {jour(D["debut_ms"])} au {jour(D["fin_ms"])} 2010, protocole strict, poids <code>weights_holdout</code>, <code>reports/pilot.json</code>).</p>
<nav>{nav}</nav></div>
{frames}
<div class="foot">Maquette statique construite par <code>maquette/build_maquette.py</code> à partir des fichiers du dépôt ; les écrans existent aussi en PNG et en PDF dans ce dossier. Source du planning affiché : {esc(D["source_plan"])}.</div>
<script>
(function(){{function fit(){{var s=Math.min(1,(document.documentElement.clientWidth-32)/{W_FRAME});var ws=document.querySelectorAll('.scaler');for(var i=0;i<ws.length;i++){{ws[i].style.height=({H_FRAME}*s)+'px';ws[i].firstElementChild.style.transform='scale('+s+')';}}}}
window.addEventListener('resize',fit);fit();}})();
</script>
</body>
</html>
"""


def artboard(titre: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{esc(titre)}</title>
<script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;display=swap" rel="stylesheet">
<style>
body{{margin:0}}
a{{color:{PRIMARY}}}a:hover{{color:#104281}}
{CSS}
</style>
</helmet>
{body}
</x-dc>
<script type="text/x-dc" data-dc-script data-props='{{"$preview":{{"width":{W_FRAME},"height":{H_FRAME}}}}}'>
class Component extends DCLogic {{
  renderVals() {{
    return {{}};
  }}
}}
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", default=None, help="dossier où écrire project/canvas.json et les artboards .dc.html")
    a = ap.parse_args()
    D = charger()
    ecrans = [
        ("ecran-1-reve", "Rêve sur une fenêtre", "le planning réel simulé en boucle ouverte, avec la bande d’incertitude et le planning aligné", ecran_reve(D)),
        ("ecran-2-scenarios", "Comparer des plannings", "quatre scénarios « et si » en comparaison appariée, membre par membre", ecran_scenarios(D)),
        ("ecran-3-optimiseur", "Optimiser par recherche locale", "120 permutations d’heures de pointage, 15 gardées", ecran_optimiseur(D)),
    ]
    (OUT / "ecrans").mkdir(parents=True, exist_ok=True)
    for slug, t, _, body in ecrans:
        (OUT / "ecrans" / f"{slug}.html").write_text(page_ecran(f"{t} — maquette du pilote", body), encoding="utf-8")
    (OUT / "index.html").write_text(page_index(ecrans, D), encoding="utf-8")
    print(f"maquette/index.html et {len(ecrans)} écrans écrits ; planning : {D['source_plan']} ; RMSE fenêtre {D['rmse_fen']:.4f} ; couverture ±2σ {100 * D['couverture']:.0f} %")
    if a.canvas:
        root = Path(a.canvas) / "project"
        root.mkdir(parents=True, exist_ok=True)
        noms = ["Main.dc.html", "Scenarios.dc.html", "Optimiseur.dc.html"]
        boards, order = {}, []
        for i, ((slug, t, _, body), nom) in enumerate(zip(ecrans, noms)):
            (root / nom).write_text(artboard(f"Écran {i + 1} — {t}", body), encoding="utf-8")
            boards[nom] = {"x": i * (W_FRAME + 80), "y": 0, "w": W_FRAME, "h": H_FRAME, "title": f"Écran {i + 1} — {t}"}
            order.append(nom)
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        canvas = {"v": 3, "createdOnFiles": {"v": 1, "at": now}, "title": "Maquette du pilote", "launch": {"view": "canvas"},
                  "pages": [], "boards": boards, "order": order,
                  "notes": {"titre": {"x": 0, "y": -300, "text": "Console de planification du pilote — trois écrans", "kind": "title1", "maxW": 3 * W_FRAME + 160},
                            "contexte": {"x": 0, "y": H_FRAME + 60, "w": 560, "text": "Chiffres réels du projet : fenêtre du 25 oct. au 7 nov. 2010, protocole strict (annee1 → annee2), poids weights_holdout, reports/pilot.json. Les boutons et champs montrent l’usage prévu ; rien n’est câblé.", "fill": "blue"}},
                  "designSystems": []}
        (root / "canvas.json").write_text(json.dumps(canvas, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"toile : {root}")


if __name__ == "__main__":
    main()
