"""
Diagnostic systematique du creux de NPWD2551 pendant la deuxieme saison d'eclipses de annee2 (2011) :
le simulateur, entraine sur annee1, le sur-predit de ~0,3 pendant plusieurs semaines.

Questions posees aux donnees du kit, sans a priori :
  1. ou et quand : semaines de biais > +0,10 du melange livre (weights_holdout/val_pred.csv) ;
  2. a umbra egale, NPWD2551 consomme-t-il moins en annee2 qu'en annee1 sur la meme saison (jours 400-490) ?
  3. quelles colonnes de contexte / planning (saaf, ltdata, evtf, ftl, comptes dmop par sous-systeme et par stem)
     different le plus entre les deux saisons (difference standardisee) ?
  4. d'autres circuits changent-ils de niveau en meme temps (signature d'un changement de configuration) ?
  5. point de rupture dans annee2 : quels stems dmop apparaissent dans les 48 h autour du decrochage et sont rares ailleurs ?

    python experiments/diag_2551.py        -> reports/diag_2551.json, reports/diag_2551.md
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dio.data import get_vocab, load_period
from dio.features import _read_dmop, split_stem

N = 168; C = "NPWD2551"


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Table markdown sans dependance (tabulate absent)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r.tolist()) + " |")
    return "\n".join(lines)
kit = Path("kit_etudiants"); v = get_vocab(kit, ["annee1", "annee2"])
X1, Y1 = load_period(kit / "annee1", v); X2, Y2 = load_period(kit / "annee2", v)
out = {}

# 1. biais par semaine du melange livre (protocole strict)
pred = pd.read_csv("weights_holdout/val_pred.csv", index_col="ut_ms")[C].reindex(Y2.index)
bias = (pred - Y2[C]).iloc[N:]
wk = (np.arange(len(bias)) // N)
bw = bias.groupby(wk).mean()
bad = bw[bw > 0.10]
out["semaines_biais_sup_0_10"] = {int(k): round(float(x), 3) for k, x in bad.items()}
t2 = pd.to_datetime(Y2.index, unit="ms")
if len(bad):
    a, b = int(bad.index.min()), int(bad.index.max())
    out["fenetre_anormale"] = {"debut": str(t2[N + a * N].date()), "fin": str(t2[min(len(t2) - 1, N + (b + 1) * N)].date()),
                               "biais_moyen": round(float(bias.iloc[a * N:(b + 1) * N].mean()), 3)}

# 2. meme saison (jours 400-490) dans les deux annees : NPWD2551 en fonction de l'umbra
def season(X, Y, d0=400, d1=490):
    sl = slice(d0 * 24, d1 * 24); return X.iloc[sl], Y.iloc[sl]
Xs1, Ys1 = season(X1, Y1); Xs2, Ys2 = season(X2, Y2)
bins = [-0.001, 0.0, 0.1, 0.3, 0.6, 1.0]
def by_umbra(Xs, Ys):
    g = pd.cut(Xs["ev_umbra"], bins); return Ys[C].groupby(g, observed=True).agg(["mean", "count"])
u1, u2 = by_umbra(Xs1, Ys1), by_umbra(Xs2, Ys2)
tab = pd.concat([u1.add_suffix("_a1"), u2.add_suffix("_a2")], axis=1)
tab["ecart_a2_moins_a1"] = tab["mean_a2"] - tab["mean_a1"]
out["npwd2551_par_umbra_saison2"] = {str(k): {kk: (round(float(vv), 3) if pd.notna(vv) else None) for kk, vv in r.items()} for k, r in tab.iterrows()}
out["umbra_moyen_saison2"] = {"annee1": round(float(Xs1["ev_umbra"].mean()), 4), "annee2": round(float(Xs2["ev_umbra"].mean()), 4)}

# 3. colonnes de contexte / planning qui different le plus entre les deux saisons
base_cols = [c for c in X1.columns if c.startswith(("saaf_", "lt_", "ev_", "ftl_", "dm_p_", "dm_sc_")) or c == "dm_total"]
rows = []
for c in base_cols:
    a, b = Xs1[c].to_numpy(dtype=float), Xs2[c].to_numpy(dtype=float)
    sd = np.sqrt((a.var() + b.var()) / 2)
    if sd < 1e-9:
        continue
    rows.append((c, float(a.mean()), float(b.mean()), float((b.mean() - a.mean()) / sd)))
diff = pd.DataFrame(rows, columns=["colonne", "moyenne_a1", "moyenne_a2", "d_standardise"]).sort_values("d_standardise", key=np.abs, ascending=False)
out["colonnes_les_plus_differentes"] = diff.head(25).round(4).to_dict(orient="records")

# 4. autres circuits : niveau a umbra egale (heures sans umbra, et heures d'umbra > 0,3) dans les deux saisons
rows = []
for c in Y1.columns:
    r = {"circuit": c}
    for name, m1, m2 in [("hors_umbra", Xs1["ev_umbra"] == 0, Xs2["ev_umbra"] == 0), ("umbra_sup_0_3", Xs1["ev_umbra"] > 0.3, Xs2["ev_umbra"] > 0.3)]:
        r[f"{name}_a1"] = float(Ys1.loc[m1, c].mean()); r[f"{name}_a2"] = float(Ys2.loc[m2, c].mean())
        r[f"{name}_ecart"] = r[f"{name}_a2"] - r[f"{name}_a1"]
    rows.append(r)
shift = pd.DataFrame(rows).sort_values("umbra_sup_0_3_ecart", key=np.abs, ascending=False)
out["circuits_qui_changent_de_niveau_saison2"] = shift.head(10).round(3).to_dict(orient="records")

# 5. point de rupture dans annee2 : premiere journee de la saison ou la moyenne de NPWD2551 aux heures d'umbra
#    passe durablement sous la courbe de annee1 ; stems dmop dans les 48 h autour, rares ailleurs
d = Ys2[C].where(Xs2["ev_umbra"] > 0.3).groupby(pd.to_datetime(Ys2.index, unit="ms").date).mean()
ref = float(Ys1.loc[Xs1["ev_umbra"] > 0.3, C].mean())
below = d[d < ref - 0.15]
rupture = None
if len(below):
    for day in below.index:
        later = d[d.index >= day]
        if len(later) >= 5 and (later.iloc[:5] < ref - 0.15).all():
            rupture = day; break
out["rupture"] = {"reference_annee1_umbra_sup_0_3": round(ref, 3), "jour": str(rupture) if rupture else None}
if rupture is not None:
    t0 = pd.Timestamp(rupture).value // 10 ** 6
    dm = _read_dmop(kit / "annee2" / "dmop.csv"); dm = dm[~dm["cmd"].str.contains(r"\.", regex=True)]
    dm["stem"] = [split_stem(c)[0] for c in dm["cmd"]]
    win = dm[(dm["ut_ms"] >= t0 - 48 * 3600000) & (dm["ut_ms"] < t0 + 48 * 3600000)]
    tot = dm["stem"].value_counts(); w = win["stem"].value_counts()
    rate = (w / tot.reindex(w.index)).sort_values(ascending=False)   # part de toutes les occurrences de l'annee tombant dans ces 96 h
    out["stems_concentres_autour_de_la_rupture"] = [{"stem": s, "dans_la_fenetre": int(w[s]), "sur_l_annee": int(tot[s]), "part": round(float(rate[s]), 3)} for s in rate.index[:15]]
    ev = pd.read_csv(kit / "annee2" / "evtf.csv"); ev = ev[(ev["ut_ms"] >= t0 - 48 * 3600000) & (ev["ut_ms"] < t0 + 48 * 3600000)]
    out["evenements_evtf_autour_de_la_rupture"] = ev["description"].astype(str).str.replace(r"\d+", "#", regex=True).value_counts().head(15).to_dict()

# 6. ce que le planning de annee2 contient que annee1 n'avait pas : trous dans dmop.csv, mode de pointage D4PNPO,
#    famille de commandes AACF ; et table journaliere autour de la rupture
H = 3600000
out["planning"] = {}
for yr, X in [("annee1", X1), ("annee2", X2)]:
    dm = pd.read_csv(kit / yr / "dmop.csv"); t = np.sort(dm["ut_ms"].to_numpy()); gaps = np.diff(t)
    big = np.where(gaps > 24 * H)[0]
    out["planning"][yr] = {"trous_dmop_sup_24h": [{"debut": str(pd.to_datetime(t[i], unit="ms").date()), "jours": round(float(gaps[i] / H / 24), 1)} for i in big],
                           "heures_D4PNPO": int((X["ftl_D4PNPO"] > 0).sum()) if "ftl_D4PNPO" in X.columns else 0,
                           "commandes_AACF": int(dm["subsystem"].astype(str).str.strip('"').str.startswith("AACF").sum())}
if rupture is not None:
    t2i = pd.to_datetime(X2.index, unit="ms"); r = pd.Timestamp(rupture)
    w = (t2i >= r - pd.Timedelta(days=8)) & (t2i < r + pd.Timedelta(days=10))
    daily = pd.DataFrame({"NPWD2551": Y2[C].to_numpy(), "umbra": X2["ev_umbra"].to_numpy(), "D4PNPO": X2["ftl_D4PNPO"].to_numpy(),
                          "comms": X2["ftl_comms"].to_numpy(), "commandes_par_h": X2["dm_total"].to_numpy()}, index=t2i)[w].resample("D").mean().round(3)
    daily.index = daily.index.date
    out["journalier_autour_rupture"] = {str(k): v for k, v in daily.to_dict(orient="index").items()}
rows = []
for nm, Xs, Ys in [("annee1", Xs1, Ys1), ("annee2", Xs2, Ys2)]:
    cols = [c for c in Xs.columns if c.startswith("ftl_") and c not in ("ftl_comms", "ftl_n_switch", "ftl_other")]
    mode = Xs[cols].idxmax(axis=1).str.replace("ftl_", ""); m = Xs["ev_umbra"] > 0.3
    for md_, g in Ys.loc[m, C].groupby(mode[m]):
        rows.append({"annee": nm, "mode": md_, "NPWD2551_umbra_sup_0_3": round(float(g.mean()), 3), "heures": int(len(g))})
out["npwd2551_par_mode_en_umbra"] = rows

Path("reports/diag_2551.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
lines = ["# Diagnostic NPWD2551, saison d'eclipses 2 (jours 400-490)", "",
         f"Fenetre de biais > 0,10 : {out.get('fenetre_anormale')}", "", "## NPWD2551 par tranche d'umbra (moyenne, effectif)", "",
         md(tab.round(3)), "", "## Colonnes de planning / contexte les plus differentes entre les deux saisons", "",
         md(diff.head(25).round(4), index=False), "", "## Circuits dont le niveau change (heures d'umbra > 0,3)", "",
         md(shift.head(10).round(3), index=False), "", f"## Rupture : {out['rupture']}", ""]
lines += ["## Ce que le planning de annee2 contient que annee1 n'avait pas", "", "```", json.dumps(out["planning"], indent=1, ensure_ascii=False), "```", ""]
if "journalier_autour_rupture" in out:
    lines += ["## Moyennes journalieres autour de la rupture", "", md(daily), ""]
lines += ["## NPWD2551 en umbra > 0,3 selon le mode de pointage, saison 2", "", md(pd.DataFrame(rows), index=False), ""]
if "stems_concentres_autour_de_la_rupture" in out:
    lines += [md(pd.DataFrame(out["stems_concentres_autour_de_la_rupture"]), index=False), "",
              "Evenements evtf (48 h autour) : " + json.dumps(out["evenements_evtf_autour_de_la_rupture"], ensure_ascii=False)]
Path("reports/diag_2551.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:40]))
