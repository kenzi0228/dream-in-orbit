"""
Construction des features horaires a partir des 5 fichiers de contexte du kit.

Tout est aligne sur une grille d'heures (debut d'heure en ms UTC, comme evaluate.py).
Chaque feature decrit ce qui se passe pendant l'heure [h, h + 1h) :
    - saaf   : angles du Soleil (moyenne horaire)
    - ltdata : distances / constante solaire (interpolation lineaire des valeurs journalieres)
    - evtf   : fraction d'heure en ombre (umbra / penumbra), en occultation, en contact
               radio, phase orbitale et altitude (interpolee entre les evenements)
    - ftl    : fraction d'heure dans chaque mode de pointage + communications
    - dmop   : nombre de commandes par sous-systeme / par commande frequente,
               recence (heures depuis la derniere emission) et intervalles start/end

Le vocabulaire (commandes dmop frequentes, modes ftl) est appris sur le train et
re-utilise tel quel sur le test, pour que les colonnes soient identiques.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

H = 3600 * 1000  # une heure en millisecondes
FEATURE_VERSION = 7  # a incrementer quand on change les colonnes (invalide le cache)

# Altitudes (km) associees aux evenements orbitaux d'evtf, pour reconstruire une
# altitude approximative. Pericentre / apocentre de Mars Express ~ 300 km / 10 100 km.
ALT_PERI_KM = 330.0
ALT_APO_KM = 10100.0

MIN_COUNT_CMD = 50   # une commande entre dans le vocabulaire si elle apparait >= 50 fois
MAX_CMDS = 250       # et on garde au plus 250 commandes
MIN_COUNT_STEM = 50  # un stem (commande sans sa lettre de variante) entre dans le vocabulaire s'il apparait >= 50 fois (20 : 161 stems, GRU moins bon)
RECENCY_CAP_H = 24 * 30


# --------------------------------------------------------------------------- utils

def hourly_grid(start_ms: int, end_ms: int) -> np.ndarray:
    """Heures (debut d'heure) de floor(start) a floor(end) inclus."""
    a = (int(start_ms) // H) * H
    b = (int(end_ms) // H) * H
    return np.arange(a, b + H, H, dtype=np.int64)


def interval_fraction(starts: np.ndarray, ends: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Fraction de chaque heure de `grid` couverte par l'union des intervalles [s, e)."""
    if len(starts) == 0:
        return np.zeros(len(grid))
    order = np.argsort(starts)
    s, e = np.asarray(starts, dtype=np.int64)[order], np.asarray(ends, dtype=np.int64)[order]
    # fusion des intervalles qui se chevauchent
    ms, me = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a)
            me.append(b)
    ms, me = np.array(ms), np.array(me)
    lengths = me - ms
    cum = np.concatenate([[0], np.cumsum(lengths)])

    def coverage(t):
        i = np.searchsorted(ms, t, side="right") - 1
        base = cum[np.clip(i, 0, len(ms))]
        inside = np.where(i >= 0, np.clip(t - ms[np.clip(i, 0, len(ms) - 1)], 0, lengths[np.clip(i, 0, len(ms) - 1)]), 0)
        return base + inside

    return (coverage(grid + H) - coverage(grid)) / H


def pair_intervals(df: pd.DataFrame, start_mask, end_mask) -> tuple[np.ndarray, np.ndarray]:
    """Apparie des evenements START / END dans l'ordre chronologique (un START ouvre, le END suivant ferme)."""
    ev = df[start_mask | end_mask].sort_values("ut_ms")
    is_start = start_mask[ev.index].to_numpy()
    t = ev["ut_ms"].to_numpy()
    starts, ends, open_t = [], [], None
    for ti, st in zip(t, is_start):
        if st:
            open_t = ti
        elif open_t is not None:
            starts.append(open_t)
            ends.append(ti)
            open_t = None
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def _reindex_interp(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Reindexe une serie horaire sur la grille, interpole les trous, remplit les bords."""
    out = df.reindex(df.index.union(grid)).interpolate(method="index", limit_direction="both")
    return out.reindex(grid).ffill().bfill()


# --------------------------------------------------------------------------- saaf

def features_saaf(path: Path, grid: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["h"] = (df["ut_ms"] // H) * H
    cols = ["sa", "sx", "sy", "sz"]
    for c in cols:
        df[f"cos_{c}"] = np.cos(np.deg2rad(df[c]))
    g = df.groupby("h")
    mean = g[cols + [f"cos_{c}" for c in cols]].mean()
    mean.columns = [f"saaf_{c}" for c in mean.columns]
    std = g[["sa"]].std().fillna(0)
    std.columns = ["saaf_sa_std"]
    out = pd.concat([mean, std], axis=1)
    return _reindex_interp(out, grid)


# --------------------------------------------------------------------------- ltdata

def features_ltdata(path: Path, grid: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("ut_ms").sort_index()
    df.columns = [f"lt_{c}" for c in df.columns]
    return _reindex_interp(df, grid)


# --------------------------------------------------------------------------- evtf

_RE_KM = re.compile(r"^(\d+)_KM_(ASCEND|DESCEND)$")
_RE_STATION = re.compile(r"^([A-Z0-9]{3})_(AOS|LOS)_")


def features_evtf(path: Path, grid: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = df["description"].astype(str)
    out = pd.DataFrame(index=grid)

    def frac(start_mask, end_mask):
        s, e = pair_intervals(df, start_mask, end_mask)
        return interval_fraction(s, e, grid)

    out["ev_umbra"] = frac(d == "MAR_UMBRA_START", d == "MAR_UMBRA_END")
    out["ev_penumbra"] = frac(d == "MAR_PENUMBRA_START", d == "MAR_PENUMBRA_END")
    moon_s = d.str.match(r"^(PHO|DEI)_(UMBRA|PENUMBRA)_START$")
    moon_e = d.str.match(r"^(PHO|DEI)_(UMBRA|PENUMBRA)_END$")
    out["ev_moon_shadow"] = frac(moon_s, moon_e)
    out["ev_occ_mars"] = frac(d.str.match(r"^OCC_MARS_START"), d.str.match(r"^OCC_MARS_END"))
    out["ev_occ_mars_km"] = frac(d.str.match(r"^OCC_MARS_\d+KM_START"), d.str.match(r"^OCC_MARS_\d+KM_END"))
    out["ev_mrb_range"] = frac(d.str.match(r"^MRB_/_RANGE_\d+KM_START"), d.str.match(r"^MRB_/_RANGE_\d+KM_END"))

    # contacts avec les stations sol : union et somme des couvertures par station
    union_s, union_e, total = [], [], np.zeros(len(grid))
    stations = sorted({m.group(1) for m in (_RE_STATION.match(x) for x in d) if m})
    for st in stations:
        s, e = pair_intervals(df, d.str.startswith(f"{st}_AOS_"), d.str.startswith(f"{st}_LOS_"))
        total += interval_fraction(s, e, grid)
        union_s.append(s)
        union_e.append(e)
    if union_s:
        out["ev_gs_any"] = interval_fraction(np.concatenate(union_s), np.concatenate(union_e), grid)
    else:
        out["ev_gs_any"] = 0.0
    out["ev_gs_sum"] = total

    # phase orbitale a partir des passages au pericentre / apocentre
    peri = np.sort(df.loc[d.str.startswith("PERICENTRE_PASSAGE"), "ut_ms"].to_numpy(dtype=np.int64))
    apo = np.sort(df.loc[d.str.startswith("APOCENTRE_PASSAGE"), "ut_ms"].to_numpy(dtype=np.int64))
    mid = grid + H // 2
    if len(peri) >= 2:
        i = np.searchsorted(peri, mid, side="right") - 1
        prev_p = peri[np.clip(i, 0, len(peri) - 1)]
        next_p = peri[np.clip(i + 1, 0, len(peri) - 1)]
        period = np.where(next_p > prev_p, next_p - prev_p, np.median(np.diff(peri)))
        since = np.where(i >= 0, mid - prev_p, 0)
        phase = np.clip(since / period, 0, 1)
        out["ev_since_peri_h"] = since / H
        out["ev_to_peri_h"] = np.clip(next_p - mid, 0, None) / H
        out["ev_orbit_phase_cos"] = np.cos(2 * np.pi * phase)
        out["ev_orbit_phase_sin"] = np.sin(2 * np.pi * phase)
        out["ev_orbit_period_h"] = period / H
    else:
        for c in ["ev_since_peri_h", "ev_to_peri_h", "ev_orbit_phase_cos", "ev_orbit_phase_sin", "ev_orbit_period_h"]:
            out[c] = 0.0

    # altitude approximative : interpolation de log(altitude) entre les evenements connus
    km_t, km_v = [], []
    for t, desc in zip(df["ut_ms"].to_numpy(), d):
        m = _RE_KM.match(desc)
        if m:
            km_t.append(t)
            km_v.append(float(m.group(1)))
    km_t = np.concatenate([np.array(km_t, dtype=np.int64), peri, apo])
    km_v = np.concatenate([np.array(km_v), np.full(len(peri), ALT_PERI_KM), np.full(len(apo), ALT_APO_KM)])
    if len(km_t) >= 2:
        order = np.argsort(km_t)
        km_t, km_v = km_t[order], np.log(km_v[order])
        sub = np.concatenate([grid + k * H // 4 + H // 8 for k in range(4)])  # 4 sous-points par heure
        alt = np.interp(sub, km_t, km_v).reshape(4, len(grid)).mean(axis=0)
        out["ev_log_alt"] = alt
    else:
        out["ev_log_alt"] = np.log(ALT_APO_KM)
    out["ev_n_events"] = pd.Series(1, index=(df["ut_ms"] // H) * H).groupby(level=0).sum().reindex(grid).fillna(0).to_numpy()
    return out


# --------------------------------------------------------------------------- ftl

def features_ftl(path: Path, grid: np.ndarray, types: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["type"] = df["type"].astype(str).str.strip('"')
    out = pd.DataFrame(index=grid)
    known = np.zeros(len(grid))
    for t in types:
        m = df["type"] == t
        f = interval_fraction(df.loc[m, "utb_ms"].to_numpy(), df.loc[m, "ute_ms"].to_numpy(), grid)
        out[f"ftl_{t}"] = f
        known += f
    out["ftl_other"] = np.clip(1 - known, 0, 1)
    comms = df["flagcomms"].astype(str).str.upper().str.contains("TRUE")
    out["ftl_comms"] = interval_fraction(df.loc[comms, "utb_ms"].to_numpy(), df.loc[comms, "ute_ms"].to_numpy(), grid)
    # nombre de changements de pointage dans l'heure (manoeuvres)
    out["ftl_n_switch"] = pd.Series(1, index=(df["utb_ms"] // H) * H).groupby(level=0).sum().reindex(grid).fillna(0).to_numpy()
    return out


def ftl_types(path: Path, until_ms: int | None = None) -> list[str]:
    df = pd.read_csv(path)
    if until_ms is not None:
        df = df[df["utb_ms"] < until_ms]
    return sorted(df["type"].astype(str).str.strip('"').unique())


# --------------------------------------------------------------------------- dmop

_PAIR_PREFIXES = {"PENS": "PENE", "MOCS": "MOCE", "PPNS": "PPNE", "PDNS": "PDNE", "UPBS": "UPBE", "UDBS": "UDBE"}


def _read_dmop(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cmd"] = df["subsystem"].astype(str).str.strip('"')
    df["prefix"] = df["cmd"].str[:4]
    df["h"] = (df["ut_ms"] // H) * H
    return df


_RE_LETTER = re.compile(r"[A-Z]")


def split_stem(cmd: str) -> tuple[str, str]:
    """AAAAF20C1 -> ('AAAAF20_1', 'C') : la derniere lettre (apres le sous-systeme) est la variante
    de la commande (souvent ON / OFF, ou un mode) ; le reste est le 'stem'."""
    m = list(_RE_LETTER.finditer(cmd[4:]))
    if not m:
        return cmd, ""
    k = 4 + m[-1].start()
    return cmd[:k] + "_" + cmd[k + 1:], cmd[k]


def dmop_vocab(path: Path, until_ms: int | None = None) -> dict:
    df = _read_dmop(path)
    if until_ms is not None:
        df = df[df["h"] < until_ms]  # vocabulaire appris sur la periode d'entrainement seulement
    plain = df[~df["cmd"].str.contains(r"\.", regex=True)]
    counts = plain["cmd"].value_counts()
    cmds = counts[counts >= MIN_COUNT_CMD].index.tolist()[:MAX_CMDS]
    prefixes = sorted(df["prefix"].unique().tolist())
    # interrupteurs : stems ayant >= 2 variantes, chacune emise >= MIN_COUNT_CMD fois
    sv = pd.DataFrame([split_stem(c) for c in plain["cmd"]], columns=["stem", "var"])
    cnt = sv.groupby(["stem", "var"]).size()
    cnt = cnt[cnt >= MIN_COUNT_CMD]
    stems = {}
    for (st, va), _ in cnt.items():
        stems.setdefault(st, []).append(va)
    switches = sorted([st, va] for st, vs in stems.items() if len(vs) >= 2 for va in vs)
    stem_counts = sv["stem"].value_counts()
    stems_freq = sorted(stem_counts[stem_counts >= MIN_COUNT_STEM].index.tolist())
    return {"cmds": sorted(cmds), "prefixes": prefixes, "switches": switches, "stems": stems_freq}


def features_dmop(path: Path, grid: np.ndarray, vocab: dict) -> pd.DataFrame:
    df = _read_dmop(path)
    out = pd.DataFrame(index=grid)
    grid_s = pd.Series(np.arange(len(grid)), index=grid)

    # comptes par prefixe de sous-systeme
    for p in vocab["prefixes"]:
        c = df[df["prefix"] == p].groupby("h").size()
        out[f"dm_p_{p}"] = c.reindex(grid).fillna(0).to_numpy()
    out["dm_total"] = df.groupby("h").size().reindex(grid).fillna(0).to_numpy()

    # comptes + recence par commande frequente
    cmd_set = set(vocab["cmds"])
    sub = df[df["cmd"].isin(cmd_set)]
    counts = sub.groupby(["h", "cmd"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=grid, columns=vocab["cmds"], fill_value=0)
    cnt = counts.to_numpy(dtype=np.float32)
    # recence : heures ecoulees depuis la derniere occurrence au debut de l'heure
    rec = np.full(cnt.shape, RECENCY_CAP_H, dtype=np.float32)
    last = np.full(cnt.shape[1], -np.inf)
    for i in range(len(grid)):
        rec[i] = np.minimum(i - last, RECENCY_CAP_H)
        last = np.where(cnt[i] > 0, i, last)
    cnt_df = pd.DataFrame(cnt, index=grid, columns=[f"dm_c_{c}" for c in vocab["cmds"]])
    rec_df = pd.DataFrame(np.log1p(rec), index=grid, columns=[f"dm_r_{c}" for c in vocab["cmds"]])

    # recence par sous-systeme (prefixe) et par stem, toutes variantes confondues :
    # "depuis combien d'heures le sous-systeme X / la commande X (quelle que soit sa lettre)
    # n'a pas ete sollicite". Robuste aux changements de variante d'une annee a l'autre.
    def recency_matrix(keys: pd.Series, vocab_keys: list[str], prefix: str) -> pd.DataFrame:
        if not vocab_keys:
            return pd.DataFrame(index=grid)
        c = pd.DataFrame({"h": df["h"], "k": keys})
        c = c[c["k"].isin(set(vocab_keys))].groupby(["h", "k"]).size().unstack(fill_value=0)
        c = c.reindex(index=grid, columns=vocab_keys, fill_value=0).to_numpy(dtype=np.float32)
        r = np.full(c.shape, RECENCY_CAP_H, dtype=np.float32)
        lst = np.full(c.shape[1], -np.inf)
        for i in range(len(grid)):
            r[i] = np.minimum(i - lst, RECENCY_CAP_H)
            lst = np.where(c[i] > 0, i, lst)
        return pd.DataFrame(np.log1p(r), index=grid, columns=[f"{prefix}{k}" for k in vocab_keys])

    stems_all = pd.Series([split_stem(c)[0] for c in df["cmd"]], index=df.index)
    rec_prefix = recency_matrix(df["prefix"], vocab.get("prefixes", []), "dm_pr_")
    rec_stem = recency_matrix(stems_all, vocab.get("stems", []), "dm_st_")

    # etat des interrupteurs : pour chaque stem, quelle variante a ete emise en dernier
    # (1 pour la variante active, 0 pour les autres ; tout a 0 avant la premiere emission)
    sw = pd.DataFrame(index=grid)
    switches = vocab.get("switches", [])
    if switches:
        stems = {}
        for st, va in switches:
            stems.setdefault(st, []).append(va)
        sv = pd.DataFrame([split_stem(c) for c in df["cmd"]], columns=["stem", "var"], index=df.index)
        df2 = pd.concat([df, sv], axis=1)
        for st, vs in stems.items():
            sub = df2[(df2["stem"] == st) & (df2["var"].isin(vs))].sort_values("ut_ms")
            # derniere variante emise avant la fin de chaque heure (code entier, -1 = jamais emise)
            codes = sub["var"].map({va: i for i, va in enumerate(vs)}).astype(float)
            last = codes.groupby(sub["h"]).last().reindex(grid).ffill().fillna(-1).to_numpy()
            for i, va in enumerate(vs):
                sw[f"dm_s_{st}{va}"] = (last == i).astype(np.float32)

    # comptes horaires par stem (toutes lettres confondues) : robustes au changement de lettre d'une annee a l'autre
    sc = pd.DataFrame(index=grid); sn = pd.DataFrame(index=grid)
    stems_vocab = vocab.get("stems", [])
    if stems_vocab:
        c = pd.DataFrame({"h": df["h"], "k": stems_all})
        c = c[c["k"].isin(set(stems_vocab))].groupby(["h", "k"]).size().unstack(fill_value=0)
        c = c.reindex(index=grid, columns=stems_vocab, fill_value=0).to_numpy(dtype=np.float32)
        sc = pd.DataFrame(c, index=grid, columns=[f"dm_sc_{k}" for k in stems_vocab])
        # commandes a venir (le planning est connu) : compte du stem dans l'heure suivante et sur les 6 heures suivantes
        n = len(grid); z = np.zeros((1, c.shape[1]), np.float32)
        sn1 = np.vstack([c[1:], z])
        cs = np.vstack([z, np.cumsum(c, axis=0)])
        idx = np.arange(n); sn6 = cs[np.minimum(idx + 7, n)] - cs[idx + 1]
        sn = pd.DataFrame(np.hstack([sn1, sn6]).astype(np.float32), index=grid,
                          columns=[f"dm_sn1_{k}" for k in stems_vocab] + [f"dm_sn6_{k}" for k in stems_vocab])

    # interrupteurs sans les lettres : pour chaque stem d'interrupteur, toutes variantes confondues (meme une
    # lettre inconnue du vocabulaire), heures depuis le dernier changement de variante (dm_sw_X_chg) et
    # heures jusqu'au prochain changement (dm_sw_X_next, le planning est connu), en log1p, plafonnees.
    rob = pd.DataFrame(index=grid)
    if switches:
        n = len(grid); idx = np.arange(n); cap = float(RECENCY_CAP_H)
        for st in stems:
            sub = df2[df2["stem"] == st].sort_values("ut_ms")
            vars_ = sub["var"].to_numpy()
            change = np.r_[False, vars_[1:] != vars_[:-1]] if len(vars_) else np.zeros(0, dtype=bool)
            pos = np.searchsorted(grid, sub["h"].to_numpy()[change], side="right") - 1
            pos = np.unique(pos[(pos >= 0) & (pos < n)])
            last = np.full(n, -np.inf); nxt = np.full(n, np.inf)
            if len(pos):
                last[pos] = pos; last = np.maximum.accumulate(last)
                nxt[pos] = pos; nxt = np.minimum.accumulate(nxt[::-1])[::-1]
            since = np.minimum(idx - last, cap)
            until = np.minimum(np.where(np.isfinite(nxt), nxt - idx, cap), cap)
            rob[f"dm_sw_{st}_chg"] = np.log1p(since).astype(np.float32)
            rob[f"dm_sw_{st}_next"] = np.log1p(np.maximum(until, 0)).astype(np.float32)

    # intervalles start / end apparies par identifiant (PENS.000123 ... PENE.000123)
    pair = pd.DataFrame(index=grid)
    for ps, pe in _PAIR_PREFIXES.items():
        s = df[df["prefix"] == ps]
        e = df[df["prefix"] == pe]
        if len(s) == 0 or len(e) == 0:
            pair[f"dm_act_{ps}"] = 0.0
            continue
        s_id = s["cmd"].str.split(".").str[-1]
        e_id = e["cmd"].str.split(".").str[-1]
        m = pd.merge(pd.DataFrame({"id": s_id.values, "s": s["ut_ms"].values}),
                     pd.DataFrame({"id": e_id.values, "e": e["ut_ms"].values}), on="id")
        m = m[m["e"] > m["s"]]
        pair[f"dm_act_{ps}"] = interval_fraction(m["s"].to_numpy(), m["e"].to_numpy(), grid)
    return pd.concat([out, pair, sw, cnt_df, rec_df, rec_prefix, rec_stem, sc, sn, rob], axis=1)


# --------------------------------------------------------------------------- assemblage

def build_vocab(ctx_dirs: list[Path], until_ms: int | None = None) -> dict:
    """Vocabulaire appris sur un ou plusieurs dossiers de contexte (union), limite aux heures < until_ms :
    les noms de colonnes (commandes, stems, interrupteurs, types de pointage) ne doivent rien a la periode de test."""
    cmds, prefixes, types, switches, stems = set(), set(), set(), set(), set()
    for d in ctx_dirs:
        v = dmop_vocab(d / "dmop.csv", until_ms)
        cmds |= set(v["cmds"])
        prefixes |= set(v["prefixes"])
        switches |= {tuple(x) for x in v["switches"]}
        stems |= set(v["stems"])
        types |= set(ftl_types(d / "ftl.csv", until_ms))
    return {"cmds": sorted(cmds), "prefixes": sorted(prefixes), "switches": sorted(list(x) for x in switches),
            "stems": sorted(stems), "ftl_types": sorted(types)}


def build_features(ctx_dir: Path, grid: np.ndarray, vocab: dict) -> pd.DataFrame:
    ctx_dir = Path(ctx_dir)
    parts = [
        features_saaf(ctx_dir / "saaf.csv", grid),
        features_ltdata(ctx_dir / "ltdata.csv", grid),
        features_evtf(ctx_dir / "evtf.csv", grid),
        features_ftl(ctx_dir / "ftl.csv", grid, vocab["ftl_types"]),
        features_dmop(ctx_dir / "dmop.csv", grid, vocab),
    ]
    X = pd.concat(parts, axis=1)
    X = pd.concat([X, derived_features(X)], axis=1)
    X.index.name = "ut_ms"
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return X.astype(np.float32)


LEADLAG_COLS = ["ev_umbra", "ev_penumbra", "saaf_cos_sa", "saaf_cos_sx", "saaf_cos_sy", "saaf_cos_sz",
                "ftl_comms", "ftl_EARTH", "ftl_SLEW", "ev_log_alt"]
LEADLAG_SHIFTS = [-3, -2, -1, 1, 2, 3]


DERIVED_PREFIXES = ("fx_", "hist_", "ll_", "roll_", "ewm_")


def recompute_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Recalcule les features derivees a partir des colonnes de base (apres modification d'un planning,
    par exemple dans le pilote) ; les colonnes et leur ordre sont conserves."""
    base = X[[c for c in X.columns if not c.startswith(DERIVED_PREFIXES)]]
    out = pd.concat([base, derived_features(base)], axis=1)
    missing = [c for c in X.columns if c not in out.columns]
    if missing:
        raise ValueError(f"recompute_derived : colonnes derivees non recalculables : {missing[:5]}")
    return out[X.columns]


def derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Features derivees des sources de base :
      fx_*   : flux solaire recu par face = constante solaire x cos(angle) tronque a 0,
               pour chaque face (+/-x, +/-y, +/-z, et l'axe du capteur sa), corrige de l'ombre
      hist_* : memoire thermique explicite : heures depuis la derniere ombre, ombre cumulee sur
               les 7 et 24 dernieres heures, duree de la derniere ombre, flux moyen 6 h / 24 h
      ll_*   : anticipation / retard : valeurs des heures t-3..t+3 de quelques colonnes cles
               (le planning est connu a l'avance, c'est une entree legitime)
    """
    out = {}
    sol = X["lt_solarconstantmars"] / 600.0  # ~1 a la constante solaire moyenne
    shade = (1.0 - X["ev_umbra"] - 0.5 * X["ev_penumbra"]).clip(0, 1)
    for ax in ["sa", "sx", "sy", "sz"]:
        c = X[f"saaf_cos_{ax}"]
        out[f"fx_{ax}_plus"] = sol * c.clip(lower=0) * shade
        out[f"fx_{ax}_minus"] = sol * (-c).clip(lower=0) * shade
    out["fx_total"] = sol * shade
    umb = X["ev_umbra"].to_numpy()
    since = np.zeros(len(umb))
    last = -RECENCY_CAP_H
    for i in range(len(umb)):
        if umb[i] > 0:
            last = i
        since[i] = min(i - last, RECENCY_CAP_H)
    out["hist_h_since_umbra"] = np.log1p(since)
    out["hist_umbra_7h"] = X["ev_umbra"].rolling(7, min_periods=1).sum()
    out["hist_umbra_24h"] = X["ev_umbra"].rolling(24, min_periods=1).sum()
    out["hist_umbra_7d"] = X["ev_umbra"].rolling(24 * 7, min_periods=1).mean()
    out["hist_fx_6h"] = out["fx_total"].rolling(6, min_periods=1).mean()
    out["hist_fx_24h"] = out["fx_total"].rolling(24, min_periods=1).mean()
    out["hist_comms_24h"] = X["ftl_comms"].rolling(24, min_periods=1).mean()
    for c in LEADLAG_COLS:
        if c not in X.columns:
            continue
        for k in LEADLAG_SHIFTS:
            out[f"ll_{c}_{k:+d}"] = X[c].shift(-k)  # k>0 : valeur future (t+k), k<0 : passee
    # roll_* : statistiques glissantes multi-echelles (passe) : ce qu'un arbre ne peut pas calculer
    # seul, et qui porte l'inertie thermique (Soleil recu, ombre, pointage, activite des sous-systemes)
    base = {"fx_total": out["fx_total"]}
    for c in ROLL_COLS:
        if c in X.columns:
            base[c] = X[c]
    for c, serie in base.items():
        for w in ROLL_WINDOWS:
            out[f"roll_{c}_{w}h"] = serie.rolling(w, min_periods=1).mean()
    for c in ["ev_umbra", "ftl_comms", "fx_total"]:
        serie = base.get(c)
        if serie is not None:
            for w in [6, 24]:
                out[f"roll_{c}_next{w}h"] = serie[::-1].rolling(w, min_periods=1).mean()[::-1]  # futur (planning connu)
    # ewm_* : inertie thermique du premier ordre : moyennes exponentielles (constantes de temps 3 h,
    # 12 h, 48 h, 168 h) du flux recu, de l'ombre, des angles et du pointage
    for c in EWM_COLS:
        serie = base.get(c)
        if serie is None:
            continue
        for tau in EWM_TAUS:
            out[f"ewm_{c}_{tau}h"] = serie.ewm(halflife=tau, min_periods=1).mean()
    return pd.DataFrame(out, index=X.index)


ROLL_COLS = ["saaf_cos_sa", "saaf_cos_sx", "saaf_cos_sy", "saaf_cos_sz", "ev_umbra", "ev_penumbra", "ev_occ_mars",
             "ftl_comms", "ftl_EARTH", "ftl_SLEW", "ftl_INERTIAL", "ftl_NADIR", "ftl_MAINTENANCE", "ftl_WARMUP",
             "dm_total", "dm_p_ATTT", "dm_p_AAAA", "dm_p_APSF", "dm_p_AMMM", "dm_p_ASSS", "dm_p_AHHH", "dm_p_AOOO",
             "dm_p_AVVV", "dm_p_AXXX", "dm_p_AACF"]
ROLL_WINDOWS = [6, 24, 72, 168]
EWM_COLS = ["fx_total", "ev_umbra", "ev_penumbra", "saaf_cos_sa", "saaf_cos_sx", "saaf_cos_sy", "saaf_cos_sz",
            "ftl_comms", "ftl_EARTH", "ftl_SLEW", "ftl_INERTIAL", "ftl_NADIR", "ftl_MAINTENANCE", "ftl_WARMUP"]
EWM_TAUS = [3, 12, 48, 168]


def load_power_hourly(power_csv: Path, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Moyenne horaire des 33 circuits, exactement comme make_validation.py (NaN si aucune mesure)."""
    raw = pd.read_csv(power_csv)
    raw["ut_ms"] = (raw["ut_ms"] // H) * H
    y = raw.groupby("ut_ms").mean().sort_index()
    if grid is not None:
        y = y.reindex(grid)
    y.index.name = "ut_ms"
    return y.astype(np.float32)
