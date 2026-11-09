"""
Chargement des periodes du kit (annee1, annee2) sous forme de tables horaires,
avec cache sur disque (la lecture de power.csv prend ~1 min).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_VERSION, H, build_features, build_vocab, hourly_grid, load_power_hourly

CACHE = Path("cache")


def _vocab_hash(vocab: dict) -> str:
    return hashlib.md5(json.dumps(vocab, sort_keys=True).encode()).hexdigest()[:10]


def load_period(ctx_dir: Path, vocab: dict, power_csv: Path | None = None, use_cache: bool = True):
    """Retourne (X, y) horaires pour un dossier du kit. y = None si pas de power.csv."""
    ctx_dir = Path(ctx_dir)
    power_csv = power_csv or (ctx_dir / "power.csv")
    key = f"{ctx_dir.name}_v{FEATURE_VERSION}_{_vocab_hash(vocab)}"
    fx, fy = CACHE / f"{key}_X.parquet", CACHE / f"{ctx_dir.name}_y.parquet"
    if use_cache and fx.exists() and fy.exists():
        X = pd.read_parquet(fx)
        y = pd.read_parquet(fy).reindex(X.index)
        y.index.name = "ut_ms"
        return X, y

    if use_cache and fy.exists():
        y = pd.read_parquet(fy)
    else:
        y = load_power_hourly(power_csv)
        if use_cache:
            CACHE.mkdir(exist_ok=True)
            y.to_parquet(fy)
    grid = hourly_grid(y.index.min(), y.index.max())
    y = y.reindex(grid)
    y.index.name = "ut_ms"
    X = build_features(ctx_dir, grid, vocab)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        X.to_parquet(fx)
    return X, y


def get_vocab(kit: Path, years: list[str], until_ms: int | None = None, use_cache: bool = True) -> dict:
    """Vocabulaire des commandes et des types de pointage appris sur `years` (heures < until_ms si donne) :
    toujours la periode d'entrainement, jamais la periode de validation."""
    kit = Path(kit)
    suffix = "" if until_ms is None else f"_until{int(until_ms)}"
    f = CACHE / f"vocab_v{FEATURE_VERSION}_{'_'.join(years)}{suffix}.json"   # le vocabulaire depend des seuils : versionne
    if use_cache and f.exists():
        return json.loads(f.read_text())
    v = build_vocab([kit / y for y in years], until_ms)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(v))
    return v


def period_grid(ctx_dir: Path, power_csv: Path | None = None, use_cache: bool = True) -> np.ndarray:
    """Grille horaire d'une periode (depuis power.csv, meme cache que load_period), sans calculer les features."""
    ctx_dir = Path(ctx_dir)
    fy = CACHE / f"{ctx_dir.name}_y.parquet"
    y = pd.read_parquet(fy) if (use_cache and fy.exists()) else load_power_hourly(power_csv or ctx_dir / "power.csv")
    return hourly_grid(y.index.min(), y.index.max())


def feature_groups(columns) -> dict[str, list[str]]:
    """Regroupe les colonnes par source (pour les ablations)."""
    groups = {"saaf": [], "ltdata": [], "evtf": [], "ftl": [], "dmop_base": [], "dmop_switch": [], "dmop_cmd": [],
              "dmop_prefix_rec": [], "dmop_stem_rec": [], "dmop_sw": [], "dmop_stem_cnt": [], "dmop_stem_next": [],
              "flux": [], "hist": [], "leadlag": [], "roll": [], "ewm": []}
    for c in columns:
        if c.startswith("dm_s_"):
            groups["dmop_switch"].append(c)
        elif c.startswith("dm_c_") or c.startswith("dm_r_"):
            groups["dmop_cmd"].append(c)
        elif c.startswith("dm_pr_"):
            groups["dmop_prefix_rec"].append(c)
        elif c.startswith("dm_st_"):
            groups["dmop_stem_rec"].append(c)
        elif c.startswith("dm_sw_"):
            groups["dmop_sw"].append(c)
        elif c.startswith("dm_sc_"):
            groups["dmop_stem_cnt"].append(c)
        elif c.startswith("dm_sn1_") or c.startswith("dm_sn6_"):
            groups["dmop_stem_next"].append(c)
        elif c.startswith("dm_"):
            groups["dmop_base"].append(c)
        elif c.startswith("fx_"):
            groups["flux"].append(c)
        elif c.startswith("hist_"):
            groups["hist"].append(c)
        elif c.startswith("ll_"):
            groups["leadlag"].append(c)
        elif c.startswith("roll_"):
            groups["roll"].append(c)
        elif c.startswith("ewm_"):
            groups["ewm"].append(c)
        elif c.startswith("saaf_"):
            groups["saaf"].append(c)
        elif c.startswith("lt_"):
            groups["ltdata"].append(c)
        elif c.startswith("ev_"):
            groups["evtf"].append(c)
        elif c.startswith("ftl_"):
            groups["ftl"].append(c)
    return groups


def select_features(columns, keep: str = "", drop: str = "") -> list[str]:
    """Selection de groupes : keep = liste a garder (vide = tous), drop = liste a retirer.
    'dmop' designe les trois groupes dmop_base, dmop_switch, dmop_cmd."""
    groups = feature_groups(columns)

    def expand(spec):
        out = []
        for g in [x for x in spec.split(",") if x]:
            out += ["dmop_base", "dmop_switch", "dmop_cmd", "dmop_prefix_rec", "dmop_stem_rec", "dmop_sw", "dmop_stem_cnt", "dmop_stem_next"] if g == "dmop" else [g]
        return out

    kept = expand(keep) or list(groups)
    kept = [g for g in kept if g not in expand(drop)]
    allowed = {c for g in kept for c in groups[g]}
    return [c for c in columns if c in allowed]
