"""
Ablations : ré-entraîne le world model en retirant une source d'information, et mesure
la RMSE de simulation sur annee2 (entraînement sur annee1, 7 jours réels puis rêve).

    python ablations.py --seeds 2 --epochs 12 --out reports/ablations.json

Chaque ligne appelle train.py avec les mêmes hyperparamètres que le modèle final,
plus l'option d'ablation. Les résultats (RMSE totale, par circuit, par semaine) sont
collectés depuis les val_report.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Les ablations du GRU sont mesurees sur le GRU seul (--combine none) pour rester lisibles ;
# les deux premieres lignes donnent le systeme complet (GRU + composant gradient boosting) et le GRU seul.
GRU = ["--combine", "none"]
ABLATIONS = {
    "complet (GRU + gradient boosting)": [],
    "GRU seul (sans composant gradient boosting)": GRU,
    "sans_memoire (MLP heure par heure, avec re-injection)": GRU + ["--no-memory"],
    "sans_reinjection (planning seul)": GRU + ["--no-feedback"],
    "sans_scheduled_sampling (teacher forcing seul)": GRU + ["--tf-epochs", "999"],
    "fenetres_independantes (pas d'etat transmis)": GRU + ["--windows-mode", "--epochs", "20", "--tf-epochs", "6"],
    "sans_eclipses_evenements (evtf)": GRU + ["--drop-groups", "evtf"],
    "sans_soleil (saaf)": GRU + ["--drop-groups", "saaf"],
    "sans_tendances (ltdata)": GRU + ["--drop-groups", "ltdata"],
    "sans_pointage (ftl)": GRU + ["--drop-groups", "ftl"],
    "sans_comptes_commandes (dmop_stem_cnt)": GRU + ["--drop-groups", "dmop_stem_cnt"],
    "avec_interrupteurs_par_lettre (dmop_switch)": GRU + ["--keep-groups", "saaf,ltdata,evtf,ftl,dmop_stem_cnt,dmop_switch"],
    "avec_toutes_commandes_dmop (toutes les colonnes)": GRU + ["--keep-groups", ""],
    "avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre)": GRU + ["--drop-cols", "^$"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/ablations.json")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", default="", help="sous-ensemble d'ablations (noms separes par des virgules)")
    ap.add_argument("--extra", default="", help="options supplementaires passees a train.py (hyperparametres du modele final)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    for name, opts in ABLATIONS.items():
        if a.only and name.split(" ")[0] not in a.only.split(","):
            continue
        tag = "abl_" + name.split(" ")[0]
        cmd = [sys.executable, "train.py", "--holdout", "--seeds", str(a.seeds), "--tag", tag] + opts + a.extra.split()
        if a.epochs:
            cmd += ["--epochs", str(a.epochs)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        rep = json.loads(Path(f"weights_holdout_{tag}/val_report.json").read_text())
        results[name] = {"val_rmse": rep["val_rmse"], "rmse_by_week": rep["val_rmse_by_week"],
                         "worst_circuits": dict(sorted(rep["val_rmse_per_circuit"].items(), key=lambda kv: -kv[1])[:5])}
        out.write_text(json.dumps(results, indent=1))
        print(json.dumps({name: results[name]["val_rmse"]}), flush=True)
    # on ne garde que les ablations de la liste courante (les anciens noms disparaissent du json et du tableau)
    results = {name: results[name] for name in ABLATIONS if name in results}
    out.write_text(json.dumps(results, indent=1))
    base = results.get("GRU seul (sans composant gradient boosting)", {}).get("val_rmse")
    lines = ["| Ablation | RMSE annee2 | écart vs GRU seul |", "|---|---|---|"]
    for name, r in results.items():
        d = "" if base is None else f"{r['val_rmse'] - base:+.4f}"
        lines.append(f"| {name} | {r['val_rmse']:.4f} | {d} |")
    Path(out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
