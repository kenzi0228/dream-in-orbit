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

ABLATIONS = {
    "complet": [],
    "sans_memoire (MLP heure par heure, avec re-injection)": ["--no-memory"],
    "sans_reinjection (planning seul)": ["--no-feedback"],
    "sans_scheduled_sampling (teacher forcing seul)": ["--tf-epochs", "999"],
    "fenetres_independantes (pas d'etat transmis)": ["--windows-mode", "--epochs", "20", "--tf-epochs", "6"],
    "sans_eclipses_evenements (evtf)": ["--drop-groups", "evtf"],
    "sans_soleil (saaf)": ["--drop-groups", "saaf"],
    "sans_tendances (ltdata)": ["--drop-groups", "ltdata"],
    "sans_pointage (ftl)": ["--drop-groups", "ftl"],
    "avec_commandes_dmop (prefixes + interrupteurs)": ["--keep-groups", "saaf,ltdata,evtf,ftl,dmop_base,dmop_switch"],
    "avec_toutes_commandes_dmop (560 features)": ["--keep-groups", ""],
    "avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre)": ["--drop-cols", "^$"],
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
    base = results.get("complet", {}).get("val_rmse")
    lines = ["| Ablation | RMSE annee2 | écart vs complet |", "|---|---|---|"]
    for name, r in results.items():
        d = "" if base is None else f"{r['val_rmse'] - base:+.4f}"
        lines.append(f"| {name} | {r['val_rmse']:.4f} | {d} |")
    Path(out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
