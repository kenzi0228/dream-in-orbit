"""
Harnais d'experiences : chaque configuration est entrainee dans les deux protocoles
(strict annee1 -> annee2, et chronologique annee1 + 1/2 annee2 -> 1/2 annee2), 1 graine,
et les RMSE sont consignees dans reports/experiments.csv.

    python experiments/sweep.py --batch experiments/batch1.json [--parallel 3]

batch.json : {"nom": "options de train.py", ...}
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "experiments.csv"


def run_one(name: str, args: str, protocol: str, threads: int) -> float:
    tag = f"exp_{name}_{protocol}"
    proto = ["--holdout"] if protocol == "strict" else ["--holdout-split", "0.5"]
    cmd = [sys.executable, "train.py", "--seeds", "1", "--threads", str(threads), "--eval-every", "20", "--tag", tag] + proto + args.split()
    log = ROOT / "reports" / "sweep" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    d = ROOT / f"weights_holdout_{tag}"
    rep = json.loads((d / "val_report.json").read_text())
    shutil.rmtree(d, ignore_errors=True)
    return rep["val_rmse"]


def run_config(name: str, args: str, threads: int) -> dict:
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        fs = {p: ex.submit(run_one, name, args, p, threads) for p in ["strict", "split"]}
        res = {p: f.result() for p, f in fs.items()}
    row = {"name": name, "args": args, "strict": round(res["strict"], 5), "split": round(res["split"], 5),
           "mean": round((res["strict"] + res["split"]) / 2, 5), "minutes": round((time.time() - t0) / 60, 1)}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--parallel", type=int, default=3, help="configurations en parallele (chacune = 2 processus)")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    configs = json.loads(Path(a.batch).read_text())
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda kv: run_config(kv[0], kv[1], a.threads), configs.items()))
    print("BATCH DONE", a.batch)


if __name__ == "__main__":
    main()
