#!/bin/bash
# Relance toute la chaine de livrables avec le systeme courant (defauts de train.py) :
#   validation stricte (annee1 -> annee2) et chrono, flux officiel predict + evaluate, modele final,
#   demo, pilote, lignes d'ablation hybrides. Usage : ./run_all.sh [threads]   (journal : reports/run_all.log)
set -e
T=${1:-6}
PY=.venv/bin/python
export OMP_NUM_THREADS=$T
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a reports/run_all.log; }
log "=== validation stricte (5 graines) ==="
$PY train.py --holdout --seeds 5 --threads $T 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
log "=== validation chrono (3 graines) ==="
$PY train.py --holdout-split 0.5 --seeds 3 --out weights_holdout_split --threads $T 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
log "=== flux officiel : predict + evaluate sur validation/ ==="
[ -d validation/test ] || (cd kit_etudiants && ../$PY make_validation.py --annee annee2 --out ../validation)
$PY predict.py --test_dir validation/test --out pred.csv --weights weights_holdout --std_out pred_std.csv 2>&1 | grep -v Warning | tail -n 2 | tee -a reports/run_all.log
$PY kit_etudiants/evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv 2>&1 | tee -a reports/run_all.log
log "=== modele final (annee1 + annee2, 5 graines) ==="
$PY train.py --seeds 5 --threads $T 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
log "=== demo + pilote ==="
$PY demo.py --weights weights_holdout 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
$PY pilot.py --weights weights_holdout --start-day 100 --days 14 --iters 120 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
log "=== ablations (lignes hybrides, 1 graine) ==="
$PY ablations.py --seeds 1 --only complet,GRU 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log
log "=== termine ==="
