#!/bin/bash
# Relance toute la chaine de livrables avec le systeme courant (defauts de train.py) :
#   validations strict (annee1 -> annee2), inverse (annee2 -> annee1) et chronologique ; flux officiel predict + evaluate ;
#   calibration de l'incertitude ; modele final ; demo ; pilote ; ablations ; test de bout en bout.
# Usage : ./run_all.sh [threads]      (journal : reports/run_all.log ; ~4 h sur 6 coeurs, le GBM est mis en cache)
set -e
T=${1:-6}
PY=.venv/bin/python
export OMP_NUM_THREADS=$T
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a reports/run_all.log; }
run() { "$@" 2>&1 | grep -v Warning | tail -n 3 | tee -a reports/run_all.log; }
log "=== validation stricte : annee1 -> annee2 (5 graines) ==="
run $PY train.py --holdout --seeds 5 --threads $T
log "=== validation inverse : annee2 -> annee1 (5 graines) ==="
run $PY train.py --holdout-reverse --seeds 5 --threads $T
log "=== validation chronologique : annee1 + 1/2 annee2 -> 1/2 annee2 (3 graines) ==="
run $PY train.py --holdout-split 0.5 --seeds 3 --threads $T
log "=== GRU seul (reference des tableaux) en inverse et en chronologique (1 graine) ==="
run $PY train.py --holdout-reverse --combine none --seeds 1 --threads $T --tag gru
run $PY train.py --holdout-split 0.5 --combine none --seeds 1 --threads $T --tag gru
log "=== flux officiel : predict + evaluate sur validation/ ==="
[ -d validation/test ] || (cd kit_etudiants && ../$PY make_validation.py --annee annee2 --out ../validation)
run $PY predict.py --test_dir validation/test --out pred.csv --weights weights_holdout --std_out pred_std.csv
$PY kit_etudiants/evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv 2>&1 | tee -a reports/run_all.log
log "=== calibration de l'incertitude ==="
# pour la demo : facteurs ajustes en inverse (annee1 en test), verifies en strict (annee2 en test) et chrono
run $PY calibrate.py --fit weights_holdout_reverse --check weights_holdout,weights_holdout_split --out weights_holdout/calibration.json --report reports/calibration_reverse.json
# pour le modele livre : facteurs ajustes sur les deux annees de test, verifies en chrono
run $PY calibrate.py --fit weights_holdout,weights_holdout_reverse --check weights_holdout_split --out reports/calibration_livree.json --report reports/calibration.json
log "=== modele final (annee1 + annee2, 5 graines) ==="
run $PY train.py --seeds 5 --threads $T
cp reports/calibration_livree.json weights/calibration.json
log "=== demo + pilote ==="
run $PY demo.py --weights weights_holdout --calibration weights_holdout/calibration.json
run $PY pilot.py --weights weights_holdout --start-day 100 --days 14 --iters 120
log "=== ablations (1 graine ; le GBM vient du cache) ==="
run $PY ablations.py --seeds 1 --extra "--threads $T"
log "=== tableaux du rapport + diagnostic NPWD2551 ==="
run $PY experiments/tableaux_rapport.py --weights weights_holdout --gru weights_holdout_abl_GRU
run $PY experiments/tableaux_rapport.py --weights weights_holdout_reverse --gru weights_holdout_reverse_gru
run $PY experiments/diag_2551.py
log "=== test de bout en bout ==="
run $PY tests/test_predict_smoke.py
log "=== termine ==="
