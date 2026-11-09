# experiments/ : harnais et analyses

Scripts d'expérience et d'analyse qui ont produit les chiffres du rapport (`reports/*.json`, `reports/experiments.csv`).
Ils ne font pas partie du livrable (`train.py`, `predict.py`, `weights/`) et supposent le dépôt comme répertoire courant.

| Script | Rôle | Sorties |
|---|---|---|
| `sweep.py`, `batch_*.json` | harnais bi-protocole : une ligne de `experiments.csv` par variante de `train.py` | `reports/experiments.csv` |
| `gbm_sets.py`, `gbm_bagging.py`, `gbm_letterdrop.py`, `gbm_recency.py` | jeux de features, hyperparamètres, bagging temporel, robustesse aux lettres du gradient boosting | `reports/gbm_*.json` |
| `reblend_sets.py`, `reblend.py`, `blend_all.py`, `poids_commun.py` | re-mélange hors ligne GRU / GBM, poids de mélange sur les deux protocoles | `reports/reblend_*.json`, `reports/poids_commun.json` |
| `analyse_erreurs.py`, `analyse_circuits.py`, `analyse_trous.py`, `analyse_conjonction.py`, `analyse_variantes.py` | d'où vient l'erreur : circuits, semaines, trous du planning, lettres de variante | `reports/analyse_erreurs.json` |
| `diag_2551.py` | diagnostic systématique du creux de NPWD2551 en 2011 (trous de dmop, mode D4PNPO, famille AACF) | `reports/diag_2551.json`, `.md` |
| `tableaux_rapport.py` | chiffres de la section Résultats (par circuit, dérive, corrélation des erreurs) | `reports/tableaux_rapport.json` |
| `offline_tests.py` | tests hors ligne de post-traitements (déjà écartés) | — |

Les scripts antérieurs à la troisième passe (`sweep.py`, `gbm_*.py`, `reblend*.py`, `blend_all.py`, `analyse_*.py`)
construisent le vocabulaire des commandes sur les deux années : leurs chiffres servent à comparer des variantes
entre elles, pas à estimer la RMSE finale. Les estimations du rapport viennent de `train.py`, dont le vocabulaire
est appris sur la période d'entraînement seule.
