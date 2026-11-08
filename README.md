# Dream in Orbit — un world model de la consommation électrique de Mars Express

Un GRU apprend, sur les données de télémétrie du kit (deux années martiennes), comment
la consommation des 33 circuits réagit au planning (commandes, pointages) et au contexte
(Soleil, éclipses, distances). Il simule ensuite seul, heure par heure, une période
entière à partir de ses 7 premiers jours réels :

    état_{t-1} + actions_t + contexte_t  --(GRU)-->  état_t

La prédiction est la moyenne d'un ensemble de modèles (Deep Ensembles) ; leur
écart-type donne l'incertitude.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Testé avec Python 3.9 sur macOS (CPU uniquement, pas de GPU nécessaire).
Le kit doit se trouver dans `kit_etudiants/` (`annee1/`, `annee2/`, `make_validation.py`, `evaluate.py`).

## Utilisation

```bash
# Entraîne tout (annee1 + annee2) et écrit les poids dans weights/
.venv/bin/python train.py

# Simule une période de test au format de l'évaluation
.venv/bin/python predict.py --test_dir DOSSIER --out predictions.csv
```

Validation dans les conditions exactes de l'évaluation (entraînement sur annee1 seulement) :

```bash
.venv/bin/python train.py --holdout                    # -> weights_holdout/
cd kit_etudiants && ../.venv/bin/python make_validation.py --annee annee2 --out ../validation && cd ..
.venv/bin/python predict.py --test_dir validation/test --out pred.csv --weights weights_holdout
.venv/bin/python kit_etudiants/evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv
```

Autres scripts :

| Script | Rôle |
|---|---|
| `baselines.py` | prédiction constante, persistance, gradient boosting heure par heure |
| `ablations.py` | ré-entraîne le world model sans tel ou tel groupe de features / sans mémoire / sans ré-injection |
| `demo.py` | figures « rêve contre réalité », courbe de dérive, erreur par circuit, calibration de l'incertitude |
| `pilot.py` | bonus : utilise le simulateur pour comparer des plannings alternatifs |

## Organisation du code

| Fichier | Contenu |
|---|---|
| `dio/features.py` | alignement des 5 fichiers de contexte sur la grille horaire (fiches des données dans `RAPPORT.md`) |
| `dio/data.py` | chargement avec cache (`cache/`), groupes de features |
| `dio/model.py` | le world model GRU (teacher forcing, rollout avec scheduled sampling), normalisation |
| `dio/simulate.py` | simulation d'une période avec un ensemble de modèles |
| `dio/metrics.py` | RMSE identique à `evaluate.py`, RMSE par circuit et par horizon |
| `train.py` | entraînement en une commande (ensemble de 5 GRU) |
| `predict.py` | interface officielle |

Le rapport (fiches des données, ablations, dérive, analyse des erreurs) est dans `RAPPORT.md`,
les figures dans `reports/figures/`.

## Résultats (RMSE de simulation sur annee2, après entraînement sur annee1 seule)

| Modèle | RMSE |
|---|---|
| Constante | 0,106 |
| Gradient boosting heure par heure | 0,092 |
| **World model (ensemble de 5 GRU)** | **0,082** |

Le modèle livré dans `weights/` (5 fichiers `model_k.pt`, 2,6 Mo au total, plus `meta.json`
qui contient le vocabulaire et la liste des features) est entraîné sur annee1 + annee2
(`python train.py`, 80 époques par modèle, ~25 min sur un portable 6 cœurs, CPU).
Appliqué à annee2 il donne 0,066, mais annee2 est dans son entraînement : la seule
estimation honnête est celle de `weights_holdout` (0,082). Il ne lit que les features physiques (Soleil,
ombre, orbite, pointage, constante solaire, durées d'éclipse) : les commandes dmop et
les colonnes « calendrier » de ltdata dégradent la généralisation d'une année à l'autre
(détails et ablations dans le rapport).

## Reproduire le rapport

```bash
.venv/bin/python baselines.py                       # reports/baselines.json
.venv/bin/python train.py --holdout --seeds 5       # weights_holdout/ (validation stricte)
.venv/bin/python demo.py --weights weights_holdout  # reports/figures/
.venv/bin/python ablations.py --seeds 1             # reports/ablations.json + .md
.venv/bin/python pilot.py --weights weights_holdout --start-day 100 --days 14 --iters 120
.venv/bin/python tests/test_features.py             # tests unitaires des features
```
