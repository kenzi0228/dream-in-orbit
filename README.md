# Dream in Orbit — simuler la consommation électrique de Mars Express

Un ensemble de cinq GRU apprend, sur la télémétrie du kit (deux années martiennes), comment la
consommation des 33 circuits réagit au planning (commandes, pointages) et au contexte (Soleil,
éclipses, saison). Il simule ensuite seul, heure par heure, une période entière à partir de ses
7 premiers jours réels :

    état_{t-1} + actions_t + contexte_t  --(GRU, mémoire h_t)-->  état_t

La prédiction livrée est la moyenne des cinq GRU, mélangée à parts égales avec un composant
gradient boosting heure par heure (`dio/gbm.py`, 5 sacs temporels, déterministe). Les deux
familles de modèles font des erreurs peu corrélées (0,8) et le mélange bat chacune d'elles dans
les trois protocoles de validation (0,005 à 0,008 de RMSE de mieux que l'ensemble de GRU seul).
`python train.py --combine none` donne la version réseau de neurones pur.

Les commandes dmop sont lues **sans leur lettre de variante** (`AXXX301A` et `AXXX301C` sont la
même commande) : les lettres changent d'une année martienne à l'autre pour 20 des 29
interrupteurs, et tout ce qui est appris par lettre est perdu l'année suivante (`RAPPORT.md`, § 8).
Le vocabulaire des features est appris sur la période d'entraînement seule.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Testé avec Python 3.9 sur macOS (CPU uniquement, pas de GPU nécessaire).
Le kit doit se trouver dans `kit_etudiants/` (`annee1/`, `annee2/`, `make_validation.py`, `evaluate.py`).

## Utilisation

```bash
# Entraîne tout (annee1 + annee2) et écrit les poids dans weights/   (~70 min sur 6 cœurs)
.venv/bin/python train.py

# Simule une période de test au format de l'évaluation (+ incertitude calibrée, optionnelle)
.venv/bin/python predict.py --test_dir DOSSIER --out predictions.csv [--std_out incertitude.csv]
```

Validation dans les conditions exactes de l'évaluation (entraînement sur annee1 seulement) :

```bash
.venv/bin/python train.py --holdout                    # -> weights_holdout/
cd kit_etudiants && ../.venv/bin/python make_validation.py --annee annee2 --out ../validation && cd ..
.venv/bin/python predict.py --test_dir validation/test --out pred.csv --weights weights_holdout
.venv/bin/python kit_etudiants/evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv
```

`train.py --holdout-reverse` (annee2 → annee1) et `train.py --holdout-split 0.5` (deux ans →
½ annee2) sont les deux autres protocoles du rapport.

## Résultats (RMSE de simulation, 7 jours réels puis rêve sur toute la période)

| Modèle | strict (annee1 → annee2) | inversé (annee2 → annee1) | chronologique (2 ans → ½ annee2) |
|---|---|---|---|
| Constante | 0,106 | 0,125 | 0,105 |
| Gradient boosting seul (5 sacs, sans lettre) | 0,0792 | 0,0908 | 0,0816 |
| Ensemble de GRU seul | 0,0796 | 0,0952 | 0,0884 |
| **World model livré (0,5 GRU + 0,5 GBM)** | **0,0750** | **0,0901** | **0,0805** |

Flux officiel (`predict.py` → `evaluate.py`, poids `weights_holdout`) : 0,074968. Le protocole
inversé n'a servi à aucune décision de conception : il mesure l'optimisme de sélection des deux
autres. L'incertitude livrée (`--std_out`) est l'écart-type de l'ensemble calibré par circuit
(`calibrate.py`, `weights/calibration.json`), vérifié d'une année de test sur l'autre.

Le modèle livré dans `weights/` (5 fichiers `model_k.pt`, 3 Mo ; `gbm.pkl`, le composant gradient
boosting compressé, 45 Mo ; `meta.json`, vocabulaire et liste des features ; `calibration.json`)
est entraîné sur annee1 + annee2. La seule estimation honnête de sa RMSE est celle des trois
protocoles ci-dessus : attendez-vous à 0,0805-0,0901 sur une année nouvelle. Deux circuits de
chauffage font 63 % de l'erreur, et leur décrochage de 2011 coïncide avec un trou de 14 jours dans
le fichier de commandes (`RAPPORT.md`, § 5).

## Organisation du code

| Fichier | Contenu |
|---|---|
| `train.py` | entraînement en une commande ; protocoles `--holdout`, `--holdout-reverse`, `--holdout-split` ; `--combine blend|none|stack` |
| `predict.py` | interface officielle (+ `--std_out`, incertitude calibrée) |
| `calibrate.py` | facteurs d'échelle de l'incertitude par circuit, ajustés sur une validation, vérifiés sur les autres |
| `dio/features.py` | alignement des 5 fichiers de contexte sur la grille horaire, vocabulaire appris sur l'entraînement (fiches dans `RAPPORT.md`) |
| `dio/data.py` | chargement avec cache (`cache/`), groupes de features |
| `dio/model.py` | le world model GRU (teacher forcing, rollout avec scheduled sampling), normalisation |
| `dio/gbm.py` | le composant gradient boosting (sacs temporels, prédictions hors-échantillon, cache) |
| `dio/simulate.py` | simulation d'une période avec l'ensemble (membres, moyenne, écart-type) |
| `dio/metrics.py` | RMSE identique à `evaluate.py`, par circuit, par horizon |
| `baselines.py` | constante, persistance, gradient boosting heure par heure |
| `ablations.py` | ré-entraîne le GRU sans tel ou tel ingrédient (`reports/ablations.md`) |
| `demo.py` | figures « rêve contre réalité », dérive, erreur par circuit, calibration (`reports/figures/`) |
| `pilot.py` | bonus : comparaison appariée de plannings alternatifs (`reports/pilot.json`) |
| `experiments/` | harnais d'expériences et analyses (voir `experiments/README.md`) |
| `tests/` | tests unitaires des features et test de bout en bout `train.py` → `predict.py` |
| `run_all.sh` | relance toute la chaîne de livrables (~4 h sur 6 cœurs) |

## Reproduire le rapport

```bash
./run_all.sh 6
```

soit, dans l'ordre : les trois validations, le flux officiel, la calibration, le modèle final,
la démo, le pilote, les ablations, les tableaux du rapport, le diagnostic de NPWD2551 et les tests.
Chaque étape est aussi une commande seule (voir le tableau ci-dessus) ; `python tests/test_features.py`
et `python tests/test_predict_smoke.py` lancent les tests.
