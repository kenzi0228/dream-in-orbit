# Dream in Orbit — rapport

Satellite : Mars Express (ESA). Les deux périodes du kit, `annee1` et `annee2`, couvrent
chacune 687 jours terrestres = une année martienne, consécutives (22/08/2008 → 09/07/2010
et 10/07/2010 → 26/05/2012). Le jeu de test de l'orga est la période suivante.

Ce que le système livré est, en une phrase honnête : un ensemble de cinq GRU qui lisent le
planning et le contexte heure par heure et portent une mémoire sur l'état caché, mélangé à
parts égales avec un gradient boosting heure par heure ; la mémoire sert à la stabilité de la
simulation et aux cycles horaires, la physique (Soleil, ombre, saison, pointage) fait l'essentiel
de la prédiction, et les deux circuits de chauffage qui pèsent 63 % de l'erreur changent de
régime d'une année à l'autre pour des raisons que le planning ne contient pas (§ 5).

## 1. Fiches des données

Tout est aligné sur une grille horaire : une ligne par heure UTC (début d'heure, `ut_ms`),
16 456 heures pour `annee1`, 16 242 pour `annee2`. La vérité est la moyenne horaire des mesures de
`power.csv` (exactement comme `make_validation.py`) ; aucune heure n'est sans mesure, et une cellule
sans vérité serait ignorée dans la perte comme dans le score.

### power.csv — l'état (33 circuits NPWDxxxx)

| | |
|---|---|
| Lignes | 1,83 M (annee1), 1,77 M (annee2), soit une mesure toutes les ~32 s |
| Valeurs | courant moyen par circuit, de 0 à ~2,2 ; 13 à 17 circuits selon l'année sont quasi constants (écart-type < 0,001) |
| Qui pèse dans la RMSE | NPWD2551 (36 % de l'erreur quadratique de la constante), NPWD2451 (25 %), NPWD2851 (9 %), NPWD2532 (9 %), NPWD2531 (5 %), NPWD2881 (4 %) |
| Comportement | NPWD2551 et NPWD2451 sont des lignes de chauffage à thermostat : fortes pendant les saisons d'éclipses (moyenne 0,5-0,8 contre 0,03-0,1 hors saison), cycles ON/OFF à l'échelle de l'orbite (7 h), autocorrélation à 1 h ≈ 0,6 seulement |
| Pièges | changements de régime d'une année à l'autre : NPWD2531 passe de 0,4-0,6 (automne 2008) à ~0 ensuite ; NPWD2551 s'éteint en janvier 2011 malgré les éclipses, décroche le 30 août 2011 (§ 5) et s'allume en février-mars 2012 sans éclipse |

### dmop.csv — les commandes (actions)

| | |
|---|---|
| Lignes | 187 k (annee1), 144 k (annee2) ; `ut_ms` + nom de commande |
| Format | `AAAAF56A1` : 4 lettres de sous-système (AAAA, APSF, AMMM, ATTT = thermique, AXXX…), un code, une lettre de variante. Les entrées `MAPO.000…`/`MPER.…` marquent apocentre/péricentre, `PENS/PENE`, `MOCS/MOCE`… sont des paires début/fin |
| Vocabulaire | 9 280 commandes distinctes dans annee1 ; le vocabulaire des features est appris sur la période d'entraînement seule (§ 2) : en strict, 148 commandes fréquentes, 117 « stems » (commande sans sa lettre de variante) émis ≥ 50 fois, 18 interrupteurs (stems à plusieurs variantes fréquentes, 49 couples stem/lettre) |
| Features livrées | **sans lettre de variante** : comptes horaires par stem (`dm_sc_`), commandes à venir dans l'heure et les 6 heures suivantes (`dm_sn1_`, `dm_sn6_`), heures depuis / jusqu'au prochain changement de variante de chaque interrupteur (`dm_sw_`), comptes par sous-système, récences par sous-système et par stem, fraction d'heure dans les intervalles début/fin. Les features par lettre (`dm_s_` états, `dm_c_` comptes, `dm_r_` récences par commande) existent dans le cache mais aucun modèle livré ne les lit |
| Pièges | dérive du vocabulaire d'une année à l'autre : les lettres de variante changent pour 20 des 29 interrupteurs, le sous-système AACF (5 825 commandes) n'existe que dans annee2 ; **trous** : `dmop.csv` est vide 83 jours dans annee2 (deux trous de 34 et 35 jours, un de 14, un de 15) contre 56 jours dans annee1 (aucun trou de plus de 15 jours) |

### ftl.csv — les pointages (actions)

| | |
|---|---|
| Lignes | 19,6 k / 20 k intervalles `[utb_ms, ute_ms)` avec un type et un drapeau `flagcomms` |
| Types | EARTH (59 % du temps), SLEW (14 %), INERTIAL, MAINTENANCE, NADIR, WARMUP, RADIO_SCIENCE, ACROSS_TRACK, D4PNPO… (16 types) ; D4PNPO couvre 658 heures de annee1 et 1 958 de annee2 |
| Features | fraction d'heure dans chaque type, fraction avec communication (`ftl_comms`, 28 % du temps), nombre de changements de pointage dans l'heure |

### saaf.csv — angles du Soleil (contexte)

| | |
|---|---|
| Lignes | 1,26 M / 1,31 M, une mesure toutes les ~47 s ; `sa` (aspect solaire), `sx`, `sy`, `sz` (angles avec les axes) |
| Features | moyenne horaire des 4 angles et de leurs cosinus, écart-type horaire de `sa` ; trous comblés par interpolation |
| Lien physique | hors saison d'éclipses, NPWD2551 passe de 0,09 (sa ≤ 2°) à 0,3-0,46 (sa > 20°) : moins de Soleil sur la face chauffée → plus de chauffage |

### evtf.csv — événements (contexte)

| | |
|---|---|
| Lignes | 76,8 k / 75,2 k ; `MAR_UMBRA_START/END`, `MAR_PENUMBRA_…`, `OCC_MARS_…`, `PERICENTRE_PASSAGE`, `APOCENTRE_PASSAGE`, `800/1200/2000/4000_KM_ASCEND/DESCEND`, `XXX_AOS/LOS` (stations sol) |
| Features | fraction d'heure en ombre (umbra, pénombre, ombre de Phobos/Deimos), en occultation, en contact station ; temps depuis / jusqu'au péricentre, phase orbitale (cos, sin), log de l'altitude interpolée (péricentre ≈ 330 km, apocentre ≈ 10 100 km) |
| Lien physique | en saison d'éclipses, NPWD2551 vaut 0,73-0,84 dans l'heure de l'umbra contre 0,35-0,55 ensuite |

### ltdata.csv — tendances journalières (contexte)

| | |
|---|---|
| Lignes | 687 / 687, une par jour : distances Soleil-Mars et Terre-Mars, angle Soleil-Mars-Terre, constante solaire, durée d'éclipse et d'occultation |
| Features | interpolation linéaire à l'heure |
| Piège | **distance Terre-Mars et angle Soleil-Mars-Terre jouent le rôle de calendrier** : leur période (780 j) diffère de l'année martienne (687 j), donc le couple (distance Soleil, distance Terre) identifie la date de façon unique. Un réseau les utilise pour mémoriser l'année d'entraînement : on les retire (ablation : +0,013) |

## 2. Protocole d'évaluation

Trois validations, toujours chronologiques, jamais aléatoires, toutes sous la forme exacte de
l'évaluation : 7 jours réels, puis simulation libre de tout le reste, RMSE de `evaluate.py`.

| Protocole | Entraînement | Test | Ce qu'il mesure |
|---|---|---|---|
| **strict** | annee1 | annee2 (16 242 h) | une année nouvelle, avec de nouvelles lettres de commandes et de nouveaux régimes : le cas de l'évaluation finale |
| **inversé** | annee2 | annee1 | la même chose dans l'autre sens, sur une année de test que **aucune** décision de conception n'a vue |
| **chronologique** | annee1 + ½ annee2 | ½ annee2 (8 121 h) | deux ans d'entraînement, lettres de commandes stables : ce que le modèle gagne à connaître le vocabulaire de l'année testée |

Deux règles d'hygiène. **Le vocabulaire des features est appris sur la période d'entraînement
seule** (`train.py` : noms des stems, interrupteurs, types de pointage ; en chronologique, à la
date de coupure) : rien de la période de test n'entre dans la définition des colonnes, comme au
test final où le vocabulaire vient de `weights/meta.json`. Et **le protocole inversé a été ajouté
après toutes les décisions** (features, hyperparamètres, poids de mélange, nombre de sacs, toutes
prises sur strict et chronologique) : il mesure l'optimisme de sélection, inévitable quand une
seule année de test sert à des dizaines de choix. Le bruit entre graines est de ±0,001 sur la RMSE
d'un GRU seul (strict : membres du mélange entre 0,0749 et 0,0763) : un écart inférieur à 0,002 n'est pas significatif.

Repères (RMSE de références naïves) : constante = 0,1058 (strict), 0,1252 (inversé), 0,1048
(chronologique) ; constante calée sur les 7 premiers jours = 0,1116 / 0,1494 / 0,1269 ;
persistance de la dernière heure = 0,1378 / 0,1689 / 0,1416 (`baselines.py`).

## 3. Le modèle

```
x_t = [features de l'heure t : contexte + planning, 162 colonnes standardisées]
y_{t-1} = consommation de l'heure précédente (33 circuits, standardisés)
h_t = GRU(h_{t-1}, MLP([x_t, y_{t-1}]))                  état caché (128), mémoire
ŷ_t^GRU = tête([h_t])                                     33 sorties, perte MSE sur l'échelle brute
ŷ_t = 0,5 × moyenne_5_graines(ŷ_t^GRU) + 0,5 × GBM(x_t^GBM)   684 colonnes pour le GBM
```

* **Entrées du GRU** (`--keep-groups saaf,ltdata,evtf,ftl,dmop_stem_cnt`) : angles solaires et
  cosinus, ombres et occultations horaires, phase orbitale et altitude, pointage (fraction par
  type, communication), constante solaire et durées d'éclipse et d'occultation, et les comptes
  horaires de commandes par stem, sans lettre de variante. Pas de calendrier (distance Terre-Mars,
  angle Soleil-Mars-Terre), pas de récences (« heures depuis… » : des rampes de plusieurs semaines
  qui servent d'horloge au réseau, § 8).
* **Entraînement « stateful »** : la période est découpée en 16 flux contigus ; on avance par
  segments de 168 h en transmettant l'état caché (BPTT tronquée), avec un décalage aléatoire à
  chaque époque. **Scheduled sampling** : 15 époques en teacher forcing, puis la probabilité de
  ré-injecter la vérité décroît vers 0 ; bruit gaussien (0,05 standardisé) sur l'état ré-injecté.
  80 époques, AdamW, OneCycle, dropout 0,3, dropout de features 0,2, 5 graines.
* **Composant gradient boosting** (`dio/gbm.py`) : un HistGradientBoostingRegressor par circuit
  (600 arbres à 0,02, 15 feuilles, 100 par feuille, l2 = 3), sur toutes les colonnes sauf les
  calendriers, les statistiques glissantes et toute feature encodée par lettre ; **bagging
  temporel en 5 sacs**, chaque sac entraîné sans un bloc contigu d'un cinquième de la période,
  prédiction = moyenne des sacs. Déterministe, sans mémoire, aveugle aux 7 jours réels.
* **Mélange à parts égales**, poids choisi sur les protocoles strict et chronologique avant
  l'existence du protocole inversé (§ 4). L'empilement (le GRU lit la prédiction hors-échantillon
  du GBM et apprend à la corriger) a été essayé et nettement battu par le mélange : les prédictions
  hors-échantillon des sacs sont des erreurs d'extrapolation de saison (un sac n'a jamais vu
  d'éclipses) qui ne ressemblent pas aux erreurs du GBM complet au test ; le GRU apprend à corriger
  des erreurs qui n'existent plus (§ 8).
* **Incertitude** : écart-type des cinq prédictions finales (Deep Ensembles), multiplié par un
  facteur par circuit ajusté sur une année de validation et vérifié sur l'autre (`calibrate.py`,
  `weights/calibration.json`, § 4). Sans ce facteur, l'écart-type brut est sous-dispersé d'un
  facteur 11 : il classe les heures incertaines, il ne les quantifie pas.
* **Ce que la mémoire fait vraiment** (§ 6) : l'ablation « sans ré-injection » (le GRU ne relit
  pas y_{t-1}) donne la même RMSE ; la boucle d'état passe par h_t. Le GRU sans mémoire (MLP) perd
  0,002, les fenêtres indépendantes 0,010, l'absence de scheduled sampling 0,012. La mémoire est
  ce qui rend la simulation libre stable sur 16 000 pas, pas ce qui prédit le niveau.

**Règlement.** Le modèle final est un réseau de neurones (l'ensemble de GRU) auquel un modèle
classique est adjoint comme composant d'un mélange à parts égales, ce que le règlement autorise
(« les modèles classiques servent de baseline ou de composant uniquement »). `train.py --combine
none` donne la version réseau pur (0,0796 en strict), `--gbm-weight 0.3` une version où le
réseau pèse 70 % (0,0758), si l'orga préfère l'un ou l'autre.

## 4. Résultats

| Modèle | strict (annee1 → annee2) | inversé (annee2 → annee1) | chronologique (2 ans → ½ annee2) |
|---|---|---|---|
| Constante (moyenne du train) | 0,1058 | 0,1252 | 0,1048 |
| Gradient boosting seul (5 sacs, sans lettre) | 0,0792 | 0,0908 | 0,0816 |
| Ensemble de GRU seul (5 graines ; 3 en chronologique) | 0,0796 | 0,0952 | 0,0884 |
| **World model livré : 0,5 GRU + 0,5 GBM** | **0,0750** | **0,0901** | **0,0805** |
| réduction par rapport à la constante | 29 % | 28 % | 23 % |

Le flux officiel (`make_validation.py` → `predict.py` → `evaluate.py`, poids `weights_holdout`)
donne 0,074968 sur 530 409 cellules notées. Les cinq membres de l'ensemble strict sont entre
0,0749 et 0,0763 ; les moyenner gagne peu : les erreurs restantes sont systématiques,
pas du bruit de graine. Le protocole inversé, qui n'a servi à aucun choix, donne une réduction du
même ordre que le strict : l'optimisme de sélection sur le strict est au plus de quelques
millièmes.

Le poids du GBM, poids par poids (le choix de 0,5 a été fait sur strict et chronologique ; la
colonne inversée est une vérification a posteriori) :

| poids du GBM | 0,3 | 0,4 | **0,5** | 0,6 | 0,7 |
|---|---|---|---|---|---|
| strict | 0,0758 | 0,0752 | **0,0750** | 0,0751 | 0,0756 |
| inversé | 0,0915 | 0,0907 | **0,0901** | 0,0898 | 0,0897 |
| chronologique | 0,0826 | 0,0814 | **0,0805** | 0,0799 | 0,0798 |

### Erreur par circuit (strict, `experiments/tableaux_rapport.py`)

| Circuit | RMSE livré | GRU seul | GBM seul | écart-type réel | part de la MSE |
|---|---|---|---|---|---|
| NPWD2551 | 0,261 | 0,282 | 0,279 | 0,366 | 37 % |
| NPWD2451 | 0,221 | 0,237 | 0,228 | 0,299 | 26 % |
| NPWD2532 | 0,129 | 0,141 | 0,125 | 0,158 | 9 % |
| NPWD2851 | 0,124 | 0,118 | 0,149 | 0,136 | 8 % |
| NPWD2881 | 0,083 | 0,092 | 0,087 | 0,122 | 4 % |
| NPWD2561 | 0,082 | 0,090 | 0,087 | 0,120 | 4 % |
| NPWD2531 | 0,078 | 0,075 | 0,093 | 0,081 | 3 % |

Deux circuits font 63 % de l'erreur quadratique ; sur chacun, le mélange bat ses deux
composantes. Les erreurs du GRU et du GBM ne sont corrélées qu'à 0,78 : c'est pour cela que
leur moyenne gagne, alors que deux GRU de graines différentes sont corrélés à 0,96-0,99.

### Courbe de dérive (`reports/figures/derive.png`)

| Horizon depuis le début du rêve | RMSE moyenne des semaines |
|---|---|
| semaines 1-4 | 0,039 |
| semaines 5-50 | 0,064 |
| semaines 51-98 | 0,083 |

L'erreur ne s'accumule pas avec l'horizon : elle suit les saisons (pics pendant les deux saisons
d'éclipses, où la constante et le gradient boosting montent à 0,13-0,15). La pire semaine
(semaine 63, 0,141) est dans la deuxième saison d'éclipses de 2011, au décrochage
de NPWD2551 (§ 5). Le mécanisme anti-dérive est double : état caché entraîné sur des flux longs,
et apprentissage sur ses propres prédictions (scheduled sampling jusqu'à p = 0).

### Incertitude : calibration et couverture (`calibrate.py`, `reports/calibration_reverse.json`)

L'écart-type brut de l'ensemble suit l'erreur (corrélation de Spearman 0,82 entre
écart-type et erreur absolue) mais il est 11 fois trop petit. Facteurs par circuit
ajustés sur le protocole inversé (annee1 en test), puis appliqués tels quels aux deux autres :

| protocole (test) | couverture ±1σ brute | ±2σ brute | ±1σ calibrée | ±2σ calibrée | RMSE / RMS(σ) calibré |
|---|---|---|---|---|---|
| strict (annee2) | 13 % | 22 % | 45 % | 64 % | 1,63 |
| chronologique (½ annee2) | 11 % | 18 % | 38 % | 57 % | 1,64 |
| inversé (annee1, ajustement) | — | — | 60 % | 81 % | 1,00 |

Attendu sous une loi gaussienne : 68 % et 95 %. Les facteurs transférés d'une année de test à
l'autre donnent des intervalles encore trop étroits d'un facteur 1,6 : la couverture à ±2σ triple (22 % → 64 % en strict,
18 % → 57 % en chronologique) sans atteindre les 95 % nominaux, parce que l'année de test contient le
décrochage de 2011, où les cinq membres se trompent ensemble ; l'écart-type reste un classement
fiable des heures incertaines (Spearman 0,82-0,86). Le fichier livré `weights/calibration.json`
moyenne (géométriquement) les facteurs des deux années de test. La démo (`demo.py`,
`reports/figures/`) trace les bandes avec les facteurs ajustés sur l'autre année : ce qu'elle
montre n'a pas été ajusté sur ce qu'elle montre.

## 5. D'où vient l'erreur

* **Deux circuits, 1,5 % des cellules.** NPWD2551 et NPWD2451 font 63 % de l'erreur
  quadratique ; 60 % de l'erreur vient de 1,5 % des cellules circuit × heure (|erreur| > 0,3) : des
  marches, pas du bruit. NPWD2551 est un chauffage à paliers (à zéro sur 78 % des mesures brutes, sinon des paliers à
  0,22 / 0,62 / 1,04 / 1,64 / 1,86), allumé pendant les deux saisons d'éclipses ; NPWD2451 vaut 0,05, 1,1 ou 2,16 (un ou deux
  éléments), sa moyenne horaire est un rapport cyclique de thermostat anticorrélé au flux solaire,
  autocorrélé à 0,5 seulement d'une heure à l'autre.
* **Le décrochage de 2011** (`experiments/diag_2551.py`, `reports/diag_2551.md`). Pendant la
  deuxième saison d'éclipses de annee2 (jours 400-490), à ombre égale, NPWD2551 consomme 0,30 à
  0,45 de moins qu'en annee1 pour les heures à umbra partiel (0 à 0,6), et 0,12 de plus pour les
  heures d'umbra pleine ; le niveau en umbra baisse dans tous les modes de pointage (INERTIAL 0,87
  → 0,61, SLEW 0,89 → 0,79). D'autres lignes changent de niveau en même temps (NPWD2851 +0,11,
  NPWD2532 +0,18 hors ombre, NPWD2451 −0,11 en ombre) : la signature d'un changement de
  configuration thermique, pas d'un défaut de mesure. Le décrochage commence le **30 août 2011,
  premier jour d'un trou de 14 jours dans `dmop.csv`** (0 commande par heure, communications
  normales), suivi d'un trou de 35 jours à partir du 10 octobre ; cette saison contient aussi la
  famille de commandes AACF (0,8 commande par heure, absente de annee1) et le mode de pointage
  D4PNPO (14 % des heures contre 0,8 % la même saison de annee1). Un modèle entraîné sur annee1 n'a
  jamais vu ni un trou de planning de plus de 15 jours, ni AACF, ni autant de D4PNPO : il n'a
  aucun moyen d'apprendre ce que ces heures font. Ce que le kit ne contient pas (consignes de
  thermostat, état réel des équipements) reste inconnu : « inexpliqué » est la conclusion exacte,
  pas « inexplicable ».
* **Ce que ça borne.** Un oracle qui retirerait le biais hebdomadaire de NPWD2551 seul gagnerait
  environ 0,004 (0,0750 → 0,0710 ; 0,0691 en corrigeant aussi NPWD2451) : la RMSE plancher avec ces
  entrées est vers 0,071. Le reste de l'erreur sur les
  deux circuits de chauffage est le rapport cyclique du thermostat, imprévisible heure par heure à
  partir du planning.
* **Ce qui marche** : les saisons d'éclipses (montée et descente de NPWD2551 et NPWD2451, cycles
  horaires à l'échelle de l'orbite, `zoom_horaire.png`), la saison solaire de NPWD2451, le niveau
  de NPWD2532 ; les lignes à deux états (NPWD2881, NPWD2561, NPWD2791) depuis que le GRU lit les
  comptes de commandes.

## 6. Ablations (`ablations.py`, `reports/ablations.md`)

Chaque ligne ré-entraîne le modèle (1 graine, 80 époques, protocole strict) en retirant ou en
ajoutant un ingrédient, à partir du jeu de features livré. Les ablations du GRU sont mesurées sur
le GRU seul (0,0802) ; la première ligne situe le système complet. Bruit entre graines : ±0,001.

| Ablation | RMSE | écart | ce que ça montre |
|---|---|---|---|
| **système complet : GRU + composant gradient boosting (0,5)** | 0,0751 | −0,005 | le GBM corrige les deux chauffages (0,28 → 0,26 ; 0,24 → 0,22) sans toucher au reste (pires : NPWD2551 0,26, NPWD2451 0,22) |
| GRU seul | 0,0802 | référence | une graine ; la moyenne de 5 graines donne 0,0796 (pires : NPWD2551 0,28, NPWD2451 0,24) |
| sans mémoire (MLP heure par heure) | 0,0818 | +0,002 | la mémoire vaut 0,002, à peine plus que le bruit de graine (pires : NPWD2551 0,28, NPWD2451 0,25) |
| sans ré-injection (planning seul) | 0,0794 | −0,001 | relire y_{t-1} n'apporte rien : la boucle d'état passe par h_t (pires : NPWD2551 0,28, NPWD2451 0,24) |
| sans scheduled sampling (teacher forcing seul) | 0,0924 | +0,012 | apprendre sur ses propres sorties est ce qui tient le rêve 16 000 pas (pires : NPWD2551 0,36, NPWD2451 0,27) |
| fenêtres indépendantes (état remis à zéro, 20 époques) | 0,0898 | +0,010 | sans état transmis, pas de régimes longs ; confondu avec 20 époques au lieu de 80 (pires : NPWD2551 0,33, NPWD2451 0,26) |
| sans éclipses ni événements (evtf) | 0,0808 | +0,001 | l'ombre est déjà dans les angles solaires et les durées d'éclipse de ltdata (pires : NPWD2551 0,29, NPWD2451 0,24) |
| sans Soleil (saaf) | 0,1004 | +0,020 | l'ingrédient n° 1 : NPWD2532 passe de 0,14 à 0,24, NPWD2451 de 0,24 à 0,33 (pires : NPWD2451 0,33, NPWD2551 0,30) |
| sans tendances (ltdata : constante solaire, durées d'éclipse et d'occultation) | 0,0885 | +0,008 | la saison (constante solaire, durées d'éclipse) fixe le niveau de l'année (pires : NPWD2551 0,31, NPWD2451 0,28) |
| sans pointage (ftl) | 0,0898 | +0,010 | l'attitude (quelle face au Soleil) : NPWD2551 passe de 0,28 à 0,36 (pires : NPWD2551 0,36, NPWD2451 0,25) |
| sans comptes de commandes (dmop_stem_cnt : 45 features physiques) | 0,0824 | +0,002 | gain petit mais net sur les lignes à deux états : NPWD2881 entre dans les pires (pires : NPWD2551 0,29, NPWD2451 0,23) |
| avec interrupteurs dmop par lettre (dmop_switch) | 0,0854 | +0,005 | les lettres changent d'année : ce qu'elles font apprendre est perdu au test (pires : NPWD2551 0,31, NPWD2451 0,25) |
| avec toutes les colonnes dmop | 0,0867 | +0,006 | des centaines de colonnes de plus : NPWD2531 (0,19) et NPWD2851 (0,17) décrochent (pires : NPWD2551 0,27, NPWD2451 0,23) |
| avec calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0,0888 | +0,009 | une horloge : le réseau mémorise l'année d'entraînement au lieu de la physique (pires : NPWD2551 0,34, NPWD2451 0,25) |

Trois lectures. **La physique d'abord** : retirer le Soleil coûte 0,020, le pointage 0,010, les
tendances 0,008, soit dix fois ce que vaut la mémoire ; les éclipses d'evtf sont presque redondantes
(0,001) avec les angles solaires et les durées d'éclipse journalières. **La procédure d'entraînement
compte plus que l'architecture** : le scheduled sampling (0,012) et l'état transmis entre segments
(0,010) sont ce qui rend une simulation de 16 000 pas stable, alors que remplacer le GRU par un MLP
coûte 0,002 et que retirer la ré-injection de y_{t-1} ne coûte rien : la boucle d'état est portée par
h_t, pas par la sortie relue. C'est la limite honnête du récit « world model » (§ 9). **Ajouter
nuit** : lettres de variante (+0,005), toutes les colonnes dmop (+0,006), calendrier (+0,009) donnent
au réseau de quoi mémoriser l'année d'entraînement au lieu d'en apprendre la physique ; le seul ajout
qui paie est le compte horaire de commandes par stem (0,002, concentré sur les lignes à deux états).
Enfin le composant gradient boosting vaut 0,005 sur le GRU seul, autant que le deuxième ingrédient le
plus utile : deux modèles qui se trompent différemment valent mieux qu'un modèle plus gros (§ 8).

## 7. Bonus : le pilote (`pilot.py`, `reports/pilot.json`)

Le world model sert de simulateur pour comparer des plannings sur 14 jours en saison d'éclipses
(2010-10-25), initialisés par les 7 jours réels précédents, poids `weights_holdout`. Les
fenêtres de communication avec la Terre ne sont jamais touchées ; après chaque modification du
planning, les features dérivées (anticipation, inertie) sont recalculées. **Comparaison appariée** :
chaque membre de l'ensemble simule le planning réel et le planning modifié ; on rapporte la
moyenne des différences, leur dispersion entre membres et l'accord des membres sur le signe. C'est
l'incertitude du modèle sur la différence, bien plus petite que celle sur le niveau (±10,2).
Énergie = somme horaire des 33 circuits sur les 14 jours.

| Planning | Énergie simulée | Écart | dispersion entre membres | membres d'accord |
|---|---|---|---|---|
| Réel (mesuré : 1 726,2) | 1 728,4 | référence (+0,1 % du réel) | | |
| Hors communication, tout en pointage EARTH (plus d'observation) | 1 717,6 | −0,62 % | ±0,07 % | 5/5 |
| Hors communication, tout en NADIR (observation continue) | 1 718,0 | −0,60 % | ±0,26 % | 5/5 |
| Hors communication, sans manœuvres (SLEW rendu à EARTH) | 1 724,9 | −0,20 % | ±0,03 % | 5/5 |
| **Recherche locale** : permutations d'heures de pointage dans chaque journée, 120 essais, 15 gardées | 1 727,6 | −0,05 % | ±0,01 % | 5/5 |

Lecture : le chauffage dépend d'abord de la physique (ombre, Soleil), pas du planning ; les
écarts sont de l'ordre du pour cent au plus. Le gain « facile » (tout en EARTH) consiste à ne plus
observer, ce qui n'est pas un planning acceptable. La recherche locale ne retire aucune observation : elle déplace des heures de pointage à l'intérieur
de chaque journée et ne garde une permutation que si les cinq membres la jugent favorable ; elle
trouve −0,05 % en 120 essais (15 permutations gardées), un gain cohérent entre
membres mais près de trois fois plus petit que l'écart entre le planning réel simulé et le réel mesuré
(+0,1 %). Rien de tout cela n'est
vérifiable : ces plannings n'ont pas volé. Le pilote mesure ce que le modèle croit, et la
comparaison appariée dit seulement si le modèle est sûr de lui.

## 8. Journal des essais

Tout ce qui suit a été mesuré dans les protocoles strict et chronologique (`reports/experiments.csv`,
`reports/gbm_*.json`, `reports/reblend_*.json`) ; seuls les résultats utiles au lecteur sont gardés.

**Première semaine : apprendre à rêver.** Un GRU en teacher forcing pur dérive en quelques jours
de simulation (0,13) ; le scheduled sampling et l'entraînement stateful le ramènent à 0,088. Les
features de calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) font mémoriser l'année
d'entraînement : les retirer donne 0,083. Les 77 états d'interrupteurs dmop par lettre font tomber
le GRU à 0,103 (strict) et 0,121 (chrono). Le GRU « physique » (47 features) atteint 0,0821 / 0,0901.

**Deuxième semaine : le plateau du GRU, puis un second modèle.** Aucune variante seule ne déplace le
GRU physique de plus que le bruit : LSTM 0,0825, 256 neurones 0,0836, 2 couches 0,0835 (0,0993 en
chrono), SWA 0,0832, Huber 0,0825-0,0837, dropout / bruit / lr / weight decay 0,082-0,083 ;
adaptation au test sur les 7 jours réels 0,084 → 0,104 (sur-apprend une semaine) ; mémoire thermique
explicite, flux par face, anticipation t±3 : dans le bruit ou pire en chrono. Deux GRU de graines
différentes ont des erreurs corrélées à 0,96-0,99 : les moyenner n'apporte rien. Un gradient
boosting heure par heure sur toutes les colonnes est moins bon seul (0,0891) mais corrélé à 0,75
seulement : mélangé à 0,45, 0,0779 / 0,0830. ExtraTrees et RandomForest en plus : rien ; poids de
mélange par circuit : sur-apprennent la période de validation (choisis sur un protocole, pires sur
l'autre) ; second GRU sur toutes les features : rien de plus une fois le GBM présent.

**Troisième passe : les lettres.** L'erreur décomposée (deux circuits, 1,5 % des cellules, § 5) et une
découverte : les lettres de variante des commandes changent d'une année à l'autre (20 interrupteurs
sur 29, 7 sans aucune lettre commune, 49 commandes du vocabulaire présentes seulement en annee2).
Tout ce qui est appris par lettre est perdu l'année suivante ; aucun encodage sans lettre de
l'**état** d'un interrupteur n'est stable (rang d'apparition, parité, variante majoritaire : tous
s'inversent). Les **événements** le sont : comptes par stem, changements de variante à venir et
passés. GBM sans aucune lettre + ces features : 0,0822 / 0,0829 contre 0,0891 / 0,0823 ; bagging
temporel en 5 sacs : 0,0796 / 0,0820 ; GRU + comptes par stem : 0,0795 / 0,0901 (première
amélioration du GRU depuis la première semaine), alors que les récences par stem (0,1029) et les
changements d'interrupteurs (0,0920) lui nuisent : des rampes de plusieurs semaines sont des
horloges. Négatifs : moyennes exponentielles pour le GBM (0,0909 / 0,0866), pondération des données
récentes (0,0827-0,0836 contre 0,0823), mélange pondéré par l'incertitude (0,0003), projection de
rang faible, 7 ou 31 feuilles (0,0803 / 0,0849 et 0,0845 / 0,0824 : le compromis est 15), 161 stems
au lieu de 115-149 (GRU 0,0810 / 0,0925). Système : 0,0751 / 0,0808.

**Quatrième passe : la méthode.** Vocabulaire appris sur l'entraînement seul (le strict passe de
0,07509 à 0,0750 avec cinq graines : la fuite de noms de colonnes ne valait rien, mais elle
existait) ; protocole inversé comme année de test vierge ; empilement essayé et battu (GRU lisant
le GBM hors-échantillon et apprenant le résidu : 0,0908 ; résidu seul : 0,0884 ; contre 0,0751 pour
le mélange, explication § 3) ; calibration de l'incertitude par circuit, vérifiée d'une année sur
l'autre ; pilote en comparaison appariée ; diagnostic du décrochage de 2011 ; test de bout en bout
de `train.py` → `predict.py` ; cache du GBM pour que les relances et les ablations ne le
ré-ajustent pas ; poids du GBM compressés (45 Mo au lieu de 206).

## 9. Limites, et ce qu'il faudrait pour aller plus loin

* **Le récit « world model » est plus ambitieux que ce que les ablations soutiennent.** La mémoire
  vaut 0,002, la ré-injection de l'état rien ; les 7 jours réels n'informent que la première
  semaine. Le problème, tel que les données le posent, est une régression contextuelle avec une
  contrainte de stabilité sur 16 000 pas, et c'est ce que le système fait.
* **Une seule année de test par sens.** Deux protocoles sur deux années sont le maximum que le kit
  permet ; l'optimisme de sélection résiduel est de l'ordre du millième, pas plus, mais le vrai test
  (une troisième année, de nouveaux régimes) peut sortir de la fourchette 0,0805-0,0901.
* **Ce qui n'est pas dans le kit ne peut pas être appris** : consignes de thermostat, trous de
  planning de 35 jours, familles de commandes nouvelles. Une troisième année martienne de données
  compterait plus que n'importe quel modèle : le GBM passe de 0,089 à 0,082 avec une demi-année de
  plus.
* **L'incertitude calibrée est une échelle, pas une loi** : elle est juste en moyenne par circuit,
  pas heure par heure ; les erreurs sont des marches, pas des gaussiennes.
* **Le pilote est une démonstration d'usage**, pas un résultat : ses écarts sont de l'ordre du pour
  cent et invérifiables.
