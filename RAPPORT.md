# Dream in Orbit — rapport

Satellite : Mars Express (ESA). Les deux périodes du kit, `annee1` et `annee2`, couvrent
chacune 687 jours terrestres = une année martienne, consécutives (22/08/2008 → 09/07/2010
et 10/07/2010 → 26/05/2012). Le jeu de test de l'orga est la période suivante.

## 1. Fiches des données

Tout est aligné sur une grille horaire : une ligne par heure UTC (début d'heure, `ut_ms`),
16 488 heures par période. La vérité est la moyenne horaire des mesures de `power.csv`
(exactement comme `make_validation.py`) ; 32 heures de `annee1` n'ont aucune mesure
(elles sont ignorées dans la perte et dans le score).

### power.csv — l'état (33 circuits NPWDxxxx)

| | |
|---|---|
| Lignes | 1,83 M (annee1), 1,77 M (annee2), soit une mesure toutes les ~32 s |
| Valeurs | courant moyen par circuit, de 0 à ~2,2 ; 19 circuits sont quasi constants (écart-type < 0,001) |
| Qui pèse dans la RMSE | NPWD2551 (36 % de l'erreur quadratique de la constante), NPWD2451 (25 %), NPWD2851 (9 %), NPWD2532 (9 %), NPWD2531 (5 %), NPWD2881 (4 %) |
| Comportement | NPWD2551 et NPWD2451 sont des lignes de chauffage à thermostat : fortes pendant les saisons d'éclipses (moyenne 0,5-0,8 contre 0,03-0,1 hors saison), cycles ON/OFF à l'échelle de l'orbite (7 h), autocorrélation à 1 h ≈ 0,6 seulement |
| Pièges | changements de régime d'une année à l'autre : NPWD2531 passe de 0,4-0,6 (automne 2008) à ~0 ensuite ; NPWD2551 s'éteint en janvier 2011 malgré les éclipses et s'allume en février-mars 2012 sans éclipse |

### dmop.csv — les commandes (actions)

| | |
|---|---|
| Lignes | 187 k (annee1), 144 k (annee2) ; `ut_ms` + nom de commande |
| Format | `AAAAF56A1` : 4 lettres de sous-système (AAAA, APSF, AMMM, ATTT = thermique, AXXX…), un code, une lettre de variante. Les entrées `MAPO.000…`/`MPER.…` marquent apocentre/péricentre, `PENS/PENE`, `MOCS/MOCE`… sont des paires début/fin |
| Vocabulaire | 9 280 commandes distinctes dans annee1 ; 198 apparaissent ≥ 50 fois dans l'une des deux années ; 39 « stems » ont plusieurs variantes (interrupteurs : `ATTTF310A`/`ATTTF310B` émises 3 600 fois chacune, en alternance) |
| Features | comptes par sous-système et par commande fréquente dans l'heure ; recence (log des heures depuis la dernière émission) ; **état des interrupteurs** (dernière variante émise pour chaque stem, 77 colonnes) ; fraction d'heure dans les intervalles début/fin |
| Pièges | dérive du vocabulaire : le sous-système AACF (5 800 commandes) n'existe que dans annee2 ; l'interrupteur `AXXX301`/`AXXX305` signale la saison d'éclipses par la variante A dans annee1 et par une autre variante dans annee2 |

### ftl.csv — les pointages (actions)

| | |
|---|---|
| Lignes | 19,6 k / 20 k intervalles `[utb_ms, ute_ms)` avec un type et un drapeau `flagcomms` |
| Types | EARTH (59 % du temps), SLEW (14 %), INERTIAL, MAINTENANCE, NADIR, WARMUP, RADIO_SCIENCE, ACROSS_TRACK, D4PNPO… (16 types) |
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
| Piège | **distance Terre-Mars et angle Soleil-Mars-Terre jouent le rôle de calendrier** : leur période (780 j) diffère de l'année martienne (687 j), donc le couple (distance Soleil, distance Terre) identifie la date de façon unique. Un réseau les utilise pour mémoriser l'année d'entraînement : on les retire |

## 2. Protocole d'évaluation

Deux validations, toujours chronologiques, jamais aléatoires :

* **stricte** : entraînement sur annee1, simulation de annee2 à partir de ses 7 premiers jours (le script `make_validation.py` + `evaluate.py` de l'orga) ;
* **chronologique à deux ans** : entraînement sur annee1 + première moitié de annee2, simulation de la seconde moitié. C'est la situation du test final (le modèle livré est entraîné sur les deux années), elle montre ce que le modèle gagne à connaître le vocabulaire des deux années.

Repères (RMSE, protocole strict) : constante = 0,1058 ; constante calée sur les 7 premiers jours = 0,1116 ; persistance de la dernière heure = 0,1378 ; gradient boosting heure par heure = 0,092 (features physiques).

## 3. Le world model

```
x_t = [features de l'heure t (planning + contexte)]       ~50 à 160 colonnes standardisées
y_{t-1} = consommation de l'heure précédente (33 circuits, standardisés)
h_t = GRU(h_{t-1}, MLP([x_t, y_{t-1}]))                   état caché = mémoire thermique
ŷ_t = tête([h_t])                                         33 sorties, perte MSE sur l'échelle brute
```

* **Entraînement « stateful »** (`--stateful`) : la période d'entraînement est découpée en
  16 flux contigus ; on avance par segments de 168 h en transmettant l'état caché d'un
  segment au suivant (BPTT tronquée). Le modèle apprend donc avec un état caché qui a
  évolué sur des milliers d'heures, comme en simulation. À chaque époque les flux sont
  décalés aléatoirement.
* **Scheduled sampling** : les 15 premières époques en teacher forcing (y_{t-1} réel), puis
  la probabilité de ré-injecter la vérité décroît linéairement vers 0 : en fin
  d'entraînement le modèle ne voit plus que ses propres prédictions, exactement comme
  au test. Un bruit gaussien (0,05 en unités standardisées) est ajouté à l'état
  ré-injecté pour qu'il apprenne à corriger ses propres erreurs.
* **Simulation** : les 7 jours réels initialisent l'état caché (teacher forcing), puis
  rêve libre ; prédictions bornées à 0. Moyenne de 5 modèles (graines différentes),
  l'écart-type de l'ensemble sert d'incertitude.
* **Perte** : MSE brute (celle de la métrique) ; les sorties sont dé-standardisées avec
  les statistiques du train, donc les gros circuits pèsent naturellement plus.
* **Entrées du GRU** : les quatre fichiers de contexte et de pointage (saaf, ltdata sans les
  deux colonnes « calendrier », evtf, ftl) et, depuis la troisième passe (section 8), les
  comptes horaires de commandes dmop par stem, sans la lettre de variante (196 entrées).
* **Composant gradient boosting** (`dio/gbm.py`, sections 7 et 8) : un
  HistGradientBoostingRegressor par circuit, entraîné sur toutes les features sauf les
  calendriers, les statistiques glissantes et toute feature encodée par lettre de commande ;
  5 sacs temporels moyennés ; déterministe. Prédiction finale = 0,5 × ensemble de GRU +
  0,5 × GBM. Le GBM ne voit pas les 7 jours réels ; l'incertitude affichée est celle de
  l'ensemble de GRU.
* **Ce qui porte vraiment la mémoire** : l'ablation « sans ré-injection » (le GRU ne lit
  que le planning, section 5) donne la même RMSE. La boucle état → état suivant
  passe donc par l'état caché h_t, pas par la prédiction ré-injectée ; les 7 jours
  réels servent à initialiser h_t. Le scheduled sampling reste nécessaire dès que la
  ré-injection existe (sans lui : +0,016).

### Les échecs, dans l'ordre

| Essai | RMSE annee2 (strict) | Diagnostic |
|---|---|---|
| GRU, fenêtres de 3 semaines, 483 features, 30 époques | 0,12 (train 0,064) | mémorise annee1 ; validation pire que la constante (0,106) |
| idem, 64 neurones, dropout 0,3, 164 features | 0,12 | la régularisation ne suffit pas |
| GRU sans ré-injection (planning seul) | 0,125 | ce n'est donc pas la dérive |
| GRU stateful, mêmes features | 0,145 (train 0,06) | encore plus de mémorisation : l'état caché peut « compter » |
| MLP avec ré-injection, teacher forcing seul | 467 | explosion : sans scheduled sampling le rêve diverge |
| **retrait des features calendrier** (distance Terre-Mars, angle Soleil-Mars-Terre), 47 features physiques | **0,083** | le couple (distance Soleil, distance Terre) identifiait la date ; sans lui, le modèle doit s'appuyer sur la physique |
| + normalisation robuste (plancher sur l'écart-type, clip ±10) | — | des modes de pointage absents de annee1 donnaient des entrées standardisées à 900 000 |

Leçon : avec une seule année d'entraînement, tout ce qui permet de reconnaître la date
est un piège ; il faut donner au modèle les causes (Soleil, ombre, pointage, commandes)
et rien qui ressemble à une horloge.

## 4. Résultats

RMSE de simulation (7 jours réels puis rêve sur tout le reste), évaluée avec `evaluate.py` :

| Modèle | Strict : annee1 → annee2 | 2 ans : annee1 + ½ annee2 → ½ annee2 |
|---|---|---|
| Constante (moyenne du train) | 0,1058 | 0,1048 |
| Constante calée sur les 7 premiers jours | 0,1116 | 0,1269 |
| Persistance de la dernière heure | 0,1378 | 0,1416 |
| Gradient boosting heure par heure, features physiques (sans mémoire) | 0,0920 | 0,0998 |
| Gradient boosting, toutes features par lettre (composant de la deuxième semaine) | 0,0891 | 0,0823 |
| Gradient boosting, commandes sans lettre, 5 sacs temporels (le composant livré) | 0,0793 | 0,0818 |
| GRU physique seul (1 modèle, 128 neurones, 47 features) | 0,0821 | 0,0901 |
| GRU physique + comptes horaires de commandes (1 modèle, 196 features) | 0,0795 | 0,0901 |
| Ensemble de 5 GRU (3 en chrono) | 0,0797 | 0,0897 |
| World model de la deuxième semaine = 0,55 × GRU physique + 0,45 × gradient boosting par lettre | 0,0779 | 0,0830 |
| **World model livré = 0,5 × ensemble de GRU + 0,5 × gradient boosting** | **0,0751** | **0,0808** |

Par rapport à la constante, le système livré réduit la RMSE de 29 % en strict et de 23 % en
chronologique. Le flux officiel (`make_validation.py` → `predict.py` → `evaluate.py`) donne
exactement 0,075095 sur 530 409 cellules notées. Le détail de la recherche qui a mené au
mélange est en section 7 (deuxième semaine) et en section 8 (troisième passe).

L'ensemble de GRU n'améliore presque pas la moyenne (0,0797 contre 0,0795-0,0804 pour
chaque modèle) : les erreurs restantes des GRU sont systématiques (régimes non prévisibles
depuis le planning), pas du bruit d'entraînement ; c'est un modèle d'une autre famille, le
gradient boosting, qui apporte la diversité utile (sections 7 et 8) : les erreurs des deux
composantes ne sont corrélées qu'à 0,79. L'écart-type de l'ensemble reste utile comme
indicateur : la corrélation de Spearman entre l'écart-type et l'erreur absolue est de
0,83, mais il est très sous-dispersé (RMSE observée ≈ 16 fois l'écart-type moyen, figure
`calibration_incertitude.png` ; le composant gradient boosting, déterministe, pèse la
moitié du mélange et ne contribue pas à la dispersion) : il classe bien les heures
incertaines, il ne les quantifie pas.

### Courbe de dérive (`derive.png`)

RMSE par semaine depuis le début du rêve, sur 98 semaines :

| Horizon | RMSE moyenne |
|---|---|
| semaines 1-4 | 0,039 |
| semaines 5-50 | 0,064 |
| semaines 51-98 | 0,084 |

L'erreur ne s'accumule pas : elle suit les saisons (pics pendant les saisons
d'éclipses, semaines 25-35 et 55-70, où même la constante et le gradient boosting
montent à 0,13-0,15). La courbe du world model reste sous celle des deux références
presque partout ; la pire semaine (semaine 63, 0,141) tombe dans la deuxième saison
d'éclipses, pendant la conjonction solaire, sur le creux inexpliqué de NPWD2551 en
octobre 2011 (section 8). Le mécanisme anti-dérive est double : l'état caché est
entraîné sur des flux longs (stateful) et le modèle a appris à vivre avec ses propres
prédictions (scheduled sampling jusqu'à p = 0).

### Analyse des erreurs (`erreur_par_circuit.png`, `reve_vs_realite.png`)

| Circuit | RMSE du mélange livré | RMSE ensemble de GRU | RMSE gradient boosting | écart-type réel | part de la MSE totale |
|---|---|---|---|---|---|
| NPWD2551 | 0,261 | 0,280 | 0,279 | 0,366 | 37 % |
| NPWD2451 | 0,222 | 0,236 | 0,228 | 0,299 | 26 % |
| NPWD2532 | 0,129 | 0,139 | 0,125 | 0,158 | 9 % |
| NPWD2851 | 0,124 | 0,120 | 0,149 | 0,136 | 8 % |
| NPWD2881 | 0,083 | 0,087 | 0,087 | 0,122 | 4 % |
| NPWD2561 | 0,082 | 0,088 | 0,087 | 0,120 | 4 % |
| NPWD2531 | 0,078 | 0,075 | 0,094 | 0,081 | 3 % |

(`experiments/tableaux_rapport.py`, `reports/tableaux_rapport.json`.) Deux circuits font
63 % de l'erreur quadratique ; sur chacun, le mélange bat ses deux composantes.

* **Ce qui marche** : les saisons d'éclipses (montée et descente de NPWD2551 et NPWD2451
  en octobre 2010 et juillet 2011, cycles horaires de chauffage à l'échelle de
  l'orbite, voir `zoom_horaire.png`), la saison solaire de NPWD2451 (bosse de février
  2012), le niveau de NPWD2532.
* **Ce qui ne marche pas** : les changements de régime que le planning ne contient pas.
  NPWD2551 s'éteint fin janvier 2011 (le modèle maintient 0,2 pendant deux semaines),
  se rallume en février 2011 puis en février-mars 2012 sans éclipse ; NPWD2532 saute à
  1,48 en janvier 2011 (le modèle reste à 1,42). Les circuits NPWD2881, NPWD2561 et
  NPWD2791 sont des lignes à deux états (0 ou 0,1) qui commutent de façon
  imprévisible : le modèle prédit la moyenne, ce qui minimise la RMSE mais ne
  reproduit pas les créneaux ; depuis que le GRU lit les comptes de commandes, il y fait
  jeu égal avec le gradient boosting (0,087 contre 0,087 sur NPWD2881, où le mélange est
  passé de 0,124 à 0,083).
* **Ce que les commandes apportent, et comment** : avec les 77 états d'interrupteurs dmop
  codés par lettre de variante, le GRU tombe à 0,103 (strict) et 0,121 (2 ans) : il apprend
  des coïncidences entre commandes et régimes de l'année d'entraînement qui ne se
  reproduisent pas, parce que les lettres changent d'une année à l'autre (section 8). Les
  mêmes commandes lues **sans leur lettre**, comme des comptes horaires par stem (149
  colonnes), font gagner 0,0026 au GRU en strict sans rien coûter en chronologique. La
  version livrée lit donc la physique (Soleil, ombre, orbite, pointage, constante solaire,
  durée d'éclipse : 47 features) et ces comptes : 196 features.

## 5. Ablations (`ablations.py`, `reports/ablations.md`)

Chaque ligne ré-entraîne le modèle (1 graine, 80 époques, protocole strict annee1 → annee2)
en retirant ou en ajoutant un ingrédient, à partir du jeu de features livré (physique +
comptes de commandes, 196 colonnes). Les ablations du GRU sont mesurées sur le GRU seul
(0,0795) pour rester lisibles ; les deux premières lignes situent le système complet. Le
bruit entre graines est de ±0,001 : un écart inférieur à 0,002 ne veut rien dire.

| Ablation | RMSE | écart | ce que ça montre |
|---|---|---|---|
| **système complet : GRU + composant gradient boosting (0,5)** | **0,0749** | −0,005 | la diversité des familles de modèles vaut plus que n'importe quelle feature |
| GRU seul | 0,0795 | référence | |
| sans Soleil (saaf) | 0,1029 | +0,023 | l'ingrédient le plus important : NPWD2451 passe de 0,24 à 0,35 et NPWD2532 de 0,14 à 0,26 ; le chauffage est d'abord une question d'exposition au Soleil |
| sans tendances (ltdata : constante solaire, durées d'éclipse et d'occultation) | 0,0885 | +0,009 | la saison compte, même avec les éclipses heure par heure |
| sans pointage (ftl) | 0,0847 | +0,005 | NPWD2551 passe de 0,28 à 0,33 : l'orientation (communication vers la Terre ou non) change la face exposée |
| sans éclipses ni événements (evtf) | 0,0794 | −0,000 | la durée d'éclipse journalière de ltdata et les angles solaires portent tout pour le GRU ; l'umbra horaire ne lui ajoute rien (le gradient boosting, lui, la lit) |
| sans comptes de commandes (dmop_stem_cnt : 47 features physiques) | 0,0821 | +0,003 | ce que les commandes apportent une fois débarrassées de leur lettre : surtout les lignes à deux états (NPWD2881 : 0,125 → 0,088 ; NPWD2561 : 0,119 → 0,087) |
| sans scheduled sampling (teacher forcing seul) | 0,0950 | +0,016 | sans apprendre à rêver, le modèle dérive : NPWD2551 monte à 0,39 |
| fenêtres indépendantes (état remis à zéro, 20 époques) | 0,0852 | +0,006 | l'état caché doit être entraîné sur des flux longs pour tenir 16 000 pas |
| sans mémoire (MLP heure par heure) | 0,0823 | +0,003 | la mémoire aide peu une fois les bonnes features en place ; sa valeur est dans les cycles horaires et la stabilité, pas dans la moyenne |
| sans ré-injection (planning seul) | 0,0788 | −0,001 | relire sa propre prédiction n'apporte rien : la mémoire utile est dans h_t, pas dans y_{t-1}. Les 7 jours réels servent à initialiser l'état caché |
| avec calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0,0923 | +0,013 | les features « horloge » datent chaque heure et le modèle mémorise l'année d'entraînement : NPWD2551 à 0,37 |
| avec interrupteurs dmop par lettre (dmop_switch : 273 features) | 0,0938 | +0,014 | NPWD2531 passe de 0,08 à 0,22 : le modèle associe des lettres de commande à des régimes de annee1, et les lettres changent dans annee2 (section 8) |
| avec toutes les colonnes dmop (1 485 features) | 0,1012 | +0,022 | NPWD2531 à 0,33, NPWD2851 à 0,22 : plus d'erreur que la constante sur ces deux circuits |

Ce que ça dit du problème : la consommation de Mars Express est pilotée par la physique
(Soleil, saison, orientation). Les commandes aident si on les lit sans leur lettre de
variante, comme des comptes d'événements, et nuisent dès qu'elles sont codées par lettre ou
qu'on en donne toutes les colonnes à un réseau entraîné sur une seule année. La mémoire sert
surtout à garder la simulation stable sur deux ans, et le scheduled sampling est ce qui la
rend possible.

## 6. Bonus : le pilote (`pilot.py`, `reports/pilot.json`)

Le world model sert de simulateur pour comparer des plannings sur une fenêtre de 14
jours en saison d'éclipses (25 octobre → 8 novembre 2010), initialisée par les 7 jours
réels précédents. Les fenêtres de communication avec la Terre (`flagcomms`) ne sont
jamais touchées. Énergie = somme horaire des 33 circuits sur les 14 jours.

| Planning | Énergie simulée | Écart |
|---|---|---|
| Réel (mesuré : 1 726,2) | 1 731,2 ± 1,0 | référence (le simulateur retrouve la somme réelle à 0,3 % près) |
| Hors communication, tout en pointage EARTH (plus d'observation) | 1 721,4 | −0,56 % |
| Hors communication, tout en NADIR (observation continue) | 1 726,8 | −0,25 % |
| Hors communication, sans manœuvres (SLEW rendu à EARTH) | 1 728,2 | −0,17 % |
| Hors communication, aucune commande dmop | 1 729,7 | −0,09 % |
| **Recherche locale** : permutations d'heures de pointage à l'intérieur de chaque journée, même volume d'observation par mode et par jour, 120 essais | 1 730,2 | −0,06 % |

Lecture : le chauffage dépend surtout de la physique (ombre, Soleil), pas du
planning ; ce que le pilote peut gagner à communications et observations constantes
est de l'ordre de 0,1 à 0,3 %. Le gain « facile » (−0,56 %) consiste à ne plus
observer, ce qui n'est pas un planning acceptable. À la soutenance : le simulateur est
fidèle sur l'énergie totale (+0,3 %), mais ses marges (±1,0, écart-type entre les cinq
GRU) sont du même ordre que les gains de la recherche locale, donc ces 0,06 % ne sont pas
démontrés ; seuls « tout en NADIR » (−0,25 %) et « tout en EARTH » (−0,56 %) sortent de la
marge. Le système de la deuxième semaine donnait des écarts deux fois plus grands (−1,2 %
pour « tout en EARTH », −0,4 % pour « moins de manœuvres ») : le modèle actuel, qui lit les
commandes sans leur lettre et moyenne cinq sacs de gradient boosting, réagit moins au
planning. Aucune des deux réponses n'est vérifiable, ces plannings n'ayant pas été volés :
c'est la limite du bonus. Comme les deux composantes lisent les commandes, le simulateur
réagit aussi à leur retrait (−0,09 %).

## 7. Deuxième semaine : ce qui a été essayé pour aller plus loin

*Les chiffres de cette section sont ceux du système de la deuxième semaine (GRU physique à
47 features, gradient boosting lisant les commandes par lettre, mélange 0,55 / 0,45) ; la
section 8 décrit ce qui a changé depuis et le système livré.*

Toutes les idées ci-dessous ont été mesurées dans les **deux protocoles** (strict annee1 → annee2,
et chronologique annee1 + ½ annee2 → ½ annee2), une graine chacune, avec le harnais
`experiments/sweep.py` (`reports/experiments.csv`). Le bruit entre graines est de ±0,001
(trois graines du modèle de base : strict 0,0821 / 0,0828 / 0,0830 ; chrono 0,0901 / 0,0915 / 0,0916) :
une différence inférieure à 0,002 n'est pas significative.

### Features (`dio/features.py`, groupes `flux`, `hist`, `leadlag`, `dmop_prefix_rec`, `dmop_stem_rec`)

| Ajout au GRU physique (47 features) | strict | chrono | moyenne | lecture |
|---|---|---|---|---|
| aucun (base) | 0,0821 | 0,0901 | 0,0861 | référence |
| mémoire thermique explicite (heures depuis l'ombre, ombre cumulée 7 h / 24 h / 7 j, flux moyen 6 h / 24 h) | 0,0814 | 0,0903 | 0,0858 | dans le bruit : le GRU calcule déjà cela |
| flux solaire par face (constante solaire × cos tronqué, corrigé de l'ombre) | 0,0822 | 0,0918 | 0,0870 | rien |
| anticipation t−3…t+3 des colonnes clés | 0,0827 | 0,1019 | 0,0923 | nuit en chrono : plus de colonnes = plus de mémorisation |
| recence par sous-système dmop (31 col.) | 0,0854 | 0,0938 | 0,0896 | nuit |
| comptes par sous-système dmop (38 col.) | 0,0821 | 0,1009 | 0,0915 | nuit en chrono |
| recence par stem dmop, toutes variantes confondues (149 col.) | 0,1029 | 0,0942 | 0,0986 | nuit : même rendues robustes au changement de lettre, les commandes font sur-apprendre le GRU |

### Architecture et entraînement

| Variante | strict | chrono | moyenne |
|---|---|---|---|
| LSTM au lieu de GRU | 0,0825 | 0,0899 | 0,0862 |
| 256 neurones | 0,0836 | 0,0903 | 0,0869 |
| 2 couches | 0,0835 | 0,0993 | 0,0914 |
| 120 époques + moyenne des poids des 20 dernières (SWA) | 0,0832 | 0,0926 | 0,0879 |
| segments de BPTT de 336 h / 32 flux | 0,0830 / 0,0833 | 0,0975 / 0,0983 | 0,0903 / 0,0908 |
| dropout 0,2 / 0,4 ; bruit d'état 0 / 0,15 ; lr 5e-4 / 2e-3 ; weight decay 1e-2 | 0,082-0,083 | 0,090-0,097 | 0,086-0,090 |
| perte de Huber (seuil 0,3 / 0,1) | 0,0825 / 0,0837 | 0,0916 / 0,1034 | 0,0871 / 0,0935 |
| adaptation au test : 10 à 100 pas de gradient sur les 7 jours réels avant le rêve | 0,084 → 0,104 | — | sur-apprend une semaine |

Conclusion : le GRU physique est à son plateau (≈ 0,082 / 0,090). Aucune variante seule ne le
déplace de plus que le bruit ; plusieurs le dégradent nettement en protocole chronologique, qui
est le plus sensible à la mémorisation.

### Ce qui a marché : un second modèle différent, pas un GRU de plus

Les erreurs de deux GRU (graines, variantes d'architecture, avec ou sans ré-injection) sont
corrélées à 0,96-0,99 : les moyenner n'apporte rien. Un gradient boosting heure par heure
entraîné sur **toutes** les features (816 colonnes, commandes dmop comprises, sans les
calendriers) est moins bon seul en strict (0,089) mais ses erreurs ne sont corrélées qu'à 0,75
avec celles du GRU : il lit les commandes, que le GRU ne peut pas exploiter sans sur-apprendre,
et un arbre extrapole en escalier là où un réseau s'emballe. Mélangé, il fait baisser la RMSE
dans les deux protocoles (`reports/gbm_blend.json`, `reports/gbm_params.json`) :

| | strict | chrono |
|---|---|---|
| GRU (ensemble de 5 / 1 modèle) | 0,0821 | 0,0901 |
| gradient boosting seul, toutes features sauf calendrier | 0,0891 | 0,0823 |
| 0,7 GRU + 0,3 GBM | 0,0780 | 0,0849 |
| 0,6 GRU + 0,4 GBM | 0,0778 | 0,0836 |
| **0,55 GRU + 0,45 GBM (livré en deuxième semaine)** | **0,0779** | **0,0830** |
| 0,5 GRU + 0,5 GBM | 0,0783 | 0,0825 |
| 0,3 GRU + 0,7 GBM | 0,0810 | 0,0814 |

(ensembles de 5 GRU en strict, 3 en chronologique.) Le poids optimal n'est pas le même dans
les deux protocoles : 0,4 de GBM en strict, 0,7 en chronologique. À deux ans d'entraînement,
le gradient boosting seul fait mieux que le GRU (0,082 contre 0,091) : plus il y a d'années,
plus les commandes deviennent exploitables, et le test final est entraîné sur deux ans. Le
poids retenu alors, 0,45, était le compromis : à 0,0002 du meilleur en strict, à 0,0016 en
chronologique, et le réseau de neurones reste majoritaire. Les hyperparamètres du GBM
comptent peu (cinq réglages à ±0,0005, `reports/gbm_params.json`).

Ce qui n'a rien apporté de plus (`reports/blend_all.json`, `reports/forests.json`,
`reports/gbm_roll.json`) :

* une troisième et une quatrième famille, ExtraTrees (0,090 / 0,093 seul) et RandomForest
  (0,093 / 0,091) : la grille de poids sur les quatre modèles leur donne 0 à 0,1 et ne gagne
  rien (0,0776 contre 0,0778 en strict) ;
* des statistiques glissantes multi-échelles (6 h, 24 h, 3 j, 7 j, 110 colonnes) pour le
  GBM : 0,0900 contre 0,0891 seul ;
* des poids de mélange **par circuit** : choisis sur un protocole, ils sont pires que les
  poids globaux sur l'autre (strict → chrono 0,0851 ; chrono → strict 0,0824) : ils
  sur-apprennent la période de validation ;
* un second GRU entraîné sur toutes les features (0,106 seul, 0,0803 en mélange 0,8 / 0,2
  avec le GRU physique, mais rien de plus une fois le GBM présent).

**Règlement.** Le modèle final reste un réseau de neurones (l'ensemble de GRU, 55 % du
mélange à ce stade, 50 % dans la version livrée, section 8) ; le gradient boosting est un
composant, ce que le règlement autorise explicitement (« les modèles classiques servent de
baseline ou de composant uniquement »). `train.py --gbm-weight 0` donne la version réseau
pur (0,080 en strict pour la version livrée), si l'orga préfère.

## 8. Troisième passe : d'où vient l'erreur, et pourquoi une année ne ressemble pas à la suivante

Avant d'empiler d'autres variantes de modèles, cette passe a décomposé l'erreur du système
livré (`experiments/analyse_erreurs.py`, `analyse_circuits.py`, `analyse_trous.py`,
`analyse_conjonction.py`, `reports/analyse_erreurs.json`).

### Deux circuits, deux pour cent des heures

* **Deux circuits font 60 % de l'erreur quadratique** : NPWD2551 (35 à 39 % selon le
  protocole) et NPWD2451 (25 %). Les 31 autres se partagent le reste.
* **60 % de l'erreur vient de 2 % des heures** (|erreur| > 0,3) : des marches, pas du bruit.
* NPWD2551 est un chauffage à paliers : 78 % du temps à zéro, sinon des niveaux discrets
  (0,26 ; 0,67 ; 1,08 ; 1,6 ; 1,9). Il s'allume pendant les deux saisons d'éclipses de
  l'année martienne (corrélation 0,6 avec l'ombre), avec une configuration commandée par
  l'interrupteur AXXX301.
* NPWD2451 vaut 0,05, 1,1 ou 2,16 (un ou deux éléments allumés) ; sa moyenne horaire est un
  rapport cyclique de thermostat, anticorrélé au flux solaire reçu. Autocorrélation horaire
  de 0,5 seulement : c'est la partie vraiment difficile.
* **Les pires semaines des deux protocoles sont les mêmes** : les jours 400 à 490 de
  l'année 2, deuxième saison d'éclipses, qui contient une **conjonction solaire** : cinq
  semaines sans aucune commande DMOP ni contact sol (`dmop.csv` a des trous de 35 jours aux
  jours 191 et 457 de l'année 2 ; l'année 1 n'a que des trous de 7 à 15 jours). Sur ces
  semaines, GRU et GBM sur-prédisent NPWD2551 de +0,30 en moyenne. Aucune feature isolée
  n'explique cette sur-prédiction (test d'intervention : forcer un groupe à sa valeur des
  semaines précédentes ne ramène la prédiction que de 0,72 à 0,60 au mieux) ; un oracle qui
  retirerait le biais hebdomadaire de ce seul circuit donnerait 0,0726 au lieu de 0,0777.

### Les lettres des commandes changent d'une année à l'autre

Le point le plus important de cette passe. Une commande DMOP se termine par une lettre de
variante (`AXXX301A`, `AXXX301B`). Pour NPWD2551, l'interrupteur AXXX301 vaut A ou B en
année 1, mais B, C ou E en année 2 : la configuration « saison d'éclipses » s'appelle A la
première année et C la seconde. Ce n'est pas un cas isolé (`experiments/analyse_variantes.py`) :

| | |
|---|---|
| stems d'interrupteurs dont le jeu de lettres change entre les deux années | 20 sur 29 |
| stems sans aucune lettre commune entre les deux années | 7 sur 29 |
| commandes du vocabulaire (198) présentes uniquement en année 2 | 49 |
| commandes présentes uniquement en année 1 | 9 |

Conséquence : toute feature encodée **par lettre** (`dm_s_` états d'interrupteurs, `dm_c_`
comptes et `dm_r_` récences par commande) est constante ou jamais vue l'année suivante. Un
arbre qui a appris « AXXX301_A = 1 ⇒ chauffage » se retrouve en année 2 avec A = 0 et B = 0
en permanence, une combinaison qu'il n'a vue que trois jours : hors saison d'éclipses, il
sur-prédit NPWD2551 de +0,33 toute l'année 2. Le GRU, lui, n'utilise pas les commandes, et il
est à peu près juste sur ces semaines.

Le protocole strict (année 1 → année 2) mesure exactement cette dérive, et il est le meilleur
proxy de l'évaluation finale (deux années d'entraînement, une troisième en test, avec de
nouvelles lettres). Le protocole chronologique, lui, teste à lettres constantes. Les deux
disent des choses différentes et vraies : sans les features par commande, le GBM passe de
0,0891 à 0,0868 en strict, mais de 0,0823 à 0,0862 en chronologique. À lettres stables, les
commandes portent un vrai signal ; à lettres changeantes, elles trompent.

Aucun encodage « sans lettre » de l'**état** d'un interrupteur n'est stable d'une année à
l'autre (rang d'apparition de la variante, parité des changements, variante majoritaire :
tous s'inversent entre année 1 et année 2). Ce qui est stable, ce sont les **événements** :
le nombre de commandes d'un stem dans l'heure, toutes lettres confondues (`dm_sc_`, 149
colonnes), et pour chaque interrupteur les heures écoulées depuis le dernier changement de
variante et les heures restant jusqu'au prochain, le planning étant connu (`dm_sw_`, 58
colonnes). Ces features entrent dans la version 5 du cache.

### Ce que ça change, modèle par modèle (`reports/gbm_lettres.json`, `reports/reblend_lettres.json`, `reports/experiments.csv`)

Gradient boosting seul, mêmes hyperparamètres, GRU de référence inchangé pour les mélanges :

| features du GBM | strict | chrono | mélange 0,55 / 0,45 strict | chrono |
|---|---|---|---|---|
| toutes (par lettre), livraison précédente | 0,0891 | 0,0823 | 0,0779 | 0,0830 |
| sans les comptes / récences par commande (`dm_c_`, `dm_r_`) | 0,0868 | 0,0862 | 0,0783 | 0,0838 |
| sans les états d'interrupteurs par lettre (`dm_s_`) | 0,0915 | 0,0836 | 0,0786 | 0,0834 |
| sans les deux (plus aucune lettre) | 0,0818 | 0,0848 | 0,0769 | 0,0832 |
| **sans les deux + comptes par stem + changements d'interrupteurs** | **0,0822** | **0,0829** | **0,0766** | **0,0824** |
| toutes + comptes par stem + changements | 0,0891 | 0,0817 | 0,0777 | 0,0821 |
| « oubli des lettres » (train ×3, lettres masquées par stem au hasard) | 0,0903 | 0,0817 | 0,0778 | 0,0825 |

Retirer les lettres fait gagner 0,007 au GBM en strict ; les features sans lettre rendent en
chrono ce que les lettres apportaient (0,0829 contre 0,0848 sans rien). Tant que les features
par lettre sont présentes, les arbres les préfèrent et l'augmentation ne les rend pas robustes.
Le jeu retenu est le cinquième ; les hyperparamètres n'y changent rien d'utile : 7 feuilles
0,0803 / 0,0849, 15 feuilles (défaut) 0,0822 / 0,0829, 31 feuilles 0,0845 / 0,0824
(`reports/gbm_p7.json`, `gbm_p31.json`). Les arbres peu profonds généralisent d'une année à
l'autre, les profonds exploitent les lettres stables ; le défaut est le compromis.

GRU seul (1 graine, `experiments/sweep.py`) :

| entrées du GRU | strict | chrono |
|---|---|---|
| physique seule (saaf, ltdata, evtf, ftl), référence | 0,0821 | 0,0901 |
| **+ comptes horaires par stem (`dmop_stem_cnt`, 149 colonnes)** | **0,0795** | **0,0901** |
| idem, autre graine | 0,0804 (réf. 0,0828) | 0,0902 (réf. 0,0915) |
| + changements d'interrupteurs (`dmop_sw`, récences) | 0,0920 | 0,1037 |
| + comptes par stem + changements | 0,1007 | 0,1121 |
| + comptes par stem + comptes par sous-système et intervalles (`dmop_base`) | 0,0823 | 0,0956 |
| + comptes par stem + comptes par sous-système seuls | 0,0801 | 0,0908 |

C'est la première amélioration du GRU depuis la fin de la première semaine, et elle vient des
commandes : le GRU peut lire les commandes à condition qu'elles soient **sans lettre et
ponctuelles** (un compte dans l'heure). Les récences (« depuis combien d'heures… ») lui
nuisent toujours : ce sont des rampes de plusieurs semaines, c'est-à-dire des horloges, et le
GRU s'en sert pour mémoriser la date (même mécanisme que les features de calendrier, § 3).
Avec ces nouvelles entrées, le dropout de features (0,1 : 0,0850 ; 0,3 : 0,0798) et une
taille cachée de 192 (0,0821) ne font pas mieux que les défauts.

Négatifs de la passe, pour mémoire (`reports/gbm_ewm.json`, `gbm_recency.json`,
`analyse_erreurs.json`) : moyennes exponentielles (inertie thermique à 3 h, 12 h, 48 h, 7 j)
pour le GBM, 0,0909 / 0,0866 contre 0,0891 / 0,0823 (lisses, donc mémorisables) ;
pondération des données récentes pour le GBM (année 1 à poids 0,5 ou 0,25), 0,0827 et
0,0836 contre 0,0823 ; mélange pondéré par l'incertitude du GRU, 0,0003 au mieux ;
projection de rang faible des prédictions, rien.

### Le système livré après cette passe (`weights_holdout`, `weights_holdout_split`, `reports/poids_commun.json`)

* GRU : entrées physiques (saaf, ltdata, evtf, ftl) **+ comptes horaires de commandes par
  stem** (196 entrées au lieu de 47), 5 graines en strict, 3 en chronologique ; rien d'autre
  ne change (`--keep-groups saaf,ltdata,evtf,ftl,dmop_stem_cnt`).
* Gradient boosting : toutes les colonnes sauf calendrier, statistiques glissantes, moyennes
  exponentielles et **toute feature par lettre** (`dm_s_`, `dm_c_`, `dm_r_`), soit 846
  colonnes dont les comptes par stem, les commandes à venir (heure suivante, 6 heures
  suivantes) et les changements d'interrupteurs ; 600 arbres à 0,02, 15 feuilles, 100 par
  feuille, l2 = 3 ; **bagging temporel en 5 sacs** (chaque sac est entraîné sans un bloc
  contigu d'un cinquième de la période, puis moyenne). Les sacs valent 0,0796 contre 0,0822
  pour un ajustement unique en strict (0,0820 contre 0,0829 en chronologique) : chaque sac
  doit extrapoler la saison qu'il n'a pas vue, et la moyenne lisse ces extrapolations.
* Mélange à parts égales (`--gbm-weight 0.5`).

| | strict (année 1 → année 2) | chrono (2 ans → ½ année 2) |
|---|---|---|
| ensemble de GRU seul | 0,0797 (était 0,0821) | 0,0897 (était 0,0907) |
| gradient boosting seul, 5 sacs | 0,0793 (était 0,0891) | 0,0818 (était 0,0823) |
| **mélange 0,5 / 0,5, livré** | **0,0751** (était 0,0779) | **0,0808** (était 0,0830) |

Le poids du GBM, poids par poids (`experiments/poids_commun.py`) :

| poids du GBM | 0,3 | 0,4 | **0,5** | 0,6 | 0,7 | 0,8 |
|---|---|---|---|---|---|---|
| strict | 0,0759 | 0,0753 | **0,0751** | 0,0752 | 0,0757 | 0,0766 |
| chrono | 0,0833 | 0,0819 | **0,0808** | 0,0802 | 0,0800 | 0,0802 |
| moyenne | 0,0796 | 0,0786 | **0,0780** | 0,0777 | 0,0779 | 0,0784 |

L'optimum strict est à 0,5, l'optimum chronologique à 0,7, la moyenne à 0,6 (0,0002 de
mieux que 0,5). On livre 0,5 : c'est l'optimum du protocole qui ressemble le plus à
l'évaluation finale (nouvelle année, nouvelles lettres), et le réseau de neurones reste à
parité dans le mélange, ce que demande le règlement.

Les semaines de la conjonction (63 à 70 de l'année 2) restent à 0,11-0,14 de RMSE dans les
deux systèmes : le chauffage NPWD2551 y consomme 0,3 de moins qu'en année 1 à ombre égale,
un décalage de niveau que rien dans le planning n'annonce. Le gain de cette passe vient de
tout le reste de l'année (semaines 1 à 8 : 0,040 contre 0,045 ; hors saisons d'éclipses, le
GBM ne sur-prédit plus NPWD2551).
