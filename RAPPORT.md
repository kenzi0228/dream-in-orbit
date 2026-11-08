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
* **Ce qui porte vraiment la mémoire** : l'ablation « sans ré-injection » (le GRU ne lit
  que le planning, section 5) donne la même RMSE. La boucle état → état suivant
  passe donc par l'état caché h_t, pas par la prédiction ré-injectée ; les 7 jours
  réels servent à initialiser h_t. Le scheduled sampling reste nécessaire dès que la
  ré-injection existe (sans lui : +0,006, et le MLP sans mémoire diverge).

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
| Gradient boosting heure par heure (sans mémoire) | 0,0920 | 0,0998 |
| GRU seul (1 modèle, 128 neurones) | 0,0821 | 0,0901 |
| **World model = ensemble de 5 GRU** | **0,0821** | — |

Par rapport à la constante, l'ensemble réduit la RMSE de 22 % ; par rapport au meilleur
modèle sans mémoire, de 11 %. Le flux officiel (`make_validation.py` → `predict.py` →
`evaluate.py`) donne exactement la même valeur, 0,082106, sur 530 409 cellules notées.

L'ensemble n'améliore presque pas la moyenne (0,0821 contre 0,0821-0,0831 pour chaque
modèle) : les erreurs restantes sont systématiques (régimes non prévisibles depuis le
planning), pas du bruit d'entraînement. L'écart-type de l'ensemble reste utile comme
indicateur : la corrélation de Spearman entre l'écart-type et l'erreur absolue est de
0,86, mais il est sous-dispersé (RMSE observée ≈ 8 fois l'écart-type, figure
`calibration_incertitude.png`) : il classe bien les heures incertaines, il ne les
quantifie pas.

### Courbe de dérive (`derive.png`)

RMSE par semaine depuis le début du rêve, sur 98 semaines :

| Horizon | RMSE moyenne |
|---|---|
| semaines 1-4 | 0,059 |
| semaines 5-50 | 0,070 |
| semaines 51-98 | 0,091 |

L'erreur ne s'accumule pas : elle suit les saisons (pics pendant les saisons
d'éclipses, semaines 25-35 et 55-70, où même la constante et le gradient boosting
montent à 0,13-0,15). La courbe du world model reste sous celle des deux références
presque partout ; la pire semaine (semaine 65, 0,151) correspond au creux inexpliqué
de NPWD2551 en octobre 2011. Le mécanisme anti-dérive est double : l'état caché est
entraîné sur des flux longs (stateful) et le modèle a appris à vivre avec ses propres
prédictions (scheduled sampling jusqu'à p = 0).

### Analyse des erreurs (`erreur_par_circuit.png`, `reve_vs_realite.png`)

| Circuit | RMSE world model | RMSE gradient boosting | écart-type réel |
|---|---|---|---|
| NPWD2551 | 0,286 | 0,408 | 0,366 |
| NPWD2451 | 0,227 | 0,278 | 0,299 |
| NPWD2532 | 0,136 | 0,151 | 0,158 |
| NPWD2881 | 0,124 | 0,101 | 0,122 |
| NPWD2851 | 0,120 | 0,164 | 0,136 |
| NPWD2561 | 0,117 | 0,098 | 0,120 |
| NPWD2531 | 0,081 | 0,184 | 0,081 |

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
  reproduit pas les créneaux ; le gradient boosting fait mieux sur ceux-là (il
  exploite des commandes dmop que le GRU n'a pas).
* **Ce que les commandes n'apportent pas** : avec les 77 états d'interrupteurs dmop,
  le GRU tombe à 0,103 (strict) et 0,121 (2 ans) : il apprend des coïncidences entre
  commandes et régimes de l'année d'entraînement qui ne se reproduisent pas. La
  version finale ne lit donc que la physique (Soleil, ombre, orbite, pointage,
  constante solaire, durée d'éclipse) : 47 features.

## 5. Ablations (`ablations.py`, `reports/ablations.md`)

Chaque ligne ré-entraîne le modèle (1 graine, 80 époques, protocole strict annee1 → annee2)
en retirant ou en ajoutant un ingrédient. Modèle complet : 0,0821.

| Ablation | RMSE | écart | ce que ça montre |
|---|---|---|---|
| sans Soleil (saaf) | 0,0988 | +0,017 | l'ingrédient le plus important : NPWD2451 passe de 0,23 à 0,33 ; le chauffage est d'abord une question d'exposition au Soleil |
| sans tendances (ltdata : constante solaire, durées d'éclipse et d'occultation) | 0,0891 | +0,007 | la saison compte, même avec les éclipses heure par heure |
| sans pointage (ftl) | 0,0880 | +0,006 | NPWD2551 passe de 0,29 à 0,34 : l'orientation (communication vers la Terre ou non) change la face exposée |
| sans éclipses ni événements (evtf) | 0,0838 | +0,002 | surprenant : la durée d'éclipse journalière de ltdata et les angles solaires portent déjà l'essentiel ; l'umbra horaire affine les cycles |
| sans scheduled sampling (teacher forcing seul) | 0,0876 | +0,006 | sans apprendre à rêver, le modèle dérive davantage (NPWD2451 0,26 contre 0,23) |
| fenêtres indépendantes (état remis à zéro, 20 époques) | 0,0865 | +0,004 | l'état caché doit être entraîné sur des flux longs pour tenir 16 000 pas |
| sans mémoire (MLP heure par heure) | 0,0841 | +0,002 | la mémoire aide peu une fois les bonnes features en place ; sa valeur est dans les cycles horaires et la stabilité, pas dans la moyenne |
| sans ré-injection (planning seul) | 0,0818 | −0,000 | relire sa propre prédiction n'apporte rien : la mémoire utile est dans h_t, pas dans y_{t-1}. Les 7 jours réels servent à initialiser l'état caché |
| avec calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0,0859 | +0,004 | les features « horloge » dégradent ; avec le reste de la régularisation l'effet est modéré, il était massif sur les premiers essais |
| avec commandes dmop (préfixes + 77 interrupteurs) | 0,1025 | +0,020 | NPWD2531 passe de 0,08 à 0,31 : le modèle associe des commandes à des régimes de annee1 qui changent de nom dans annee2 |
| avec toutes les commandes dmop (560 features) | 0,1061 | +0,024 | pire que la constante sur NPWD2551 (0,42) |

Ce que ça dit du problème : la consommation de Mars Express est pilotée par la
physique (Soleil, saison, orientation) ; les commandes, telles qu'elles sont codées
dans dmop, sont trop instables d'une année à l'autre pour qu'un modèle entraîné sur
une seule année en tire une loi. Et la mémoire sert surtout à garder la simulation
stable sur deux ans.

## 6. Bonus : le pilote (`pilot.py`, `reports/pilot.json`)

Le world model sert de simulateur pour comparer des plannings sur une fenêtre de 14
jours en saison d'éclipses (25 octobre → 8 novembre 2010), initialisée par les 7 jours
réels précédents. Les fenêtres de communication avec la Terre (`flagcomms`) ne sont
jamais touchées. Énergie = somme horaire des 33 circuits sur les 14 jours.

| Planning | Énergie simulée | Écart |
|---|---|---|
| Réel (mesuré : 1 726,2) | 1 727,4 ± 0,9 | référence (le simulateur retrouve la somme réelle à 0,07 % près) |
| Hors communication, tout en pointage EARTH (plus d'observation) | 1 693,9 | −1,9 % |
| Hors communication, tout en NADIR (observation continue) | 1 716,5 | −0,6 % |
| Hors communication, sans manœuvres (SLEW rendu à EARTH) | 1 717,5 | −0,6 % |
| **Recherche locale** : permutations d'heures de pointage à l'intérieur de chaque journée, même volume d'observation par mode et par jour, 120 essais | 1 724,0 | −0,2 % |

Lecture : le chauffage dépend surtout de la physique (ombre, Soleil), pas du
planning ; ce que le pilote peut gagner à communications et observations constantes
est de l'ordre de 0,2 à 0,6 %. Le gain « facile » (−1,9 %) consiste à ne plus
observer, ce qui n'est pas un planning acceptable. À la soutenance : le simulateur est
fidèle sur l'énergie totale, mais ses marges (±0,9) sont du même ordre que les gains
de la recherche locale, donc ces 0,2 % ne sont pas démontrés ; les scénarios à −0,6 %
(moins de manœuvres, observation continue plutôt que fragmentée) sont plus robustes.
