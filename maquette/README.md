# Maquette de l'interface du pilote

Trois écrans (1440 × 1000 px) de la console de planification que le world model rend possible :

1. **Rêve sur une fenêtre** : le planning réel simulé en boucle ouverte sur 14 jours, la puissance
   totale mesurée et prédite avec sa bande ±2σ, le planning des pointages aligné, les circuits qui
   pèsent le plus.
2. **Comparer des plannings** : quatre scénarios « et si » en comparaison appariée (chaque membre de
   l'ensemble simule les deux plannings), l'écart d'énergie avec la dispersion des membres, le détail
   du scénario choisi en heures par mode.
3. **Optimiser par recherche locale** : l'énergie simulée au fil des 120 permutations, les 15 gardées,
   le planning proposé.

Tous les chiffres sont les résultats réels du projet (`reports/pilot.json`, `weights_holdout/`,
`validation/`), fenêtre du 25 octobre au 7 novembre 2010, protocole strict. Les boutons et champs
montrent l'usage prévu ; rien n'est câblé.

```bash
.venv/bin/python maquette/build_maquette.py    # -> index.html (les trois écrans), ecrans/*.html (un par écran)
```

`index.html` s'ouvre dans un navigateur sans serveur ; `maquette.pdf` (un écran par page) et les
`ecran-*.png` sont des rendus de ces mêmes pages.
