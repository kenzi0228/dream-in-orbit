| Ablation | RMSE annee2 | écart vs complet |
|---|---|---|
| complet | 0.0821 | +0.0000 |
| sans_memoire (MLP heure par heure, avec re-injection) | 0.0841 | +0.0020 |
| sans_reinjection (planning seul) | 0.0818 | -0.0003 |
| sans_scheduled_sampling (teacher forcing seul) | 0.0876 | +0.0055 |
| fenetres_independantes (pas d'etat transmis) | 0.0865 | +0.0044 |
| sans_eclipses_evenements (evtf) | 0.0838 | +0.0017 |
| sans_soleil (saaf) | 0.0988 | +0.0167 |
| sans_tendances (ltdata) | 0.0891 | +0.0070 |
| sans_pointage (ftl) | 0.0880 | +0.0059 |
| avec_commandes_dmop (prefixes + interrupteurs) | 0.1025 | +0.0204 |
| avec_toutes_commandes_dmop (560 features) | 0.1061 | +0.0240 |
| avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0.0859 | +0.0038 |
