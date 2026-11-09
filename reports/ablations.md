| Ablation | RMSE annee2 | écart vs GRU seul |
|---|---|---|
| complet (GRU + gradient boosting) | 0.0749 | -0.0046 |
| GRU seul (sans composant gradient boosting) | 0.0795 | +0.0000 |
| sans_memoire (MLP heure par heure, avec re-injection) | 0.0841 | +0.0046 |
| sans_reinjection (planning seul) | 0.0818 | +0.0022 |
| sans_scheduled_sampling (teacher forcing seul) | 0.0876 | +0.0080 |
| fenetres_independantes (pas d'etat transmis) | 0.0865 | +0.0070 |
| sans_eclipses_evenements (evtf) | 0.0838 | +0.0043 |
| sans_soleil (saaf) | 0.0988 | +0.0193 |
| sans_tendances (ltdata) | 0.0891 | +0.0095 |
| sans_pointage (ftl) | 0.0880 | +0.0085 |
| avec_commandes_dmop (prefixes + interrupteurs) | 0.1025 | +0.0230 |
| avec_toutes_commandes_dmop (560 features) | 0.1061 | +0.0266 |
| avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0.0859 | +0.0064 |
