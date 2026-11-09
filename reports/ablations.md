| Ablation | RMSE annee2 | écart vs GRU seul |
|---|---|---|
| complet (GRU + gradient boosting) | 0.0751 | -0.0051 |
| GRU seul (sans composant gradient boosting) | 0.0802 | +0.0000 |
| sans_memoire (MLP heure par heure, avec re-injection) | 0.0818 | +0.0016 |
| sans_reinjection (planning seul) | 0.0794 | -0.0008 |
| sans_scheduled_sampling (teacher forcing seul) | 0.0924 | +0.0122 |
| fenetres_independantes (pas d'etat transmis) | 0.0898 | +0.0096 |
| sans_eclipses_evenements (evtf) | 0.0808 | +0.0006 |
| sans_soleil (saaf) | 0.1004 | +0.0201 |
| sans_tendances (ltdata) | 0.0885 | +0.0082 |
| sans_pointage (ftl) | 0.0898 | +0.0096 |
| sans_comptes_commandes (dmop_stem_cnt) | 0.0824 | +0.0022 |
| avec_interrupteurs_par_lettre (dmop_switch) | 0.0854 | +0.0051 |
| avec_toutes_commandes_dmop (toutes les colonnes) | 0.0867 | +0.0064 |
| avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0.0888 | +0.0086 |
