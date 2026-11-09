| Ablation | RMSE annee2 | écart vs GRU seul |
|---|---|---|
| complet (GRU + gradient boosting) | 0.0749 | -0.0046 |
| GRU seul (sans composant gradient boosting) | 0.0795 | +0.0000 |
| sans_memoire (MLP heure par heure, avec re-injection) | 0.0823 | +0.0028 |
| sans_reinjection (planning seul) | 0.0788 | -0.0007 |
| sans_scheduled_sampling (teacher forcing seul) | 0.0950 | +0.0155 |
| fenetres_independantes (pas d'etat transmis) | 0.0852 | +0.0057 |
| sans_eclipses_evenements (evtf) | 0.0794 | -0.0001 |
| sans_soleil (saaf) | 0.1029 | +0.0234 |
| sans_tendances (ltdata) | 0.0885 | +0.0090 |
| sans_pointage (ftl) | 0.0847 | +0.0052 |
| sans_comptes_commandes (dmop_stem_cnt) | 0.0821 | +0.0026 |
| avec_interrupteurs_par_lettre (dmop_switch) | 0.0938 | +0.0143 |
| avec_toutes_commandes_dmop (toutes les colonnes) | 0.1012 | +0.0217 |
| avec_calendrier (distance Terre-Mars, angle Soleil-Mars-Terre) | 0.0923 | +0.0127 |
