| cycle | stratégie | km | H2 g/100 km | H2 corr. g/100 km | Δ vs règles % | Δ vs ECMS % | ΔSOC pts | SOC min | SOC max | η_fc moy | violations | bilan % | calcul s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wltc | rules_baseline | 23.27 | 861.7 | 821.6 | 0.0 | 2.2 | 10.11 | 0.563 | 0.701 | 0.495 | 0 | 2.9e-12 | 0.4 |
| wltc | ecms_const | 23.27 | 803.4 | 804.1 | -2.1 | 0.0 | -0.17 | 0.476 | 0.756 | 0.496 | 0 | 1.7e-12 | 0.29 |
| wltc | iems | 23.27 | 820.0 | 815.6 | -0.7 | 1.4 | 1.11 | 0.435 | 0.63 | 0.497 | 0 | 2.7e-12 | 0.96 |
| synth_mixed | rules_baseline | 30.61 | 850.3 | 832.3 | 0.0 | 1.1 | 6.4 | 0.551 | 0.688 | 0.529 | 0 | 7.4e-13 | 0.33 |
| synth_mixed | ecms_const | 30.61 | 824.6 | 823.5 | -1.1 | 0.0 | 0.39 | 0.323 | 0.797 | 0.54 | 0 | 1.3e-13 | 0.5 |
| synth_mixed | iems | 30.61 | 830.6 | 822.8 | -1.1 | -0.1 | 2.76 | 0.426 | 0.631 | 0.532 | 0 | 1.1e-12 | 0.56 |
| synth_urban | rules_baseline | 6.99 | 1018.3 | 969.8 | 0.0 | -1.4 | 2.44 | 0.591 | 0.629 | 0.329 | 0 | 1.2e-11 | 0.22 |
| synth_urban | ecms_const | 6.99 | 983.5 | 983.2 | 1.4 | 0.0 | 0.02 | 0.556 | 0.616 | 0.318 | 0 | 1.2e-11 | 0.22 |
| synth_urban | iems | 6.99 | 1023.3 | 942.3 | -2.8 | -4.2 | 4.13 | 0.599 | 0.641 | 0.333 | 0 | 1.2e-11 | 0.35 |
| synth_highway | rules_baseline | 47.89 | 952.7 | 936.4 | 0.0 | 1.5 | 9.61 | 0.561 | 0.696 | 0.563 | 0 | 1.6e-13 | 0.24 |
| synth_highway | ecms_const | 47.89 | 922.2 | 922.4 | -1.5 | 0.0 | -0.12 | 0.363 | 0.765 | 0.574 | 0 | 1.4e-13 | 0.23 |
| synth_highway | iems | 47.89 | 930.9 | 929.2 | -0.8 | 0.7 | 1.01 | 0.378 | 0.61 | 0.569 | 0 | 5.7e-14 | 0.4 |
| synth_sport | rules_baseline | 21.18 | 915.7 | 890.9 | 0.0 | 0.9 | 6.23 | 0.561 | 0.712 | 0.54 | 0 | 4.7e-13 | 0.17 |
| synth_sport | ecms_const | 21.18 | 883.0 | 882.7 | -0.9 | 0.0 | 0.09 | 0.468 | 0.61 | 0.54 | 0 | 5.0e-13 | 0.16 |
| synth_sport | iems | 21.18 | 890.9 | 879.6 | -1.3 | -0.4 | 2.86 | 0.538 | 0.636 | 0.542 | 0 | 4.5e-13 | 0.33 |

*ECMS λ constant calibré par dichotomie avec prédiction exacte (référence hors ligne) ; règles et iEMS avec le prédicteur bruité par défaut. Consommations corrigées du SOC final.*
