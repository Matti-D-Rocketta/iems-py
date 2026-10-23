# Reprise M1 — état au 23/10/2026

Fait : config, modèles, cycles (WLTC reconstruit, synthétiques à jerk borné), candidats,
balance, optimiseur, dpt v1, stratégies (+ calibration ECMS), simulateur, métriques.

Choix de modélisation (à expliquer dans le README) :
- PàC : consommation convexe a + bP + cP² ajustée sur la carte η (sinon Hamiltonien
  tout-ou-rien et calibration de λ mal posée).
- Prédicteur : erreur lisse et autocorrélée (étirement temporel + biais d'amplitude) ;
  un bruit blanc sur la vitesse créait ±50 kW de demande fantôme.
- Admissibilité : plancher/plafond PàC anticipés sur Hp, robustes à ±1 pas ; en cas de
  conflit, consigne au milieu.

Mesures actuelles (WLTC, iEMS) : 824 g/100 km corrigé, ΔSOC +10,3 pts, 68 pas en violation.

Reste à faire :
1. Charge-sustaining iEMS : ΔSOC ≈ +10 pts sur WLTC. λ moyen ≈ 1,45 (< λ0 = 1,55), car
   l'usage batterie urbain/trafic tire λ vers le bas ; recaler la table d'usage, soc_ref et Kp.
2. Violations résiduelles (synth_sport, bruit de prédiction) : fixer un niveau de bruit de
   référence (0,02/0,03 ?) et publier la sensibilité au bruit dans le rapport.
3. Tests pytest (section 7), examples/compare_strategies.py, rapport de conformité M1.

Astuce console Windows : lancer Python avec PYTHONIOENCODING=utf-8 (caractères Δ, λ).
