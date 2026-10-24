# Note de reprise — M1 terminé le 24/10/2026, prochain jalon M2

## État

- Jalon M1 complet : 6/6 critères vérifiés par `examples/m1_report.py` (`docs/m1_report.md`),
  358 tests pytest verts (≈ 40 s), ruff propre, tout poussé sur `origin/main`.
- Résultats par défaut : 0 violation sur 5 cycles × 3 stratégies ; iEMS ≤ baseline sur 5/5 ;
  écart ≤ 1,6 % à l'ECMS calibré hors ligne ; ΔSOC iEMS WLTC −0,55 pt ; même SOC final
  (≈ 60 %) depuis 35 % ou 75 %.
- Sensibilité connue : tout tient jusqu'à 2× le bruit de prédiction par défaut ; à 3× des
  violations réapparaissent (55 sur WLTC). DPR : précision moyenne 0,87 sur les synthétiques.

## Choix de modélisation à défendre en entretien

1. Pile : consommation convexe `a + bP + cP²` ajustée sur la carte η (Hamiltonien convexe).
2. Prédicteur v1 : erreurs d'ordre 2 en τ, spécifiées à τ = 10 s (vitesse et accélération
   courantes mesurées) ; marge inertielle `m·v·σ_a/η` dans l'admissibilité.
3. Charge-sustaining : SOC de référence abaissé avec v² (réserve cinétique, ≈ 0,8 mesuré)
   + λ = λ0 + Kp·(SOC_cible − SOC), λ0 = λ* de la dichotomie ECMS sur WLTC (1,51).
4. WLTC = reconstruction structurelle (phases/distances/vitesses max GTR 15), remplaçable par
   la table officielle dans `src/iems/data/wltc_class3.csv`.

## Points ouverts (non bloquants)

- Près de `soc_max`, le ralenti de la pile (1 kW > P_aux 500 W) ne peut pas être écoulé à
  l'arrêt : dépassements structurels si `soc0` est pris dans la bande de dérating haute.
  L'iEMS s'en tient éloigné par construction ; à traiter si M4 expose le SOC initial.
- Paramètres `# TODO-thèse` à recaler depuis le manuscrit (masses, carte η, usage par schéma,
  seuils DPR → centres FCM).
- v2 : LSTM (torch) et flou (scikit-fuzzy) à iso-interface (`SpeedPredictor`, `Balance`).

## Pour démarrer M2

Prompt en section 10 de `PASSATION_iEMS_Python.md` (fichier local, exclu du git).
Points d'entrée : `iems.simulator.run/simulate`, `iems.metrics.compute_metrics(...).row()`,
`iems.models.list_cycles/get_cycle`, `iems.strategies.STRATEGIES`, `IEMSConfig.with_updates`
(validé en JSON aller-retour, `usage_table` compris). `examples/compare_strategies.run_campaign`
fournit déjà le tableau comparatif pour `POST /compare`.

Console Windows : `PYTHONUTF8=1` devant Python (caractères Δ, λ).
