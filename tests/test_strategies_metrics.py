"""Tests de ``strategies.py`` et ``metrics.py`` : critères M1 ②③④, registre, baseline, métriques.

Les tests marqués ``slow`` (campagne 5 cycles, calibration ECMS) prennent plusieurs secondes ;
le reste tient en moins de 10 s. Toutes les fixtures sont locales au fichier.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from iems.config import IEMSConfig
from iems.metrics import Metrics, Violations, compute_metrics
from iems.models.cycles import get_cycle, list_cycles
from iems.simulator import SimResult, run, simulate
from iems.strategies import (
    IEMS,
    STRATEGIES,
    ECMSConst,
    EMSObservation,
    RulesBaseline,
    calibrate_ecms_const,
    make_strategy,
)

CYCLES = ("wltc", "synth_mixed", "synth_urban", "synth_highway", "synth_sport")
STRATEGY_NAMES = ("rules_baseline", "ecms_const", "iems")
N_PRED = 101  # Hp = 10 s au pas modèle 0,1 s → 101 points de demande prédite
TOL_W = 1e-6  # tolérance numérique sur les puissances (W)
WLTC_KM = 23.267  # 3,095 + 4,756 + 7,162 + 8,254 km (GTR n° 15)
ROW_KEYS = {
    "cycle",
    "stratégie",
    "km",
    "H2 g/100km",
    "H2 corr. g/100km",
    "ΔSOC pts",
    "violations",
    "bilan %",
}


# --------------------------------------------------------------------------------------
# Fixtures (module : une simulation WLTC ≈ 0,4 s, réutilisée par les tests rapides)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> IEMSConfig:
    return IEMSConfig()


@pytest.fixture(scope="module")
def res_iems_wltc(cfg: IEMSConfig) -> SimResult:
    return simulate(get_cycle("wltc"), IEMS(cfg), cfg)


@pytest.fixture(scope="module")
def metrics_iems_wltc(res_iems_wltc: SimResult) -> Metrics:
    return compute_metrics(res_iems_wltc)


@pytest.fixture(scope="module")
def campaign(cfg: IEMSConfig) -> dict[tuple[str, str], Metrics]:
    """Métriques iems / rules_baseline sur les 5 cycles (évaluée par les seuls tests slow)."""
    return {
        (cyc, strat): compute_metrics(run(cyc, strat, cfg))
        for cyc in CYCLES
        for strat in ("iems", "rules_baseline")
    }


@pytest.fixture(scope="module")
def calibration(cfg: IEMSConfig):
    return calibrate_ecms_const(get_cycle("wltc"), cfg)


def make_obs(
    soc: float,
    P_dem: float = 20e3,
    P_fc: float = 20e3,
    v: float = 10.0,
    k: int = 0,
    P_dem_pred: np.ndarray | None = None,
) -> EMSObservation:
    """Observation EMS construite à la main (demande prédite constante par défaut)."""
    pred = np.full(N_PRED, P_dem) if P_dem_pred is None else np.asarray(P_dem_pred, dtype=float)
    return EMSObservation(
        k=k,
        t=float(k),
        v=v,
        soc=soc,
        P_fc=P_fc,
        P_bat=0.0,
        P_dem=P_dem,
        P_dem_pred=pred,
        v_hist=np.full(k + 1, v),
    )


def with_soc_final(res: SimResult, soc_final: float) -> SimResult:
    """Copie du résultat dont seul le SOC final est modifié."""
    soc = res.soc.copy()
    soc[-1] = soc_final
    return dataclasses.replace(res, soc=soc)


# --------------------------------------------------------------------------------------
# (a) Critère ③ : charge-sustaining iEMS sur WLTC
# --------------------------------------------------------------------------------------
def test_iems_wltc_charge_sustaining_sous_2_pts(metrics_iems_wltc: Metrics) -> None:
    """Critère ③ : |ΔSOC| < 2 pts pour l'iEMS sur WLTC en configuration par défaut."""
    assert abs(metrics_iems_wltc.delta_soc_pts) < 2.0
    assert metrics_iems_wltc.charge_sustaining is True


def test_iems_wltc_zero_violation(metrics_iems_wltc: Metrics) -> None:
    """Critère ② sur WLTC : aucune violation de bornes pour l'iEMS."""
    assert metrics_iems_wltc.violations.total == 0


def test_iems_wltc_bilan_energie_boucle(metrics_iems_wltc: Metrics) -> None:
    """Critère ① : bilan d'énergie bouclé bien sous 1 % (intégrateur exact : ~1e-14 attendu)."""
    assert metrics_iems_wltc.energy.rel_error < 1e-6


# --------------------------------------------------------------------------------------
# (b) Critère ② : zéro violation iEMS sur les 5 cycles
# --------------------------------------------------------------------------------------
def test_catalogue_cycles_est_la_campagne_m1() -> None:
    """Le catalogue de cycles est exactement celui de la campagne M1."""
    assert set(list_cycles()) == set(CYCLES)


@pytest.mark.slow
@pytest.mark.parametrize("cycle", CYCLES)
def test_iems_zero_violation_sur_cycle(campaign: dict, cycle: str) -> None:
    """Critère ② : violations.total == 0 pour l'iEMS sur chaque cycle du catalogue."""
    assert campaign[(cycle, "iems")].violations.total == 0


# --------------------------------------------------------------------------------------
# (c) Critère ④ : iEMS ≤ baseline (conso corrigée du SOC) sur ≥ 4 cycles / 5
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_iems_conso_corrigee_inferieure_baseline_sur_4_cycles_sur_5(campaign: dict) -> None:
    """Critère ④ : h2_corr(iems) ≤ h2_corr(rules_baseline) sur au moins 4 des 5 cycles."""
    gagnes = [
        cyc
        for cyc in CYCLES
        if campaign[(cyc, "iems")].h2_corr_g_per_100km
        <= campaign[(cyc, "rules_baseline")].h2_corr_g_per_100km
    ]
    assert len(gagnes) >= 4, f"iEMS meilleur que la baseline sur {gagnes} seulement"


@pytest.mark.slow
def test_rules_baseline_zero_violation_sur_5_cycles(campaign: dict) -> None:
    """La baseline passe elle aussi par le bloc Candidates : zéro violation sur les 5 cycles."""
    assert all(campaign[(cyc, "rules_baseline")].violations.total == 0 for cyc in CYCLES)


# --------------------------------------------------------------------------------------
# (d) Calibration ECMS à λ constant sur WLTC
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_calibrate_ecms_const_wltc_delta_soc_sous_tolerance(calibration, cfg: IEMSConfig) -> None:
    """La dichotomie ramène |SOC(T) − SOC(0)| sous 0,005 avec λ dans lam_bounds."""
    lo, hi = 1.0, 4.0  # lam_bounds par défaut de calibrate_ecms_const
    assert abs(calibration.delta_soc) < 0.005
    assert lo <= calibration.lam <= hi
    assert calibration.result.strategy == "ecms_const"
    assert calibration.result.soc_final - cfg.soc0 == pytest.approx(
        calibration.delta_soc, abs=1e-12
    )
    assert 2 <= calibration.iterations <= 25


@pytest.mark.slow
def test_calibrate_ecms_const_wltc_zero_violation(calibration) -> None:
    """L'ECMS calibré (charge-sustaining) ne produit aucune violation sur WLTC."""
    assert compute_metrics(calibration.result).violations.total == 0


@pytest.mark.slow
def test_ecms_const_soc_final_croissant_en_lambda(cfg: IEMSConfig) -> None:
    """Monotonie : λ = 1,2 (batterie bon marché) finit plus bas en SOC que λ = 2,0."""
    cycle = get_cycle("wltc")
    soc_bas = simulate(cycle, ECMSConst(cfg, lam=1.2), cfg).soc_final
    soc_haut = simulate(cycle, ECMSConst(cfg, lam=2.0), cfg).soc_final
    assert soc_bas < soc_haut
    assert soc_haut - soc_bas > 0.05  # écart net (≈ 0,45 vs ≈ 0,80 attendus), pas un bruit


def test_ecms_const_lambda0_defaut_wltc_zero_violation(cfg: IEMSConfig) -> None:
    """Un ECMS au λ0 par défaut (= λ* calibré sur WLTC) reste dans les bornes (critère ②)."""
    m = compute_metrics(run("wltc", "ecms_const", cfg))
    assert m.violations.total == 0


# --------------------------------------------------------------------------------------
# (e) Registre et instanciation
# --------------------------------------------------------------------------------------
def test_registre_contient_les_trois_strategies() -> None:
    """Le registre STRATEGIES expose exactement rules_baseline, ecms_const et iems."""
    assert set(STRATEGIES) == set(STRATEGY_NAMES)


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_make_strategy_nom_du_registre_instancie_avec_name_correct(
    name: str, cfg: IEMSConfig
) -> None:
    """make_strategy renvoie une instance de la classe du registre dont .name est le nom demandé."""
    strat = make_strategy(name, cfg)
    assert strat.name == name
    assert isinstance(strat, STRATEGIES[name])


def test_make_strategy_nom_inconnu_leve_keyerror(cfg: IEMSConfig) -> None:
    """Un nom absent du registre lève KeyError avec la liste des stratégies disponibles."""
    with pytest.raises(KeyError, match="inconnue"):
        make_strategy("pmp_offline", cfg)


def test_ecms_const_lam_explicite_conserve(cfg: IEMSConfig) -> None:
    """ECMSConst garde le λ passé explicitement et prend λ0 de la config par défaut."""
    assert ECMSConst(cfg, lam=1.7).lam == 1.7
    assert ECMSConst(cfg).lam == cfg.ems.lambda0
    assert make_strategy("ecms_const", cfg, lam=1.7).lam == 1.7


def test_rules_baseline_reset_idempotent(cfg: IEMSConfig) -> None:
    """reset() ramène toujours l'état initial (filtre vide, mode 0), répété ou après un pas."""
    strat = RulesBaseline(cfg)
    initial = (strat._P_filt, strat._mode)
    assert initial == (None, 0)
    strat.reset()
    strat.reset()
    assert (strat._P_filt, strat._mode) == initial
    strat.step(make_obs(soc=cfg.ems.baseline_soc_low - 0.05))
    assert (strat._P_filt, strat._mode) != initial
    strat.reset()
    assert (strat._P_filt, strat._mode) == initial
    frais = RulesBaseline(cfg).step(make_obs(soc=cfg.soc0))
    assert strat.step(make_obs(soc=cfg.soc0)) == frais


# --------------------------------------------------------------------------------------
# (f) Baseline : hystérésis, et contrat de sortie commun à toutes les stratégies
# --------------------------------------------------------------------------------------
def test_rules_baseline_hysteresis_modes(cfg: IEMSConfig) -> None:
    """Recharge sous soc_low, délestage au-dessus de soc_high, retour à 0 en repassant soc_ref."""
    ems = cfg.ems
    strat = RulesBaseline(cfg)
    sequence = [
        (ems.baseline_soc_low - 0.05, 1),  # sous le seuil bas → recharge
        (0.5 * (ems.baseline_soc_low + ems.soc_ref), 1),  # remonté mais < soc_ref : reste
        (ems.soc_ref, 0),  # soc_ref atteint → normal
        (ems.baseline_soc_high + 0.03, -1),  # au-dessus du seuil haut → délestage
        (0.5 * (ems.soc_ref + ems.baseline_soc_high), -1),  # redescendu mais > soc_ref : reste
        (ems.soc_ref, 0),  # soc_ref atteint → normal
    ]
    for soc, mode_attendu in sequence:
        cmd = strat.step(make_obs(soc=soc))
        assert cmd.info["mode"] == mode_attendu, f"soc={soc:.3f}"


def test_rules_baseline_mode_decale_la_consigne_de_charge_W(cfg: IEMSConfig) -> None:
    """Au premier pas la consigne vaut clip(P_dem + mode·baseline_charge_W, P_lo, P_hi)."""
    ems = cfg.ems
    P_dem = 20e3
    cmds = {}
    for soc in (ems.baseline_soc_low - 0.05, ems.soc_ref, ems.baseline_soc_high + 0.03):
        cmd = RulesBaseline(cfg).step(make_obs(soc=soc, P_dem=P_dem, P_fc=P_dem))
        cible = P_dem + cmd.info["mode"] * ems.baseline_charge_W
        attendu = min(max(cible, cmd.info["P_lo"]), cmd.info["P_hi"])
        assert cmd.FC_Power_cmd == pytest.approx(attendu, abs=TOL_W)
        cmds[cmd.info["mode"]] = cmd.FC_Power_cmd
    assert cmds[1] > cmds[0] > cmds[-1]


OBSERVATIONS = {
    "roulage_soc_bas": make_obs(soc=0.45),
    "roulage_soc_ref": make_obs(soc=0.60),
    "roulage_soc_haut": make_obs(soc=0.75),
    "demande_croissante": make_obs(
        soc=0.60, P_dem=10e3, P_dem_pred=np.linspace(10e3, 30e3, N_PRED)
    ),
    "freinage_soc_haut": make_obs(soc=0.78, P_dem=-15e3, P_fc=5e3, v=15.0),
    "arret_pac_ralenti": make_obs(soc=0.60, P_dem=500.0, P_fc=1e3, v=0.0),
}


@pytest.mark.parametrize("name", STRATEGY_NAMES)
@pytest.mark.parametrize("obs_id", list(OBSERVATIONS))
def test_strategie_consigne_dans_bornes_et_batterie_boucle_le_bus(
    name: str, obs_id: str, cfg: IEMSConfig
) -> None:
    """FC_Power_cmd ∈ [P_lo, P_hi] de info et BAT_Power_cmd = demande moyenne − FC_Power_cmd."""
    obs = OBSERVATIONS[obs_id]
    cmd = make_strategy(name, cfg).step(obs)
    P_lo, P_hi = cmd.info["P_lo"], cmd.info["P_hi"]
    assert P_lo <= P_hi
    assert P_lo - TOL_W <= cmd.FC_Power_cmd <= P_hi + TOL_W
    assert cfg.fuel_cell.P_min_W <= cmd.FC_Power_cmd <= cfg.fuel_cell.P_max_W
    # Demande moyenne sur les `substeps` pas modèle que la consigne va couvrir (cf. candidates).
    demande_moyenne = float(np.mean(obs.P_dem_pred[: cfg.ems.substeps]))
    assert cmd.BAT_Power_cmd == pytest.approx(demande_moyenne - cmd.FC_Power_cmd, abs=TOL_W)
    assert np.isfinite(cmd.FC_Power_cmd) and np.isfinite(cmd.BAT_Power_cmd)


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategie_respecte_la_rampe_pac_sur_un_pas(name: str, cfg: IEMSConfig) -> None:
    """La consigne ne s'écarte jamais de plus de r_fc·Ts_ems de la puissance PàC courante."""
    r_pas = cfg.fuel_cell.ramp_W_per_s * cfg.ems.Ts_ems
    for P_fc in (1e3, 20e3, 60e3, 95e3):
        for P_dem in (-20e3, 500.0, 30e3, 90e3):
            cmd = make_strategy(name, cfg).step(make_obs(soc=0.6, P_dem=P_dem, P_fc=P_fc))
            assert abs(cmd.FC_Power_cmd - P_fc) <= r_pas + TOL_W


# --------------------------------------------------------------------------------------
# (g) Métriques
# --------------------------------------------------------------------------------------
def test_violations_total_est_la_somme_des_champs() -> None:
    """Violations.total additionne les six compteurs ; zéro par défaut."""
    assert Violations().total == 0
    assert Violations(1, 2, 3, 4, 5, 6).total == 21
    assert Violations(bat_charge=7).total == 7


def test_h2_corr_superieur_a_h2_si_soc_final_plus_bas(res_iems_wltc: SimResult) -> None:
    """Batterie déchargée en fin de cycle : la conso corrigée dépasse la conso brute."""
    m = compute_metrics(with_soc_final(res_iems_wltc, res_iems_wltc.cfg.soc0 - 0.05))
    assert m.delta_soc_pts == pytest.approx(-5.0, abs=1e-9)
    assert m.h2_corr_g > m.h2_g
    assert m.h2_corr_g_per_100km > m.h2_g_per_100km


def test_h2_corr_inferieur_a_h2_si_soc_final_plus_haut(res_iems_wltc: SimResult) -> None:
    """Batterie plus chargée en fin de cycle : la conso corrigée est sous la conso brute."""
    m = compute_metrics(with_soc_final(res_iems_wltc, res_iems_wltc.cfg.soc0 + 0.05))
    assert m.delta_soc_pts == pytest.approx(5.0, abs=1e-9)
    assert m.h2_corr_g < m.h2_g
    assert m.h2_corr_g_per_100km < m.h2_g_per_100km


def test_h2_corr_egal_a_h2_si_soc_final_egal_soc0(res_iems_wltc: SimResult) -> None:
    """ΔSOC nul : aucune correction, et la conso brute ne dépend pas du SOC final."""
    m0 = compute_metrics(with_soc_final(res_iems_wltc, res_iems_wltc.cfg.soc0))
    m_bas = compute_metrics(with_soc_final(res_iems_wltc, res_iems_wltc.cfg.soc0 - 0.05))
    assert m0.h2_corr_g == pytest.approx(m0.h2_g, rel=1e-12)
    assert m_bas.h2_g == pytest.approx(m0.h2_g, rel=1e-12)


def test_h2_corr_decharge_plus_chere_que_la_recharge_rapportee(res_iems_wltc: SimResult) -> None:
    """Asymétrie : combler −ΔE coûte plus d'H2 que +ΔE n'en économise (η_ch·η_dis < 1)."""
    soc0 = res_iems_wltc.cfg.soc0
    m0 = compute_metrics(with_soc_final(res_iems_wltc, soc0))
    m_bas = compute_metrics(with_soc_final(res_iems_wltc, soc0 - 0.05))
    m_haut = compute_metrics(with_soc_final(res_iems_wltc, soc0 + 0.05))
    assert m_bas.h2_corr_g - m0.h2_g > m0.h2_g - m_haut.h2_corr_g > 0


def test_metrics_row_a_les_cles_attendues(metrics_iems_wltc: Metrics) -> None:
    """row() fournit les huit colonnes du tableau comparatif avec les valeurs du run."""
    row = metrics_iems_wltc.row()
    assert set(row) == ROW_KEYS
    assert row["cycle"] == "wltc"
    assert row["stratégie"] == "iems"
    assert row["violations"] == metrics_iems_wltc.violations.total
    assert row["km"] == pytest.approx(metrics_iems_wltc.distance_km, abs=0.005)
    assert row["ΔSOC pts"] == pytest.approx(metrics_iems_wltc.delta_soc_pts, abs=0.005)


def test_metrics_distance_km_wltc(metrics_iems_wltc: Metrics) -> None:
    """La distance du WLTC vaut 23,27 km à 1 % près."""
    assert metrics_iems_wltc.distance_km == pytest.approx(WLTC_KM, rel=0.01)


def test_metrics_champs_coherents_avec_le_run(
    metrics_iems_wltc: Metrics, res_iems_wltc: SimResult, cfg: IEMSConfig
) -> None:
    """Les champs scalaires dérivent bien des journaux : SOC, conso par 100 km, rendement."""
    m = metrics_iems_wltc
    assert m.soc0 == cfg.soc0
    assert m.soc_final == res_iems_wltc.soc_final
    assert m.delta_soc_pts == pytest.approx(100.0 * (m.soc_final - m.soc0), abs=1e-9)
    assert cfg.battery.soc_min <= m.soc_min <= m.soc0 <= m.soc_max <= cfg.battery.soc_max
    assert m.h2_g_per_100km == pytest.approx(100.0 * m.h2_g / m.distance_km, rel=1e-12)
    assert m.h2_corr_g_per_100km == pytest.approx(100.0 * m.h2_corr_g / m.distance_km, rel=1e-12)
    assert 0.3 < m.eta_fc_mean < 0.6  # rendement moyen dans la cloche de la carte η
    assert m.energy.E_h2_chem == pytest.approx(1e-3 * m.h2_g * cfg.fuel_cell.lhv_J_per_kg)
    assert m.energy.E_fc == pytest.approx(m.eta_fc_mean * m.energy.E_h2_chem)
    assert m.runtime_s >= 0.0
