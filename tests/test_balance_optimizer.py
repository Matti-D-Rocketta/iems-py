"""Tests de ``balance.py`` (usage, SOC cible, co-état) et ``optimizer.py`` (Hamiltonien, argmin)."""

from __future__ import annotations

import numpy as np
import pytest

from iems.balance import N_GRID, Balance, BalanceOutput, build_usage_table, kinetic_reserve
from iems.candidates import N_CANDIDATES, CandidateSet, build_candidates
from iems.config import IEMSConfig
from iems.models.battery import Battery
from iems.models.fuel_cell import FuelCell
from iems.optimizer import apply_penalties, hamiltonian, homogenization_factor, optimize
from iems.patterns import DrivingPattern

# Tolérances explicites
ABS_TIGHT = 1e-12  # identités algébriques en double précision
ABS_USAGE = 0.01  # usage nominal (demandé par la spec de test)
REL_FORMULA = 1e-9  # recalcul d'une formule fermée
GRID = np.linspace(0.0, 1.0, N_GRID)
PATTERNS = list(DrivingPattern)
PATTERN_IDS = [p.name for p in PATTERNS]


# --------------------------------------------------------------------------------------------
# Fixtures locales (pas de conftest)
# --------------------------------------------------------------------------------------------
@pytest.fixture
def cfg() -> IEMSConfig:
    return IEMSConfig()


@pytest.fixture
def battery(cfg: IEMSConfig) -> Battery:
    return Battery(cfg.battery)


@pytest.fixture
def fc(cfg: IEMSConfig) -> FuelCell:
    return FuelCell(cfg.fuel_cell)


@pytest.fixture
def balance(cfg: IEMSConfig, battery: Battery) -> Balance:
    return Balance(cfg, battery)


def _make_cands(P_fc, P_dem: float, valid, excess) -> CandidateSet:
    """Construit un CandidateSet à la main (bornes/plancher/plafond sans importance ici)."""
    P_fc = np.asarray(P_fc, dtype=float)
    return CandidateSet(
        P_fc=P_fc,
        P_bat=P_dem - P_fc,
        valid=np.asarray(valid, dtype=bool),
        excess=np.asarray(excess, dtype=float),
        P_lo=float(P_fc[0]),
        P_hi=float(P_fc[-1]),
        floor=float(P_fc[0]),
        ceiling=float(P_fc[-1]),
        degenerate=False,
        conflict=False,
    )


# --------------------------------------------------------------------------------------------
# (a) build_usage_table
# --------------------------------------------------------------------------------------------
def test_usage_table_shape_is_5x9x9(cfg: IEMSConfig) -> None:
    """La table d'usage a une entrée par schéma et une grille 9 × 9 (v_norm, soc_norm)."""
    table = build_usage_table(cfg)
    assert table.shape == (len(DrivingPattern), N_GRID, N_GRID) == (5, 9, 9)


def test_usage_table_values_within_unit_interval(cfg: IEMSConfig) -> None:
    """Tous les usages sont dans [0, 1] après saturation."""
    table = build_usage_table(cfg)
    assert np.all(np.isfinite(table))
    assert table.min() >= 0.0
    assert table.max() <= 1.0


@pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_IDS)
def test_usage_table_non_decreasing_along_soc(cfg: IEMSConfig, pattern: DrivingPattern) -> None:
    """À vitesse fixée, l'usage croît (au sens large) avec le SOC pour chaque schéma."""
    table = build_usage_table(cfg)
    assert np.all(np.diff(table[pattern], axis=1) >= -ABS_TIGHT)
    # strictement croissant au moins entre deux nœuds non saturés (autour de soc_ref)
    assert table[pattern, 0, 5] > table[pattern, 0, 4]


@pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_IDS)
def test_usage_table_non_decreasing_along_speed_kinetic_reserve(
    cfg: IEMSConfig, pattern: DrivingPattern
) -> None:
    """À SOC fixé, l'usage croît avec la vitesse : la référence SOC baisse (réserve cinétique)."""
    # Note : la référence SOC_ref(v) décroît en v et u croît avec (SOC − SOC_ref(v)), donc
    # u est croissant en v à SOC fixé (laisser de la place au freinage récupératif).
    table = build_usage_table(cfg)
    assert np.all(np.diff(table[pattern], axis=0) >= -ABS_TIGHT)
    assert table[pattern, -1, 4] > table[pattern, 0, 4]


@pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_IDS)
def test_usage_table_matches_affine_soc_anchoring_formula(
    cfg: IEMSConfig, balance: Balance, pattern: DrivingPattern
) -> None:
    """Chaque nœud vaut clip(u_nominal + gain · (SOC − SOC_ref(v)) / largeur, 0, 1)."""
    ems, bat = cfg.ems, cfg.battery
    width = bat.soc_max - bat.soc_min
    table = build_usage_table(cfg)
    for i, vn in enumerate(GRID):
        soc_ref_v = balance.soc_ref(vn * ems.v_norm_max)
        soc = bat.soc_min + GRID * width
        affine = ems.usage_table[pattern] + ems.usage_soc_gain * (soc - soc_ref_v) / width
        expected = np.clip(affine, 0.0, 1.0)
        np.testing.assert_allclose(table[pattern, i], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_IDS)
def test_usage_table_node_at_rest_and_soc_ref_equals_nominal(
    cfg: IEMSConfig, pattern: DrivingPattern
) -> None:
    """Sur un nœud (v = 0, SOC = soc_ref), la table vaut l'usage nominal du schéma à ± 0,01."""
    # soc_ref = 0,55 tombe exactement sur le nœud soc_norm = 0,5 (indice 4) de [0,30 ; 0,80]
    cfg_node = cfg.with_updates(ems={"soc_ref": 0.55})
    bat = cfg_node.battery
    j = 4
    assert bat.soc_min + GRID[j] * (bat.soc_max - bat.soc_min) == pytest.approx(0.55, abs=1e-12)
    table = build_usage_table(cfg_node)
    assert table[pattern, 0, j] == pytest.approx(cfg_node.ems.usage_table[pattern], abs=ABS_USAGE)


@pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_IDS)
def test_usage_at_rest_and_soc_ref_equals_nominal(
    cfg: IEMSConfig, balance: Balance, pattern: DrivingPattern
) -> None:
    """Balance.usage(v = 0, SOC = soc_ref) interpolé vaut l'usage nominal du schéma à ± 0,01."""
    u = balance.usage(0.0, cfg.ems.soc_ref, pattern)
    assert u == pytest.approx(cfg.ems.usage_table[pattern], abs=ABS_USAGE)


def test_usage_pattern_order_sportif_above_urbain_above_autoroute(
    cfg: IEMSConfig, balance: Balance
) -> None:
    """À v = 0 et SOC = soc_ref : usage sportif > urbain > autoroute."""
    soc = cfg.ems.soc_ref
    u_sport = balance.usage(0.0, soc, DrivingPattern.SPORTIF)
    u_urb = balance.usage(0.0, soc, DrivingPattern.URBAIN)
    u_auto = balance.usage(0.0, soc, DrivingPattern.AUTOROUTE)
    assert u_sport > u_urb > u_auto
    assert u_sport == pytest.approx(0.85, abs=ABS_USAGE)
    assert u_auto == pytest.approx(0.30, abs=ABS_USAGE)


def test_usage_saturates_outside_grid_inputs(cfg: IEMSConfig, balance: Balance) -> None:
    """Entrées hors grille (v > v_norm_max, SOC hors fenêtre) : usage dans [0, 1], sans NaN."""
    for v, soc in [(2 * cfg.ems.v_norm_max, 0.9), (-1.0, 0.1), (0.0, 1.5)]:
        for p in DrivingPattern:
            u = balance.usage(v, soc, p)
            assert np.isfinite(u)
            assert 0.0 <= u <= 1.0


# --------------------------------------------------------------------------------------------
# (b) réserve cinétique et SOC de référence
# --------------------------------------------------------------------------------------------
def test_kinetic_reserve_zero_at_rest(cfg: IEMSConfig) -> None:
    """À l'arrêt, aucune réserve cinétique n'est nécessaire."""
    assert float(kinetic_reserve(0.0, cfg)) == pytest.approx(0.0, abs=ABS_TIGHT)


def test_kinetic_reserve_at_v_norm_max_matches_formula(cfg: IEMSConfig) -> None:
    """kinetic_reserve(1) = kinetic_recovery · ½ · m · v_norm_max² / E_cap_J."""
    ems, veh, bat = cfg.ems, cfg.vehicle, cfg.battery
    expected = ems.kinetic_recovery * 0.5 * veh.mass_kg * ems.v_norm_max**2 / bat.E_cap_J
    assert float(kinetic_reserve(1.0, cfg)) == pytest.approx(expected, rel=REL_FORMULA)
    # ordre de grandeur attendu avec les valeurs par défaut (≈ 15 points de SOC)
    assert expected == pytest.approx(0.1527, abs=1e-3)


def test_kinetic_reserve_vectorized_is_quadratic(cfg: IEMSConfig) -> None:
    """Sur un tableau, la réserve est quadratique en v_norm (facteur 4 entre 0,5 et 1)."""
    r = kinetic_reserve(np.array([0.0, 0.5, 1.0]), cfg)
    assert r.shape == (3,)
    assert r[2] == pytest.approx(4.0 * r[1], rel=REL_FORMULA)


def test_soc_ref_at_rest_equals_config(cfg: IEMSConfig, balance: Balance) -> None:
    """soc_ref(0) vaut le soc_ref de la config."""
    assert balance.soc_ref(0.0) == pytest.approx(cfg.ems.soc_ref, abs=ABS_TIGHT)


def test_soc_ref_strictly_decreasing_in_speed(balance: Balance) -> None:
    """La référence de SOC décroît strictement avec la vitesse."""
    v = np.linspace(0.0, 40.0, 41)
    refs = np.array([balance.soc_ref(float(x)) for x in v])
    assert np.all(np.diff(refs) < 0.0)


def test_soc_ref_at_v_norm_max_equals_soc_ref_minus_reserve(
    cfg: IEMSConfig, balance: Balance
) -> None:
    """soc_ref(v_norm_max) = soc_ref − kinetic_reserve(1)."""
    expected = cfg.ems.soc_ref - float(kinetic_reserve(1.0, cfg))
    assert balance.soc_ref(cfg.ems.v_norm_max) == pytest.approx(expected, abs=ABS_TIGHT)


# --------------------------------------------------------------------------------------------
# (c) SOC cible
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("usage", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("soc", [0.31, 0.45, 0.60, 0.79])
def test_soc_target_within_reachable_range(balance: Balance, soc: float, usage: float) -> None:
    """SOC_target reste dans la plage atteignable [lo, hi], quel que soit l'usage."""
    lo, hi, target = balance.soc_target(soc, usage)
    assert lo <= hi
    assert lo - ABS_TIGHT <= target <= hi + ABS_TIGHT


def test_soc_target_range_matches_battery_reachable_soc(
    cfg: IEMSConfig, balance: Balance, battery: Battery
) -> None:
    """[lo, hi] est la plage atteignable de la batterie sur l'horizon Hp."""
    soc = 0.60
    lo, hi, _ = balance.soc_target(soc, 0.5)
    lo_b, hi_b = battery.reachable_soc(soc, cfg.ems.Hp)
    assert lo == pytest.approx(lo_b, abs=ABS_TIGHT)
    assert hi == pytest.approx(hi_b, abs=ABS_TIGHT)
    assert lo < soc < hi


def test_soc_target_usage_zero_gives_hi(balance: Balance) -> None:
    """Usage 0 (recharger) : la cible est le haut de la plage atteignable."""
    lo, hi, target = balance.soc_target(0.60, 0.0)
    assert target == pytest.approx(hi, abs=ABS_TIGHT)
    assert target > lo


def test_soc_target_usage_one_gives_lo(balance: Balance) -> None:
    """Usage 1 (solliciter la batterie) : la cible est le bas de la plage atteignable."""
    lo, hi, target = balance.soc_target(0.60, 1.0)
    assert target == pytest.approx(lo, abs=ABS_TIGHT)
    assert target < hi


def test_soc_target_half_usage_gives_midpoint(balance: Balance) -> None:
    """Usage 0,5 : la cible est le milieu de la plage."""
    lo, hi, target = balance.soc_target(0.60, 0.5)
    assert target == pytest.approx(0.5 * (lo + hi), abs=ABS_TIGHT)


# --------------------------------------------------------------------------------------------
# (d) co-état
# --------------------------------------------------------------------------------------------
def test_costate_unsaturated_matches_proportional_law(cfg: IEMSConfig, balance: Balance) -> None:
    """λ = λ0 + Kp · (SOC_target − SOC) quand on est dans [λ_min, λ_max]."""
    ems = cfg.ems
    soc, delta = 0.60, 0.02
    expected = ems.lambda0 + ems.Kp * delta
    assert ems.lambda_min < expected < ems.lambda_max
    assert balance.costate(soc, soc + delta) == pytest.approx(expected, abs=ABS_TIGHT)
    assert balance.costate(soc, soc - delta) == pytest.approx(
        ems.lambda0 - ems.Kp * delta, abs=ABS_TIGHT
    )


def test_costate_at_target_equals_lambda0(cfg: IEMSConfig, balance: Balance) -> None:
    """SOC sur sa cible : λ = λ0."""
    assert balance.costate(0.55, 0.55) == pytest.approx(cfg.ems.lambda0, abs=ABS_TIGHT)


def test_costate_saturates_at_lambda_max(cfg: IEMSConfig, balance: Balance) -> None:
    """Cible très au-dessus du SOC : λ est saturé à λ_max."""
    ems = cfg.ems
    delta = (ems.lambda_max - ems.lambda0) / ems.Kp + 0.5  # bien au-delà du seuil de saturation
    assert balance.costate(0.3, 0.3 + delta) == pytest.approx(ems.lambda_max, abs=ABS_TIGHT)


def test_costate_saturates_at_lambda_min(cfg: IEMSConfig, balance: Balance) -> None:
    """Cible très en dessous du SOC : λ est saturé à λ_min."""
    ems = cfg.ems
    delta = (ems.lambda0 - ems.lambda_min) / ems.Kp + 0.5
    assert balance.costate(0.8, 0.8 - delta) == pytest.approx(ems.lambda_min, abs=ABS_TIGHT)


def test_costate_uses_config_gain_and_bounds(cfg: IEMSConfig, battery: Battery) -> None:
    """Kp, λ_min et λ_max de la config sont bien ceux appliqués."""
    cfg2 = cfg.with_updates(ems={"Kp": 20.0, "lambda_min": 1.0, "lambda_max": 3.0, "lambda0": 2.0})
    bal = Balance(cfg2, battery)
    assert bal.costate(0.6, 0.62) == pytest.approx(2.0 + 20.0 * 0.02, abs=ABS_TIGHT)
    assert bal.costate(0.6, 0.70) == pytest.approx(3.0, abs=ABS_TIGHT)
    assert bal.costate(0.6, 0.50) == pytest.approx(1.0, abs=ABS_TIGHT)


def test_step_output_consistent_with_submethods(cfg: IEMSConfig, balance: Balance) -> None:
    """Balance.step assemble usage, soc_ref, plage, cible et λ de façon cohérente."""
    v, soc, pattern = 15.0, 0.58, DrivingPattern.URBAIN
    out = balance.step(v, soc, pattern)
    assert isinstance(out, BalanceOutput)
    assert out.usage == pytest.approx(balance.usage(v, soc, pattern), abs=ABS_TIGHT)
    assert out.soc_ref_v == pytest.approx(balance.soc_ref(v), abs=ABS_TIGHT)
    lo, hi, target = balance.soc_target(soc, out.usage)
    assert (out.soc_lo, out.soc_hi) == pytest.approx((lo, hi), abs=ABS_TIGHT)
    assert out.soc_target == pytest.approx(target, abs=ABS_TIGHT)
    assert out.lam == pytest.approx(balance.costate(soc, target), abs=ABS_TIGHT)
    assert cfg.ems.lambda_min <= out.lam <= cfg.ems.lambda_max
    assert 0.0 <= out.usage <= 1.0


# --------------------------------------------------------------------------------------------
# (e) optimiseur
# --------------------------------------------------------------------------------------------
def test_homogenization_factor_equals_ecap_over_lhv(
    cfg: IEMSConfig, battery: Battery, fc: FuelCell
) -> None:
    """k_h = E_cap_J / PCI (kg)."""
    k_h = homogenization_factor(battery, fc)
    assert k_h == pytest.approx(cfg.battery.E_cap_J / cfg.fuel_cell.lhv_J_per_kg, rel=1e-12)
    assert k_h == pytest.approx(1600.0 * 3600.0 / 120e6, rel=1e-12)  # 0,048 kg par défaut


def test_hamiltonian_zero_lambda_equals_h2_rate(battery: Battery, fc: FuelCell) -> None:
    """À λ = 0, le Hamiltonien se réduit au débit d'H2."""
    P_fc = np.linspace(fc.p.P_min_W, fc.p.P_max_W, 9)
    H = hamiltonian(P_fc, 40e3, 0.0, fc, battery)
    np.testing.assert_allclose(H, fc.h2_rate(P_fc), rtol=0, atol=1e-15)


def test_hamiltonian_matches_reference_equation(battery: Battery, fc: FuelCell) -> None:
    """H · PCI = a + b·P + c·P² + λ · P_int(P_dem − P), décharge comme charge."""
    P_dem, lam = 30e3, 1.8
    P_fc = np.array([5e3, 20e3, 30e3, 45e3, 90e3])
    H = hamiltonian(P_fc, P_dem, lam, fc, battery)
    P_bat = P_dem - P_fc
    expected = (
        fc.a + fc.b * P_fc + fc.c * P_fc**2 + lam * battery.internal_power(P_bat)
    ) / fc.p.lhv_J_per_kg
    np.testing.assert_allclose(H, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("lam", "P_dem"),
    [(2.0, 60e3), (1.55, 30e3), (2.5, 95e3)],
    ids=["lam2_Pdem60k", "lam1.55_Pdem30k", "lam2.5_Pdem95k"],
)
def test_hamiltonian_argmin_in_discharge_matches_analytic_optimum(
    cfg: IEMSConfig, battery: Battery, fc: FuelCell, lam: float, P_dem: float
) -> None:
    """En décharge, l'argmin de H sur grille fine est en P* = (λ/η_dis − b)/(2c) à ± 1 % P_max."""
    P_star = (lam / cfg.battery.eta_dis - fc.b) / (2.0 * fc.c)
    # P* intérieur : au-dessus du ralenti, sous la demande (décharge) et sous P_max
    assert fc.p.P_min_W < P_star < min(P_dem, fc.p.P_max_W)
    grid = np.linspace(fc.p.P_min_W, fc.p.P_max_W, 9401)  # pas 10 W
    H = hamiltonian(grid, P_dem, lam, fc, battery)
    P_argmin = float(grid[np.argmin(H)])
    assert abs(P_argmin - P_star) <= 0.01 * fc.p.P_max_W
    assert P_argmin < P_dem  # l'optimum est bien en décharge


def test_hamiltonian_increases_with_lambda_in_discharge(battery: Battery, fc: FuelCell) -> None:
    """P_fc < P_dem (batterie en décharge) : H croît strictement avec λ."""
    P_dem = 40e3
    P_fc = np.array([1e3, 10e3, 25e3, 39e3])
    H1 = hamiltonian(P_fc, P_dem, 1.0, fc, battery)
    H2 = hamiltonian(P_fc, P_dem, 2.0, fc, battery)
    assert np.all(H2 > H1)


def test_hamiltonian_decreases_with_lambda_in_charge(battery: Battery, fc: FuelCell) -> None:
    """P_fc > P_dem (batterie en charge) : H décroît strictement avec λ."""
    P_dem = 20e3
    P_fc = np.array([21e3, 40e3, 70e3, 95e3])
    H1 = hamiltonian(P_fc, P_dem, 1.0, fc, battery)
    H2 = hamiltonian(P_fc, P_dem, 2.0, fc, battery)
    assert np.all(H2 < H1)


def test_hamiltonian_independent_of_lambda_at_zero_battery_power(
    battery: Battery, fc: FuelCell
) -> None:
    """P_fc = P_dem : la batterie ne bouge pas, H ne dépend pas de λ."""
    P_dem = 30e3
    H1 = hamiltonian(np.array([P_dem]), P_dem, 0.5, fc, battery)
    H2 = hamiltonian(np.array([P_dem]), P_dem, 5.0, fc, battery)
    assert H1[0] == pytest.approx(H2[0], abs=1e-18)


def test_apply_penalties_is_passthrough() -> None:
    """v1 : apply_penalties renvoie le même objet, avec ou sans contexte."""
    H = np.array([1.0, 2.0, 3.0])
    assert apply_penalties(H) is H
    out = apply_penalties(H, {"ramp": 1.0})
    assert out is H
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_optimize_restricts_argmin_to_valid_candidates(battery: Battery, fc: FuelCell) -> None:
    """j* est l'argmin de H parmi les seuls candidats valides, pas l'argmin global."""
    P_dem, lam = 40e3, 2.0
    P_fc = np.linspace(fc.p.P_min_W, fc.p.P_max_W, N_CANDIDATES)
    H = hamiltonian(P_fc, P_dem, lam, fc, battery)
    j_global = int(np.argmin(H))
    valid = np.ones(N_CANDIDATES, dtype=bool)
    valid[j_global] = False  # on invalide l'argmin global et son voisin direct
    valid[min(j_global + 1, N_CANDIDATES - 1)] = False
    excess = np.where(valid, 0.0, 500.0)
    cands = _make_cands(P_fc, P_dem, valid, excess)

    out = optimize(cands, P_dem, lam, fc, battery)

    expected = int(np.argmin(np.where(valid, H, np.inf)))
    assert expected != j_global
    assert out.j_star == expected
    assert valid[out.j_star]
    assert out.FC_Power_cmd == pytest.approx(P_fc[expected], abs=ABS_TIGHT)
    np.testing.assert_allclose(out.H, H, rtol=0, atol=1e-18)


def test_optimize_falls_back_to_minimal_excess_when_none_valid(
    battery: Battery, fc: FuelCell
) -> None:
    """Aucun candidat valide : j* est l'argmin de H parmi ceux de dépassement minimal."""
    P_dem, lam = 40e3, 2.0
    P_fc = np.linspace(fc.p.P_min_W, fc.p.P_max_W, N_CANDIDATES)
    H = hamiltonian(P_fc, P_dem, lam, fc, battery)
    j_global = int(np.argmin(H))
    assert j_global >= 3  # avec ces réglages l'argmin global est hors du trio de tête
    excess = np.full(N_CANDIDATES, 5000.0)
    excess[:3] = 1000.0  # dépassement minimal partagé par les candidats 0, 1, 2
    cands = _make_cands(P_fc, P_dem, np.zeros(N_CANDIDATES, dtype=bool), excess)

    out = optimize(cands, P_dem, lam, fc, battery)

    expected = int(np.argmin(np.where(excess <= excess.min(), H, np.inf)))
    assert out.j_star == expected
    assert out.j_star in (0, 1, 2)
    assert out.j_star != j_global
    assert excess[out.j_star] == pytest.approx(excess.min(), abs=ABS_TIGHT)


def test_optimize_all_valid_picks_global_argmin(battery: Battery, fc: FuelCell) -> None:
    """Tous valides : j* est l'argmin global de H."""
    P_dem, lam = 40e3, 2.0
    P_fc = np.linspace(fc.p.P_min_W, fc.p.P_max_W, N_CANDIDATES)
    cands = _make_cands(P_fc, P_dem, np.ones(N_CANDIDATES, dtype=bool), np.zeros(N_CANDIDATES))
    out = optimize(cands, P_dem, lam, fc, battery)
    assert out.j_star == int(np.argmin(hamiltonian(P_fc, P_dem, lam, fc, battery)))


@pytest.mark.parametrize("P_dem", [-10e3, 0.0, 12.5e3, 40e3])
def test_optimize_power_split_sums_to_demand(battery: Battery, fc: FuelCell, P_dem: float) -> None:
    """FC_Power_cmd + BAT_Power_cmd = P_dem et la consigne PàC est le candidat j*."""
    P_fc = np.linspace(fc.p.P_min_W, fc.p.P_max_W, N_CANDIDATES)
    cands = _make_cands(P_fc, P_dem, np.ones(N_CANDIDATES, dtype=bool), np.zeros(N_CANDIDATES))
    out = optimize(cands, P_dem, 1.55, fc, battery)
    assert out.FC_Power_cmd + out.BAT_Power_cmd == pytest.approx(P_dem, abs=1e-9)
    assert out.FC_Power_cmd == pytest.approx(P_fc[out.j_star], abs=ABS_TIGHT)
    assert out.H.shape == (N_CANDIDATES,)


def test_optimize_on_real_candidate_set_is_consistent(
    cfg: IEMSConfig, battery: Battery, fc: FuelCell
) -> None:
    """Sur un CandidateSet issu de build_candidates : répartition exacte, candidat valide choisi."""
    P_dem = 30e3
    n_pred = round(cfg.ems.Hp / cfg.ems.Ts_model) + 1
    cands = build_candidates(20e3, np.full(n_pred, P_dem), 0.60, 0.0, cfg, fc, battery)
    assert cands.valid.any()
    out = optimize(cands, P_dem, cfg.ems.lambda0, fc, battery)
    assert out.FC_Power_cmd + out.BAT_Power_cmd == pytest.approx(P_dem, abs=1e-9)
    assert out.FC_Power_cmd == pytest.approx(cands.P_fc[out.j_star], abs=ABS_TIGHT)
    assert cands.valid[out.j_star]
    assert cands.P_lo <= out.FC_Power_cmd <= cands.P_hi
