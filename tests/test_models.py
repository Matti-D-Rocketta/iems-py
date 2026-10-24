"""Tests des modèles basiques : véhicule, batterie, pile à combustible et cycles.

Référence : PASSATION_iEMS_Python.md § 5 (équations) et § 7 (monotonie des limites batterie).
Tous les tests sont indépendants de l'ordre d'exécution et tournent en moins de 10 s.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from iems.config import BatteryParams, FuelCellParams, VehicleParams
from iems.models.battery import Battery
from iems.models.cycles import (
    SEGMENT_SPECS,
    Cycle,
    get_cycle,
    list_cycles,
    load_wltc,
    synthetic_cycle,
)
from iems.models.fuel_cell import FuelCell
from iems.models.vehicle import electrical_demand, power_demand, wheel_power
from iems.patterns import DrivingPattern

# Tolérances explicites (SI).
REL_EXACT = 1e-12  # égalité « exacte » en flottant (formules recalculées)
ABS_W = 1e-6  # puissance (W)
ABS_SOC = 1e-12  # SOC (fraction)
MONO_TOL_W = 1e-9  # tolérance de monotonie sur les limites BMS (W)

CATALOG = ("wltc", "synth_mixed", "synth_urban", "synth_highway", "synth_sport")
SYNTH_CATALOG = tuple(n for n in CATALOG if n != "wltc")


# ---------------------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------------------
@pytest.fixture
def vehicle() -> VehicleParams:
    return VehicleParams()


@pytest.fixture
def battery() -> Battery:
    return Battery(BatteryParams())


@pytest.fixture
def fuel_cell() -> FuelCell:
    return FuelCell(FuelCellParams())


@pytest.fixture
def wltc() -> Cycle:
    return load_wltc()


@pytest.fixture
def three_segments() -> list[tuple[DrivingPattern, int]]:
    """Trois segments de schémas distincts, durées distinctes pour repérer les frontières."""
    return [
        (DrivingPattern.TRAFIC, 60),
        (DrivingPattern.AUTOROUTE, 150),
        (DrivingPattern.SPORTIF, 90),
    ]


def _wheel_formula(v: float, a: float, p: VehicleParams) -> float:
    """P_roue = v · (m·a + ½·ρ·Cd·Af·v² + Crr·m·g) (route plate, v > 0)."""
    return v * (
        p.mass_kg * a + 0.5 * p.rho_air * p.Cd * p.frontal_area_m2 * v**2 + p.Crr * p.mass_kg * p.g
    )


def _linear_ramp(v0: float, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Grille 1 Hz de 3 points à accélération constante : np.gradient y rend ``a`` exactement."""
    t = np.array([0.0, 1.0, 2.0])
    return t, np.array([v0 - a, v0, v0 + a])


# ---------------------------------------------------------------------------------------
# (a) Véhicule
# ---------------------------------------------------------------------------------------
def test_power_demand_traction_formule_recalculee(vehicle: VehicleParams) -> None:
    """P_dem à (20 m/s, +0,5 m/s²) vaut P_roue / η_trans + P_aux, soit 27 151,35 W par défaut."""
    v0, a = 20.0, 0.5
    t, v = _linear_ramp(v0, a)
    pd = power_demand(t, v, vehicle)
    assert pd.a[1] == pytest.approx(a, rel=REL_EXACT)
    p_wheel = _wheel_formula(v0, a, vehicle)
    assert pd.P_wheel[1] == pytest.approx(p_wheel, rel=REL_EXACT)
    expected = p_wheel / vehicle.eta_trans + vehicle.P_aux_W
    assert pd.P_dem[1] == pytest.approx(expected, rel=REL_EXACT)
    # Valeur numérique à la main avec les défauts : 20·(900 + 167,04 + 158,922) = 24 519,24 W roue.
    assert pd.P_wheel[1] == pytest.approx(24519.24, abs=1e-6)
    assert pd.P_dem[1] == pytest.approx(27151.3478, abs=1e-3)


def test_power_demand_freinage_pertes_multiplicatives(vehicle: VehicleParams) -> None:
    """En freinage (P_roue < 0), P_dem = P_roue · η_trans + P_aux : −59 742,30 W par défaut."""
    v0, a = 20.0, -2.0
    t, v = _linear_ramp(v0, a)
    pd = power_demand(t, v, vehicle)
    p_wheel = _wheel_formula(v0, a, vehicle)
    assert p_wheel < 0.0
    assert pd.P_wheel[1] == pytest.approx(p_wheel, rel=REL_EXACT)
    expected = p_wheel * vehicle.eta_trans + vehicle.P_aux_W
    assert pd.P_dem[1] == pytest.approx(expected, rel=REL_EXACT)
    assert pd.P_dem[1] == pytest.approx(-59742.2992, abs=1e-3)
    # Les pertes réduisent la puissance récupérée au bus : |P_dem − P_aux| < |P_roue|.
    assert abs(pd.P_dem[1] - vehicle.P_aux_W) < abs(pd.P_wheel[1])


def test_electrical_demand_pertes_dans_les_deux_sens(vehicle: VehicleParams) -> None:
    """La transmission divise par η en traction et multiplie par η en freinage."""
    x = 10_000.0
    out = electrical_demand(np.array([x, -x, 0.0]), vehicle)
    expected = np.array([x / vehicle.eta_trans, -x * vehicle.eta_trans, 0.0]) + vehicle.P_aux_W
    np.testing.assert_allclose(out, expected, rtol=REL_EXACT)
    # Dans les deux sens la pile+batterie fournissent plus / récupèrent moins que la roue.
    assert out[0] - vehicle.P_aux_W > x
    assert abs(out[1] - vehicle.P_aux_W) < x


def test_power_demand_arret_egale_p_aux(vehicle: VehicleParams) -> None:
    """À l'arrêt prolongé, P_roue = 0 et P_dem = P_aux sur toute la grille."""
    t = np.arange(10, dtype=float)
    pd = power_demand(t, np.zeros_like(t), vehicle)
    np.testing.assert_allclose(pd.P_wheel, 0.0, atol=ABS_W)
    np.testing.assert_allclose(pd.P_dem, vehicle.P_aux_W, atol=ABS_W)


def test_wheel_power_pas_de_roulement_a_vitesse_nulle(vehicle: VehicleParams) -> None:
    """À v = 0 la puissance roue est nulle quelle que soit l'accélération (démarrage compris)."""
    for a in (-1.0, 0.0, 1.0, 3.0):
        assert wheel_power(np.array([0.0]), np.array([a]), vehicle)[0] == pytest.approx(
            0.0, abs=ABS_W
        )
    # Démarrage : premier échantillon à l'arrêt avec accélération non nulle → P_dem = P_aux.
    t = np.array([0.0, 1.0, 2.0])
    pd = power_demand(t, np.array([0.0, 1.0, 2.0]), vehicle)
    assert pd.a[0] > 0.0
    assert pd.P_dem[0] == pytest.approx(vehicle.P_aux_W, abs=ABS_W)
    # Dès que v > 0 à accélération nulle, le roulement (Crr·m·g) et l'aéro apparaissent.
    v = 1.0
    expected = v * (0.5 * vehicle.rho_air * vehicle.Cd * vehicle.frontal_area_m2 * v**2)
    expected += v * vehicle.Crr * vehicle.mass_kg * vehicle.g
    assert wheel_power(np.array([v]), np.array([0.0]), vehicle)[0] == pytest.approx(
        expected, rel=REL_EXACT
    )


def test_power_demand_formes_coherentes_avec_la_grille(vehicle: VehicleParams) -> None:
    """Les séries renvoyées (t, v, a, P_roue, P_dem) ont toutes la taille de la grille."""
    t = np.arange(0.0, 30.0, 0.1)
    v = 5.0 * (1.0 - np.cos(t / 5.0))
    pd = power_demand(t, v, vehicle)
    for arr in (pd.t, pd.v, pd.a, pd.P_wheel, pd.P_dem):
        assert arr.shape == t.shape
    assert np.all(np.isfinite(pd.P_dem))


# ---------------------------------------------------------------------------------------
# (b) Batterie
# ---------------------------------------------------------------------------------------
def test_soc_rate_signe_decharge_baisse_charge_monte(battery: Battery) -> None:
    """P_bat > 0 (décharge) fait baisser le SOC, P_bat < 0 (charge) le fait monter."""
    rates = battery.soc_rate(np.array([10e3, -10e3, 0.0]))
    assert rates[0] < 0.0
    assert rates[1] > 0.0
    assert rates[2] == pytest.approx(0.0, abs=ABS_SOC)
    # Asymétrie des rendements : à même |P|, la décharge coûte plus que la charge ne rapporte.
    assert abs(rates[0]) > abs(rates[1])


def test_step_puissance_constante_formule_exacte(battery: Battery) -> None:
    """step(soc, P, dt) = soc − P_int·dt/E_cap_J, en décharge comme en charge."""
    p = battery.p
    assert p.E_cap_J == pytest.approx(p.E_cap_Wh * 3600.0, rel=REL_EXACT)
    soc_dis = battery.step(0.60, 10e3, 1.0)
    assert soc_dis == pytest.approx(0.60 - (10e3 / p.eta_dis) * 1.0 / p.E_cap_J, rel=REL_EXACT)
    assert soc_dis == pytest.approx(0.598172514619883, abs=1e-12)  # défauts : 1,6 kWh, η = 0,95
    soc_ch = battery.step(0.60, -10e3, 2.0)
    assert soc_ch == pytest.approx(0.60 + (10e3 * p.eta_ch) * 2.0 / p.E_cap_J, rel=REL_EXACT)
    assert soc_ch > 0.60 > soc_dis


def test_step_lineaire_en_dt_a_puissance_constante(battery: Battery) -> None:
    """À puissance constante, 10 sous-pas de dt/10 donnent le même SOC qu'un pas dt."""
    soc, P_bat, dt = 0.55, 8e3, 1.0
    one_step = battery.step(soc, P_bat, dt)
    s = soc
    for _ in range(10):
        s = battery.step(s, P_bat, dt / 10)
    assert s == pytest.approx(one_step, abs=ABS_SOC)
    # Cohérence avec soc_rate : ΔSOC = soc_rate(P)·dt.
    assert one_step - soc == pytest.approx(float(battery.soc_rate(P_bat)) * dt, rel=REL_EXACT)


def test_internal_power_rendements_decharge_et_charge(battery: Battery) -> None:
    """P_int = P/η_dis en décharge, P·η_ch en charge, 0 à 0 ; scalaire et vectoriel identiques."""
    p = battery.p
    P_bat = np.array([10e3, -10e3, 0.0])
    out = battery.internal_power(P_bat)
    np.testing.assert_allclose(out, [10e3 / p.eta_dis, -10e3 * p.eta_ch, 0.0], rtol=REL_EXACT)
    for P, ref in zip(P_bat, out, strict=True):
        assert battery.internal_power_scalar(float(P)) == pytest.approx(ref, rel=REL_EXACT)
    # Les pertes vont toujours dans le même sens : plus sorti du stock, moins rentré.
    assert out[0] > 10e3
    assert abs(out[1]) < 10e3


def test_discharge_limit_croissante_avec_soc(battery: Battery) -> None:
    """La limite de décharge est croissante (au sens large) sur une grille fine de SOC."""
    grid = np.linspace(0.0, 1.0, 2001)
    lim = battery.discharge_limit_v(grid)
    assert np.all(np.diff(lim) >= -MONO_TOL_W)
    assert lim[0] == pytest.approx(0.0, abs=ABS_W)
    assert lim[-1] == pytest.approx(battery.p.P_dis_max_W, abs=ABS_W)
    # Strictement croissante à l'intérieur de la bande de dérating.
    p = battery.p
    band = np.linspace(p.soc_min + 1e-3, p.soc_min + p.derate_band - 1e-3, 50)
    assert np.all(np.diff(battery.discharge_limit_v(band)) > 0.0)


def test_charge_limit_decroissante_avec_soc(battery: Battery) -> None:
    """La limite de charge est décroissante (au sens large) sur une grille fine de SOC."""
    grid = np.linspace(0.0, 1.0, 2001)
    lim = battery.charge_limit_v(grid)
    assert np.all(np.diff(lim) <= MONO_TOL_W)
    assert lim[0] == pytest.approx(battery.p.P_ch_max_W, abs=ABS_W)
    assert lim[-1] == pytest.approx(0.0, abs=ABS_W)
    p = battery.p
    band = np.linspace(p.soc_max - p.derate_band + 1e-3, p.soc_max - 1e-3, 50)
    assert np.all(np.diff(battery.charge_limit_v(band)) < 0.0)


def test_limites_scalaires_et_vectorielles_coherentes(battery: Battery) -> None:
    """discharge_limit / charge_limit (scalaires) coïncident avec les versions vectorielles."""
    grid = np.linspace(0.0, 1.0, 501)
    dis_scalar = np.array([battery.discharge_limit(float(s)) for s in grid])
    ch_scalar = np.array([battery.charge_limit(float(s)) for s in grid])
    np.testing.assert_allclose(dis_scalar, battery.discharge_limit_v(grid), atol=ABS_W)
    np.testing.assert_allclose(ch_scalar, battery.charge_limit_v(grid), atol=ABS_W)
    # Un scalaire passé à la version vectorielle rend un tableau 0-d de même valeur.
    assert float(battery.discharge_limit_v(0.6)) == pytest.approx(battery.discharge_limit(0.6))


def test_limites_nulles_aux_bornes_de_la_fenetre(battery: Battery) -> None:
    """Décharge nulle à soc_min, charge nulle à soc_max, pleines puissances à l'autre borne."""
    p = battery.p
    assert battery.discharge_limit(p.soc_min) == pytest.approx(0.0, abs=ABS_W)
    assert battery.charge_limit(p.soc_max) == pytest.approx(0.0, abs=ABS_W)
    assert battery.discharge_limit(p.soc_max) == pytest.approx(p.P_dis_max_W, abs=ABS_W)
    assert battery.charge_limit(p.soc_min) == pytest.approx(p.P_ch_max_W, abs=ABS_W)
    # Rampe linéaire : mi-bande = moitié de la puissance max.
    half = 0.5 * p.derate_band
    assert battery.discharge_limit(p.soc_min + half) == pytest.approx(0.5 * p.P_dis_max_W, rel=1e-9)
    assert battery.charge_limit(p.soc_max - half) == pytest.approx(0.5 * p.P_ch_max_W, rel=1e-9)
    # Hors fenêtre, les limites restent saturées à 0 (jamais négatives).
    assert battery.discharge_limit(p.soc_min - 0.1) == 0.0
    assert battery.charge_limit(p.soc_max + 0.1) == 0.0


def test_limites_bornees_entre_zero_et_pmax(battery: Battery) -> None:
    """Sur tout [0, 1], 0 ≤ limite ≤ P_max correspondant."""
    grid = np.linspace(0.0, 1.0, 1001)
    dis, ch = battery.discharge_limit_v(grid), battery.charge_limit_v(grid)
    assert np.all(dis >= 0.0) and np.all(dis <= battery.p.P_dis_max_W + ABS_W)
    assert np.all(ch >= 0.0) and np.all(ch <= battery.p.P_ch_max_W + ABS_W)


def test_reachable_soc_dans_fenetre_et_encadre_soc(battery: Battery) -> None:
    """Pour tout SOC utile et tout horizon, soc_min ≤ lo ≤ soc ≤ hi ≤ soc_max."""
    p = battery.p
    for soc in np.linspace(p.soc_min, p.soc_max, 26):
        for horizon in (0.1, 1.0, 10.0, 100.0, 1e6):
            lo, hi = battery.reachable_soc(float(soc), horizon)
            assert p.soc_min - ABS_SOC <= lo <= soc + ABS_SOC
            assert soc - ABS_SOC <= hi <= p.soc_max + ABS_SOC


def test_reachable_soc_valeur_recalculee_et_horizon_infini(battery: Battery) -> None:
    """À 60 % sur 10 s : lo = soc − P_dis/η_dis·H/E_cap, hi = soc + P_ch·η_ch·H/E_cap."""
    p = battery.p
    soc, horizon = 0.60, 10.0
    lo, hi = battery.reachable_soc(soc, horizon)
    assert lo == pytest.approx(soc - p.P_dis_max_W / p.eta_dis * horizon / p.E_cap_J, rel=REL_EXACT)
    assert hi == pytest.approx(soc + p.P_ch_max_W * p.eta_ch * horizon / p.E_cap_J, rel=REL_EXACT)
    assert (lo, hi) == pytest.approx((0.5451754385964912, 0.6494791666666666), abs=1e-12)
    # Horizon nul : la plage se réduit au SOC courant ; horizon immense : toute la fenêtre.
    assert battery.reachable_soc(soc, 0.0) == pytest.approx((soc, soc), abs=ABS_SOC)
    assert battery.reachable_soc(soc, 1e9) == pytest.approx((p.soc_min, p.soc_max), abs=ABS_SOC)


def test_battery_params_fenetre_trop_etroite_rejetee() -> None:
    """Une fenêtre SOC plus étroite que deux bandes de dérating est refusée par pydantic."""
    with pytest.raises(ValidationError):
        BatteryParams(soc_min=0.45, soc_max=0.52, derate_band=0.05)  # 0,07 < 2 × 0,05


# ---------------------------------------------------------------------------------------
# (c) Pile à combustible
# ---------------------------------------------------------------------------------------
def test_h2_power_convexe_differences_secondes_positives(fuel_cell: FuelCell) -> None:
    """P_H2(P) est convexe (différences secondes ≥ 0) et strictement positive sur [0, P_max]."""
    P = np.linspace(0.0, fuel_cell.p.P_max_W, 401)
    h2 = fuel_cell.h2_power(P)
    assert np.all(np.diff(h2, 2) >= 0.0)
    assert np.all(h2 > 0.0)
    assert np.all(np.diff(h2) > 0.0)  # consommation croissante avec la puissance nette
    assert fuel_cell.a > 0.0 and fuel_cell.c > 0.0


def test_efficiency_en_cloche_pic_entre_15_et_35_pct(fuel_cell: FuelCell) -> None:
    """η(P) est unimodale, culmine entre 15 et 35 % de P_max et η(P_max) < pic."""
    p_max = fuel_cell.p.P_max_W
    P = np.linspace(0.0, p_max, 2001)
    eta = fuel_cell.efficiency(P)
    k = int(np.argmax(eta))
    assert 0.15 <= P[k] / p_max <= 0.35
    assert eta[-1] < eta[k]
    assert eta[0] == pytest.approx(0.0, abs=1e-12)  # débit de ralenti sans puissance nette
    assert np.all(np.diff(eta[: k + 1]) >= -1e-12)  # montée
    assert np.all(np.diff(eta[k:]) <= 1e-12)  # descente
    assert np.all((eta >= 0.0) & (eta < 1.0))
    # La propriété peak_power (√(a/c)) coïncide avec l'argmax de la grille.
    assert fuel_cell.peak_power == pytest.approx(P[k], abs=p_max / 2000)
    assert 0.15 <= fuel_cell.peak_power / p_max <= 0.35


def test_efficiency_reproduit_la_carte_de_config(fuel_cell: FuelCell) -> None:
    """Le modèle convexe ajusté rend chaque point de la carte η à mieux que 1 %."""
    p = fuel_cell.p
    P_map = np.asarray(p.eta_map_frac) * p.P_max_W
    eta_fit = fuel_cell.efficiency(P_map)
    np.testing.assert_allclose(eta_fit, p.eta_map_eta, rtol=1e-2)


def test_h2_rate_ralenti_egal_a_sur_pci(fuel_cell: FuelCell) -> None:
    """ṁ_H2(0) = a / PCI > 0 ; une puissance négative est traitée comme 0 ; ṁ = P_H2 / PCI."""
    rate0 = float(fuel_cell.h2_rate(0.0))
    assert rate0 == pytest.approx(fuel_cell.a / fuel_cell.p.lhv_J_per_kg, rel=REL_EXACT)
    assert rate0 > 0.0
    assert float(fuel_cell.h2_rate(-5e3)) == pytest.approx(rate0, rel=REL_EXACT)
    P = np.array([5e3, 20e3, 95e3])
    np.testing.assert_allclose(
        fuel_cell.h2_rate(P), fuel_cell.h2_power(P) / fuel_cell.p.lhv_J_per_kg, rtol=REL_EXACT
    )
    # Cohérence conso / rendement : ṁ·PCI·η = P.
    np.testing.assert_allclose(
        fuel_cell.h2_rate(P) * fuel_cell.p.lhv_J_per_kg * fuel_cell.efficiency(P), P, rtol=1e-12
    )


def test_clip_borne_par_pmin_pmax(fuel_cell: FuelCell) -> None:
    """clip sature dans [P_min, P_max] et laisse l'intérieur inchangé."""
    p = fuel_cell.p
    assert fuel_cell.clip(-1e9) == p.P_min_W
    assert fuel_cell.clip(0.0) == p.P_min_W
    assert fuel_cell.clip(1e9) == p.P_max_W
    assert fuel_cell.clip(p.P_min_W) == p.P_min_W
    assert fuel_cell.clip(p.P_max_W) == p.P_max_W
    mid = 0.5 * (p.P_min_W + p.P_max_W)
    assert fuel_cell.clip(mid) == mid


def test_ramp_limits_bornes_par_pmin_pmax_et_rampe(fuel_cell: FuelCell) -> None:
    """Pour P_now ∈ [P_min, P_max] : P_min ≤ lo ≤ P_now ≤ hi ≤ P_max et |hi − lo| ≤ 2·r·dt."""
    p = fuel_cell.p
    for P_now in np.linspace(p.P_min_W, p.P_max_W, 41):
        for dt in (0.1, 1.0, 5.0):
            lo, hi = fuel_cell.ramp_limits(float(P_now), dt)
            assert p.P_min_W <= lo <= P_now <= hi <= p.P_max_W
            assert hi - lo <= 2.0 * p.ramp_W_per_s * dt + ABS_W
    # Intérieur : plage symétrique ±r·dt ; aux bornes : saturation sur la borne statique.
    P_mid, dt = 50e3, 1.0
    dP = p.ramp_W_per_s * dt
    assert fuel_cell.ramp_limits(P_mid, dt) == pytest.approx((P_mid - dP, P_mid + dP), abs=ABS_W)
    assert fuel_cell.ramp_limits(p.P_min_W, dt) == pytest.approx(
        (p.P_min_W, p.P_min_W + dP), abs=ABS_W
    )
    assert fuel_cell.ramp_limits(p.P_max_W, dt) == pytest.approx(
        (p.P_max_W - dP, p.P_max_W), abs=ABS_W
    )


def test_fuel_cell_params_carte_non_croissante_rejetee() -> None:
    """Une carte dont les fractions de puissance ne croissent pas strictement est refusée."""
    with pytest.raises(ValidationError):
        FuelCellParams(
            eta_map_frac=(0.10, 0.05, 0.20, 0.50, 1.0), eta_map_eta=(0.40, 0.50, 0.55, 0.50, 0.45)
        )
    with pytest.raises(ValidationError):  # doublon = non strictement croissante
        FuelCellParams(eta_map_frac=(0.1, 0.2, 0.2, 1.0), eta_map_eta=(0.4, 0.5, 0.5, 0.45))


def test_fuel_cell_params_pmin_superieur_ou_egal_pmax_rejete() -> None:
    """P_min_W ≥ P_max_W est refusé (égalité comprise)."""
    with pytest.raises(ValidationError):
        FuelCellParams(P_min_W=95e3, P_max_W=95e3)
    with pytest.raises(ValidationError):
        FuelCellParams(P_min_W=100e3, P_max_W=95e3)


def test_fuel_cell_params_carte_incoherente_rejetee() -> None:
    """Longueurs différentes, rendement hors ]0, 1[ ou carte trop courte sont refusés."""
    with pytest.raises(ValidationError):
        FuelCellParams(eta_map_frac=(0.1, 0.5, 1.0), eta_map_eta=(0.4, 0.5))
    with pytest.raises(ValidationError):
        FuelCellParams(eta_map_frac=(0.1, 0.5, 1.0), eta_map_eta=(0.4, 1.2, 0.45))
    with pytest.raises(ValidationError):
        FuelCellParams(eta_map_frac=(0.1, 1.0), eta_map_eta=(0.4, 0.45))


def test_fuel_cell_params_par_defaut_valides() -> None:
    """Les paramètres par défaut construisent une pile avec a > 0, c > 0 sans exception."""
    fc = FuelCell(FuelCellParams())
    assert fc.a > 0.0 and fc.c > 0.0
    assert fc.p.P_min_W < fc.peak_power < fc.p.P_max_W


# ---------------------------------------------------------------------------------------
# (d) Cycles
# ---------------------------------------------------------------------------------------
def test_load_wltc_1801_points_a_1hz_duree_1800s(wltc: Cycle) -> None:
    """Le WLTC embarqué compte 1801 échantillons espacés de 1 s, soit 1800 s."""
    assert wltc.name == "wltc"
    assert len(wltc.t) == len(wltc.v) == 1801
    np.testing.assert_allclose(np.diff(wltc.t), 1.0, atol=1e-12)
    assert wltc.duration == pytest.approx(1800.0, abs=1e-9)
    assert wltc.labels is None


def test_load_wltc_distance_et_vitesse_max(wltc: Cycle) -> None:
    """Distance totale 23,27 km ± 1 % et vitesse max 131,3 km/h ± 0,5."""
    assert wltc.distance_m / 1000.0 == pytest.approx(23.27, rel=1e-2)
    assert float(wltc.v.max()) * 3.6 == pytest.approx(131.3, abs=0.5)


def test_load_wltc_depart_et_arrivee_a_l_arret(wltc: Cycle) -> None:
    """Le WLTC commence et finit à l'arrêt, sans vitesse négative."""
    assert wltc.v[0] == 0.0
    assert wltc.v[-1] == 0.0
    assert np.all(wltc.v >= 0.0)
    assert np.all(np.isfinite(wltc.v))


def test_synthetic_cycle_deterministe_a_graine_egale(three_segments) -> None:
    """Deux appels avec la même graine donnent exactement le même cycle."""
    c1 = synthetic_cycle(three_segments, seed=7)
    c2 = synthetic_cycle(three_segments, seed=7)
    np.testing.assert_array_equal(c1.t, c2.t)
    np.testing.assert_array_equal(c1.v, c2.v)
    np.testing.assert_array_equal(c1.labels, c2.labels)
    assert c1.meta == c2.meta == {"seed": 7}


def test_synthetic_cycle_different_a_graine_differente(three_segments) -> None:
    """Changer la graine change le profil de vitesse (longueur ou valeurs)."""
    c1 = synthetic_cycle(three_segments, seed=7)
    c3 = synthetic_cycle(three_segments, seed=8)
    assert c1.v.shape != c3.v.shape or not np.array_equal(c1.v, c3.v)


def test_synthetic_cycle_labels_meme_longueur_que_v(three_segments) -> None:
    """Les étiquettes ont exactement la taille du profil, sur les cycles du catalogue aussi."""
    c = synthetic_cycle(three_segments, seed=3)
    assert c.labels is not None
    assert len(c.labels) == len(c.v) == len(c.t)
    for name in SYNTH_CATALOG:
        cyc = get_cycle(name)
        assert cyc.labels is not None
        assert len(cyc.labels) == len(cyc.v) == len(cyc.t)


def test_synthetic_cycle_chaque_segment_etiquete_avec_son_schema(three_segments) -> None:
    """Chaque segment porte l'étiquette de son schéma ; lead-in et queue portent 1er / dernier."""
    lead_in = 5
    for seed in (0, 1, 2):
        c = synthetic_cycle(three_segments, seed=seed, lead_in_s=lead_in)
        assert c.labels is not None
        pos = lead_in
        np.testing.assert_array_equal(c.labels[:lead_in], int(three_segments[0][0]))
        for pattern, dur in three_segments:
            np.testing.assert_array_equal(c.labels[pos : pos + dur], int(pattern))
            pos += dur
        assert pos < len(c.labels)  # il reste la décélération finale + 5 s d'arrêt
        np.testing.assert_array_equal(c.labels[pos:], int(three_segments[-1][0]))
        assert set(np.unique(c.labels)) == {int(p) for p, _ in three_segments}


def test_synthetic_cycle_depart_arrivee_arret_vitesses_positives(three_segments) -> None:
    """Chaque cycle synthétique part de 0, revient à 0 et ne contient aucune vitesse négative."""
    cycles = [synthetic_cycle(three_segments, seed=s) for s in range(3)]
    cycles += [get_cycle(name) for name in SYNTH_CATALOG]
    for c in cycles:
        assert c.v[0] == 0.0
        assert c.v[-1] == 0.0
        assert np.all(c.v >= 0.0)
        assert np.all(np.isfinite(c.v))
        np.testing.assert_allclose(np.diff(c.t), 1.0, atol=1e-12)  # 1 Hz
        assert c.duration == pytest.approx(len(c.v) - 1, abs=1e-9)


def test_synthetic_cycle_vitesses_dans_la_plage_du_schema() -> None:
    """Un segment autoroute seul ne dépasse jamais la borne haute de vitesse de sa spec."""
    spec = SEGMENT_SPECS[DrivingPattern.AUTOROUTE]
    c = synthetic_cycle([(DrivingPattern.AUTOROUTE, 400)], seed=5)
    assert float(c.v.max()) * 3.6 <= spec.v_kmh[1] + 1e-9
    assert float(c.v.max()) * 3.6 > spec.v_kmh[0]  # on atteint bien le régime autoroutier


def test_synth_mixed_couvre_les_cinq_schemas() -> None:
    """Le cycle mixte du catalogue contient les cinq schémas de conduite."""
    c = get_cycle("synth_mixed")
    assert c.labels is not None
    assert set(np.unique(c.labels)) == {int(p) for p in DrivingPattern}
    assert set(SEGMENT_SPECS) == set(DrivingPattern)


def test_resample_conserve_points_origine_duree_distance_labels(three_segments) -> None:
    """resample(0.1) garde les échantillons 1 Hz, la durée, la distance (± 0,1 %) et les labels."""
    c = synthetic_cycle(three_segments, seed=11)
    r = c.resample(0.1)
    assert len(r.t) == 10 * (len(c.t) - 1) + 1
    np.testing.assert_allclose(r.t[::10], c.t, atol=1e-9)
    np.testing.assert_allclose(r.v[::10], c.v, atol=1e-9)
    assert r.duration == pytest.approx(c.duration, abs=1e-9)
    assert r.distance_m == pytest.approx(c.distance_m, rel=1e-3)
    assert r.labels is not None
    assert len(r.labels) == len(r.v)
    np.testing.assert_array_equal(r.labels[::10], c.labels)
    assert r.name == c.name and r.meta == c.meta
    # Les sous-échantillons portent l'étiquette de l'échantillon 1 Hz qui les précède.
    np.testing.assert_array_equal(r.labels[:10], c.labels[0])


def test_resample_wltc_conserve_points_duree_distance(wltc: Cycle) -> None:
    """Le WLTC rééchantillonné à 0,1 s a 18001 points, même durée, même distance (± 0,1 %)."""
    r = wltc.resample(0.1)
    assert len(r.t) == 18001
    np.testing.assert_allclose(r.v[::10], wltc.v, atol=1e-9)
    assert r.duration == pytest.approx(1800.0, abs=1e-9)
    assert r.distance_m == pytest.approx(wltc.distance_m, rel=1e-3)
    assert r.labels is None
    np.testing.assert_allclose(np.diff(r.t), 0.1, atol=1e-9)


def test_resample_lineaire_reste_dans_l_enveloppe(wltc: Cycle) -> None:
    """L'interpolation linéaire ne crée ni dépassement ni vitesse négative."""
    r = wltc.resample(0.1)
    assert float(r.v.min()) >= 0.0
    assert float(r.v.max()) <= float(wltc.v.max()) + 1e-12


def test_get_cycle_inconnu_leve_keyerror() -> None:
    """Un nom absent du catalogue lève KeyError."""
    with pytest.raises(KeyError):
        get_cycle("inconnu")


def test_list_cycles_et_get_cycle_catalogue_complet() -> None:
    """list_cycles expose les 5 cycles attendus et get_cycle renvoie un cycle du bon nom."""
    assert set(list_cycles()) == set(CATALOG)
    for name in CATALOG:
        c = get_cycle(name)
        assert c.name == name
        assert len(c.t) == len(c.v) > 100
