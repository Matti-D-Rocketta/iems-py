"""Tests du bloc Candidates (``iems.candidates``) : bornes, 9 candidats, vecteur de secours.

Couvre aussi la monotonie des limites batterie vs SOC (section 7 de la passation), dont
dépendent directement les plafonds/planchers anticipés.
"""

from __future__ import annotations

import numpy as np
import pytest

from iems.candidates import (
    N_CANDIDATES,
    CandidateSet,
    anticipation_bounds,
    build_candidates,
    dilate,
    interval_extrema,
)
from iems.config import IEMSConfig
from iems.models.battery import Battery
from iems.models.fuel_cell import FuelCell

# --- constantes de test -----------------------------------------------------------------
SOC_MID = 0.60  # SOC loin des bandes de dérating : limites batterie pleines
N_PRED = 101  # Hp = 10 s au pas Ts_model = 0,1 s, instant courant inclus
ATOL_W = 1e-6  # tolérance absolue sur les puissances (arithmétique exacte attendue)
VALID_EPS_W = 1e-6  # seuil de validité utilisé par candidates.py (excess <= 1e-6)


# --- fixtures locales -------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> IEMSConfig:
    return IEMSConfig()


@pytest.fixture(scope="module")
def fc(cfg: IEMSConfig) -> FuelCell:
    return FuelCell(cfg.fuel_cell)


@pytest.fixture(scope="module")
def battery(cfg: IEMSConfig) -> Battery:
    return Battery(cfg.battery)


@pytest.fixture(scope="module")
def substeps(cfg: IEMSConfig) -> int:
    return cfg.ems.substeps


# --- aides ------------------------------------------------------------------------------
def constant_demand(p_W: float, n: int = N_PRED) -> np.ndarray:
    return np.full(n, float(p_W))


def step_demand(p0_W: float, p1_W: float, t_step_s: float, ts_model: float) -> np.ndarray:
    """Demande ``p0`` puis ``p1`` à partir de ``t_step_s`` (échantillon inclus)."""
    t = np.arange(N_PRED) * ts_model
    return np.where(t < t_step_s - 1e-12, float(p0_W), float(p1_W))


def derated_limits(cfg: IEMSConfig, battery: Battery, soc: float) -> tuple[float, float]:
    """Reproduit (P_dis, P_ch) minorés tels que documentés dans ``anticipation_bounds``."""
    ems, bp = cfg.ems, battery.p
    d_soc = max(bp.P_dis_max_W, bp.P_ch_max_W) * ems.Ts_ems / (bp.eta_dis * bp.E_cap_J)
    keep = 1.0 - ems.safety_margin
    p_dis = max(battery.discharge_limit(soc - d_soc) * keep - ems.safety_margin_W, 0.0)
    p_ch = max(battery.charge_limit(soc + d_soc) * keep - ems.safety_margin_W, 0.0)
    return p_dis, p_ch


def assert_well_formed(cs: CandidateSet, fc: FuelCell, P_fc_now: float, Ts_ems: float) -> None:
    """Invariants communs : 9 valeurs finies, triées, dans la plage utilisable et la rampe PàC."""
    assert cs.P_fc.shape == (N_CANDIDATES,)
    assert cs.P_bat.shape == (N_CANDIDATES,)
    assert cs.valid.shape == (N_CANDIDATES,)
    assert cs.excess.shape == (N_CANDIDATES,)
    assert np.all(np.isfinite(cs.P_fc))
    assert np.all(np.isfinite(cs.P_bat))
    assert np.all(np.isfinite(cs.excess))
    assert np.isfinite(cs.P_lo) and np.isfinite(cs.P_hi)
    assert np.isfinite(cs.floor) and np.isfinite(cs.ceiling)
    assert cs.P_lo <= cs.P_hi
    assert np.all(np.diff(cs.P_fc) >= -ATOL_W)
    assert np.all(cs.P_fc >= cs.P_lo - ATOL_W)
    assert np.all(cs.P_fc <= cs.P_hi + ATOL_W)
    r_lo, r_hi = fc.ramp_limits(P_fc_now, Ts_ems)
    assert np.all(cs.P_fc >= r_lo - ATOL_W)
    assert np.all(cs.P_fc <= r_hi + ATOL_W)
    assert np.all(cs.P_fc >= fc.p.P_min_W - ATOL_W)
    assert np.all(cs.P_fc <= fc.p.P_max_W + ATOL_W)
    assert np.all(cs.excess >= 0.0)
    assert np.array_equal(cs.valid, cs.excess <= VALID_EPS_W)


# =========================================================================================
# (a) Neuf candidats par quarts d'écart
# =========================================================================================
def test_build_candidates_nominal_neuf_candidats_tries_dans_plage(cfg, fc, battery):
    """Cas nominal 20 kW/20 kW : 9 candidats triés, de P_lo à P_hi, cinquième = P_fc courant."""
    P_now = 20e3
    cs = build_candidates(P_now, constant_demand(20e3), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    assert not cs.conflict
    assert len(cs.P_fc) == 9
    assert cs.P_fc[0] == pytest.approx(cs.P_lo, abs=ATOL_W)
    assert cs.P_fc[-1] == pytest.approx(cs.P_hi, abs=ATOL_W)
    assert cs.P_fc[4] == pytest.approx(P_now, abs=ATOL_W)
    # Plage = rampe PàC pleine (anticipation non contraignante à mi-SOC)
    r_lo, r_hi = fc.ramp_limits(P_now, cfg.ems.Ts_ems)
    assert cs.P_lo == pytest.approx(r_lo, abs=ATOL_W)
    assert cs.P_hi == pytest.approx(r_hi, abs=ATOL_W)


def test_build_candidates_quarts_ecart_delta_moins_et_delta_plus(cfg, fc, battery):
    """Les 4 premiers pas valent Δ⁻ = (c − lo)/4 et les 4 derniers Δ⁺ = (hi − c)/4."""
    P_now = 3e3  # proche du ralenti : lo = P_min, d'où Δ⁻ ≠ Δ⁺
    cs = build_candidates(P_now, constant_demand(5e3), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    c = cs.P_fc[4]
    d_minus = (c - cs.P_lo) / 4.0
    d_plus = (cs.P_hi - c) / 4.0
    assert d_minus != pytest.approx(d_plus, abs=1.0)  # asymétrie effectivement testée
    np.testing.assert_allclose(np.diff(cs.P_fc[:5]), d_minus, rtol=0, atol=ATOL_W)
    np.testing.assert_allclose(np.diff(cs.P_fc[4:]), d_plus, rtol=0, atol=ATOL_W)
    expected = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4250.0, 5500.0, 6750.0, 8000.0]
    np.testing.assert_allclose(cs.P_fc, expected, rtol=0, atol=ATOL_W)


def test_build_candidates_centre_ramene_sur_borne_basse(cfg, fc, battery):
    """P_fc courant sous le plancher : c = lo, les 5 premiers candidats sont confondus en lo."""
    P_now = 20e3
    soc = cfg.battery.soc_min  # décharge interdite : plancher = demande majorée > P_now
    cs = build_candidates(P_now, constant_demand(20e3), soc, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    assert cs.P_lo > P_now
    np.testing.assert_allclose(cs.P_fc[:5], cs.P_lo, rtol=0, atol=ATOL_W)
    d_plus = (cs.P_hi - cs.P_lo) / 4.0
    np.testing.assert_allclose(np.diff(cs.P_fc[4:]), d_plus, rtol=0, atol=ATOL_W)
    assert cs.P_fc[-1] == pytest.approx(cs.P_hi, abs=ATOL_W)


def test_build_candidates_centre_ramene_sur_borne_haute(cfg, fc, battery):
    """P_fc courant au-dessus du plafond : c = hi, les 5 derniers candidats sont confondus en hi."""
    P_now = 20e3
    soc = cfg.battery.soc_max  # charge interdite : plafond = demande minorée < P_now
    cs = build_candidates(P_now, constant_demand(20e3), soc, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    assert cs.P_hi < P_now
    np.testing.assert_allclose(cs.P_fc[4:], cs.P_hi, rtol=0, atol=ATOL_W)
    d_minus = (cs.P_hi - cs.P_lo) / 4.0
    np.testing.assert_allclose(np.diff(cs.P_fc[:5]), d_minus, rtol=0, atol=ATOL_W)
    assert cs.P_fc[0] == pytest.approx(cs.P_lo, abs=ATOL_W)


@pytest.mark.parametrize(
    ("P_now", "p_dem"),
    [
        (1e3, 0.0),
        (1e3, 15e3),
        (20e3, 20e3),
        (20e3, -10e3),
        (50e3, 60e3),
        (94e3, 90e3),
        (95e3, 95e3),
    ],
    ids=["idle-0", "idle-15k", "20k-20k", "20k-brake", "50k-60k", "94k-90k", "max-95k"],
)
def test_build_candidates_bornes_pac_et_rampe_respectees(cfg, fc, battery, P_now, p_dem):
    """Tous les candidats restent dans [P_min, P_max] et dans la rampe atteignable en Ts_ems."""
    cs = build_candidates(P_now, constant_demand(p_dem), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)


def test_build_candidates_ralenti_demande_nulle_plage_rampe_complete(cfg, fc, battery):
    """Au ralenti sans demande, la plage est [P_min, P_min + r·Ts] et la batterie se charge."""
    P_now = cfg.fuel_cell.P_min_W
    cs = build_candidates(P_now, constant_demand(0.0), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    assert cs.P_lo == pytest.approx(cfg.fuel_cell.P_min_W, abs=ATOL_W)
    assert cs.P_hi == pytest.approx(
        cfg.fuel_cell.P_min_W + cfg.fuel_cell.ramp_W_per_s * cfg.ems.Ts_ems, abs=ATOL_W
    )
    assert np.all(cs.P_bat <= 0.0)  # tout l'excédent PàC part en charge batterie
    assert np.all(cs.valid)


# =========================================================================================
# (b) Dégénérescence : vecteur de secours
# =========================================================================================
def _degenerate_cases(cfg: IEMSConfig) -> list:
    fcp, bp, ems = cfg.fuel_cell, cfg.battery, cfg.ems
    r_ts = fcp.ramp_W_per_s * ems.Ts_ems
    return [
        # PàC à P_max, demande faible : plafond < P_lo → secours en P_lo = P_max − r·Ts
        pytest.param(
            fcp.P_max_W, constant_demand(20e3), SOC_MID, cfg, fcp.P_max_W - r_ts, id="pmax-low"
        ),
        # PàC à P_max, demande hors de portée : plancher > P_hi → secours en P_max
        pytest.param(
            fcp.P_max_W, constant_demand(200e3), SOC_MID, cfg, fcp.P_max_W, id="pmax-high"
        ),
        # Marche +40 kW dans 2 s, PàC basse : plancher > P_hi → secours en P_hi (rampe saturée)
        pytest.param(
            5e3, step_demand(5e3, 45e3, 2.0, ems.Ts_model), SOC_MID, cfg, 5e3 + r_ts, id="step-ramp"
        ),
        # SOC max et freinage : plafond (P_aux) < P_lo → secours en P_lo
        pytest.param(20e3, constant_demand(-10e3), bp.soc_max, cfg, 20e3 - r_ts, id="socmax-brake"),
    ]


@pytest.mark.parametrize(
    ("P_now", "p_dem", "soc", "cfg_case", "expected"), _degenerate_cases(IEMSConfig())
)
def test_build_candidates_degenere_neuf_copies_sans_nan(P_now, p_dem, soc, cfg_case, expected):
    """Plage réduite à un point : degenerate True, 9 copies de la valeur admissible, sans NaN."""
    fc_case, bat_case = FuelCell(cfg_case.fuel_cell), Battery(cfg_case.battery)
    cs = build_candidates(P_now, p_dem, soc, 0.0, cfg_case, fc_case, bat_case)
    assert_well_formed(cs, fc_case, P_now, cfg_case.ems.Ts_ems)
    assert cs.degenerate
    assert cs.P_hi - cs.P_lo < 1.0
    np.testing.assert_allclose(cs.P_fc, expected, rtol=0, atol=ATOL_W)
    assert np.all(cs.P_fc == cs.P_fc[0])
    assert np.all(cs.P_bat == cs.P_bat[0])
    assert np.all(cs.excess == cs.excess[0])


def test_build_candidates_rampe_quasi_nulle_plancher_egal_plafond(cfg, battery):
    """Rampe PàC de 0,1 W/s : P_hi − P_lo = 0,2 W < 1 W → secours en P_lo, sans NaN."""
    cfg_r = cfg.with_updates(fuel_cell={"ramp_W_per_s": 0.1})
    fc_r = FuelCell(cfg_r.fuel_cell)
    P_now = 20e3
    cs = build_candidates(P_now, constant_demand(20e3), SOC_MID, 0.0, cfg_r, fc_r, battery)
    assert_well_formed(cs, fc_r, P_now, cfg_r.ems.Ts_ems)
    assert cs.degenerate
    assert cs.P_hi - cs.P_lo == pytest.approx(0.2, abs=1e-9)
    np.testing.assert_allclose(cs.P_fc, cs.P_lo, rtol=0, atol=ATOL_W)
    np.testing.assert_allclose(cs.P_bat, 20e3 - cs.P_lo, rtol=0, atol=ATOL_W)
    assert np.all(cs.valid)


def test_build_candidates_conflit_reel_plancher_egal_plafond(cfg, fc, battery):
    """Marche hors de portée PàC + batterie : conflict True, plancher = plafond, secours fini."""
    P_now = 20e3
    p_dem = step_demand(5e3, 100e3, 1.0, cfg.ems.Ts_model)
    cs = build_candidates(P_now, p_dem, SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert cs.conflict
    assert cs.degenerate
    assert cs.floor == pytest.approx(cs.ceiling, abs=ATOL_W)
    # Milieu de [f0, c0] sans marge : f0 = 100 kW − P_dis, c0 = 5 kW + P_ch
    p_dis, p_ch = derated_limits(cfg, battery, SOC_MID)
    assert cs.floor == pytest.approx(0.5 * ((100e3 - p_dis) + (5e3 + p_ch)), abs=ATOL_W)


def test_build_candidates_marges_reduites_par_dichotomie_sans_conflit(cfg, fc, battery):
    """Marges contradictoires mais réductibles : conflict False, plancher = plafond."""
    P_now = 20e3
    soc = cfg.battery.soc_min  # P_dis = 0 : les marges inertielles se croisent à forte demande
    v = 36.0  # m/s : marge inertielle maximale
    p_dem = constant_demand(150e3)
    p_max, p_min, _ = interval_extrema(p_dem, cfg.ems.substeps)
    f1, c1 = anticipation_bounds(p_max, p_min, soc, cfg, battery, v=v)
    f0, c0 = anticipation_bounds(p_max, p_min, soc, cfg, battery, v=v, margin_scale=0.0)
    assert f1 > c1  # prémisse : contradiction avec marges pleines…
    assert f0 <= c0  # … levée sans marge
    cs = build_candidates(P_now, p_dem, soc, 0.0, cfg, fc, battery, v=v)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.conflict
    assert cs.degenerate
    assert cs.floor == pytest.approx(cs.ceiling, abs=ATOL_W)
    assert f0 - ATOL_W <= cs.floor <= c0 + ATOL_W


# =========================================================================================
# (c) interval_extrema
# =========================================================================================
def test_interval_extrema_rampe_lineaire_bornes_incluses(substeps):
    """Sur une rampe 0..100 W, max/min par intervalle incluent les deux bornes (i·s et (i+1)·s)."""
    p = np.arange(N_PRED, dtype=float)
    p_max, p_min, p_mean = interval_extrema(p, substeps)
    n_int = (N_PRED - 1) // substeps
    assert p_max.shape == p_min.shape == p_mean.shape == (n_int,)
    i = np.arange(n_int)
    np.testing.assert_allclose(p_max, substeps * (i + 1), rtol=0, atol=ATOL_W)
    np.testing.assert_allclose(p_min, substeps * i, rtol=0, atol=ATOL_W)
    # Moyenne sur les `substeps` pas modèle de l'intervalle (ceux que le simulateur applique)
    np.testing.assert_allclose(p_mean, substeps * i + (substeps - 1) / 2.0, rtol=0, atol=ATOL_W)


def test_interval_extrema_pic_interieur_et_pic_frontiere(substeps):
    """Un pic intérieur n'affecte que son intervalle ; un pic en frontière compte des deux côtés."""
    p = constant_demand(10e3)
    p[15] = 50e3  # intérieur de l'intervalle 1 ([10, 20])
    p[40] = -5e3  # frontière entre les intervalles 3 ([30, 40]) et 4 ([40, 50])
    p_max, p_min, _ = interval_extrema(p, substeps)
    assert p_max[1] == 50e3
    assert np.all(np.delete(p_max, 1) == 10e3)
    assert p_min[3] == -5e3
    assert p_min[4] == -5e3
    assert np.all(np.delete(p_min, [3, 4]) == 10e3)


def test_interval_extrema_nombre_intervalles_horizon(cfg, substeps):
    """101 points à 0,1 s donnent Hp/Ts_ems = 10 intervalles EMS."""
    p_max, _, _ = interval_extrema(constant_demand(1.0), substeps)
    assert len(p_max) == round(cfg.ems.Hp / cfg.ems.Ts_ems) == 10


def test_interval_extrema_prediction_courte_padding_bord(substeps):
    """Prédiction plus courte qu'un intervalle : un seul intervalle, bord maintenu, sans erreur."""
    p_max, p_min, p_mean = interval_extrema(np.array([1.0, 2.0, 3.0]), substeps)
    assert p_max.shape == p_min.shape == p_mean.shape == (1,)
    assert p_max[0] == 3.0
    assert p_min[0] == 1.0
    assert p_mean[0] == pytest.approx((1.0 + 2.0 + 3.0 * (substeps - 2)) / substeps, abs=1e-12)


def test_interval_extrema_point_unique_constant(substeps):
    """Un seul échantillon (stratégie sans prédiction) : max = min = moyenne = cette valeur."""
    p_max, p_min, p_mean = interval_extrema(np.array([7.5e3]), substeps)
    assert p_max.tolist() == [7.5e3]
    assert p_min.tolist() == [7.5e3]
    assert p_mean.tolist() == [7.5e3]


def test_interval_extrema_queue_incomplete_ignoree(substeps):
    """Une queue plus courte qu'un intervalle n'ajoute pas d'intervalle partiel."""
    p = np.arange(2 * substeps + 5, dtype=float)
    p_max, _, _ = interval_extrema(p, substeps)
    assert len(p_max) == 2
    assert p_max[-1] == 2 * substeps


# =========================================================================================
# (d) dilate
# =========================================================================================
def test_dilate_max_min_glissant_pm1_longueur_conservee():
    """Dilatation ±1 : chaque case prend le max (resp. min) de ses voisines, longueur inchangée."""
    p_max = np.array([1.0, 5.0, 2.0, 0.0])
    p_min = np.array([1.0, 5.0, 2.0, 0.0])
    hi, lo = dilate(p_max, p_min)
    assert hi.shape == p_max.shape
    assert lo.shape == p_min.shape
    np.testing.assert_array_equal(hi, [5.0, 5.0, 5.0, 2.0])
    np.testing.assert_array_equal(lo, [1.0, 1.0, 0.0, 0.0])


def test_dilate_longueur_1_identite():
    """Un seul intervalle : la dilatation rend les entrées inchangées."""
    hi, lo = dilate(np.array([3.0]), np.array([1.0]))
    np.testing.assert_array_equal(hi, [3.0])
    np.testing.assert_array_equal(lo, [1.0])


def test_dilate_longueur_2_bords_repliques():
    """Deux intervalles : chacun voit l'autre, max et min deviennent uniformes."""
    hi, lo = dilate(np.array([3.0, 1.0]), np.array([1.0, 3.0]))
    np.testing.assert_array_equal(hi, [3.0, 3.0])
    np.testing.assert_array_equal(lo, [1.0, 1.0])


def test_dilate_encadre_toujours_les_entrees():
    """Sur des profils aléatoires (graine fixée), hi ≥ p_max et lo ≤ p_min case par case."""
    rng = np.random.default_rng(1234)
    for n in (2, 3, 10, 17):
        p_max = rng.normal(size=n) * 1e4
        p_min = p_max - rng.uniform(0, 1e4, size=n)
        hi, lo = dilate(p_max, p_min)
        assert np.all(hi >= p_max)
        assert np.all(lo <= p_min)
        # Chaque valeur dilatée est l'une des trois voisines (max/min glissant exact)
        for i in range(n):
            window = slice(max(i - 1, 0), min(i + 2, n))
            assert hi[i] == p_max[window].max()
            assert lo[i] == p_min[window].min()


# =========================================================================================
# (e) anticipation_bounds
# =========================================================================================
def test_anticipation_bounds_demande_faible_plancher_sous_pmin_plafond_large(
    cfg, battery, substeps
):
    """Demande constante 5 kW : plancher ≤ P_min et plafond ≥ demande + P_ch (limite minorée)."""
    p_dem = 5e3
    p_max, p_min, _ = interval_extrema(constant_demand(p_dem), substeps)
    _, p_ch = derated_limits(cfg, battery, SOC_MID)
    floor, ceiling = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery)
    assert floor <= cfg.fuel_cell.P_min_W
    assert ceiling >= p_dem + p_ch - cfg.ems.prediction_margin * p_dem - ATOL_W
    assert ceiling >= p_dem + 0.8 * battery.charge_limit(SOC_MID)  # borne lisible
    # Sans marge de prédiction, le plafond vaut exactement demande + P_ch minoré
    floor0, ceiling0 = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, margin_scale=0.0)
    assert ceiling0 == pytest.approx(p_dem + p_ch, abs=ATOL_W)
    assert floor0 <= cfg.fuel_cell.P_min_W


def test_anticipation_bounds_marche_40kw_plancher_anticipe(cfg, battery, substeps):
    """Marche +40 kW dans 2 s : plancher > demande courante − P_dis, la PàC doit monter avant."""
    p_now, p_after = 5e3, 45e3
    p_max, p_min, _ = interval_extrema(step_demand(p_now, p_after, 2.0, cfg.ems.Ts_model), substeps)
    p_dis, _ = derated_limits(cfg, battery, SOC_MID)
    floor, ceiling = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery)
    assert floor > p_now - p_dis
    assert floor > cfg.fuel_cell.P_min_W
    assert floor < ceiling
    # Valeur attendue : la marche (majorée) moins la décharge minorée, sans marge de rampe
    expected = p_after * (1.0 + cfg.ems.prediction_margin) - p_dis
    assert floor == pytest.approx(expected, abs=ATOL_W)
    # Et strictement au-dessus du plancher d'une demande constante à 5 kW
    floor_flat, _ = anticipation_bounds(
        *interval_extrema(constant_demand(p_now), substeps)[:2], SOC_MID, cfg, battery
    )
    assert floor > floor_flat


def test_anticipation_bounds_marche_lointaine_relachee_par_la_rampe(cfg, battery, substeps):
    """Plus la marche est lointaine, plus le plancher est bas (marge de rampe r·Ts par pas EMS)."""
    floors = []
    for t_step in (2.0, 4.0, 6.0, 8.0):
        p_max, p_min, _ = interval_extrema(
            step_demand(5e3, 45e3, t_step, cfg.ems.Ts_model), substeps
        )
        floors.append(anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery)[0])
    assert np.all(np.diff(floors) < 0)
    # D'un pas EMS à l'autre, le plancher baisse exactement de r_fc · Ts_ems
    r_ts = cfg.fuel_cell.ramp_W_per_s * cfg.ems.Ts_ems
    np.testing.assert_allclose(np.diff(floors), -2 * r_ts, rtol=0, atol=ATOL_W)


@pytest.mark.parametrize(
    ("p_dem", "v"),
    [
        (constant_demand(5e3), 0.0),
        (constant_demand(5e3), 30.0),
        (constant_demand(-12e3), 20.0),
        (step_demand(5e3, 45e3, 2.0, 0.1), 15.0),
        (step_demand(30e3, -20e3, 3.0, 0.1), 25.0),
    ],
    ids=["flat-v0", "flat-v30", "brake-v20", "step-up", "step-down"],
)
def test_anticipation_bounds_margin_scale_zero_moins_conservatif(cfg, battery, substeps, p_dem, v):
    """margin_scale = 0 donne un plancher ≤ et un plafond ≥ ceux obtenus avec margin_scale = 1."""
    p_max, p_min, _ = interval_extrema(p_dem, substeps)
    f1, c1 = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, v=v, margin_scale=1.0)
    f0, c0 = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, v=v, margin_scale=0.0)
    fh, ch = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, v=v, margin_scale=0.5)
    assert f0 <= fh <= f1
    assert c0 >= ch >= c1
    assert np.isfinite([f0, f1, c0, c1]).all()


def test_anticipation_bounds_vitesse_resserre_les_bornes(cfg, battery, substeps):
    """La marge inertielle m·v·σ_a/η relève le plancher et abaisse le plafond quand v croît."""
    p_max, p_min, _ = interval_extrema(constant_demand(20e3), substeps)
    f0, c0 = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, v=0.0)
    f30, c30 = anticipation_bounds(p_max, p_min, SOC_MID, cfg, battery, v=30.0)
    veh, ems = cfg.vehicle, cfg.ems
    inertial = veh.mass_kg * 30.0 * ems.prediction_margin_accel / veh.eta_trans
    assert f30 - f0 == pytest.approx(inertial, abs=ATOL_W)
    assert c0 - c30 == pytest.approx(inertial, abs=ATOL_W)


def test_anticipation_bounds_soc_aux_bornes_limites_nulles(cfg, battery, substeps):
    """À soc_min (P_dis = 0) plancher = demande majorée ; à soc_max (P_ch = 0) plafond minoré."""
    p_dem = 20e3
    p_max, p_min, _ = interval_extrema(constant_demand(p_dem), substeps)
    floor_min, _ = anticipation_bounds(p_max, p_min, cfg.battery.soc_min, cfg, battery)
    assert floor_min == pytest.approx(p_dem * (1.0 + cfg.ems.prediction_margin), abs=ATOL_W)
    _, ceiling_max = anticipation_bounds(p_max, p_min, cfg.battery.soc_max, cfg, battery)
    assert ceiling_max == pytest.approx(p_dem * (1.0 - cfg.ems.prediction_margin), abs=ATOL_W)


# =========================================================================================
# (f) build_candidates : P_bat, validité, SOC aux bornes
# =========================================================================================
def test_build_candidates_p_bat_complement_demande_moyenne(cfg, fc, battery, substeps):
    """P_bat = P_dem_next − P_fc pour chaque candidat, P_dem_next = moyenne du prochain pas EMS."""
    P_now = 20e3
    p_dem = 10e3 + 100.0 * np.arange(N_PRED)  # rampe : la moyenne diffère du premier point
    cs = build_candidates(P_now, p_dem, SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    p_dem_next = float(np.mean(p_dem[:substeps]))
    assert p_dem_next != p_dem[0]
    np.testing.assert_allclose(cs.P_bat, p_dem_next - cs.P_fc, rtol=0, atol=ATOL_W)


def test_build_candidates_valid_equivaut_excess_nul(cfg, fc, battery):
    """valid ⇔ excess ≈ 0 : tous valides (excès nul) en nominal, tous invalides en surcharge."""
    cs_ok = build_candidates(20e3, constant_demand(20e3), SOC_MID, 0.0, cfg, fc, battery)
    assert np.all(cs_ok.valid)
    np.testing.assert_allclose(cs_ok.excess, 0.0, rtol=0, atol=VALID_EPS_W)
    cs_ko = build_candidates(20e3, constant_demand(60e3), SOC_MID, 0.0, cfg, fc, battery)
    assert not np.any(cs_ko.valid)
    assert np.all(cs_ko.excess > VALID_EPS_W)
    for cs in (cs_ok, cs_ko):
        assert np.array_equal(cs.valid, cs.excess <= VALID_EPS_W)


def test_build_candidates_depassement_decharge_invalide(cfg, fc, battery):
    """Demande 60 kW hors de portée : P_bat > discharge_limit(soc), invalide, excès = écart."""
    P_now = 20e3
    cs = build_candidates(P_now, constant_demand(60e3), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    p_dis = battery.discharge_limit(SOC_MID)
    assert np.all(cs.P_bat > p_dis)
    assert not np.any(cs.valid)
    np.testing.assert_allclose(cs.excess, cs.P_bat - p_dis, rtol=0, atol=ATOL_W)


def test_build_candidates_depassement_charge_invalide_freinage_soc_max(cfg, fc, battery):
    """À soc_max en freinage, la batterie n'absorbe rien : excès = charge demandée − friction."""
    P_now, p_dem = 20e3, -10e3
    soc = cfg.battery.soc_max
    cs = build_candidates(P_now, constant_demand(p_dem), soc, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert battery.charge_limit(soc) == 0.0
    brake = max(cfg.vehicle.P_aux_W - p_dem, 0.0)
    assert not np.any(cs.valid)
    np.testing.assert_allclose(cs.excess, -brake - cs.P_bat, rtol=0, atol=ATOL_W)
    assert np.all(cs.excess > 0.0)


def test_build_candidates_rampe_batterie_depassee_invalide(cfg, fc, battery):
    """Un saut de P_bat supérieur à r_bat·Ts rend le candidat invalide (excès = dépassement)."""
    P_now = 20e3
    P_bat_prev = -1e6  # saut irréaliste pour isoler le terme de rampe
    cs = build_candidates(P_now, constant_demand(20e3), SOC_MID, P_bat_prev, cfg, fc, battery)
    r_bat = cfg.battery.ramp_W_per_s * cfg.ems.Ts_ems
    assert not np.any(cs.valid)
    np.testing.assert_allclose(
        cs.excess, np.abs(cs.P_bat - P_bat_prev) - r_bat, rtol=0, atol=ATOL_W
    )
    # Le même pas avec P_bat_prev cohérent est entièrement valide
    cs_ok = build_candidates(P_now, constant_demand(20e3), SOC_MID, 0.0, cfg, fc, battery)
    assert np.all(cs_ok.valid)


@pytest.mark.parametrize("p_dem", [20e3, 0.0, -10e3], ids=["20k", "0", "brake"])
@pytest.mark.parametrize("soc_name", ["soc_min", "soc_max"])
def test_build_candidates_soc_aux_bornes_sans_exception(cfg, fc, battery, soc_name, p_dem):
    """soc = soc_min ou soc_max (une limite nulle) : aucune exception, 9 candidats finis."""
    soc = getattr(cfg.battery, soc_name)
    P_now = 20e3
    cs = build_candidates(P_now, constant_demand(p_dem), soc, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert isinstance(cs.degenerate, bool)
    assert isinstance(cs.conflict, bool)


def test_build_candidates_valides_respectent_limites_batterie_courantes(cfg, fc, battery, substeps):
    """Tout candidat valide a P_bat dans [−P_ch(soc) − friction, P_dis(soc)] au SOC courant."""
    scenarios = [
        (20e3, constant_demand(20e3), SOC_MID),
        (3e3, constant_demand(5e3), SOC_MID),
        (20e3, step_demand(5e3, 45e3, 2.0, cfg.ems.Ts_model), SOC_MID),
        (20e3, constant_demand(-10e3), cfg.battery.soc_min),
        (20e3, constant_demand(20e3), cfg.battery.soc_min + 0.01),
        (20e3, constant_demand(20e3), cfg.battery.soc_max - 0.01),
    ]
    n_valid = 0
    for P_now, p_dem, soc in scenarios:
        cs = build_candidates(P_now, p_dem, soc, 0.0, cfg, fc, battery)
        assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
        p_next = float(np.mean(p_dem[:substeps]))
        brake = max(cfg.vehicle.P_aux_W - p_next, 0.0)
        ok = cs.valid
        n_valid += int(ok.sum())
        assert np.all(cs.P_bat[ok] <= battery.discharge_limit(soc) + ATOL_W)
        assert np.all(cs.P_bat[ok] >= -battery.charge_limit(soc) - brake - ATOL_W)
    assert n_valid > 0


def test_build_candidates_demande_scalaire_sans_prediction(cfg, fc, battery):
    """Un seul point de demande (stratégie sans prédiction) est accepté : P_bat = P_dem − P_fc."""
    P_now, p_dem = 20e3, 25e3
    cs = build_candidates(P_now, np.array([p_dem]), SOC_MID, 0.0, cfg, fc, battery)
    assert_well_formed(cs, fc, P_now, cfg.ems.Ts_ems)
    assert not cs.degenerate
    np.testing.assert_allclose(cs.P_bat, p_dem - cs.P_fc, rtol=0, atol=ATOL_W)


def test_build_candidates_candidate_set_immuable(cfg, fc, battery):
    """CandidateSet est un dataclass gelé : la réaffectation d'un champ lève FrozenInstanceError."""
    cs = build_candidates(20e3, constant_demand(20e3), SOC_MID, 0.0, cfg, fc, battery)
    with pytest.raises(AttributeError):
        cs.P_lo = 0.0  # type: ignore[misc]


# =========================================================================================
# (g) Monotonie des limites batterie vs SOC (section 7)
# =========================================================================================
def test_battery_discharge_limit_v_croissante_soc(cfg, battery):
    """discharge_limit_v est croissante en SOC, nulle sous soc_min, pleine au-delà de la bande."""
    soc = np.linspace(0.0, 1.0, 401)
    lim = battery.discharge_limit_v(soc)
    assert lim.shape == soc.shape
    assert np.all(np.diff(lim) >= -ATOL_W)
    assert np.all(lim >= 0.0)
    bp = cfg.battery
    assert np.all(lim[soc <= bp.soc_min + 1e-12] == 0.0)
    assert np.all(lim[soc >= bp.soc_min + bp.derate_band - 1e-12] == bp.P_dis_max_W)
    assert lim.max() == bp.P_dis_max_W


def test_battery_charge_limit_v_decroissante_soc(cfg, battery):
    """charge_limit_v est décroissante en SOC, nulle au-dessus de soc_max, pleine sous la bande."""
    soc = np.linspace(0.0, 1.0, 401)
    lim = battery.charge_limit_v(soc)
    assert lim.shape == soc.shape
    assert np.all(np.diff(lim) <= ATOL_W)
    assert np.all(lim >= 0.0)
    bp = cfg.battery
    assert np.all(lim[soc >= bp.soc_max - 1e-12] == 0.0)
    assert np.all(lim[soc <= bp.soc_max - bp.derate_band + 1e-12] == bp.P_ch_max_W)
    assert lim.max() == bp.P_ch_max_W


def test_battery_limites_scalaires_et_vectorielles_coherentes(battery):
    """Les versions scalaires et vectorielles des limites coïncident sur toute la plage de SOC."""
    soc = np.linspace(0.0, 1.0, 101)
    dis_scalar = np.array([battery.discharge_limit(float(s)) for s in soc])
    ch_scalar = np.array([battery.charge_limit(float(s)) for s in soc])
    np.testing.assert_allclose(battery.discharge_limit_v(soc), dis_scalar, rtol=0, atol=ATOL_W)
    np.testing.assert_allclose(battery.charge_limit_v(soc), ch_scalar, rtol=0, atol=ATOL_W)


def test_battery_limites_nulles_aux_bornes_de_la_fenetre(cfg, battery):
    """discharge_limit(soc_min) = 0 et charge_limit(soc_max) = 0 : le SOC reste dans la fenêtre."""
    bp = cfg.battery
    assert battery.discharge_limit(bp.soc_min) == 0.0
    assert battery.charge_limit(bp.soc_max) == 0.0
    # À mi-bande, exactement la moitié de la puissance max (rampe linéaire)
    assert battery.discharge_limit(bp.soc_min + bp.derate_band / 2) == pytest.approx(
        bp.P_dis_max_W / 2, abs=ATOL_W
    )
    assert battery.charge_limit(bp.soc_max - bp.derate_band / 2) == pytest.approx(
        bp.P_ch_max_W / 2, abs=ATOL_W
    )
