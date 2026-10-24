"""Tests du simulateur à deux cadences (``simulator.py``) et du bilan d'énergie (``metrics``).

Repères : WLTC = 1800 s, ``Ts_model`` = 0,1 s, ``Ts_ems`` = 1 s (section 5 de la passation).
``ecms_const`` est pris comme dans la campagne M1 (``examples/compare_strategies.py``) : λ
constant calibré hors ligne **avec prédiction exacte** ; la valeur calibrée est figée ici
(``LAM_ECMS_WLTC``) et un test lent vérifie qu'elle reste celle de ``calibrate_ecms_const``.
Les fixtures sont locales au fichier ; les résultats mis en cache ne sont jamais modifiés.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from iems.config import IEMSConfig
from iems.metrics import compute_metrics
from iems.models.battery import Battery
from iems.models.cycles import Cycle, get_cycle, list_cycles, synthetic_cycle
from iems.patterns import DrivingPattern
from iems.simulator import SimResult, run, simulate
from iems.strategies import ECMSConst, Strategy, calibrate_ecms_const, make_strategy

# --- Tolérances explicites -------------------------------------------------------------
TOL_ENERGY_M1 = 1e-2  # critère M1 ① : bilan d'énergie bouclé à < 1 %
TOL_ENERGY_NUM = 1e-6  # précision attendue de l'intégrateur (observé ≈ 1e-14)
TOL_BUS_W = 1e-3  # résidu du bus DC (W)
TOL_RAMP_W = 1e-6  # marge sur la rampe PàC appliquée (W par pas modèle)
TOL_POWER_W = 1e-9  # comparaisons de puissances (W)
TOL_SOC = 1e-9  # fenêtre SOC
TOL_TIME_S = 1e-9  # grille temporelle
RUNTIME_MAX_S = 5.0  # section 7 : un WLTC simulé en < 5 s

# λ* de calibrate_ecms_const(WLTC, prédiction exacte) ≈ 1,510 (ΔSOC ≈ +0,1 pt, 0 violation).
LAM_ECMS_WLTC = 1.51
STRATEGY_NAMES = ("rules_baseline", "ecms_const", "iems")
ECMS_LAMBDA0 = "ecms_const_lambda0"  # ECMS constant avec le λ0 par défaut (non calibré)
CADENCES = [(0.1, 1.0), (0.2, 1.0), (0.1, 4.0), (0.5, 4.0)]  # (Ts_model, Ts_ems)


# --- Aides locales -----------------------------------------------------------------------
def _exact_cfg(**ems_overrides: float) -> IEMSConfig:
    """Config par défaut avec prédicteur exact (bruit et étirement nuls), surcharges EMS en plus."""
    return IEMSConfig().with_updates(
        ems={"predictor_noise": 0.0, "predictor_time_warp": 0.0, **ems_overrides}
    )


def _campaign_strategy(name: str, cfg: IEMSConfig) -> Strategy:
    """Stratégie ``name`` sur ``cfg`` ; ``ecms_const`` reçoit le λ calibré de la campagne."""
    if name == "ecms_const":
        return ECMSConst(cfg, LAM_ECMS_WLTC)
    if name == ECMS_LAMBDA0:
        return ECMSConst(cfg)
    return make_strategy(name, cfg)


def _campaign_setup(name: str) -> tuple[IEMSConfig, Strategy]:
    """Config et stratégie comme dans la campagne M1 (ECMS calibré avec prédiction exacte)."""
    cfg = _exact_cfg() if name == "ecms_const" else IEMSConfig()
    return cfg, _campaign_strategy(name, cfg)


def _braking_cycle() -> Cycle:
    """Cycle 1 Hz de 45 s : accélération, palier à 25 m/s, freinage à −2,5 m/s², arrêt."""
    v = np.concatenate(
        [np.linspace(0.0, 25.0, 21), np.full(5, 25.0), np.linspace(25.0, 0.0, 11)[1:], np.zeros(10)]
    )
    return Cycle("freinage", np.arange(len(v), dtype=float), v)


def _sport_cycle() -> Cycle:
    """Cycle synthétique court (≈ 105 s) de conduite sportive : freinages appuyés."""
    return synthetic_cycle([(DrivingPattern.SPORTIF, 90)], seed=0, name="sport_court")


def _interval_index(res: SimResult) -> np.ndarray:
    """Indice du pas EMS auquel appartient chaque pas modèle."""
    return np.arange(len(res.P_fc)) // res.cfg.ems.substeps


class ExactSlicePredictor:
    """Prédicteur minimal respectant le Protocol : tranche exacte du cycle rééchantillonné."""

    def __init__(self, v: np.ndarray, dt: float, Hp: float) -> None:
        self._v = np.asarray(v, dtype=float)
        self._n_h = round(Hp / dt)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict(self, i: int) -> np.ndarray:
        future = self._v[i : i + self._n_h + 1]
        if len(future) < self._n_h + 1:
            future = np.pad(future, (0, self._n_h + 1 - len(future)), mode="edge")
        return future


# --- Fixtures ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wltc() -> Cycle:
    return get_cycle("wltc")


@pytest.fixture(scope="module")
def wltc_result(wltc: Cycle) -> Callable[[str], SimResult]:
    """Fabrique avec cache des résultats WLTC par stratégie (résultats jamais modifiés)."""
    cache: dict[str, SimResult] = {}

    def get(name: str) -> SimResult:
        if name not in cache:
            cfg, strategy = _campaign_setup(name)
            cache[name] = simulate(wltc, strategy, cfg)
        return cache[name]

    return get


@pytest.fixture(scope="module")
def braking_results() -> dict[tuple[float, float], SimResult]:
    """iEMS sur le cycle de freinage pour plusieurs cadences (Ts_model, Ts_ems)."""
    cycle = _braking_cycle()
    out: dict[tuple[float, float], SimResult] = {}
    for ts_model, ts_ems in CADENCES:
        cfg = IEMSConfig().with_updates(ems={"Ts_model": ts_model, "Ts_ems": ts_ems})
        out[(ts_model, ts_ems)] = simulate(cycle, make_strategy("iems", cfg), cfg)
    return out


@pytest.fixture(scope="module")
def regen_results() -> dict[str, SimResult]:
    """Les 3 stratégies sur le cycle sportif court avec un SOC initial proche de soc_max."""
    cycle = _sport_cycle()
    cfg = IEMSConfig(soc0=0.79)
    return {name: simulate(cycle, make_strategy(name, cfg), cfg) for name in STRATEGY_NAMES}


# --- (a) Conservation d'énergie ----------------------------------------------------------
@pytest.mark.parametrize("name", [*STRATEGY_NAMES, ECMS_LAMBDA0])
def test_bilan_energie_wltc_boucle(name, wltc_result):
    """Le bilan d'énergie boucle à < 1 % (M1 ①) et à la précision numérique sur WLTC."""
    res = wltc_result(name)
    energy = compute_metrics(res).energy
    assert energy.rel_error < TOL_ENERGY_M1
    assert energy.rel_error < TOL_ENERGY_NUM
    assert res.max_bus_residual_W < TOL_BUS_W


@pytest.mark.parametrize("name", [*STRATEGY_NAMES, ECMS_LAMBDA0])
def test_bilan_bus_journaux_coherents(name, wltc_result):
    """P_fc + P_bat = P_dem + P_friction + P_spill − P_unmet à chaque pas modèle journalisé."""
    res = wltc_result(name)
    lhs = res.P_fc + res.P_bat
    rhs = res.P_dem[:-1] + res.P_friction + res.P_spill - res.P_unmet
    assert float(np.max(np.abs(lhs - rhs))) < TOL_BUS_W
    assert np.all(res.P_friction >= 0.0)
    assert np.all(res.P_spill >= 0.0)
    assert np.all(res.P_unmet >= 0.0)


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_bilan_energie_termes_coherents(name, wltc_result):
    """Les termes d'EnergyBalance ont le sens attendu (stock, PàC, pertes batterie, PCI)."""
    res = wltc_result(name)
    energy = compute_metrics(res).energy
    cfg = res.cfg
    assert energy.E_fc == pytest.approx(float(np.sum(res.P_fc)) * res.dt, rel=1e-12)
    assert energy.E_storage == pytest.approx((cfg.soc0 - res.soc_final) * cfg.battery.E_cap_J)
    assert energy.E_bat_losses >= 0.0
    assert 0.0 < energy.E_fc < energy.E_h2_chem  # η_fc < 1
    sources = energy.E_fc + energy.E_storage
    uses = energy.E_demand_served + energy.E_friction + energy.E_spill + energy.E_bat_losses
    scale = float(np.sum(np.abs(res.P_dem[:-1]))) * res.dt
    assert abs(sources - uses) / scale == pytest.approx(energy.rel_error, abs=1e-15)


# --- (b) Deux cadences -------------------------------------------------------------------
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_grille_ems_wltc_longueur_et_multiples(name, wltc_result):
    """len(t_ems) = ceil(N / substeps), t_ems multiples de Ts_ems, journaux aux bonnes tailles."""
    res = wltc_result(name)
    ems = res.cfg.ems
    n_model = len(res.P_fc)
    n_ems = len(res.t_ems)
    assert n_ems == math.ceil(n_model / ems.substeps)
    assert len(res.t) == len(res.soc) == len(res.v) == len(res.P_dem) == n_model + 1
    assert len(res.FC_Power_cmd) == len(res.BAT_Power_cmd) == len(res.P_fc_at_ems) == n_ems
    ratio = res.t_ems / ems.Ts_ems
    assert float(np.max(np.abs(ratio - np.round(ratio)))) < TOL_TIME_S
    assert np.allclose(np.diff(res.t_ems), ems.Ts_ems, atol=TOL_TIME_S)
    assert np.allclose(res.t_ems, res.t[:: ems.substeps][:n_ems], atol=TOL_TIME_S)


@pytest.mark.parametrize("cadence", CADENCES, ids=lambda c: f"Ts_model={c[0]}_Ts_ems={c[1]}")
def test_grille_ems_cadences_variees_ceil(cadence, braking_results):
    """Pour plusieurs cadences, len(t_ems) = ceil(N / substeps) et la grille EMS est régulière."""
    res = braking_results[cadence]
    ems = res.cfg.ems
    assert ems.substeps == round(cadence[1] / cadence[0])
    n_model = len(res.P_fc)
    assert len(res.t_ems) == math.ceil(n_model / ems.substeps)
    assert np.allclose(res.t_ems, np.arange(len(res.t_ems)) * ems.Ts_ems, atol=TOL_TIME_S)


def test_dernier_intervalle_partiel_compte_un_pas_ems(braking_results):
    """Un dernier intervalle EMS incomplet (N non multiple de substeps) est bien simulé."""
    res = braking_results[(0.1, 4.0)]
    n_sub = res.cfg.ems.substeps
    n_model = len(res.P_fc)
    assert n_model % n_sub != 0  # 45 s / 0,1 s = 450 pas, substeps = 40 → reste 10
    assert len(res.t_ems) == n_model // n_sub + 1
    assert res.t_ems[-1] == pytest.approx((len(res.t_ems) - 1) * res.cfg.ems.Ts_ems)
    tail = res.P_fc[(len(res.t_ems) - 1) * n_sub :]
    assert len(tail) == n_model % n_sub
    assert np.all(np.isfinite(tail))
    assert np.all(np.isfinite(res.soc))


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_consigne_bloquee_entre_deux_pas_ems(name, wltc_result):
    """Dans chaque intervalle EMS, la PàC progresse vers FC_Power_cmd[k] sans jamais la dépasser."""
    res = wltc_result(name)
    k_of_i = _interval_index(res)
    start = res.P_fc_at_ems[k_of_i]
    target = res.FC_Power_cmd[k_of_i]
    direction = np.sign(target - start)
    assert np.all((res.P_fc - start) * direction >= -TOL_POWER_W)  # va vers la consigne
    assert np.all((target - res.P_fc) * direction >= -TOL_POWER_W)  # sans la dépasser
    dist = np.abs(target - res.P_fc)
    first = np.arange(len(res.P_fc)) % res.cfg.ems.substeps == 0
    prev = np.where(first, np.abs(target - start), np.roll(dist, 1))
    assert np.all(dist <= prev + TOL_POWER_W)  # distance à la consigne non croissante


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_p_fc_at_ems_est_la_puissance_appliquee_a_la_decision(name, wltc_result):
    """P_fc_at_ems[k] est la PàC appliquée juste avant le pas EMS k (P_min au départ)."""
    res = wltc_result(name)
    n_sub = res.cfg.ems.substeps
    n_ems = len(res.t_ems)
    assert res.P_fc_at_ems[0] == res.cfg.fuel_cell.P_min_W
    assert np.array_equal(res.P_fc_at_ems[1:], res.P_fc[n_sub - 1 : (n_ems - 1) * n_sub : n_sub])


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_rampe_pac_respectee_a_chaque_pas_modele(name, wltc_result):
    """|ΔP_fc| ≤ r_fc·Ts_model + 1e-6 à chaque pas modèle, premier pas compris (départ à P_min)."""
    res = wltc_result(name)
    fc = res.cfg.fuel_cell
    p_full = np.concatenate(([fc.P_min_W], res.P_fc))
    assert float(np.max(np.abs(np.diff(p_full)))) <= fc.ramp_W_per_s * res.dt + TOL_RAMP_W
    assert compute_metrics(res).violations.fc_ramp == 0


@pytest.mark.parametrize("cadence", CADENCES, ids=lambda c: f"Ts_model={c[0]}_Ts_ems={c[1]}")
def test_rampe_pac_respectee_cadences_variees(cadence, braking_results):
    """La rampe PàC appliquée suit Ts_model quelle que soit la cadence choisie."""
    res = braking_results[cadence]
    fc = res.cfg.fuel_cell
    p_full = np.concatenate(([fc.P_min_W], res.P_fc))
    assert float(np.max(np.abs(np.diff(p_full)))) <= fc.ramp_W_per_s * cadence[0] + TOL_RAMP_W


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_p_fc_dans_bornes_statiques(name, wltc_result):
    """La PàC appliquée reste dans [P_min, P_max] à chaque pas modèle."""
    res = wltc_result(name)
    fc = res.cfg.fuel_cell
    assert float(res.P_fc.min()) >= fc.P_min_W - TOL_POWER_W
    assert float(res.P_fc.max()) <= fc.P_max_W + TOL_POWER_W
    assert compute_metrics(res).violations.fc_bounds == 0


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_consigne_atteignable_en_un_pas_ems(name, wltc_result):
    """La consigne EMS reste à portée de rampe sur Ts_ems depuis la PàC appliquée."""
    res = wltc_result(name)
    fc, ems = res.cfg.fuel_cell, res.cfg.ems
    gap = np.abs(res.FC_Power_cmd - res.P_fc_at_ems)
    assert float(gap.max()) <= fc.ramp_W_per_s * ems.Ts_ems + TOL_POWER_W
    assert compute_metrics(res).violations.cmd_ramp == 0


# --- (c) Fenêtre SOC ---------------------------------------------------------------------
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_soc_reste_dans_fenetre_utile_wltc(name, wltc_result):
    """Le SOC reste dans [soc_min, soc_max] à chaque pas modèle sur WLTC."""
    res = wltc_result(name)
    bat = res.cfg.battery
    assert float(res.soc.min()) >= bat.soc_min - TOL_SOC
    assert float(res.soc.max()) <= bat.soc_max + TOL_SOC
    assert res.soc[0] == res.cfg.soc0
    assert compute_metrics(res).violations.soc_window == 0


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_campagne_wltc_sans_violation(name, wltc_result):
    """Comme dans la campagne M1 (critère ②), aucune violation de bornes sur WLTC."""
    assert compute_metrics(wltc_result(name)).violations.total == 0


# --- (d) Déterminisme --------------------------------------------------------------------
def test_deux_simulations_identiques_strictement_egales(wltc):
    """Deux simulate identiques (prédicteur bruité à graine fixe) donnent des tableaux égaux."""
    cfg = IEMSConfig()
    res_a = simulate(wltc, make_strategy("iems", cfg), cfg)
    res_b = simulate(wltc, make_strategy("iems", cfg), cfg)
    assert np.array_equal(res_a.soc, res_b.soc)
    assert np.array_equal(res_a.P_fc, res_b.P_fc)
    assert np.array_equal(res_a.FC_Power_cmd, res_b.FC_Power_cmd)
    assert np.array_equal(res_a.BAT_Power_cmd, res_b.BAT_Power_cmd)
    assert np.array_equal(res_a.ems_info["lam"], res_b.ems_info["lam"])


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_reutilisation_instance_strategie_resultats_identiques(name):
    """Réutiliser la même instance de stratégie (reset par simulate) redonne les mêmes tableaux."""
    cycle = _braking_cycle()
    cfg = IEMSConfig()
    strategy = make_strategy(name, cfg)
    res_a = simulate(cycle, strategy, cfg)
    res_b = simulate(cycle, strategy, cfg)
    assert np.array_equal(res_a.soc, res_b.soc)
    assert np.array_equal(res_a.P_fc, res_b.P_fc)
    assert np.array_equal(res_a.FC_Power_cmd, res_b.FC_Power_cmd)


# --- (e) Performance ---------------------------------------------------------------------
def test_run_wltc_iems_moins_de_5_s():
    """Un WLTC complet avec l'iEMS se simule en moins de 5 s (section 7)."""
    res = run("wltc", "iems")
    assert res.runtime_s < RUNTIME_MAX_S
    assert res.strategy == "iems"
    assert res.cycle.name == "wltc"
    assert res.cycle.duration == pytest.approx(1800.0)


def test_run_nom_inconnu_leve_keyerror():
    """run() refuse un cycle ou une stratégie hors catalogue avec KeyError."""
    with pytest.raises(KeyError):
        run("cycle_inconnu", "iems")
    with pytest.raises(KeyError):
        run("wltc", "strategie_inconnue")


# --- (f) Freinage régénératif (piège 8.3) ------------------------------------------------
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_freinage_fort_excedent_en_friction_jamais_en_soc(name, regen_results):
    """Près de soc_max, l'excédent de freinage part en friction et le SOC ne dépasse pas soc_max."""
    res = regen_results[name]
    cfg = res.cfg
    assert cfg.soc0 == 0.79
    assert np.any(res.P_dem < 0.0)  # le cycle freine vraiment
    assert np.any(res.P_friction > 0.0)
    assert float(res.soc.max()) <= cfg.battery.soc_max + TOL_SOC
    regen_available = np.maximum(cfg.vehicle.P_aux_W - res.P_dem[:-1], 0.0)
    assert np.all(res.P_friction <= regen_available + TOL_POWER_W)
    assert np.all((res.P_friction == 0.0) | (res.P_dem[:-1] < cfg.vehicle.P_aux_W))


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_freinage_fort_limite_de_charge_batterie_respectee(name, regen_results):
    """En freinage, la puissance de charge batterie ne dépasse jamais sa limite au SOC courant."""
    res = regen_results[name]
    bat = Battery(res.cfg.battery)
    assert np.all(res.P_bat >= -bat.charge_limit_v(res.soc[:-1]) - TOL_POWER_W)
    assert np.all(res.P_bat <= bat.discharge_limit_v(res.soc[:-1]) + TOL_POWER_W)


def test_freinage_fort_cycle_manuel_friction_et_bilan(braking_results):
    """Sur le cycle de freinage à −2,5 m/s², friction > 0 et bilan d'énergie bouclé."""
    res = braking_results[(0.1, 1.0)]
    assert float(res.P_dem.min()) < -50e3  # freinage largement au-delà des 30 kW batterie
    assert np.any(res.P_friction > 0.0)
    assert float(res.soc.max()) <= res.cfg.battery.soc_max + TOL_SOC
    assert compute_metrics(res).energy.rel_error < TOL_ENERGY_NUM
    assert res.max_bus_residual_W < TOL_BUS_W


# --- (g) Traction servie, surplus absorbé ------------------------------------------------
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_p_unmet_et_p_spill_nuls_wltc(name, wltc_result):
    """Sur WLTC, aucune traction non servie ni surplus PàC non absorbé pour les 3 stratégies."""
    res = wltc_result(name)
    assert np.all(res.P_unmet == 0.0)
    assert np.all(res.P_spill == 0.0)
    viol = compute_metrics(res).violations
    assert viol.bat_discharge == 0
    assert viol.bat_charge == 0


def test_ecms_lambda0_defaut_sans_spill_wltc(wltc_result):
    """Avec le λ0 par défaut (= λ* calibré sur WLTC), l'ECMS constant reste sans P_spill.

    Avec un λ0 trop haut (1,55), le SOC saturait à soc_max et le ralenti de la pile
    (P_min = 1 kW > P_aux) n'était plus absorbable à l'arrêt : 311 pas de P_spill.
    """
    res = wltc_result(ECMS_LAMBDA0)
    assert np.all(res.P_spill == 0.0)


# --- (h) Journaux EMS --------------------------------------------------------------------
def test_ems_info_iems_contient_lam_soc_target_pattern(wltc_result):
    """ems_info de l'iEMS contient lam, soc_target et pattern, de longueur n_ems et sans NaN."""
    res = wltc_result("iems")
    n_ems = len(res.t_ems)
    ems, bat = res.cfg.ems, res.cfg.battery
    for key in ("lam", "soc_target", "pattern"):
        assert key in res.ems_info
        assert res.ems_info[key].shape == (n_ems,)
        assert not np.any(np.isnan(res.ems_info[key]))
    assert np.all((res.ems_info["lam"] >= ems.lambda_min) & (res.ems_info["lam"] <= ems.lambda_max))
    soc_target = res.ems_info["soc_target"]
    assert np.all((soc_target >= bat.soc_min - TOL_SOC) & (soc_target <= bat.soc_max + TOL_SOC))
    patterns = set(np.unique(res.ems_info["pattern"]).astype(int).tolist())
    assert patterns <= {int(p) for p in DrivingPattern}


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_ems_info_toutes_les_series_ont_la_longueur_n_ems(name, wltc_result):
    """Chaque série de ems_info est un tableau float de longueur n_ems."""
    res = wltc_result(name)
    assert res.ems_info  # au moins les bornes candidates
    for key, arr in res.ems_info.items():
        assert isinstance(arr, np.ndarray), key
        assert arr.dtype == np.float64, key
        assert arr.shape == (len(res.t_ems),), key


def test_simresult_proprietes_soc_final_et_dt(wltc_result):
    """soc_final et dt de SimResult reflètent le dernier SOC et Ts_model."""
    res = wltc_result("iems")
    assert res.soc_final == float(res.soc[-1])
    assert res.dt == res.cfg.ems.Ts_model
    assert np.allclose(np.diff(res.t), res.dt, atol=TOL_TIME_S)


# --- (i) Prédicteur personnalisé ---------------------------------------------------------
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_predicteur_personnalise_identique_a_oracle_exact(name, wltc):
    """Un SpeedPredictor minimal (tranche exacte) reproduit exactement l'oracle non bruité."""
    cfg = _exact_cfg()
    dt, hp = cfg.ems.Ts_model, cfg.ems.Hp
    predictor = ExactSlicePredictor(wltc.resample(dt).v, dt, hp)
    res_ref = simulate(wltc, _campaign_strategy(name, cfg), cfg)
    res_custom = simulate(wltc, _campaign_strategy(name, cfg), cfg, predictor=predictor)
    assert predictor.reset_calls == 1
    assert np.array_equal(res_ref.soc, res_custom.soc)
    assert np.array_equal(res_ref.P_fc, res_custom.P_fc)
    assert np.array_equal(res_ref.FC_Power_cmd, res_custom.FC_Power_cmd)
    assert np.array_equal(res_ref.BAT_Power_cmd, res_custom.BAT_Power_cmd)


# --- Tests lents (> 2 s) -----------------------------------------------------------------
@pytest.mark.slow
def test_ecms_calibre_wltc_lambda_fige_a_jour(wltc):
    """calibrate_ecms_const (prédiction exacte) redonne LAM_ECMS_WLTC, sans dérive ni violation."""
    cfg = _exact_cfg()
    cal = calibrate_ecms_const(wltc, cfg)
    assert abs(cal.lam - LAM_ECMS_WLTC) <= 0.02
    assert abs(cal.delta_soc) <= 0.005  # charge-sustaining à 0,5 pt près
    metrics = compute_metrics(cal.result)
    assert metrics.violations.total == 0
    assert metrics.energy.rel_error < TOL_ENERGY_NUM
    assert np.all(cal.result.P_spill == 0.0)
    assert np.all(cal.result.P_unmet == 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("cycle_name", list_cycles())
def test_bilan_energie_tous_cycles_boucle(cycle_name):
    """Le bilan d'énergie boucle à la précision numérique sur les 5 cycles × 3 stratégies."""
    cycle = get_cycle(cycle_name)
    for name in STRATEGY_NAMES:
        cfg, strategy = _campaign_setup(name)
        res = simulate(cycle, strategy, cfg)
        energy = compute_metrics(res).energy
        assert energy.rel_error < TOL_ENERGY_NUM, (cycle_name, name)
        assert res.max_bus_residual_W < TOL_BUS_W, (cycle_name, name)
        assert float(res.soc.max()) <= cfg.battery.soc_max + TOL_SOC, (cycle_name, name)
        assert float(res.soc.min()) >= cfg.battery.soc_min - TOL_SOC, (cycle_name, name)
