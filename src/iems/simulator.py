"""Simulateur à pas fixe, deux cadences.

* Le **modèle** tourne à ``Ts_model`` : la PàC suit la consigne avec sa rampe, la batterie
  boucle le bus (``P_bat = P_dem − P_fc``) dans ses limites, le SOC est intégré.
* L'**EMS** tourne à ``Ts_ems`` (multiple entier de ``Ts_model``). La consigne
  ``FC_Power_cmd`` est **bloquée** (bloqueur d'ordre 0) entre deux pas EMS : c'est la
  transition de cadence explicite vers les commandes.

Bus DC, à chaque pas modèle (vérifié à 1e-6 W près) :
``P_fc + P_bat = P_dem + P_friction + P_spill − P_unmet`` où

* ``P_friction`` : régénération déviée vers les freins mécaniques quand la batterie est à sa
  limite de charge (piège n° 3 : l'excédent de freinage part en friction, pas en SOC) ;
* ``P_spill`` : surplus PàC que la batterie ne peut pas absorber (violation) ;
* ``P_unmet`` : traction non servie car la batterie est à sa limite de décharge (violation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iems.config import IEMSConfig
from iems.dpt.predictor import NoisyOraclePredictor, SpeedPredictor
from iems.models.battery import Battery
from iems.models.cycles import Cycle
from iems.models.fuel_cell import FuelCell
from iems.models.vehicle import power_demand
from iems.strategies import EMSObservation, Strategy

FloatArray = NDArray[np.float64]
_BUS_TOL_W = 1e-6


@dataclass
class SimResult:
    """Journaux d'une simulation (pas modèle et pas EMS)."""

    cycle: Cycle
    strategy: str
    cfg: IEMSConfig
    # pas modèle (N+1 points pour t, v, P_dem, soc ; N pas pour les puissances appliquées)
    t: FloatArray
    v: FloatArray
    P_dem: FloatArray
    P_fc: FloatArray
    P_bat: FloatArray
    P_friction: FloatArray
    P_spill: FloatArray
    P_unmet: FloatArray
    soc: FloatArray
    m_h2: FloatArray  # masse d'H2 consommée cumulée (kg), N+1 points
    # pas EMS
    t_ems: FloatArray
    FC_Power_cmd: FloatArray
    BAT_Power_cmd: FloatArray
    P_fc_at_ems: FloatArray  # puissance PàC appliquée au moment de la décision
    ems_info: dict[str, FloatArray] = field(default_factory=dict)
    runtime_s: float = 0.0
    max_bus_residual_W: float = 0.0

    @property
    def soc_final(self) -> float:
        return float(self.soc[-1])

    @property
    def dt(self) -> float:
        return self.cfg.ems.Ts_model


def _demand_from_speed(v: FloatArray, dt: float, cfg: IEMSConfig) -> FloatArray:
    t = np.arange(len(v)) * dt
    return power_demand(t, v, cfg.vehicle).P_dem


def simulate(
    cycle: Cycle,
    strategy: Strategy,
    cfg: IEMSConfig | None = None,
    predictor: SpeedPredictor | None = None,
) -> SimResult:
    """Simule ``strategy`` sur ``cycle`` et renvoie les journaux complets."""
    cfg = cfg or IEMSConfig()
    ems = cfg.ems
    dt, n_sub = ems.Ts_model, ems.substeps
    fc, bat = FuelCell(cfg.fuel_cell), Battery(cfg.battery)
    r_fc_dt = cfg.fuel_cell.ramp_W_per_s * dt
    P_aux = cfg.vehicle.P_aux_W

    c = cycle.resample(dt)
    dem = power_demand(c.t, c.v, cfg.vehicle)
    N = len(c.t) - 1
    if N < 1:
        raise ValueError("cycle trop court : au moins deux échantillons après rééchantillonnage")
    n_ems = -(-N // n_sub)
    if predictor is None:
        predictor = NoisyOraclePredictor(
            c.v, dt, ems.Hp, ems.predictor_noise, ems.seed, ems.predictor_time_warp
        )
    predictor.reset()
    strategy.reset()

    P_fc_log = np.zeros(N)
    P_bat_log = np.zeros(N)
    fr_log = np.zeros(N)
    spill_log = np.zeros(N)
    unmet_log = np.zeros(N)
    soc_log = np.zeros(N + 1)
    h2_log = np.zeros(N + 1)
    lhv = cfg.fuel_cell.lhv_J_per_kg
    t_ems = np.zeros(n_ems)
    fc_cmd = np.zeros(n_ems)
    bat_cmd = np.zeros(n_ems)
    fc_at = np.zeros(n_ems)
    info_log: dict[str, list[float]] = {}

    soc = cfg.soc0
    soc_log[0] = soc
    P_fc = cfg.fuel_cell.P_min_W
    P_bat = 0.0
    v_hist = np.zeros(n_ems)
    max_res = 0.0
    tic = time.perf_counter()

    for k in range(n_ems):
        i0 = k * n_sub
        v_hist[k] = c.v[i0]
        obs = EMSObservation(
            k=k,
            t=float(c.t[i0]),
            v=float(c.v[i0]),
            soc=soc,
            P_fc=P_fc,
            P_bat=P_bat,
            P_dem=float(dem.P_dem[i0]),
            P_dem_pred=_demand_from_speed(predictor.predict(i0), dt, cfg),
            v_hist=v_hist[: k + 1],
        )
        cmd = strategy.step(obs)
        t_ems[k], fc_cmd[k], bat_cmd[k], fc_at[k] = (
            c.t[i0],
            cmd.FC_Power_cmd,
            cmd.BAT_Power_cmd,
            P_fc,
        )
        for key, val in cmd.info.items():
            info_log.setdefault(key, [np.nan] * k).append(float(val))
        for lst in info_log.values():  # clés absentes à ce pas
            if len(lst) < k + 1:
                lst.append(np.nan)

        target = cmd.FC_Power_cmd
        for i in range(i0, min(i0 + n_sub, N)):
            # PàC : suit la consigne bloquée, rampe et bornes physiques
            P_fc += min(max(target - P_fc, -r_fc_dt), r_fc_dt)
            P_fc = fc.clip(P_fc)
            P_d = float(dem.P_dem[i])
            # Batterie : boucle le bus dans ses limites
            P_req = P_d - P_fc
            P_ch, P_dis = bat.charge_limit(soc), bat.discharge_limit(soc)
            friction = spill = unmet = 0.0
            if P_req < -P_ch:
                excess = -P_ch - P_req
                friction = min(excess, max(P_aux - P_d, 0.0))  # régén. déviée vers les freins
                spill = excess - friction
                P_bat = -P_ch
            elif P_req > P_dis:
                unmet = P_req - P_dis
                P_bat = P_dis
            else:
                P_bat = P_req
            res = abs(P_fc + P_bat - (P_d + friction + spill - unmet))
            if res > _BUS_TOL_W:
                raise RuntimeError(f"bilan de bus non bouclé à t={c.t[i]:.1f}s : {res:.3g} W")
            max_res = max(max_res, res)
            soc = bat.step(soc, P_bat, dt)
            P_fc_log[i], P_bat_log[i] = P_fc, P_bat
            fr_log[i], spill_log[i], unmet_log[i] = friction, spill, unmet
            soc_log[i + 1] = soc
            h2_log[i + 1] = h2_log[i] + (fc.a + fc.b * P_fc + fc.c * P_fc * P_fc) / lhv * dt

    return SimResult(
        cycle=cycle,
        strategy=strategy.name,
        cfg=cfg,
        t=c.t,
        v=c.v,
        P_dem=dem.P_dem,
        P_fc=P_fc_log,
        P_bat=P_bat_log,
        P_friction=fr_log,
        P_spill=spill_log,
        P_unmet=unmet_log,
        soc=soc_log,
        m_h2=h2_log,
        t_ems=t_ems,
        FC_Power_cmd=fc_cmd,
        BAT_Power_cmd=bat_cmd,
        P_fc_at_ems=fc_at,
        ems_info={k: np.asarray(v, dtype=float) for k, v in info_log.items()},
        runtime_s=time.perf_counter() - tic,
        max_bus_residual_W=max_res,
    )


def run(cycle_name: str, strategy_name: str, cfg: IEMSConfig | None = None, **kw: Any) -> SimResult:
    """Raccourci : simule une stratégie du registre sur un cycle du catalogue."""
    from iems.models.cycles import get_cycle
    from iems.strategies import make_strategy

    cfg = cfg or IEMSConfig()
    return simulate(get_cycle(cycle_name), make_strategy(strategy_name, cfg, **kw), cfg)
