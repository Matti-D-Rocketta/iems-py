"""Métriques d'une simulation : consommation H2, ΔSOC, violations, bilan d'énergie.

Consommation **corrigée du SOC** (comparaison équitable entre stratégies qui ne finissent
pas au même SOC) : l'énergie stockée consommée ``ΔE = (SOC0 − SOC_T)·E_cap`` est convertie en
H2 au rendement PàC moyen du run :

* batterie déchargée (ΔE > 0) : il faudrait la recharger par la PàC → ``+ΔE / (η_fc·η_ch·PCI)`` ;
* batterie plus chargée (ΔE < 0) : l'excédent remplacerait de la PàC → ``ΔE·η_dis / (η_fc·PCI)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from iems.models.battery import Battery
from iems.simulator import SimResult

_TOL_W = 1.0  # W : tolérance numérique sur les dépassements de puissance
_TOL_SOC = 1e-9


@dataclass(frozen=True)
class Violations:
    soc_window: int = 0  # pas modèle hors fenêtre SOC utile
    fc_bounds: int = 0  # PàC hors [P_min, P_max]
    fc_ramp: int = 0  # rampe PàC appliquée dépassée
    cmd_ramp: int = 0  # consigne EMS hors de portée de la rampe sur Ts_ems
    bat_discharge: int = 0  # traction non servie (batterie à sa limite de décharge)
    bat_charge: int = 0  # surplus PàC non absorbable (batterie à sa limite de charge)

    @property
    def total(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class EnergyBalance:
    """Bilan global (J) : sources = usages + pertes, à la précision de l'intégrateur près."""

    E_fc: float  # énergie électrique PàC
    E_storage: float  # énergie déstockée (SOC0 − SOC_T)·E_cap
    E_demand_served: float
    E_friction: float
    E_spill: float
    E_bat_losses: float
    E_h2_chem: float  # énergie chimique de l'H2 consommé (PCI)
    rel_error: float  # |sources − usages| / ∫|P_dem|


@dataclass(frozen=True)
class Metrics:
    cycle: str
    strategy: str
    distance_km: float
    h2_g: float
    h2_g_per_100km: float
    h2_corr_g: float
    h2_corr_g_per_100km: float
    soc0: float
    soc_final: float
    delta_soc_pts: float
    soc_min: float
    soc_max: float
    eta_fc_mean: float
    violations: Violations
    energy: EnergyBalance
    runtime_s: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def charge_sustaining(self) -> bool:
        return abs(self.delta_soc_pts) < 2.0

    def row(self) -> dict[str, object]:
        """Ligne de tableau comparatif."""
        return {
            "cycle": self.cycle,
            "stratégie": self.strategy,
            "km": round(self.distance_km, 2),
            "H2 g/100km": round(self.h2_g_per_100km, 1),
            "H2 corr. g/100km": round(self.h2_corr_g_per_100km, 1),
            "ΔSOC pts": round(self.delta_soc_pts, 2),
            "violations": self.violations.total,
            "bilan %": round(100 * self.energy.rel_error, 4),
        }


def compute_metrics(res: SimResult) -> Metrics:
    cfg = res.cfg
    dt = res.dt
    bat = Battery(cfg.battery)
    lhv = cfg.fuel_cell.lhv_J_per_kg

    m_h2 = float(res.m_h2[-1])  # intégrée par le simulateur (= Σ ṁ(P_fc)·dt, P_fc constant par pas)
    E_fc = float(np.sum(res.P_fc) * dt)
    E_h2 = m_h2 * lhv
    eta_fc = E_fc / E_h2 if E_h2 > 0 else float(np.mean(cfg.fuel_cell.eta_map_eta))
    E_storage = (cfg.soc0 - res.soc_final) * cfg.battery.E_cap_J
    if E_storage >= 0:
        m_corr = m_h2 + E_storage / (eta_fc * cfg.battery.eta_ch * lhv)
    else:
        m_corr = m_h2 + E_storage * cfg.battery.eta_dis / (eta_fc * lhv)

    P_dem = res.P_dem[:-1]
    E_served = float(np.sum(P_dem - res.P_unmet) * dt)
    E_fric = float(np.sum(res.P_friction) * dt)
    E_spill = float(np.sum(res.P_spill) * dt)
    E_bat_loss = float(np.sum(bat.internal_power(res.P_bat) - res.P_bat) * dt)
    sources = E_fc + E_storage
    uses = E_served + E_fric + E_spill + E_bat_loss
    scale = float(np.sum(np.abs(P_dem)) * dt)
    energy = EnergyBalance(
        E_fc, E_storage, E_served, E_fric, E_spill, E_bat_loss, E_h2, abs(sources - uses) / scale
    )

    p_fc = cfg.fuel_cell
    r = p_fc.ramp_W_per_s
    P_fc_full = np.concatenate(([p_fc.P_min_W], res.P_fc))
    viol = Violations(
        soc_window=int(
            np.sum(
                (res.soc < cfg.battery.soc_min - _TOL_SOC)
                | (res.soc > cfg.battery.soc_max + _TOL_SOC)
            )
        ),
        fc_bounds=int(
            np.sum((res.P_fc < p_fc.P_min_W - _TOL_W) | (res.P_fc > p_fc.P_max_W + _TOL_W))
        ),
        fc_ramp=int(np.sum(np.abs(np.diff(P_fc_full)) > r * dt + _TOL_W * dt)),
        cmd_ramp=int(
            np.sum(np.abs(res.FC_Power_cmd - res.P_fc_at_ems) > r * cfg.ems.Ts_ems + _TOL_W)
        ),
        bat_discharge=int(np.sum(res.P_unmet > _TOL_W)),
        bat_charge=int(np.sum(res.P_spill > _TOL_W)),
    )

    dist_km = float(np.trapezoid(res.v, res.t)) / 1000.0
    per100 = 100.0 / dist_km if dist_km > 0 else float("nan")
    return Metrics(
        cycle=res.cycle.name,
        strategy=res.strategy,
        distance_km=dist_km,
        h2_g=1e3 * m_h2,
        h2_g_per_100km=1e3 * m_h2 * per100,
        h2_corr_g=1e3 * m_corr,
        h2_corr_g_per_100km=1e3 * m_corr * per100,
        soc0=cfg.soc0,
        soc_final=res.soc_final,
        delta_soc_pts=100.0 * (res.soc_final - cfg.soc0),
        soc_min=float(res.soc.min()),
        soc_max=float(res.soc.max()),
        eta_fc_mean=eta_fc,
        violations=viol,
        energy=energy,
        runtime_s=res.runtime_s,
    )
