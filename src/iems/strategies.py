"""Stratégies de gestion d'énergie, toutes derrière la même interface ``Strategy.step``.

* ``rules_baseline`` : power follower borné (la PàC suit ``P_dem`` filtré passe-bas, la
  batterie absorbe le reste) + hystérésis SOC à trois modes (recharge / normal / délestage).
* ``ecms_const`` : Hamiltonien à λ constant. λ est calibré **hors ligne** par dichotomie pour
  obtenir ``SOC(T) ≈ SOC(0)`` sur le cycle : c'est la référence quasi-optimale.
* ``iems`` : pipeline complet DPR → Candidates → Balance (λ adaptatif) → Optimizer.

Toutes passent par le même bloc Candidates (admissibilité PàC/batterie) : la comparaison
porte sur la décision énergétique, pas sur le respect des contraintes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from iems.balance import Balance
from iems.candidates import CandidateSet, build_candidates
from iems.config import IEMSConfig
from iems.dpt.dpr import DrivingPatternRecognizer
from iems.models.battery import Battery
from iems.models.fuel_cell import FuelCell
from iems.optimizer import optimize
from iems.patterns import DrivingPattern

if TYPE_CHECKING:
    from iems.models.cycles import Cycle
    from iems.simulator import SimResult


@dataclass(frozen=True)
class EMSObservation:
    """Ce que voit l'EMS à un pas ``Ts_ems``."""

    k: int  # indice du pas EMS
    t: float
    v: float  # m/s
    soc: float
    P_fc: float  # puissance PàC appliquée à l'instant courant (W)
    P_bat: float  # dernière puissance batterie appliquée (W)
    P_dem: float  # demande courante (W)
    P_dem_pred: NDArray[np.float64]  # demande prédite au pas modèle sur [t, t+Hp]
    v_hist: NDArray[np.float64]  # historique de vitesse au pas EMS (le dernier = v)


@dataclass(frozen=True)
class EMSCommand:
    FC_Power_cmd: float  # W
    BAT_Power_cmd: float  # W
    info: dict[str, Any] = field(default_factory=dict)  # diagnostics journalisés


class Strategy(Protocol):
    name: str

    def reset(self) -> None: ...

    def step(self, obs: EMSObservation) -> EMSCommand: ...


class _Base:
    name = "base"

    def __init__(self, cfg: IEMSConfig) -> None:
        self.cfg = cfg
        self.fc = FuelCell(cfg.fuel_cell)
        self.battery = Battery(cfg.battery)

    def reset(self) -> None:
        pass

    def _candidates(self, obs: EMSObservation) -> CandidateSet:
        return build_candidates(
            obs.P_fc, obs.P_dem_pred, obs.soc, obs.P_bat, self.cfg, self.fc, self.battery, obs.v
        )

    def _next_demand(self, obs: EMSObservation) -> float:
        """Demande moyenne prédite sur le prochain intervalle EMS."""
        n = self.cfg.ems.substeps
        return float(np.mean(obs.P_dem_pred[: n + 1]))

    @staticmethod
    def _cand_info(c: CandidateSet) -> dict[str, Any]:
        return {"P_lo": c.P_lo, "P_hi": c.P_hi, "degenerate": c.degenerate, "conflict": c.conflict}


class RulesBaseline(_Base):
    """Power follower borné + hystérésis SOC."""

    name = "rules_baseline"

    def reset(self) -> None:
        self._P_filt: float | None = None
        self._mode = 0  # -1 délestage, 0 normal, +1 recharge

    def __init__(self, cfg: IEMSConfig) -> None:
        super().__init__(cfg)
        self.reset()

    def step(self, obs: EMSObservation) -> EMSCommand:
        ems = self.cfg.ems
        if self._P_filt is None:
            self._P_filt = obs.P_dem
        alpha = ems.Ts_ems / (ems.baseline_tau_s + ems.Ts_ems)
        self._P_filt += alpha * (obs.P_dem - self._P_filt)
        # Hystérésis : on quitte un mode en repassant le SOC de référence.
        if obs.soc < ems.baseline_soc_low:
            self._mode = 1
        elif obs.soc > ems.baseline_soc_high:
            self._mode = -1
        elif (self._mode == 1 and obs.soc >= ems.soc_ref) or (
            self._mode == -1 and obs.soc <= ems.soc_ref
        ):
            self._mode = 0
        target = self._P_filt + self._mode * ems.baseline_charge_W
        c = self._candidates(obs)
        cmd = min(max(target, c.P_lo), c.P_hi)
        info = {"mode": self._mode, **self._cand_info(c)}
        return EMSCommand(cmd, self._next_demand(obs) - cmd, info)


class ECMSConst(_Base):
    """ECMS à co-état constant."""

    name = "ecms_const"

    def __init__(self, cfg: IEMSConfig, lam: float | None = None) -> None:
        super().__init__(cfg)
        self.lam = cfg.ems.lambda0 if lam is None else lam

    def step(self, obs: EMSObservation) -> EMSCommand:
        c = self._candidates(obs)
        out = optimize(c, self._next_demand(obs), self.lam, self.fc, self.battery)
        info = {"lam": self.lam, "j_star": out.j_star, **self._cand_info(c)}
        return EMSCommand(out.FC_Power_cmd, out.BAT_Power_cmd, info)


class IEMS(_Base):
    """iEMS : DPR + Balance (SOC cible, λ adaptatif) + PMP en ligne sur candidats."""

    name = "iems"

    def __init__(self, cfg: IEMSConfig) -> None:
        super().__init__(cfg)
        self.dpr = DrivingPatternRecognizer(cfg.ems.dpr_window_s, cfg.ems.Ts_ems)
        self.balance = Balance(cfg, self.battery)

    def step(self, obs: EMSObservation) -> EMSCommand:
        pattern: DrivingPattern = self.dpr.recognize(obs.v_hist)
        bal = self.balance.step(obs.v, obs.soc, pattern)
        c = self._candidates(obs)
        out = optimize(c, self._next_demand(obs), bal.lam, self.fc, self.battery)
        info = {
            "lam": bal.lam,
            "soc_target": bal.soc_target,
            "soc_ref_v": bal.soc_ref_v,
            "usage": bal.usage,
            "pattern": int(pattern),
            "j_star": out.j_star,
            **self._cand_info(c),
        }
        return EMSCommand(out.FC_Power_cmd, out.BAT_Power_cmd, info)


STRATEGIES: dict[str, Callable[..., Strategy]] = {
    RulesBaseline.name: RulesBaseline,
    ECMSConst.name: ECMSConst,
    IEMS.name: IEMS,
}


def make_strategy(name: str, cfg: IEMSConfig, **kwargs: Any) -> Strategy:
    """Instancie une stratégie du registre par son nom."""
    try:
        return STRATEGIES[name](cfg, **kwargs)
    except KeyError:
        raise KeyError(
            f"stratégie inconnue : {name!r} (disponibles : {list(STRATEGIES)})"
        ) from None


@dataclass(frozen=True)
class Calibration:
    lam: float
    delta_soc: float  # SOC(T) − SOC(0) obtenu avec ``lam``
    iterations: int
    result: SimResult


def calibrate_ecms_const(
    cycle: Cycle,
    cfg: IEMSConfig,
    lam_bounds: tuple[float, float] = (1.0, 4.0),
    tol_soc: float = 0.002,
    max_iter: int = 25,
) -> Calibration:
    """Dichotomie sur λ pour que ``SOC(T) ≈ SOC(0)`` (charge-sustaining) sur ``cycle``.

    ``SOC(T)`` est croissant en λ (batterie plus « chère » → plus de recharge par la PàC).
    """
    from iems.simulator import simulate  # import local : simulator dépend de ce module

    def run(lam: float) -> tuple[float, SimResult]:
        res = simulate(cycle, ECMSConst(cfg, lam), cfg)
        return res.soc_final - cfg.soc0, res

    lo, hi = lam_bounds
    d_lo, r_lo = run(lo)
    d_hi, r_hi = run(hi)
    if d_lo > 0 or d_hi < 0:
        best = (lo, d_lo, r_lo) if abs(d_lo) < abs(d_hi) else (hi, d_hi, r_hi)
        return Calibration(best[0], best[1], 2, best[2])
    best = (lo, d_lo, r_lo) if abs(d_lo) < abs(d_hi) else (hi, d_hi, r_hi)
    it = 2
    while it < max_iter and abs(best[1]) > tol_soc:
        mid = 0.5 * (lo + hi)
        d_mid, r_mid = run(mid)
        it += 1
        if abs(d_mid) < abs(best[1]):
            best = (mid, d_mid, r_mid)
        if d_mid < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return Calibration(best[0], best[1], it, best[2])
