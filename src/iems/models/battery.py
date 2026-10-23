"""Batterie v1 : modèle énergétique à rendements constants.

``dSOC/dt = −P_int / E_cap`` avec ``E_cap`` en joules (soit ``3600 · E_cap[Wh]``) et
``P_int = P_bat / η_dis`` en décharge, ``P_bat · η_ch`` en charge (``P_bat`` > 0 = décharge,
mesurée aux bornes).

Limites BMS : la puissance de décharge admissible décroît linéairement jusqu'à 0 sur une
bande ``derate_band`` au-dessus de ``soc_min`` ; symétriquement pour la charge sous ``soc_max``.
Ces rampes douces garantissent que le SOC ne sort pas de la fenêtre utile.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iems.config import BatteryParams


class Battery:
    """Batterie énergétique (v1). Les méthodes scalaires sont écrites pour la vitesse."""

    def __init__(self, params: BatteryParams) -> None:
        self.p = params

    # --- limites -------------------------------------------------------------------------
    def discharge_limit(self, soc: float) -> float:
        """Puissance de décharge max aux bornes (W, ≥ 0), croissante avec le SOC."""
        p = self.p
        frac = (soc - p.soc_min) / p.derate_band
        return p.P_dis_max_W * min(max(frac, 0.0), 1.0)

    def charge_limit(self, soc: float) -> float:
        """Puissance de charge max aux bornes (W, ≥ 0), décroissante avec le SOC."""
        p = self.p
        frac = (p.soc_max - soc) / p.derate_band
        return p.P_ch_max_W * min(max(frac, 0.0), 1.0)

    def discharge_limit_v(self, soc: ArrayLike) -> NDArray[np.float64]:
        frac = (np.asarray(soc, dtype=float) - self.p.soc_min) / self.p.derate_band
        return self.p.P_dis_max_W * np.clip(frac, 0.0, 1.0)

    def charge_limit_v(self, soc: ArrayLike) -> NDArray[np.float64]:
        frac = (self.p.soc_max - np.asarray(soc, dtype=float)) / self.p.derate_band
        return self.p.P_ch_max_W * np.clip(frac, 0.0, 1.0)

    # --- dynamique -----------------------------------------------------------------------
    def internal_power(self, P_bat: ArrayLike) -> NDArray[np.float64]:
        """Puissance interne (W) : ce qui sort réellement du stock d'énergie."""
        P = np.asarray(P_bat, dtype=float)
        return np.where(P >= 0.0, P / self.p.eta_dis, P * self.p.eta_ch)

    def internal_power_scalar(self, P_bat: float) -> float:
        return P_bat / self.p.eta_dis if P_bat >= 0.0 else P_bat * self.p.eta_ch

    def soc_rate(self, P_bat: ArrayLike) -> NDArray[np.float64]:
        """dSOC/dt (1/s)."""
        return -self.internal_power(P_bat) / self.p.E_cap_J

    def step(self, soc: float, P_bat: float, dt: float) -> float:
        """Intègre le SOC sur ``dt`` (Euler explicite exact à puissance constante)."""
        return soc - self.internal_power_scalar(P_bat) * dt / self.p.E_cap_J

    def reachable_soc(self, soc: float, horizon: float) -> tuple[float, float]:
        """Plage de SOC atteignable en ``horizon`` secondes aux limites courantes.

        Approximation : limites gelées au SOC courant, bornée par la fenêtre utile.
        """
        p = self.p
        lo = soc - self.discharge_limit(soc) / p.eta_dis * horizon / p.E_cap_J
        hi = soc + self.charge_limit(soc) * p.eta_ch * horizon / p.E_cap_J
        return max(lo, p.soc_min), min(hi, p.soc_max)
