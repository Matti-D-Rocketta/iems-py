"""Système pile à combustible : rendement η(P), consommation H2, bornes et rampe."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iems.config import FuelCellParams


class FuelCell:
    """Modèle statique de système PàC (carte de rendement interpolée)."""

    def __init__(self, params: FuelCellParams) -> None:
        self.p = params
        self._P_map = np.asarray(params.eta_map_frac, dtype=float) * params.P_max_W
        self._eta_map = np.asarray(params.eta_map_eta, dtype=float)

    def efficiency(self, P_fc: ArrayLike) -> NDArray[np.float64]:
        """Rendement système (PCI) interpolé linéairement, saturé aux extrémités."""
        return np.interp(np.asarray(P_fc, dtype=float), self._P_map, self._eta_map)

    def h2_rate(self, P_fc: ArrayLike) -> NDArray[np.float64]:
        """Débit massique d'H2 (kg/s) : ``ṁ = P_fc / (η_fc(P_fc) · PCI)``, nul si P_fc ≤ 0."""
        P = np.maximum(np.asarray(P_fc, dtype=float), 0.0)
        return P / (self.efficiency(P) * self.p.lhv_J_per_kg)

    def clip(self, P_fc: float) -> float:
        """Sature dans ``[P_min, P_max]``."""
        return min(max(P_fc, self.p.P_min_W), self.p.P_max_W)

    def ramp_limits(self, P_now: float, dt: float) -> tuple[float, float]:
        """Plage atteignable en ``dt`` compte tenu de la rampe et des bornes statiques."""
        dP = self.p.ramp_W_per_s * dt
        return max(self.p.P_min_W, P_now - dP), min(self.p.P_max_W, P_now + dP)
