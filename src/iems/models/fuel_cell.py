"""Système pile à combustible : consommation H2, rendement, bornes et rampe.

La consommation est modélisée par une **parabole convexe** en puissance nette :

    P_H2(P) = ṁ_H2 · PCI = a + b·P + c·P²      (a > 0 : débit de ralenti, c > 0)

ajustée aux moindres carrés sur la carte de rendement de la config (``P_i / η_i``). Ce modèle
classique (pertes des auxiliaires à vide + pertes ohmiques/activation croissantes) produit
naturellement la cloche η(P) = P / P_H2(P) : faible à basse charge à cause du débit de
ralenti, maximale vers ``√(a/c)``, décroissante à forte charge.

Pourquoi ne pas interpoler η linéairement : avec une carte qui passe par η(0) > 0, le coût
``P/η(P)`` est **concave** à faible charge et linéaire sur un plateau de rendement. Couplé à
une batterie à rendement constant, le Hamiltonien n'a alors pas de minimum intérieur : la
commande devient tout-ou-rien et la calibration de λ mal posée (ΔSOC saute de −16 à +20
points pour Δλ = 0,2). La parabole convexe rend le Hamiltonien strictement convexe.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iems.config import FuelCellParams


def fit_consumption(params: FuelCellParams) -> tuple[float, float, float]:
    """Coefficients ``(a, b, c)`` de ``P_H2 = a + b·P + c·P²`` (W) ajustés sur la carte η."""
    P = np.asarray(params.eta_map_frac, dtype=float) * params.P_max_W
    E = P / np.asarray(params.eta_map_eta, dtype=float)
    c, b, a = np.polyfit(P, E, 2)
    if c <= 0 or a < 0:
        raise ValueError("la carte de rendement ne correspond pas à une consommation convexe")
    return float(a), float(b), float(c)


class FuelCell:
    """Modèle statique de système PàC."""

    def __init__(self, params: FuelCellParams) -> None:
        self.p = params
        self.a, self.b, self.c = fit_consumption(params)

    def h2_power(self, P_fc: ArrayLike) -> NDArray[np.float64]:
        """Puissance chimique H2 consommée (W, au PCI) pour une puissance nette ``P_fc``."""
        P = np.maximum(np.asarray(P_fc, dtype=float), 0.0)
        return self.a + self.b * P + self.c * P * P

    def h2_rate(self, P_fc: ArrayLike) -> NDArray[np.float64]:
        """Débit massique d'H2 (kg/s)."""
        return self.h2_power(P_fc) / self.p.lhv_J_per_kg

    def efficiency(self, P_fc: ArrayLike) -> NDArray[np.float64]:
        """Rendement système (PCI) ``η = P / P_H2(P)``."""
        P = np.maximum(np.asarray(P_fc, dtype=float), 0.0)
        return P / self.h2_power(P)

    @property
    def peak_power(self) -> float:
        """Puissance de rendement maximal ``√(a/c)`` (W)."""
        return float(np.sqrt(self.a / self.c))

    def clip(self, P_fc: float) -> float:
        """Sature dans ``[P_min, P_max]``."""
        return min(max(P_fc, self.p.P_min_W), self.p.P_max_W)

    def ramp_limits(self, P_now: float, dt: float) -> tuple[float, float]:
        """Plage atteignable en ``dt`` compte tenu de la rampe et des bornes statiques."""
        dP = self.p.ramp_W_per_s * dt
        return max(self.p.P_min_W, P_now - dP), min(self.p.P_max_W, P_now + dP)
