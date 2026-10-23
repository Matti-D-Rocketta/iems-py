"""Puissance électrique demandée au bus DC à partir d'un profil de vitesse.

``P_dem = P_roue / η_trans + P_aux`` en traction et ``P_roue · η_trans + P_aux`` en
freinage (la chaîne de traction a des pertes dans les deux sens). ``P_roue`` vient de
l'équation longitudinale sur route plate :
``P_roue = v · (m·a + ½·ρ·Cd·Af·v² + Crr·m·g)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iems.config import VehicleParams

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PowerDemand:
    """Décomposition de la demande sur une grille temporelle."""

    t: FloatArray
    v: FloatArray
    a: FloatArray
    P_wheel: FloatArray  # puissance mécanique à la roue (W), signée
    P_dem: FloatArray  # puissance électrique demandée au bus (W), signée


def acceleration(t: FloatArray, v: FloatArray) -> FloatArray:
    """Accélération par différences finies centrées (ordre 2, extrémités d'ordre 1)."""
    if len(v) < 2:
        return np.zeros_like(v)
    return np.gradient(v, t)


def wheel_power(v: FloatArray, a: FloatArray, p: VehicleParams) -> FloatArray:
    """Puissance mécanique à la roue (W)."""
    v = np.asarray(v, dtype=float)
    aero = 0.5 * p.rho_air * p.Cd * p.frontal_area_m2 * v**2
    roll = np.where(v > 0.0, p.Crr * p.mass_kg * p.g, 0.0)
    return v * (p.mass_kg * np.asarray(a, dtype=float) + aero + roll)


def electrical_demand(P_wheel: FloatArray, p: VehicleParams) -> FloatArray:
    """Puissance électrique au bus DC (W), auxiliaires compris."""
    P_wheel = np.asarray(P_wheel, dtype=float)
    P_motor = np.where(P_wheel >= 0.0, P_wheel / p.eta_trans, P_wheel * p.eta_trans)
    return P_motor + p.P_aux_W


def power_demand(t: FloatArray, v: FloatArray, p: VehicleParams) -> PowerDemand:
    """Calcule la demande complète sur la grille ``t``."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    a = acceleration(t, v)
    P_wheel = wheel_power(v, a, p)
    return PowerDemand(t=t, v=v, a=a, P_wheel=P_wheel, P_dem=electrical_demand(P_wheel, p))
