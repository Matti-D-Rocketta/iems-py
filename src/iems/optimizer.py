"""Optimiseur : Hamiltonien par candidat, pénalités, argmin → ``FC_Power_cmd`` / ``BAT_Power_cmd``.

Hamiltonien (PMP / ECMS) pour le candidat ``j`` :

    H_j = ṁ_H2,j − λ · k_h · dSOC_j/dt

* ``ṁ_H2,j = P_fc,j / (η_fc(P_fc,j) · PCI)`` en kg/s ;
* ``dSOC_j/dt = −P_int(P_bat,j) / E_cap`` en 1/s, avec ``P_bat,j = P_dem − P_fc,j`` ;
* ``k_h = E_cap / PCI`` en kg : **facteur d'homogénéisation**. Il convertit une variation de
  SOC en masse d'H2 qui contiendrait la même énergie (au PCI). Avec ce choix,
  ``λ · k_h · dSOC/dt = −λ · P_int / PCI`` : λ est un **facteur d'équivalence sans
  dimension**, l'inverse d'un rendement moyen de recharge par la PàC (ordre de grandeur
  1/0,5 ≈ 2). C'est ce qui rend le réglage de λ0 et Kp interprétable.

Convention de signe : décharger (dSOC/dt < 0) coûte ``λ·k_h·|dSOC/dt|`` ; λ > 0.

Pénalités : passe-plat en v1 (``H'_j = H_j``) ; crochet pour pénaliser rampes et
violations en v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iems.candidates import CandidateSet
from iems.models.battery import Battery
from iems.models.fuel_cell import FuelCell


def homogenization_factor(battery: Battery, fc: FuelCell) -> float:
    """``k_h = E_cap / PCI`` (kg) : SOC (sans dimension) → masse d'H2 équivalente."""
    return battery.p.E_cap_J / fc.p.lhv_J_per_kg


def hamiltonian(
    P_fc: NDArray[np.float64],
    P_dem: float,
    lam: float,
    fc: FuelCell,
    battery: Battery,
) -> NDArray[np.float64]:
    """Hamiltonien (kg/s) pour chaque candidat ``P_fc``."""
    mdot = fc.h2_rate(P_fc)
    dsoc = battery.soc_rate(P_dem - np.asarray(P_fc, dtype=float))
    return mdot - lam * homogenization_factor(battery, fc) * dsoc


def apply_penalties(
    H: NDArray[np.float64], context: dict[str, Any] | None = None
) -> NDArray[np.float64]:
    """Sous-système « Penalties » — v1 : passe-plat (sortie = entrée)."""
    del context  # crochet v2 : pénalités de rampe, de violation, d'usure
    return H


@dataclass(frozen=True)
class OptimizerOutput:
    FC_Power_cmd: float  # W
    BAT_Power_cmd: float  # W
    j_star: int
    H: NDArray[np.float64] = field(repr=False)


def optimize(
    cands: CandidateSet, P_dem: float, lam: float, fc: FuelCell, battery: Battery
) -> OptimizerOutput:
    """``j* = argmin H'_j`` parmi les candidats valides.

    Si aucun candidat n'est valide, l'argmin porte sur ceux de dépassement minimal.
    """
    H = apply_penalties(hamiltonian(cands.P_fc, P_dem, lam, fc, battery))
    keep = cands.valid if cands.valid.any() else cands.excess <= cands.excess.min() + 1e-6
    H_eff = np.where(keep, H, np.inf)
    j = int(np.argmin(H_eff))
    P_fc_cmd = float(cands.P_fc[j])
    return OptimizerOutput(P_fc_cmd, P_dem - P_fc_cmd, j, H)
