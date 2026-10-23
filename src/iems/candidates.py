"""Bornes de puissance PàC / batterie et vecteur de 9 candidats.

Trois couches de limites sur la consigne PàC du prochain pas EMS :

1. **PàC** : bornes statiques ``[P_min, P_max]`` et rampe ``r_fc`` → ``[P_lo, P_hi]``
   atteignable depuis ``P_fc(t)`` en ``Ts_ems`` (crochet température : non implémenté en v1).
2. **Batterie sur l'horizon** : la batterie doit pouvoir boucler ``P_bat = P_dem − P_fc``.
   Sur chaque intervalle ``i`` de la demande prédite, la PàC doit rester dans
   ``[P_dem_max,i − P_dis, max(P_dem_min,i, P_aux) + P_ch]`` ; comme elle ne peut bouger que
   de ``r_fc`` par seconde, la consigne d'aujourd'hui hérite d'un **plancher** et d'un
   **plafond** anticipés (la PàC « se prépare » à une accélération ou à un freinage).
   Les limites batterie sont minorées de ``safety_margin`` (bruit de prédiction, dérive SOC).
3. **Synthèse** : intersection des deux ; si elle est vide (plancher > plafond, typiquement
   une marche de +40 kW en 1 s), consigne au milieu, ce qui répartit l'érosion des marges.

Les 9 candidats sont construits par quarts d'écart entre la puissance courante (ramenée dans
la plage utilisable) et chacune des bornes. Si la plage est dégénérée (bornes confondues),
le **vecteur de secours** contient 9 copies de la seule valeur admissible : jamais de NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iems.config import IEMSConfig
from iems.models.battery import Battery
from iems.models.fuel_cell import FuelCell

N_CANDIDATES = 9
_DEGENERATE_EPS_W = 1.0  # plage < 1 W ⇒ dégénérée


@dataclass(frozen=True)
class CandidateSet:
    """Résultat du bloc Candidates pour un pas EMS."""

    P_fc: NDArray[np.float64]  # (9,) consignes PàC candidates (W)
    P_bat: NDArray[np.float64]  # (9,) puissance batterie correspondante (W)
    valid: NDArray[np.bool_]  # (9,) candidat bouclable par la batterie au pas courant
    excess: NDArray[np.float64]  # (9,) dépassement des limites batterie (W), 0 si valide
    P_lo: float  # borne basse utilisable (W)
    P_hi: float  # borne haute utilisable (W)
    floor: float  # plancher anticipé (W)
    ceiling: float  # plafond anticipé (W)
    degenerate: bool  # vecteur de secours utilisé
    conflict: bool  # plancher > plafond : la batterie ne peut pas tout boucler


def interval_extrema(
    P_dem_pred: NDArray[np.float64], substeps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Max, min et moyenne de la demande prédite (pas modèle) par intervalle EMS.

    ``P_dem_pred[0]`` est l'instant courant ; l'intervalle ``i`` couvre les échantillons
    ``[i·substeps, (i+1)·substeps]`` (bornes incluses).
    """
    n_int = max((len(P_dem_pred) - 1) // substeps, 1)
    n = n_int * substeps
    if len(P_dem_pred) < n + 1:  # demande plus courte qu'un intervalle : maintien du dernier point
        P_dem_pred = np.pad(P_dem_pred, (0, n + 1 - len(P_dem_pred)), mode="edge")
    body = P_dem_pred[:n].reshape(n_int, substeps)
    ends = P_dem_pred[substeps : n + 1 : substeps]
    p_max = np.maximum(body.max(axis=1), ends)
    p_min = np.minimum(body.min(axis=1), ends)
    return p_max, p_min, body.mean(axis=1)


def dilate(
    p_max: NDArray[np.float64], p_min: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Robustesse à une erreur de calendrier d'un intervalle : max/min glissant sur ±1."""
    if len(p_max) < 2:
        return p_max, p_min
    hi = np.maximum(p_max, np.maximum(np.r_[p_max[1:], p_max[-1]], np.r_[p_max[0], p_max[:-1]]))
    lo = np.minimum(p_min, np.minimum(np.r_[p_min[1:], p_min[-1]], np.r_[p_min[0], p_min[:-1]]))
    return hi, lo


def anticipation_bounds(
    p_max: NDArray[np.float64],
    p_min: NDArray[np.float64],
    soc: float,
    cfg: IEMSConfig,
    battery: Battery,
) -> tuple[float, float]:
    """Plancher et plafond sur la consigne PàC, déduits de la demande prédite par intervalle."""
    ems = cfg.ems
    bp = battery.p
    # Limites évaluées au SOC le plus défavorable atteignable en un pas EMS (près des bornes,
    # le dérating BMS se resserre pendant que la consigne est bloquée), puis minorées d'une
    # marge relative et absolue (erreur de prédiction).
    d_soc = max(bp.P_dis_max_W, bp.P_ch_max_W) * ems.Ts_ems / (bp.eta_dis * bp.E_cap_J)
    keep = 1.0 - ems.safety_margin
    P_dis = max(battery.discharge_limit(soc - d_soc) * keep - ems.safety_margin_W, 0.0)
    P_ch = max(battery.charge_limit(soc + d_soc) * keep - ems.safety_margin_W, 0.0)
    # Marge de rampe : à partir de l'intervalle 2, la PàC peut encore bouger de r·Ts par pas.
    slack = cfg.fuel_cell.ramp_W_per_s * ems.Ts_ems * np.maximum(np.arange(len(p_max)) - 1, 0)
    # Marge de prédiction proportionnelle à la demande : l'erreur croît avec elle.
    k = ems.prediction_margin
    floor = float(np.max(p_max + k * np.abs(p_max) - P_dis - slack))
    cap = np.maximum(p_min - k * np.abs(p_min), cfg.vehicle.P_aux_W) + P_ch
    ceiling = float(np.min(cap + slack))
    return floor, ceiling


def build_candidates(
    P_fc_now: float,
    P_dem_pred: NDArray[np.float64],
    soc: float,
    P_bat_prev: float,
    cfg: IEMSConfig,
    fc: FuelCell,
    battery: Battery,
) -> CandidateSet:
    """Construit les bornes utilisables et les 9 candidats (ou le vecteur de secours).

    ``P_dem_pred`` : demande prédite au pas modèle depuis l'instant courant (indice 0)
    jusqu'à ``t + Hp``. Pour une stratégie sans prédiction, passer ``[P_dem(t)]``.
    """
    ems = cfg.ems
    P_lo, P_hi = fc.ramp_limits(P_fc_now, ems.Ts_ems)

    p_max, p_min, p_mean = interval_extrema(np.asarray(P_dem_pred, dtype=float), ems.substeps)
    floor, ceiling = anticipation_bounds(*dilate(p_max, p_min), soc, cfg, battery)
    conflict = floor > ceiling
    if conflict:
        # Les marges sont conservatrices : couper la poire en deux laisse les deux limites
        # réelles (sans marge) respectées tant que le recouvrement reste inférieur aux marges.
        floor = ceiling = 0.5 * (floor + ceiling)

    lo = max(P_lo, min(floor, P_hi))
    hi = max(min(P_hi, ceiling), lo)

    if hi - lo < _DEGENERATE_EPS_W:
        P_fc = np.full(N_CANDIDATES, lo)  # vecteur de secours
        degenerate = True
    else:
        c = min(max(P_fc_now, lo), hi)
        d_minus, d_plus = (c - lo) / 4.0, (hi - c) / 4.0
        P_fc = np.array(
            [
                lo,
                c - 3 * d_minus,
                c - 2 * d_minus,
                c - d_minus,
                c,
                c + d_plus,
                c + 2 * d_plus,
                c + 3 * d_plus,
                hi,
            ]
        )
        degenerate = False

    P_dem_next = float(p_mean[0])
    P_bat = P_dem_next - P_fc
    brake = max(cfg.vehicle.P_aux_W - P_dem_next, 0.0)  # régénération déviable vers la friction
    r_bat = battery.p.ramp_W_per_s * ems.Ts_ems
    excess = (
        np.maximum(P_bat - battery.discharge_limit(soc), 0.0)
        + np.maximum(-battery.charge_limit(soc) - brake - P_bat, 0.0)
        + np.maximum(np.abs(P_bat - P_bat_prev) - r_bat, 0.0)
    )
    valid = excess <= 1e-6
    return CandidateSet(P_fc, P_bat, valid, excess, lo, hi, floor, ceiling, degenerate, conflict)
