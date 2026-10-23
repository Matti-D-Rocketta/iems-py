"""Balance : pourcentage d'utilisation batterie → SOC cible → co-état λ.

Pipeline du bloc (section 4.3 de la passation) :

1. **Usage batterie** ``u ∈ [0, 1] = f(v_norm, SOC_norm, schéma)``. v1 : table bilinéaire par
   schéma, construite à partir de trois effets explicites (v2 : contrôleur flou, même
   interface) :

   * **schéma de conduite** : usage nominal (sportif ≈ 0,85 ; autoroute ≈ 0,30 ; trafic et
     urbain intermédiaires). ``u = 0,5`` est neutre (λ = λ0) ; au-dessus, l'EMS sollicite la
     batterie, en dessous il la recharge ;
   * **réserve cinétique** : la référence de SOC baisse avec la vitesse,
     ``SOC_ref(v) = soc_ref − kinetic_recovery · ½·m·v² / E_cap``. À 120 km/h, le véhicule
     « stocke » ≈ 1 MJ d'énergie cinétique dont une partie reviendra dans la batterie au
     freinage : il faut lui laisser la place (règle classique des HEV, cf. gestion du SOC
     Toyota THS ; littérature : SOC de référence fonction de la vitesse) ;
   * **ancrage SOC** : ``u`` croît avec l'écart ``SOC − SOC_ref(v)`` (gain ``usage_soc_gain``),
     ce qui ramène le SOC vers sa référence : c'est le terme proportionnel du charge-sustaining.

2. **SOC cible** : plage de SOC atteignable en fin d'horizon ``Hp`` (intégration des limites
   batterie), ``SOC_target = hi − u · (hi − lo)`` : usage 1 → bas de la plage (la batterie
   fournit au maximum), usage 0 → haut de la plage.

3. **Co-état** : ``λ = λ0 + Kp · (SOC_target − SOC)`` (adaptation proportionnelle type A-ECMS),
   saturé dans ``[λ_min, λ_max]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iems.config import IEMSConfig
from iems.models.battery import Battery
from iems.patterns import DrivingPattern

N_GRID = 9
_GRID = np.linspace(0.0, 1.0, N_GRID)  # nœuds de la table en v_norm et SOC_norm


@dataclass(frozen=True)
class BalanceOutput:
    usage: float  # pourcentage d'utilisation batterie ∈ [0, 1]
    soc_ref_v: float  # référence de SOC à la vitesse courante (réserve cinétique déduite)
    soc_lo: float  # bas de la plage atteignable en fin d'horizon
    soc_hi: float  # haut de la plage atteignable
    soc_target: float
    lam: float  # co-état (facteur d'équivalence, sans dimension)


def kinetic_reserve(
    v_norm: float | NDArray[np.float64], cfg: IEMSConfig
) -> float | NDArray[np.float64]:
    """Part de SOC à réserver pour la récupération de l'énergie cinétique à ``v_norm``."""
    ems = cfg.ems
    e_kin = 0.5 * cfg.vehicle.mass_kg * (np.asarray(v_norm) * ems.v_norm_max) ** 2
    return ems.kinetic_recovery * e_kin / cfg.battery.E_cap_J


def build_usage_table(cfg: IEMSConfig) -> NDArray[np.float64]:
    """Table ``[schéma, v_norm, soc_norm]`` des usages batterie aux nœuds de ``_GRID``."""
    ems, bat = cfg.ems, cfg.battery
    width = bat.soc_max - bat.soc_min
    vn, sn = np.meshgrid(_GRID, _GRID, indexing="ij")
    soc = bat.soc_min + sn * width
    soc_ref_v = ems.soc_ref - kinetic_reserve(vn, cfg)
    table = np.empty((len(DrivingPattern), N_GRID, N_GRID))
    for p in DrivingPattern:
        table[p] = ems.usage_table[p] + ems.usage_soc_gain * (soc - soc_ref_v) / width
    return np.clip(table, 0.0, 1.0)


def _bilinear(table2d: NDArray[np.float64], x: float, y: float) -> float:
    """Interpolation bilinéaire sur la grille régulière ``_GRID`` × ``_GRID`` (entrées saturées)."""
    n = N_GRID - 1
    fx = min(max(x, 0.0), 1.0) * n
    fy = min(max(y, 0.0), 1.0) * n
    i, j = min(int(fx), n - 1), min(int(fy), n - 1)
    tx, ty = fx - i, fy - j
    return float(
        table2d[i, j] * (1 - tx) * (1 - ty)
        + table2d[i + 1, j] * tx * (1 - ty)
        + table2d[i, j + 1] * (1 - tx) * ty
        + table2d[i + 1, j + 1] * tx * ty
    )


class Balance:
    """Bloc Balance de l'iEMS."""

    def __init__(self, cfg: IEMSConfig, battery: Battery) -> None:
        self.cfg = cfg
        self.battery = battery
        self.table = build_usage_table(cfg)

    def soc_ref(self, v: float) -> float:
        """Référence de SOC à la vitesse ``v`` (m/s), réserve cinétique déduite."""
        return float(self.cfg.ems.soc_ref - kinetic_reserve(v / self.cfg.ems.v_norm_max, self.cfg))

    def usage(self, v: float, soc: float, pattern: DrivingPattern) -> float:
        bat = self.cfg.battery
        v_norm = v / self.cfg.ems.v_norm_max
        soc_norm = (soc - bat.soc_min) / (bat.soc_max - bat.soc_min)
        return _bilinear(self.table[int(pattern)], v_norm, soc_norm)

    def soc_target(self, soc: float, usage: float) -> tuple[float, float, float]:
        lo, hi = self.battery.reachable_soc(soc, self.cfg.ems.Hp)
        return lo, hi, hi - usage * (hi - lo)

    def costate(self, soc: float, soc_target: float) -> float:
        ems = self.cfg.ems
        lam = ems.lambda0 + ems.Kp * (soc_target - soc)
        return min(max(lam, ems.lambda_min), ems.lambda_max)

    def step(self, v: float, soc: float, pattern: DrivingPattern) -> BalanceOutput:
        u = self.usage(v, soc, pattern)
        lo, hi, target = self.soc_target(soc, u)
        return BalanceOutput(u, self.soc_ref(v), lo, hi, target, self.costate(soc, target))
