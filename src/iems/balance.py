"""Balance : pourcentage d'utilisation batterie → SOC cible → co-état λ.

v1 : l'usage batterie est lu dans une **table interpolée** (bilinéaire) indexée par le schéma
de conduite, la vitesse normalisée et le SOC normalisé. La table est construite à partir de
l'usage nominal par schéma (sportif ≈ 85 %, autoroute ≈ 30 %…) modulé par le SOC (SOC haut →
on peut davantage solliciter la batterie) et la vitesse. v2 : contrôleur flou (scikit-fuzzy)
à interface identique.

SOC cible : point de la plage de SOC atteignable en fin d'horizon ``Hp``, pondéré par l'usage
(usage 1 → bas de la plage, la batterie fournit au maximum ; usage 0 → haut de la plage).

Co-état : ``λ(t) = λ0 + Kp · (SOC_target − SOC(t))`` (adaptation proportionnelle type A-ECMS),
saturé dans ``[λ_min, λ_max]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iems.config import IEMSConfig
from iems.models.battery import Battery
from iems.patterns import DrivingPattern

_GRID = np.linspace(0.0, 1.0, 5)  # nœuds de la table en v_norm et SOC_norm


@dataclass(frozen=True)
class BalanceOutput:
    usage: float  # pourcentage d'utilisation batterie ∈ [0, 1]
    soc_lo: float  # bas de la plage atteignable en fin d'horizon
    soc_hi: float  # haut de la plage atteignable
    soc_target: float
    lam: float  # co-état (facteur d'équivalence, sans dimension)


def build_usage_table(cfg: IEMSConfig) -> NDArray[np.float64]:
    """Table ``[schéma, v_norm, soc_norm]`` des usages batterie aux nœuds de ``_GRID``."""
    ems, bat = cfg.ems, cfg.battery
    sn_ref = (ems.soc_ref - bat.soc_min) / (bat.soc_max - bat.soc_min)
    vn, sn = np.meshgrid(_GRID, _GRID, indexing="ij")
    table = np.empty((len(DrivingPattern), len(_GRID), len(_GRID)))
    for p in DrivingPattern:
        base = ems.usage_table[p]
        table[p] = base + ems.usage_soc_gain * (sn - sn_ref) - ems.usage_v_gain * (vn - 0.5)
    return np.clip(table, 0.0, 1.0)


def _bilinear(table2d: NDArray[np.float64], x: float, y: float) -> float:
    """Interpolation bilinéaire sur la grille régulière ``_GRID`` × ``_GRID`` (entrées saturées)."""
    n = len(_GRID) - 1
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
        return BalanceOutput(u, lo, hi, target, self.costate(soc, target))
