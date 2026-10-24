"""DPR — reconnaissance du schéma de conduite.

v1 : **règles à seuils** sur des caractéristiques calculées sur une fenêtre glissante du
passé (vitesse moyenne et max, écart-type, accélérations, part d'arrêt). v2 : Fuzzy C-means
(scikit-fuzzy) entraîné hors ligne, même interface.

Seuils choisis pour séparer les segments du générateur synthétique (vérité terrain) ;
ce sont des ``# TODO-thèse`` à recaler sur les centres de classes FCM de la thèse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iems.patterns import DrivingPattern

KMH = 1.0 / 3.6
_STOP_SPEED = 0.5  # m/s : en dessous, le véhicule est considéré à l'arrêt


@dataclass(frozen=True)
class DPRFeatures:
    v_mean: float  # m/s
    v_max: float  # m/s
    v_std: float  # m/s
    a_pos_mean: float  # m/s², moyenne des accélérations positives
    a_abs_p95: float  # m/s², 95e centile de |a|
    stop_ratio: float  # part du temps à l'arrêt


@dataclass(frozen=True)
class DPRThresholds:
    # Seuils recalés par recherche en grille sur les 4 cycles synthétiques étiquetés (précision
    # moyenne 0,80 → 0,87, rappel sportif 0,47 → 0,85) ; TODO-thèse : centres de classes FCM.
    sport_a_p95: float = 1.3  # |a| au 95e centile au-delà duquel → Sportif
    highway_v_mean: float = 95 * KMH
    road_v_mean: float = 50 * KMH  # TODO-thèse
    traffic_v_max: float = 30 * KMH  # TODO-thèse
    traffic_stop_ratio: float = 0.45  # TODO-thèse


def features(v: NDArray[np.float64], dt: float) -> DPRFeatures:
    """Caractéristiques de la fenêtre ``v`` échantillonnée à ``dt``."""
    v = np.asarray(v, dtype=float)
    a = np.diff(v) / dt if len(v) > 1 else np.zeros(1)
    a_pos = a[a > 0.05]
    return DPRFeatures(
        v_mean=float(v.mean()),
        v_max=float(v.max()),
        v_std=float(v.std()),
        a_pos_mean=float(a_pos.mean()) if len(a_pos) else 0.0,
        a_abs_p95=float(np.percentile(np.abs(a), 95)),
        stop_ratio=float(np.mean(v < _STOP_SPEED)),
    )


def classify(f: DPRFeatures, th: DPRThresholds | None = None) -> DrivingPattern:
    """Règles à seuils, de la plus discriminante à la plus générique."""
    th = th or DPRThresholds()
    if f.a_abs_p95 > th.sport_a_p95:
        return DrivingPattern.SPORTIF
    if f.v_mean > th.highway_v_mean:
        return DrivingPattern.AUTOROUTE
    if f.v_mean > th.road_v_mean:
        return DrivingPattern.ROUTE
    if f.v_max < th.traffic_v_max or f.stop_ratio > th.traffic_stop_ratio:
        return DrivingPattern.TRAFIC
    return DrivingPattern.URBAIN


class DrivingPatternRecognizer:
    """DPR en ligne : fenêtre glissante sur l'historique de vitesse (pas EMS)."""

    def __init__(
        self,
        window_s: float,
        dt: float,
        thresholds: DPRThresholds | None = None,
        default: DrivingPattern = DrivingPattern.URBAIN,
        min_samples: int = 10,
    ) -> None:
        self.n = max(round(window_s / dt), 2)
        self.dt = dt
        self.th = thresholds or DPRThresholds()
        self.default = default
        # Jamais plus que la fenêtre, sinon le DPR resterait muet (toujours ``default``).
        self.min_samples = min(min_samples, self.n)

    def recognize(self, v_hist: NDArray[np.float64]) -> DrivingPattern:
        """Schéma courant à partir des ``window_s`` dernières secondes de ``v_hist``."""
        window = np.asarray(v_hist[-self.n :], dtype=float)
        if len(window) < self.min_samples:
            return self.default
        return classify(features(window, self.dt), self.th)

    def label_cycle(self, v: NDArray[np.float64]) -> NDArray[np.int64]:
        """Applique le DPR en ligne sur tout un profil (pour évaluer la précision)."""
        return np.array([int(self.recognize(v[: i + 1])) for i in range(len(v))], dtype=np.int64)
