"""Cycles de conduite : WLTC classe 3 embarqué et générateur synthétique étiqueté.

**WLTC** : le fichier ``data/wltc_class3.csv`` est une *reconstruction* du WLTC classe 3b qui
respecte la structure publiée du cycle (GTR n° 15 : 4 phases Low/Medium/High/Extra-high de
589/433/455/323 s, distances 3,095/4,756/7,162/8,254 km, vitesses max 56,5/76,6/97,4/131,3 km/h,
arrêt en fin de phase). Ce n'est pas la table officielle point à point (indisponible hors
ligne) : déposer la table officielle (colonnes ``t_s,v_kmh``) au même chemin la remplace sans
autre changement. Le CSV est produit par :func:`build_wltc_class3_approx`.

**Synthétique** : concaténation de segments typés (trafic, urbain, route, autoroute, sportif)
construits par micro-trajets (accélération plafonnée en puissance, croisière ondulée,
décélération, arrêt). Chaque échantillon porte l'étiquette de son segment : c'est la vérité
terrain du DPR. Générateur déterministe à graine fixée.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from iems.patterns import DrivingPattern

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

KMH = 1.0 / 3.6
_MIN_DV = 3.0  # m/s : écart minimal entre deux vitesses cibles successives
_MASS_REF_KG = 1800.0  # masse de référence pour plafonner l'accélération en puissance


@dataclass(frozen=True)
class Cycle:
    """Profil de vitesse échantillonné (t en s, v en m/s), éventuellement étiqueté."""

    name: str
    t: FloatArray
    v: FloatArray
    labels: IntArray | None = None  # DrivingPattern par échantillon (vérité terrain)
    description: str = ""
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def distance_m(self) -> float:
        return float(np.trapezoid(self.v, self.t))

    def resample(self, dt: float) -> Cycle:
        """Rééchantillonne sur une grille régulière de pas ``dt`` (interpolation linéaire).

        Linéaire, comme le prescrit la procédure WLTP pour les cycles à 1 Hz : l'accélération
        est constante entre deux points, donc la puissance demandée ne dépasse jamais celle
        que le profil 1 Hz implique. Une interpolation cubique (même monotone) peut tripler
        localement la dérivée et créer des pics de puissance qui n'existent pas dans le cycle.
        """
        n = round(self.duration / dt) + 1
        t_new = self.t[0] + dt * np.arange(n)
        v_new = np.interp(t_new, self.t, self.v)
        labels = None
        if self.labels is not None:
            idx = np.clip(np.searchsorted(self.t, t_new, side="right") - 1, 0, len(self.t) - 1)
            labels = self.labels[idx]
        return Cycle(self.name, t_new, v_new, labels, self.description, dict(self.meta))


# --------------------------------------------------------------------------------------
# Briques de construction de profils (1 Hz)
# --------------------------------------------------------------------------------------
def _ramp(
    v_from: float, v_to: float, rate: float, p_cap_w: float | None, jerk: float = 1.0
) -> list[float]:
    """Rampe de vitesse 1 Hz de ``v_from`` à ``v_to`` (exclu ``v_from``, inclus ``v_to``).

    L'accélération s'établit et retombe avec un jerk borné (comme un conducteur réel) :
    sans cela, la demande passe de 5 à 55 kW en un échantillon, ce qu'aucune PàC limitée
    en rampe ne peut suivre. En accélération, le taux est aussi plafonné par
    ``p_cap_w / (m·v)`` (puissance roue max).
    """
    out: list[float] = []
    v, a = v_from, 0.0
    sign = 1.0 if v_to >= v_from else -1.0
    while sign * (v_to - v) > 1e-6:
        a_cap = rate
        if p_cap_w is not None and sign > 0:
            a_cap = min(a_cap, p_cap_w / (_MASS_REF_KG * max(v, 3.0)))
        remaining = sign * (v_to - v)
        a_land = max(np.sqrt(2.0 * jerk * remaining), 0.1)  # retombée douce de l'accélération
        a = min(a + jerk, a_cap, a_land)
        v = v + sign * a
        if sign * (v_to - v) < 0:
            v = v_to
        out.append(v)
    return out


def _cruise(v: float, duration: int, amp: float) -> list[float]:
    """Croisière ondulée sous ``v`` qui commence et finit exactement à ``v``.

    Période ≥ 40 s environ ; l'amplitude est réduite pour les croisières courtes afin de ne
    pas créer d'accélérations parasites (une ondulation de 4 % sur 8 s à 130 km/h, c'est
    ±25 kW de demande).
    """
    if duration <= 0:
        return []
    n = max(1, round(duration / 40))
    amp_eff = amp * min(1.0, duration / 40.0)
    k = np.arange(1, duration + 1)
    return list(v * (1.0 - amp_eff * 0.5 * (1.0 - np.cos(2.0 * np.pi * n * k / duration))))


@dataclass(frozen=True)
class SegmentSpec:
    """Paramètres statistiques d'un type de segment."""

    v_kmh: tuple[float, float]
    accel: tuple[float, float]
    decel: tuple[float, float]
    cruise_s: tuple[int, int]
    stop_s: tuple[int, int]
    p_stop: float
    amp: float = 0.04
    p_wheel_cap_w: float = 45e3
    jerk: float = 0.8  # m/s³


SEGMENT_SPECS: dict[DrivingPattern, SegmentSpec] = {
    DrivingPattern.TRAFIC: SegmentSpec((6, 22), (0.4, 0.8), (0.5, 0.9), (4, 12), (8, 25), 0.75),
    DrivingPattern.URBAIN: SegmentSpec((30, 50), (0.8, 1.3), (0.8, 1.2), (10, 30), (8, 25), 0.55),
    DrivingPattern.ROUTE: SegmentSpec((60, 90), (0.6, 1.0), (0.6, 1.0), (30, 80), (8, 15), 0.08),
    DrivingPattern.AUTOROUTE: SegmentSpec(
        (105, 130), (0.3, 0.6), (0.3, 0.6), (60, 160), (5, 10), 0.0, amp=0.03
    ),
    # Sportif : fortes accélérations à basse vitesse, freinages appuyés, croisières courtes.
    # Puissance roue plafonnée à 35 kW : au-delà, passer d'une croisière à 5 kW à une pleine
    # accélération en 1-2 s exige plus que PàC (5 kW/s) + batterie (30 kW) ne peuvent fournir.
    DrivingPattern.SPORTIF: SegmentSpec(
        (45, 95),
        (2.0, 2.8),
        (2.5, 3.2),
        (6, 15),
        (2, 5),
        0.15,
        amp=0.02,
        p_wheel_cap_w=35e3,
        jerk=1.5,
    ),
}


def _segment(
    pattern: DrivingPattern, duration: int, v_start: float, rng: np.random.Generator
) -> list[float]:
    """Génère ``duration`` échantillons 1 Hz d'un segment, en partant de ``v_start``."""
    spec = SEGMENT_SPECS[pattern]
    out: list[float] = []
    v = v_start
    while len(out) < duration:
        v_tgt = rng.uniform(*spec.v_kmh) * KMH
        if 0.0 < abs(v_tgt - v) < _MIN_DV:  # pas de micro-impulsion d'accélération
            v_tgt = v + _MIN_DV if v_tgt > v or v + _MIN_DV <= spec.v_kmh[1] * KMH else v - _MIN_DV
        out += _ramp(
            v,
            v_tgt,
            rng.uniform(*spec.accel) if v_tgt > v else rng.uniform(*spec.decel),
            spec.p_wheel_cap_w,
            spec.jerk,
        )
        out += _cruise(v_tgt, int(rng.integers(*spec.cruise_s, endpoint=True)), spec.amp)
        v = out[-1] if out else v
        if rng.random() < spec.p_stop:
            out += _ramp(v, 0.0, rng.uniform(*spec.decel), None, spec.jerk)
            out += [0.0] * int(rng.integers(*spec.stop_s, endpoint=True))
            v = 0.0
    return out[:duration]


def synthetic_cycle(
    segments: Sequence[tuple[DrivingPattern, int]],
    seed: int = 0,
    name: str = "synthetic",
    lead_in_s: int = 5,
) -> Cycle:
    """Concatène des segments typés ``(schéma, durée_s)`` en un cycle étiqueté.

    Le cycle démarre et finit à l'arrêt ; la décélération finale porte l'étiquette du
    dernier segment.
    """
    rng = np.random.default_rng(seed)
    v_list: list[float] = [0.0] * lead_in_s
    lab_list: list[int] = [int(segments[0][0])] * lead_in_s
    v = 0.0
    for pattern, dur in segments:
        seg = _segment(pattern, int(dur), v, rng)
        v_list += seg
        lab_list += [int(pattern)] * len(seg)
        v = seg[-1]
    spec_last = SEGMENT_SPECS[segments[-1][0]]
    tail = _ramp(v, 0.0, float(np.mean(spec_last.decel)), None) + [0.0] * 5
    v_list += tail
    lab_list += [int(segments[-1][0])] * len(tail)
    v_arr = np.asarray(v_list, dtype=float)
    t_arr = np.arange(len(v_arr), dtype=float)
    desc = " + ".join(f"{p.label} {d}s" for p, d in segments)
    return Cycle(name, t_arr, v_arr, np.asarray(lab_list, dtype=np.int64), desc, {"seed": seed})


# --------------------------------------------------------------------------------------
# WLTC classe 3b — reconstruction structurelle
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _WltcPhase:
    name: str
    duration: int
    distance_m: float
    vmax_kmh: float
    # points de passage : (vitesse cible km/h, maintien nominal s). 0 = arrêt.
    waypoints: tuple[tuple[float, float], ...]


_WLTC_ACC = 1.0  # m/s², accélération nominale (plafonnée en puissance)
_WLTC_DEC = 0.8  # m/s²
_WLTC_PCAP_W = 35e3  # puissance roue max allouée à l'accélération (≈ 0,6 m/s² à 110 km/h)

_WLTC_PHASES: tuple[_WltcPhase, ...] = (
    _WltcPhase(
        "Low",
        589,
        3095.0,
        56.5,
        (
            (0, 11),
            (44, 6),
            (20, 4),
            (35, 8),
            (0, 20),
            (30, 6),
            (18, 5),
            (40, 10),
            (0, 25),
            (56.5, 15),
            (38, 6),
            (50, 12),
            (0, 30),
            (25, 5),
            (45, 20),
            (30, 8),
            (0, 18),
            (35, 12),
            (0, 20),
        ),
    ),
    _WltcPhase(
        "Medium",
        433,
        4756.0,
        76.6,
        (
            (0, 10),
            (50, 10),
            (30, 6),
            (65, 20),
            (76.6, 12),
            (55, 10),
            (70, 25),
            (0, 15),
            (45, 8),
            (60, 20),
            (40, 6),
            (0, 10),
        ),
    ),
    _WltcPhase(
        "High",
        455,
        7162.0,
        97.4,
        (
            (0, 10),
            (60, 15),
            (80, 30),
            (65, 10),
            (97.4, 20),
            (85, 40),
            (70, 20),
            (90, 30),
            (50, 10),
            (0, 8),
        ),
    ),
    _WltcPhase(
        "Extra-high",
        323,
        8254.0,
        131.3,
        (
            (0, 8),
            (70, 10),
            (105, 40),
            (120, 30),
            (131.3, 15),
            (110, 20),
            (125, 25),
            (0, 6),
        ),
    ),
)


def _build_phase(ph: _WltcPhase, k: float) -> list[float] | None:
    """Profil 1 Hz d'une phase ; ``k`` met à l'échelle les vitesses autres que le max.

    Les maintiens sont étirés uniformément pour remplir exactement la durée de phase.
    Renvoie ``None`` si les rampes seules dépassent la durée.
    """
    targets: list[float] = []
    ramps: list[list[float]] = []
    v = 0.0
    for v_kmh, _ in ph.waypoints:
        tgt = v_kmh if v_kmh == ph.vmax_kmh else min(v_kmh * k, 0.98 * ph.vmax_kmh)
        tgt *= KMH
        up = tgt > v
        ramps.append(_ramp(v, tgt, _WLTC_ACC if up else _WLTC_DEC, _WLTC_PCAP_W if up else None))
        targets.append(tgt)
        v = tgt
    free = ph.duration - sum(len(r) for r in ramps)
    nominal = [h for _, h in ph.waypoints]
    if free < 2 * len(nominal):
        return None
    scale = free / sum(nominal)
    holds = [max(1, round(h * scale)) for h in nominal]
    holds[-1] += free - sum(holds)
    if holds[-1] < 1:
        return None
    out: list[float] = []
    for ramp, tgt, hold in zip(ramps, targets, holds, strict=True):
        out += ramp
        out += _cruise(tgt, hold, 0.05) if tgt > 0 else [0.0] * hold
    return out


def build_wltc_class3_approx() -> tuple[FloatArray, FloatArray]:
    """Construit la reconstruction WLTC classe 3b (t_s, v_kmh), 1801 points à 1 Hz."""
    v_all: list[float] = [0.0]  # t = 0
    for ph in _WLTC_PHASES:
        lo, hi = 0.3, 1.6
        for _ in range(60):  # dichotomie sur k pour caler la distance de phase
            k = 0.5 * (lo + hi)
            prof = _build_phase(ph, k)
            if prof is not None and float(np.sum(prof)) < ph.distance_m:  # Δt = 1 s
                lo = k
            else:
                hi = k
        prof = _build_phase(ph, lo)
        if prof is None:
            raise ValueError(f"phase {ph.name} : points de passage incompatibles avec la durée")
        v_all += prof
    v_kmh = np.round(np.asarray(v_all) / KMH, 1)
    return np.arange(len(v_kmh), dtype=float), v_kmh


WLTC_PHASE_BOUNDS_S: tuple[int, ...] = (0, 589, 1022, 1477, 1800)


def load_wltc() -> Cycle:
    """Charge le WLTC classe 3 embarqué (``data/wltc_class3.csv``)."""
    path = resources.files("iems") / "data" / "wltc_class3.csv"
    with resources.as_file(path) as p:
        data = np.loadtxt(
            p, delimiter=",", comments="#", skiprows=0, ndmin=2, converters=None, encoding="utf-8"
        )
    t, v = data[:, 0], data[:, 1] * KMH
    return Cycle(
        "wltc",
        t,
        v,
        None,
        "WLTC classe 3b (reconstruction structurelle, cf. cycles.py)",
        {"phase_bounds_s": WLTC_PHASE_BOUNDS_S},
    )


# --------------------------------------------------------------------------------------
# Catalogue des cycles de la campagne M1
# --------------------------------------------------------------------------------------
P = DrivingPattern
_SYNTH: dict[str, tuple[list[tuple[DrivingPattern, int]], int, str]] = {
    "synth_mixed": (
        [
            (P.URBAIN, 240),
            (P.TRAFIC, 180),
            (P.URBAIN, 180),
            (P.ROUTE, 300),
            (P.AUTOROUTE, 360),
            (P.ROUTE, 180),
            (P.SPORTIF, 180),
            (P.URBAIN, 180),
        ],
        1,
        "Cycle mixte : les cinq schémas",
    ),
    "synth_urban": (
        [(P.TRAFIC, 300), (P.URBAIN, 300), (P.TRAFIC, 240), (P.URBAIN, 360), (P.TRAFIC, 300)],
        2,
        "Ville dense : trafic et urbain",
    ),
    "synth_highway": (
        [(P.ROUTE, 300), (P.AUTOROUTE, 600), (P.ROUTE, 240), (P.AUTOROUTE, 480)],
        3,
        "Route principale et autoroute",
    ),
    "synth_sport": (
        [(P.ROUTE, 180), (P.SPORTIF, 300), (P.URBAIN, 180), (P.SPORTIF, 300), (P.ROUTE, 180)],
        4,
        "Conduite sportive",
    ),
}


def list_cycles() -> list[str]:
    return ["wltc", *_SYNTH]


def get_cycle(name: str) -> Cycle:
    """Renvoie un cycle du catalogue par son nom."""
    if name == "wltc":
        return load_wltc()
    if name in _SYNTH:
        segs, seed, desc = _SYNTH[name]
        c = synthetic_cycle(segs, seed=seed, name=name)
        return Cycle(c.name, c.t, c.v, c.labels, f"{desc} — {c.description}", c.meta)
    raise KeyError(f"cycle inconnu : {name!r} (disponibles : {list_cycles()})")
