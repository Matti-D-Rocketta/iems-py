"""Vocabulaire des schémas de conduite (DPR) partagé par les cycles, le DPR et la balance."""

from __future__ import annotations

from enum import IntEnum


class DrivingPattern(IntEnum):
    """Les cinq schémas de conduite reconnus par l'iEMS."""

    TRAFIC = 0
    URBAIN = 1
    ROUTE = 2  # route principale
    AUTOROUTE = 3
    SPORTIF = 4

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    DrivingPattern.TRAFIC: "Trafic",
    DrivingPattern.URBAIN: "Urbain",
    DrivingPattern.ROUTE: "Route principale",
    DrivingPattern.AUTOROUTE: "Autoroute",
    DrivingPattern.SPORTIF: "Sportif",
}
