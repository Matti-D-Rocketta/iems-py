"""Prédiction du profil de vitesse sur l'horizon ``Hp``.

v1 : **oracle bruité**. On lit la tranche future du cycle et on la déforme par un biais
multiplicatif lisse qui croît avec l'horizon (marche aléatoire à pas de 1 s), ce qui imite
l'erreur d'un prédicteur réel : exacte à l'instant courant, de plus en plus incertaine au loin.
Le bruit est lisse pour ne pas créer d'accélérations parasites. Graine fixée : déterministe.

v2 : LSTM (torch, optionnel) + filtre exponentiel en post-traitement, même interface.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SpeedPredictor(Protocol):
    """Interface commune des prédicteurs de vitesse."""

    def reset(self) -> None: ...

    def predict(self, i: int) -> FloatArray:
        """Vitesse prédite au pas modèle de ``t_i`` à ``t_i + Hp`` (indice 0 = instant courant)."""
        ...


class NoisyOraclePredictor:
    """Oracle bruité : tranche future du cycle × (1 + biais aléatoire lisse)."""

    def __init__(self, v: FloatArray, dt: float, Hp: float, noise: float, seed: int) -> None:
        self.v = np.asarray(v, dtype=float)
        self.dt = dt
        self.n_h = round(Hp / dt)
        self.noise = noise
        self.seed = seed
        self._n_sec = int(np.ceil(Hp)) + 1
        self._t_sec = np.arange(self._n_sec, dtype=float)
        self._t_fine = np.arange(self.n_h + 1) * dt
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def predict(self, i: int) -> FloatArray:
        end = i + self.n_h + 1
        future = self.v[i:end]
        if len(future) < self.n_h + 1:  # au-delà de la fin du cycle : arrêt maintenu
            future = np.pad(future, (0, self.n_h + 1 - len(future)), mode="edge")
        if self.noise == 0.0:
            return future.copy()
        walk = np.concatenate(([0.0], np.cumsum(self.rng.normal(0.0, self.noise, self._n_sec - 1))))
        bias = np.interp(self._t_fine, self._t_sec, walk)
        return np.maximum(future * (1.0 + bias), 0.0)
