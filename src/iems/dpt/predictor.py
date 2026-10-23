"""Prédiction du profil de vitesse sur l'horizon ``Hp``.

v1 : **oracle bruité**. On lit la tranche future du cycle et on la déforme de deux façons,
tirées à chaque appel (graine fixée : déterministe) :

* **erreur de calendrier** : la tranche est étirée/comprimée dans le temps d'un facteur
  ``1 + β`` (une accélération prévue arrive un peu trop tôt ou trop tard) ;
* **erreur d'amplitude** : biais multiplicatif ``γ·τ/Hp`` qui croît linéairement de 0 (instant
  courant exact) à ``γ`` en bout d'horizon.

β et γ suivent des processus AR(1) d'un appel à l'autre (corrélation ``rho``) : comme un
prédicteur réel, l'erreur est persistante, elle ne saute pas aléatoirement d'une seconde à
l'autre (un bruit indépendant à chaque appel fait osciller les décisions de l'EMS).

Pourquoi pas un bruit blanc ou une marche aléatoire sur la vitesse : la demande de puissance
dépend de l'accélération, donc de la *dérivée* de la vitesse prédite. Un bruit qui oscille
seconde par seconde crée des accélérations fantômes (± 0,9 m/s² à 110 km/h pour 3 %/s, soit
± 50 kW) qui n'ont rien de réaliste. Les deux erreurs ci-dessus sont lisses : l'accélération
prédite reste celle du cycle, à ``(1+β)`` près, plus ``γ·v/Hp``.

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
    """Oracle bruité : tranche future du cycle, étirée dans le temps et biaisée en amplitude."""

    def __init__(
        self,
        v: FloatArray,
        dt: float,
        Hp: float,
        noise: float,
        seed: int,
        time_warp: float = 0.10,
        rho: float = 0.95,
    ) -> None:
        self.v = np.asarray(v, dtype=float)
        self.dt = dt
        self.n_h = round(Hp / dt)
        self.noise = noise  # écart-type du biais d'amplitude en bout d'horizon
        self.time_warp = time_warp  # écart-type de l'étirement temporel
        self.seed = seed
        self.rho = rho
        self._tau = np.arange(self.n_h + 1, dtype=float)  # en pas modèle
        self._ramp = self._tau / max(self.n_h, 1)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._beta = float(self.rng.normal(0.0, self.time_warp))
        self._gamma = float(self.rng.normal(0.0, self.noise))

    def predict(self, i: int) -> FloatArray:
        end = i + self.n_h + 1
        future = self.v[i:end]
        if len(future) < self.n_h + 1:  # au-delà de la fin du cycle : arrêt maintenu
            future = np.pad(future, (0, self.n_h + 1 - len(future)), mode="edge")
        if self.noise == 0.0 and self.time_warp == 0.0:
            return future.copy()
        innov = np.sqrt(1.0 - self.rho**2)
        self._beta = self.rho * self._beta + innov * float(self.rng.normal(0.0, self.time_warp))
        self._gamma = self.rho * self._gamma + innov * float(self.rng.normal(0.0, self.noise))
        beta = float(np.clip(self._beta, -0.5, 0.5))
        gamma = self._gamma
        warped = np.interp(np.minimum(self._tau * (1.0 + beta), self.n_h), self._tau, future)
        return np.maximum(warped * (1.0 + gamma * self._ramp), 0.0)
