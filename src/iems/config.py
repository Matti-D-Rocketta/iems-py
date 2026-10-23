"""Paramètres du véhicule, des composants et de l'EMS.

Toutes les grandeurs sont en unités SI (W, J, s, kg, m). Les valeurs par défaut sont des
valeurs génériques de la littérature pour une berline FCHEV ; celles qui doivent être
reprises du manuscrit de thèse sont marquées ``# TODO-thèse``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iems.patterns import DrivingPattern

LHV_H2_J_PER_KG = 120e6  # littérature : PCI de l'hydrogène ≈ 120 MJ/kg


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VehicleParams(_Params):
    """Dynamique longitudinale (route plate en v1)."""

    mass_kg: float = Field(1800.0, gt=0)  # TODO-thèse : masse du véhicule d'étude
    Cd: float = Field(0.29, gt=0)  # TODO-thèse
    frontal_area_m2: float = Field(2.4, gt=0)  # TODO-thèse
    Crr: float = Field(0.009, ge=0)  # TODO-thèse : résistance au roulement
    eta_trans: float = Field(0.92, gt=0, le=1)  # TODO-thèse : rendement chaîne de traction
    P_aux_W: float = Field(500.0, ge=0)  # TODO-thèse : auxiliaires
    rho_air: float = 1.2  # littérature : masse volumique de l'air à 20 °C
    g: float = 9.81


class FuelCellParams(_Params):
    """Système pile à combustible : carte de rendement en cloche, bornes et rampe."""

    P_max_W: float = Field(95e3, gt=0)  # TODO-thèse
    P_min_W: float = Field(1e3, ge=0)  # TODO-thèse : puissance de ralenti
    ramp_W_per_s: float = Field(5e3, gt=0)  # TODO-thèse : rampe BOP
    # Carte η(P/P_max) en cloche : ≈0,40 à 3 %, pic ≈0,58 vers 20 %, ≈0,45 à pleine charge
    # (littérature : allure typique d'un système PEMFC automobile avec auxiliaires). Les points
    # sont cohérents avec une consommation convexe à débit de ralenti non nul (≈ 3 kW d'H2,
    # cf. fuel_cell.py), d'où η ≈ 0,22 à 1 % de charge.
    eta_map_frac: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)  # TODO-thèse
    eta_map_eta: tuple[float, ...] = (
        0.221,
        0.480,
        0.555,
        0.580,
        0.571,
        0.536,
        0.491,
        0.450,
    )  # TODO-thèse
    lhv_J_per_kg: float = LHV_H2_J_PER_KG

    @model_validator(mode="after")
    def _check_map(self) -> FuelCellParams:
        if len(self.eta_map_frac) != len(self.eta_map_eta):
            raise ValueError("eta_map_frac et eta_map_eta doivent avoir la même longueur")
        if any(b <= a for a, b in zip(self.eta_map_frac, self.eta_map_frac[1:], strict=False)):
            raise ValueError("eta_map_frac doit être strictement croissante")
        if not all(0 < e < 1 for e in self.eta_map_eta):
            raise ValueError("les rendements doivent être dans ]0, 1[")
        if len(self.eta_map_frac) < 3 or self.eta_map_frac[0] <= 0:
            raise ValueError("au moins 3 points de carte, à puissance strictement positive")
        if self.P_min_W >= self.P_max_W:
            raise ValueError("P_min_W doit être < P_max_W")
        return self


class BatteryParams(_Params):
    """Batterie v1 (modèle énergétique)."""

    E_cap_Wh: float = Field(1600.0, gt=0)  # TODO-thèse : capacité énergétique
    P_dis_max_W: float = Field(30e3, gt=0)  # TODO-thèse
    P_ch_max_W: float = Field(30e3, gt=0)  # TODO-thèse
    soc_min: float = Field(0.30, ge=0, le=1)  # TODO-thèse : fenêtre SOC utile
    soc_max: float = Field(0.80, ge=0, le=1)  # TODO-thèse
    # Largeur (en SOC) de la rampe linéaire qui ramène la puissance admissible à 0
    # au bord de la fenêtre (littérature : dérating BMS usuel de quelques points).
    derate_band: float = Field(0.05, gt=0)  # TODO-thèse
    eta_ch: float = Field(0.95, gt=0, le=1)  # TODO-thèse
    eta_dis: float = Field(0.95, gt=0, le=1)  # TODO-thèse
    # Rampe batterie : très rapide devant la PàC (électronique de puissance).
    ramp_W_per_s: float = Field(300e3, gt=0)  # littérature : ordre de grandeur onduleur/BMS

    @property
    def E_cap_J(self) -> float:
        return self.E_cap_Wh * 3600.0

    @model_validator(mode="after")
    def _check_window(self) -> BatteryParams:
        if self.soc_max - self.soc_min <= 2 * self.derate_band:
            raise ValueError("fenêtre SOC trop étroite pour la bande de dérating")
        return self


def _default_usage_table() -> dict[DrivingPattern, float]:
    # Pourcentage d'utilisation batterie « nominal » par schéma (soutenance : sportif ≈ 85 %,
    # autoroute ≈ 30 %, trafic/urbain intermédiaires).
    return {
        DrivingPattern.TRAFIC: 0.60,  # TODO-thèse
        DrivingPattern.URBAIN: 0.55,  # TODO-thèse
        DrivingPattern.ROUTE: 0.45,  # TODO-thèse
        DrivingPattern.AUTOROUTE: 0.30,  # TODO-thèse
        DrivingPattern.SPORTIF: 0.85,  # TODO-thèse
    }


class EMSParams(_Params):
    """Paramètres de l'EMS (cadences, horizon, co-état, DPR, prédicteur, baseline)."""

    Ts_model: float = Field(0.1, gt=0)  # pas du modèle physique
    Ts_ems: float = Field(1.0, gt=0)  # pas de l'EMS (commandes bloquées entre deux pas)
    Hp: float = Field(10.0, gt=0)  # horizon de prédiction glissant

    # Co-état λ (facteur d'équivalence, sans dimension grâce à k_h, cf. optimizer.py).
    # λ0 calibré en M1 par dichotomie ECMS sur WLTC (voir examples/compare_strategies.py).
    lambda0: float = 1.55  # TODO-thèse : recalé par calibration ecms_const (M1)
    Kp: float = 5.0  # TODO-thèse : gain proportionnel d'adaptation du co-état
    lambda_min: float = 0.5
    lambda_max: float = 5.0

    # Balance : usage batterie = table(schéma) modulée par SOC et vitesse normalisés.
    usage_table: dict[DrivingPattern, float] = Field(default_factory=_default_usage_table)
    usage_soc_gain: float = 3.0  # TODO-thèse : pente de l'usage vs écart de SOC normalisé
    soc_ref: float = 0.60  # SOC de référence à l'arrêt (milieu haut de la fenêtre) — TODO-thèse
    # Fraction de l'énergie cinétique qui revient effectivement dans la batterie au freinage
    # (pertes aéro/roulement pendant le freinage, η_trans, η_ch, écrêtage à P_ch_max).
    kinetic_recovery: float = Field(0.75, ge=0, le=1)  # TODO-thèse : ≈ 0,8 mesuré sur WLTC
    v_norm_max: float = 36.1  # 130 km/h : vitesse de normalisation

    # Admissibilité : marge de sécurité appliquée aux limites batterie dans l'anticipation.
    safety_margin: float = Field(0.05, ge=0, lt=1)  # relative
    safety_margin_W: float = Field(1e3, ge=0)  # absolue (W)
    prediction_margin: float = Field(0.05, ge=0, lt=1)  # relative à la demande prédite
    prediction_margin_accel: float = Field(0.10, ge=0)  # m/s² : incertitude d'accélération à 1 s

    # DPR v1 (règles à seuils sur fenêtre glissante).
    dpr_window_s: float = Field(60.0, gt=0)

    # Prédicteur v1 (oracle bruité).
    predictor_noise: float = Field(0.03, ge=0)  # écart-type du biais d'amplitude à t+Hp
    predictor_time_warp: float = Field(0.03, ge=0)  # écart-type de l'étirement temporel
    seed: int = 42

    # Baseline « power follower » + hystérésis SOC.
    baseline_tau_s: float = Field(5.0, gt=0)  # constante de temps du filtre passe-bas
    baseline_soc_low: float = 0.50  # entrée en mode recharge
    baseline_soc_high: float = 0.62  # sortie du mode recharge
    baseline_charge_W: float = 8e3  # surplus PàC en mode recharge

    @model_validator(mode="after")
    def _check_rates(self) -> EMSParams:
        ratio = self.Ts_ems / self.Ts_model
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError("Ts_ems doit être un multiple entier de Ts_model")
        if self.Hp < self.Ts_ems:
            raise ValueError("Hp doit être >= Ts_ems")
        return self

    @property
    def substeps(self) -> int:
        """Nombre de pas modèle par pas EMS."""
        return round(self.Ts_ems / self.Ts_model)


class IEMSConfig(_Params):
    """Configuration complète d'une simulation."""

    vehicle: VehicleParams = Field(default_factory=VehicleParams)
    fuel_cell: FuelCellParams = Field(default_factory=FuelCellParams)
    battery: BatteryParams = Field(default_factory=BatteryParams)
    ems: EMSParams = Field(default_factory=EMSParams)
    soc0: float = Field(0.60, ge=0, le=1)  # SOC initial

    @model_validator(mode="after")
    def _check_soc0(self) -> IEMSConfig:
        if not self.battery.soc_min <= self.soc0 <= self.battery.soc_max:
            raise ValueError("soc0 hors de la fenêtre SOC utile")
        return self

    def with_updates(self, **sections: dict) -> IEMSConfig:
        """Copie avec mise à jour partielle, ex. ``cfg.with_updates(ems={"Kp": 5})``."""
        data = self.model_dump()
        for key, upd in sections.items():
            if isinstance(upd, dict):
                data[key].update(upd)
            else:
                data[key] = upd
        return IEMSConfig.model_validate(data)
