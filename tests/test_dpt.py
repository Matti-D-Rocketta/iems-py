"""Tests du module DPT : oracle bruité (``dpt/predictor.py``) et DPR à seuils (``dpt/dpr.py``).

Conventions : unités SI (m/s, m/s², s), tolérances explicites, aucune fixture partagée hors de
ce fichier. Les cycles synthétiques portent la vérité terrain du DPR dans ``Cycle.labels``.
"""

from __future__ import annotations

import numpy as np
import pytest

from iems.dpt.dpr import (
    KMH,
    DPRFeatures,
    DPRThresholds,
    DrivingPatternRecognizer,
    classify,
    features,
)
from iems.dpt.predictor import NoisyOraclePredictor
from iems.models.cycles import get_cycle
from iems.patterns import DrivingPattern

HP_S = 10.0  # horizon de prédiction (s), valeur par défaut d'EMSParams.Hp
DT_EMS = 1.0  # pas EMS (s) : cycles 1 Hz
DT_MODEL = 0.1  # pas modèle (s) : celui que le simulateur passe au prédicteur
NOISE_DEFAULT = 0.03  # EMSParams.predictor_noise
TIME_WARP_DEFAULT = 0.03  # EMSParams.predictor_time_warp
SEED = 42


# --------------------------------------------------------------------------------------
# Fixtures locales
# --------------------------------------------------------------------------------------
@pytest.fixture
def wltc_v() -> np.ndarray:
    """Profil WLTC 1 Hz (m/s), copie fraîche pour chaque test."""
    return get_cycle("wltc").v.copy()


@pytest.fixture
def wltc_v_fine() -> np.ndarray:
    """Profil WLTC rééchantillonné au pas modèle (0,1 s)."""
    return get_cycle("wltc").resample(DT_MODEL).v.copy()


def _oracle(
    v: np.ndarray,
    noise: float,
    time_warp: float,
    seed: int = SEED,
    dt: float = DT_EMS,
    Hp: float = HP_S,
) -> NoisyOraclePredictor:
    return NoisyOraclePredictor(v, dt, Hp, noise, seed, time_warp)


def _f(
    v_mean: float,
    v_max: float,
    a_abs_p95: float = 0.5,
    stop_ratio: float = 0.0,
    v_std: float = 2.0,
    a_pos_mean: float = 0.5,
) -> DPRFeatures:
    """Construit des DPRFeatures à la main (vitesses en m/s, accélérations en m/s²)."""
    return DPRFeatures(
        v_mean=v_mean,
        v_max=v_max,
        v_std=v_std,
        a_pos_mean=a_pos_mean,
        a_abs_p95=a_abs_p95,
        stop_ratio=stop_ratio,
    )


# --------------------------------------------------------------------------------------
# (a) NoisyOraclePredictor
# --------------------------------------------------------------------------------------
def test_predict_sans_bruit_renvoie_tranche_exacte(wltc_v):
    """Sans bruit ni étirement, predict(i) est exactement v[i : i + n_h + 1]."""
    p = _oracle(wltc_v, noise=0.0, time_warp=0.0)
    assert p.n_h == 10
    for i in (0, 1, 250, 1000, 1700, len(wltc_v) - p.n_h - 1):
        pred = p.predict(i)
        assert pred.shape == (p.n_h + 1,)
        assert np.array_equal(pred, wltc_v[i : i + p.n_h + 1])


def test_predict_sans_bruit_pas_modele_renvoie_tranche_exacte(wltc_v_fine):
    """Au pas modèle (0,1 s), n_h = 100 et la tranche exacte est renvoyée sans bruit."""
    p = _oracle(wltc_v_fine, noise=0.0, time_warp=0.0, dt=DT_MODEL)
    assert p.n_h == 100
    for i in (0, 3000, 12345):
        assert np.array_equal(p.predict(i), wltc_v_fine[i : i + 101])


def test_predict_sans_bruit_ne_partage_pas_la_memoire_du_cycle(wltc_v):
    """La prédiction sans bruit est une copie : la modifier ne touche pas le cycle."""
    p = _oracle(wltc_v, noise=0.0, time_warp=0.0)
    ref = wltc_v.copy()
    pred = p.predict(300)
    pred[:] = -1.0
    assert np.array_equal(wltc_v, ref)


def test_predict_fin_de_cycle_longueur_et_padding_bord(wltc_v):
    """En fin de cycle, la longueur reste n_h + 1 et les points manquants valent v[-1]."""
    p = _oracle(wltc_v, noise=0.0, time_warp=0.0)
    n = len(wltc_v)
    for i in (n - 1, n - 4, n - p.n_h):
        pred = p.predict(i)
        assert pred.shape == (p.n_h + 1,)
        n_real = n - i
        assert np.array_equal(pred[:n_real], wltc_v[i:])
        assert np.all(pred[n_real:] == wltc_v[-1])


def test_predict_avec_bruit_fin_de_cycle_longueur_constante(wltc_v):
    """Avec bruit, la longueur reste n_h + 1 même quand l'horizon dépasse la fin du cycle."""
    p = _oracle(wltc_v, noise=0.5, time_warp=0.3)
    n = len(wltc_v)
    for i in (n - 1, n - 3, n - p.n_h - 1):
        pred = p.predict(i)
        assert pred.shape == (p.n_h + 1,)
        assert np.all(np.isfinite(pred))


def test_predict_avec_bruit_valeur_courante_exacte(wltc_v):
    """Avec bruit, predict(i)[0] vaut exactement v[i] : la vitesse courante est mesurée."""
    p = _oracle(wltc_v, noise=0.1, time_warp=0.1)
    for i in range(len(wltc_v) - p.n_h):
        assert p.predict(i)[0] == wltc_v[i]


@pytest.mark.parametrize(
    ("dt", "noise", "time_warp", "seed"),
    [
        (DT_EMS, NOISE_DEFAULT, TIME_WARP_DEFAULT, 0),
        (DT_EMS, NOISE_DEFAULT, TIME_WARP_DEFAULT, 123),
        (DT_EMS, 0.1, 0.1, SEED),
        (DT_MODEL, NOISE_DEFAULT, TIME_WARP_DEFAULT, SEED),
    ],
)
def test_predict_avec_bruit_pente_initiale_preservee_a_l_ordre_2(dt, noise, time_warp, seed):
    """La pente au 1er pas n'est perturbée que par les termes d'ordre 2 : O(β/n_h) et O(γ/n_h²).

    Les erreurs sont nulles en valeur et en dérivée à τ = 0 (docstring du module) ; la
    différence finie pred[1] − pred[0] porte donc uniquement les termes du second ordre :
    (β/n_h)·Δv_voisin pour le calendrier et (γ/n_h²)·v pour l'amplitude, soit un facteur n_h
    (resp. n_h²) sous l'accélération fantôme γ·v/n_h qu'imposerait un biais linéaire en τ.
    β et γ sont des AR(1) stationnaires d'écart-type time_warp et noise : on borne par 5σ et 6σ.
    """
    v = get_cycle("wltc").resample(dt).v if dt != DT_EMS else get_cycle("wltc").v
    p = _oracle(v, noise=noise, time_warp=time_warp, seed=seed, dt=dt)
    n_h = p.n_h
    for i in range(0, len(v) - n_h - 2, 3):
        pred = p.predict(i)
        f0, f1, f2 = v[i], v[i + 1], v[i + 2]
        err = (pred[1] - pred[0]) - (f1 - f0)
        tol_warp = 5.0 * time_warp / n_h * max(abs(f2 - f1), abs(f1 - f0))
        tol_amp = 6.0 * noise / n_h**2 * max(f0, f1, f2)
        assert abs(err) <= tol_warp + tol_amp + 1e-12, f"i={i}: err={err:.3e}"


def test_predict_biais_amplitude_quadratique_en_tau():
    """À vitesse constante et sans étirement, l'erreur relative vaut exactement γ·(τ/Hp)²."""
    c = 20.0
    v = np.full(200, c)
    p = _oracle(v, noise=0.05, time_warp=0.0, seed=3)
    k = np.arange(p.n_h + 1, dtype=float)
    for i in range(0, 50, 7):
        pred = p.predict(i)
        gamma = pred[-1] / c - 1.0
        assert gamma != 0.0
        assert np.allclose((pred - c) / c, gamma * (k / p.n_h) ** 2, rtol=0.0, atol=1e-9)
    # conséquence : la pente au premier pas n'est perturbée que de c·γ/n_h²
    pred = p.predict(60)
    assert abs(pred[1] - pred[0]) <= 5 * 0.05 * c / p.n_h**2


def test_predict_erreur_calendrier_quadratique_en_tau():
    """Sur une rampe linéaire et sans bruit d'amplitude, l'erreur croît comme τ² (ordre 2)."""
    a0 = 0.8  # m/s²
    v = a0 * np.arange(400, dtype=float)
    p = _oracle(v, noise=0.0, time_warp=0.1, seed=5, Hp=20.0)
    k = np.arange(1, p.n_h // 2 + 1)  # moitié d'horizon : τ étiré jamais écrêté (|β| ≤ 0,5)
    for i in (50, 100, 200):
        pred = p.predict(i)
        err = pred[k] - v[i + k]
        assert err[0] != 0.0
        assert np.allclose(err / err[0], k.astype(float) ** 2, rtol=1e-8, atol=1e-8)


def test_predict_vitesses_toujours_positives(wltc_v):
    """Même avec un bruit d'amplitude énorme, aucune vitesse prédite n'est négative."""
    p = _oracle(wltc_v, noise=3.0, time_warp=0.3, seed=9)
    for i in range(0, len(wltc_v), 5):
        assert np.all(p.predict(i) >= 0.0)


def test_predict_meme_graine_memes_sequences_apres_reset(wltc_v):
    """Même graine → mêmes séquences, et reset() rejoue la séquence à l'identique."""
    p1 = _oracle(wltc_v, noise=NOISE_DEFAULT, time_warp=TIME_WARP_DEFAULT, seed=7)
    p2 = _oracle(wltc_v, noise=NOISE_DEFAULT, time_warp=TIME_WARP_DEFAULT, seed=7)
    first = [p1.predict(i) for i in range(100, 130)]
    for _ in range(25):  # fait avancer l'AR(1) de p1 avant le reset
        p1.predict(500)
    p1.reset()
    replay = [p1.predict(i) for i in range(100, 130)]
    fresh = [p2.predict(i) for i in range(100, 130)]
    for a, b, c in zip(first, replay, fresh, strict=True):
        assert np.array_equal(a, b)
        assert np.array_equal(a, c)


def test_predict_graines_differentes_sequences_differentes(wltc_v):
    """Deux graines différentes donnent des prédictions différentes sur une zone roulante."""
    p1 = _oracle(wltc_v, noise=NOISE_DEFAULT, time_warp=TIME_WARP_DEFAULT, seed=1)
    p2 = _oracle(wltc_v, noise=NOISE_DEFAULT, time_warp=TIME_WARP_DEFAULT, seed=2)
    idx = [i for i in range(len(wltc_v) - 11) if wltc_v[i + 10] > 5.0][:20]
    assert any(not np.array_equal(p1.predict(i), p2.predict(i)) for i in idx)


@pytest.mark.parametrize(
    ("noise", "time_warp"),
    [
        (NOISE_DEFAULT, 0.0),
        (0.1, 0.0),
        (NOISE_DEFAULT, TIME_WARP_DEFAULT),
        (0.1, TIME_WARP_DEFAULT),
    ],
)
def test_predict_ecart_type_erreur_bout_horizon_de_l_ordre_du_bruit(wltc_v, noise, time_warp):
    """L'écart-type de l'erreur relative en bout d'horizon vaut noise à un facteur 0,3-3 près."""
    p = _oracle(wltc_v, noise=noise, time_warp=time_warp)
    errs = []
    for i in range(len(wltc_v) - p.n_h):
        v_end = wltc_v[i + p.n_h]
        if v_end > 1.0:  # erreur relative définie hors arrêt
            errs.append((p.predict(i)[-1] - v_end) / v_end)
    assert len(errs) > 1000
    std = float(np.std(errs))
    assert 0.3 * noise <= std <= 3.0 * noise, f"std={std:.4f} pour noise={noise}"


# --------------------------------------------------------------------------------------
# (b) dpr.features
# --------------------------------------------------------------------------------------
def test_features_vitesse_constante_std_et_accelerations_nulles():
    """Vitesse constante : v_std = 0, a_pos_mean = 0, a_abs_p95 = 0, stop_ratio = 0."""
    c = 15.0
    f = features(np.full(60, c), DT_EMS)
    assert f.v_mean == pytest.approx(c, abs=1e-12)
    assert f.v_max == c
    assert f.v_std == 0.0
    assert f.a_pos_mean == 0.0
    assert f.a_abs_p95 == 0.0
    assert f.stop_ratio == 0.0


def test_features_vitesse_nulle_stop_ratio_un():
    """Véhicule à l'arrêt tout le long : stop_ratio = 1 et toutes les vitesses nulles."""
    f = features(np.zeros(60), DT_EMS)
    assert f.stop_ratio == 1.0
    assert f.v_mean == 0.0
    assert f.v_max == 0.0
    assert f.a_pos_mean == 0.0


def test_features_rampe_lineaire_acceleration_en_unites_si():
    """Sur une rampe a0·t échantillonnée à dt = 0,5 s, les accélérations valent a0 (m/s²)."""
    a0, dt = 1.2, 0.5
    v = a0 * dt * np.arange(30, dtype=float)
    f = features(v, dt)
    assert f.a_pos_mean == pytest.approx(a0, rel=1e-9)
    assert f.a_abs_p95 == pytest.approx(a0, rel=1e-9)
    assert f.v_max == pytest.approx(v[-1], rel=1e-12)
    # seul le premier échantillon (v = 0) est sous le seuil d'arrêt de 0,5 m/s
    assert f.stop_ratio == pytest.approx(1 / 30, rel=1e-12)


def test_features_stop_ratio_est_la_fraction_sous_seuil_arret():
    """stop_ratio compte les échantillons sous 0,5 m/s : moitié arrêt, moitié roulage → 0,5."""
    v = np.concatenate([np.zeros(40), np.full(40, 10.0)])
    f = features(v, DT_EMS)
    assert f.stop_ratio == pytest.approx(0.5, abs=1e-12)
    assert f.v_mean == pytest.approx(5.0, abs=1e-12)


def test_features_fenetre_d_un_echantillon_sans_nan():
    """Une fenêtre d'un seul échantillon donne des caractéristiques finies, accélérations nulles."""
    f = features(np.array([12.0]), DT_EMS)
    assert f.v_mean == 12.0
    assert f.a_pos_mean == 0.0
    assert f.a_abs_p95 == 0.0
    assert all(np.isfinite(x) for x in vars(f).values())


# --------------------------------------------------------------------------------------
# (c) dpr.classify
# --------------------------------------------------------------------------------------
def test_thresholds_par_defaut_en_metres_par_seconde():
    """Les seuils de vitesse par défaut sont en m/s (95/50/30 km/h) et ordonnés."""
    th = DPRThresholds()
    assert th.highway_v_mean == pytest.approx(95 * KMH, rel=1e-12)
    assert th.road_v_mean == pytest.approx(50 * KMH, rel=1e-12)
    assert th.traffic_v_max == pytest.approx(30 * KMH, rel=1e-12)
    assert th.highway_v_mean > th.road_v_mean > th.traffic_v_max
    assert 0.0 < th.traffic_stop_ratio < 1.0


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        pytest.param(
            _f(v_mean=10.0, v_max=15.0, a_abs_p95=2.0), DrivingPattern.SPORTIF, id="sport"
        ),
        pytest.param(_f(v_mean=30.0, v_max=35.0), DrivingPattern.AUTOROUTE, id="autoroute"),
        pytest.param(_f(v_mean=18.0, v_max=25.0), DrivingPattern.ROUTE, id="route"),
        pytest.param(_f(v_mean=5.0, v_max=7.0), DrivingPattern.TRAFIC, id="trafic_v_max"),
        pytest.param(
            _f(v_mean=8.0, v_max=12.0, stop_ratio=0.5), DrivingPattern.TRAFIC, id="trafic_stop"
        ),
        pytest.param(
            _f(v_mean=8.0, v_max=12.0, stop_ratio=0.2), DrivingPattern.URBAIN, id="urbain"
        ),
    ],
)
def test_classify_chaque_schema_atteignable(f, expected):
    """Chaque schéma est atteint par la règle qui lui est propre (seuils par défaut)."""
    assert classify(f) is expected


def test_classify_priorite_sportif_sur_autoroute():
    """Un |a| au 95e centile élevé l'emporte sur une vitesse moyenne d'autoroute."""
    f = _f(v_mean=30.0, v_max=36.0, a_abs_p95=2.5)
    assert classify(f) is DrivingPattern.SPORTIF
    assert classify(_f(v_mean=30.0, v_max=36.0, a_abs_p95=0.5)) is DrivingPattern.AUTOROUTE


def test_classify_priorite_vitesse_moyenne_sur_criteres_trafic():
    """Une vitesse moyenne de route l'emporte sur un stop_ratio de trafic."""
    f = _f(v_mean=18.0, v_max=25.0, stop_ratio=0.6)
    assert classify(f) is DrivingPattern.ROUTE


def test_classify_egalite_au_seuil_ne_declenche_pas():
    """Les règles sont strictes : une valeur égale au seuil ne bascule pas le schéma."""
    th = DPRThresholds()
    assert classify(_f(v_mean=10.0, v_max=15.0, a_abs_p95=th.sport_a_p95)) is DrivingPattern.URBAIN
    assert classify(_f(v_mean=th.highway_v_mean, v_max=30.0)) is DrivingPattern.ROUTE
    assert classify(_f(v_mean=8.0, v_max=12.0, stop_ratio=th.traffic_stop_ratio)) is (
        DrivingPattern.URBAIN
    )


def test_classify_seuils_personnalises_pris_en_compte():
    """Des seuils fournis explicitement remplacent les seuils par défaut."""
    f = _f(v_mean=10.0, v_max=15.0, a_abs_p95=1.0)
    assert classify(f) is DrivingPattern.URBAIN
    assert classify(f, DPRThresholds(sport_a_p95=0.9)) is DrivingPattern.SPORTIF
    assert classify(f, DPRThresholds(road_v_mean=9.0)) is DrivingPattern.ROUTE


def test_classify_renvoie_un_membre_de_l_enum():
    """classify renvoie toujours un DrivingPattern (IntEnum), jamais un int nu."""
    out = classify(_f(v_mean=8.0, v_max=12.0))
    assert isinstance(out, DrivingPattern)
    assert out.label == "Urbain"


# --------------------------------------------------------------------------------------
# (d) DrivingPatternRecognizer
# --------------------------------------------------------------------------------------
def test_recognizer_moins_de_min_samples_renvoie_default():
    """Avec moins de min_samples échantillons, le DPR renvoie le schéma par défaut."""
    rec = DrivingPatternRecognizer(60.0, DT_EMS)
    assert rec.min_samples == 10
    assert rec.recognize(np.full(9, 30.0)) is DrivingPattern.URBAIN  # sinon → AUTOROUTE
    assert rec.recognize(np.array([])) is DrivingPattern.URBAIN
    assert rec.recognize(np.full(10, 30.0)) is DrivingPattern.AUTOROUTE
    rec_t = DrivingPatternRecognizer(60.0, DT_EMS, default=DrivingPattern.TRAFIC, min_samples=20)
    assert rec_t.recognize(np.full(19, 30.0)) is DrivingPattern.TRAFIC
    assert rec_t.recognize(np.full(20, 30.0)) is DrivingPattern.AUTOROUTE


def test_recognizer_fenetre_glissante_ignore_historique_ancien():
    """Seules les window_s dernières secondes comptent : un vieux passage autoroutier est oublié."""
    rec = DrivingPatternRecognizer(60.0, DT_EMS)
    assert rec.n == 60
    highway_then_stop = np.concatenate([np.full(200, 35.0), np.zeros(60)])
    assert rec.recognize(highway_then_stop) is DrivingPattern.TRAFIC
    # arrêt plus court que la fenêtre : la moitié autoroutière reste visible → pas TRAFIC
    assert rec.recognize(np.concatenate([np.full(200, 35.0), np.zeros(30)])) is not (
        DrivingPattern.TRAFIC
    )


def test_recognizer_taille_fenetre_en_echantillons():
    """n = window_s / dt arrondi, au minimum 2 échantillons."""
    assert DrivingPatternRecognizer(60.0, DT_EMS).n == 60
    assert DrivingPatternRecognizer(60.0, DT_MODEL).n == 600
    assert DrivingPatternRecognizer(0.5, DT_EMS).n == 2


def test_label_cycle_longueur_egale_a_v():
    """label_cycle renvoie une étiquette entière par échantillon de v."""
    cycle = get_cycle("synth_urban")
    labels = DrivingPatternRecognizer(60.0, DT_EMS).label_cycle(cycle.v)
    assert labels.shape == cycle.v.shape
    assert labels.dtype == np.int64
    valid = {int(p) for p in DrivingPattern}
    assert set(np.unique(labels).tolist()) <= valid


def test_label_cycle_debut_au_schema_par_defaut():
    """Les min_samples − 1 premiers échantillons portent le schéma par défaut."""
    rec = DrivingPatternRecognizer(60.0, DT_EMS, default=DrivingPattern.ROUTE)
    labels = rec.label_cycle(get_cycle("synth_highway").v)
    assert np.all(labels[: rec.min_samples - 1] == int(DrivingPattern.ROUTE))


@pytest.mark.parametrize(
    ("name", "min_accuracy"),
    [
        ("synth_urban", 0.85),
        ("synth_highway", 0.85),
        ("synth_mixed", 0.70),
        # Sportif : seuils TODO-thèse en cours de recalage (précision mesurée ≈ 0,58) ;
        # on ne garantit qu'un plancher bas en attendant les centres de classes FCM.
        ("synth_sport", 0.45),
    ],
)
def test_recognizer_precision_sur_cycles_synthetiques(name, min_accuracy):
    """La précision du DPR face à la vérité terrain des cycles synthétiques dépasse le plancher."""
    cycle = get_cycle(name)
    assert cycle.labels is not None
    labels = DrivingPatternRecognizer(60.0, DT_EMS).label_cycle(cycle.v)
    accuracy = float(np.mean(labels == cycle.labels))
    assert accuracy >= min_accuracy, f"{name}: précision {accuracy:.3f} < {min_accuracy}"


# --------------------------------------------------------------------------------------
# (e) Déterminisme et pureté
# --------------------------------------------------------------------------------------
def test_recognizer_deterministe_et_pur():
    """Deux appels identiques donnent le même résultat et l'historique n'est pas modifié."""
    cycle = get_cycle("synth_mixed")
    v_ref = cycle.v.copy()
    rec = DrivingPatternRecognizer(60.0, DT_EMS)
    first = rec.label_cycle(cycle.v)
    second = rec.label_cycle(cycle.v)
    assert np.array_equal(first, second)
    assert np.array_equal(cycle.v, v_ref)
    other = DrivingPatternRecognizer(60.0, DT_EMS).label_cycle(cycle.v)
    assert np.array_equal(first, other)


def test_recognize_independant_de_l_ordre_des_appels():
    """recognize ne garde aucun état : le résultat ne dépend pas des appels précédents."""
    rec = DrivingPatternRecognizer(60.0, DT_EMS)
    urban = np.concatenate([np.zeros(20), np.full(40, 10.0)])
    highway = np.full(60, 33.0)
    before = rec.recognize(urban)
    for _ in range(5):
        rec.recognize(highway)
    assert rec.recognize(urban) is before
    assert before is DrivingPattern.URBAIN
