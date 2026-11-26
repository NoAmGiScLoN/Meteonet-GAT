"""Estimation du mouvement entre deux cartes de pluie par corrélation de phase (tracker v2).

Convention : le déplacement (vy, vx) est en pixels pleine résolution par pas, de la carte a vers
la carte b (b(p) ~ a(p - v)) ; vy > 0 vers le bas (indices de lignes croissants).
"""
import logging
from dataclasses import dataclass

import numpy as np

from scipy.ndimage import distance_transform_edt

from cell_delimitation import MISSING, THRESHOLD_RAW

logger = logging.getLogger(__name__)

RAW_TO_MM_H = 12 / 100  # taux (mm/h) = brut * 12 / 100
DEFAULT_COARSE = 4
# Rapport pic / second maximum : sur 20 bruits purs il reste dans [1,0 ; 1,51], sur carte réelle
# (2016-01-01 12:00) il vaut 6,5. Seuil 2,0 : marge au-dessus du bruit, sous les vrais pics.
DEFAULT_MIN_PEAK_RATIO = 2.0
MIN_RAINY_PIXELS = 30  # pixels pluvieux (pleine résolution) minimum dans chaque carte
PEAK_EXCLUSION = 2  # demi-côté (px grossiers) du voisinage du pic exclu pour le second maximum
EPS = 1e-12
REL_EPS = 1e-6  # plancher relatif du spectre croisé (fraction de son maximum)
TAPER_PX = 6  # largeur (px grossiers) de l'effacement près des pixels manquants
VALID_FRACTION_MIN = 0.5  # fraction de pixels valides pour qu'un bloc grossier soit valide
ALIAS_MARGIN = 2.0  # px grossiers : au-delà de (taille/2 - marge) le décalage est ambigu (repliement)
NULL_MOTION = (0.0, 0.0, 0.0, False)


@dataclass(frozen=True)
class Motion:
    """Déplacement estimé (px pleine résolution par pas), rapport de pic et fiabilité."""
    vy: float
    vx: float
    peak_ratio: float
    reliable: bool


@dataclass(frozen=True, eq=False)  # eq=False : champs ndarray
class GridMotion:
    """Mouvement par tuile : v (n_ty, n_tx, 2) = (vy, vx), reliable (n_ty, n_tx) booléen."""
    v: np.ndarray
    reliable: np.ndarray
    tile: int


def _taper(valid: np.ndarray) -> np.ndarray:
    """Poids 0..1 qui s'annule en douceur (cosinus) près des pixels invalides, 1 au-delà de TAPER_PX."""
    if valid.all():
        return np.ones(valid.shape)
    ramp = np.clip(distance_transform_edt(valid) / TAPER_PX, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * ramp))


def _blocks(values: np.ndarray, coarse: int) -> np.ndarray:
    """Moyenne par blocs coarse x coarse (carte recadrée à un multiple de coarse)."""
    h, w = (values.shape[0] // coarse) * coarse, (values.shape[1] // coarse) * coarse
    return values[:h, :w].reshape(h // coarse, coarse, w // coarse, coarse).mean(axis=(1, 3))


def _prepare(R: np.ndarray, coarse: int) -> tuple[np.ndarray, float]:
    """sqrt(taux) sans manquants, moyenné par blocs, apodisé (Hann x effacement près des -1).

    Le poids annule le champ près des pixels manquants : un bord -1 statique n'injecte pas
    d'énergie haute fréquence (pic parasite en (0, 0)). Retourne (champ, nombre de pixels
    pluvieux pondéré par ce poids, en pixels pleine résolution).
    """
    invalid = R == MISSING
    rate = np.where(invalid, 0.0, np.clip(R.astype(np.float64), 0.0, None)) * RAW_TO_MM_H
    valid = _blocks((~invalid).astype(np.float64), coarse) > VALID_FRACTION_MIN
    ny, nx = valid.shape
    weight = _taper(valid) * np.outer(np.hanning(ny), np.hanning(nx))
    rainy = _blocks((~invalid & (R > THRESHOLD_RAW)).astype(np.float64), coarse)
    return _blocks(np.sqrt(rate), coarse) * weight, float((rainy * weight).sum() * coarse ** 2)


def _parabolic(m1: float, c: float, p1: float) -> float:
    """Décalage sous-pixel (borné à ±0,5) du sommet de la parabole passant par trois échantillons."""
    denom = m1 - 2 * c + p1
    return 0.0 if abs(denom) < EPS else float(np.clip(0.5 * (m1 - p1) / denom, -0.5, 0.5))


def _second_peak(corr: np.ndarray, iy: int, ix: int) -> float:
    """Plus grand échantillon hors du voisinage (circulaire) du pic : second maximum local."""
    masked = corr.copy()
    ys = np.arange(iy - PEAK_EXCLUSION, iy + PEAK_EXCLUSION + 1) % corr.shape[0]
    xs = np.arange(ix - PEAK_EXCLUSION, ix + PEAK_EXCLUSION + 1) % corr.shape[1]
    masked[np.ix_(ys, xs)] = -np.inf
    return float(masked.max())


def _signed(index: float, size: int) -> float:
    """Indice circulaire -> décalage signé dans [-size/2, size/2)."""
    return index - size if index >= size / 2 else index


def phase_correlation(a: np.ndarray, b: np.ndarray, coarse: int = DEFAULT_COARSE,
                      min_peak_ratio: float = DEFAULT_MIN_PEAK_RATIO) -> Motion:
    """Déplacement global de la carte a vers la carte b (px pleine résolution, sur l'intervalle a -> b).

    peak_ratio = pic / second maximum (hors voisinage du pic) de la surface de corrélation.
    Fiable si peak_ratio >= min_peak_ratio, si chaque carte a >= MIN_RAINY_PIXELS pixels pluvieux
    (pondérés par l'apodisation) et si le décalage reste à plus de ALIAS_MARGIN px grossiers de la
    limite de repliement (|décalage| < taille analysée / 2 / coarse). Cartes trop pauvres, trop
    petites ou sans énergie : Motion(0, 0, 0, False).
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Cartes 2-D de même forme requises : {a.shape} vs {b.shape}")
    if coarse < 1:
        raise ValueError(f"coarse doit être >= 1 (reçu {coarse})")
    if min(a.shape) // coarse < 2 * PEAK_EXCLUSION + 2:
        logger.warning("Carte %s trop petite pour coarse=%d", a.shape, coarse)
        return Motion(*NULL_MOTION)
    (pa, rain_a), (pb, rain_b) = _prepare(a, coarse), _prepare(b, coarse)
    if min(rain_a, rain_b) < MIN_RAINY_PIXELS:
        return Motion(*NULL_MOTION)
    cross = np.fft.fft2(pb) * np.conj(np.fft.fft2(pa))
    floor = REL_EPS * float(np.abs(cross).max())
    if not floor > 0:
        return Motion(*NULL_MOTION)
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), floor)).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    ny, nx = corr.shape
    peak, second = float(corr[iy, ix]), _second_peak(corr, int(iy), int(ix))
    ratio = peak / second if second > EPS else float("inf")
    sy = _signed(iy + _parabolic(corr[(iy - 1) % ny, ix], peak, corr[(iy + 1) % ny, ix]), ny)
    sx = _signed(ix + _parabolic(corr[iy, (ix - 1) % nx], peak, corr[iy, (ix + 1) % nx]), nx)
    aliased = abs(sy) > ny / 2 - ALIAS_MARGIN or abs(sx) > nx / 2 - ALIAS_MARGIN
    motion = Motion(float(sy * coarse), float(sx * coarse), float(ratio),
                    bool(ratio >= min_peak_ratio and not aliased))
    logger.debug("Corrélation de phase : %s", motion)
    return motion


def grid_motion(a: np.ndarray, b: np.ndarray, tile: int = 128, coarse: int = DEFAULT_COARSE,
               min_peak_ratio: float = DEFAULT_MIN_PEAK_RATIO) -> GridMotion:
    """Mouvement par tuile ; le reste est fusionné dans la dernière tuile de chaque axe (tout est couvert).

    Les tuiles non fiables reprennent le mouvement global ((0, 0) s'il est lui-même non fiable).
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Cartes 2-D de même forme requises : {a.shape} vs {b.shape}")
    if tile < 1:
        raise ValueError(f"tile doit être >= 1 (reçu {tile})")
    n_ty, n_tx = max(1, a.shape[0] // tile), max(1, a.shape[1] // tile)
    glob = phase_correlation(a, b, coarse, min_peak_ratio)
    fallback = (glob.vy, glob.vx) if glob.reliable else (0.0, 0.0)
    v = np.empty((n_ty, n_tx, 2))
    reliable = np.zeros((n_ty, n_tx), dtype=bool)
    for ty in range(n_ty):
        y1 = a.shape[0] if ty == n_ty - 1 else (ty + 1) * tile
        for tx in range(n_tx):
            x1 = a.shape[1] if tx == n_tx - 1 else (tx + 1) * tile
            sl = (slice(ty * tile, y1), slice(tx * tile, x1))
            m = phase_correlation(a[sl], b[sl], coarse, min_peak_ratio)
            reliable[ty, tx] = m.reliable
            v[ty, tx] = (m.vy, m.vx) if m.reliable else fallback
    return GridMotion(v=v, reliable=reliable, tile=tile)


def motion_at(grid: GridMotion, y: int, x: int) -> tuple[float, float]:
    """Mouvement (vy, vx) de la tuile contenant le pixel (y, x) ; indices bornés à la grille."""
    ty = min(max(y // grid.tile, 0), grid.v.shape[0] - 1)
    tx = min(max(x // grid.tile, 0), grid.v.shape[1] - 1)
    return (float(grid.v[ty, tx, 0]), float(grid.v[ty, tx, 1]))


def smooth_velocity(v_prev: tuple[float, float] | None, v_obs: tuple[float, float],
                    alpha: float = 0.5, dt_steps: int = 1) -> tuple[float, float]:
    """Moyenne exponentielle de la vitesse par pas ; v_obs est un déplacement sur dt_steps pas."""
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha doit être dans ]0, 1] (reçu {alpha})")
    if dt_steps < 1:
        raise ValueError(f"dt_steps doit être >= 1 (reçu {dt_steps})")
    per_step = (v_obs[0] / dt_steps, v_obs[1] / dt_steps)
    if v_prev is None:
        return per_step
    return (alpha * per_step[0] + (1 - alpha) * v_prev[0],
            alpha * per_step[1] + (1 - alpha) * v_prev[1])


def predict_displacement(v: tuple[float, float], dt_steps: int) -> tuple[float, float]:
    """Déplacement attendu après dt_steps pas pour une vitesse v par pas."""
    if dt_steps < 1:
        raise ValueError(f"dt_steps doit être >= 1 (reçu {dt_steps})")
    return (v[0] * dt_steps, v[1] * dt_steps)
