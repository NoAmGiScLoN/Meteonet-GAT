"""Statistiques de residu par cellule d'une reconstruction ajustee (domaine racine).

Residu r = sqrt(max(rec, 0)) - sqrt(rate), comme la loss et la nrmse de l'ajusteur. Masque de la
cellule : rate > 0 (zero hors cellule). Les statistiques sont sauvegardees avec les sources de chaque
cellule (correction de biais, modele de bruit, features du GAT).
"""
import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

import noise_floor

log = logging.getLogger(__name__)

EDGE_BAND_PX = 2  # demi-largeur (px) de la bande a cheval sur le bord du masque
SPECTRAL_MIN_PX = 16  # cote minimal (px) de la boite englobante pour estimer la pente
SPECTRAL_MIN_BINS = 4  # nombre minimal de classes radiales pour ajuster la pente
STRUCT_8 = np.ones((3, 3), dtype=bool)  # voisinage 8-connexe (comme cell_delimitation.STRUCT_8)
NYQUIST = 0.5  # cycles/px
POWER_REL_FLOOR = 1e-12  # classes de puissance <= ce facteur x max exclues (ajustement quasi parfait)


@dataclass(frozen=True)
class ResidualStats:
    """Statistiques de residu d'une cellule.

    Convention de signe : residu = sqrt(rec) - sqrt(rate), donc positif = rec surestime.
    - peak_bias, mass_bias : biais relatifs (rec - rate) / rate du maximum et de la somme, sur tout
      le crop, spill compris. mass_bias recouvre donc spill_frac par construction.
    - spill_frac : masse de rec hors masque / masse de rate.
    - resid_std_in, resid_std_edge : ecart-type du residu (racine) dans le masque / dans la bande de
      bord. Ils retirent la moyenne : un biais uniforme se voit dans mass_bias, pas dans la std.
      La bande de bord inclut des pixels hors masque (a cheval sur le bord).
    - spectral_slope : pente log-log de la PSD radiale du residu (voir radial_psd_slope).
    - nrmse_mask : nrmse en racine dans le masque.
    NaN = non estimable (bande vide, boite trop petite, residu nul...).
    """
    peak_bias: float
    mass_bias: float
    resid_std_in: float
    resid_std_edge: float
    spill_frac: float
    spectral_slope: float
    nrmse_mask: float

    def as_dict(self) -> dict[str, float]:
        """Dictionnaire a ordre de cles stable (STAT_COLUMNS)."""
        return {name: float(value) for name, value in asdict(self).items()}


STAT_COLUMNS: tuple[str, ...] = (
    "peak_bias",
    "mass_bias",
    "resid_std_in",
    "resid_std_edge",
    "spill_frac",
    "spectral_slope",
    "nrmse_mask",
)


def cell_mask(rate: np.ndarray) -> np.ndarray:
    """Masque de la cellule : rate > 0."""
    return np.asarray(rate) > 0


def edge_band(mask: np.ndarray, width: int) -> np.ndarray:
    """Bande a cheval sur le bord : dilate(mask, width) & ~erode(mask, width), 8-connexe.

    Le bord du crop n'est pas un bord de masque (erosion avec border_value=1)."""
    if width < 1:
        raise ValueError("width must be >= 1")
    mask = np.asarray(mask, dtype=bool)
    dilated = ndi.binary_dilation(mask, structure=STRUCT_8, iterations=width)
    eroded = ndi.binary_erosion(mask, structure=STRUCT_8, iterations=width, border_value=1)
    return dilated & ~eroded


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice]:
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _radial_power(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequence moyenne et puissance moyenne par classe radiale (k=0 exclu, k <= Nyquist)."""
    height, width = crop.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fft2(crop * window)) ** 2
    k = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :])
    bin_idx = np.rint(k * max(height, width)).astype(int)
    keep = (bin_idx >= 1) & (k <= NYQUIST)
    counts = np.bincount(bin_idx[keep])
    sum_power = np.bincount(bin_idx[keep], weights=power[keep])
    sum_k = np.bincount(bin_idx[keep], weights=k[keep])
    mean_power = np.divide(sum_power, counts, out=np.zeros(counts.shape), where=counts > 0)
    valid = (counts > 0) & (mean_power > POWER_REL_FLOOR * mean_power.max())
    return sum_k[valid] / counts[valid], mean_power[valid]


def radial_psd_slope(field: np.ndarray, mask: np.ndarray) -> float:
    """Pente log-log du spectre de puissance radial de `field` dans la boite englobante du masque.

    field mis a 0 hors masque, fenetre de Hann 2-D, k=0 exclu. NaN si le plus petit cote de la
    boite est < SPECTRAL_MIN_PX ou s'il y a moins de SPECTRAL_MIN_BINS classes radiales dont la
    puissance depasse POWER_REL_FLOOR x le max (residu nul ou quasi nul -> NaN).

    Methode et biais connus (la pente est un descripteur de texture RELATIF, pas un spectre absolu) :
    - le residu est mis a 0 hors masque : la forme du masque ajoute sa propre signature spectrale ;
    - la fenetre de Hann sur la boite englobante attenue le bord, donc les pixels proches du bord
      du masque pesent moins ;
    - le spill (rec hors masque) est exclu de l'analyse ;
    - l'ajustement est non pondere sur des classes lineaires en k : les hautes frequences, plus
      nombreuses, dominent la pente ;
    - un sur-echantillonnage par interpolation vide le haut du spectre et raidit la pente."""
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if field.shape != mask.shape or field.ndim != 2:
        raise ValueError("field and mask must be 2-D with the same shape")
    if not mask.any():
        raise ValueError("mask is empty")
    box = _bounding_box(mask)
    crop = np.where(mask, field, 0.0)[box]
    if min(crop.shape) < SPECTRAL_MIN_PX:
        return float("nan")
    k_mean, power_mean = _radial_power(crop)
    if k_mean.size < SPECTRAL_MIN_BINS:
        return float("nan")
    return float(np.polyfit(np.log(k_mean), np.log(power_mean), 1)[0])


def _validate(rec: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copies float64 validees (2-D, meme forme, finies, rate >= 0, masque non vide, somme > 0)."""
    rate_arr = np.asarray(rate, dtype=np.float64)
    rec_arr = np.asarray(rec, dtype=np.float64)
    if rate_arr.ndim != 2 or rec_arr.ndim != 2:
        raise ValueError("rec and rate must be 2-D")
    if not np.all(np.isfinite(rate_arr)):
        raise ValueError("rate contains NaN or inf")
    if np.any(rate_arr < 0):
        raise ValueError("rate contains negative values")
    if rec_arr.shape != rate_arr.shape:
        raise ValueError(f"shape mismatch: {rec_arr.shape} vs {rate_arr.shape}")
    if not np.all(np.isfinite(rec_arr)):
        raise ValueError("rec contains NaN or inf")
    if not (rate_arr > 0).any():
        raise ValueError("cell mask is empty (rate has no positive pixel)")
    return np.clip(rec_arr, 0.0, None), rate_arr


def _std_or_nan(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def compute_residual_stats(rec: np.ndarray, rate: np.ndarray) -> ResidualStats:
    """Statistiques de residu de `rec` (mm/h, rendu de l'ajusteur) contre `rate` (crop observe)."""
    rec_pos, rate_arr = _validate(rec, rate)
    mask = cell_mask(rate_arr)
    resid = np.sqrt(rec_pos) - np.sqrt(rate_arr)
    total_rate = rate_arr.sum()
    stats = ResidualStats(
        peak_bias=float((rec_pos.max() - rate_arr.max()) / rate_arr.max()),
        mass_bias=float((rec_pos.sum() - total_rate) / total_rate),
        resid_std_in=_std_or_nan(resid[mask]),
        resid_std_edge=_std_or_nan(resid[edge_band(mask, EDGE_BAND_PX)]),
        spill_frac=float(rec_pos[~mask].sum() / total_rate),
        spectral_slope=radial_psd_slope(resid, mask),
        nrmse_mask=noise_floor.nrmse_sqrt(rate_arr, rec_pos, mask),
    )
    log.debug("residual stats: %s", stats)
    return stats
