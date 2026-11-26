"""Helpers synthétiques réutilisables pour les tests du tracker v2 (fonctions simples, sans fixtures).

Les cartes sont en unités brutes MeteoNet (int16) : taux (mm/h) = brut * 12 / 100, donc le
seuil 1 mm/h correspond à brut > 8.33. Les pixels hors couverture valent -1.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_SHAPE = (60, 80)
T0 = datetime(2016, 1, 1, 12, 0)
STEP_MIN = 5
MISSING = -1
MIN_RAW = 9  # plus petite valeur brute > 8.33 : le bord des blobs reste toujours pluvieux
DEFAULT_PEAK = 60

# (cy, cx, ry, rx, pic) ; rectangle manquant : (y0, y1, x0, x1)
Blob = tuple[float, float, float, float, int]
Rect = tuple[int, int, int, int]


def rain_frame(shape: tuple[int, int], blobs: Iterable[Blob],
               missing: Rect | None = None) -> np.ndarray:
    """Carte int16 : blobs elliptiques (pic au centre, décroissance jusqu'au bord), zone -1 optionnelle."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    R = np.zeros(shape, dtype=np.int16)
    for cy, cx, ry, rx, peak in blobs:
        d2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        value = np.maximum(MIN_RAW, np.round(peak * (1 - 0.5 * d2))).astype(np.int16)
        R = np.where(d2 <= 1.0, np.maximum(R, value), R).astype(np.int16)
    if missing is not None:
        y0, y1, x0, x1 = missing
        R[y0:y1, x0:x1] = MISSING
    return R


def make_times(n: int, t0: datetime = T0, step_min: int = STEP_MIN,
               gap_after: int | None = None, gap_min: int | None = None) -> list[datetime]:
    """n instants espacés de step_min ; l'écart entre les cartes gap_after et gap_after+1 vaut gap_min."""
    times = [t0 + timedelta(minutes=step_min * i) for i in range(n)]
    if gap_after is None or gap_min is None:
        return times
    extra = timedelta(minutes=gap_min - step_min)
    return [t + extra if i > gap_after else t for i, t in enumerate(times)]


Sequence_ = tuple[list[np.ndarray], list[datetime]]


def disc_frame(shape: tuple[int, int], center: tuple[float, float], area: int,
               peak: int = DEFAULT_PEAK) -> np.ndarray:
    """Carte avec un disque pluvieux d'aire EXACTE `area` px : les `area` pixels les plus proches du centre."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dist = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2).ravel()
    keep = np.argsort(dist, kind="stable")[:area]
    R = np.zeros(shape, dtype=np.int16).ravel()
    R[keep] = peak
    return R.reshape(shape)


def _build(blobs_per_frame: Sequence[Sequence[Blob]], shape: tuple[int, int] = DEFAULT_SHAPE,
           missing: Rect | None = None, gap_after: int | None = None,
           gap_min: int | None = None) -> Sequence_:
    """Construit (cartes, instants) à partir des blobs de chaque carte."""
    frames = [rain_frame(shape, blobs, missing) for blobs in blobs_per_frame]
    return frames, make_times(len(frames), T0, STEP_MIN, gap_after, gap_min)


def translating_blob(n: int = 6, shape: tuple[int, int] = DEFAULT_SHAPE, start: tuple[float, float] = (30, 15),
                     velocity: tuple[float, float] = (0, 3), radii: tuple[float, float] = (5, 5),
                     gap_after: int | None = None, gap_min: int | None = None) -> Sequence_:
    """Un blob qui se translate de `velocity` (dy, dx) px par carte."""
    blobs = [[(start[0] + velocity[0] * i, start[1] + velocity[1] * i, *radii, DEFAULT_PEAK)] for i in range(n)]
    return _build(blobs, shape, None, gap_after, gap_min)


def merging_blobs(n: int = 6, shape: tuple[int, int] = DEFAULT_SHAPE,
                  gap_after: int | None = None, gap_min: int | None = None) -> Sequence_:
    """Deux blobs qui se rapprochent : 2 composantes au début, 1 (contact) à la fin."""
    blobs = [[(30, 14 + 5 * i, 6, 6, DEFAULT_PEAK), (30, 66 - 5 * i, 6, 6, DEFAULT_PEAK)] for i in range(n)]
    return _build(blobs, shape, None, gap_after, gap_min)


def splitting_blob(n: int = 6, shape: tuple[int, int] = DEFAULT_SHAPE,
                   gap_after: int | None = None, gap_min: int | None = None) -> Sequence_:
    """Un blob qui se scinde en deux : séquence de la fusion parcourue à l'envers."""
    frames, _ = merging_blobs(n, shape)
    return frames[::-1], make_times(n, T0, STEP_MIN, gap_after, gap_min)


def blob_exits_edge(n: int = 5, shape: tuple[int, int] = DEFAULT_SHAPE) -> Sequence_:
    """Un blob qui sort par le bord droit : carte finale vide."""
    return _build([[(30, shape[1] - 10 + 5 * i, 5, 5, DEFAULT_PEAK)] for i in range(n)], shape)


def blob_enters_missing(n: int = 5, shape: tuple[int, int] = DEFAULT_SHAPE) -> Sequence_:
    """Un blob qui avance vers une zone -1 (rectangle à droite) et y est englouti."""
    missing = (0, shape[0], shape[1] - 20, shape[1])
    blobs = [[(30, shape[1] - 40 + 7 * i, 5, 5, DEFAULT_PEAK)] for i in range(n)]
    return _build(blobs, shape, missing)


def blob_vanishes_one_frame(n: int = 5, vanish_at: int = 2, shape: tuple[int, int] = DEFAULT_SHAPE) -> Sequence_:
    """Un blob qui se translate mais est absent à la carte `vanish_at` seulement."""
    return _build([[] if i == vanish_at else [(30, 20 + 3 * i, 5, 5, DEFAULT_PEAK)] for i in range(n)], shape)


def area_sequence(areas: Sequence[int] = (52, 38, 45), shape: tuple[int, int] = DEFAULT_SHAPE) -> Sequence_:
    """Un disque fixe dont l'aire pluvieuse vaut exactement `areas` (52 -> 38 -> 45 px)."""
    frames = [disc_frame(shape, (30, 40), a) for a in areas]
    return frames, make_times(len(frames))


def write_npz_files(root: Path, files_spec: Iterable[tuple[int, int, int, Sequence[datetime], np.ndarray]]) -> list[Path]:
    """Écrit de faux npz MeteoNet (clés data, dates, miss_dates) dans l'arborescence réelle.

    files_spec : (année, mois, k, dates, data) ; chemin
    root/NW_rainfall_<y>/rainfall-NW-<y>-<mm>/rainfall_NW_<y>_<mm>.<k>.npz
    """
    paths = []
    for year, month, k, dates, data in files_spec:
        month_dir = Path(root) / f"NW_rainfall_{year}" / f"rainfall-NW-{year}-{month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        path = month_dir / f"rainfall_NW_{year}_{month:02d}.{k}.npz"
        date_array = np.empty(len(dates), dtype=object)
        date_array[:] = list(dates)
        np.savez(path, data=np.asarray(data, dtype=np.int16), dates=date_array,
                 miss_dates=np.empty(0, dtype=object))
        paths.append(path)
    return paths
