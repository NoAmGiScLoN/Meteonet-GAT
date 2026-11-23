"""Flux de cartes MeteoNet pour le tracker v2 : itération chronologique à cheval sur les fichiers.

Un seul fichier npz (~2,6 Go) est chargé à la fois, libéré avant le suivant.
"""
import logging
import re
from dataclasses import dataclass
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np

from cell_delimitation import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_MAX_GAP_MIN = 15
DEFAULT_STEP_MIN = 5
_FILE_RE = re.compile(r"rainfall_NW_(\d{4})_(\d{2})\.(\d+)\.npz$")


@dataclass(frozen=True, eq=False)  # eq=False : champ ndarray, pas de comparaison ambiguë
class Frame:
    """Une carte de pluie : instant, pixels bruts (lecture seule), pas écoulés depuis la précédente."""
    t: datetime
    R: np.ndarray  # int16, -1 = manquant ; copie en lecture seule
    dt_steps: int  # 1 = un pas (5 min) depuis la carte précédente ; 0 pour la première
    gap_break: bool  # True si l'écart dépasse max_gap_min (trajectoires à couper)


class NoFramesError(ValueError):
    """Aucune carte dans l'intervalle demandé (données absentes ou lacunaires)."""


def _months_between(start: datetime, end: datetime) -> Iterator[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def list_files(data_dir: Path, start: datetime, end: datetime) -> list[Path]:
    """Fichiers npz des mois recouvrant [start, end], triés (année, mois, k)."""
    found: list[tuple[tuple[int, int, int], Path]] = []
    for year, month in _months_between(start, end):
        month_dir = Path(data_dir) / f"NW_rainfall_{year}" / f"rainfall-NW-{year}-{month:02d}"
        for path in month_dir.glob("rainfall_NW_*.npz"):
            match = _FILE_RE.match(path.name)
            if match:
                found.append(((year, month, int(match.group(3))), path))
    return [path for _, path in sorted(found)]


def check_step(previous: datetime, current: datetime, step_min: int) -> int:
    """Nombre de pas entre deux instants ; ValueError si non croissant ou non multiple de step_min."""
    seconds = (current - previous).total_seconds()
    if seconds <= 0:
        raise ValueError(f"Horodatages non strictement croissants (doublon ou retour arrière) : {previous} -> {current}")
    if seconds % (step_min * 60) != 0:
        raise ValueError(f"Écart {previous} -> {current} non multiple de {step_min} min")
    return int(seconds // (step_min * 60))


def iter_frames(start: datetime, end: datetime, data_dir: Path = DATA_DIR,
                max_gap_min: int = DEFAULT_MAX_GAP_MIN,
                step_min: int = DEFAULT_STEP_MIN) -> Iterator[Frame]:
    """Cartes avec start <= t <= end, dans l'ordre chronologique, d'un fichier à l'autre.

    Les arguments sont validés à l'appel (pas à la première itération).
    """
    if start > end:
        raise ValueError(f"Le début {start} est postérieur à la fin {end}")
    paths = list_files(data_dir, start, end)
    if not paths:
        raise FileNotFoundError(f"Aucun fichier MeteoNet dans {data_dir} pour la période {start} -> {end}")
    return _generate(paths, start, end, max_gap_min, step_min)


def _generate(paths: list[Path], start: datetime, end: datetime,
              max_gap_min: int, step_min: int) -> Iterator[Frame]:
    """Générateur interne : un fichier chargé à la fois, libéré avant le suivant."""
    previous: datetime | None = None
    for path in paths:
        # allow_pickle=True : `dates` est un tableau d'objets datetime ; données MeteoNet locales de confiance
        with np.load(path, allow_pickle=True) as npz:
            dates = list(npz["dates"])  # lecture paresseuse : `data` n'est pas encore chargé
            selected = [i for i, t in enumerate(dates) if start <= t <= end]
            if selected:
                logger.info("Chargement de %s (%d cartes retenues)", path.name, len(selected))
                data = npz["data"]
                if len(data) != len(dates):
                    raise ValueError(f"{path.name} : longueur de data ({len(data)}) "
                                     f"différente de celle de dates ({len(dates)})")
                for i in selected:
                    t = dates[i]
                    dt_steps = 0 if previous is None else check_step(previous, t, step_min)
                    gap_break = previous is not None and (t - previous).total_seconds() / 60 > max_gap_min
                    pixels = np.array(data[i], dtype=np.int16)  # copie : n'expose pas le gros tableau
                    pixels.setflags(write=False)
                    previous = t
                    yield Frame(t=t, R=pixels, dt_steps=dt_steps, gap_break=gap_break)
                del data
        if any(t > end for t in dates):
            break  # fichiers suivants entièrement postérieurs à `end`
    if previous is None:
        raise NoFramesError(f"Aucune carte entre {start} et {end} dans les fichiers trouvés")
