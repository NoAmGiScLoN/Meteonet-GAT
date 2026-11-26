"""Chargement des sorties de track_run pour l'analyse : tables concaténées, colonnes utiles
seulement (mémoire), classes de taille des pistes."""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from track_run import DONE_FILE, DTYPES

logger = logging.getLogger(__name__)

STEP = timedelta(minutes=5)
SIZE_BINS = (-np.inf, 200, 1000, 5000, np.inf)
SIZE_LABELS = ("50-200", "200-1000", "1000-5000", ">=5000")
NEEDED = {
    "detections": ["timestamp", "local_id", "area_px", "cy", "cx", "intensity_sum", "track_id",
                   "v_y", "v_x", "velocity_reset"],
    "events": ["t", "track_id", "event", "truncated", "area"],
    "lineage": list(DTYPES["lineage"]),
}


@dataclass(frozen=True, eq=False)
class RunTables:
    """Tables concaténées d'une exécution ; lineage est vide sauf demande explicite."""
    detections: pd.DataFrame
    lineage: pd.DataFrame
    events: pd.DataFrame


def _read(dirs: Sequence[Path], kind: str) -> pd.DataFrame:
    columns = NEEDED[kind]
    parts = [pd.read_parquet(d / f"{kind}.parquet", columns=columns) for d in dirs]
    parts = [p for p in parts if len(p)]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return table.astype({c: DTYPES[kind][c] for c in columns})


def load_run(out_dir: Path, months: Sequence[str] | None = None,
             with_lineage: bool = False) -> RunTables:
    """Concatène les mois terminés (DONE) de out_dir (ou ceux de `months`, « aaaa-mm »).

    Seules les colonnes utiles aux métriques sont lues ; la lignée seulement sur demande."""
    out_dir = Path(out_dir)
    names = list(months) if months else sorted(p.parent.name for p in out_dir.glob(
        f"*/{DONE_FILE}"))
    dirs = []
    for name in names:
        if (out_dir / name / DONE_FILE).exists():
            dirs.append(out_dir / name)
        else:
            logger.warning("%s : mois non terminé, ignoré", name)
    if not dirs:
        raise FileNotFoundError(f"Aucun mois terminé (DONE) dans {out_dir}")
    lineage = _read(dirs, "lineage") if with_lineage else pd.DataFrame(columns=NEEDED["lineage"])
    return RunTables(_read(dirs, "detections"), lineage, _read(dirs, "events"))


def size_class(max_area: pd.Series) -> pd.Series:
    """Classe de taille d'après l'aire maximale (px)."""
    return pd.cut(max_area, bins=SIZE_BINS, labels=SIZE_LABELS, right=False)
