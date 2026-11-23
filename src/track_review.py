"""Planches de revue visuelle des événements du tracker v2 : échantillon stratifié par type,
chaque événement en 3 cartes (t-1, t, t+1) de taux de pluie avec contours colorés par piste
(piste de l'événement en rouge). Seules les cartes utiles sont lues (fenêtres de 3 pas).
"""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure  # API objet : aucun backend global imposé
from scipy import ndimage as ndi

from cell_delimitation import MISSING, segment
from frame_stream import Frame, NoFramesError, iter_frames
from lineage import EVENT_TYPES

logger = logging.getLogger(__name__)

EVENTS_PER_PAGE = 8
PER_ROW = 2                 # événements par ligne de la planche
CROP_PX = 64                # côté de la vignette centrée sur la piste
STEP = timedelta(minutes=5)
RATE_FACTOR = 12 / 100      # brut MeteoNet -> mm/h
RATE_MAX = 20.0             # saturation de l'échelle de couleur (mm/h)
HIGHLIGHT = (1.0, 0.0, 0.0, 1.0)
OTHERS_UNTRACKED = (0.5, 0.5, 0.5, 1.0)
CSV_COLUMNS = ["event", "t", "track_id", "area", "page", "ok"]
KEY = ["event", "t", "track_id"]
LOOKBACK = 3                # cartes avant t lues dans les détections (centre d'une piste morte)


@dataclass(frozen=True)
class ReviewEvent:
    """Événement échantillonné pour la revue, avec la page de planche où il figure."""
    event: str
    t: datetime
    track_id: int
    area: int
    page: int


def sample_events(events: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n événements au plus, répartis à tour de rôle entre les types présents (stratifié)."""
    rng = np.random.default_rng(seed)
    pools = {}
    for kind in EVENT_TYPES:
        idx = np.flatnonzero((events.event == kind).to_numpy())
        if len(idx):
            pools[kind] = list(rng.permutation(idx))
    chosen: list[int] = []
    while len(chosen) < n and any(pools.values()):
        for pool in pools.values():
            if pool and len(chosen) < n:
                chosen.append(pool.pop())
    picked = events.iloc[chosen].copy()
    picked["_order"] = picked.event.map({k: i for i, k in enumerate(EVENT_TYPES)})
    picked = picked.sort_values(["_order", "t"], kind="stable").drop(columns="_order")
    return picked.reset_index(drop=True)


def _boundaries(labels: np.ndarray) -> np.ndarray:
    """Pixels de contour : étiquette non nulle ayant un voisin (4-connexité) différent."""
    pad = np.pad(labels, 1)
    centre = pad[1:-1, 1:-1]
    differs = np.zeros(labels.shape, dtype=bool)
    for sl in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        differs |= sl != centre
    return (centre > 0) & differs


def overlay(labels: np.ndarray, dets: pd.DataFrame, track_id: int) -> np.ndarray:
    """Image RGBA des contours : couleur par piste, piste de l'événement en rouge épaissie.

    Les local_id absents de la carte (détections d'une autre carte) sont ignorés, avec alerte."""
    top = int(labels.max())
    ids = dets.local_id.to_numpy(int)
    ok = (ids > 0) & (ids <= top)
    if not ok.all():
        logger.warning("%d local_id hors de la carte (max %d) : contours ignorés",
                       int((~ok).sum()), top)
    lut = np.zeros(top + 1, dtype=np.int64)
    lut[ids[ok]] = dets.track_id.to_numpy(int)[ok]
    tracks = lut[labels]
    edge = _boundaries(labels)
    out = np.zeros(labels.shape + (4,))
    cmap = matplotlib.colormaps["tab20"]
    for tid in np.unique(tracks[edge]):
        out[edge & (tracks == tid)] = OTHERS_UNTRACKED if tid == 0 else cmap(int(tid) % 20)
    if track_id > 0:
        out[ndi.binary_dilation(edge & (tracks == track_id))] = HIGHLIGHT
    return out


def _crop(centre: tuple[float, float] | None, shape: tuple[int, int]) -> tuple[slice, slice]:
    if centre is None:
        return slice(0, shape[0]), slice(0, shape[1])
    spans = []
    for c, size in zip(centre, shape):
        lo = int(min(max(round(c) - CROP_PX // 2, 0), max(size - CROP_PX, 0)))
        spans.append(slice(lo, min(lo + CROP_PX, size)))
    return spans[0], spans[1]


def _read_window(t: datetime, data_dir: Path) -> dict[datetime, Frame]:
    try:
        return {f.t: f for f in iter_frames(t - STEP, t + STEP, data_dir)}
    except (FileNotFoundError, NoFramesError, ValueError) as exc:
        logger.warning("%s : cartes illisibles (%s)", t, exc)
        return {}


def _centre(dets: pd.DataFrame, track_id: int, t: datetime) -> tuple[float, float] | None:
    mine = dets[dets.track_id == track_id]
    if not len(mine):
        return None
    row = mine.iloc[(mine.timestamp - t).abs().argmin()]
    return float(row.cy), float(row.cx)


def _draw_panel(ax: Axes, frame: Frame | None, dets: pd.DataFrame, ev: pd.Series,
                centre: tuple[float, float] | None, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    if frame is None:
        ax.text(0.5, 0.5, "pas de carte", ha="center", va="center", fontsize=7)
        return
    rate = np.where(frame.R == MISSING, np.nan, frame.R * RATE_FACTOR)
    ys, xs = _crop(centre, rate.shape)
    ax.imshow(rate[ys, xs], cmap="Blues", vmin=0, vmax=RATE_MAX)
    labels, _, _ = segment(frame.R)
    here = dets[dets.timestamp == frame.t]
    if labels.max() > 0 and len(here):
        ax.imshow(overlay(labels, here, int(ev.track_id))[ys, xs])


def _draw_page(rows: pd.DataFrame, offset: int, dets: pd.DataFrame, data_dir: Path,
               path: Path) -> None:
    n_rows = -(-len(rows) // PER_ROW)
    fig = Figure(figsize=(3 * 3 * PER_ROW * 0.8, 2.6 * n_rows))
    axes = fig.subplots(n_rows, 3 * PER_ROW, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, ev in enumerate(rows.itertuples(index=False)):
        ev = pd.Series(ev._asdict())
        frames = _read_window(ev.t, data_dir)
        centre = _centre(dets, int(ev.track_id), ev.t)
        r, c = divmod(k, PER_ROW)
        for j, shift in enumerate((-1, 0, 1)):
            ax = axes[r, 3 * c + j]
            ax.axis("on")
            name = f"t{shift:+d}" if shift else (f"#{offset + k + 1} {ev.event} tr={ev.track_id}"
                                                 f" A={ev.area}\n{ev.t:%m-%d %H:%M}")
            _draw_panel(ax, frames.get(ev.t + shift * STEP), dets, ev, centre, name)
    fig.tight_layout()
    fig.savefig(path, dpi=70)


def _merge_ok(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Reprend les « ok » déjà remplis d'un review_events.csv existant (clé event, t,
    track_id). Si des lignes remplies ne figurent plus dans le nouvel échantillon, l'ancien
    fichier est conservé sous review_events.prevN.csv : rien de saisi n'est jamais perdu."""
    if not path.exists():
        return table
    try:
        old = pd.read_csv(path, dtype={"ok": str}, keep_default_na=False)
        old["t"] = pd.to_datetime(old.t).astype(str)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        logger.warning("%s illisible : remplacé", path)
        return table
    filled = old[old.ok.str.strip() != ""]
    key = lambda df: list(zip(df.event, df.t.astype(str), df.track_id.astype(int)))  # noqa: E731
    if set(key(filled)) - set(key(table)):
        n = len(list(path.parent.glob("review_events.prev*.csv"))) + 1
        path.replace(path.with_name(f"review_events.prev{n}.csv"))
    ok = dict(zip(key(filled), filled.ok))
    return table.assign(ok=[ok.get(k, "") for k in key(table)])


def write_review(events: pd.DataFrame, detections: pd.DataFrame, data_dir: Path, n: int,
                 seed: int, png_path: Path) -> list[ReviewEvent]:
    """Planches PNG (<stem>_01.png...) et review_events.csv à côté ; retourne l'échantillon.

    Un review_events.csv existant n'est jamais écrasé : ses « ok » remplis sont conservés."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    picked = sample_events(events, n, seed)
    needed = {t + s * STEP for t in picked.t for s in range(-LOOKBACK, 2)}
    dets = detections[detections.timestamp.isin(needed)]
    sample = []
    for p, start in enumerate(range(0, len(picked), EVENTS_PER_PAGE), start=1):
        rows = picked.iloc[start:start + EVENTS_PER_PAGE]
        _draw_page(rows, start, dets, Path(data_dir),
                   png_path.with_name(f"{png_path.stem}_{p:02d}.png"))
        sample += [ReviewEvent(r.event, r.t.to_pydatetime(), int(r.track_id), int(r.area), p)
                   for r in rows.itertuples()]
    table = pd.DataFrame([{**e.__dict__, "ok": ""} for e in sample], columns=CSV_COLUMNS)
    csv_path = png_path.parent / "review_events.csv"
    _merge_ok(table.astype({"t": "datetime64[ns]"}), csv_path).to_csv(csv_path, index=False)
    return sample


__all__ = ("EVENTS_PER_PAGE", "ReviewEvent", "overlay", "sample_events", "write_review")
