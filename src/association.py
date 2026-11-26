"""Association cellules <-> pistes (tracker v2) : advection, recouvrement, coût, Hungarian.

Même coût que le v1 (track_cells.match) : w_pos * dist / d_max + (1 - w_pos) * (1 - IoU du masque
translaté), porte « dist > d_max et IoU < 0,1 ».
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from motion import Motion, predict_displacement
from tracker_state import Track, TrackerConfig

BIG = 1e6  # coût des paires écartées par la porte
GATE_IOU = 0.1


def advect_coords(ys: np.ndarray, xs: np.ndarray, disp: tuple[float, float],
                  shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Translate les pixels de disp arrondi, garde ceux dans la grille : (ys2, xs2, frac_out)."""
    yy, xx = ys + int(round(disp[0])), xs + int(round(disp[1]))
    inside = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    frac_out = 0.0 if ys.size == 0 else float(1.0 - inside.sum() / ys.size)
    return yy[inside], xx[inside], frac_out


def overlap_counts(tracks: Sequence[Track], displacements: Sequence[tuple[float, float]],
                   labels: np.ndarray) -> np.ndarray:
    """Pixels communs piste advectée / cellule : matrice (n_pistes, labels.max() + 1), col. 0 = fond.

    La largeur vient de labels.max() : des identifiants non contigus (trous) restent indexables.
    """
    counts = np.zeros((len(tracks), int(labels.max()) + 1), dtype=np.int64)
    for r, (track, disp) in enumerate(zip(tracks, displacements)):
        yy, xx, _ = advect_coords(track.ys, track.xs, disp, labels.shape)
        counts[r] = np.bincount(labels[yy, xx], minlength=counts.shape[1])
    return counts


def cost_matrix(tracks: Sequence[Track], cells: pd.DataFrame, overlaps: np.ndarray,
                predicted_centres: np.ndarray,
                cfg: TrackerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (C, iou, dist), formes (n_pistes, n_cellules) ; colonnes dans l'ordre de `cells`.

    L'IoU utilise l'aire pleine de la piste (sémantique v1), même si des pixels sortent.
    """
    shape = (len(tracks), len(cells))
    C, iou, dist = np.full(shape, BIG), np.zeros(shape), np.zeros(shape)
    if 0 in shape:
        return C, iou, dist
    lids = cells.local_id.to_numpy(dtype=int)
    if lids.min() < 0 or lids.max() >= overlaps.shape[1]:
        raise ValueError(f"local_id hors de la matrice de recouvrement "
                         f"({overlaps.shape[1]} colonnes) : {lids.min()}..{lids.max()}")
    area_c = cells.area_px.to_numpy(dtype=float)
    area_t = np.array([t.area for t in tracks], dtype=float)[:, None]
    inter = overlaps[:, lids].astype(float)
    iou = inter / (area_t + area_c[None, :] - inter)
    cy, cx = cells.cy.to_numpy(dtype=float), cells.cx.to_numpy(dtype=float)
    dy, dx = cy[None, :] - predicted_centres[:, 0:1], cx[None, :] - predicted_centres[:, 1:2]
    dist = np.hypot(dy, dx)
    cost = cfg.w_pos * dist / cfg.d_max + (1 - cfg.w_pos) * (1 - iou)
    return np.where((dist > cfg.d_max) & (iou < GATE_IOU), BIG, cost), iou, dist


def assign(C: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Affectation hongroise ; les paires de coût > max_cost (dont celles écartées) sont abandonnées."""
    if C.size == 0:
        return []
    rows, cols = linear_sum_assignment(C)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if C[r, c] <= max_cost]


def prediction_for(track: Track, field_motion: Motion | None, dt_steps: int,
                   cfg: TrackerConfig) -> tuple[float, float]:
    """Déplacement sur dt_steps pas : vitesse de la piste (mûre), sinon champ fiable, sinon nul."""
    if track.age >= cfg.young_frames:
        return predict_displacement(track.v, dt_steps)
    if field_motion is not None and field_motion.reliable:
        return predict_displacement((field_motion.vy, field_motion.vx), dt_steps)
    return predict_displacement((0.0, 0.0), dt_steps)
