"""Tracker v2 : un pas de suivi (segmentation, mouvement, association, lignée, événements).

Règles : naissance à >= min_area_start px, continuation jusqu'à min_area_keep px (hystérésis) ;
une piste manquée dérive (coast) max_coast carte(s) avec une prédiction multipliée par les pas
écoulés, puis se termine ; un écart > max_gap_min coupe toutes les pistes (gap_break) et la
carte suivante repart à neuf. Vitesse : EMA par piste ; pistes jeunes (age < young_frames) :
mouvement global fiable de la corrélation de phase (par pas), sinon nul.
"""
from collections.abc import Iterable
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd

from association import advect_coords, assign, cost_matrix, overlap_counts, prediction_for
from cell_delimitation import describe, segment
from frame_stream import Frame
from lineage import classify_groups, edge_event, is_truncated, overlap_links
from motion import Motion, phase_correlation, smooth_velocity
from tracker_state import Track, TrackerConfig, TrackerState

CELL_COLUMNS = ["timestamp", "local_id", "area_px", "cy", "cx", "cy_w", "cx_w", "intensity_sum",
                "intensity_mean", "intensity_max", "border_flag", "missing_neighbor_flag",
                "n_holes_bbox"]
DET_EXTRA = ["track_id", "truncated", "coasted_before", "v_y", "v_x", "v_src", "pred_src",
             "velocity_reset"]
LINEAGE_COLUMNS = ["t", "parent_track", "child_track", "link_type", "overlap_frac_parent",
                   "overlap_frac_child", "share_of_child", "share_of_parent", "is_main"]
EVENT_COLUMNS = ["t", "track_id", "event", "truncated", "area", "notes"]
NO_FIELD = Motion(0.0, 0.0, 0.0, False)


class Vel(NamedTuple):
    """Vitesse rapportée d'une cellule : valeur par pas, source APRÈS mise à jour, source de la
    prédiction, drapeau de réinitialisation (déplacement observé aberrant)."""
    v: tuple[float, float]
    src: str          # track | field | none
    pred_src: str     # track | field | none
    reset: bool


NO_VEL = Vel((0.0, 0.0), "none", "none", False)


@dataclass(frozen=True, eq=False)  # eq=False : champs DataFrame
class StepOutput:
    """Sorties d'un pas : détections (track_id 0 = sans piste), lignée, événements."""
    detections: pd.DataFrame
    lineage: pd.DataFrame
    events: pd.DataFrame


def _cells_table(frame: Frame) -> tuple[np.ndarray, pd.DataFrame]:
    labels, n, valid = segment(frame.R)
    table = describe(frame.R, labels, n, valid, frame.t)
    return labels, (pd.DataFrame(columns=CELL_COLUMNS) if n == 0 else table)


def _pixels_by_label(labels: np.ndarray, wanted: Iterable[int]) -> dict[int, tuple]:
    """{local_id: (ys, xs)} pour les étiquettes demandées, en un seul tri."""
    flat = labels.ravel()
    idx = np.flatnonzero(flat)
    order = np.argsort(flat[idx], kind="stable")
    idx, lab = idx[order], flat[idx][order]
    out = {}
    for lid in wanted:
        lo, hi = np.searchsorted(lab, [lid, lid + 1])
        ys, xs = np.divmod(idx[lo:hi], labels.shape[1])
        out[int(lid)] = (ys.astype(np.int32), xs.astype(np.int32))
    return out


def _field_motion(frame: Frame, prev_R: np.ndarray | None, cfg: TrackerConfig) -> Motion:
    """Mouvement global par pas entre la carte précédente et celle-ci (non fiable si absent).

    Limite : pour dt_steps > 1 la corrélation de phase porte sur un déplacement multiple ; la
    division par dt_steps suppose un mouvement uniforme, et un décalage proche de la limite de
    repliement est déjà marqué non fiable par motion.phase_correlation.
    """
    if prev_R is None or frame.dt_steps < 1:
        return NO_FIELD
    m = phase_correlation(prev_R, frame.R, cfg.coarse, cfg.min_peak_ratio)
    return Motion(m.vy / frame.dt_steps, m.vx / frame.dt_steps, m.peak_ratio, m.reliable)


def _source(track_age: int, field: Motion, cfg: TrackerConfig) -> str:
    if track_age >= cfg.young_frames:
        return "track"
    return "field" if field.reliable else "none"


def _event(t: datetime, track_id: int, event: str, truncated: bool, area: int,
           notes: str = "") -> dict:
    return {"t": t, "track_id": track_id, "event": event, "truncated": truncated,
            "area": area, "notes": notes}


def _end_event(track: Track, disp: tuple[float, float], shape: tuple[int, int]) -> str:
    _, _, frac_out = advect_coords(track.ys, track.xs, disp, shape)
    flags = {"border_flag": track.border_flag, "missing_neighbor_flag": track.missing_flag}
    return edge_event(flags, "end", frac_out)


def _new_track(track_id: int, row: pd.Series, pix: tuple, v: tuple[float, float]) -> Track:
    return Track(track_id=track_id, local_id=int(row.local_id), cy=float(row.cy),
                 cx=float(row.cx), v=v, area=int(row.area_px), mass=float(row.intensity_sum),
                 ys=pix[0], xs=pix[1], age=1, coasted=0, truncated=is_truncated(row),
                 border_flag=bool(row.border_flag),
                 missing_flag=bool(row.missing_neighbor_flag))


def _continue_track(track: Track, row: pd.Series, pix: tuple, cfg: TrackerConfig,
                    pred: tuple[float, float], field: Motion) -> tuple[Track, Vel]:
    """Continue une piste. EMA de la vitesse : jeune -> amorcée par le champ fiable ; si le
    déplacement observé s'écarte de la prédiction de plus de d_max (saut de centre après
    fusion / scission), la vitesse prédite est conservée (velocity_reset)."""
    steps = track.gap_steps
    obs = (float(row.cy) - track.cy, float(row.cx) - track.cx)
    young = track.age < cfg.young_frames
    pred_src = _source(track.age, field, cfg)
    field_v = (field.vy, field.vx) if field.reliable else None
    if math.hypot(obs[0] - pred[0], obs[1] - pred[1]) > cfg.d_max:
        v, src = ((field_v, "field") if young and field_v else (track.v, "track"))
        reset = True
    else:
        v_prev = field_v if young and field_v else (None if track.age <= 1 else track.v)
        v, src, reset = smooth_velocity(v_prev, obs, cfg.alpha, steps), "track", False
    new = replace(_new_track(track.track_id, row, pix, v), age=track.age + 1)
    return new, Vel(v, src, pred_src, reset)


def _validate_frame(state: TrackerState, frame: Frame) -> None:
    if state.tracks and not frame.gap_break and frame.dt_steps < 1:
        raise ValueError("dt_steps doit être >= 1 avec des pistes actives "
                         f"(reçu {frame.dt_steps})")


def _check_unique(detections: pd.DataFrame) -> None:
    tracked = detections[detections.track_id > 0]
    if tracked.duplicated(["timestamp", "track_id"]).any():
        raise ValueError("Invariant violé : (t, track_id) dupliqué dans les détections")


def step(state: TrackerState, frame: Frame, cfg: TrackerConfig,
         prev_R: np.ndarray | None) -> tuple[TrackerState, StepOutput]:
    """Traite une carte : retourne (nouvel état, sorties) ; `state` n'est jamais modifié."""
    _validate_frame(state, frame)
    t, events = frame.t, []
    if frame.gap_break:
        events += [_event(t, k.track_id, "gap_break", k.truncated, k.area) for k in state.tracks]
        state = state.with_tracks(())
        prev_R = None
    labels, table = _cells_table(frame)
    table = table.reset_index(drop=True)
    kept_tracks = list(state.tracks)
    eligible = table[table.area_px >= cfg.min_area_keep].reset_index(drop=True)
    pix = _pixels_by_label(labels, eligible.local_id) if len(eligible) else {}
    field = _field_motion(frame, prev_R, cfg)
    dt = max(frame.dt_steps, 1)
    disps = [prediction_for(k, field, k.gap_steps + dt, cfg) for k in kept_tracks]
    tracks = [replace(k, gap_steps=k.gap_steps + dt) for k in kept_tracks]
    result, groups_new = _associate(tracks, eligible, labels, disps, cfg)
    rows = {int(r.local_id): eligible.iloc[i] for i, r in enumerate(eligible.itertuples())}
    by_id = {k.track_id: k for k in tracks}
    disp_of = {k.track_id: d for k, d in zip(tracks, disps)}
    new_tracks, lid_track, coasted_before, vel = [], {}, {}, {}
    for tid, lid in result.continuations.items():
        k, vel[lid] = _continue_track(by_id[tid], rows[lid], pix[lid], cfg, disp_of[tid], field)
        new_tracks.append(k)
        lid_track[lid], coasted_before[lid] = tid, by_id[tid].coasted
    for tid, why in result.ended.items():
        events.append(_event(t, tid, why, by_id[tid].truncated, by_id[tid].area))
    for tid in result.unmatched_tracks:
        k = by_id[tid]
        if k.coasted < cfg.max_coast:
            new_tracks.append(replace(k, coasted=k.coasted + 1))
            events.append(_event(t, tid, "coast", k.truncated, k.area))
        else:
            events.append(_event(t, tid, _end_event(k, disp_of[tid], labels.shape), k.truncated,
                                 k.area))
    state, started = _start_tracks(state, result, groups_new, rows, pix, field, t, events, by_id)
    new_tracks += [k for k, _ in started]
    for k, v in started:
        lid_track[k.local_id], coasted_before[k.local_id], vel[k.local_id] = k.track_id, 0, v
    detections = _detections(table, lid_track, coasted_before, vel)
    _check_unique(detections)
    lineage = _lineage(result, lid_track, t)
    new_state = state.with_tracks(sorted(new_tracks, key=lambda k: k.track_id)).with_time(t)
    return new_state, StepOutput(detections, lineage,
                                 pd.DataFrame(events, columns=EVENT_COLUMNS))


def _associate(tracks: list[Track], cells: pd.DataFrame, labels: np.ndarray,
               disps: list[tuple[float, float]], cfg: TrackerConfig):
    """Association + classification ; cellules vides ou sans piste : tout non apparié."""
    eligible_new = set(cells.local_id[cells.area_px >= cfg.min_area_start])
    if not len(cells) or not tracks:
        return classify_groups([], tracks, cells, [], eligible_new), eligible_new
    counts = overlap_counts(tracks, disps, labels)
    centres = np.array([(k.cy + d[0], k.cx + d[1]) for k, d in zip(tracks, disps)])
    C, _, _ = cost_matrix(tracks, cells, counts, centres, cfg)
    pairs = assign(C, cfg.max_cost)
    cands = overlap_links(counts, tracks, cells, cfg)
    return classify_groups(cands, tracks, cells, pairs, eligible_new), eligible_new


def _start_tracks(state: TrackerState, result, eligible_new: set, rows: dict, pix: dict,
                  field: Motion, t: datetime, events: list,
                  by_id: dict[int, Track]) -> tuple[TrackerState, list]:
    """Crée les pistes : enfants de scission (vitesse du parent) puis cellules non appariées
    éligibles (vitesse du champ fiable, sinon nulle)."""
    started = []
    v_field = (field.vy, field.vx) if field.reliable else (0.0, 0.0)
    wanted = [(lid, parent, ev) for lid, parent, ev in result.new_children]
    wanted += [(lid, None, None) for lid in result.unmatched_cells if lid in eligible_new]
    for lid, parent, ev in wanted:
        tid, state = state.allocate_id()
        if parent is not None:
            vel = Vel(by_id[parent].v, "track", "none", False)
        else:
            vel = Vel(v_field, "field" if field.reliable else "none", "none", False)
        k = _new_track(tid, rows[lid], pix[lid], vel.v)
        if ev is None:
            ev = edge_event(rows[lid], "start")
        notes = f"parent={parent}" if parent is not None else ""
        events.append(_event(t, tid, ev, k.truncated, k.area, notes))
        started.append((k, vel))
    return state, started


def _detections(table: pd.DataFrame, lid_track: dict, coasted_before: dict,
                vel: dict) -> pd.DataFrame:
    out = table.copy()
    lids = out.local_id.astype(int)
    vels = [vel.get(int(l), NO_VEL) for l in lids]
    out["track_id"] = lids.map(lambda l: lid_track.get(l, 0)).astype(int)
    out["truncated"] = out.border_flag.astype(bool) | out.missing_neighbor_flag.astype(bool)
    out["coasted_before"] = lids.map(lambda l: coasted_before.get(l, 0)).astype(int)
    out["v_y"] = [v.v[0] for v in vels]
    out["v_x"] = [v.v[1] for v in vels]
    out["v_src"] = [v.src for v in vels]
    out["pred_src"] = [v.pred_src for v in vels]
    out["velocity_reset"] = [v.reset for v in vels]
    return out.reset_index(drop=True)[CELL_COLUMNS + DET_EXTRA]


def _lineage(result, lid_track: dict, t: datetime) -> pd.DataFrame:
    rows = [{"t": t, "parent_track": lk.parent_track,
             "child_track": lid_track.get(lk.child_local_id, 0), "link_type": lk.link_type,
             "overlap_frac_parent": lk.overlap_frac_parent,
             "overlap_frac_child": lk.overlap_frac_child, "share_of_child": lk.share_of_child,
             "share_of_parent": lk.share_of_parent, "is_main": lk.is_main}
            for lk in result.links]
    return pd.DataFrame(rows, columns=LINEAGE_COLUMNS)


def track_sequence(frames: Iterable[Frame], cfg: TrackerConfig,
                   state: TrackerState | None = None) -> tuple[TrackerState, list[StepOutput]]:
    """Boucle de commodité : enchaîne `step` en conservant la carte précédente."""
    state = TrackerState() if state is None else state
    outputs, prev_R = [], None
    for frame in frames:
        state, out = step(state, frame, cfg, prev_R)
        outputs.append(out)
        prev_R = frame.R
    return state, outputs
