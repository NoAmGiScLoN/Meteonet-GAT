"""Tests de l'association (v2) : advection, recouvrements, coûts, affectation hongroise, prédiction."""
import numpy as np
import pandas as pd
import pytest

from association import BIG, advect_coords, assign, cost_matrix, overlap_counts, prediction_for
from cell_delimitation import describe, segment
from motion import Motion
from tracker_state import Track, TrackerConfig
from tracking_synth import disc_frame

SHAPE = (60, 80)
CFG = TrackerConfig()


def _cells(R: np.ndarray):
    labels, n, valid = segment(R)
    return labels, n, describe(R, labels, n, valid, None)


def _track_from(R: np.ndarray, track_id: int = 1, v=(0.0, 0.0), age: int = 3) -> Track:
    labels, n, df = _cells(R)
    row = df.iloc[0]
    ys, xs = np.nonzero(labels == int(row.local_id))
    return Track(track_id=track_id, local_id=int(row.local_id), cy=row.cy, cx=row.cx, v=v,
                 area=int(row.area_px), mass=float(row.intensity_sum), ys=ys, xs=xs,
                 age=age, coasted=0, truncated=False)


def _predicted(tracks, disps):
    return np.array([(t.cy + d[0], t.cx + d[1]) for t, d in zip(tracks, disps)])


def _run(tracks, disps, R1, cfg=CFG):
    labels, n, df = _cells(R1)
    ov = overlap_counts(tracks, disps, labels)
    C, iou, dist = cost_matrix(tracks, df, ov, _predicted(tracks, disps), cfg)
    return C, iou, dist


# ------------------------------------------------------------ advect_coords
def test_advect_coords_shifts_rounded_and_reports_no_loss():
    ys, xs = np.array([5, 6]), np.array([5, 5])
    y2, x2, frac = advect_coords(ys, xs, (1.4, -2.6), (20, 20))
    assert y2.tolist() == [6, 7] and x2.tolist() == [2, 2] and frac == 0.0


def test_advect_coords_clips_at_edges_and_reports_frac_out():
    ys, xs = np.array([0, 5, 9, 9]), np.array([5, 5, 5, 9])
    y2, x2, frac = advect_coords(ys, xs, (-1, 0), (10, 10))
    assert y2.tolist() == [4, 8, 8] and x2.tolist() == [5, 5, 9]
    assert frac == pytest.approx(0.25)


def test_advect_coords_everything_outside():
    y2, x2, frac = advect_coords(np.array([1, 2]), np.array([1, 2]), (0, 100), (10, 10))
    assert y2.size == 0 and x2.size == 0 and frac == 1.0


def test_advect_coords_empty_input():
    y2, x2, frac = advect_coords(np.array([], dtype=int), np.array([], dtype=int), (1, 1), (10, 10))
    assert y2.size == 0 and frac == 0.0


# ------------------------------------------------------------ overlap_counts
def test_overlap_counts_matches_brute_force():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(30, 30))
    n_cells = 3
    tracks = []
    for tid in range(2):
        mask = rng.random((30, 30)) < 0.2
        ys, xs = np.nonzero(mask)
        tracks.append(Track(tid + 1, 1, 0.0, 0.0, (0.0, 0.0), ys.size, 1.0, ys, xs, 3, 0, False))
    disps = [(2.2, -3.0), (0.0, 1.0)]
    counts = overlap_counts(tracks, disps, labels)
    assert counts.shape == (2, n_cells + 1)
    for r, (t, d) in enumerate(zip(tracks, disps)):
        expected = np.zeros(n_cells + 1, dtype=int)
        for y, x in zip(t.ys, t.xs):
            yy, xx = y + int(round(d[0])), x + int(round(d[1]))
            if 0 <= yy < 30 and 0 <= xx < 30:
                expected[labels[yy, xx]] += 1
        assert counts[r].tolist() == expected.tolist()


def test_overlap_counts_no_tracks_gives_empty_matrix():
    counts = overlap_counts([], [], np.array([[0, 2], [1, 0]]))
    assert counts.shape == (0, 3)


# --------------------------------------------------------------- cost_matrix
def test_translating_blob_gives_one_low_cost_assignment():
    R0, R1 = disc_frame(SHAPE, (30, 15), 80), disc_frame(SHAPE, (30, 18), 80)
    track = _track_from(R0)
    C, iou, dist = _run([track], [(0.0, 3.0)], R1)
    assert C.shape == (1, 1) and dist[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert iou[0, 0] == pytest.approx(1.0)
    assert C[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert assign(C, CFG.max_cost) == [(0, 0)]


def test_cost_uses_v1_formula_for_partial_overlap():
    R0, R1 = disc_frame(SHAPE, (30, 15), 80), disc_frame(SHAPE, (30, 18), 80)
    track = _track_from(R0)
    C, iou, dist = _run([track], [(0.0, 0.0)], R1)  # prédiction nulle : recouvrement partiel
    expected = CFG.w_pos * dist[0, 0] / CFG.d_max + (1 - CFG.w_pos) * (1 - iou[0, 0])
    assert 0 < iou[0, 0] < 1 and dist[0, 0] == pytest.approx(3.0, abs=0.2)
    assert C[0, 0] == pytest.approx(expected)


def test_gating_excludes_far_non_overlapping_cells():
    R0 = disc_frame(SHAPE, (30, 10), 80)
    R1 = disc_frame(SHAPE, (30, 70), 80)
    C, iou, dist = _run([_track_from(R0)], [(0.0, 0.0)], R1)
    assert C[0, 0] == BIG and iou[0, 0] == 0.0 and dist[0, 0] > CFG.d_max
    assert assign(C, CFG.max_cost) == []


def test_gating_keeps_far_centre_when_overlap_is_large():
    # dist > d_max mais IoU >= 0,1 : non écarté (règle v1 : dist > d_max ET iou < 0,1)
    cfg = TrackerConfig(d_max=2.0)
    R0, R1 = disc_frame(SHAPE, (30, 15), 80), disc_frame(SHAPE, (30, 19), 80)
    C, iou, dist = _run([_track_from(R0)], [(0.0, 0.0)], R1, cfg)
    assert dist[0, 0] > cfg.d_max and iou[0, 0] >= 0.1 and C[0, 0] < BIG


def test_cost_matrix_empty_inputs():
    labels, n, df = _cells(disc_frame(SHAPE, (30, 15), 80))
    ov = overlap_counts([], [], labels)
    C, iou, dist = cost_matrix([], df, ov, np.empty((0, 2)), CFG)
    assert C.shape == iou.shape == dist.shape == (0, 1)
    C2, _, _ = cost_matrix([_track_from(disc_frame(SHAPE, (30, 15), 80))], pd.DataFrame(
        columns=["local_id", "area_px", "cy", "cx"]), np.zeros((1, 1), dtype=int), np.zeros((1, 2)), CFG)
    assert C2.shape == (1, 0)


def test_two_track_swap_resolved_globally_by_hungarian():
    # Un glouton lierait (0, 0) (coût 0,20) puis laisserait la piste 1 sans cellule ;
    # l'optimum global lie (0, 1) et (1, 0) : les deux pistes sont affectées.
    C = np.array([[0.20, 0.30],
                  [0.25, BIG]])
    assert sorted(assign(C, 0.75)) == [(0, 1), (1, 0)]


def test_swap_scenario_with_real_blobs_keeps_identities():
    R0 = np.maximum(disc_frame(SHAPE, (30, 15), 80), disc_frame(SHAPE, (30, 40), 80))
    labels0, n0, df0 = _cells(R0)
    tracks = []
    for _, row in df0.iterrows():
        ys, xs = np.nonzero(labels0 == int(row.local_id))
        tracks.append(Track(int(row.local_id), int(row.local_id), row.cy, row.cx, (0.0, 0.0),
                            int(row.area_px), 1.0, ys, xs, 3, 0, False))
    R1 = np.maximum(disc_frame(SHAPE, (30, 18), 80), disc_frame(SHAPE, (30, 43), 80))
    C, _, _ = _run(tracks, [(0.0, 3.0)] * 2, R1)
    assert sorted(assign(C, CFG.max_cost)) == [(0, 0), (1, 1)]


def test_assign_drops_pairs_above_max_cost_and_gated_pairs():
    C = np.array([[0.1, BIG], [BIG, 0.9]])
    assert assign(C, 0.75) == [(0, 0)]


def test_assign_handles_rectangular_and_empty():
    assert assign(np.array([[0.2, 0.1, 0.5]]), 0.75) == [(0, 1)]
    assert assign(np.empty((0, 3)), 0.75) == []
    assert assign(np.empty((2, 0)), 0.75) == []


# ------------------------------------------------------------ prediction_for
def _t(age: int, v=(1.0, 2.0)) -> Track:
    return Track(1, 1, 0.0, 0.0, v, 1, 1.0, np.array([0]), np.array([0]), age, 0, False)


def test_prediction_uses_track_velocity_for_old_tracks():
    field = Motion(5.0, 5.0, 9.0, True)
    assert prediction_for(_t(age=2), field, 1, CFG) == (1.0, 2.0)


def test_prediction_uses_field_motion_for_young_tracks():
    field = Motion(-3.0, 4.0, 9.0, True)
    assert prediction_for(_t(age=1), field, 1, CFG) == (-3.0, 4.0)


def test_prediction_is_zero_when_young_and_field_unreliable_or_missing():
    assert prediction_for(_t(age=1), Motion(3.0, 3.0, 1.0, False), 1, CFG) == (0.0, 0.0)
    assert prediction_for(_t(age=1), None, 1, CFG) == (0.0, 0.0)


def test_prediction_scales_with_dt_steps():
    assert prediction_for(_t(age=5), None, 3, CFG) == (3.0, 6.0)
    assert prediction_for(_t(age=1), Motion(1.0, -2.0, 9.0, True), 2, CFG) == (2.0, -4.0)


def test_prediction_rejects_invalid_dt_steps():
    with pytest.raises(ValueError):
        prediction_for(_t(age=5), None, 0, CFG)


# ------------------------------------------------------- revue : trous de labels
def _gap_labels() -> np.ndarray:
    labels = np.zeros((10, 10), dtype=int)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 2
    labels[8:10, 0:2] = 5  # identifiants 3 et 4 absents (n_cells = 3 ne suffirait pas)
    return labels


def _unit_track(ys, xs, track_id: int = 1) -> Track:
    ys, xs = np.asarray(ys), np.asarray(xs)
    return Track(track_id, 1, float(ys.mean()), float(xs.mean()), (0.0, 0.0), ys.size, 1.0,
                 ys, xs, 3, 0, False)


def test_overlap_counts_sizes_from_largest_label_when_ids_have_gaps():
    track = _unit_track([8, 8, 9, 9], [0, 1, 0, 1])
    counts = overlap_counts([track], [(0.0, 0.0)], _gap_labels())
    assert counts.shape == (1, 6) and counts[0, 5] == 4 and counts.sum() == 4


def test_cost_matrix_works_with_label_gaps():
    track = _unit_track([8, 8, 9, 9], [0, 1, 0, 1])
    cells = pd.DataFrame({"local_id": [1, 2, 5], "area_px": [4, 4, 4],
                          "cy": [1.5, 5.5, 8.5], "cx": [1.5, 5.5, 0.5]})
    ov = overlap_counts([track], [(0.0, 0.0)], _gap_labels())
    C, iou, _ = cost_matrix([track], cells, ov, np.array([[8.5, 0.5]]), CFG)
    assert iou[0].tolist() == [0.0, 0.0, 1.0] and C[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_cost_matrix_rejects_local_id_beyond_overlap_columns():
    track = _unit_track([0], [0])
    cells = pd.DataFrame({"local_id": [1, 5], "area_px": [4, 4], "cy": [0.0, 1.0], "cx": [0.0, 1.0]})
    with pytest.raises(ValueError, match="local_id"):
        cost_matrix([track], cells, np.zeros((1, 3), dtype=int), np.zeros((1, 2)), CFG)


# --------------------------------------------------- revue : égalités, lignes BIG, bord
def test_assign_ties_resolve_deterministically_lowest_index_first():
    # scipy (linear_sum_assignment) rend, pour une matrice constante, l'identité : lignes/colonnes
    # de plus petit indice d'abord. On fige ce comportement.
    C = np.full((2, 2), 0.4)
    assert assign(C, 0.75) == [(0, 0), (1, 1)]
    assert assign(np.full((1, 3), 0.4), 0.75) == [(0, 0)]


def test_assign_all_big_rows_return_no_pair():
    assert assign(np.full((2, 3), BIG), 0.75) == []


def test_edge_exit_iou_uses_full_track_area_v1_semantics():
    # Piste de 4 px advectée de 2 px à gauche : 2 px sortent, 2 recouvrent la cellule (2 px).
    # IoU v1 = 2 / (4 + 2 - 2) = 0,5 (aire pleine de la piste) ; pas 1,0 (pixels dans la grille).
    track = _unit_track([5, 5, 5, 5], [0, 1, 2, 3])
    labels = np.zeros((10, 10), dtype=int)
    labels[5, 0:2] = 1
    cells = pd.DataFrame({"local_id": [1], "area_px": [2], "cy": [5.0], "cx": [0.5]})
    _, _, frac_out = advect_coords(track.ys, track.xs, (0.0, -2.0), (10, 10))
    ov = overlap_counts([track], [(0.0, -2.0)], labels)
    _, iou, _ = cost_matrix([track], cells, ov, np.array([[5.0, 0.5]]), CFG)
    assert frac_out == pytest.approx(0.5) and ov[0, 1] == 2
    assert iou[0, 0] == pytest.approx(0.5)
