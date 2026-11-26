"""Tests de la généalogie (v2) : liens de recouvrement, fusions, scissions, événements de bord."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from association import overlap_counts
from cell_delimitation import describe, segment
from lineage import (EVENT_TYPES, GroupResult, Link, classify_groups, edge_event,
                      is_truncated, overlap_links)
from tracker_state import Track, TrackerConfig
from tracking_synth import (blob_enters_missing, blob_exits_edge, merging_blobs,
                            splitting_blob)

CFG = TrackerConfig()


def _track(track_id: int, area: int = 100, mass: float = 100.0) -> Track:
    ys = np.arange(area)
    return Track(track_id=track_id, local_id=track_id, cy=0.0, cx=0.0, v=(0.0, 0.0), area=area,
                 mass=mass, ys=ys, xs=np.zeros(area), age=3, coasted=0, truncated=False)


def _cells(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """rows : (local_id, area_px, intensity_sum)."""
    return pd.DataFrame({"local_id": [r[0] for r in rows], "area_px": [r[1] for r in rows],
                         "intensity_sum": [r[2] for r in rows],
                         "border_flag": False, "missing_neighbor_flag": False})


def _cand(t: int, lid: int, inter: int = 50, fp: float = 0.5, fc: float = 0.5):
    return (t, lid, inter, fp, fc)


def _links_by_pair(result: GroupResult) -> dict:
    return {(lk.parent_track, lk.child_local_id): lk for lk in result.links}


# ------------------------------------------------------------ overlap_links
def test_overlap_links_keeps_pair_when_both_fractions_reach_f_min():
    tracks, cells = [_track(1, area=100)], _cells([(1, 100, 10.0)])
    out = overlap_links(np.array([[0, 10]]), tracks, cells, CFG)  # 0.1 et 0.1
    assert out == [(0, 1, 10, 0.1, 0.1)]


def test_overlap_links_drops_small_fragment_touching_large_cell():
    # 10/100 = 0.1 de la piste (petite) mais 10/1000 = 0.01 de la cellule : pas de lien (ET)
    tracks, cells = [_track(1, area=100)], _cells([(1, 1000, 10.0)])
    assert overlap_links(np.array([[0, 10]]), tracks, cells, CFG) == []


def test_overlap_links_drops_small_cell_touching_large_track():
    tracks, cells = [_track(1, area=1000)], _cells([(1, 100, 10.0)])
    assert overlap_links(np.array([[0, 10]]), tracks, cells, CFG) == []


def test_overlap_links_drops_pair_below_f_min_in_both_directions():
    tracks, cells = [_track(1, area=100)], _cells([(1, 100, 10.0)])
    assert overlap_links(np.array([[0, 9]]), tracks, cells, CFG) == []


def test_overlap_links_ignores_background_and_zero_columns():
    tracks, cells = [_track(1)], _cells([(2, 100, 10.0)])
    out = overlap_links(np.array([[500, 0, 80]]), tracks, cells, CFG)
    assert [(c[0], c[1], c[2]) for c in out] == [(0, 2, 80)]


def test_overlap_links_empty_inputs():
    assert overlap_links(np.zeros((0, 3), dtype=int), [], _cells([(1, 10, 1.0)]), CFG) == []


def test_overlap_links_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        overlap_links(np.zeros((2, 3), dtype=int), [_track(1)], _cells([(1, 10, 1.0)]), CFG)
    with pytest.raises(ValueError):
        overlap_links(np.zeros((1, 2), dtype=int), [_track(1)], _cells([(5, 10, 1.0)]), CFG)
    with pytest.raises(ValueError):
        overlap_links(np.zeros(3, dtype=int), [_track(1)], _cells([(1, 10, 1.0)]), CFG)


# --------------------------------------------------------- classify_groups
def test_continuation_one_to_one():
    res = classify_groups([_cand(0, 1)], [_track(7)], _cells([(1, 100, 100.0)]), [])
    assert dict(res.continuations) == {7: 1}
    assert dict(res.ended) == {} and res.new_children == ()
    (link,) = res.links
    assert (link.link_type, link.is_main) == ("continuation", True)
    assert (link.share_of_child, link.share_of_parent) == (1.0, 1.0)


def test_merge_heavier_parent_continues_and_shares_sum_to_one():
    tracks = [_track(1, mass=300.0), _track(2, mass=100.0)]
    res = classify_groups([_cand(0, 1), _cand(1, 1)], tracks, _cells([(1, 200, 400.0)]), [])
    assert dict(res.continuations) == {1: 1}
    assert dict(res.ended) == {2: "merge_end"}
    links = _links_by_pair(res)
    assert links[(1, 1)].is_main and not links[(2, 1)].is_main
    assert links[(1, 1)].share_of_child == pytest.approx(0.75)
    assert sum(lk.share_of_child for lk in res.links) == pytest.approx(1.0)
    assert [lk.share_of_parent for lk in res.links] == [1.0, 1.0]
    assert {lk.link_type for lk in res.links} == {"merge"}


def test_merge_equal_masses_resolved_by_overlap_then_id():
    cells = _cells([(1, 200, 400.0)])
    tracks = [_track(1, mass=50.0), _track(2, mass=50.0)]
    by_overlap = classify_groups([_cand(0, 1, inter=40), _cand(1, 1, inter=60)], tracks, cells, [])
    assert dict(by_overlap.continuations) == {2: 1}
    by_id = classify_groups([_cand(0, 1, inter=50), _cand(1, 1, inter=50)], tracks, cells, [])
    assert dict(by_id.continuations) == {1: 1} and dict(by_id.ended) == {2: "merge_end"}


def test_split_heavier_child_inherits_and_other_starts_with_parent():
    cells = _cells([(1, 50, 30.0), (2, 80, 90.0)])
    res = classify_groups([_cand(0, 1), _cand(0, 2)], [_track(5)], cells, [])
    assert dict(res.continuations) == {5: 2}
    assert dict(res.ended) == {}
    assert res.new_children == ((1, 5, "split_start"),)
    links = _links_by_pair(res)
    assert links[(5, 2)].share_of_parent == pytest.approx(0.75) and links[(5, 2)].is_main
    assert links[(5, 1)].share_of_parent == pytest.approx(0.25) and not links[(5, 1)].is_main
    assert [lk.share_of_child for lk in res.links] == [1.0, 1.0]
    assert {lk.link_type for lk in res.links} == {"split"}


def test_split_equal_masses_prefers_larger_overlap_then_lower_local_id():
    cells = _cells([(1, 50, 30.0), (2, 50, 30.0)])
    big = classify_groups([_cand(0, 1, inter=20), _cand(0, 2, inter=30)], [_track(5)], cells, [])
    assert dict(big.continuations) == {5: 2}
    tie = classify_groups([_cand(0, 1, inter=20), _cand(0, 2, inter=20)], [_track(5)], cells, [])
    assert dict(tie.continuations) == {5: 1}


def test_many_to_many_is_deterministic_and_records_all_links():
    # t1 (m=300) -> c1, c2 ; t2 (m=100) -> c2 ; c1 m=50, c2 m=200
    tracks = [_track(1, mass=300.0), _track(2, mass=100.0)]
    cells = _cells([(1, 60, 50.0), (2, 140, 200.0)])
    cands = [_cand(0, 1), _cand(0, 2), _cand(1, 2)]
    res = classify_groups(cands, tracks, cells, [])
    # c1 -> parent t1 ; c2 -> parent t1 (plus lourd) ; t1 garde c2 (enfant le plus lourd)
    assert dict(res.continuations) == {1: 2}
    assert dict(res.ended) == {2: "merge_end"}
    assert res.new_children == ((1, 1, "split_start"),)
    assert len(res.links) == 3
    assert sum(lk.is_main for lk in res.links) == 1
    assert classify_groups(list(reversed(cands)), tracks, cells, []) == res
    links = _links_by_pair(res)
    assert links[(1, 1)].share_of_child == 1.0
    assert links[(1, 1)].share_of_parent == pytest.approx(0.2)
    assert links[(1, 2)].share_of_child == pytest.approx(0.75)
    assert links[(1, 2)].share_of_parent == pytest.approx(0.8)
    assert links[(2, 2)].share_of_child == pytest.approx(0.25)
    assert links[(2, 2)].share_of_parent == 1.0


def test_many_to_many_cross_links_heavier_track_wins_both_cells_and_keeps_one():
    # liens croisés complets ; t1 (plus lourde) est parent principal des deux cellules
    tracks = [_track(1, mass=300.0), _track(2, mass=200.0)]
    cells = _cells([(1, 100, 100.0), (2, 100, 100.0)])
    cands = [_cand(0, 1), _cand(0, 2), _cand(1, 1), _cand(1, 2)]
    res = classify_groups(cands, tracks, cells, [])
    # t1 est parent principal des deux cellules ; garde la plus lourde (égalité -> recouvr. -> id)
    assert dict(res.continuations) == {1: 1}
    assert dict(res.ended) == {2: "merge_end"}
    assert res.new_children == ((2, 1, "split_start"),)


def test_hungarian_distance_only_continuation_is_kept():
    res = classify_groups([], [_track(3)], _cells([(4, 40, 10.0)]), [(0, 0)])
    assert dict(res.continuations) == {3: 4}
    assert res.links == () and dict(res.ended) == {}


def test_hungarian_pair_with_overlap_only_on_cell_leaves_track_unmatched():
    # c1 recouverte par t1 ; la paire hongroise (t2, c1) n'a aucun recouvrement côté t2
    tracks = [_track(1), _track(2)]
    res = classify_groups([_cand(0, 1)], tracks, _cells([(1, 100, 10.0)]), [(1, 0)])
    assert dict(res.continuations) == {1: 1}
    assert res.unmatched_tracks == (2,) and res.unmatched_cells == ()


def test_hungarian_pair_with_overlap_only_on_track_leaves_cell_unmatched():
    tracks = [_track(1)]
    cells = _cells([(1, 100, 10.0), (2, 100, 10.0)])
    res = classify_groups([_cand(0, 1)], tracks, cells, [(0, 1)])
    assert dict(res.continuations) == {1: 1}
    assert res.unmatched_cells == (2,) and res.unmatched_tracks == ()


def test_unlinked_tracks_and_cells_are_reported_unmatched():
    tracks = [_track(1), _track(2)]
    cells = _cells([(1, 100, 10.0), (2, 100, 10.0)])
    res = classify_groups([_cand(0, 1)], tracks, cells, [])
    assert res.unmatched_tracks == (2,) and res.unmatched_cells == (2,)


def test_non_eligible_non_main_child_is_not_started_but_link_kept():
    cells = _cells([(1, 50, 30.0), (2, 80, 90.0)])
    res = classify_groups([_cand(0, 1), _cand(0, 2)], [_track(5)], cells, [],
                          eligible_new=frozenset())
    assert dict(res.continuations) == {5: 2}
    assert res.new_children == () and res.unmatched_cells == (1,)
    assert _links_by_pair(res)[(5, 1)].link_type == "split"
    assert not _links_by_pair(res)[(5, 1)].is_main


def test_eligible_non_main_child_is_started():
    cells = _cells([(1, 50, 30.0), (2, 80, 90.0)])
    res = classify_groups([_cand(0, 1), _cand(0, 2)], [_track(5)], cells, [],
                          eligible_new=frozenset({1}))
    assert res.new_children == ((1, 5, "split_start"),) and res.unmatched_cells == ()


def test_hungarian_pair_inside_overlap_group_does_not_override_overlap():
    tracks = [_track(1, mass=10.0), _track(2, mass=90.0)]
    res = classify_groups([_cand(0, 1), _cand(1, 1)], tracks, _cells([(1, 100, 100.0)]), [(0, 0)])
    assert dict(res.continuations) == {2: 1}


def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    n_t, n_c = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    areas_t, areas_c = rng.integers(20, 200, n_t), rng.integers(20, 200, n_c)
    tracks = [_track(10 + i, area=int(areas_t[i]), mass=float(rng.integers(0, 5) * 50))
              for i in range(n_t)]
    cells = _cells([(i + 1, int(areas_c[i]), float(rng.integers(0, 5) * 50)) for i in range(n_c)])
    overlaps = np.zeros((n_t, n_c + 1), dtype=int)
    for t in range(n_t):
        for c in range(n_c):
            if rng.random() < 0.4:
                overlaps[t, c + 1] = rng.integers(1, min(areas_t[t], areas_c[c]) + 1)
    rows, cols = rng.permutation(n_t), rng.permutation(n_c)
    k = int(rng.integers(0, min(n_t, n_c) + 1))
    pairs = [(int(r), int(c)) for r, c in zip(rows[:k], cols[:k])]
    eligible = frozenset(int(i + 1) for i in range(n_c) if rng.random() < 0.5)
    return tracks, cells, overlaps, pairs, eligible


@pytest.mark.parametrize("seed", range(60))
def test_result_partitions_tracks_and_cells_and_shares_sum_to_one(seed):
    tracks, cells, overlaps, pairs, eligible = _random_case(seed)
    cands = overlap_links(overlaps, tracks, cells, CFG)
    res = classify_groups(cands, tracks, cells, pairs, eligible_new=eligible)
    t_parts = [list(res.continuations), list(res.ended), list(res.unmatched_tracks)]
    assert sorted(sum(t_parts, [])) == sorted(t.track_id for t in tracks)
    c_parts = [list(res.continuations.values()), [c[0] for c in res.new_children],
               list(res.unmatched_cells)]
    assert sorted(sum(c_parts, [])) == sorted(int(i) for i in cells.local_id)
    for lid in {lk.child_local_id for lk in res.links}:
        group = [lk for lk in res.links if lk.child_local_id == lid]
        assert sum(lk.share_of_child for lk in group) == pytest.approx(1.0)
    for tid in {lk.parent_track for lk in res.links}:
        group = [lk for lk in res.links if lk.parent_track == tid]
        assert sum(lk.share_of_parent for lk in group) == pytest.approx(1.0)
    assert not set(res.continuations.values()) & {c[0] for c in res.new_children}
    assert all(c[0] in eligible for c in res.new_children if c[2] == "split_start")


@pytest.mark.parametrize("seed", range(20))
def test_output_is_invariant_under_permutation_of_tracks_and_cells(seed):
    tracks, cells, overlaps, pairs, eligible = _random_case(seed)
    base = classify_groups(overlap_links(overlaps, tracks, cells, CFG), tracks, cells, pairs,
                           eligible_new=eligible)
    rng = np.random.default_rng(1000 + seed)
    tp, cp = rng.permutation(len(tracks)), rng.permutation(len(cells))
    tracks2 = [tracks[i] for i in tp]
    cells2 = cells.iloc[cp].reset_index(drop=True)
    inv_t, inv_c = np.argsort(tp), np.argsort(cp)
    overlaps2 = overlaps[tp]  # colonnes = local_id, inchangés
    pairs2 = [(int(inv_t[t]), int(inv_c[c])) for t, c in pairs]
    again = classify_groups(overlap_links(overlaps2, tracks2, cells2, CFG), tracks2, cells2,
                            pairs2, eligible_new=eligible)
    assert again == base
    assert [(lk.parent_track, lk.child_local_id) for lk in again.links] == sorted(
        (lk.parent_track, lk.child_local_id) for lk in again.links)


def test_independent_groups_are_resolved_separately():
    tracks = [_track(1), _track(2)]
    cells = _cells([(1, 100, 10.0), (2, 100, 10.0)])
    res = classify_groups([_cand(0, 1), _cand(1, 2)], tracks, cells, [])
    assert dict(res.continuations) == {1: 1, 2: 2}


def test_zero_mass_parents_share_equally():
    tracks = [_track(1, mass=0.0), _track(2, mass=0.0)]
    res = classify_groups([_cand(0, 1), _cand(1, 1)], tracks, _cells([(1, 100, 0.0)]), [])
    assert [lk.share_of_child for lk in res.links] == [0.5, 0.5]


def test_outputs_are_frozen():
    res = classify_groups([_cand(0, 1)], [_track(7)], _cells([(1, 100, 100.0)]), [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.links = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        res.continuations[8] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        res.ended[8] = "death"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.links[0].is_main = False  # type: ignore[misc]


@pytest.mark.parametrize("candidates, hungarian", [
    ([_cand(3, 1)], []),                       # indice de piste hors bornes
    ([_cand(0, 9)], []),                       # local_id inconnu
    ([_cand(0, 1), _cand(0, 1)], []),          # paire dupliquée
    ([_cand(0, 1, fp=1.5)], []),               # fraction hors [0, 1]
    ([], [(0, 5)]),                            # indice cellule hors bornes
    ([], [(0, 0), (0, 0)]),                    # piste dupliquée
])
def test_classify_groups_rejects_inconsistent_inputs(candidates, hungarian):
    with pytest.raises(ValueError):
        classify_groups(candidates, [_track(1)], _cells([(1, 100, 10.0)]), hungarian)


def test_classify_groups_rejects_duplicate_track_ids_and_local_ids():
    with pytest.raises(ValueError):
        classify_groups([], [_track(1), _track(1)], _cells([(1, 10, 1.0)]), [])
    with pytest.raises(ValueError):
        classify_groups([], [_track(1)], _cells([(1, 10, 1.0), (1, 10, 1.0)]), [])


@pytest.mark.parametrize("row", [(1, 100, float("nan")), (1, 100, -1.0), (1, 0, 10.0),
                                 (1, 100, float("inf"))])
def test_classify_groups_rejects_invalid_cell_masses_and_areas(row):
    with pytest.raises(ValueError):
        classify_groups([], [_track(1)], _cells([row]), [])


def test_link_validates_fields():
    with pytest.raises(ValueError):
        Link(1, 1, "bogus", 0.5, 0.5, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        Link(1, 1, "merge", 0.5, 0.5, 1.5, 1.0, True)


# ------------------------------------------------- séquences synthétiques
def _segment_frame(R: np.ndarray):
    labels, n, valid = segment(R)
    return labels, describe(R, labels, n, valid, None)


def _tracks_from(R: np.ndarray, labels: np.ndarray, df: pd.DataFrame) -> list[Track]:
    out = []
    for row in df.itertuples():
        ys, xs = np.nonzero(labels == row.local_id)
        out.append(Track(track_id=row.local_id, local_id=row.local_id, cy=row.cy, cx=row.cx,
                         v=(0.0, 0.0), area=int(row.area_px), mass=float(row.intensity_sum),
                         ys=ys, xs=xs, age=3, coasted=0, truncated=False))
    return out


def _run(R0: np.ndarray, R1: np.ndarray) -> GroupResult:
    l0, df0 = _segment_frame(R0)
    l1, df1 = _segment_frame(R1)
    tracks = _tracks_from(R0, l0, df0)
    ov = overlap_counts(tracks, [(0.0, 0.0)] * len(tracks), l1)
    return classify_groups(overlap_links(ov, tracks, df1, CFG), tracks, df1, [])


def test_synthetic_merge_sequence():
    frames, _ = merging_blobs()
    res = _run(frames[3], frames[4])
    assert len(res.continuations) == 1 and len(res.ended) == 1
    assert set(res.ended.values()) == {"merge_end"}
    assert sum(lk.share_of_child for lk in res.links) == pytest.approx(1.0)


def test_synthetic_split_sequence():
    frames, _ = splitting_blob()
    res = _run(frames[1], frames[2])
    assert len(res.continuations) == 1 and len(res.new_children) == 1
    assert res.new_children[0][2] == "split_start"
    assert all(lk.link_type == "split" for lk in res.links)


# ------------------------------------------------------------- événements
def _flags(border: bool = False, missing: bool = False) -> dict:
    return {"border_flag": border, "missing_neighbor_flag": missing}


@pytest.mark.parametrize("flags, frac, expected", [
    (_flags(border=True), 0.0, "exit_edge"),
    (_flags(), 0.5, "exit_edge"),
    (_flags(), 0.49, "death"),
    (_flags(missing=True), 0.0, "exit_missing"),
    (_flags(border=True, missing=True), 0.0, "exit_edge"),
    (_flags(missing=True), 0.6, "exit_edge"),
    (_flags(), 0.0, "death"),
])
def test_edge_event_end(flags, frac, expected):
    assert edge_event(flags, "end", frac) == expected


@pytest.mark.parametrize("flags, expected", [
    (_flags(border=True), "entry_edge"),
    (_flags(missing=True), "entry_missing"),
    (_flags(border=True, missing=True), "entry_edge"),
    (_flags(), "birth"),
])
def test_edge_event_start(flags, expected):
    assert edge_event(flags, "start", 0.0) == expected


def test_edge_event_accepts_series_and_validates():
    row = pd.Series(_flags(border=True))
    assert edge_event(row, "start", 0.0) == "entry_edge"
    with pytest.raises(ValueError):
        edge_event(_flags(), "middle", 0.0)
    with pytest.raises(ValueError):
        edge_event(_flags(), "end", 1.5)
    with pytest.raises(ValueError):
        edge_event({"border_flag": True}, "end", 0.0)


def test_is_truncated():
    assert is_truncated(_flags(border=True)) and is_truncated(_flags(missing=True))
    assert not is_truncated(_flags())
    assert is_truncated(pd.Series(_flags(missing=True)))


def test_synthetic_edge_exit_is_classified_exit_edge():
    frames, _ = blob_exits_edge()
    _, df = _segment_frame(frames[2])
    assert bool(df.iloc[0].border_flag)
    assert edge_event(df.iloc[0], "end", 0.0) == "exit_edge"


def test_synthetic_missing_entry_is_classified_exit_missing():
    frames, _ = blob_enters_missing()
    _, df = _segment_frame(frames[3])
    row = df.iloc[0]
    assert bool(row.missing_neighbor_flag) and not bool(row.border_flag)
    assert edge_event(row, "end", 0.0) == "exit_missing"
    assert edge_event(row, "start", 0.0) == "entry_missing"


def test_event_types_cover_the_spec():
    assert set(EVENT_TYPES) == {"birth", "death", "entry_edge", "exit_edge", "entry_missing",
                                "exit_missing", "merge_end", "split_start", "coast", "gap_break"}
