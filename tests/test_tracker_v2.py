"""Tests de bout en bout du tracker v2 (step / track_sequence) sur séquences synthétiques."""
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cell_delimitation import DATA_DIR
from frame_stream import Frame, iter_frames
from motion import Motion
from tracker_state import Track, TrackerConfig, TrackerState
from tracker_v2 import StepOutput, _continue_track, step, track_sequence
from tracking_synth import (T0, area_sequence, blob_enters_missing, blob_exits_edge,
                            blob_vanishes_one_frame, disc_frame, make_times, merging_blobs,
                            rain_frame, splitting_blob, translating_blob, DEFAULT_SHAPE)

CFG = TrackerConfig()
PEAK = 60
NO_FIELD = Motion(0.0, 0.0, 0.0, False)


def _frames(maps: list[np.ndarray], times: list[datetime]) -> list[Frame]:
    frames = []
    for i, (R, t) in enumerate(zip(maps, times)):
        minutes = 0.0 if i == 0 else (t - times[i - 1]).total_seconds() / 60
        frames.append(Frame(t=t, R=R, dt_steps=0 if i == 0 else int(minutes // 5),
                            gap_break=minutes > 15))
    return frames


def _run(seq: tuple[list[np.ndarray], list[datetime]], cfg: TrackerConfig = CFG):
    return track_sequence(_frames(*seq), cfg)


def _all(outs: list[StepOutput], name: str) -> pd.DataFrame:
    parts = [getattr(o, name) for o in outs if len(getattr(o, name))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ids(out: StepOutput) -> list[int]:
    return sorted(int(i) for i in out.detections.track_id if i > 0)


def _events(outs: list[StepOutput]) -> list[str]:
    return list(_all(outs, "events").event)


def test_translating_blob_keeps_one_id_and_velocity_converges():
    # Arrange / Act
    _, outs = _run(translating_blob(n=7, velocity=(0, 3)))
    # Assert
    ids = {i for o in outs for i in _ids(o)}
    assert len(ids) == 1 and all(len(_ids(o)) == 1 for o in outs)
    last = outs[-1].detections.iloc[0]
    assert last.v_src == "track"
    assert last.v_x == pytest.approx(3.0, abs=0.3) and last.v_y == pytest.approx(0.0, abs=0.3)


def test_ten_minute_step_still_links_with_doubled_prediction():
    # Arrange : 0, 5, 10, 20, 25 min, vitesse réelle 3 px / pas
    times = make_times(5, gap_after=2, gap_min=10)
    steps = [0, 1, 2, 4, 5]
    maps = [rain_frame(DEFAULT_SHAPE, [(30, 15 + 3 * s, 5, 5, PEAK)]) for s in steps]
    # Act
    _, outs = _run((maps, times))
    # Assert
    assert len({i for o in outs for i in _ids(o)}) == 1
    assert outs[3].detections.iloc[0].v_x == pytest.approx(3.0, abs=0.3)


def test_blob_missing_one_frame_coasts_and_keeps_id():
    # Act
    _, outs = _run(blob_vanishes_one_frame(n=5, vanish_at=2))
    # Assert
    assert len({i for o in outs for i in _ids(o)}) == 1
    assert _ids(outs[2]) == []
    assert "coast" in list(outs[2].events.event)
    assert outs[3].detections.iloc[0].coasted_before == 1
    assert "death" not in _events(outs)


def test_blob_missing_two_frames_ends_with_death_and_new_id_after():
    # Arrange
    blobs = [[(30, 20, 5, 5, PEAK)]] * 2 + [[]] * 2 + [[(30, 20, 5, 5, PEAK)]]
    maps = [rain_frame(DEFAULT_SHAPE, b) for b in blobs]
    # Act
    _, outs = _run((maps, make_times(5)))
    # Assert
    assert list(outs[2].events.event) == ["coast"]
    assert list(outs[3].events.event) == ["death"]
    assert _ids(outs[4]) != _ids(outs[0]) and _ids(outs[4]) != []


def test_gap_over_fifteen_minutes_ends_all_and_restarts_fresh():
    # Act
    _, outs = _run(translating_blob(n=6, gap_after=2, gap_min=20))
    # Assert
    ev = outs[3].events
    assert list(ev[ev.event == "gap_break"].track_id) == _ids(outs[2])
    assert _ids(outs[3]) != [] and not set(_ids(outs[3])) & set(_ids(outs[2]))
    assert list(outs[3].events.event).count("birth") + list(
        outs[3].events.event).count("entry_edge") == 1


def test_area_between_hysteresis_thresholds_keeps_one_track():
    # Act
    _, outs = _run(area_sequence((52, 38, 45)))
    # Assert
    ids = [_ids(o) for o in outs]
    assert ids[0] != [] and ids[0] == ids[1] == ids[2]


def test_new_cell_of_forty_pixels_never_starts_a_track():
    # Act
    _, outs = _run(area_sequence((40, 40, 40)))
    # Assert
    assert all(_ids(o) == [] for o in outs)
    assert len(_all(outs, "events")) == 0
    assert len(outs[0].detections) == 1


def test_merge_heavier_parent_continues_and_other_ends():
    # Arrange : gros blob à gauche (masse plus grande), petit à droite
    blobs = [[(30, 14 + 5 * i, 8, 8, PEAK), (30, 66 - 5 * i, 5, 5, PEAK)] for i in range(5)]
    maps = [rain_frame(DEFAULT_SHAPE, b) for b in blobs]
    # Act
    _, outs = _run((maps, make_times(5)))
    # Assert
    heavy = _ids(outs[0])[0] if len(_ids(outs[0])) == 2 else None
    big = outs[0].detections.sort_values("area_px").track_id.iloc[-1]
    small = outs[0].detections.sort_values("area_px").track_id.iloc[0]
    assert heavy is not None and _ids(outs[4]) == [big]
    ended = _all(outs, "events")
    assert list(ended[ended.event == "merge_end"].track_id) == [small]
    lin = outs[4].lineage
    assert set(lin.parent_track) == {big, small} and set(lin.link_type) == {"merge"}
    assert list(lin[lin.is_main].parent_track) == [big]


def test_split_heavier_child_inherits_and_other_starts_with_parent():
    # Arrange : fusion de l'asymétrique parcourue à l'envers
    blobs = [[(30, 14 + 5 * i, 8, 8, PEAK), (30, 66 - 5 * i, 5, 5, PEAK)] for i in range(5)]
    maps = [rain_frame(DEFAULT_SHAPE, b) for b in blobs][::-1]
    # Act
    _, outs = _run((maps, make_times(5)))
    # Assert
    parent = _ids(outs[0])[0]
    det = outs[1].detections.sort_values("area_px")
    first_split = next(o for o in outs if len(_ids(o)) == 2)
    d = first_split.detections.sort_values("area_px")
    assert d.track_id.iloc[-1] == parent and d.track_id.iloc[0] != parent
    ev = first_split.events
    row = ev[ev.event == "split_start"]
    assert list(row.track_id) == [d.track_id.iloc[0]] and str(parent) in row.notes.iloc[0]
    assert set(first_split.lineage.parent_track) == {parent}
    assert det is not None


def test_edge_exit_gives_exit_edge_with_truncated_last_frames():
    # Act
    _, outs = _run(blob_exits_edge(n=5))
    # Assert
    ev = _all(outs, "events")
    assert list(ev[ev.event == "exit_edge"].truncated) == [True]
    assert outs[2].detections.iloc[0].truncated


def test_entry_from_edge_gives_entry_edge():
    # Arrange
    maps, times = blob_exits_edge(n=5)
    # Act
    _, outs = _run((maps[::-1], times))
    # Assert
    ev = _all(outs, "events")
    assert list(ev[ev.event == "entry_edge"].truncated) == [True]
    assert "birth" not in list(ev.event)


def test_moving_into_missing_region_gives_exit_missing():
    # Act
    _, outs = _run(blob_enters_missing(n=5))
    # Assert
    assert _events(outs).count("exit_missing") == 1


def test_first_frame_creates_births_and_entries_for_cells_over_fifty_px():
    # Arrange : 1 intérieur, 1 au bord, 1 petit
    maps = [rain_frame(DEFAULT_SHAPE, [(30, 20, 5, 5, PEAK), (10, 77, 6, 6, PEAK),
                                       (45, 40, 3, 3, PEAK)])]
    # Act
    _, outs = _run((maps, make_times(1)))
    # Assert
    assert sorted(outs[0].events.event) == ["birth", "entry_edge"]
    assert len(outs[0].detections) == 3 and len(_ids(outs[0])) == 2
    assert (outs[0].detections.v_src == "none").all()


def test_detections_have_unique_time_and_track_id():
    # Act
    _, outs = _run(merging_blobs(n=6))
    _, outs2 = _run(splitting_blob(n=6))
    # Assert
    for o in outs + outs2:
        tracked = o.detections[o.detections.track_id > 0]
        assert not tracked.duplicated(["timestamp", "track_id"]).any()


def _snapshot(state: TrackerState) -> list:
    return [(k.track_id, k.ys.copy(), k.xs.copy(), k.v, k.area, k.mass, k.age, k.coasted,
             k.gap_steps, k.cy, k.cx) for k in state.tracks] + [state.next_id, state.t]


def _same(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, tuple):
            ok = all(np.array_equal(p, q) if isinstance(p, np.ndarray) else p == q
                     for p, q in zip(x, y))
        else:
            ok = x == y
        if not ok:
            return False
    return True


def test_step_does_not_mutate_input_state():
    # Arrange
    frames = _frames(*translating_blob(n=3))
    state, _ = track_sequence(frames[:2], CFG)
    before = _snapshot(state)
    # Act
    new_state, _ = step(state, frames[2], CFG, frames[1].R)
    # Assert : contenus identiques (ids, coordonnées, vitesses, aires...) avant / après
    assert _same(before, _snapshot(state))
    assert not _same(before, _snapshot(new_state))
    assert new_state.t == frames[2].t


def test_step_rejects_non_positive_dt_with_active_tracks():
    # Arrange
    frames = _frames(*translating_blob(n=2))
    state, _ = track_sequence(frames[:1], CFG)
    bad = Frame(t=frames[1].t, R=frames[1].R, dt_steps=0, gap_break=False)
    # Act / Assert
    with pytest.raises(ValueError):
        step(state, bad, CFG, frames[0].R)


def test_step_output_is_frozen():
    # Act
    _, outs = _run(translating_blob(n=2))
    # Assert
    with pytest.raises(Exception):
        outs[0].detections = pd.DataFrame()  # type: ignore[misc]


def _track(age: int, v: tuple[float, float] = (0.0, 3.0), gap_steps: int = 1) -> Track:
    ys, xs = np.mgrid[20:30, 20:30]
    return Track(track_id=1, local_id=1, cy=25.0, cx=25.0, v=v, area=100, mass=100.0,
                 ys=ys.ravel(), xs=xs.ravel(), age=age, coasted=0, truncated=False,
                 gap_steps=gap_steps)


def _row(cy: float, cx: float) -> pd.Series:
    return pd.Series({"local_id": 1, "cy": cy, "cx": cx, "area_px": 100, "intensity_sum": 100.0,
                      "border_flag": False, "missing_neighbor_flag": False})


PIX = (np.arange(100, dtype=np.int32), np.arange(100, dtype=np.int32))


def test_young_track_velocity_blends_field_instead_of_discarding_it():
    # Arrange : champ fiable (0, 4) par pas, déplacement observé (0, 2)
    field = Motion(0.0, 4.0, 5.0, True)
    # Act
    _, vel = _continue_track(_track(age=1), _row(25.0, 27.0), PIX, CFG, (0.0, 4.0), field)
    # Assert : EMA alpha 0.5 amorcée par le champ : (4 + 2) / 2 = 3
    assert vel.v == pytest.approx((0.0, 3.0)) and vel.src == "track"
    assert vel.pred_src == "field" and not vel.reset


def test_implausible_displacement_keeps_predicted_velocity_and_flags_reset():
    # Arrange : saut de centre de 40 px (> d_max) après fusion
    track = _track(age=5, v=(0.0, 3.0))
    # Act
    new, vel = _continue_track(track, _row(25.0, 65.0), PIX, CFG, (0.0, 3.0), NO_FIELD)
    # Assert
    assert vel.reset and new.v == (0.0, 3.0) and vel.src == "track"


def test_implausible_displacement_of_young_track_falls_back_on_field():
    # Act
    _, vel = _continue_track(_track(age=1), _row(25.0, 65.0), PIX, CFG, (0.0, 4.0),
                             Motion(0.0, 4.0, 5.0, True))
    # Assert
    assert vel.reset and vel.v == (0.0, 4.0) and vel.src == "field"


def test_merge_keeps_velocity_of_kept_track_close_to_true_motion():
    # Arrange : gros blob (+5 px / pas) fusionné avec un petit
    blobs = [[(30, 14 + 5 * i, 8, 8, PEAK), (30, 66 - 5 * i, 5, 5, PEAK)] for i in range(5)]
    maps = [rain_frame(DEFAULT_SHAPE, b) for b in blobs]
    # Act
    _, outs = _run((maps, make_times(5)))
    # Assert
    assert outs[4].detections.iloc[0].v_x == pytest.approx(5.0, abs=2.0)


def test_v_src_and_pred_src_values_over_a_sequence():
    # Act
    _, outs = _run(translating_blob(n=5, velocity=(0, 3)))
    # Assert
    first, last = outs[0].detections.iloc[0], outs[-1].detections.iloc[0]
    assert (first.v_src, first.pred_src) == ("none", "none")
    assert (last.v_src, last.pred_src) == ("track", "track")
    assert not outs[-1].detections.velocity_reset.any()


def test_split_children_inherit_parent_velocity():
    # Arrange
    blobs = [[(30, 14 + 5 * i, 8, 8, PEAK), (30, 66 - 5 * i, 5, 5, PEAK)] for i in range(5)]
    maps = [rain_frame(DEFAULT_SHAPE, b) for b in blobs][::-1]
    # Act
    _, outs = _run((maps, make_times(5)))
    # Assert
    k = next(i for i, o in enumerate(outs) if len(_ids(o)) == 2)
    parent = outs[k - 1].detections.iloc[0]
    child = outs[k].detections.sort_values("area_px").iloc[0]
    assert child.v_x == parent.v_x and child.v_src == "track"


def test_fifteen_minute_step_without_break_links_moving_blob():
    # Arrange : 0, 5, 20, 25 min (écart de 15 min = 3 pas, pas de coupure), 3 px / pas
    times = make_times(4, gap_after=1, gap_min=15)
    maps = [rain_frame(DEFAULT_SHAPE, [(30, 15 + 3 * s, 5, 5, PEAK)]) for s in (0, 1, 4, 5)]
    # Act
    _, outs = _run((maps, times))
    # Assert
    assert len({i for o in outs for i in _ids(o)}) == 1
    assert "gap_break" not in _events(outs)


def test_coasted_track_rematched_with_velocity_divided_by_elapsed_steps():
    # Act
    _, outs = _run(blob_vanishes_one_frame(n=5, vanish_at=2))
    # Assert : déplacement de 6 px sur 2 pas -> 3 px / pas
    after = outs[3].detections.iloc[0]
    assert after.coasted_before == 1 and after.v_x == pytest.approx(3.0, abs=0.5)
    assert after.pred_src == "track"


def test_track_rejects_non_boolean_flags():
    # Act / Assert
    with pytest.raises(ValueError):
        Track(track_id=1, local_id=1, cy=0.0, cx=0.0, v=(0.0, 0.0), area=1, mass=1.0,
              ys=np.zeros(1), xs=np.zeros(1), age=1, coasted=0, truncated=False,
              border_flag="oui")  # type: ignore[arg-type]


@pytest.mark.integration
def test_real_frames_tracked_without_error_and_timed():
    if not DATA_DIR.exists():
        pytest.skip("DATA/ absent")
    start = datetime(2016, 1, 1, 12, 0)
    frames = list(iter_frames(start, start + timedelta(minutes=55)))
    assert len(frames) == 12
    state, prev, outs, durations = TrackerState(), None, [], []
    for f in frames:
        t0 = time.perf_counter()
        state, out = step(state, f, CFG, prev)
        durations.append(time.perf_counter() - t0)
        outs.append(out)
        prev = f.R
    for o in outs:
        tracked = o.detections[o.detections.track_id > 0]
        assert not tracked.duplicated(["timestamp", "track_id"]).any()
    big = [o.detections[o.detections.area_px >= 200] for o in outs[1:]]
    share = sum((b.track_id > 0).sum() for b in big) / max(sum(len(b) for b in big), 1)
    counts = _all(outs, "events").event.value_counts().to_dict()
    dets = _all(outs, "detections")
    life = dets[dets.track_id > 0].groupby("track_id").size()
    n_tracks = len(life)
    short = int((life <= 2).sum())
    print(f"\nper frame: {np.mean(durations):.2f}s, events: {counts}, tracks: {n_tracks}, "
          f"<=2 frames: {short}")
    assert share >= 0.9
    assert counts.get("split_start", 0) + counts.get("merge_end", 0) <= 2 * n_tracks
    assert short <= 0.8 * n_tracks
