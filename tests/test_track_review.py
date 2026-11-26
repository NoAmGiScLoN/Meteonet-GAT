"""Tests des planches de revue : cohérence panneau/détections, événements de fin, légende."""
from datetime import timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from cell_delimitation import segment
from frame_stream import iter_frames
from test_track_metrics import T0, _write_run
from track_review import (HIGHLIGHT, OTHERS_UNTRACKED, STEP, anchor_time, panel_overlay,
                          panel_times, predicted_position, write_review)
from track_tables import load_run

CMAP = matplotlib.colormaps["tab20"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root, out = tmp_path_factory.mktemp("raw"), tmp_path_factory.mktemp("out")
    _write_run(root, out)
    frames = {f.t: f for f in iter_frames(T0, T0 + 13 * STEP, root)}
    return root, load_run(out), frames


def _colour_mask(image: np.ndarray, colour) -> np.ndarray:
    return np.all(np.isclose(image, colour), axis=-1)


def test_each_panel_outlines_match_detections_of_its_own_timestamp(run):
    _, tables, frames = run
    det = tables.detections
    for t, frame in frames.items():
        here = det[det.timestamp == t]
        image = panel_overlay(frame, det, track_id=-1)
        labels, _, _ = segment(frame.R)
        expected = np.zeros(labels.shape, dtype=bool)
        for row in here.itertuples():
            mask = labels == row.local_id
            edge = mask & ~ndi.binary_erosion(mask, border_value=0)
            colour = OTHERS_UNTRACKED if row.track_id == 0 else CMAP(row.track_id % 20)
            assert np.array_equal(_colour_mask(image, colour) & edge, edge)
            expected |= edge
        assert np.array_equal(image[..., 3] > 0, expected), t


def test_vanished_blob_has_no_outline_in_its_absent_frame(run):
    _, tables, frames = run
    det = tables.detections
    gone = T0 + 6 * STEP            # le blob A est absent aux cartes 6 et 7
    tid = int(det[det.timestamp == T0 + 5 * STEP].sort_values("cx").track_id.iloc[0])
    assert tid not in set(det[det.timestamp == gone].track_id)
    image = panel_overlay(frames[gone], det, track_id=tid)
    assert not _colour_mask(image, HIGHLIGHT).any()
    assert not _colour_mask(image, CMAP(tid % 20)).any()


def test_panel_times_for_end_and_other_events():
    t, anchor = T0 + 7 * STEP, T0 + 5 * STEP
    assert panel_times("death", t, anchor) == (anchor, t, t + STEP)
    assert panel_times("coast", t, anchor) == (anchor, t, t + STEP)
    assert panel_times("birth", t, anchor) == (t - STEP, t, t + STEP)
    assert panel_times("death", t, None) == (t - STEP, t, t + STEP)


def test_anchor_time_is_last_detection_before_event():
    det = pd.DataFrame({"track_id": [1, 1, 1, 2],
                        "timestamp": [T0, T0 + STEP, T0 + 4 * STEP, T0 + 2 * STEP]})
    assert anchor_time(det, 1, T0 + 6 * STEP) == T0 + 4 * STEP
    assert anchor_time(det, 1, T0) is None
    assert anchor_time(det, 3, T0 + 6 * STEP) is None
    assert anchor_time(det, 1, T0 + 400 * STEP) is None  # trop ancien : repli sur t - 1


def test_predicted_position_extrapolates_last_detection():
    last = pd.Series({"cy": 10.0, "cx": 20.0, "v_y": 1.0, "v_x": -3.0})
    assert predicted_position(last, steps=2) == (12.0, 14.0)


def test_end_event_highlights_track_at_last_detection_not_at_t(run):
    _, tables, frames = run
    death = tables.events[tables.events.event == "death"].iloc[0]
    tid = int(death.track_id)
    anchor = anchor_time(tables.detections, tid, death.t)
    left, mid, _ = panel_times("death", death.t, anchor)
    assert anchor is not None and left == anchor < mid
    assert _colour_mask(panel_overlay(frames[left], tables.detections, tid), HIGHLIGHT).any()
    assert not _colour_mask(panel_overlay(frames[mid], tables.detections, tid), HIGHLIGHT).any()


def test_review_page_with_legend_and_missing_data_renders(run, tmp_path):
    root, tables, _ = run
    png = tmp_path / "review.png"
    sample = write_review(tables.events, tables.detections, root, 100, 0, png)
    assert {s.event for s in sample} >= {"death", "coast"}
    first = tmp_path / "review_01.png"
    assert first.exists() and first.stat().st_size > 5000
