"""Tests des helpers synthétiques du tracker v2 (tracking_synth)."""
from datetime import timedelta

import numpy as np
import pytest

from cell_delimitation import MISSING, segment
from tracking_synth import (
    DEFAULT_SHAPE, T0, area_sequence, blob_enters_missing, blob_exits_edge,
    blob_vanishes_one_frame, merging_blobs, rain_frame, splitting_blob,
    translating_blob, make_times,
)


@pytest.mark.unit
def test_rain_frame_blob_has_expected_area_and_dtype():
    # Arrange / Act
    R = rain_frame(DEFAULT_SHAPE, [(30, 40, 5, 5, 60)])
    _, n, _ = segment(R)
    area = int((R > 8.33).sum())
    # Assert
    assert R.dtype == np.int16 and R.shape == DEFAULT_SHAPE
    assert n == 1
    assert abs(area - np.pi * 25) <= 0.15 * np.pi * 25


@pytest.mark.unit
def test_rain_frame_missing_rectangle_is_marked_missing():
    R = rain_frame(DEFAULT_SHAPE, [(10, 10, 4, 4, 50)], missing=(0, 5, 0, 5))
    assert (R[:5, :5] == MISSING).all()
    assert (R == MISSING).sum() == 25


@pytest.mark.unit
def test_make_times_default_and_custom_gap():
    regular = make_times(4)
    gapped = make_times(4, gap_after=1, gap_min=20)
    assert [b - a for a, b in zip(regular, regular[1:])] == [timedelta(minutes=5)] * 3
    assert gapped[2] - gapped[1] == timedelta(minutes=20)
    assert gapped[0] == T0 and gapped[3] - gapped[2] == timedelta(minutes=5)


@pytest.mark.unit
def test_merging_blobs_goes_from_two_components_to_one():
    frames, times = merging_blobs()
    counts = [segment(R)[1] for R in frames]
    assert counts[0] == 2 and counts[-1] == 1
    assert len(frames) == len(times)


@pytest.mark.unit
def test_splitting_blob_is_the_reverse_of_merging():
    frames, _ = splitting_blob()
    counts = [segment(R)[1] for R in frames]
    assert counts[0] == 1 and counts[-1] == 2


@pytest.mark.unit
def test_translating_blob_moves_right():
    frames, _ = translating_blob()
    xs = [np.nonzero(R > 8.33)[1].mean() for R in frames]
    assert all(b > a for a, b in zip(xs, xs[1:]))


@pytest.mark.unit
def test_blob_exits_through_grid_edge():
    frames, _ = blob_exits_edge()
    assert (frames[0] > 8.33).any() and not (frames[-1] > 8.33).any()


@pytest.mark.unit
def test_blob_entering_missing_region_has_minus_one_pixels():
    frames, _ = blob_enters_missing()
    assert all((R == MISSING).any() for R in frames)
    rain = [int((R > 8.33).sum()) for R in frames]
    assert rain[-1] < rain[0]


@pytest.mark.unit
def test_blob_vanishes_for_exactly_one_frame():
    frames, _ = blob_vanishes_one_frame()
    present = [bool((R > 8.33).any()) for R in frames]
    assert present.count(False) == 1 and present[0] and present[-1]


@pytest.mark.unit
def test_area_sequence_follows_52_38_45():
    frames, _ = area_sequence()
    areas = [int((R > 8.33).sum()) for R in frames]
    assert areas == [52, 38, 45]
    assert [segment(R)[1] for R in frames] == [1, 1, 1]
