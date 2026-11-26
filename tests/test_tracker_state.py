"""Tests de l'état du tracker v2 : configuration validée, Track et TrackerState immuables."""
from dataclasses import FrozenInstanceError
from datetime import datetime

import numpy as np
import pytest

from tracker_state import Track, TrackerConfig, TrackerState


def _track(track_id: int = 1, **kw) -> Track:
    args = dict(track_id=track_id, local_id=3, cy=10.0, cx=20.0, v=(0.0, 2.0), area=3, mass=500.0,
                ys=np.array([9, 10, 11]), xs=np.array([20, 20, 20]), age=3, coasted=0, truncated=False)
    return Track(**{**args, **kw})


def test_config_defaults_match_plan():
    cfg = TrackerConfig()
    assert (cfg.min_area_start, cfg.min_area_keep, cfg.max_coast) == (50, 35, 1)
    assert (cfg.max_gap_min, cfg.d_max, cfg.w_pos, cfg.max_cost) == (15, 25.0, 0.5, 0.75)
    assert (cfg.f_min, cfg.alpha, cfg.young_frames, cfg.coarse, cfg.min_peak_ratio) == (0.1, 0.5, 2, 4, 2.0)


@pytest.mark.parametrize("kw", [
    dict(min_area_keep=60, min_area_start=50),
    dict(min_area_keep=0),
    dict(max_coast=-1),
    dict(max_gap_min=0),
    dict(d_max=0.0),
    dict(d_max=-1.0),
    dict(w_pos=0.0), dict(w_pos=1.5),
    dict(max_cost=0.0), dict(max_cost=1.2),
    dict(f_min=0.0), dict(f_min=1.1),
    dict(alpha=0.0), dict(alpha=1.5),
    dict(young_frames=0),
    dict(coarse=0),
    dict(min_peak_ratio=0.0),
])
def test_config_rejects_invalid_values(kw):
    with pytest.raises(ValueError):
        TrackerConfig(**kw)


def test_config_accepts_boundaries():
    cfg = TrackerConfig(min_area_keep=50, min_area_start=50, w_pos=1.0, alpha=1.0, f_min=1.0, max_coast=0)
    assert cfg.max_coast == 0


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        TrackerConfig().d_max = 3.0  # type: ignore[misc]


def test_track_is_frozen_and_arrays_read_only():
    ys = np.array([1, 2, 3])
    track = _track(ys=ys)
    with pytest.raises(FrozenInstanceError):
        track.cy = 0.0  # type: ignore[misc]
    assert not track.ys.flags.writeable and not track.xs.flags.writeable
    assert track.ys.dtype == np.int32
    ys[0] = 99  # l'entrée de l'appelant n'affecte pas le Track
    assert track.ys[0] == 1


def test_track_rejects_mismatched_pixel_arrays():
    with pytest.raises(ValueError):
        _track(ys=np.array([1, 2]), xs=np.array([1]))


def test_track_update_with_replace_does_not_mutate_original():
    from dataclasses import replace
    original = _track()
    updated = replace(original, age=original.age + 1, coasted=1)
    assert (original.age, original.coasted) == (3, 0)
    assert (updated.age, updated.coasted) == (4, 1)


def test_state_default_is_empty():
    state = TrackerState()
    assert state.tracks == () and state.next_id == 1 and state.t is None


def test_state_allocate_id_returns_new_state():
    state = TrackerState()
    new_id, after = state.allocate_id()
    assert new_id == 1 and after.next_id == 2 and state.next_id == 1


def test_state_with_tracks_and_time_do_not_mutate():
    state = TrackerState()
    t = datetime(2016, 1, 1, 12, 0)
    after = state.with_tracks((_track(),)).with_time(t)
    assert state.tracks == () and state.t is None
    assert len(after.tracks) == 1 and after.t == t


def test_state_with_tracks_accepts_any_iterable_and_stores_tuple():
    after = TrackerState().with_tracks([_track(1), _track(2)])
    assert isinstance(after.tracks, tuple) and [k.track_id for k in after.tracks] == [1, 2]


def test_state_is_frozen():
    with pytest.raises(FrozenInstanceError):
        TrackerState().next_id = 5  # type: ignore[misc]


@pytest.mark.parametrize("kw", [
    dict(area=0), dict(area=4),  # area doit valoir ys.size (3) et être > 0
    dict(age=-1), dict(coasted=-1),
    dict(ys=np.array([-1, 10, 11])), dict(xs=np.array([20, -2, 20])),
    dict(ys=np.array([9.5, 10.0, 11.0])),
    dict(v=(1.0,)), dict(v=(1.0, 2.0, 3.0)),
    dict(v=(float("nan"), 0.0)), dict(v=(0.0, float("inf"))),
    dict(mass=float("nan")), dict(mass=-1.0), dict(mass=float("inf")),
])
def test_track_rejects_invalid_fields(kw):
    with pytest.raises(ValueError):
        _track(**kw)


def test_track_accepts_float_valued_integer_pixel_arrays_only_if_integral():
    track = _track(ys=np.array([9.0, 10.0, 11.0]))
    assert track.ys.tolist() == [9, 10, 11]


@pytest.mark.parametrize("name", ["min_area_start", "min_area_keep", "max_coast", "max_gap_min",
                                  "young_frames", "coarse"])
@pytest.mark.parametrize("bad", [True, 2.5, "3"])
def test_config_rejects_non_integer_integer_parameters(name, bad):
    with pytest.raises(ValueError):
        TrackerConfig(**{name: bad})


@pytest.mark.parametrize("name", ["d_max", "w_pos", "max_cost", "f_min", "alpha", "min_peak_ratio"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_config_rejects_non_finite_floats(name, bad):
    with pytest.raises(ValueError):
        TrackerConfig(**{name: bad})
