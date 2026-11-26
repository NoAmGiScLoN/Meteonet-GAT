"""Tests du flux de cartes (frame_stream) : continuité, pas de temps, ouverture des fichiers."""
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from cell_delimitation import DATA_DIR
from frame_stream import Frame, iter_frames, list_files
from tracking_synth import make_times, write_npz_files

SHAPE = (4, 5)
T = datetime(2016, 1, 1, 12, 0)


def _data(n: int, offset: int = 0) -> np.ndarray:
    """Cartes identifiables : la carte i vaut partout offset + i."""
    return np.stack([np.full(SHAPE, offset + i, dtype=np.int16) for i in range(n)])


def _file(year, month, k, times, offset=0):
    return (year, month, k, times, _data(len(times), offset))


class _Spy:
    """Enregistre ouvertures, chargements de `data` et fermetures de fichiers npz."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple[str, str]] = []
        self.max_open = 0
        self._open = 0
        real_load = np.load

        spy = self

        class Proxy:
            def __init__(self, path: Path) -> None:
                self.name = path.name
                self.f = real_load(path, allow_pickle=True)
                spy._open += 1
                spy.max_open = max(spy.max_open, spy._open)
                spy.events.append(("open", self.name))

            def __enter__(self) -> "Proxy":
                return self

            def __exit__(self, *exc: object) -> None:
                self.f.close()
                spy._open -= 1
                spy.events.append(("close", self.name))

            def __getitem__(self, key: str) -> np.ndarray:
                if key == "data":
                    spy.events.append(("data", self.name))
                return self.f[key]

        monkeypatch.setattr(np, "load", lambda path, **kw: Proxy(path))

    @property
    def open_count(self) -> int:
        """Nombre de fichiers actuellement ouverts."""
        return self._open

    def loaded(self) -> list[str]:
        return [name for kind, name in self.events if kind == "data"]

    def opened(self) -> list[str]:
        return [name for kind, name in self.events if kind == "open"]


@pytest.mark.unit
def test_continuity_across_file_boundary(tmp_path):
    t1 = make_times(3, T)
    t2 = make_times(3, T + timedelta(minutes=15))
    write_npz_files(tmp_path, [_file(2016, 1, 1, t1), _file(2016, 1, 2, t2, offset=10)])

    frames = list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))

    assert [f.t for f in frames] == t1 + t2
    assert [f.dt_steps for f in frames] == [0, 1, 1, 1, 1, 1]
    assert frames[2].R[0, 0] == 2 and frames[3].R[0, 0] == 10


@pytest.mark.unit
def test_ten_minute_step_gives_two_steps_without_break(tmp_path):
    times = make_times(3, T, gap_after=0, gap_min=10)
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])

    frames = list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))

    assert [f.dt_steps for f in frames] == [0, 2, 1]
    assert [f.gap_break for f in frames] == [False, False, False]


@pytest.mark.unit
def test_twenty_minute_step_gives_four_steps_with_break(tmp_path):
    times = make_times(3, T, gap_after=0, gap_min=20)
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])

    frames = list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))

    assert [f.dt_steps for f in frames] == [0, 4, 1]
    assert [f.gap_break for f in frames] == [False, True, False]


@pytest.mark.unit
def test_custom_max_gap_changes_break_flag(tmp_path):
    times = make_times(2, T, gap_after=0, gap_min=10)
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])

    frames = list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path, max_gap_min=5))

    assert frames[1].gap_break is True


@pytest.mark.unit
def test_start_end_filtering_inside_a_file(tmp_path):
    times = make_times(8, T)
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])

    frames = list(iter_frames(times[2], times[5], data_dir=tmp_path))

    assert [f.t for f in frames] == times[2:6]
    assert frames[0].dt_steps == 0 and frames[0].gap_break is False
    assert [int(f.R[0, 0]) for f in frames] == [2, 3, 4, 5]


@pytest.mark.unit
def test_files_outside_range_are_not_opened(tmp_path, monkeypatch):
    t_jan = make_times(3, datetime(2016, 1, 1, 0, 0))
    t_feb1 = make_times(3, datetime(2016, 2, 1, 0, 0))
    t_feb2 = make_times(3, datetime(2016, 2, 10, 0, 0))
    t_mar = make_times(3, datetime(2016, 3, 1, 0, 0))
    write_npz_files(tmp_path, [
        _file(2016, 1, 1, t_jan), _file(2016, 2, 1, t_feb1),
        _file(2016, 2, 2, t_feb2), _file(2016, 3, 1, t_mar)])
    spy = _Spy(monkeypatch)

    frames = list(iter_frames(t_feb1[0], t_feb1[-1], data_dir=tmp_path))

    assert len(frames) == 3
    assert "rainfall_NW_2016_01.1.npz" not in spy.opened()
    assert "rainfall_NW_2016_03.1.npz" not in spy.opened()
    assert spy.loaded() == ["rainfall_NW_2016_02.1.npz"]  # le .2 est postérieur à `end`


@pytest.mark.unit
def test_files_are_loaded_once_each_and_one_at_a_time(tmp_path, monkeypatch):
    t1 = make_times(3, T)
    t2 = make_times(3, T + timedelta(minutes=15))
    t3 = make_times(3, T + timedelta(minutes=30))
    write_npz_files(tmp_path, [_file(2016, 1, 1, t1), _file(2016, 1, 2, t2),
                               _file(2016, 1, 3, t3)])
    spy = _Spy(monkeypatch)

    frames = list(iter_frames(T, T + timedelta(hours=2), data_dir=tmp_path))

    assert len(frames) == 9
    assert spy.loaded() == [f"rainfall_NW_2016_01.{k}.npz" for k in (1, 2, 3)]
    assert spy.max_open == 1
    assert spy.open_count == 0


@pytest.mark.unit
def test_list_files_sorted_and_limited_to_overlapping_months(tmp_path):
    write_npz_files(tmp_path, [
        _file(2016, 2, 1, make_times(1, datetime(2016, 2, 1))),
        _file(2015, 12, 3, make_times(1, datetime(2015, 12, 25))),
        _file(2016, 1, 2, make_times(1, datetime(2016, 1, 11))),
        _file(2016, 1, 1, make_times(1, datetime(2016, 1, 1))),
        _file(2016, 3, 1, make_times(1, datetime(2016, 3, 1)))])

    paths = list_files(tmp_path, datetime(2015, 12, 31), datetime(2016, 2, 5))

    assert [p.name for p in paths] == [
        "rainfall_NW_2015_12.3.npz", "rainfall_NW_2016_01.1.npz",
        "rainfall_NW_2016_01.2.npz", "rainfall_NW_2016_02.1.npz"]


@pytest.mark.unit
def test_start_after_end_raises(tmp_path):
    write_npz_files(tmp_path, [_file(2016, 1, 1, make_times(3, T))])
    with pytest.raises(ValueError, match="début"):
        list(iter_frames(T + timedelta(hours=1), T, data_dir=tmp_path))


@pytest.mark.unit
def test_no_file_covering_range_raises_french_message(tmp_path):
    write_npz_files(tmp_path, [_file(2016, 1, 1, make_times(3, T))])
    with pytest.raises(FileNotFoundError, match="Aucun fichier"):
        list(iter_frames(datetime(2017, 5, 1), datetime(2017, 5, 2), data_dir=tmp_path))


@pytest.mark.unit
def test_validation_errors_are_raised_without_iterating(tmp_path):
    write_npz_files(tmp_path, [_file(2016, 1, 1, make_times(3, T))])
    with pytest.raises(ValueError, match="début"):
        iter_frames(T + timedelta(hours=1), T, data_dir=tmp_path)  # pas de list()
    with pytest.raises(FileNotFoundError, match="Aucun fichier"):
        iter_frames(datetime(2017, 5, 1), datetime(2017, 5, 2), data_dir=tmp_path)


@pytest.mark.unit
def test_data_and_dates_length_mismatch_raises(tmp_path):
    write_npz_files(tmp_path, [(2016, 1, 1, make_times(3, T), _data(2))])
    with pytest.raises(ValueError, match="longueur"):
        list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))


@pytest.mark.unit
def test_files_exist_but_no_frame_in_range_raises(tmp_path):
    write_npz_files(tmp_path, [_file(2016, 1, 1, make_times(3, T))])
    with pytest.raises(ValueError, match="Aucune carte"):
        list(iter_frames(T + timedelta(hours=5), T + timedelta(hours=6), data_dir=tmp_path))


@pytest.mark.unit
def test_missing_middle_file_uses_last_yielded_frame(tmp_path):
    t1 = make_times(2, T)  # 12:00, 12:05
    t3 = make_times(2, T + timedelta(minutes=45))  # 12:45, 12:50 (le .2 manque)
    write_npz_files(tmp_path, [_file(2016, 1, 1, t1), _file(2016, 1, 3, t3)])

    frames = list(iter_frames(T, T + timedelta(hours=2), data_dir=tmp_path))

    assert [f.dt_steps for f in frames] == [0, 1, 8, 1]  # 12:05 -> 12:45 = 40 min
    assert [f.gap_break for f in frames] == [False, False, True, False]


@pytest.mark.unit
def test_non_multiple_step_raises(tmp_path):
    times = [T, T + timedelta(minutes=7)]
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])
    with pytest.raises(ValueError, match="multiple"):
        list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))


@pytest.mark.unit
def test_duplicated_timestamp_raises(tmp_path):
    times = [T, T + timedelta(minutes=5), T + timedelta(minutes=5)]
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])
    with pytest.raises(ValueError, match="croissant|dupliqu"):
        list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))


@pytest.mark.unit
def test_backwards_timestamp_raises(tmp_path):
    times = [T + timedelta(minutes=5), T]
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])
    with pytest.raises(ValueError, match="croissant|dupliqu"):
        list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))


@pytest.mark.unit
def test_frame_is_frozen_and_pixels_are_isolated(tmp_path):
    write_npz_files(tmp_path, [_file(2016, 1, 1, make_times(3, T))])

    frames = list(iter_frames(T, T + timedelta(hours=1), data_dir=tmp_path))

    with pytest.raises(FrozenInstanceError):
        frames[0].dt_steps = 5  # type: ignore[misc]
    assert isinstance(frames[0], Frame) and frames[0].R.dtype == np.int16
    assert not frames[0].R.flags.writeable
    with pytest.raises(ValueError):
        frames[0].R[0, 0] = 99
    assert frames[1].R[0, 0] == 1


@pytest.mark.integration
def test_real_data_crosses_january_file_boundary():
    if not DATA_DIR.exists():
        pytest.skip("DATA/ absent")
    start, end = datetime(2016, 1, 10, 23, 50), datetime(2016, 1, 11, 0, 10)

    frames = list(iter_frames(start, end))

    assert len(frames) == 5
    assert [f.dt_steps for f in frames] == [0, 1, 1, 1, 1]
    assert frames[0].t == start and frames[-1].t == end


@pytest.mark.unit
def test_off_grid_first_frame_is_rejected_before_loading_data(tmp_path):
    # Arrange : fichier sans clé `data` : tout chargement de data échouerait en KeyError
    month_dir = tmp_path / "NW_rainfall_2016" / "rainfall-NW-2016-01"
    month_dir.mkdir(parents=True)
    dates = np.empty(2, dtype=object)
    dates[:] = [datetime(2016, 1, 1, 0, 3), datetime(2016, 1, 1, 0, 8)]
    np.savez(month_dir / "rainfall_NW_2016_01.1.npz", dates=dates)
    # Act / Assert
    with pytest.raises(ValueError, match="grille"):
        list(iter_frames(datetime(2016, 1, 1), datetime(2016, 1, 2), data_dir=tmp_path))


@pytest.mark.unit
def test_frames_carry_their_source_file(tmp_path):
    times = make_times(2)
    write_npz_files(tmp_path, [_file(2016, 1, 1, times)])
    frames = list(iter_frames(times[0], times[-1], data_dir=tmp_path))
    assert all(f.source is not None and f.source.name == "rainfall_NW_2016_01.1.npz" for f in frames)
