"""Tests de track_run : sorties parquet par mois, reprise, continuité entre mois (fausses données)."""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import track_run

from cell_delimitation import DATA_DIR
from frame_stream import iter_frames
from track_run import assemble, main
from tracker_state import TrackerConfig
from tracker_v2 import track_sequence
from tracking_synth import DEFAULT_SHAPE, rain_frame, write_npz_files

CFG = TrackerConfig()
N_PER_MONTH = 12
JAN_START = datetime(2016, 1, 31, 23, 0)
KINDS = ("detections", "lineage", "events")


def _blobs(i: int) -> list[tuple]:
    return [(30, 10 + 2 * i, 6, 6, 60), (10 + i, 60, 5, 5, 60)]


def _build(root: Path, gap_min: int) -> tuple[Path, datetime, datetime]:
    """Deux mois de blobs mobiles ; `gap_min` = écart entre la dernière carte de janvier et la 1re de février."""
    jan = [JAN_START + timedelta(minutes=5 * i) for i in range(N_PER_MONTH)]
    feb0 = jan[-1] + timedelta(minutes=gap_min)
    feb = [feb0 + timedelta(minutes=5 * i) for i in range(N_PER_MONTH)]
    maps = [rain_frame(DEFAULT_SHAPE, _blobs(i)) for i in range(2 * N_PER_MONTH)]
    write_npz_files(root, [(2016, 1, 0, jan, maps[:N_PER_MONTH]),
                           (2016, 2, 0, feb, maps[N_PER_MONTH:])])
    return root, jan[0], feb[-1]


@pytest.fixture(scope="module", params=[5, 15, 25], ids=["contigu", "limite_15", "avec_ecart"])
def fake(request, tmp_path_factory):
    return _build(tmp_path_factory.mktemp("meteonet"), request.param)


def _run(fake, out: Path, *extra: str, end: str = "2016-02") -> int:
    root, _, _ = fake
    return main(["--start", "2016-01", "--end", end, "--out", str(out),
                 "--data-dir", str(root), *extra])


def _read(out: Path, kind: str) -> pd.DataFrame:
    parts = [pd.read_parquet(out / m / f"{kind}.parquet") for m in ("2016-01", "2016-02")]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def _continuous(fake) -> dict[str, pd.DataFrame]:
    root, start, end = fake
    frames = iter_frames(start, end, root, CFG.max_gap_min)
    _, outs = track_sequence(frames, CFG)
    return {k: assemble([getattr(o, k) for o in outs], k) for k in KINDS}


def _assert_same(out: Path, expected: dict[str, pd.DataFrame]) -> None:
    for kind in KINDS:
        got = _read(out, kind).reset_index(drop=True)
        want = expected[kind][expected[kind].columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want)


def test_month_by_month_equals_continuous_run(fake, tmp_path):
    expected = _continuous(fake)
    assert main(["--start", "2016-01", "--end", "2016-02", "--out", str(tmp_path),
                 "--data-dir", str(fake[0])]) == 0
    _assert_same(tmp_path, expected)
    assert (expected["detections"].track_id > 0).any()


def test_resume_after_first_month_equals_continuous(fake, tmp_path):
    expected = _continuous(fake)
    _run(fake, tmp_path, end="2016-01")
    assert not (tmp_path / "2016-02").exists()
    _run(fake, tmp_path, "--resume")
    _assert_same(tmp_path, expected)


def test_done_month_is_skipped_on_resume(fake, tmp_path):
    _run(fake, tmp_path)
    target = tmp_path / "2016-01" / "detections.parquet"
    before = target.stat().st_mtime_ns
    _run(fake, tmp_path, "--resume")
    assert target.stat().st_mtime_ns == before
    assert (tmp_path / "2016-01" / "DONE").exists()


def test_config_mismatch_refuses_resume(fake, tmp_path):
    _run(fake, tmp_path, end="2016-01")
    with pytest.raises(ValueError, match="run_config"):
        _run(fake, tmp_path, "--resume", "--min-area-start", "80")


def test_existing_output_without_flag_is_refused(fake, tmp_path):
    _run(fake, tmp_path, end="2016-01")
    with pytest.raises(FileExistsError):
        _run(fake, tmp_path, end="2016-01")


def test_parquet_round_trip_keeps_dtypes(fake, tmp_path):
    _run(fake, tmp_path)
    det = _read(tmp_path, "detections")
    ev = _read(tmp_path, "events")
    assert det.track_id.dtype.kind == "i" and ev.track_id.dtype.kind == "i"
    assert det.truncated.dtype == bool and det.velocity_reset.dtype == bool
    assert det.border_flag.dtype == bool
    assert str(det.timestamp.dtype).startswith("datetime64")
    assert str(ev.t.dtype).startswith("datetime64")
    lin = pd.read_parquet(tmp_path / "2016-01" / "lineage.parquet")
    assert lin.is_main.dtype == bool and lin.child_track.dtype.kind == "i"


def test_no_temporary_file_is_left(fake, tmp_path):
    _run(fake, tmp_path)
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_overwrite_clears_previous_output(fake, tmp_path):
    _run(fake, tmp_path)
    stale = tmp_path / "2016-02" / "stale.txt"
    stale.write_text("x")
    _run(fake, tmp_path, "--overwrite", end="2016-01")
    assert not stale.exists() and not (tmp_path / "2016-02").exists()
    assert (tmp_path / "2016-01" / "DONE").exists()


def test_track_ids_are_global_and_unique_per_timestamp(fake, tmp_path):
    _run(fake, tmp_path)
    det = _read(tmp_path, "detections")
    tracked = det[det.track_id > 0]
    assert not tracked.duplicated(["timestamp", "track_id"]).any()
    feb = pd.read_parquet(tmp_path / "2016-02" / "detections.parquet")
    jan = pd.read_parquet(tmp_path / "2016-01" / "detections.parquet")
    new_in_feb = set(feb.track_id[feb.track_id > 0]) - set(jan.track_id)
    assert all(tid > jan.track_id.max() for tid in new_in_feb)


def test_month_summary_and_state_files(fake, tmp_path):
    _run(fake, tmp_path, end="2016-01")
    summary = pd.read_json(tmp_path / "2016-01" / "month_summary.json", typ="series")
    assert summary["frames"] == N_PER_MONTH
    assert summary["tracked_cells"] > 0
    assert (tmp_path / "2016-01" / "state_end.pkl").exists()
    assert (tmp_path / "run_config.json").exists()


def test_invalid_range_is_refused(fake, tmp_path):
    with pytest.raises(ValueError):
        main(["--start", "2016-02", "--end", "2016-01", "--out", str(tmp_path),
              "--data-dir", str(fake[0])])


@pytest.mark.skipif(not (DATA_DIR / "NW_rainfall_2016").exists(), reason="DATA/ absent")
def test_real_data_six_hours(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    code = main(["--start", "2016-01", "--end", "2016-01", "--out", str(tmp_path),
                 "--start-time", "2016-01-01T00:00", "--end-time", "2016-01-01T06:00"])
    month = tmp_path / "2016-01"
    assert code == 0 and (month / "DONE").exists()
    assert (month / "detections.parquet").exists() and (month / "events.parquet").exists()
    assert "cartes/s" in caplog.text
    assert os.path.getsize(month / "month_summary.json") > 0


def _schema(out: Path, month: str, kind: str):
    return pq.read_schema(out / month / f"{kind}.parquet")


def test_resume_survives_git_sha_change(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(track_run, "git_sha", lambda: "aaaa")
    _run(fake, tmp_path, end="2016-01")
    monkeypatch.setattr(track_run, "git_sha", lambda: "bbbb")
    _run(fake, tmp_path, "--resume")
    config = json.loads((tmp_path / "run_config.json").read_text())
    summary = json.loads((tmp_path / "2016-02" / "month_summary.json").read_text())
    assert config["provenance"]["git_sha"] == "aaaa" and summary["git_sha"] == "bbbb"
    assert "data_dir" in config["provenance"]


def test_state_schema_change_refuses_resume(fake, tmp_path, monkeypatch):
    _run(fake, tmp_path, end="2016-01")
    monkeypatch.setattr(track_run, "STATE_SCHEMA", track_run.STATE_SCHEMA + 1)
    with pytest.raises(ValueError, match="run_config"):
        _run(fake, tmp_path, "--resume")


def test_resume_without_run_config_is_refused_when_months_are_done(fake, tmp_path):
    _run(fake, tmp_path, end="2016-01")
    (tmp_path / "run_config.json").unlink()
    with pytest.raises(ValueError, match="run_config"):
        _run(fake, tmp_path, "--resume")


def test_dry_month_has_same_schema_and_dtypes(fake, tmp_path, caplog):
    _run(fake, tmp_path, end="2016-03")  # mars : aucun fichier
    dry = tmp_path / "2016-03"
    assert (dry / "DONE").exists()
    assert json.loads((dry / "month_summary.json").read_text())["frames"] == 0
    for kind in KINDS:
        assert _schema(tmp_path, "2016-03", kind).equals(_schema(tmp_path, "2016-01", kind))
    det = pd.concat([pd.read_parquet(tmp_path / m / "detections.parquet")
                     for m in ("2016-01", "2016-03")])
    assert det.track_id.dtype == "int64" and det.truncated.dtype == bool
    assert det.v_src.dtype == pd.read_parquet(dry / "detections.parquet").v_src.dtype


def test_missing_middle_month_does_not_abort_and_ids_stay_global(tmp_path):
    jan = [JAN_START + timedelta(minutes=5 * i) for i in range(N_PER_MONTH)]
    mar = [datetime(2016, 3, 1) + timedelta(minutes=5 * i) for i in range(N_PER_MONTH)]
    maps = [rain_frame(DEFAULT_SHAPE, _blobs(i)) for i in range(N_PER_MONTH)]
    write_npz_files(tmp_path / "d", [(2016, 1, 0, jan, maps), (2016, 3, 0, mar, maps)])
    out = tmp_path / "out"
    main(["--start", "2016-01", "--end", "2016-03", "--out", str(out),
          "--data-dir", str(tmp_path / "d")])
    assert (out / "2016-02" / "DONE").exists()
    jan_ids = pd.read_parquet(out / "2016-01" / "detections.parquet").track_id
    mar_det = pd.read_parquet(out / "2016-03" / "detections.parquet")
    assert mar_det.track_id.max() > jan_ids.max()
    ev = pd.read_parquet(out / "2016-03" / "events.parquet")
    assert len(mar_det) > 0 and len(ev) > 0


def test_corrupt_state_of_last_done_month_is_redone(fake, tmp_path):
    expected = _continuous(fake)
    _run(fake, tmp_path)
    pkl = tmp_path / "2016-02" / "state_end.pkl"
    pkl.write_bytes(pkl.read_bytes()[:20])
    _run(fake, tmp_path, "--resume", end="2016-03")
    _assert_same(tmp_path, expected)
    assert track_run.load_state(tmp_path / "2016-02")[0].t is not None


def test_unreadable_summary_of_last_done_month_is_redone(fake, tmp_path):
    _run(fake, tmp_path)
    (tmp_path / "2016-02" / "month_summary.json").write_text("{")
    _run(fake, tmp_path, "--resume")
    summary = json.loads((tmp_path / "2016-02" / "month_summary.json").read_text())
    assert summary["frames"] == N_PER_MONTH


def test_atomic_write_syncs_file_before_replace(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    track_run._atomic_write(tmp_path / "x.bin", b"abc")
    assert (tmp_path / "x.bin").read_bytes() == b"abc" and len(calls) >= 1
