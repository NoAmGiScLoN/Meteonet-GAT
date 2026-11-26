"""CLI scripts/extract_cells.py : petit run sur de faux npz, reprise/écrasement, jours exclus, donnée réelle."""
import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import extract_cells as cli
from cell_delimitation import DATA_DIR
from cell_shard_dataset import ShardIndex
from data_split import SplitConfig, benchmark_days, excluded_days, split_of
from tracking_synth import disc_frame, write_npz_files

CFG = SplitConfig()


def _day_with_split(wanted: str, excluded: frozenset[date] = frozenset()) -> date:
    d = date(2016, 3, 1)
    while split_of(d, CFG, excluded) != wanted:
        d += timedelta(days=1)
    return d


def _fake_benchmark(tmp_path: Path, day: date = date(2016, 12, 20)) -> Path:
    path = tmp_path / "sources.pkl"
    path.write_bytes(pickle.dumps({0: {"date": f"{day} 12:00:00"}}))
    return path


def _fake_data(tmp_path: Path, day: date, n_frames: int = 24) -> Path:
    root = tmp_path / "data"
    t0 = datetime(day.year, day.month, day.day)
    dates = [t0 + timedelta(minutes=5 * i) for i in range(n_frames)]
    frames = []
    for i in range(n_frames):
        a = np.maximum(disc_frame((60, 80), (15, 15), 100), disc_frame((60, 80), (40, 50), 900))
        frames.append(np.where(a > 0, 30 + i, 0).astype(np.int16))
    write_npz_files(root, [(day.year, day.month, 1, dates, np.stack(frames))])
    return root


def _args(tmp_path: Path, day: date, root: Path, *extra: str, bench_day: date = date(2016, 12, 20)) -> list[str]:
    return ["--start", str(day), "--end", str(day), "--out", str(tmp_path / "out"), "--data-dir", str(root),
            "--benchmark", str(_fake_benchmark(tmp_path, bench_day)),
            "--quota-train", "50-200=5,200-1000=4,1000-5000=3,>=5000=2",
            "--quota-val", "50-200=1,200-1000=1,1000-5000=1,>=5000=1", "--shard-size", "3", *extra]


@pytest.mark.unit
def test_tiny_run_writes_shards_index_and_summary(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    out = tmp_path / "out"
    summary = pd.read_csv(out / "summary.csv")
    train = summary[(summary.split == "train") & (summary["bin"] != "total")].set_index("bin")
    # 24 trames : 50-200 et 200-1000 aux indices 0,6,12,18 -> 4 candidats (quotas 5 et 4)
    assert train.loc["50-200", "n_cells"] == 4 and train.loc["200-1000", "n_cells"] == 4
    index = pd.read_csv(out / "index.csv")
    assert len(index) == summary[summary["bin"] != "total"].n_cells.sum()
    assert all((out / s).exists() for s in index.shard.unique())
    assert (out / "run_config.json").exists()
    assert len(ShardIndex.from_dir(out)) == len(index)
    assert set(index.split) == {"train"}  # le jour est en train
    assert index.groupby("shard").size().max() <= 3  # shard-size respecté
    assert "raw_mb" in summary.columns


@pytest.mark.unit
def test_excluded_days_are_skipped(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root, bench_day=day)) == 0
    assert len(pd.read_csv(tmp_path / "out" / "index.csv")) == 0


@pytest.mark.unit
def test_test_days_are_skipped_since_test_has_no_quota(tmp_path: Path):
    day = _day_with_split("test")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    assert len(pd.read_csv(tmp_path / "out" / "index.csv")) == 0


@pytest.mark.unit
def test_resume_refuses_different_config_and_accepts_same(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    assert cli.main(_args(tmp_path, day, root, "--resume")) == 0
    with pytest.raises(ValueError, match="reprise refusée"):
        cli.main(_args(tmp_path, day, root, "--resume", "--salt", "autre"))


@pytest.mark.unit
def test_existing_output_requires_resume_or_overwrite(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    with pytest.raises(FileExistsError):
        cli.main(_args(tmp_path, day, root))
    assert cli.main(_args(tmp_path, day, root, "--overwrite", "--salt", "autre")) == 0
    assert json.loads((tmp_path / "out" / "run_config.json").read_text())["salt"] == "autre"


@pytest.mark.unit
def test_selection_is_deterministic_across_runs(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    cli.main(_args(tmp_path, day, root))
    first = pd.read_csv(tmp_path / "out" / "index.csv")
    cli.main(_args(tmp_path, day, root, "--overwrite"))
    pd.testing.assert_frame_equal(first, pd.read_csv(tmp_path / "out" / "index.csv"))


@pytest.mark.unit
@pytest.mark.parametrize("spec,expected", [("50-200=5", {"50-200": 5}), (">=5000=7,200-1000=1", {">=5000": 7, "200-1000": 1})])
def test_parse_quota_overrides(spec, expected):
    assert cli.parse_quota_spec(spec) == expected


@pytest.mark.unit
@pytest.mark.parametrize("spec", ["50-200", "zzz=3", "50-200=-1", "50-200=x"])
def test_parse_quota_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        cli.parse_quota_spec(spec)


@pytest.mark.integration
@pytest.mark.skipif(not (DATA_DIR / "NW_rainfall_2016" / "rainfall-NW-2016-02").exists(), reason="DATA absent")
def test_real_day_extraction_with_small_quotas(tmp_path: Path):
    day = date(2016, 2, 3)
    argv = ["--start", str(day), "--end", str(day), "--out", str(tmp_path / "out"),
            "--quota-train", "50-200=20,200-1000=20,1000-5000=20,>=5000=5",
            "--quota-val", "50-200=2,200-1000=2,1000-5000=2,>=5000=1", "--log-every", "200"]
    assert cli.main(argv) == 0
    index = pd.read_csv(tmp_path / "out" / "index.csv")
    assert 0 <= len(index) <= 100
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert {"split", "bin", "n_cells", "raw_mb"} <= set(summary.columns)


def _two_file_data(tmp_path: Path, day: date) -> Path:
    """Deux fichiers (k=1 : matinée, k=2 : après-midi) du même jour, cellules aux trois tailles."""
    root = tmp_path / "data"
    t0 = datetime(day.year, day.month, day.day)
    spec = []
    for k, first in ((1, 0), (2, 12)):
        dates = [t0 + timedelta(minutes=5 * (first + i)) for i in range(12)]
        frames = []
        for i in range(12):
            a = np.maximum(disc_frame((60, 80), (15, 15), 100), disc_frame((60, 80), (40, 50), 900))
            frames.append(np.where(a > 0, 30 + i + first, 0).astype(np.int16))
        spec.append((day.year, day.month, k, dates, np.stack(frames)))
    write_npz_files(root, spec)
    return root


def _records_by_shard(out: Path) -> dict:
    import cell_shards as cs
    return {p.name: [(r.t, r.local_id, r.raw.tobytes()) for r in cs.read_shard(p)]
            for p in sorted(out.glob("*.shard"))}


@pytest.mark.unit
def test_resume_after_interruption_equals_uninterrupted_run(tmp_path: Path, monkeypatch):
    # Arrange
    day = _day_with_split("train")
    root = _two_file_data(tmp_path, day)
    ref_args = _args(tmp_path, day, root)
    ref_args[ref_args.index("--out") + 1] = str(tmp_path / "ref")
    assert cli.main(ref_args) == 0
    real = cli.iter_frames

    def crashing(*a, **k):
        seen_second = 0
        for frame in real(*a, **k):
            if frame.source.name.endswith(".2.npz"):
                seen_second += 1
                if seen_second == 2:  # le checkpoint de fin du fichier 1 est déjà écrit
                    raise RuntimeError("panne simulée")
            yield frame

    monkeypatch.setattr(cli, "iter_frames", crashing)
    with pytest.raises(RuntimeError, match="panne"):
        cli.main(_args(tmp_path, day, root))
    out = tmp_path / "out"
    assert (out / "_partial" / "selector.pkl").exists() and not (out / "index.csv").exists()
    monkeypatch.setattr(cli, "iter_frames", real)
    # Act
    assert cli.main(_args(tmp_path, day, root, "--resume")) == 0
    # Assert
    assert _records_by_shard(out) == _records_by_shard(tmp_path / "ref")
    pd.testing.assert_frame_equal(pd.read_csv(out / "index.csv"), pd.read_csv(tmp_path / "ref" / "index.csv"))
    assert not (out / "_partial").exists()


@pytest.mark.unit
def test_resumed_run_processes_only_frames_after_checkpoint(tmp_path: Path, monkeypatch):
    day = _day_with_split("train")
    root = _two_file_data(tmp_path, day)
    real = cli.iter_frames

    def crashing(*a, **k):
        for frame in real(*a, **k):
            if frame.source.name.endswith(".2.npz") and frame.t.hour >= 1 and frame.t.minute >= 5:
                raise RuntimeError("panne simulée")
            yield frame

    monkeypatch.setattr(cli, "iter_frames", crashing)
    with pytest.raises(RuntimeError):
        cli.main(_args(tmp_path, day, root))
    calls = []
    monkeypatch.setattr(cli, "iter_frames", lambda start, end, *a, **k: (calls.append(start), real(start, end, *a, **k))[1])
    cli.main(_args(tmp_path, day, root, "--resume"))
    assert calls[0] == datetime(day.year, day.month, day.day, 0, 55, 0) + timedelta(minutes=5)


@pytest.mark.unit
def test_no_val_or_excluded_day_in_train_shards(tmp_path: Path):
    # Arrange : jours train, val, exclu et test dans un même fichier mensuel
    excluded_day = _day_with_split("train")
    excluded = excluded_days(benchmark_days([_fake_benchmark(tmp_path, excluded_day)]), CFG)
    train_day, val_day, test_day = (_day_with_split(s, excluded) for s in ("train", "val", "test"))
    days = sorted({excluded_day, train_day, val_day, test_day})
    assert len(days) == 4 and len({d.month for d in days}) >= 1
    root = tmp_path / "data"
    by_month: dict = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(d)
    spec = []
    for (y, m), month_days in by_month.items():
        dates, frames = [], []
        for d in month_days:
            for i in range(12):
                dates.append(datetime(d.year, d.month, d.day) + timedelta(minutes=5 * i))
                a = np.maximum(disc_frame((60, 80), (15, 15), 100), disc_frame((60, 80), (40, 50), 900))
                frames.append(np.where(a > 0, 30, 0).astype(np.int16))
        spec.append((y, m, 1, dates, np.stack(frames)))
    write_npz_files(root, spec)
    argv = ["--start", str(days[0]), "--end", str(days[-1]), "--out", str(tmp_path / "out"), "--data-dir", str(root),
            "--benchmark", str(_fake_benchmark(tmp_path, excluded_day)), "--quota-train", "50-200=50,200-1000=50,1000-5000=50,>=5000=50",
            "--quota-val", "50-200=50,200-1000=50,1000-5000=50,>=5000=50"]
    # Act
    assert cli.main(argv) == 0
    # Assert
    import cell_shards as cs
    seen = {"train": set(), "val": set()}
    for path in (tmp_path / "out").glob("*.shard"):
        for r in cs.read_shard(path):
            seen[r.split].add(r.t.date())
            assert path.name.startswith(r.split + "_")
    assert seen["train"] == {train_day} and seen["val"] == {val_day}


@pytest.mark.unit
def test_min_area_below_smallest_bin_is_refused(tmp_path: Path):
    day = _day_with_split("train")
    with pytest.raises(SystemExit):
        cli.main(_args(tmp_path, day, _fake_data(tmp_path, day), "--min-area", "10"))


@pytest.mark.unit
def test_max_ram_guard_raises_clear_error(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    with pytest.raises(cli.RamBudgetExceeded, match="max-ram-gb"):
        cli.main(_args(tmp_path, day, root, "--max-ram-gb", "0.000001"))


@pytest.mark.unit
def test_is_complete_requires_every_listed_shard_and_matching_counts(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    out = tmp_path / "out"
    assert cli._is_complete(out)
    victim = next(out.glob("*.shard"))
    import shutil
    shutil.rmtree(victim)
    assert not cli._is_complete(out)


@pytest.mark.unit
def test_is_complete_rejects_summary_count_mismatch(tmp_path: Path):
    day = _day_with_split("train")
    root = _fake_data(tmp_path, day)
    assert cli.main(_args(tmp_path, day, root)) == 0
    out = tmp_path / "out"
    summary = pd.read_csv(out / "summary.csv")
    summary.loc[0, "n_cells"] += 1
    summary.to_csv(out / "summary.csv", index=False)
    assert not cli._is_complete(out)


@pytest.mark.unit
def test_no_tmp_files_left_after_run(tmp_path: Path):
    day = _day_with_split("train")
    assert cli.main(_args(tmp_path, day, _fake_data(tmp_path, day))) == 0
    assert not [p for p in (tmp_path / "out").rglob("*") if p.name.endswith(".tmp")]
