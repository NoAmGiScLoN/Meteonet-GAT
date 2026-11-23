"""Tests du découpage par jours (train/val/test) sans fuite vers le benchmark E5."""
import pickle
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from data_split import (SPLITS, SplitConfig, benchmark_days, default_benchmark_paths,
                        excluded_days, split_of, split_table)

ROOT = Path(__file__).resolve().parent.parent
NO_EXCL: frozenset[date] = frozenset()
PROPORTION_TOL = 0.015
GOLDEN_DAYS = (date(2016, 1, 1), date(2016, 6, 15), date(2017, 3, 3), date(2018, 11, 30), date(2016, 2, 15))
GOLDEN: dict[str, str] = {
    '2016-01-01': 'train', '2016-06-15': 'test', '2017-03-03': 'train',
    '2018-11-30': 'train', '2016-02-15': 'val',
}


def _days(start: date = date(2016, 1, 1), end: date = date(2018, 12, 31)) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _write(path: Path, obj: object) -> Path:
    path.write_bytes(pickle.dumps(obj))
    return path


@pytest.mark.unit
def test_split_is_deterministic() -> None:
    cfg = SplitConfig()
    a = [split_of(d, cfg, NO_EXCL) for d in _days()]
    b = [split_of(d, cfg, NO_EXCL) for d in _days()]
    assert a == b


@pytest.mark.unit
def test_salt_changes_assignments() -> None:
    a = [split_of(d, SplitConfig(), NO_EXCL) for d in _days()]
    b = [split_of(d, SplitConfig(salt="autre"), NO_EXCL) for d in _days()]
    assert a != b


@pytest.mark.unit
def test_proportions_close_to_80_10_10() -> None:
    cfg = SplitConfig()
    excl = excluded_days(frozenset({date(2016, 1, 9), date(2017, 4, 26), date(2017, 9, 15)}), cfg)
    counts = Counter(s for d in _days() if (s := split_of(d, cfg, excl)) != "excluded")
    total = sum(counts.values())
    assert abs(counts["train"] / total - 0.8) < PROPORTION_TOL
    assert abs(counts["val"] / total - 0.1) < PROPORTION_TOL
    assert abs(counts["test"] / total - 0.1) < PROPORTION_TOL


@pytest.mark.unit
def test_every_split_covers_every_season_and_year() -> None:
    table = split_table(_days(), SplitConfig(), NO_EXCL)
    for split in SPLITS:
        sub = table[table["split"] == split]
        assert set(sub["season"]) == {"DJF", "MAM", "JJA", "SON"}
        assert set(sub["year"]) == {2016, 2017, 2018}


@pytest.mark.unit
def test_benchmark_days_and_margin_are_excluded() -> None:
    bench = frozenset({date(2016, 1, 9)})
    excl = excluded_days(bench, SplitConfig())
    assert excl == {date(2016, 1, 8), date(2016, 1, 9), date(2016, 1, 10)}
    for d in excl:
        assert split_of(d, SplitConfig(), excl) == "excluded"
    assert split_of(date(2016, 1, 11), SplitConfig(), excl) != "excluded"


@pytest.mark.unit
def test_zero_margin_keeps_only_benchmark_days() -> None:
    bench = frozenset({date(2016, 1, 9)})
    assert excluded_days(bench, SplitConfig(exclude_margin_days=0)) == bench


@pytest.mark.unit
@pytest.mark.parametrize("cfg", [
    dict(val_frac=0.0), dict(test_frac=-0.1), dict(val_frac=0.5, test_frac=0.5),
    dict(exclude_margin_days=-1), dict(val_frac=float('nan')), dict(test_frac=float('inf')),
    dict(block_days=0),
])
def test_invalid_config_raises(cfg: dict) -> None:
    with pytest.raises(ValueError):
        SplitConfig(**cfg)


@pytest.mark.unit
def test_excluded_days_empty_benchmark_raises() -> None:
    with pytest.raises(ValueError):
        excluded_days(frozenset(), SplitConfig())


@pytest.mark.unit
def test_benchmark_days_reads_pickle(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.pkl", {1: {"date": "2016-01-09 22:00:00"}, 2: {"date": "2016-01-10 01:00:00"}})
    assert benchmark_days([p]) == {date(2016, 1, 9), date(2016, 1, 10)}


@pytest.mark.unit
@pytest.mark.parametrize("obj", [[1, 2], {1: "pas un dict"}, {1: {"tag": "x"}}, {1: {"date": "pas une date"}}])
def test_malformed_pickle_raises(tmp_path: Path, obj: object) -> None:
    with pytest.raises(ValueError):
        benchmark_days([_write(tmp_path / "bad.pkl", obj)])


@pytest.mark.unit
def test_empty_paths_raises() -> None:
    with pytest.raises(ValueError):
        benchmark_days([])


@pytest.mark.unit
def test_empty_result_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        benchmark_days([_write(tmp_path / "vide.pkl", {})])


@pytest.mark.unit
@pytest.mark.parametrize("stamp", ["2016-01-09T22:00:00+00:00", "2016-01-09 22:00:00Z"])
def test_timezone_aware_dates_raise(tmp_path: Path, stamp: str) -> None:
    with pytest.raises(ValueError):
        benchmark_days([_write(tmp_path / "tz.pkl", {1: {"date": stamp}})])


@pytest.mark.unit
def test_blocks_share_one_split() -> None:
    cfg = SplitConfig()
    for start in (date(2016, 1, 1), date(2016, 12, 29), date(2017, 6, 1)):
        base = (start - date(2016, 1, 1)).days // 3 * 3
        first = date(2016, 1, 1) + timedelta(days=base)
        block = {split_of(first + timedelta(days=k), cfg, NO_EXCL) for k in range(3)}
        assert len(block) == 1


@pytest.mark.unit
def test_each_year_season_stratum_has_val_and_test_blocks() -> None:
    table = split_table(_days(), SplitConfig(), NO_EXCL)
    first = date(2016, 1, 1)
    table["block"] = [(d - first).days // 3 for d in table["day"]]
    blocks = table.groupby("block").agg(split=("split", "first"), season=("season", "first"), year=("year", "first"))
    for _, grp in blocks.groupby(["year", "season"]):
        assert {"val", "test"} <= set(grp["split"])


@pytest.mark.unit
@pytest.mark.parametrize("day", [date(2015, 12, 31), date(2019, 1, 1)])
def test_day_outside_range_raises(day: date) -> None:
    with pytest.raises(ValueError):
        split_of(day, SplitConfig(), NO_EXCL)


@pytest.mark.unit
def test_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        SplitConfig(start=date(2018, 1, 1), end=date(2016, 1, 1))


@pytest.mark.unit
def test_block_days_one_gives_per_day_assignment() -> None:
    cfg = SplitConfig(block_days=1)
    assert len({split_of(d, cfg, NO_EXCL) for d in _days(date(2016, 1, 1), date(2016, 3, 31))}) == 3


@pytest.mark.unit
def test_golden_assignments_are_stable() -> None:
    cfg = SplitConfig()
    got = {d.isoformat(): split_of(d, cfg, NO_EXCL) for d in GOLDEN_DAYS}
    assert got == GOLDEN


@pytest.mark.unit
def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        benchmark_days([tmp_path / "absent.pkl"])


@pytest.mark.integration
def test_real_e5_sources_are_all_excluded() -> None:
    paths = default_benchmark_paths(ROOT)
    if not all(p.exists() for p in paths):
        pytest.skip("sources.pkl E5 absents")
    days = benchmark_days(paths)
    cfg = SplitConfig()
    excl = excluded_days(days, cfg)
    assert days and days <= excl
    assert all(split_of(d, cfg, excl) == "excluded" for d in excl)
    for d in days:
        for k in (-1, 1):
            assert d + timedelta(days=k) in excl


@pytest.mark.unit
@pytest.mark.parametrize("val_frac,test_frac,n,expected", [
    (0.1, 0.1, 3, (1, 1)),      # n = 3 : au moins 1 val, 1 test, 1 train
    (0.45, 0.45, 3, (1, 1)),
    (0.45, 0.45, 4, (2, 1)),    # n_val + n_test = 4 > n - 1 : test réduit, mais >= 1
    (0.49, 0.49, 5, (2, 2)),
    (0.5, 0.25, 10, (5, 3)),    # 2.5 -> 3 (round() donnerait 2 : arrondi au supérieur explicite)
    (0.1, 0.1, 2, (0, 0)),      # n < 3 : pas de minimum imposé
    (0.4, 0.4, 2, (1, 1)),
    (0.1, 0.1, 0, (0, 0)),
])
def test_stratum_quotas_never_exceed_the_stratum_and_round_half_up(val_frac, test_frac, n, expected):
    import data_split as ds
    cfg = ds.SplitConfig(val_frac=val_frac, test_frac=test_frac)
    n_val, n_test = ds._stratum_quotas(cfg, n)
    assert (n_val, n_test) == expected
    assert n_val + n_test <= (n - 1 if n >= 3 else n)


@pytest.mark.unit
def test_block_splits_cache_is_immutable():
    import data_split as ds
    table = ds._block_splits(ds.SplitConfig())
    with pytest.raises(TypeError):
        table[0] = "val"  # type: ignore[index]
