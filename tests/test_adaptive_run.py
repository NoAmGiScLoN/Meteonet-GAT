"""Tests de src/adaptive_run.py : CLI de la session B (parser, selection, sorties incrementales, resume, resume)."""
import json
import logging
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

import adaptive_run as ar
import variable_source_fit as vsf
from residual_stats import STAT_COLUMNS

TINY_FLAGS = ["--n-w", "4", "--n-r", "32", "--lut-n", "32", "--lut-steps", "60"]


def fake_sample(spec: list[tuple[str, int]]) -> list[dict]:
    return [dict(bin=b, area=a, tag=f"t{i}") for i, (b, a) in enumerate(spec)]


def base_args(tmp_path, sample_path, *extra) -> list[str]:
    return ["--sample", str(sample_path), "--out", str(tmp_path / "out"), "--iters", "2", "--k-cap", "2",
            "--target", "0", "--device", "cpu", *TINY_FLAGS, *extra]


@pytest.fixture
def sample_path(tmp_path, make_cell):
    cells = [make_cell(i, size=20 + i) for i in range(4)]
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps(cells))
    return path


# ------------------------------------------------------------------ parser
@pytest.mark.unit
def test_parser_defaults():
    args = ar.build_parser().parse_args(["--sample", "s", "--out", "o"])
    assert args.target == ar.TARGET_DEFAULT
    assert args.k_cap == 128
    assert args.compression_frac == 0.5  # D24
    assert args.prune_frac == 0.01
    assert args.bins == [b[0] for b in vsf.BINS]
    assert args.iters is None and args.largest is None and args.max_per_bin is None
    assert args.resume is False


@pytest.mark.unit
def test_parser_compression_zero_means_off():
    args = ar.build_parser().parse_args(["--sample", "s", "--out", "o", "--compression-frac", "0"])
    assert args.compression_frac is None


@pytest.mark.unit
@pytest.mark.parametrize("bad", [["--k-cap", "100"], ["--target", "-1"], ["--compression-frac", "1.5"],
                                 ["--prune-frac", "1"], ["--iters", "0"], ["--largest", "0"],
                                 ["--max-per-bin", "0"], ["--bins", "zzz"], ["--target", "nan"]])
def test_parser_rejects_invalid(bad):
    with pytest.raises(SystemExit):
        ar.build_parser().parse_args(["--sample", "s", "--out", "o", *bad])


# ------------------------------------------------------------------ select_cells
@pytest.mark.unit
def test_select_cells_max_per_bin_keeps_ids_and_does_not_mutate():
    sample = fake_sample([("50-200", 60), ("200-1000", 300), ("50-200", 70), ("50-200", 80), ("200-1000", 400)])
    out = ar.select_cells(sample, ["50-200", "200-1000"], 2, None)
    assert [c["id"] for c in out] == [0, 1, 2, 4]
    assert all("id" not in c for c in sample)
    out[0]["area"] = -1
    assert sample[0]["area"] == 60


@pytest.mark.unit
def test_select_cells_filters_bins():
    sample = fake_sample([("50-200", 60), ("200-1000", 300)])
    assert [c["id"] for c in ar.select_cells(sample, ["200-1000"], None, None)] == [1]


@pytest.mark.unit
def test_select_cells_largest_in_one_bin():
    sample = fake_sample([("50-200", 60), ("200-1000", 300), ("200-1000", 500), ("200-1000", 400)])
    out = ar.select_cells(sample, [b[0] for b in vsf.BINS], None, 2)
    assert [c["id"] for c in out] == [2, 3]


@pytest.mark.unit
def test_select_cells_largest_mixed_bins_raises():
    sample = fake_sample([("50-200", 190), ("200-1000", 300)])
    with pytest.raises(ValueError, match="classe"):
        ar.select_cells(sample, [b[0] for b in vsf.BINS], None, 2)


# ------------------------------------------------------------------ summarize
def hand_df() -> pd.DataFrame:
    rows = []
    for i, (area, k, reason, nrmse) in enumerate([(100, 2, "target", 0.05), (200, 4, "cap", 0.3),
                                                  (400, 8, "compression", 0.2), (800, 16, "target", 0.08)]):
        rows.append(dict(id=i, bin="50-200", area=area, K=k, stop_reason=reason, nrmse=nrmse,
                         compression_ratio=7 * k / area, mass_bias=0.01 * i, peak_bias=-0.02 * i))
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_summarize_shares_sum_to_one_and_slope_positive():
    tables = ar.summarize(hand_df())
    assert set(tables) == {"stop", "k", "error"}
    stop = tables["stop"].iloc[0]
    assert stop["n"] == 4
    assert stop["share_target"] + stop["share_cap"] + stop["share_compression"] == pytest.approx(1.0)
    assert stop["share_target"] == 0.5
    k = tables["k"].iloc[0]
    assert k["K_median"] == 6.0
    assert k["slope_logK_logarea"] == pytest.approx(1.0)
    err = tables["error"].iloc[0]
    assert err["nrmse_median"] == pytest.approx(0.14)
    assert err["share_target_reached"] == 0.5
    assert err["mass_bias_median"] == pytest.approx(0.015)


@pytest.mark.unit
def test_summarize_single_cell_slope_is_nan():
    tables = ar.summarize(hand_df().iloc[:1])
    assert np.isnan(tables["k"].iloc[0]["slope_logK_logarea"])


# ------------------------------------------------------------------ main
@pytest.mark.integration
def test_main_writes_all_outputs(tmp_path, sample_path):
    ar.main(base_args(tmp_path, sample_path))
    out = tmp_path / "out"
    cells = pd.read_csv(out / "cells_50_200.csv")
    assert len(cells) == 4 and set(cells.id) == {0, 1, 2, 3}
    assert "stop_reason" in cells.columns
    assert set(STAT_COLUMNS) <= set(cells.columns)
    waves = pd.read_csv(out / "waves_50_200.csv")
    assert {"id", "wave", "K_active", "seconds"} <= set(waves.columns)
    sources = pickle.loads((out / "sources.pkl").read_bytes())  # nosec B301
    assert set(sources) == {0, 1, 2, 3}
    assert all(s["params"].ndim == 2 and s["params"].shape[1] == 8 for s in sources.values())
    mem = pd.read_csv(out / "memory.csv")
    assert list(mem.columns) == ["bin", "group", "n_cells", "peak_bytes", "seconds", "timestamp", "run"]
    assert mem.n_cells.sum() == 4 and mem.peak_bytes.isna().all()
    for name in ("stop", "k", "error"):
        assert len(pd.read_csv(out / f"summary_{name}.csv")) == 1


@pytest.mark.integration
def test_main_resume_skips_done_ids(tmp_path, sample_path, monkeypatch):
    ar.main(base_args(tmp_path, sample_path))
    calls = []
    real = ar.fit_adaptive
    monkeypatch.setattr(ar, "fit_adaptive", lambda *a, **k: calls.append(1) or real(*a, **k))
    ar.main(base_args(tmp_path, sample_path, "--resume"))
    assert calls == []
    assert len(pd.read_csv(tmp_path / "out" / "cells_50_200.csv")) == 4


@pytest.mark.integration
def test_main_resume_fits_only_missing_cells(tmp_path, sample_path, monkeypatch):
    ar.main(base_args(tmp_path, sample_path, "--max-per-bin", "2"))
    seen = []
    real = ar.fit_adaptive
    monkeypatch.setattr(ar, "fit_adaptive", lambda cells, *a, **k: seen.extend(c["id"] for c in cells)
                        or real(cells, *a, **k))
    ar.main(base_args(tmp_path, sample_path, "--resume"))
    assert seen[0] == 2  # amorcage sur la premiere cellule restante
    assert seen[1:] == [2, 3]  # puis seules les cellules manquantes
    cells = pd.read_csv(tmp_path / "out" / "cells_50_200.csv")
    assert sorted(cells.id) == [0, 1, 2, 3]
    assert set(pickle.loads((tmp_path / "out" / "sources.pkl").read_bytes())) == {0, 1, 2, 3}  # nosec B301


@pytest.mark.unit
def test_main_restores_device_on_error(tmp_path, sample_path, monkeypatch):
    sentinel = torch.device("meta")
    monkeypatch.setattr(vsf, "DEVICE", sentinel)

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ar, "fit_adaptive", boom)
    with pytest.raises(RuntimeError):
        ar.main(base_args(tmp_path, sample_path))
    assert vsf.DEVICE == sentinel


@pytest.mark.unit
def test_main_rejects_invalid_largest_mixed_bins(tmp_path, make_cell):
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0), make_cell(1, bin_name="200-1000")]))
    with pytest.raises(ValueError, match="classe"):
        ar.main(["--sample", str(path), "--out", str(tmp_path / "o"), "--largest", "2", "--device", "cpu",
                 *TINY_FLAGS])


@pytest.mark.unit
def test_seconds_per_wave_from_trace():
    trace = [dict(wave=0, seconds=1.0), dict(wave=0, seconds=1.0), dict(wave=1, seconds=0.5)]
    assert ar.seconds_per_wave(trace) == pytest.approx(1.25)
    assert np.isnan(ar.seconds_per_wave([]))


# ------------------------------------------------------------------ robustness (session B review)
OUT_NAME = "out"


def cell_ids(tmp_path) -> list[int]:
    return list(pd.read_csv(tmp_path / OUT_NAME / "cells_50_200.csv").id)


def source_files(tmp_path) -> set[int]:
    return {int(p.stem.rsplit("_", 1)[1]) for p in (tmp_path / OUT_NAME / "sources").glob("50_200_*.pkl")}


def crash_after_first_cell(monkeypatch):
    real = ar.fit_adaptive

    def crashing(cells, cfg, lut5, *a, on_cell_done=None, **k):
        if cfg.k_cap == 1:  # warm-up
            return real(cells, cfg, lut5, *a, **k)
        results, _, sources = real(cells, cfg, lut5, *a, **k)
        on_cell_done(results[0], sources[results[0]["id"]])
        raise RuntimeError("crash")
    monkeypatch.setattr(ar, "fit_adaptive", crashing)
    return real


@pytest.mark.integration
def test_crash_then_resume_every_row_has_a_source_and_no_duplicates(tmp_path, sample_path, monkeypatch):
    real = crash_after_first_cell(monkeypatch)
    with pytest.raises(RuntimeError, match="crash"):
        ar.main(base_args(tmp_path, sample_path))
    assert cell_ids(tmp_path) == [0] and source_files(tmp_path) == {0}  # per-cell persistence
    monkeypatch.setattr(ar, "fit_adaptive", real)
    ar.main(base_args(tmp_path, sample_path, "--resume"))
    assert sorted(cell_ids(tmp_path)) == [0, 1, 2, 3]
    assert source_files(tmp_path) == {0, 1, 2, 3}
    assert set(pickle.loads((tmp_path / OUT_NAME / "sources.pkl").read_bytes())) == {0, 1, 2, 3}  # nosec B301
    waves = pd.read_csv(tmp_path / OUT_NAME / "waves_50_200.csv")
    assert not waves.duplicated(["id", "wave"]).any()


@pytest.mark.integration
def test_resume_refits_cell_with_row_but_no_source_and_drops_stale_rows(tmp_path, sample_path, monkeypatch):
    ar.main(base_args(tmp_path, sample_path))
    (tmp_path / OUT_NAME / "sources" / "50_200_1.pkl").unlink()
    seen = []
    real = ar.fit_adaptive
    monkeypatch.setattr(ar, "fit_adaptive", lambda cells, *a, **k: seen.extend(c["id"] for c in cells)
                        or real(cells, *a, **k))
    ar.main(base_args(tmp_path, sample_path, "--resume"))
    assert seen == [1, 1]  # warm-up + refit, nothing else
    assert sorted(cell_ids(tmp_path)) == [0, 1, 2, 3]
    assert source_files(tmp_path) == {0, 1, 2, 3}
    waves = pd.read_csv(tmp_path / OUT_NAME / "waves_50_200.csv")
    assert not waves.duplicated(["id", "wave"]).any()


@pytest.mark.unit
def test_done_ids_ignores_torn_last_line(tmp_path):
    (tmp_path / "sources").mkdir()
    for i in (0, 1, 2):
        (tmp_path / "sources" / f"50_200_{i}.pkl").write_bytes(b"x")
    (tmp_path / "cells_50_200.csv").write_text("id,K\n0,2\n1,4\n2")
    assert ar.done_ids(tmp_path, "50-200") == {0, 1}


@pytest.mark.unit
def test_append_row_repairs_torn_tail_and_checks_header(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("id,K\n0,2\n1")
    ar._append_row(path, dict(id=5, K=7))
    assert path.read_text() == "id,K\n0,2\n5,7\n"
    with pytest.raises(ValueError, match="en-tete"):
        ar._append_row(path, dict(id=6, other=1))


@pytest.mark.unit
def test_atomic_write_leaves_no_tmp(tmp_path):
    ar._atomic_write_bytes(tmp_path / "f.bin", b"abc")
    assert (tmp_path / "f.bin").read_bytes() == b"abc"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]


@pytest.mark.integration
def test_existing_outputs_without_resume_error_unless_overwrite(tmp_path, sample_path):
    ar.main(base_args(tmp_path, sample_path))
    with pytest.raises(FileExistsError, match="--overwrite"):
        ar.main(base_args(tmp_path, sample_path))
    stale = tmp_path / OUT_NAME / "sources" / "50_200_99.pkl"
    stale.write_bytes(b"x")
    ar.main(base_args(tmp_path, sample_path, "--overwrite"))
    assert not stale.exists()
    assert sorted(cell_ids(tmp_path)) == [0, 1, 2, 3]


@pytest.mark.unit
def test_resume_and_overwrite_are_exclusive(sample_path):
    with pytest.raises(SystemExit):
        ar.build_parser().parse_args(["--sample", "s", "--out", "o", "--resume", "--overwrite"])


@pytest.mark.unit
def test_summaries_drop_duplicate_rows_keep_last(tmp_path):
    df = hand_df()
    dup = pd.concat([df, df.iloc[:1].assign(K=64)], ignore_index=True)
    dup.to_csv(tmp_path / "cells_50_200.csv", index=False)
    ar._write_summaries(tmp_path)
    assert pd.read_csv(tmp_path / "summary_k.csv").n.iloc[0] == 4


@pytest.mark.unit
def test_summarize_diverged_counts_and_nrmse_over_finite(caplog):
    df = hand_df()
    df.loc[1, ["stop_reason", "nrmse"]] = ["diverged", np.inf]
    df.loc[2, "nrmse"] = np.nan
    with caplog.at_level(logging.WARNING):
        tables = ar.summarize(df)
    assert "diverged" in ar.STOP_REASONS
    assert tables["stop"].iloc[0]["share_diverged"] == 0.25
    err = tables["error"].iloc[0]
    assert err["n_diverged"] == 1
    assert err["nrmse_median"] == pytest.approx(np.median([0.05, 0.08]))
    assert any("diverged" in r.message for r in caplog.records)


@pytest.mark.unit
def test_summarize_rejects_missing_bin_and_nonpositive_area():
    with pytest.raises(ValueError, match="bin"):
        ar.summarize(hand_df().drop(columns="bin"))
    with pytest.raises(ValueError, match="area"):
        ar.summarize(hand_df().assign(area=0))


@pytest.mark.integration
def test_run_config_written_and_resume_mismatch_refused(tmp_path, sample_path):
    ar.main(base_args(tmp_path, sample_path))
    cfg = json.loads((tmp_path / OUT_NAME / "run_config.json").read_text())
    assert cfg["k_cap"] == 2 and cfg["iters"] == 2 and cfg["arm"] == "diff7" and cfg["target"] == 0
    assert {"compression_frac", "prune_frac", "lut"} <= set(cfg)
    args = base_args(tmp_path, sample_path, "--resume")
    args[args.index("--k-cap") + 1] = "4"
    with pytest.raises(ValueError, match="run_config"):
        ar.main(args)


@pytest.mark.unit
def test_dedupe_bins_preserving_order(tmp_path, sample_path, monkeypatch):
    seen = []
    real = ar.fit_adaptive
    monkeypatch.setattr(ar, "fit_adaptive", lambda cells, *a, **k: seen.extend(c["id"] for c in cells)
                        or real(cells, *a, **k))
    ar.main(base_args(tmp_path, sample_path, "--bins", "50-200", "50-200"))
    assert seen == [0, 0, 1, 2, 3]
    assert ar.dedupe(["b", "a", "b"]) == ["b", "a"]


@pytest.mark.integration
def test_oom_splits_group_and_retries(tmp_path, sample_path, monkeypatch):
    class FakeOOM(RuntimeError):
        pass
    real = ar.fit_adaptive
    sizes = []

    def flaky(cells, cfg, *a, **k):
        if cfg.k_cap != 1:
            sizes.append(len(cells))
            if len(cells) > 1:
                raise FakeOOM("out of memory")
        return real(cells, cfg, *a, **k)
    monkeypatch.setattr(ar, "fit_adaptive", flaky)
    monkeypatch.setattr(ar, "OOM_ERRORS", (FakeOOM,))
    ar.main(base_args(tmp_path, sample_path))
    assert sorted(cell_ids(tmp_path)) == [0, 1, 2, 3]
    assert sizes == [4, 2, 1, 1, 2, 1, 1]
    mem = pd.read_csv(tmp_path / OUT_NAME / "memory.csv")
    assert list(mem.n_cells) == [1, 1, 1, 1]
    assert mem.groupby("run").group.apply(lambda g: g.is_unique).all()


@pytest.mark.unit
def test_oom_single_cell_is_reraised(tmp_path, sample_path, monkeypatch):
    class FakeOOM(RuntimeError):
        pass

    def always(cells, cfg, *a, **k):
        if cfg.k_cap == 1:
            return [], [], {}
        raise FakeOOM("oom")
    monkeypatch.setattr(ar, "fit_adaptive", always)
    monkeypatch.setattr(ar, "OOM_ERRORS", (FakeOOM,))
    with pytest.raises(FakeOOM):
        ar.main(base_args(tmp_path, sample_path))


@pytest.mark.unit
def test_cuda_unavailable_fails_before_loading_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="cuda"):
        ar.main(["--sample", str(tmp_path / "missing.pkl"), "--out", str(tmp_path / "o"), "--device", "cuda"])


@pytest.mark.unit
@pytest.mark.parametrize("bad", [["--w-min", "0"], ["--w-min", "5", "--w-max", "1"], ["--n-r", "0"], ["--w-max", "-1"]])
def test_lut_flags_validated(tmp_path, bad):
    with pytest.raises(SystemExit):
        ar.main(["--sample", str(tmp_path / "missing.pkl"), "--out", str(tmp_path / "o"), "--device", "cpu", *bad])


@pytest.mark.integration
def test_log_file_and_memory_run_counter_across_resumes(tmp_path, sample_path):
    ar.main(base_args(tmp_path, sample_path, "--max-per-bin", "2"))
    ar.main(base_args(tmp_path, sample_path, "--resume"))
    out = tmp_path / OUT_NAME
    assert (out / "adaptive_run.log").stat().st_size > 0
    mem = pd.read_csv(out / "memory.csv")
    assert list(mem.run) == [1, 2] and mem.timestamp.notna().all()


@pytest.mark.unit
def test_warm_up_once_per_bin(tmp_path, make_cell, monkeypatch):
    cells = [make_cell(0, size=20), make_cell(1, size=21, bin_name="200-1000")]
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps(cells))
    warm = []
    real = ar.warm_up
    monkeypatch.setattr(ar, "warm_up", lambda c, lut: warm.append(c["id"]) or real(c, lut))
    ar.main(base_args(tmp_path, path))
    assert warm == [0, 1]


@pytest.mark.unit
def test_warm_up_is_silent_labelled_and_restores_logger(make_cell, synthetic_lut, caplog):
    fit_logger = logging.getLogger("adaptive_fit")
    before = fit_logger.level
    with caplog.at_level(logging.INFO):
        ar.warm_up(make_cell(7, size=20), synthetic_lut[:1][None, None])
    messages = [r.getMessage() for r in caplog.records]
    assert not any("arret" in m for m in messages)
    assert any("echauffement GPU" in m and "cellule 7" in m and "resultat ignore" in m for m in messages)
    assert fit_logger.level == before


@pytest.mark.integration
def test_warm_up_results_never_written(tmp_path, make_cell):
    cells = [make_cell(0, size=20)]
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps(cells))
    ar.main(base_args(tmp_path, path))
    out = tmp_path / OUT_NAME
    for f in out.glob("cells_*.csv"):
        assert list(pd.read_csv(f).id) == [0]
    for f in out.glob("waves_*.csv"):
        assert set(pd.read_csv(f).id) <= {0}
