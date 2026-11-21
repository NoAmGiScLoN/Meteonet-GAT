"""Tests de src/convergence_check.py : sélection des cellules, convergence en itérations Adam."""
import copy
import pickle

import torch

import numpy as np
import pandas as pd
import pytest

import convergence_check as cc

TINY_FLAGS = ["--n-w", "4", "--n-r", "32", "--lut-n", "32", "--lut-steps", "60"]
TINY_SPEC_KW = dict(n_w=4, n_r=32, n=32, steps=60)


def sample_of(counts: dict[str, int]) -> list[dict]:
    cells, i = [], 0
    for name, n in counts.items():
        for _ in range(n):
            cells.append(dict(bin=name, area=100 + i, rate=np.zeros((2, 2), np.float32), tag=f"t{i}"))
            i += 1
    return cells


def conv_df(values: dict[tuple[int, int], float], bin_name="50-200", k_cap=8) -> pd.DataFrame:
    """values : (cell, iters) -> nrmse au K cap (plus une ligne K=1 sans effet)."""
    rows = []
    for (cell, it), v in values.items():
        rows.append(dict(bin=bin_name, cell=cell, area=100, K=k_cap, iters=it, nrmse=v))
        rows.append(dict(bin=bin_name, cell=cell, area=100, K=1, iters=it, nrmse=9.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- select_cells
@pytest.mark.unit
def test_select_cells_is_deterministic_and_balanced():
    sample = sample_of({"50-200": 12, "200-1000": 15})
    a = cc.select_cells(sample, per_bin=5, seed=0)
    b = cc.select_cells(sample, per_bin=5, seed=0)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["bin"] for c in a].count("50-200") == 5
    assert [c["bin"] for c in a].count("200-1000") == 5
    assert len({c["id"] for c in a}) == 10


@pytest.mark.unit
def test_select_cells_orders_by_bin_then_original_index_and_keeps_index_as_id():
    sample = sample_of({"50-200": 6, "200-1000": 6})
    out = cc.select_cells(sample, per_bin=3, seed=1)
    ids = [c["id"] for c in out]
    assert ids == sorted(ids)
    for c in out:
        assert sample[c["id"]]["tag"] == c["tag"]


@pytest.mark.unit
def test_select_cells_takes_all_when_bin_is_smaller():
    sample = sample_of({"50-200": 2, "200-1000": 7})
    out = cc.select_cells(sample, per_bin=4, seed=0)
    assert [c["bin"] for c in out].count("50-200") == 2


@pytest.mark.unit
def test_select_cells_does_not_mutate_input():
    sample = sample_of({"50-200": 5})
    before = copy.deepcopy(sample)
    out = cc.select_cells(sample, per_bin=3, seed=0)
    assert all("id" not in c for c in sample)
    assert [c["tag"] for c in sample] == [c["tag"] for c in before]
    assert out[0] is not sample[out[0]["id"]]


@pytest.mark.unit
def test_select_cells_rejects_bad_per_bin():
    with pytest.raises(ValueError, match="per_bin"):
        cc.select_cells(sample_of({"a": 2}), per_bin=0, seed=0)


# ---------------------------------------------------------------- choose_iters
@pytest.mark.unit
def test_choose_iters_picks_smallest_converged_level():
    vals = {}
    for cell in range(3):
        vals[(cell, 200)] = 0.30
        vals[(cell, 500)] = 0.20   # gain 0.10 >= tol
        vals[(cell, 2000)] = 0.198  # gain 0.002 < tol
    out = cc.choose_iters(conv_df(vals), tol=0.005)
    row = out.iloc[0]
    assert row["chosen_iters"] == 500
    assert row["converged"]
    assert row["k_cap"] == 8
    assert row["gain_200_500"] == pytest.approx(0.10)
    assert row["gain_500_2000"] == pytest.approx(0.002)


@pytest.mark.unit
def test_choose_iters_returns_largest_and_unconverged_when_never_converged(caplog):
    vals = {(c, it): 0.5 - 0.1 * i for c in range(2) for i, it in enumerate((200, 500, 2000))}
    with caplog.at_level("WARNING"):
        out = cc.choose_iters(conv_df(vals))
    assert out.iloc[0]["chosen_iters"] == 2000
    assert not out.iloc[0]["converged"]
    assert "50-200" in caplog.text


@pytest.mark.unit
def test_choose_iters_uses_paired_per_cell_differences():
    a = {0: 0.30, 1: 0.50, 2: 0.60}
    b = {0: 0.30, 1: 0.45, 2: 0.60}
    vals = {}
    for c in range(3):
        vals[(c, 200)], vals[(c, 500)] = a[c], b[c]
    out = cc.choose_iters(conv_df(vals), tol=0.005)
    # mediane des differences (0, 0.05, 0) = 0 ; difference des medianes = 0.50-0.45 = 0.05
    assert out.iloc[0]["gain_200_500"] == pytest.approx(0.0)
    assert out.iloc[0]["chosen_iters"] == 200
    assert out.iloc[0]["converged"]


@pytest.mark.unit
def test_choose_iters_raises_when_a_cell_misses_a_level():
    vals = {(c, it): 0.3 for c in range(2) for it in (200, 500)}
    del vals[(1, 500)]
    with pytest.raises(ValueError, match="cell"):
        cc.choose_iters(conv_df(vals))


# ---------------------------------------------------------------- run_convergence
@pytest.fixture
def tiny_lut(tmp_path):
    import variable_source_fit as vsf
    from profile_lut import LogWLutSpec
    return vsf.lut_for_arm("diff7", tmp_path, LogWLutSpec(**TINY_SPEC_KW))


@pytest.mark.integration
def test_run_convergence_gives_one_row_per_cell_k_and_level(make_cell, tiny_lut):
    cells = [make_cell(i) for i in range(2)]
    df = cc.run_convergence(cells, "diff7", (2, 4), tiny_lut, {"50-200": 3}, {"50-200": 2})
    assert list(df.columns) == ["bin", "cell", "area", "K", "iters", "nrmse", "iou", "mass_err",
                                "peak_err", "fit_time_s"]
    ks = [1, 2, 3]
    assert len(df) == 2 * len(ks) * 2
    for it in (2, 4):
        sub = df[df.iters == it]
        assert sorted(sub.cell.unique()) == [0, 1]
        assert sorted(sub.K.unique()) == ks
    assert (df.fit_time_s >= 0).all()


@pytest.mark.integration
def test_run_convergence_rejects_unknown_bin(make_cell, tiny_lut):
    with pytest.raises(ValueError, match="bin"):
        cc.run_convergence([make_cell(0, bin_name="zzz")], "diff7", (2, 3), tiny_lut, {"50-200": 3}, {"50-200": 2})


@pytest.mark.integration
def test_run_convergence_rejects_empty_iters(make_cell, tiny_lut):
    with pytest.raises(ValueError, match="iters"):
        cc.run_convergence([make_cell(0)], "diff7", (), tiny_lut, {"50-200": 3}, {"50-200": 2})


# ---------------------------------------------------------------- plot / main
@pytest.mark.unit
def test_plot_convergence_writes_png(tmp_path):
    vals = {(c, it): 0.3 / (1 + i) for c in range(2) for i, it in enumerate((200, 500))}
    path = tmp_path / "c.png"
    cc.plot_convergence(conv_df(vals), path)
    assert path.stat().st_size > 0


@pytest.mark.integration
def test_main_writes_csvs_and_png(tmp_path, make_cell):
    sample = [make_cell(0), make_cell(1), make_cell(2, bin_name="200-1000")]
    sample_path = tmp_path / "s.pkl"
    sample_path.write_bytes(pickle.dumps(sample))
    out = tmp_path / "out"
    cc.main(["--sample", str(sample_path), "--arm", "diff7", "--per-bin", "2", "--iters-list", "2", "4",
             "--bins", "50-200", "--device", "cpu", "--out", str(out), "--n-w", "4", "--n-r", "32",
             "--lut-n", "32", "--lut-steps", "60"])
    for name in ("convergence.csv", "convergence_choice.csv", "convergence.png"):
        assert (out / name).exists()
    df = pd.read_csv(out / "convergence.csv")
    assert set(df.iters) == {2, 4}
    assert set(df.bin) == {"50-200"}


# ---------------------------------------------------------------- review follow-ups
def _fail_fit(*a, **k):
    raise AssertionError("fit_group must not be called")


@pytest.mark.unit
def test_select_cells_is_stable_per_bin_when_other_bin_changes():
    full = sample_of({"50-200": 12, "200-1000": 15})
    reduced = [c for c in full if c["bin"] == "200-1000"]
    a = [c["tag"] for c in cc.select_cells(full, 5, 0) if c["bin"] == "200-1000"]
    b = [c["tag"] for c in cc.select_cells(reduced, 5, 0)]
    assert a == b


@pytest.mark.unit
def test_select_cells_interleaved_bins_grouped_in_first_appearance_order():
    sample = [dict(bin=b, area=1, tag=f"t{i}") for i, b in enumerate(["a", "b", "a", "b", "a", "b"])]
    out = cc.select_cells(sample, per_bin=2, seed=0)
    assert [c["bin"] for c in out] == ["a", "a", "b", "b"]
    for name in "ab":
        ids = [c["id"] for c in out if c["bin"] == name]
        assert ids == sorted(ids)


@pytest.mark.unit
def test_choose_iters_gain_equal_to_tol_is_not_converged():
    vals = {(c, it): v for c in range(2) for it, v in ((200, 0.5), (500, 0.25))}
    row = cc.choose_iters(conv_df(vals), tol=0.25).iloc[0]
    assert not row["converged"]
    assert row["chosen_iters"] == 500


@pytest.mark.unit
def test_choose_iters_negative_gain_counts_as_converged():
    vals = {(c, it): v for c in range(2) for it, v in ((200, 0.2), (500, 0.3))}
    row = cc.choose_iters(conv_df(vals), tol=0.005).iloc[0]
    assert row["converged"]
    assert row["chosen_iters"] == 200


@pytest.mark.unit
def test_choose_iters_reports_p90_of_per_cell_gains():
    vals = {}
    for c in range(11):
        vals[(c, 200)], vals[(c, 500)] = 1.0, 1.0 - 0.01 * c
    row = cc.choose_iters(conv_df(vals)).iloc[0]
    assert row["gain_200_500"] == pytest.approx(0.05)
    assert row["gain_p90_200_500"] == pytest.approx(0.09)


@pytest.mark.unit
def test_choose_iters_multi_bin_with_different_k_cap():
    a = conv_df({(c, it): v for c in range(2) for it, v in ((200, 0.3), (500, 0.1))}, "50-200", 8)
    b = conv_df({(c, it): v for c in range(2) for it, v in ((200, 0.3), (500, 0.299))}, "200-1000", 24)
    out = cc.choose_iters(pd.concat([a, b])).set_index("bin")
    assert out.loc["50-200", "k_cap"] == 8 and out.loc["200-1000", "k_cap"] == 24
    assert out.loc["50-200", "chosen_iters"] == 500 and not out.loc["50-200", "converged"]
    assert out.loc["200-1000", "chosen_iters"] == 200 and out.loc["200-1000", "converged"]


@pytest.mark.unit
@pytest.mark.parametrize("levels", [(5,), (5, 5), (), (0, 5)])
def test_run_convergence_rejects_bad_levels_before_any_fit(monkeypatch, make_cell, levels):
    import variable_source_fit as vsf
    monkeypatch.setattr(vsf, "fit_group", _fail_fit)
    with pytest.raises(ValueError, match="iters"):
        cc.run_convergence([make_cell(0)], "diff7", levels, None, {"50-200": 3}, {"50-200": 2})


@pytest.mark.unit
def test_run_convergence_rejects_missing_group_size(monkeypatch, make_cell):
    import variable_source_fit as vsf
    monkeypatch.setattr(vsf, "fit_group", _fail_fit)
    with pytest.raises(ValueError, match="group"):
        cc.run_convergence([make_cell(0)], "diff7", (1, 2), None, {"50-200": 3}, {})


@pytest.mark.unit
def test_main_with_single_or_duplicate_level_fails_fast(tmp_path, monkeypatch, make_cell):
    import variable_source_fit as vsf
    monkeypatch.setattr(vsf, "fit_group", _fail_fit)
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0)]))
    for levels in (["3"], ["3", "3"]):
        with pytest.raises(ValueError, match="iters"):
            cc.main(["--sample", str(path), "--iters-list", *levels, "--bins", "50-200", "--device", "cpu",
                     "--out", str(tmp_path / "o"), *TINY_FLAGS])


@pytest.mark.unit
def test_main_rejects_unknown_bin_name(tmp_path):
    with pytest.raises(SystemExit):
        cc.main(["--sample", "x", "--bins", "zzz", "--out", str(tmp_path)])


@pytest.mark.integration
def test_main_keeps_convergence_csv_when_choose_iters_raises(tmp_path, monkeypatch, make_cell):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cc, "choose_iters", boom)
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0), make_cell(1)]))
    out = tmp_path / "o"
    with pytest.raises(RuntimeError):
        cc.main(["--sample", str(path), "--iters-list", "2", "3", "--bins", "50-200", "--device", "cpu",
                 "--out", str(out), *TINY_FLAGS])
    assert len(pd.read_csv(out / "convergence.csv")) > 0


@pytest.mark.integration
def test_main_restores_device(tmp_path, make_cell):
    import variable_source_fit as vsf
    before = vsf.DEVICE
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0)]))
    cc.main(["--sample", str(path), "--iters-list", "2", "3", "--bins", "50-200", "--device", "cpu",
             "--out", str(tmp_path / "o"), *TINY_FLAGS])
    assert vsf.DEVICE is before


@pytest.mark.integration
def test_main_diffusion_arm_on_tiny_sample(tmp_path, make_cell):
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0), make_cell(1)]))
    out = tmp_path / "o"
    cc.main(["--sample", str(path), "--arm", "diffusion", "--iters-list", "2", "3", "--bins", "50-200",
             "--device", "cpu", "--out", str(out), *TINY_FLAGS])
    assert (out / "convergence_choice.csv").exists()
    assert (out / "diffusion_profiles.pt").exists()


@pytest.mark.integration
def test_run_convergence_is_deterministic(make_cell, tiny_lut):
    cells = [make_cell(i) for i in range(2)]
    args = (cells, "diff7", (2, 3), tiny_lut, {"50-200": 2}, {"50-200": 2})
    a, b = cc.run_convergence(*args), cc.run_convergence(*args)
    pd.testing.assert_series_equal(a.nrmse, b.nrmse)


@pytest.mark.integration
@pytest.mark.parametrize("size", [1, 2])
def test_run_convergence_pairing_when_bin_exceeds_group_size(monkeypatch, make_cell, tiny_lut, size):
    import variable_source_fit as vsf
    seen = []
    real = vsf.fit_group

    def spy(group, kind, kmax, iters, lut5, *a, **k):
        if iters > 1:  # ignore warm-up
            seen.append((iters, tuple(c["id"] for c in group)))
        return real(group, kind, kmax, iters, lut5, *a, **k)
    monkeypatch.setattr(vsf, "fit_group", spy)
    cells = [make_cell(i) for i in range(3)]
    df = cc.run_convergence(cells, "diff7", (2, 3), tiny_lut, {"50-200": 2}, {"50-200": size})
    comp = {it: [g for i, g in seen if i == it] for it in (2, 3)}
    assert comp[2] == comp[3]
    assert sorted(c for g in comp[2] for c in g) == [0, 1, 2]
    assert sorted(df.cell.unique()) == [0, 1, 2]


@pytest.mark.integration
def test_run_convergence_writes_csv_incrementally_and_warns_on_empty_bin(tmp_path, make_cell, tiny_lut, caplog):
    path = tmp_path / "c.csv"
    with caplog.at_level("WARNING"):
        df = cc.run_convergence([make_cell(0)], "diff7", (2, 3), tiny_lut, {"50-200": 2, "200-1000": 2},
                                {"50-200": 2, "200-1000": 2}, csv_path=path)
    assert "200-1000" in caplog.text
    assert len(pd.read_csv(path)) == len(df)
