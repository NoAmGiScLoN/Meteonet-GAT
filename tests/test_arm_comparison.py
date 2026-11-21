"""Tests de src/arm_comparison.py : comparaison appariée de bras, équivalence, accumulation de w."""
import sys

import numpy as np
import pandas as pd
import pytest

import arm_comparison as ac


def fits_df(rows):
    """rows : (bin, arm, K, cell, nrmse)."""
    return pd.DataFrame(rows, columns=["bin", "arm", "K", "cell", "nrmse"]).assign(area=100, iou=0.5,
                                                                                    mass_err=0.1, peak_err=0.1,
                                                                                    n_params=0)


@pytest.fixture
def two_arm_df():
    rows = []
    for cell, (a, b) in enumerate([(0.30, 0.31), (0.20, 0.22), (0.10, 0.12)]):
        for K in (6, 8):
            rows += [("50-200", "diff7", K, cell, a), ("50-200", "diffusion", K, cell, b)]
    return fits_df(rows)


# ------------------------------------------------------------------ load_fits
@pytest.mark.unit
def test_load_fits_concatenates_all_bin_files(tmp_path):
    fits_df([("a", "gauss", 1, 0, 0.1)]).to_csv(tmp_path / "fits_a.csv", index=False)
    fits_df([("b", "gauss", 1, 1, 0.2)]).to_csv(tmp_path / "fits_b.csv", index=False)
    df = ac.load_fits(tmp_path)
    assert sorted(df.bin) == ["a", "b"]


@pytest.mark.unit
def test_load_fits_raises_without_files(tmp_path):
    with pytest.raises(ValueError, match="fits_"):
        ac.load_fits(tmp_path)


# ------------------------------------------------------------------ paired_delta
@pytest.mark.unit
def test_paired_delta_columns_and_values(two_arm_df):
    # Act
    out = ac.paired_delta(two_arm_df, "diff7", "diffusion")
    # Assert
    assert list(out.columns) == ["bin", "K", "med_a", "med_b", "delta", "paired_delta", "n"]
    row = out[out.K == 6].iloc[0]
    assert row.med_a == pytest.approx(0.20) and row.med_b == pytest.approx(0.22)
    assert row.delta == pytest.approx(-0.02) and row.paired_delta == pytest.approx(-0.02)
    assert row.n == 3


@pytest.mark.unit
def test_paired_delta_uses_only_cells_present_in_both_arms(two_arm_df):
    df = two_arm_df[~((two_arm_df.arm == "diffusion") & (two_arm_df.cell == 0))]
    out = ac.paired_delta(df, "diff7", "diffusion")
    assert set(out.n) == {2}
    assert out.med_a.iloc[0] == pytest.approx(0.15)  # médiane des cellules 1 et 2 seulement


@pytest.mark.unit
def test_paired_delta_median_of_differences_differs_from_difference_of_medians():
    rows = []
    for cell, (a, b) in enumerate([(0.1, 0.5), (0.5, 0.1), (0.3, 0.31)]):
        rows += [("x", "A", 1, cell, a), ("x", "B", 1, cell, b)]
    out = ac.paired_delta(fits_df(rows), "A", "B").iloc[0]
    assert out.delta == pytest.approx(0.3 - 0.31)
    assert out.paired_delta == pytest.approx(-0.01)  # médiane de (-0.4, 0.4, -0.01)


@pytest.mark.unit
def test_paired_delta_unknown_arm_raises(two_arm_df):
    with pytest.raises(ValueError, match="nope"):
        ac.paired_delta(two_arm_df, "diff7", "nope")


# ------------------------------------------------------------------ equivalence_table
def delta_table(rows):
    """Valeurs données dans la colonne testée par défaut (paired_delta) ; delta reste nul."""
    return pd.DataFrame(rows, columns=["bin", "K", "paired_delta"]).assign(med_a=0, med_b=0, delta=0, n=1)


@pytest.mark.unit
def test_equivalence_uses_paired_delta_by_default():
    # Arrange : différence de médianes hors tolérance, différence appariée dans la tolérance
    d = delta_table([("x", 6, 0.002)]).assign(delta=0.02)
    # Act
    out = ac.equivalence_table(d, 0.005, 6).iloc[0]
    # Assert
    assert bool(out.passes) and out.criterion == "paired_delta"


@pytest.mark.unit
def test_equivalence_can_use_difference_of_medians():
    # Arrange
    d = delta_table([("x", 6, 0.002)]).assign(delta=0.02)
    # Act
    out = ac.equivalence_table(d, 0.005, 6, column="delta").iloc[0]
    # Assert
    assert not out.passes and out.max_abs_delta == pytest.approx(0.02)


@pytest.mark.unit
def test_equivalence_one_sided_passes_when_first_arm_is_better():
    # Arrange : bras a (diff7) meilleur de 0.02 que le bras b
    d = delta_table([("x", 6, -0.02), ("x", 8, 0.003)])
    # Act
    out = ac.equivalence_table(d, 0.005, 6).iloc[0]
    # Assert
    assert bool(out.passes) and out.sided == "one"
    assert out.max_delta == pytest.approx(0.003) and out.max_abs_delta == pytest.approx(0.02)


@pytest.mark.unit
def test_equivalence_two_sided_fails_when_first_arm_is_much_better():
    d = delta_table([("x", 6, -0.02)])
    assert not ac.equivalence_table(d, 0.005, 6, sided="two").iloc[0].passes


@pytest.mark.unit
def test_equivalence_unknown_sided_raises():
    with pytest.raises(ValueError, match="both"):
        ac.equivalence_table(delta_table([("x", 6, 0.0)]), 0.005, 6, sided="both")


@pytest.mark.unit
def test_equivalence_unknown_column_raises():
    with pytest.raises(ValueError, match="nope"):
        ac.equivalence_table(delta_table([("x", 6, 0.0)]), 0.005, 6, column="nope")


@pytest.mark.unit
def test_equivalence_passes_when_all_deltas_within_tolerance():
    d = delta_table([("x", 6, 0.004), ("x", 8, -0.003)])
    out = ac.equivalence_table(d, ac.EQUIV_TOL, ac.MIN_K_COMPARE).iloc[0]
    assert out.max_abs_delta == pytest.approx(0.004) and bool(out.passes)
    assert out.k_cap == 8 and out.delta_at_cap == pytest.approx(-0.003)


@pytest.mark.unit
def test_equivalence_fails_when_a_large_k_exceeds_tolerance():
    d = delta_table([("x", 6, 0.001), ("x", 8, 0.02)])
    assert not ac.equivalence_table(d, 0.005, 6).iloc[0].passes


@pytest.mark.unit
def test_equivalence_ignores_small_k():
    d = delta_table([("x", 1, 0.5), ("x", 6, 0.001)])
    out = ac.equivalence_table(d, 0.005, 6).iloc[0]
    assert bool(out.passes) and out.max_abs_delta == pytest.approx(0.001)


@pytest.mark.unit
def test_equivalence_bin_without_large_k_fails_with_nan():
    out = ac.equivalence_table(delta_table([("x", 1, 0.0), ("x", 2, 0.0)]), 0.005, 6).iloc[0]
    assert not out.passes and np.isnan(out.max_abs_delta)


@pytest.mark.unit
def test_equivalence_one_row_per_bin():
    d = delta_table([("x", 6, 0.0), ("y", 6, 0.1)])
    out = ac.equivalence_table(d, 0.005, 6)
    assert dict(zip(out.bin, out.passes)) == {"x": True, "y": False}


# ------------------------------------------------------------------ w_pileup
@pytest.mark.unit
def test_w_pileup_fractions_on_sigmoid_of_column_7():
    # Arrange : sigmoid(-5)=0.0067 (bas), sigmoid(5)=0.993 (haut), deux valeurs centrales
    col = np.array([-5.0, 5.0, 0.0, 0.5])
    params = {0: np.zeros((2, 8)), 1: np.zeros((2, 8))}
    params[0][:, 7], params[1][:, 7] = col[:2], col[2:]
    # Act
    out = ac.w_pileup(params, ac.PILEUP_EDGE)
    # Assert
    assert out == {"frac_low": pytest.approx(0.25), "frac_high": pytest.approx(0.25)}


@pytest.mark.unit
def test_w_pileup_rejects_empty_input_and_missing_column():
    with pytest.raises(ValueError):
        ac.w_pileup({}, 0.03)
    with pytest.raises(ValueError):
        ac.w_pileup({0: np.zeros((2, 6))}, 0.03)


@pytest.mark.unit
def test_w_pileup_rejects_bad_edge():
    with pytest.raises(ValueError):
        ac.w_pileup({0: np.zeros((1, 8))}, 0.7)


# ------------------------------------------------------------------ baseline_reproduction
@pytest.mark.unit
def test_baseline_reproduction_flags_differences_beyond_tolerance():
    new = fits_df([("x", "gauss", 1, 0, 0.100), ("x", "gauss", 2, 0, 0.200)])
    old = fits_df([("x", "gauss", 1, 0, 0.1004), ("x", "gauss", 2, 0, 0.250)])
    out = ac.baseline_reproduction(new, old, 0.005).set_index("K")
    assert bool(out.loc[1, "passes"]) and not bool(out.loc[2, "passes"])
    assert out.loc[2, "diff"] == pytest.approx(-0.05)


@pytest.mark.unit
def test_baseline_reproduction_keeps_only_common_bin_arm_k():
    new = fits_df([("x", "gauss", 1, 0, 0.1), ("x", "diffusion", 1, 0, 0.1)])
    old = fits_df([("x", "gauss", 1, 0, 0.1), ("y", "gauss", 1, 0, 0.1)])
    out = ac.baseline_reproduction(new, old, 0.005)
    assert len(out) == 1 and out.arm.iloc[0] == "gauss"


@pytest.mark.unit
def test_baseline_reproduction_without_overlap_raises():
    with pytest.raises(ValueError, match="common"):
        ac.baseline_reproduction(fits_df([("x", "gauss", 1, 0, 0.1)]), fits_df([("y", "gauss", 1, 0, 0.1)]), 0.005)


# ------------------------------------------------------------------ plot_w_histograms
@pytest.mark.unit
def test_plot_w_histograms_writes_png_with_log_axis(tmp_path):
    # Arrange
    params = {"diff7": {0: np.random.default_rng(0).normal(size=(20, 8))}}
    ws = {"diff7": np.geomspace(0.02, 0.5, 5)}
    path = tmp_path / "w.png"
    # Act
    ac.plot_w_histograms(params, ws, path)
    # Assert
    assert path.stat().st_size > 0


@pytest.mark.unit
def test_plot_w_histograms_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        ac.plot_w_histograms({}, {}, tmp_path / "w.png")


# ------------------------------------------------------------------ CLI
@pytest.mark.integration
def test_main_writes_comparison_tables(tmp_path, monkeypatch, two_arm_df):
    # Arrange
    run = tmp_path / "run"
    run.mkdir()
    two_arm_df.to_csv(run / "fits_50_200.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["arm_comparison", "--runs", str(run), "--out", str(out)])
    # Act
    ac.main()
    # Assert
    assert (out / "run_paired_delta.csv").exists()
    equiv = pd.read_csv(out / "run_equivalence.csv")
    assert list(equiv.bin) == ["50-200"]


@pytest.mark.integration
def test_compare_run_with_params_writes_pileup_and_histogram(tmp_path, two_arm_df):
    # Arrange
    import pickle
    import profile_lut as pl
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir(), out.mkdir()
    two_arm_df.to_csv(run / "fits_50_200.csv", index=False)
    rng = np.random.default_rng(0)
    for arm in ("diffusion", "diff7"):
        (run / f"params_50_200_{arm}.pkl").write_bytes(pickle.dumps({0: rng.normal(size=(10, 8))}))
    pl.build_logw_lut(pl.LogWLutSpec(n_w=3, n=32, steps=20, n_r=16), run)
    # Act
    ac.compare_run(run, out)
    # Assert
    pile = pd.read_csv(out / "run_w_pileup.csv")
    assert set(pile.arm) == {"diffusion", "diff7"} and "flag" in pile
    assert (out / "run_w_hist.png").stat().st_size > 0


@pytest.mark.unit
def test_w_nodes_raises_when_log_w_table_missing(tmp_path):
    with pytest.raises(ValueError, match="logw"):
        ac.w_nodes(tmp_path, "diff7")
    assert len(ac.w_nodes(tmp_path, "diffusion")) == 16


def build_two_tables(run):
    import profile_lut as pl
    specs = [pl.LogWLutSpec(n_w=n_w, n=32, steps=5, n_r=16) for n_w in (3, 4)]
    for spec in specs:
        pl.build_logw_lut(spec, run)
    return specs


@pytest.mark.unit
def test_w_nodes_ambiguous_tables_raise_listing_them(tmp_path):
    specs = build_two_tables(tmp_path)
    with pytest.raises(ValueError) as exc:
        ac.w_nodes(tmp_path, "diff7")
    assert all(spec.key() in str(exc.value) for spec in specs)


@pytest.mark.unit
def test_w_nodes_logw_key_picks_one_table(tmp_path):
    specs = build_two_tables(tmp_path)
    assert len(ac.w_nodes(tmp_path, "diff7", logw_key=specs[1].key())) == 4
    with pytest.raises(ValueError, match="deadbeef"):
        ac.w_nodes(tmp_path, "diff7", logw_key="deadbeef")


@pytest.mark.unit
def test_default_w_nodes_come_from_profile_lut():
    import profile_lut as pl
    assert np.array_equal(ac.w_nodes(".", "diffusion"), pl.default_w_nodes())
    assert (pl.DEFAULT_W_MIN, pl.DEFAULT_W_MAX, pl.DEFAULT_N_W) == (0.02, 0.5, 16)


# ------------------------------------------------------------------ missing arm / archived
@pytest.mark.integration
def test_compare_run_skips_run_missing_an_arm_and_logs_warning(tmp_path, two_arm_df, caplog):
    good, bad, out = tmp_path / "good", tmp_path / "bad", tmp_path / "out"
    for d in (good, bad, out):
        d.mkdir()
    two_arm_df.to_csv(good / "fits_a.csv", index=False)
    two_arm_df[two_arm_df.arm == "diffusion"].to_csv(bad / "fits_a.csv", index=False)
    with caplog.at_level("WARNING"):
        ac.compare_run(bad, out)
        ac.compare_run(good, out)
    assert "bad" in caplog.text and "diff7" in caplog.text
    assert not (out / "bad_paired_delta.csv").exists() and (out / "good_paired_delta.csv").exists()


@pytest.mark.integration
def test_main_continues_past_run_with_missing_arm(tmp_path, monkeypatch, two_arm_df):
    good, bad, out = tmp_path / "good", tmp_path / "bad", tmp_path / "out"
    good.mkdir(), bad.mkdir()
    two_arm_df.to_csv(good / "fits_a.csv", index=False)
    two_arm_df[two_arm_df.arm == "diffusion"].to_csv(bad / "fits_a.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["ac", "--runs", str(bad), str(good), "--out", str(out)])
    ac.main()
    assert (out / "good_equivalence.csv").exists()


@pytest.mark.integration
def test_main_archived_writes_reproduction_on_diffusion_arm(tmp_path, monkeypatch, two_arm_df):
    run, arch, out = tmp_path / "run", tmp_path / "arch", tmp_path / "out"
    run.mkdir(), arch.mkdir()
    two_arm_df.to_csv(run / "fits_a.csv", index=False)
    # archive: other cell ids (as with --max-per-bin), diffusion medians shifted by 0.05 at K=8
    old = two_arm_df[two_arm_df.arm == "diffusion"].assign(cell=lambda d: d.cell + 100)
    old = old.assign(nrmse=old.nrmse + (old.K == 8) * 0.05)
    old.to_csv(arch / "fits_a.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["ac", "--runs", str(run), "--out", str(out), "--archived", str(arch)])
    ac.main()
    rep = pd.read_csv(out / "reproduction.csv").set_index("K")
    assert set(rep.arm) == {"diffusion"} and set(rep.run) == {"run"}
    assert bool(rep.loc[6, "passes"]) and not bool(rep.loc[8, "passes"])


# ------------------------------------------------------------------ --reference
def write_arm(run, df, arm):
    run.mkdir(exist_ok=True)
    df[df.arm == arm].to_csv(run / "fits_50_200.csv", index=False)


@pytest.fixture
def run_and_reference(tmp_path, two_arm_df):
    """run: diff7 only; reference: diffusion only (same cells)."""
    run, ref, out = tmp_path / "run", tmp_path / "ref", tmp_path / "out"
    out.mkdir()
    write_arm(run, two_arm_df, "diff7")
    write_arm(ref, two_arm_df, "diffusion")
    return run, ref, out


@pytest.mark.integration
def test_compare_run_fills_missing_arm_from_reference(run_and_reference):
    run, ref, out = run_and_reference
    ac.compare_run(run, out, reference=ref)
    delta = pd.read_csv(out / "run_paired_delta.csv")
    assert delta[delta.K == 6].paired_delta.iloc[0] == pytest.approx(-0.02)
    assert (out / "run_equivalence.csv").exists()


@pytest.mark.integration
def test_compare_run_without_reference_still_skips_missing_arm(run_and_reference, caplog):
    run, _, out = run_and_reference
    with caplog.at_level("WARNING"):
        ac.compare_run(run, out)
    assert "diffusion" in caplog.text and not (out / "run_paired_delta.csv").exists()


@pytest.mark.integration
def test_reference_without_params_writes_comparison_and_logs_note(run_and_reference, caplog):
    import pickle
    run, ref, out = run_and_reference
    rng = np.random.default_rng(0)
    (run / "params_50_200_diff7.pkl").write_bytes(pickle.dumps({0: rng.normal(size=(10, 8))}))
    import profile_lut as pl
    pl.build_logw_lut(pl.LogWLutSpec(n_w=3, n=32, steps=20, n_r=16), run)
    with caplog.at_level("INFO"):
        ac.compare_run(run, out, reference=ref)
    assert (out / "run_equivalence.csv").exists()
    assert "no params" in caplog.text and "diffusion" in caplog.text
    assert set(pd.read_csv(out / "run_w_pileup.csv").arm) == {"diff7"}


@pytest.mark.integration
def test_reference_params_are_used_for_pileup(run_and_reference):
    import pickle
    run, ref, out = run_and_reference
    (ref / "params_50_200_diffusion.pkl").write_bytes(pickle.dumps({0: np.zeros((4, 8))}))
    ac.compare_run(run, out, reference=ref)
    assert set(pd.read_csv(out / "run_w_pileup.csv").arm) == {"diffusion"}


@pytest.mark.integration
def test_reference_mismatched_ids_use_intersection_and_warn(run_and_reference, caplog):
    run, ref, out = run_and_reference
    ref_df = pd.read_csv(ref / "fits_50_200.csv")
    ref_df[ref_df.cell != 0].assign(cell=lambda d: d.cell).to_csv(ref / "fits_50_200.csv", index=False)
    with caplog.at_level("WARNING"):
        ac.compare_run(run, out, reference=ref)
    assert "intersection" in caplog.text
    assert set(pd.read_csv(out / "run_paired_delta.csv").n) == {2}


@pytest.mark.integration
def test_reference_empty_intersection_raises(run_and_reference):
    run, ref, out = run_and_reference
    pd.read_csv(ref / "fits_50_200.csv").assign(cell=lambda d: d.cell + 100).to_csv(ref / "fits_50_200.csv", index=False)
    with pytest.raises(ValueError, match="intersection"):
        ac.compare_run(run, out, reference=ref)


@pytest.mark.integration
def test_main_reference_option(run_and_reference, monkeypatch):
    run, ref, out = run_and_reference
    monkeypatch.setattr(sys, "argv", ["ac", "--runs", str(run), "--out", str(out), "--reference", str(ref)])
    ac.main()
    assert (out / "run_equivalence.csv").exists()
