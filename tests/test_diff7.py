"""Tests du bras diff7 (7 paramètres, tau figé à 0.5, table à w log-espacé) et de ses garde-fous."""
import functools
import pickle
import sys

import numpy as np
import pytest
import torch

import profile_lut as pl
import variable_source_fit as vsf

KMAX, ITERS = 4, 30
TAU_COL, W_COL = 6, 7
TINY = dict(n=32, steps=60, n_r=32)


@pytest.fixture
def lut5_depth1(synthetic_lut):
    return synthetic_lut[:1][None, None]  # (1, 1, 1, n_w, n_r)


@pytest.fixture
def cells(make_cell):
    return [make_cell(0), make_cell(1, blobs=((10.0, 9.0, 5.0, 3.0, 5.0),))]


@pytest.mark.unit
def test_diff7_is_registered_with_7_params_and_frozen_tau_column():
    assert vsf.ARM_NPAR["diff7"] == 7
    assert vsf.ARM_FIXED["diff7"] == [TAU_COL]


@pytest.mark.unit
def test_existing_arms_registration_unchanged():
    assert vsf.ARM_NPAR["gauss"] == 6 and vsf.ARM_NPAR["diffusion"] == 8
    assert vsf.ARM_FIXED == {"diff_tau": [7], "diff_w": [6], "diff7": [6]}


@pytest.mark.unit
def test_diff7_reduces_nrmse_as_sources_are_added(cells, lut5_depth1):
    # Act
    records, _, _ = vsf.fit_group(cells, "diff7", KMAX, ITERS, lut5_depth1)
    # Assert
    first = np.median([r["nrmse"] for r in records if r["K"] == 1])
    last = np.median([r["nrmse"] for r in records if r["K"] == KMAX])
    assert last < first


@pytest.mark.unit
@pytest.mark.parametrize("use_rng", [False, True])
def test_diff7_tau_column_stays_zero_but_w_column_is_trained(cells, lut5_depth1, use_rng):
    # Arrange
    rng = np.random.default_rng(1) if use_rng else None
    # Act
    _, _, (params, active) = vsf.fit_group(cells, "diff7", KMAX, ITERS, lut5_depth1, (), rng)
    # Assert
    rows = params[active > 0]
    assert torch.all(rows[:, TAU_COL] == 0)
    assert torch.any(rows[:, W_COL] != 0)


@pytest.mark.unit
def test_diff7_n_params_is_seven_per_source(cells, lut5_depth1):
    records, _, _ = vsf.fit_group(cells, "diff7", KMAX, 2, lut5_depth1)
    assert all(r["n_params"] == 7 * r["K"] for r in records)
    assert {r["arm"] for r in records} == {"diff7"}


@pytest.mark.unit
def test_diff7_rejects_multi_tau_table(cells, synthetic_lut):
    with pytest.raises(ValueError, match="diff7"):
        vsf.fit_group(cells, "diff7", KMAX, 1, synthetic_lut[None, None])


@pytest.mark.unit
@pytest.mark.parametrize("arm", ["diffusion", "diff_tau", "diff_w"])
def test_diffusion_arms_reject_single_tau_table(cells, lut5_depth1, arm):
    with pytest.raises(ValueError, match=arm):
        vsf.fit_group(cells, arm, KMAX, 1, lut5_depth1)


# ------------------------------------------------------------------ tables par bras
@pytest.mark.unit
def test_lut_for_arm_gauss_needs_no_table(tmp_path):
    assert vsf.lut_for_arm("gauss", tmp_path, pl.LogWLutSpec()) is None


@pytest.mark.unit
def test_lut_for_arm_diff7_returns_log_w_table_without_creating_default(tmp_path):
    # Arrange
    spec = pl.LogWLutSpec(n_w=4, **TINY)
    # Act
    lut5 = vsf.lut_for_arm("diff7", tmp_path, spec)
    # Assert
    assert lut5.shape == (1, 1, 1, 4, 32)
    assert not (tmp_path / "diffusion_profiles.pt").exists()


@pytest.mark.unit
@pytest.mark.parametrize("arm", ["diffusion", "diff_tau", "diff_w"])
def test_lut_for_arm_default_arms_use_default_table(tmp_path, synthetic_lut, arm):
    # Arrange
    torch.save(synthetic_lut, tmp_path / "diffusion_profiles.pt")
    # Act
    lut5 = vsf.lut_for_arm(arm, tmp_path, pl.LogWLutSpec())
    # Assert
    assert lut5.shape == (1, 1, *synthetic_lut.shape)


# ------------------------------------------------------------------ paramètres sauvegardés
@pytest.mark.unit
def test_params_by_cell_keeps_active_rows_only_per_cell_id(cells):
    # Arrange
    params = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    active = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)
    # Act
    out = vsf.params_by_cell(cells, params, active)
    # Assert
    assert set(out) == {0, 1}
    assert out[0].shape == (3, 8) and out[1].shape == (1, 8)
    assert np.array_equal(out[1], params[1, :1].numpy())


@pytest.mark.unit
def test_params_by_cell_returns_copies(cells):
    params, active = torch.zeros(2, 2, 8), torch.ones(2, 2)
    out = vsf.params_by_cell(cells, params, active)
    out[0][:] = 5
    assert torch.all(params == 0)


# ------------------------------------------------------------------ CLI de bout en bout (petit)
@pytest.mark.integration
def test_main_runs_diff7_and_saves_params(tmp_path, monkeypatch, make_cell):
    # Arrange
    sample = [make_cell(i, bin_name="50-200") for i in range(3)]
    sample_path = tmp_path / "sample.pkl"
    sample_path.write_bytes(pickle.dumps(sample))
    out = tmp_path / "run"
    argv = ["vsf", "--sample", str(sample_path), "--arms", "gauss", "diff7", "--bins", "50-200",
            "--max-per-bin", "2", "--iters", "2", "--save-params", "--n-w", "3", "--device", "cpu",
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(vsf, "LogWLutSpec", functools.partial(pl.LogWLutSpec, **TINY))  # table minuscule
    # Act
    vsf.main()
    # Assert
    saved = pickle.loads((out / "params_50_200_diff7.pkl").read_bytes())
    assert set(saved) == {0, 1}  # --max-per-bin 2
    assert all(v.ndim == 2 and v.shape[1] == 8 for v in saved.values())
    gauss = pickle.loads((out / "params_50_200_gauss.pkl").read_bytes())
    assert all(v.shape[1] == 6 for v in gauss.values())
    assert not (out / "diffusion_profiles.pt").exists()
    assert len(list(out.glob("diffusion_profiles_logw_*.pt"))) == 1
    assert (out / "fits_50_200.csv").exists()

