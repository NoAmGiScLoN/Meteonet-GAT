"""Tests de non-régression : fit_group et build_profile_lut doivent rester identiques aux données de référence."""
import sys
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(DATA))

import make_golden as mg  # noqa: E402
import variable_source_fit as vsf  # noqa: E402
from conftest import build_synthetic_lut  # noqa: E402

ATOL = 1e-5  # cross-platform / torch-version float noise (golden generated with torch recorded in make_golden.py)
METRICS = ("nrmse", "iou", "mass_err", "peak_err")
CASES = [(k, r) for k in mg.KINDS for r in (False, True)]


@pytest.fixture(scope="module")
def golden():
    return torch.load(DATA / "fit_group_golden.pt")


@pytest.mark.unit
@pytest.mark.parametrize("kind,use_rng", CASES, ids=[f"{k}-rng{int(r)}" for k, r in CASES])
def test_fit_group_matches_golden_reference(golden, kind, use_rng):
    # Arrange
    ref = golden[f"{kind}|{int(use_rng)}"]
    # Act
    out = mg.run_case(kind, use_rng, build_synthetic_lut())
    # Assert
    assert len(out["records"]) == len(ref["records"])
    for new, old in zip(out["records"], ref["records"]):
        for key in ("cell", "bin", "area", "arm", "K", "n_params"):
            assert new[key] == old[key]
        for key in METRICS:
            assert new[key] == pytest.approx(old[key], abs=ATOL)
    assert torch.allclose(out["params"], ref["params"], atol=ATOL)
    assert torch.equal(out["active"], ref["active"])


@pytest.mark.unit
def test_build_profile_lut_matches_tiny_reference(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(vsf, "OUT", tmp_path)
    ref = torch.load(DATA / "lut_tiny_ref.pt")
    # Act
    lut = vsf.build_profile_lut(**mg.TINY_LUT_ARGS)
    # Assert
    assert lut.shape == ref.shape
    assert torch.allclose(lut, ref, atol=ATOL)


@pytest.mark.unit
def test_build_profile_lut_uses_default_path_and_caches(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(vsf, "OUT", tmp_path)
    # Act
    first = vsf.build_profile_lut(**mg.TINY_LUT_ARGS)
    second = vsf.build_profile_lut(n=8, steps=1, n_tau=2, n_w=2, n_r=8)  # arguments ignorés : cache
    # Assert
    assert (tmp_path / "diffusion_profiles.pt").exists()
    assert torch.equal(first, second)
