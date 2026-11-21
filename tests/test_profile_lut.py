"""Tests de src/profile_lut.py : coeur EDP partagé et table à w espacé en log."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import profile_lut as pl
import variable_source_fit as vsf

TINY = dict(n=32, steps=60, n_r=32)


def tiny_spec(**kw):
    return pl.LogWLutSpec(**{**dict(n_w=4), **TINY, **kw})


# ------------------------------------------------------------------ diffuse_profiles
@pytest.mark.unit
def test_diffuse_profiles_returns_flat_normalised_profiles_zero_at_rmax():
    # Arrange
    taus, ws = torch.linspace(0, 1, 3), torch.linspace(0.02, 0.5, 4)
    # Act
    prof = pl.diffuse_profiles(taus, ws, n_r=32, n=32, steps=60, sigma0=0.15, floor=0.005, dmax=0.24)
    # Assert
    assert prof.shape == (12, 32)
    assert torch.allclose(prof.max(1).values, torch.ones(12))
    assert torch.all(prof[:, -1] == 0)


@pytest.mark.unit
def test_profile_lut_l_lut_equals_vsf_l_lut():
    assert pl.L_LUT == vsf.L_LUT


# ------------------------------------------------------------------ LogWLutSpec
@pytest.mark.unit
def test_spec_defaults_match_documented_values():
    # Act
    spec = pl.LogWLutSpec()
    # Assert
    assert (spec.w_min, spec.w_max, spec.n_w, spec.tau) == (0.02, 0.5, 16, 0.5)
    assert (spec.n_r, spec.n, spec.steps) == (128, 96, 1200)


@pytest.mark.unit
def test_spec_casts_numeric_fields_to_float_so_keys_agree():
    ints = pl.LogWLutSpec(tau=1, w_max=1, dmax=1, floor=0, sigma0=1)
    floats = pl.LogWLutSpec(tau=1.0, w_max=1.0, dmax=1.0, floor=0.0, sigma0=1.0)
    assert ints.key() == floats.key() and ints == floats
    assert all(type(getattr(ints, f)) is float for f in ("w_min", "w_max", "tau", "sigma0", "floor", "dmax"))


@pytest.mark.unit
def test_default_spec_key_is_unchanged():
    assert pl.LogWLutSpec().key() == "ace17c5890"


@pytest.mark.unit
def test_spec_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        pl.LogWLutSpec().n_w = 3


@pytest.mark.unit
@pytest.mark.parametrize("kw", [dict(w_min=0.0), dict(w_min=-1.0), dict(w_min=0.5, w_max=0.5),
                                dict(w_min=0.6, w_max=0.5), dict(n_w=1), dict(tau=-0.1), dict(tau=1.1),
                                dict(n_r=0), dict(n=-3), dict(steps=0), dict(n_w=2.5)])
def test_spec_rejects_invalid_values(kw):
    with pytest.raises(ValueError):
        pl.LogWLutSpec(**kw)


@pytest.mark.unit
@pytest.mark.parametrize("field,value", [("w_min", 0.03), ("w_max", 0.6), ("n_w", 9), ("tau", 0.4), ("n_r", 64),
                                         ("n", 48), ("steps", 100), ("sigma0", 0.2), ("floor", 0.01),
                                         ("dmax", 0.2)])
def test_spec_key_changes_with_each_field(field, value):
    base = pl.LogWLutSpec()
    changed = dataclasses.replace(base, **{field: value})
    assert changed.key() != base.key()


@pytest.mark.unit
def test_spec_key_is_stable_10_hex_chars():
    key = pl.LogWLutSpec().key()
    assert key == pl.LogWLutSpec().key()
    assert len(key) == 10 and int(key, 16) >= 0


@pytest.mark.unit
def test_logw_lut_path_embeds_key(tmp_path):
    spec = tiny_spec()
    assert pl.logw_lut_path(tmp_path, spec) == tmp_path / f"diffusion_profiles_logw_{spec.key()}.pt"


# ------------------------------------------------------------------ build / load
@pytest.mark.unit
def test_build_logw_lut_has_log_spaced_nodes_and_expected_shape(tmp_path):
    # Arrange
    spec = tiny_spec(n_w=5)
    # Act
    table = pl.build_logw_lut(spec, tmp_path)
    # Assert
    ws = table["ws"].numpy()
    assert table["lut"].shape == (1, 5, 32)
    assert ws[0] == pytest.approx(spec.w_min) and ws[-1] == pytest.approx(spec.w_max)
    assert np.allclose(ws[1:] / ws[:-1], (spec.w_max / spec.w_min) ** (1 / 4), rtol=1e-5)
    assert table["spec"] == dataclasses.asdict(spec)
    assert pl.logw_lut_path(tmp_path, spec).exists()


@pytest.mark.unit
def test_logw_profiles_are_normalised_and_zero_at_rmax(tmp_path):
    lut = pl.build_logw_lut(tiny_spec(), tmp_path)["lut"]
    assert torch.allclose(lut.max(-1).values, torch.ones(1, 4))
    assert torch.all(lut[..., -1] == 0)


@pytest.mark.unit
def test_build_logw_lut_creates_missing_output_dir(tmp_path):
    out = tmp_path / "nested" / "dir"
    pl.build_logw_lut(tiny_spec(), out)
    assert pl.logw_lut_path(out, tiny_spec()).exists()


@pytest.mark.unit
def test_building_logw_lut_never_touches_default_table(tmp_path):
    # Arrange
    default = Path(vsf.__file__).resolve().parent.parent / "results" / "03_variable_sources" / "diffusion_profiles.pt"
    before = default.stat().st_mtime_ns if default.exists() else None
    # Act
    pl.load_or_build_logw_lut(tiny_spec(), tmp_path)
    # Assert
    assert not (tmp_path / "diffusion_profiles.pt").exists()
    assert (default.stat().st_mtime_ns if default.exists() else None) == before


@pytest.mark.unit
def test_load_or_build_reuses_existing_file(tmp_path, monkeypatch):
    # Arrange
    spec = tiny_spec()
    first = pl.load_or_build_logw_lut(spec, tmp_path)
    monkeypatch.setattr(pl, "diffuse_profiles", lambda *a, **k: pytest.fail("rebuilt"))
    # Act
    second = pl.load_or_build_logw_lut(spec, tmp_path)
    # Assert
    assert torch.equal(first["lut"], second["lut"])


@pytest.mark.unit
def test_load_or_build_raises_when_stored_spec_differs(tmp_path):
    # Arrange
    spec = tiny_spec()
    table = pl.build_logw_lut(dataclasses.replace(spec, n_w=5), tmp_path)
    torch.save(table, pl.logw_lut_path(tmp_path, spec))  # fichier falsifié sous la clé de `spec`
    # Act / Assert
    with pytest.raises(ValueError, match="spec"):
        pl.load_or_build_logw_lut(spec, tmp_path)


# ------------------------------------------------------------------ w_physical
@pytest.fixture
def log_nodes():
    return np.geomspace(0.02, 0.5, 5)


@pytest.mark.unit
def test_w_physical_saturates_at_bounds(log_nodes):
    out = pl.w_physical(np.array([-1e3, 1e3]), log_nodes)
    assert out[0] == pytest.approx(0.02) and out[1] == pytest.approx(0.5)


@pytest.mark.unit
def test_w_physical_zero_is_geometric_mean_for_odd_node_count(log_nodes):
    out = pl.w_physical(np.array([0.0]), log_nodes)
    assert out[0] == pytest.approx(math.sqrt(0.02 * 0.5))


@pytest.mark.unit
def test_w_physical_zero_is_mean_of_two_middle_nodes_for_even_count():
    nodes = np.geomspace(0.02, 0.5, 4)
    out = pl.w_physical(np.array([0.0]), nodes)
    assert out[0] == pytest.approx(0.5 * (nodes[1] + nodes[2]))


@pytest.mark.unit
def test_w_physical_is_monotone_and_keeps_shape(log_nodes):
    raw = np.linspace(-6, 6, 21).reshape(3, 7)
    out = pl.w_physical(raw, log_nodes)
    assert out.shape == raw.shape
    assert np.all(np.diff(out.ravel()) > 0)


@pytest.mark.unit
def test_w_physical_rejects_bad_nodes():
    with pytest.raises(ValueError):
        pl.w_physical(np.array([0.0]), np.array([0.1]))
