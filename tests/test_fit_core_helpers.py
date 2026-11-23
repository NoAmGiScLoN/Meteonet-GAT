"""Tests des briques de fit_group : make_targets, place_sources, optimise, evaluate."""
import dataclasses

import numpy as np
import pytest
import torch

import variable_source_fit as vsf
from conftest import build_cell

KMAX = 4


def _cells(n=3):
    sizes = [24, 20, 16, 22]
    return [build_cell(i, size=sizes[i]) for i in range(n)]


def _fresh(cells, kind="gauss", kmax=KMAX):
    t = vsf.make_targets(cells)
    P = 6 if kind == "gauss" else 8
    return t, torch.zeros(len(cells), kmax, P), torch.zeros(len(cells), kmax)


@pytest.mark.unit
def test_make_targets_shapes_padding_and_npix():
    # Arrange
    cells = _cells(3)
    # Act
    t = vsf.make_targets(cells)
    # Assert
    assert (t.Hm, t.Wm) == (24, 24)
    assert t.T.shape == t.sT.shape == t.valid.shape == (3, 24, 24)
    assert t.valid.dtype == torch.bool
    assert t.npix.tolist() == [24 * 24, 20 * 20, 16 * 16]
    assert torch.equal(t.T[1, :20, :20], torch.from_numpy(cells[1]["rate"]))
    assert float(t.T[1, 20:, :].abs().sum()) == 0 and float(t.T[1, :, 20:].abs().sum()) == 0
    assert torch.allclose(t.sT, t.T.sqrt())
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.Hm = 1


@pytest.mark.unit
@pytest.mark.parametrize("use_rng", [False, True])
def test_place_sources_mixed_k_gives_per_cell_counts(use_rng):
    # Arrange
    cells = _cells(3)
    t, params, active = _fresh(cells)
    rng = np.random.default_rng(3) if use_rng else None
    # Act
    vsf.place_sources(params, active, t, cells, np.array([0, 0, 0]), np.array([1, 3, 0]), "gauss", rng)
    # Assert
    assert active.sum(1).tolist() == [1, 3, 0]
    assert float(params[2].abs().sum()) == 0
    assert float(params[0, 1:].abs().sum()) == 0


@pytest.mark.unit
def test_place_sources_writes_only_new_slots():
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    vsf.place_sources(params, active, t, cells, np.array([0, 0]), np.array([1, 1]), "gauss", None)
    first = params[:, 0].clone()
    # Act
    vsf.place_sources(params, active, t, cells, np.array([1, 1]), np.array([3, 2]), "gauss", None)
    # Assert
    assert torch.equal(params[:, 0], first)
    assert active.sum(1).tolist() == [3, 2]


@pytest.mark.unit
def test_place_sources_skipped_cell_consumes_no_rng():
    # Arrange
    cells = _cells(3)
    t, p_a, a_a = _fresh(cells)
    a_a[1, :2] = 1  # cell 1 already holds 2 sources (contiguous precondition) and is skipped
    kept = [cells[0], cells[2]]
    t2, p_b, a_b = _fresh(kept)
    # Act
    vsf.place_sources(p_a, a_a, t, cells, np.array([0, 2, 0]), np.array([2, 2, 2]), "gauss", np.random.default_rng(5))
    vsf.place_sources(p_b, a_b, t2, kept, np.array([0, 0]), np.array([2, 2]), "gauss", np.random.default_rng(5))
    # Assert
    assert torch.equal(p_a[0], p_b[0])
    assert torch.equal(p_a[2], p_b[1])
    assert float(a_a[1].sum()) == 2  # skipped cell untouched


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["diffusion", "diff_tau", "diff_w", "diff7"])
def test_place_sources_frozen_columns_stay_default_with_rng(kind):
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells, kind)
    # Act
    vsf.place_sources(params, active, t, cells, np.array([0, 0]), np.array([2, 2]), kind, np.random.default_rng(1))
    # Assert
    for col in vsf.ARM_FIXED.get(kind, []):
        assert float(params[..., col].abs().sum()) == 0
    free = [c for c in (6, 7) if c not in vsf.ARM_FIXED.get(kind, [])]
    for col in free:
        assert float(params[:, :2, col].abs().sum()) > 0


@pytest.mark.unit
def test_evaluate_equals_metrics_of_render(synthetic_lut):
    # Arrange
    cells = _cells(2)
    lut5 = synthetic_lut[None, None]
    t, params, active = _fresh(cells, "diffusion")
    vsf.place_sources(params, active, t, cells, np.zeros(2, int), np.full(2, 2), "diffusion", None)
    # Act
    rec, recs = vsf.evaluate(params, active, t, "diffusion", lut5)
    # Assert
    with torch.no_grad():
        want = vsf.render_windowed(params, active, t.Hm, t.Wm, "diffusion", lut5)
    assert torch.equal(rec, want)
    assert recs == vsf.metrics(want, t.T, t.valid)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["gauss", "diff_w", "diff7"])
def test_optimise_returns_detached_and_keeps_frozen_columns(kind, synthetic_lut):
    # Arrange
    cells = _cells(2)
    lut5 = None if kind == "gauss" else synthetic_lut[None, None]
    if kind == "diff7":
        lut5 = synthetic_lut[:1][None, None]
    t, params, active = _fresh(cells, kind)
    vsf.place_sources(params, active, t, cells, np.zeros(2, int), np.full(2, 2), kind, None)
    before = params.clone()
    # Act
    out = vsf.optimise(params, active, t, kind, lut5, 3, lr=0.05, lr_schedule=None)
    # Assert
    assert not out.requires_grad
    assert out.shape == before.shape
    assert not torch.equal(out, before)
    for col in vsf.ARM_FIXED.get(kind, []):
        assert float(out[..., col].abs().sum()) == 0


def _snapshot(params, active):
    return params.clone(), active.clone()


def _place(params, active, t, cells, k_from, k_to, kind="gauss", rng=None, lut5=None):
    vsf.place_sources(params, active, t, cells, np.asarray(k_from), np.asarray(k_to), kind, rng, lut5)


@pytest.mark.unit
@pytest.mark.parametrize("active_rows", [
    [[1, 0, 1, 0], [1, 1, 0, 0]],   # hole below k_from in cell 0
    [[1, 1, 0, 0], [1, 1, 0, 1]],   # occupied slot above k_from in cell 1
    [[1, 0, 0, 0], [1, 1, 0, 0]],   # fewer active than k_from
])
def test_place_sources_rejects_non_contiguous_active_without_mutation(active_rows):
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    active[:] = torch.tensor(active_rows, dtype=torch.float32)
    params_before, active_before = _snapshot(params, active)
    # Act / Assert
    with pytest.raises(ValueError, match="contigu"):
        _place(params, active, t, cells, [2, 2], [4, 4])
    assert torch.equal(params, params_before) and torch.equal(active, active_before)


@pytest.mark.unit
@pytest.mark.parametrize("k_from, k_to", [
    ([0, 0, 0], [1, 1, 1]),          # wrong length
    ([[0, 0]], [[1, 1]]),            # wrong ndim
    ([-1, 0], [1, 1]),               # negative k_from
    ([0, 0], [1, KMAX + 1]),         # k_to above capacity
    ([KMAX + 1, 0], [1, 1]),         # k_from above capacity
    ([0.0, 0.0], [1.0, 1.0]),        # float dtype
])
def test_place_sources_rejects_invalid_bounds_without_mutation(k_from, k_to):
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    params_before, active_before = _snapshot(params, active)
    # Act / Assert
    with pytest.raises(ValueError):
        _place(params, active, t, cells, k_from, k_to)
    assert torch.equal(params, params_before) and torch.equal(active, active_before)


@pytest.mark.unit
def test_place_sources_rejects_non_array_bounds():
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    # Act / Assert
    with pytest.raises(ValueError):
        vsf.place_sources(params, active, t, cells, [0, 0], [1, 1], "gauss", None)


@pytest.mark.unit
def test_place_sources_allows_k_to_below_k_from_as_skip():
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    active[0, :3] = 1
    params_before = params.clone()
    # Act
    _place(params, active, t, cells, [3, 0], [1, 2])
    # Assert
    assert active.sum(1).tolist() == [3, 2]
    assert torch.equal(params[0], params_before[0])


@pytest.mark.unit
def test_place_sources_diffusion_without_table_and_active_sources_raises_before_mutation():
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells, "diffusion")
    active[:, 0] = 1
    params_before, active_before = _snapshot(params, active)
    # Act / Assert
    with pytest.raises(ValueError, match="table"):
        _place(params, active, t, cells, [1, 1], [2, 2], "diffusion")
    assert torch.equal(params, params_before) and torch.equal(active, active_before)


@pytest.mark.unit
def test_optimise_leaves_input_params_unmodified():
    # Arrange
    cells = _cells(2)
    t, params, active = _fresh(cells)
    _place(params, active, t, cells, [0, 0], [2, 2])
    copy = params.clone()
    # Act
    out = vsf.optimise(params, active, t, "gauss", None, 3, lr=0.05, lr_schedule=None)
    # Assert
    assert torch.equal(params, copy)
    assert not params.requires_grad
    assert not torch.equal(out, copy)


@pytest.mark.unit
def test_place_sources_initial_radius_depends_on_k_to():
    # Arrange: same cell content, same rng seed, only k_to differs
    base = build_cell(0, size=24)
    cells = [base, dict(base)]
    t, params, active = _fresh(cells)
    # Act
    _place(params, active, t, cells, [0, 0], [1, 4], rng=np.random.default_rng(7))
    # Assert: r0 = inv_softplus(0.5 * rho - 0.5), rho = clip(0.7*sqrt(area/(pi K)), 2, 40) * jitter
    assert float(params[0, 0, 3]) != float(params[1, 0, 3])
    assert float(params[0, 0, 3]) > float(params[1, 0, 3])
    assert float(params[0, 0, 3]) == float(params[0, 0, 4])


@pytest.mark.unit
def test_place_sources_initial_radius_matches_formula_without_rng():
    # Arrange
    base = build_cell(0, size=24)
    cells = [base, dict(base)]
    t, params, active = _fresh(cells)
    # Act
    _place(params, active, t, cells, [0, 0], [1, 4])
    # Assert
    for b, K in enumerate((1, 4)):
        rho = float(np.clip(0.7 * np.sqrt(base["area"] / (np.pi * K)), 2.0, 40.0))
        want = vsf.inv_softplus(max(0.5 * rho - 0.5, 0.1))
        assert float(params[b, 0, 3]) == pytest.approx(want, rel=1e-5)
