"""Insertion de sources aux pics du résidu après l'encodeur (E7, avant raffinement)."""
import numpy as np
import pytest
import torch

import variable_source_fit as vsf
from encoder_insertion import insert_sources, insertion_targets
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from source_encoder import EncoderConfig

CFG = EncoderConfig(base_ch=4, k_train=16, cap=16)
PEAK_TOL_PX = 2.0


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    return load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("l"))["lut"][None, None]


def _limit(area):
    return min(CFG.cap, max(1, int(CFG.compression_frac * area / CFG.n_par)))


@pytest.mark.unit
def test_fill_to_limit_targets_the_compression_limit():
    areas = [70, 140, 700]
    out = insertion_targets([1, 3, 2], areas, CFG, factor=None)
    assert out.tolist() == [_limit(a) for a in areas]


@pytest.mark.unit
def test_factor_scales_k_rounding_up_and_caps_at_the_limit():
    out = insertion_targets([4, 10, 2], [700, 140, 700], CFG, factor=1.5)
    assert out.tolist() == [6, _limit(140), 3]


@pytest.mark.unit
def test_factor_product_is_not_bumped_by_float_rounding():
    out = insertion_targets([10], [10_000], EncoderConfig(base_ch=4, k_train=16, cap=128), factor=1.1)
    assert out.tolist() == [11]


@pytest.mark.unit
def test_target_never_drops_below_encoder_k_and_empty_cells_get_one_source():
    out = insertion_targets([0, 20], [70, 70], CFG, factor=1.5)
    assert out.tolist() == [1, 20]


@pytest.mark.unit
@pytest.mark.parametrize("factor", [0.5, 0.0, float("nan")])
def test_factor_below_one_or_not_finite_raises(factor):
    with pytest.raises(ValueError):
        insertion_targets([2], [140], CFG, factor=factor)


@pytest.mark.unit
def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        insertion_targets([2, 3], [140], CFG, factor=None)


def _encoded(cells, ks):
    kmax = max(1, max(ks))
    params = torch.zeros(len(cells), kmax, 8)
    active = torch.zeros(len(cells), kmax)
    for b, k in enumerate(ks):
        params[b, :k, 0] = 3.0 + torch.arange(k, dtype=torch.float32)
        params[b, :k, 1] = 3.0
        params[b, :k, 2:5] = 0.5
        active[b, :k] = 1
    return params, active


@pytest.mark.unit
def test_insertion_keeps_encoder_sources_and_leaves_inputs_untouched(make_cell, lut5):
    cells = [make_cell(0, size=24), make_cell(1, size=24)]
    params, active = _encoded(cells, [2, 1])
    before_p, before_a = params.clone(), active.clone()
    new_p, new_a = insert_sources(params, active, vsf.make_targets(cells), cells, np.array([4, 3]), lut5,
                                  np.random.default_rng(0))
    assert torch.equal(params, before_p) and torch.equal(active, before_a)
    assert new_p.shape[1] == 4
    assert torch.equal(new_p[0, :2], params[0, :2]) and torch.equal(new_p[1, :1], params[1, :1])
    assert new_a.sum(1).tolist() == [4.0, 3.0]
    assert torch.equal(new_a > 0, torch.arange(4)[None, :] < torch.tensor([4, 3])[:, None])


@pytest.mark.unit
def test_no_extra_sources_returns_an_equal_copy(make_cell, lut5):
    cells = [make_cell(0, size=24)]
    params, active = _encoded(cells, [2])
    new_p, new_a = insert_sources(params, active, vsf.make_targets(cells), cells, np.array([2]), lut5, None)
    assert torch.equal(new_p, params) and torch.equal(new_a, active)
    assert new_p is not params and new_a is not active


@pytest.mark.unit
def test_first_inserted_source_sits_on_the_unexplained_peak(make_cell, lut5):
    cell = make_cell(0, blobs=((16.0, 8.0, 6.0, 2.5, 2.5),), size=24)
    params, active = _encoded([cell], [0])
    new_p, new_a = insert_sources(params, active, vsf.make_targets([cell]), [cell], np.array([1]), lut5, None)
    assert new_a.sum() == 1
    assert abs(float(new_p[0, 0, 0]) - 16.0) <= PEAK_TOL_PX and abs(float(new_p[0, 0, 1]) - 8.0) <= PEAK_TOL_PX


@pytest.mark.unit
def test_targets_below_current_k_raise(make_cell, lut5):
    cells = [make_cell(0, size=24)]
    params, active = _encoded(cells, [3])
    with pytest.raises(ValueError):
        insert_sources(params, active, vsf.make_targets(cells), cells, np.array([2]), lut5, None)
