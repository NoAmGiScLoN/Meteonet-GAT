"""Tests du plancher de bruit de la nrmse en domaine racine."""
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import ndimage as ndi

import noise_floor as nf
import variable_source_fit as vsf

FIELD_SIZE = 64
MC_REL_TOL = 0.15
ATOL_FLOAT32 = 1e-5


def quantized_field(seed=0, size=FIELD_SIZE):
    rng = np.random.default_rng(seed)
    steps = rng.integers(9, 200, size=(size, size))  # >= 9 * 0.12 = 1.08
    field = steps * nf.RATE_STEP
    field[rng.random((size, size)) < 0.3] = 0.0
    return field.astype(np.float32)


def test_nrmse_matches_fitter_metrics():
    rng = np.random.default_rng(1)
    t = rng.random((20, 20)).astype(np.float32) * 10
    r = rng.normal(3, 4, (20, 20)).astype(np.float32)
    valid = torch.ones(1, 20, 20, dtype=torch.bool)
    ref = vsf.metrics(torch.from_numpy(r)[None], torch.from_numpy(t)[None], valid)[0]["nrmse"]
    assert nf.nrmse_sqrt(t, r) == pytest.approx(ref, abs=ATOL_FLOAT32)


def test_nrmse_zero_on_identical():
    t = quantized_field()
    assert nf.nrmse_sqrt(t, t.copy()) == 0.0


def test_nrmse_rejects_bad_input():
    with pytest.raises(ValueError):
        nf.nrmse_sqrt(np.ones((3, 3)), np.ones((3, 4)))
    with pytest.raises(ValueError):
        nf.nrmse_sqrt(np.zeros((3, 3)), np.ones((3, 3)))


def test_nrmse_does_not_mutate_inputs():
    t, r = quantized_field(), -np.ones((FIELD_SIZE, FIELD_SIZE), np.float32)
    t0, r0 = t.copy(), r.copy()
    nf.nrmse_sqrt(t, r)
    assert np.array_equal(t, t0) and np.array_equal(r, r0)


def test_mc_close_to_analytic():
    f = quantized_field()
    mc = nf.quantization_floor_mc(f, np.random.default_rng(0))
    an = nf.quantization_floor_analytic(f)
    assert mc == pytest.approx(an, rel=MC_REL_TOL)


def test_mc_reproducible_and_validates_n():
    f = quantized_field()
    a = nf.quantization_floor_mc(f, np.random.default_rng(3))
    b = nf.quantization_floor_mc(f, np.random.default_rng(3))
    assert a == b
    with pytest.raises(ValueError):
        nf.quantization_floor_mc(f, np.random.default_rng(0), n=0)


def test_analytic_rejects_empty_field():
    with pytest.raises(ValueError):
        nf.quantization_floor_analytic(np.zeros((4, 4), np.float32))


def blob_on_zero(size=64, amp=20.0, blob_sigma=8.0, radius=20):
    yy, xx = np.mgrid[:size, :size] - size // 2
    r2 = yy ** 2 + xx ** 2
    field = amp * np.exp(-r2 / (2 * blob_sigma ** 2))
    field[r2 > radius ** 2] = 0.0
    return field


def padded_constant(value=4.0, size=24, pad=6):
    field = np.zeros((size, size))
    field[pad:-pad, pad:-pad] = value
    return field


def test_filter_mode_is_nearest_and_differs_from_constant_on_nonzero_border():
    assert nf.FILTER_MODE == "nearest"
    const = np.full((16, 16), 4.0)
    nearest = ndi.gaussian_filter(const, 1.0, mode="nearest")
    constant = ndi.gaussian_filter(const, 1.0, mode="constant")
    assert not np.allclose(nearest, constant)
    assert nf.smoothing_floor_full(const, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert nf.speckle_floor_full(const) == pytest.approx(0.0, abs=1e-6)


def test_full_floors_increase_with_sigma():
    rng = np.random.default_rng(2)
    noisy = 5 + rng.random((32, 32)) * 5
    assert nf.smoothing_floor_full(noisy, 1.0) > nf.smoothing_floor_full(noisy, 0.5) > 0


def test_full_floor_sees_cell_edge_but_interior_does_not():
    cell = padded_constant()
    assert nf.smoothing_floor_full(cell, 1.0) > 0.2
    assert nf.smoothing_floor_interior(cell, 1.0, 1) == pytest.approx(0.0, abs=1e-9)
    assert nf.speckle_floor_interior(cell, 1) == pytest.approx(0.0, abs=1e-9)
    assert nf.speckle_floor_full(cell) > 0


def test_smooth_blob_interior_floor_much_smaller_than_full():
    blob = blob_on_zero()
    for n in nf.EROSION_ITERS:
        for s in nf.SMOOTH_SIGMAS:
            interior = nf.smoothing_floor_interior(blob, s, n)
            assert interior < 0.02
            assert interior < 0.2 * nf.smoothing_floor_full(blob, s)


def test_interior_floor_detects_texture():
    rng = np.random.default_rng(5)
    textured = padded_constant(size=40, pad=4) * (1 + rng.random((40, 40)))
    assert nf.smoothing_floor_interior(textured, 1.0, 1) > 0.05
    spiky = padded_constant(size=24, pad=4)
    spiky[12, 12] = 40.0
    assert nf.speckle_floor_interior(spiky, 1) > 0


def test_interior_floor_nan_on_tiny_cell():
    tiny = np.zeros((9, 9))
    tiny[4, 4] = 3.0
    assert np.isnan(nf.smoothing_floor_interior(tiny, 1.0, 1))
    assert np.isnan(nf.speckle_floor_interior(tiny, 1))
    small = np.zeros((9, 9))
    small[3:6, 3:6] = 3.0
    assert not np.isnan(nf.speckle_floor_interior(small, 1))
    assert np.isnan(nf.speckle_floor_interior(small, 2))


def test_floor_functions_validate_args():
    f = padded_constant()
    with pytest.raises(ValueError):
        nf.smoothing_floor_full(f, 0)
    with pytest.raises(ValueError):
        nf.smoothing_floor_interior(f, 0, 1)
    with pytest.raises(ValueError):
        nf.smoothing_floor_interior(f, 1.0, 0)
    with pytest.raises(ValueError):
        nf.speckle_floor_interior(f, 0)
    with pytest.raises(ValueError):
        nf.speckle_floor_full(f, size=0)


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_rejects_nan_or_negative_rates(bad):
    f = padded_constant()
    f[10, 10] = bad
    rng = np.random.default_rng(0)
    for call in (
        lambda: nf.nrmse_sqrt(f, np.ones_like(f)),
        lambda: nf.quantization_floor_mc(f, rng),
        lambda: nf.quantization_floor_analytic(f),
        lambda: nf.smoothing_floor_full(f, 1.0),
        lambda: nf.smoothing_floor_interior(f, 1.0, 1),
        lambda: nf.speckle_floor_full(f),
        lambda: nf.speckle_floor_interior(f, 1),
    ):
        with pytest.raises(ValueError):
            call()


def analytic_over_mc(value):
    field = np.full((200, 200), value)
    mc = nf.quantization_floor_mc(field, np.random.default_rng(0), n=50)
    return nf.quantization_floor_analytic(field, min_rate=0.0) / mc


def test_analytic_deviates_from_mc_at_small_rates_but_not_above_min_rate():
    # Measured: the delta method does NOT overestimate here, it slightly UNDERestimates the MC
    # (ratio ~0.97 at t = 0.12), and converges to 1 as t grows.
    small = [analytic_over_mc(t) for t in (0.12, 0.2, 0.5)]
    assert small == sorted(small)  # deviation shrinks as t grows
    assert 1 - small[0] > 0.015
    assert small[0] > 0.9
    assert abs(1 - analytic_over_mc(nf.ANALYTIC_MIN_RATE)) < 0.005
    with pytest.raises(ValueError):
        nf.quantization_floor_analytic(np.full((8, 8), 0.5))  # no pixel >= ANALYTIC_MIN_RATE


def test_analytic_min_rate_constant_and_restriction():
    assert nf.ANALYTIC_MIN_RATE == pytest.approx(9 * nf.RATE_STEP)
    f = quantized_field()
    mixed = f.copy()
    mixed[0, 0] = nf.RATE_STEP  # tiny in-cell pixel, ignored by default
    assert nf.quantization_floor_analytic(mixed) == pytest.approx(
        nf.quantization_floor_analytic(np.where(mixed == nf.RATE_STEP, 0.0, mixed)))


def test_mc_default_rng_is_reproducible():
    f = quantized_field()
    assert nf.quantization_floor_mc(f) == nf.quantization_floor_mc(f)


def test_floor_columns_built_from_constants():
    expected = {"quant_mc", "quant_analytic", "speckle_full"}
    for s in nf.SMOOTH_SIGMAS:
        expected.add(f"smooth_s{s}_full")
        expected.update(f"smooth_s{s}_int{n}" for n in nf.EROSION_ITERS)
    expected.update(f"speckle_int{n}" for n in nf.EROSION_ITERS)
    assert set(nf.FLOOR_COLUMNS) == expected
    assert len(nf.FLOOR_COLUMNS) == len(expected)


def synthetic_sample():
    return [dict(tag=f"t{i}", date="2016-01-01", area=10, rate=quantized_field(i, 16), y0=0, x0=0,
                 cy=8, cx=8, bin=b) for i, b in enumerate(["a", "a", "b"])]


def test_floors_table_and_summary():
    sample = synthetic_sample()
    snapshot = [c["rate"].copy() for c in sample]
    df = nf.floors_table(sample, name="s1")
    assert len(df) == 3
    assert set(nf.FLOOR_COLUMNS) <= set(df.columns)
    assert all(np.array_equal(c["rate"], s) for c, s in zip(sample, snapshot))
    summ = nf.summarize_floors(df)
    assert len(summ) == 2
    assert "quant_mc_median" in summ.columns and "speckle_int1_p90" in summ.columns


def test_main_end_to_end(tmp_path):
    p = tmp_path / "sample_x.pkl"
    p.write_bytes(pickle.dumps(synthetic_sample()))
    out = tmp_path / "out"
    nf.main(["--samples", str(p), "--out", str(out)])
    cells = pd.read_csv(out / "noise_floor_cells.csv")
    assert len(cells) == 3
    assert (out / "noise_floor_summary.csv").exists()


def test_floors_table_validates_cell_schema():
    with pytest.raises(ValueError, match="area"):
        nf.floors_table([dict(tag="a", bin="x", rate=quantized_field())])


def test_floors_table_rejects_bad_rate():
    cell = dict(tag="a", bin="x", area=1, rate=np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        nf.floors_table([cell])


def test_floors_table_logs_empty_interiors(caplog):
    rate = np.zeros((9, 9))
    rate[4, 4] = 3.0
    with caplog.at_level("WARNING"):
        df = nf.floors_table([dict(tag="a", bin="x", area=1, rate=rate)])
    assert df["speckle_int1"].isna().all()
    assert "empty" in caplog.text.lower()


def test_summarize_warns_on_small_groups(caplog):
    df = nf.floors_table(synthetic_sample(), name="s1")
    with caplog.at_level("WARNING"):
        nf.summarize_floors(df)
    assert "fewer than" in caplog.text.lower()
