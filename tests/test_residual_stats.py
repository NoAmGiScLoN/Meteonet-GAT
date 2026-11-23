"""Tests de src/residual_stats.py : statistiques de residu par cellule (domaine racine)."""
import dataclasses

import numpy as np
import pytest
from scipy import ndimage as ndi

import cell_delimitation
import residual_stats as rs

SIZE = 40
SQUARE = slice(10, 30)


def _square_rate(value: float = 4.0) -> np.ndarray:
    rate = np.zeros((SIZE, SIZE))
    rate[SQUARE, SQUARE] = value
    return rate


def _textured_rate() -> np.ndarray:
    rng = np.random.default_rng(0)
    rate = np.zeros((SIZE, SIZE))
    rate[SQUARE, SQUARE] = rng.uniform(1.0, 9.0, size=(20, 20))
    return rate


def _power_law_field(n: int, exponent: float, seed: int = 0) -> np.ndarray:
    """Bruit blanc filtre : amplitude k^(-exponent/2) -> PSD ~ k^-exponent."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    k = np.hypot(fy, fx)
    k[0, 0] = np.inf
    spectrum = np.fft.fft2(rng.standard_normal((n, n))) * k ** (-exponent / 2)
    return np.fft.ifft2(spectrum).real


def test_identical_fields_give_zero_stats():
    rate = _textured_rate()
    stats = rs.compute_residual_stats(rate.copy(), rate)
    assert stats.peak_bias == pytest.approx(0.0)
    assert stats.mass_bias == pytest.approx(0.0)
    assert stats.spill_frac == pytest.approx(0.0)
    assert stats.resid_std_in == pytest.approx(0.0)
    assert stats.resid_std_edge == pytest.approx(0.0)
    assert stats.nrmse_mask == pytest.approx(0.0)


def test_scaled_reconstruction_gives_signed_biases():
    rate = _textured_rate()
    stats = rs.compute_residual_stats(1.1 * rate, rate)
    assert stats.mass_bias == pytest.approx(0.1)
    assert stats.peak_bias == pytest.approx(0.1)
    assert stats.spill_frac == pytest.approx(0.0)


def test_underestimation_gives_negative_bias():
    rate = _textured_rate()
    stats = rs.compute_residual_stats(0.5 * rate, rate)
    assert stats.mass_bias == pytest.approx(-0.5)
    assert stats.peak_bias == pytest.approx(-0.5)


def test_spill_ring_gives_exact_spill_fraction():
    rate = _square_rate(4.0)
    ring_value = 0.5
    mask = rs.cell_mask(rate)
    ring = rs.edge_band(mask, 1) & ~mask
    rec = rate + ring_value * ring
    stats = rs.compute_residual_stats(rec, rate)
    assert stats.spill_frac == pytest.approx(ring_value * ring.sum() / rate.sum())
    assert stats.mass_bias == pytest.approx(stats.spill_frac)


def test_negative_reconstruction_is_clipped():
    rate = _square_rate()
    rec = rate.copy()
    rec[0, 0] = -0.3
    stats = rs.compute_residual_stats(rec, rate)
    assert stats.spill_frac == pytest.approx(0.0)
    assert stats.mass_bias == pytest.approx(0.0)


def test_perturbation_on_edge_band_only():
    rate = _square_rate(4.0)
    mask = rs.cell_mask(rate)
    band = rs.edge_band(mask, rs.EDGE_BAND_PX)
    interior = mask & ~band
    rec = rate.copy()
    rec[band & mask] = 6.0
    stats = rs.compute_residual_stats(rec, rate)
    resid = np.sqrt(rec) - np.sqrt(rate)
    assert np.std(resid[interior]) == pytest.approx(0.0)
    assert stats.resid_std_edge > 0
    assert stats.resid_std_edge > stats.resid_std_in > 0


def test_edge_band_straddles_mask_edge():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    band = rs.edge_band(mask, 2)
    assert band[5, 5] and band[3, 3] and band[14, 14] and band[16, 16]
    assert not band[9, 9] and not band[0, 0] and not band[19, 19]


def test_edge_band_ignores_crop_border():
    mask = np.ones((10, 10), dtype=bool)
    assert not rs.edge_band(mask, 2).any()


def test_power_law_slope_is_recovered():
    field = _power_law_field(128, 3.0)
    mask = np.ones(field.shape, dtype=bool)
    assert rs.radial_psd_slope(field, mask) == pytest.approx(-3.0, abs=0.4)


def test_white_noise_slope_is_near_zero():
    field = np.random.default_rng(1).standard_normal((128, 128))
    mask = np.ones(field.shape, dtype=bool)
    assert rs.radial_psd_slope(field, mask) == pytest.approx(0.0, abs=0.4)


def test_tiny_mask_gives_nan_slope():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:5 + rs.SPECTRAL_MIN_PX - 1, 5:30] = True
    field = np.random.default_rng(2).standard_normal(mask.shape)
    assert np.isnan(rs.radial_psd_slope(field, mask))


def test_few_bins_gives_nan_slope(monkeypatch):
    monkeypatch.setattr(rs, "SPECTRAL_MIN_BINS", 10_000)
    mask = np.ones((32, 32), dtype=bool)
    field = np.random.default_rng(3).standard_normal(mask.shape)
    assert np.isnan(rs.radial_psd_slope(field, mask))


def test_slope_does_not_use_values_outside_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 10:50] = True
    field = np.random.default_rng(4).standard_normal(mask.shape)
    other = field.copy()
    other[~mask] += 100.0
    assert rs.radial_psd_slope(field, mask) == rs.radial_psd_slope(other, mask)


def test_inputs_are_not_mutated():
    rate = _textured_rate()
    rec = 1.1 * rate
    rec[0, 0] = -0.1
    rate_copy, rec_copy = rate.copy(), rec.copy()
    rs.compute_residual_stats(rec, rate)
    np.testing.assert_array_equal(rate, rate_copy)
    np.testing.assert_array_equal(rec, rec_copy)


def test_as_dict_keys_match_stat_columns():
    rate = _textured_rate()
    stats = rs.compute_residual_stats(1.05 * rate, rate)
    as_dict = stats.as_dict()
    assert tuple(as_dict) == rs.STAT_COLUMNS
    assert tuple(dataclasses.asdict(stats)) == rs.STAT_COLUMNS
    assert all(isinstance(v, float) for v in as_dict.values())


def test_stats_are_frozen():
    stats = rs.compute_residual_stats(_square_rate(), _square_rate())
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.peak_bias = 1.0  # type: ignore[misc]


def test_slope_is_finite_on_large_cell():
    rate = np.zeros((70, 70))
    rate[5:65, 5:65] = np.random.default_rng(5).uniform(1.0, 9.0, size=(60, 60))
    stats = rs.compute_residual_stats(rate * 1.2, rate)
    assert np.isfinite(stats.spectral_slope)


@pytest.mark.parametrize("make_bad", [
    lambda rec, rate: (rec[:-1], rate),
    lambda rec, rate: (np.where(rec > 0, np.nan, rec), rate),
    lambda rec, rate: (rec, np.where(rate > 0, np.inf, rate)),
    lambda rec, rate: (rec, -rate),
    lambda rec, rate: (rec, np.zeros_like(rate)),
    lambda rec, rate: (rec[0], rate[0]),
])
def test_invalid_inputs_raise_value_error(make_bad):
    rate = _square_rate()
    rec, bad_rate = make_bad(rate.copy(), rate.copy())
    with pytest.raises(ValueError):
        rs.compute_residual_stats(rec, bad_rate)


def test_stat_columns_are_pinned():
    assert rs.STAT_COLUMNS == (
        "peak_bias", "mass_bias", "resid_std_in", "resid_std_edge",
        "spill_frac", "spectral_slope", "nrmse_mask",
    )


def test_struct_8_matches_cell_delimitation():
    np.testing.assert_array_equal(rs.STRUCT_8, cell_delimitation.STRUCT_8)


def test_identical_fields_give_nan_slope():
    rate = np.zeros((70, 70))
    rate[5:65, 5:65] = np.random.default_rng(5).uniform(1.0, 9.0, size=(60, 60))
    stats = rs.compute_residual_stats(rate.copy(), rate)
    assert np.isnan(stats.spectral_slope)


def test_zero_field_gives_nan_slope():
    mask = np.ones((32, 32), dtype=bool)
    assert np.isnan(rs.radial_psd_slope(np.zeros(mask.shape), mask))


# Mesure (4 graines, 128x128, k^-3) : pleine boite ~ -2.8..-3.15 ; disque inscrit : decalage
# |disque - boite| < 0.005. Tolerance documentee large (0.3) pour le biais masque/Hann.
DISC_SHIFT_TOL = 0.3


def test_disc_mask_slope_stays_negative_and_close_to_full_box():
    n = 128
    yy, xx = np.mgrid[:n, :n]
    disc = (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (n / 2 - 1) ** 2
    full = np.ones((n, n), dtype=bool)
    field = _power_law_field(n, 3.0)
    slope_disc = rs.radial_psd_slope(field, disc)
    slope_full = rs.radial_psd_slope(field, full)
    assert slope_disc < 0
    assert abs(slope_disc - slope_full) < DISC_SHIFT_TOL


def test_non_square_bbox_slope_is_recovered():
    field = _power_law_field(128, 3.0)[:48, :]
    mask = np.ones(field.shape, dtype=bool)
    assert rs.radial_psd_slope(field, mask) == pytest.approx(-3.0, abs=0.5)


def test_resolution_independent_slope():
    """Meme loi k^-3 echantillonnee a n et 2n : pentes proches (tolerance 0.5)."""
    slopes = [
        rs.radial_psd_slope(f, np.ones(f.shape, dtype=bool))
        for f in (_power_law_field(64, 3.0), _power_law_field(128, 3.0))
    ]
    assert slopes[0] == pytest.approx(slopes[1], abs=0.5)


def test_interpolated_upsampling_steepens_slope():
    """Biais documente : un x2 par interpolation vide le haut des k -> pente plus raide (~ -4.7)."""
    base = _power_law_field(128, 3.0)[:64, :64]
    up = ndi.zoom(base, 2, order=1)
    slope_base = rs.radial_psd_slope(base, np.ones(base.shape, dtype=bool))
    slope_up = rs.radial_psd_slope(up, np.ones(up.shape, dtype=bool))
    assert slope_up < slope_base - 1.0


def test_mask_filling_crop_gives_nan_edge_std_and_float_dict():
    rate = np.random.default_rng(6).uniform(1.0, 9.0, size=(24, 24))
    stats = rs.compute_residual_stats(1.1 * rate, rate)
    assert np.isnan(stats.resid_std_edge)
    values = stats.as_dict()
    assert all(isinstance(v, float) for v in values.values())
    assert np.isnan(values["resid_std_edge"])


def test_mask_touching_crop_border_gives_finite_stats():
    rate = np.zeros((40, 40))
    rate[:20, :20] = np.random.default_rng(7).uniform(1.0, 9.0, size=(20, 20))
    stats = rs.compute_residual_stats(1.1 * rate, rate)
    assert all(np.isfinite(v) for v in stats.as_dict().values())


def test_one_pixel_wide_mask_is_all_edge_band():
    rate = np.zeros((20, 20))
    rate[10, 4:16] = np.linspace(1.0, 5.0, 12)
    mask = rs.cell_mask(rate)
    assert (rs.edge_band(mask, rs.EDGE_BAND_PX) & mask).sum() == mask.sum()
    stats = rs.compute_residual_stats(1.2 * rate, rate)
    assert np.isfinite(stats.resid_std_in) and np.isfinite(stats.resid_std_edge)
    assert np.isfinite(stats.nrmse_mask) and np.isfinite(stats.mass_bias)
    assert np.isnan(stats.spectral_slope)


def test_negative_rec_inside_mask_is_clipped_consistently():
    rate = _textured_rate()
    rec = 1.1 * rate
    rec[15, 15] = -2.0
    clipped = rec.copy()
    clipped[15, 15] = 0.0
    got = rs.compute_residual_stats(rec, rate)
    expected = rs.compute_residual_stats(clipped, rate)
    assert got == expected
    mask = rs.cell_mask(rate)
    resid = np.sqrt(clipped) - np.sqrt(rate)
    assert got.resid_std_in == pytest.approx(np.std(resid[mask]))
    assert got.nrmse_mask == pytest.approx(
        np.linalg.norm(resid[mask]) / np.linalg.norm(np.sqrt(rate[mask]))
    )


@pytest.mark.parametrize("bad", [np.full((4, 4, 4), 1.0), np.array([1.0, 2.0])])
def test_non_2d_rate_raises(bad):
    with pytest.raises(ValueError, match="2-D"):
        rs.compute_residual_stats(bad, bad)
