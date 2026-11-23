"""Conversion brut MeteoNet (int16) -> mm/h : pas de débordement int16 au-delà de 2730 (R * 12 > 32767)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import find_objects

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cell_dataset as cd  # noqa: E402
import make_figures as mf  # noqa: E402
import pipeline_figure as pf  # noqa: E402
import variable_source_fit as vsf  # noqa: E402
from cell_delimitation import STRUCT_8, segment  # noqa: E402


def frame(raw):
    R = np.zeros((40, 40), np.int16)
    R[10:20, 10:20] = raw
    return R


def crop_rate(raw):
    R = frame(raw)
    labels, n, _ = segment(R, STRUCT_8)
    sl = find_objects(labels)[0]
    return vsf.crop_cell(R, labels, 1, sl, 100, "t", "d")["rate"]


def test_crop_cell_no_overflow_above_2730():
    assert np.allclose(crop_rate(3000).max(), 360.0)


def test_crop_cell_unchanged_below_2730():
    assert crop_rate(2730).max() == np.float32(2730 * 12 / 100)
    assert crop_rate(100).max() == np.float32(12.0)


def test_cell_dataset_rate_helper_no_overflow():
    R = frame(3000)
    assert np.allclose(cd.raw_to_rate(R).max(), 360.0)
    assert cd.raw_to_rate(frame(100)).max() == np.float32(12.0)


def test_pipeline_figure_rate_map_no_overflow():
    R = np.array([[0, 100, 3000, -1]], np.int16)
    out = pf.rate_map(R)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 3])
    assert np.allclose(out[0, 1:3], [12.0, 360.0])


@pytest.mark.parametrize("raw", [2730, 2731, 3000, 32767])
def test_boundary_values_around_int16_overflow(raw):
    expected = np.float32(raw * 12 / 100)
    assert crop_rate(raw).max() == expected
    assert cd.raw_to_rate(frame(raw)).max() == raw * 12 / 100


def test_make_figures_rate_map_no_overflow():
    R = np.array([[0, 100, 3000, -1]], np.int16)
    out = mf.rate_map(R)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 3])
    assert np.allclose(out[0, 1:3], [12.0, 360.0])
