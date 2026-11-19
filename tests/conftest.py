"""Fixtures communes : CPU forcé, graines fixées, cellules et table de profils synthétiques."""
import math

import numpy as np
import pytest
import torch

RATE_STEP = 0.12  # quantification du taux (mm/h), comme les données radar
MIN_RATE = 1.0  # en dessous : mis à zéro (hors cellule)
DEFAULT_SIZE = 24
DEFAULT_BLOBS = ((12.0, 12.0, 6.0, 4.0, 3.0), (6.0, 15.0, 3.0, 2.5, 2.5))  # (x, y, amplitude, sx, sy)
N_THREADS = 2


def build_cell(cell_id, blobs=DEFAULT_BLOBS, size=DEFAULT_SIZE, bin_name="50-200"):
    """Cellule synthétique : somme de blobs gaussiens, quantifiée, valeurs < MIN_RATE annulées."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    field = np.zeros((size, size))
    for x, y, amp, sx, sy in blobs:
        field += amp * np.exp(-0.5 * (((xx - x) / sx) ** 2 + ((yy - y) / sy) ** 2))
    field = np.round(field / RATE_STEP) * RATE_STEP
    field[field < MIN_RATE] = 0
    rate = field.astype(np.float32)
    return dict(tag="syn", date="2016-01-01", area=int((rate > 0).sum()), rate=rate, y0=0, x0=0,
                cy=size / 2, cx=size / 2, bin=bin_name, id=cell_id)


def build_synthetic_lut(n_tau=3, n_w=4, n_r=32):
    """Table analytique (n_tau, n_w, n_r) : gaussienne de largeur croissante en tau et w, nulle au dernier bin."""
    r = torch.linspace(0, 1, n_r)
    tau = torch.linspace(0, 1, n_tau)[:, None, None]
    w = torch.linspace(0.02, 0.5, n_w)[None, :, None]
    sigma = 0.15 + 0.25 * tau + 0.3 * w
    lut = torch.exp(-r[None, None, :] ** 2 / (2 * sigma ** 2))
    lut = lut / lut.max(-1, keepdim=True).values
    lut[..., -1] = 0
    return lut


@pytest.fixture(autouse=True, scope="session")
def _force_cpu_and_seed():
    torch.set_num_threads(N_THREADS)
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    import variable_source_fit as vsf
    monkeypatch.setattr(vsf, "DEVICE", torch.device("cpu"))
    torch.manual_seed(0)


@pytest.fixture
def make_cell():
    return build_cell


@pytest.fixture
def synthetic_lut():
    return build_synthetic_lut()
