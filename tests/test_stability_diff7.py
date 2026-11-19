"""Descripteurs de stabilité pour le bras diff7 : w émis, tau absent."""
import math

import numpy as np
import pytest
import torch

import stability_test as st

CELL = dict(area=math.pi * 100, cx=10.0, cy=8.0, x0=0, y0=0)  # r_eq = 10


def hand_made_params(n_cols):
    p = torch.zeros(2, n_cols)
    p[:, 0], p[:, 1] = 10.0, 8.0
    p[1, 7 if n_cols == 8 else 0] = 0.0
    if n_cols == 8:
        p[:, 6] = 3.0  # tau brut : doit être ignoré pour diff7
        p[:, 7] = torch.tensor([0.0, 2.0])
    return p


@pytest.mark.unit
def test_names_for_diff7_list_w_and_not_tau():
    assert st.NAMES["diff7"] == ["x", "y", "logA", "log_a", "log_b", "angle", "w"]


@pytest.mark.unit
def test_descriptors_diff7_emit_seven_columns_with_w_last():
    # Arrange
    params, active = hand_made_params(8), torch.ones(2)
    # Act
    d = st.descriptors(params, active, CELL, "diff7")
    # Assert
    assert d.shape == (2, 7)
    assert d[:, 6] == pytest.approx(torch.sigmoid(torch.tensor([0.0, 2.0])).numpy())
    assert len(st.NAMES["diff7"]) == d.shape[1]


@pytest.mark.unit
def test_descriptors_diff7_ignore_tau_column():
    params, active = hand_made_params(8), torch.ones(2)
    other = params.clone()
    other[:, 6] = -5.0
    assert np.array_equal(st.descriptors(params, active, CELL, "diff7"), st.descriptors(other, active, CELL, "diff7"))


@pytest.mark.unit
def test_descriptors_diffusion_still_emits_tau_and_w():
    d = st.descriptors(hand_made_params(8), torch.ones(2), CELL, "diffusion")
    assert d.shape == (2, 8)
    assert d[0, 6] == pytest.approx(float(torch.sigmoid(torch.tensor(3.0))))


@pytest.mark.unit
def test_descriptors_gauss_unchanged():
    assert st.descriptors(hand_made_params(6), torch.ones(2), CELL, "gauss").shape == (2, 6)
