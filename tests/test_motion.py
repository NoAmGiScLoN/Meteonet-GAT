"""Tests de l'estimation de mouvement (motion) : corrélation de phase, grille, lissage, prédiction."""
import time
from dataclasses import FrozenInstanceError
from datetime import datetime

import numpy as np
import pytest

from cell_delimitation import DATA_DIR
from frame_stream import iter_frames
from motion import (GridMotion, Motion, grid_motion, motion_at, phase_correlation, predict_displacement,
                    smooth_velocity)
from tracking_synth import MISSING, rain_frame

SHAPE = (256, 256)


def _blobs(seed: int, n: int = 25, shape=SHAPE, margin: int = 40):
    """Blobs aléatoires (centres, rayons, pics) déterministes."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(margin, shape[0] - margin), rng.uniform(margin, shape[1] - margin),
             rng.uniform(6, 14), rng.uniform(6, 14), int(rng.integers(30, 150))) for _ in range(n)]


def _pair(shift, seed=0, shape=SHAPE, missing_b=None):
    """Deux cartes : la seconde est la première translatée de `shift` (dy, dx) px."""
    blobs = _blobs(seed, shape=shape)
    moved = [(cy + shift[0], cx + shift[1], ry, rx, p) for cy, cx, ry, rx, p in blobs]
    return rain_frame(shape, blobs), rain_frame(shape, moved, missing_b)


@pytest.mark.unit
@pytest.mark.parametrize("shift", [(3, -5), (2.5, 1.5), (-4, 6)])
def test_recovers_known_shift_at_full_resolution(shift):
    a, b = _pair(shift)
    m = phase_correlation(a, b, coarse=1)
    assert m.reliable
    assert m.vy == pytest.approx(shift[0], abs=0.5)
    assert m.vx == pytest.approx(shift[1], abs=0.5)


@pytest.mark.unit
@pytest.mark.parametrize("shift", [(3, -5), (6.5, 5.5), (-8, 6)])
def test_recovers_known_shift_at_coarse_4(shift):
    a, b = _pair(shift, seed=1)
    m = phase_correlation(a, b, coarse=4)
    assert m.reliable
    assert m.vy == pytest.approx(shift[0], abs=1.5)
    assert m.vx == pytest.approx(shift[1], abs=1.5)


@pytest.mark.unit
def test_missing_region_does_not_bias_estimate():
    shift = (3, -5)
    a, b = _pair(shift, seed=2, missing_b=(0, 256, 180, 256))
    m = phase_correlation(a, b, coarse=2)
    assert m.vy == pytest.approx(shift[0], abs=1.0)
    assert m.vx == pytest.approx(shift[1], abs=1.0)


@pytest.mark.unit
def test_pure_noise_is_unreliable():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 100, SHAPE).astype(np.int16)
    b = rng.integers(0, 100, SHAPE).astype(np.int16)
    assert not phase_correlation(a, b).reliable


@pytest.mark.unit
def test_empty_frames_return_null_unreliable_motion():
    z = np.zeros(SHAPE, dtype=np.int16)
    assert phase_correlation(z, z) == Motion(0.0, 0.0, 0.0, False)


@pytest.mark.unit
def test_near_empty_frame_is_unreliable():
    a, _ = _pair((2, 2))
    few = np.zeros(SHAPE, dtype=np.int16)
    few[10:13, 10:13] = 50
    assert phase_correlation(a, few) == Motion(0.0, 0.0, 0.0, False)


@pytest.mark.unit
def test_inputs_are_not_mutated():
    a, b = _pair((3, 2), missing_b=(0, 50, 0, 50))
    a0, b0 = a.copy(), b.copy()
    phase_correlation(a, b)
    grid_motion(a, b)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


@pytest.mark.unit
def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        phase_correlation(np.zeros((64, 64), np.int16), np.zeros((64, 80), np.int16))


@pytest.mark.unit
def test_motion_is_frozen():
    with pytest.raises(FrozenInstanceError):
        Motion(0.0, 0.0, 0.0, False).vy = 1.0  # type: ignore[misc]


def _split_field(seed=4, shape=(128, 256)):
    """Moitié gauche translatée de (0, +4), moitié droite de (0, -3) ; tuiles de 128."""
    rng = np.random.default_rng(seed)
    a_blobs, b_blobs = [], []
    for half, dx in ((0, 4), (1, -3)):
        for _ in range(20):
            cy, cx = rng.uniform(30, 98), rng.uniform(40, 88) + 128 * half
            r = rng.uniform(6, 12)
            p = int(rng.integers(30, 150))
            a_blobs.append((cy, cx, r, r, p))
            b_blobs.append((cy, cx + dx, r, r, p))
    return rain_frame(shape, a_blobs), rain_frame(shape, b_blobs)


@pytest.mark.unit
def test_grid_motion_recovers_each_half():
    a, b = _split_field()
    g = grid_motion(a, b, tile=128, coarse=2)
    assert isinstance(g, GridMotion) and g.tile == 128
    assert g.v.shape == (1, 2, 2) and g.reliable.shape == (1, 2)
    assert g.reliable.all()
    assert g.v[0, 0] == pytest.approx([0, 4], abs=1.0)
    assert g.v[0, 1] == pytest.approx([0, -3], abs=1.0)


@pytest.mark.unit
def test_grid_motion_unreliable_tile_falls_back_to_global():
    a, b = _pair((3, -5), seed=5, shape=(128, 256))
    a[:, 128:] = 0
    b[:, 128:] = 0  # tuile de droite vide : non fiable
    g = grid_motion(a, b, tile=128, coarse=2)
    glob = phase_correlation(a, b, coarse=2)
    assert not g.reliable[0, 1]
    assert g.v[0, 1] == pytest.approx([glob.vy, glob.vx])


@pytest.mark.unit
def test_smooth_velocity_without_previous_returns_per_step_observation():
    assert smooth_velocity(None, (4.0, -2.0), dt_steps=2) == (2.0, -1.0)


@pytest.mark.unit
def test_smooth_velocity_ema_and_dt_scaling():
    v = smooth_velocity((2.0, 0.0), (6.0, 3.0), alpha=0.25, dt_steps=3)
    assert v == pytest.approx((0.25 * 2.0 + 0.75 * 2.0, 0.75 * 0.0 + 0.25 * 1.0))
    assert smooth_velocity((2.0, 0.0), (6.0, 2.0), alpha=0.5) == pytest.approx((4.0, 1.0))
    assert smooth_velocity((2.0, 0.0), (6.0, 2.0), alpha=1.0) == pytest.approx((6.0, 2.0))


@pytest.mark.unit
@pytest.mark.parametrize("alpha,dt", [(0.0, 1), (1.5, 1), (-0.1, 1), (0.5, 0), (0.5, -2)])
def test_smooth_velocity_validation(alpha, dt):
    with pytest.raises(ValueError):
        smooth_velocity((1.0, 1.0), (1.0, 1.0), alpha=alpha, dt_steps=dt)


@pytest.mark.unit
def test_predict_displacement_scales_with_steps():
    assert predict_displacement((1.5, -2.0), 3) == (4.5, -6.0)
    with pytest.raises(ValueError):
        predict_displacement((1.5, -2.0), 0)


@pytest.mark.integration
def test_real_consecutive_frames_give_plausible_fast_motion():
    t0 = datetime(2016, 1, 1, 12, 0)
    try:
        frames = list(iter_frames(t0, datetime(2016, 1, 1, 12, 5)))
    except FileNotFoundError:
        pytest.skip("DATA/ absent")
    assert len(frames) == 2
    start = time.perf_counter()
    m = phase_correlation(frames[0].R, frames[1].R, coarse=4)
    elapsed = time.perf_counter() - start
    assert m.reliable
    assert np.hypot(m.vy, m.vx) < 15
    assert elapsed < 2.0  # test de fumée seulement : pas une mesure de performance


STATIC_HOLE = (40, 200, 60, 200)  # rectangle -1 identique dans les deux cartes, coupe la pluie


@pytest.mark.unit
@pytest.mark.parametrize("seed", [6, 7])
@pytest.mark.parametrize("coarse,tol", [(1, 0.5), (4, 1.5)])
def test_static_missing_edge_does_not_bias_toward_zero(coarse, tol, seed):
    shift = (4, -3)
    blobs = _blobs(seed, n=15)
    moved = [(cy + shift[0], cx + shift[1], ry, rx, p) for cy, cx, ry, rx, p in blobs]
    a, b = rain_frame(SHAPE, blobs, STATIC_HOLE), rain_frame(SHAPE, moved, STATIC_HOLE)
    m = phase_correlation(a, b, coarse=coarse)
    assert m.vy == pytest.approx(shift[0], abs=tol)
    assert m.vx == pytest.approx(shift[1], abs=tol)


@pytest.mark.unit
def test_zero_power_gives_finite_unreliable_motion():
    missing = np.full(SHAPE, MISSING, dtype=np.int16)
    corner = np.zeros(SHAPE, dtype=np.int16)
    corner[0:8, 0:8] = 50  # pluie sur le bord : annulée par la fenêtre de Hann
    for a, b in ((missing, missing), (corner, corner)):
        m = phase_correlation(a, b)
        assert not m.reliable
        assert all(np.isfinite([m.vy, m.vx, m.peak_ratio]))


@pytest.mark.unit
def test_shift_near_alias_limit_is_unreliable():
    blobs = [(30 + 8 * i, 20 + 6 * i, 5.0, 5.0, 80) for i in range(8)]
    moved = [(cy, cx + 60, ry, rx, p) for cy, cx, ry, rx, p in blobs]
    shape = (128, 128)  # demi-taille 64 px, limite fiable 62 px
    m = phase_correlation(rain_frame(shape, blobs), rain_frame(shape, moved), coarse=1)
    assert not m.reliable


@pytest.mark.unit
def test_too_small_frame_is_unreliable_not_an_error():
    z = np.full((8, 8), 50, dtype=np.int16)
    assert phase_correlation(z, z, coarse=4) == Motion(0.0, 0.0, 0.0, False)


@pytest.mark.unit
def test_grid_motion_covers_whole_frame_and_motion_at_clamps():
    shift = (3, -4)
    a, b = _pair(shift, seed=7, shape=(300, 390))
    g = grid_motion(a, b, tile=128, coarse=2)
    assert g.v.shape == (2, 3, 2)  # le reste est fusionné dans la dernière tuile
    vy, vx = motion_at(g, 299, 389)
    assert (vy, vx) == pytest.approx(tuple(g.v[1, 2]))
    assert motion_at(g, 0, 0) == pytest.approx(tuple(g.v[0, 0]))
    assert motion_at(g, 10_000, 10_000) == pytest.approx(tuple(g.v[1, 2]))


@pytest.mark.unit
def test_grid_motion_passes_min_peak_ratio_through():
    a, b = _split_field()
    g = grid_motion(a, b, tile=128, coarse=2, min_peak_ratio=1e9)
    assert not g.reliable.any()
