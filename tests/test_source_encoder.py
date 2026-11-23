"""Encodeur de sources (phase 4 E7) : UNet convolutif stride 2, décodage en paramètres bruts diff7."""
import dataclasses
import math
import os
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import variable_source_fit as vsf
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from source_encoder import (EncoderConfig, HeadOut, SourceEncoder, count_parameters, decode_candidates, head_grid_hw,
                            pad_batch, pad_inputs, select_sources)

TINY_CFG = EncoderConfig(base_ch=8, k_train=16)


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    table = load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("lut"))
    return table["lut"][None, None]


def _crop(h, w, seed=0):
    g = np.random.default_rng(seed)
    rate = (g.random((h, w)) * 5).astype(np.float32)
    return rate, rate > 1.0, np.ones((h, w), dtype=bool)


def _forward(h, w, cfg=TINY_CFG):
    x, _ = pad_inputs(*_crop(h, w))
    return SourceEncoder(cfg)(x[None])


@pytest.mark.unit
def test_config_validation_rejects_bad_values():
    for bad in (dict(base_ch=0), dict(depth=0), dict(gate_floor=0.0), dict(gate_floor=1.0),
                dict(radius_clamp_frac=0.0), dict(compression_frac=-1.0), dict(k_train=0), dict(cap=0)):
        with pytest.raises(ValueError):
            EncoderConfig(**bad)


@pytest.mark.unit
def test_config_is_frozen_with_planned_defaults():
    cfg = EncoderConfig()
    assert (cfg.in_ch, cfg.base_ch, cfg.depth, cfg.head_stride, cfg.k_train, cfg.cap) == (3, 24, 4, 2, 256, 128)
    assert cfg.pad_multiple == 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.base_ch = 5  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.parametrize("h,w", [(23, 39), (53, 92)])
def test_pad_inputs_pads_to_multiple_of_32_with_three_channels(h, w):
    rate, cell, valid = _crop(h, w)
    x, info = pad_inputs(rate, cell, valid)
    assert x.shape[0] == 3 and x.shape[1] % 32 == 0 and x.shape[2] % 32 == 0
    assert (info.h, info.w) == (h, w)
    assert torch.allclose(x[0, :h, :w], torch.from_numpy(rate).sqrt())
    assert x[2, h:, :].sum() == 0 and x[2, :, w:].sum() == 0  # le padding est invalide


@pytest.mark.unit
def test_pad_inputs_does_not_mutate_inputs():
    rate, cell, valid = _crop(10, 12)
    before = (rate.copy(), cell.copy(), valid.copy())
    pad_inputs(rate, cell, valid)
    assert all(np.array_equal(a, b) for a, b in zip(before, (rate, cell, valid)))


@pytest.mark.unit
def test_pad_batch_pads_to_batch_max():
    crops = [_crop(23, 39), _crop(53, 20, 1)]
    x, infos = pad_batch(crops)
    assert x.shape == (2, 3, 64, 64)
    assert [(i.h, i.w) for i in infos] == [(23, 39), (53, 20)]


@pytest.mark.unit
@pytest.mark.parametrize("h,w", [(23, 39), (53, 92)])
def test_forward_head_maps_are_half_resolution(h, w):
    out = _forward(h, w)
    hp, wp = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    assert isinstance(out, HeadOut)
    assert out.logits.shape == (1, 1, hp // 2, wp // 2)
    assert out.offsets.shape == (1, 2, hp // 2, wp // 2)
    for t in (out.raw_A, out.raw_a, out.raw_b, out.cos2t, out.sin2t, out.raw_w):
        assert t.shape == (1, 1, hp // 2, wp // 2)
    assert out.offsets.abs().max() <= 1


@pytest.mark.unit
def test_forward_rejects_non_multiple_of_32():
    with pytest.raises(ValueError):
        SourceEncoder(TINY_CFG)(torch.zeros(1, 3, 30, 32))


@pytest.mark.unit
def test_parameter_count_reported_in_planned_range():
    n = count_parameters(SourceEncoder(EncoderConfig()))
    print(f"parametres encodeur base_ch=24 : {n}")
    assert 0.8e6 < n < 2.5e6


@pytest.mark.unit
def test_initial_radius_and_amplitude_follow_bias_init():
    out = _forward(32, 32)
    a = 0.5 + F.softplus(out.raw_a)
    assert abs(float(a.detach().mean()) - 4.0) < 1.0
    assert float(F.softplus(out.raw_A).detach().mean()) < 1.5


@pytest.mark.unit
def test_decode_candidates_raw_format_and_bounds():
    h, w, k = 23, 39, 16
    out = _forward(h, w)
    params, gate, ok = decode_candidates(out, k, [(h, w)], TINY_CFG)
    assert params.shape == (1, k, 8) and gate.shape == (1, k)
    assert torch.all(params[..., 6] == 0)
    assert params[..., 0].min() >= 0 and params[..., 0].max() <= w - 1
    assert params[..., 1].min() >= 0 and params[..., 1].max() <= h - 1
    assert gate.min() >= TINY_CFG.gate_floor and gate.max() <= 1


@pytest.mark.unit
def test_decode_radius_clamp_holds_even_for_huge_raw_radius():
    h, w = 40, 64
    out = _forward(h, w)
    big = out.__class__(**{**out.__dict__, "raw_a": out.raw_a + 100.0, "raw_b": out.raw_b + 100.0})
    params, _, _ = decode_candidates(big, 8, [(h, w)], TINY_CFG)
    a, b = 0.5 + F.softplus(params[..., 3]), 0.5 + F.softplus(params[..., 4])
    assert float(2 * torch.maximum(a, b).detach().max()) <= TINY_CFG.radius_clamp_frac * max(h, w) + 1e-4


@pytest.mark.unit
def test_decode_stays_in_unpadded_area_and_orders_by_logit():
    h, w = 23, 39
    out = _forward(h, w)
    logits = torch.full_like(out.logits, -5.0)
    logits[0, 0, 0, 0] = 5.0
    logits[0, 0, -1, -1] = 9.0  # dans le padding : doit être ignoré
    hot = out.__class__(**{**out.__dict__, "logits": logits})
    params, gate, ok = decode_candidates(hot, 3, [(h, w)], TINY_CFG)
    assert gate[0, 0] > 0.99
    assert params[0, 0, 0] <= 3 and params[0, 0, 1] <= 3  # cellule (0, 0) -> près de l'origine


@pytest.mark.unit
def test_decode_rendering_and_gradients_reach_first_conv(lut5):
    h, w = 24, 24
    net = SourceEncoder(TINY_CFG)
    x, _ = pad_inputs(*_crop(h, w))
    params, gate, ok = decode_candidates(net(x[None]), 16, [(h, w)], TINY_CFG)
    rec = vsf.render_windowed(params, gate, h, w, "diffusion", lut5)
    assert rec.shape == (1, h, w) and torch.isfinite(rec).all()
    rec.sum().backward()
    first = next(p for p in net.parameters() if p.dim() == 4)
    assert first.grad is not None and first.grad.abs().sum() > 0


@pytest.mark.unit
def test_select_sources_honours_compression_limit_and_threshold():
    h, w = 64, 64
    out = _forward(h, w)
    hot = out.__class__(**{**out.__dict__, "logits": torch.full_like(out.logits, 6.0)})
    params, active = select_sources(hot, np.array([76]), [(h, w)], TINY_CFG)
    assert params.shape[-1] == 8 and active.sum() <= 5 and active.sum() > 0
    cold = out.__class__(**{**out.__dict__, "logits": torch.full_like(out.logits, -6.0)})
    _, none = select_sources(cold, np.array([76]), [(h, w)], TINY_CFG)
    assert none.sum() == 0
    _, strict = select_sources(hot, np.array([76]), [(h, w)], TINY_CFG, threshold=0.9999)
    assert strict.sum() == 0  # p = sigmoid(6) = 0.9975 < 0.9999


@pytest.mark.unit
def test_select_sources_applies_cap_128_and_active_is_contiguous():
    h, w = 64, 64
    cfg = EncoderConfig(base_ch=8)
    x, _ = pad_inputs(*_crop(h, w))
    out = SourceEncoder(cfg)(x[None])
    hot = out.__class__(**{**out.__dict__, "logits": torch.full_like(out.logits, 6.0)})
    params, active = select_sources(hot, np.array([10 ** 6]), [(h, w)], cfg)
    assert params.shape == (1, 128, 8) and int(active.sum()) == 128
    assert bool((active[:, :-1] >= active[:, 1:]).all())




def _replace(out, **kw):
    return dataclasses.replace(out, **kw)


@pytest.mark.unit
def test_decode_non_ok_slots_have_zero_gate_no_render_no_gradient(lut5):
    h, w = 23, 39  # grille valide 12 x 20 = 240 < k_train = 256
    cfg = EncoderConfig(base_ch=8)
    torch.manual_seed(0)
    net = SourceEncoder(cfg)
    x, _ = pad_inputs(*_crop(h, w))
    params, gate, ok = decode_candidates(net(x[None]), cfg.k_train, [(h, w)], cfg)
    params.retain_grad()
    assert ok.shape == gate.shape == (1, 256) and int(ok.sum()) == 240
    assert bool((gate[~ok] == 0).all()) and bool((gate[ok] >= cfg.gate_floor).all())
    rec = vsf.render_windowed(params, gate, h, w, "diffusion", lut5)
    rec_ok = vsf.render_windowed(params[:, ok[0]], gate[:, ok[0]], h, w, "diffusion", lut5)
    assert torch.allclose(rec, rec_ok)
    rec.sum().backward()
    assert bool((params.grad[~ok] == 0).all()) and params.grad[ok].abs().sum() > 0


@pytest.mark.unit
def test_decode_allows_sources_outside_cell_mask_anywhere_in_crop():
    h, w = 32, 32
    out = _forward(h, w)  # la tête ne voit aucun masque : seule la taille du recadrage borne les positions
    logits = torch.full_like(out.logits, -5.0)
    logits[0, 0, 15, 15] = 5.0  # coin bas-droit, loin de toute cellule
    params, gate, ok = decode_candidates(_replace(out, logits=logits), 2, [(h, w)], TINY_CFG)
    assert bool(ok.all()) and gate[0, 0] > 0.99
    assert 26 <= float(params[0, 0, 0]) <= w - 1 and 26 <= float(params[0, 0, 1]) <= h - 1


@pytest.mark.unit
def test_soft_radius_clamp_respects_limit_and_keeps_gradient():
    h, w = 40, 64
    out = _forward(h, w)
    limit = 0.5 * TINY_CFG.radius_clamp_frac * max(h, w)
    raw = torch.full_like(out.raw_a, limit * 1.2 - 0.5).requires_grad_(True)  # rayon 1.2 x limite
    params, _, _ = decode_candidates(_replace(out, raw_a=raw), 4, [(h, w)], TINY_CFG)
    a = 0.5 + F.softplus(params[..., 3])
    assert float(a.max()) <= limit + 1e-4
    a.sum().backward()
    assert float(raw.grad.abs().sum()) > 0
    huge = _replace(out, raw_a=out.raw_a + 100.0)
    p2, _, _ = decode_candidates(huge, 4, [(h, w)], TINY_CFG)
    assert float((0.5 + F.softplus(p2[..., 3])).max()) <= limit + 1e-4


@pytest.mark.unit
def test_theta_atan2_on_raw_outputs_with_bounded_gradient_near_zero():
    h, w = 32, 32
    out = _forward(h, w)
    zero = torch.zeros_like(out.cos2t).requires_grad_(True)
    zero_s = torch.zeros_like(out.sin2t).requires_grad_(True)
    params, _, _ = decode_candidates(_replace(out, cos2t=zero, sin2t=zero_s), 4, [(h, w)], TINY_CFG)
    params[..., 5].sum().backward()
    assert torch.isfinite(params[..., 5]).all()
    assert torch.isfinite(zero.grad).all() and torch.isfinite(zero_s.grad).all()
    cos = torch.full_like(out.cos2t, 0.0)
    sin = torch.full_like(out.sin2t, 2.0)  # pas normalisé : theta = 0.5 * atan2(2, 0) = pi / 4
    p2, _, _ = decode_candidates(_replace(out, cos2t=cos, sin2t=sin), 1, [(h, w)], TINY_CFG)
    assert float(p2[0, 0, 5]) == pytest.approx(math.pi / 4, abs=1e-4)


@pytest.mark.unit
def test_head_grid_hw_matches_head_shape():
    for h, w in [(23, 39), (53, 92), (64, 64)]:
        assert head_grid_hw(h, w) == tuple(_forward(h, w).logits.shape[2:])


@pytest.mark.unit
def test_zero_offset_source_renders_its_peak_in_the_cell_pixels(lut5):
    h, w = 32, 32
    out = _forward(h, w)
    logits = torch.full_like(out.logits, -9.0)
    logits[0, 0, 3, 5] = 9.0
    hot = _replace(out, logits=logits, offsets=torch.zeros_like(out.offsets),
                   raw_a=torch.full_like(out.raw_a, -3.0), raw_b=torch.full_like(out.raw_b, -3.0),
                   raw_A=torch.full_like(out.raw_A, 5.0))
    params, gate, _ = decode_candidates(hot, 1, [(h, w)], TINY_CFG)
    assert float(params[0, 0, 0]) == 2 * 5 + 0.5 and float(params[0, 0, 1]) == 2 * 3 + 0.5
    rec = vsf.render_windowed(params, gate, h, w, "diffusion", lut5)[0]
    y, x = divmod(int(rec.argmax()), w)
    assert (y // 2, x // 2) == (3, 5)


@pytest.mark.unit
def test_select_sources_nms_keeps_one_source_per_blob():
    h, w = 64, 64
    out = _forward(h, w)
    logits = torch.full_like(out.logits, -6.0)
    logits[0, 0, 4, 4], logits[0, 0, 4, 5], logits[0, 0, 5, 4] = 6.0, 5.5, 5.5
    logits[0, 0, 20, 20] = 4.0  # second blob, isolé
    _, active = select_sources(_replace(out, logits=logits), np.array([5000]), [(h, w)], TINY_CFG)
    assert int(active.sum()) == 2


@pytest.mark.benchmark
@pytest.mark.skipif(not os.environ.get("RUN_BENCH"), reason="benchmark : RUN_BENCH=1 pour l'exécuter")
@pytest.mark.parametrize("size", [64, 256])
def test_benchmark_forward_render_backward_k256_large_radii(size, lut5):
    cfg = EncoderConfig()
    torch.manual_seed(0)
    net = SourceEncoder(cfg)
    x, _ = pad_inputs(*_crop(size, size))
    t0 = time.perf_counter()
    out = net(x[None])
    radii = torch.empty_like(out.raw_a).uniform_(3.5, 30.0) - 0.5
    raw = torch.log(torch.expm1(radii))
    out = _replace(out, raw_a=out.raw_a - out.raw_a.detach() + raw, raw_b=out.raw_b - out.raw_b.detach() + raw,
                   logits=out.logits + torch.randn_like(out.logits))
    params, gate, _ = decode_candidates(out, 256, [(size, size)], cfg)
    vsf.render_windowed(params, gate, size, size, "diffusion", lut5).sum().backward()
    print(f"BENCH k=256 rayons 3.5-30 {size}x{size} : {time.perf_counter() - t0:.2f} s")
