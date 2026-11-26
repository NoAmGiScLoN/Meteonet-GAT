"""Pertes de l'encodeur de sources (phase 4 E7)."""
import numpy as np
import pytest
import torch

import variable_source_fit as vsf
from encoder_loss import (LossConfig, LossParts, binarise_loss, greedy_peak_map, recon_loss, sparsity_loss,
                          total_loss, warmup_heatmap_loss, heatmap_scale)
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from source_encoder import EncoderConfig, SourceEncoder, decode_candidates, head_grid_hw, pad_inputs


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    return load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("lut"))["lut"][None, None]


@pytest.mark.unit
def test_recon_loss_equals_fitter_formula(make_cell):
    cells = [make_cell(0), make_cell(1, size=16)]
    tg = vsf.make_targets(cells)
    rec = torch.rand(2, tg.Hm, tg.Wm) * 4
    expected = ((((rec.clamp(min=0) + vsf.EPS).sqrt() - tg.sT) ** 2) * tg.valid).sum((1, 2)) / tg.npix
    assert torch.allclose(recon_loss(rec, tg), expected)


@pytest.mark.unit
def test_recon_loss_zero_for_perfect_reconstruction_up_to_eps(make_cell):
    tg = vsf.make_targets([make_cell(0)])
    # sqrt(T + EPS) != sqrt(T) : l'erreur résiduelle vient de EPS seulement, très petite
    assert float(recon_loss(tg.T.clone(), tg).max()) < 1e-3
    assert float(recon_loss(tg.T + 5, tg).min()) > 0.1


@pytest.mark.unit
def test_sparsity_is_zero_below_the_compression_limit():
    # aire 100 -> k_limit = floor(0.5 * 100 / 7) = 7 ; Σp = 4 < 7
    assert float(sparsity_loss(torch.ones(2, 4), torch.tensor([100.0, 100.0]))) == 0.0


@pytest.mark.unit
def test_sparsity_is_linear_above_the_limit_in_params_per_pixel():
    areas = torch.tensor([100.0, 50.0])  # k_limit = 7 et 3
    p = torch.ones(2, 10)  # excès 3 et 7
    assert float(sparsity_loss(p, areas)) == pytest.approx((3 * 7 / 100 + 7 * 7 / 50) / 2)
    assert float(sparsity_loss(torch.ones(2, 11), areas)) == pytest.approx((4 * 7 / 100 + 8 * 7 / 50) / 2)


@pytest.mark.unit
def test_sparsity_honours_the_cap_and_gives_a_gradient_only_above_the_limit():
    area = torch.tensor([1e6])  # compression : 71428 -> plafond 128
    assert float(sparsity_loss(torch.ones(1, 130), area, cap=128)) == pytest.approx(2 * 7 / 1e6)
    p = torch.full((1, 10), 0.5, requires_grad=True)  # Σp = 5 > k_limit = 3 (aire 50)
    sparsity_loss(p, torch.tensor([50.0])).backward()
    assert (p.grad > 0).all()
    q = torch.full((1, 2), 0.5, requires_grad=True)
    sparsity_loss(q, torch.tensor([50.0])).backward()
    assert (q.grad == 0).all()


@pytest.mark.unit
def test_binarise_zero_at_extremes_and_max_at_half():
    assert float(binarise_loss(torch.tensor([[0.0, 1.0]]))) == 0.0
    vals = [float(binarise_loss(torch.full((1, 3), g))) for g in (0.1, 0.5, 0.9)]
    assert vals[1] == max(vals) == pytest.approx(0.25)


@pytest.mark.unit
def test_greedy_peak_map_first_peak_at_rate_maximum():
    rate = np.zeros((20, 30), dtype=np.float32)
    rate[5:15, 8:20] = 2.0
    rate[11, 17] = 9.0
    hm = greedy_peak_map(rate, rate > 0, k=3, sigma=1.5, grid_hw=(10, 15))
    assert hm.shape == (10, 15)
    assert hm[11 // 2, 17 // 2] == 1.0  # le premier pic est au maximum du taux
    assert hm.max() == pytest.approx(1.0) and hm.min() >= 0
    assert int((hm == 1.0).sum()) == 3  # un pic par source


@pytest.mark.unit
def test_greedy_peak_map_grid_hw_and_no_input_mutation():
    rate = np.ones((10, 10), dtype=np.float32)
    snap = rate.copy()
    hm = greedy_peak_map(rate, rate > 0, k=1, sigma=1.0, grid_hw=(16, 16))
    assert hm.shape == (16, 16) and np.array_equal(rate, snap)


@pytest.mark.unit
def test_warmup_heatmap_loss_lower_when_logits_match_peaks():
    peaks = torch.zeros(1, 1, 8, 8)
    peaks[0, 0, 3, 4] = 1.0
    good = torch.full((1, 1, 8, 8), -6.0)
    good[0, 0, 3, 4] = 6.0
    assert warmup_heatmap_loss(good, peaks) < warmup_heatmap_loss(torch.zeros(1, 1, 8, 8), peaks)
    assert warmup_heatmap_loss(-good, peaks) > warmup_heatmap_loss(good, peaks)


@pytest.mark.unit
def test_loss_config_validation_and_ramp():
    with pytest.raises(ValueError):
        LossConfig(lambda_sparsity=-1.0)
    cfg = LossConfig(lambda_binarise=0.4, binarise_ramp_steps=10)
    assert cfg.binarise_weight(0) == 0.0 and cfg.binarise_weight(5) == pytest.approx(0.2)
    assert cfg.binarise_weight(100) == pytest.approx(0.4)


@pytest.mark.unit
def test_total_loss_combines_weighted_parts(make_cell):
    tg = vsf.make_targets([make_cell(0)])
    rec = torch.rand(1, tg.Hm, tg.Wm)
    gate = torch.full((1, 4), 0.3)
    cfg = LossConfig(lambda_sparsity=0.5, lambda_binarise=2.0, binarise_ramp_steps=1, lambda_heatmap=0.0)
    parts = total_loss(rec, tg, gate, torch.tensor([100.0]), cfg, step=5)
    assert isinstance(parts, LossParts)
    expected = parts.recon + 0.5 * parts.sparsity + 2.0 * parts.binarise
    assert float(parts.total) == pytest.approx(expected, rel=1e-5)
    assert parts.heatmap == 0.0


@pytest.mark.unit
def test_overfit_one_synthetic_cell_reduces_total_loss(make_cell, lut5):
    cell = make_cell(0)
    h, w = cell["rate"].shape
    cfg = EncoderConfig(base_ch=8, k_train=16)
    lcfg = LossConfig(lambda_sparsity=0.01, lambda_binarise=0.01, binarise_ramp_steps=20, lambda_heatmap=0.5)
    torch.manual_seed(0)
    net = SourceEncoder(cfg)
    x, _ = pad_inputs(cell["rate"], cell["rate"] > 0, np.ones((h, w), dtype=bool))
    tg = vsf.make_targets([cell])
    peaks = torch.from_numpy(greedy_peak_map(cell["rate"], cell["rate"] > 0, 3, 1.5,
                                             grid_hw=head_grid_hw(h, w)))[None, None]
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    areas = torch.tensor([float(cell["area"])])
    totals = []
    for step in range(50):
        out = net(x[None])
        params, gate, ok = decode_candidates(out, cfg.k_train, [(h, w)], cfg)
        rec = vsf.render_windowed(params, gate, h, w, "diffusion", lut5)
        parts = total_loss(rec, tg, gate, areas, lcfg, step=step, logits=out.logits, peak_maps=peaks, ok=ok)
        opt.zero_grad()
        parts.total.backward()
        opt.step()
        totals.append(float(parts.total.detach()))
    assert np.mean(totals[-5:]) < 0.7 * np.mean(totals[:3])


@pytest.mark.unit
def test_greedy_peak_map_never_lands_outside_cell_or_on_invalid_pixels():
    rate = np.full((20, 20), 2.0, dtype=np.float32)
    rate[0, 0] = 50.0  # maximum hors cellule
    rate[10, 10] = 40.0  # maximum sur pixel invalide
    cell = np.zeros((20, 20), dtype=bool)
    cell[8:14, 8:14] = True
    valid = np.ones((20, 20), dtype=bool)
    valid[10, 10] = False
    hm = greedy_peak_map(rate, cell, k=2, sigma=1.0, grid_hw=(10, 10), valid=valid)
    peaks = np.argwhere(hm == 1.0)
    assert len(peaks) >= 1 and hm[0, 0] < 1.0 and hm[5, 5] < 1.0
    assert all(4 <= i <= 6 and 4 <= j <= 6 for i, j in peaks)


@pytest.mark.unit
def test_greedy_peak_map_stops_without_positive_residual_and_is_nan_safe():
    rate = np.zeros((12, 12), dtype=np.float32)
    rate[5, 5] = 3.0
    rate[2, 2] = np.nan
    cell = rate > 0
    cell[2, 2] = True
    hm = greedy_peak_map(rate, cell, k=5, sigma=1.0, grid_hw=(6, 6))
    assert np.isfinite(hm).all() and int((hm == 1.0).sum()) == 1  # un seul pixel positif : moins de k pics


@pytest.mark.unit
def test_greedy_peak_map_uses_half_pixel_cell_centres():
    rate = np.zeros((8, 8), dtype=np.float32)
    rate[2, 4] = 5.0  # pixel (y=2, x=4) : cellule (1, 2), centre en pixels (2.5, 4.5)
    hm = greedy_peak_map(rate, rate > 0, k=1, sigma=1.0, grid_hw=(4, 4))
    assert hm[1, 2] == 1.0
    assert hm[1, 1] == pytest.approx(np.exp(-((4 - 2.5) ** 2 + (2 - 2.5) ** 2) / 2), rel=1e-5)


@pytest.mark.unit
def test_masked_losses_ignore_non_ok_slots():
    gate = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    ok = torch.tensor([[True, True, False, False]])
    assert float(binarise_loss(gate, ok)) == pytest.approx(0.125)  # (0 + 0.25) / 2 ; sans masque : 0.0625
    assert float(binarise_loss(gate)) == pytest.approx(0.0625)
    # aire 10 -> k_limit = 1 : Σp valides = 1.5 (excès 0.5) ; sans masque Σp = 1.5 aussi, on teste donc le masque ci-dessous
    assert float(sparsity_loss(gate, torch.tensor([10.0]), ok=ok)) == pytest.approx(0.5 * 7 / 10)
    junk = torch.tensor([[1.0, 0.5, 1.0, 1.0]])  # places non valides à porte non nulle : ignorées
    assert float(sparsity_loss(junk, torch.tensor([10.0]), ok=ok)) == pytest.approx(0.5 * 7 / 10)


@pytest.mark.unit
def test_heatmap_scale_anneals_linearly_over_the_last_warmup_epoch():
    assert heatmap_scale(0, 5, 10, warmup_epochs=2) == 1.0  # avant la dernière époque d'échauffement
    assert heatmap_scale(1, 0, 10, warmup_epochs=2) == 1.0
    assert heatmap_scale(1, 5, 10, warmup_epochs=2) == pytest.approx(0.5)
    assert heatmap_scale(1, 9, 10, warmup_epochs=2) == pytest.approx(0.1)
    assert heatmap_scale(2, 0, 10, warmup_epochs=2) == 0.0  # après l'échauffement
    assert heatmap_scale(0, 0, 10, warmup_epochs=0) == 0.0  # démarrage à chaud : pas de carte de pics
