"""Pas d'entraînement par morceaux : somme des pertes = perte du lot entier, reprise sur OOM, budget de rendu."""
import dataclasses

import pytest
import torch

import encoder_step
from cell_shard_dataset import CellShardDataset
from encoder_loss import LossConfig, total_loss
from encoder_step import accumulate_step, encode, render_cost
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from shard_helpers import write_tiny_dir
from size_batching import collate, load_index
from source_encoder import EncoderConfig, SourceEncoder, decode_candidates, head_grid_hw
import variable_source_fit as vsf

K = 8
LOSS = LossConfig(binarise_ramp_steps=2)
HUGE = 10 ** 12


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    return load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("l"))["lut"][None, None]


@pytest.fixture
def setup(tmp_path):
    index = load_index(write_tiny_dir(tmp_path / "s", n_train=6, n_val=1))
    data = CellShardDataset(index)
    batch = collate([data[i] for i in range(5)])
    torch.manual_seed(1)
    model = SourceEncoder(EncoderConfig(base_ch=4, k_train=K, cap=16))
    h2, w2 = head_grid_hw(batch.targets.Hm, batch.targets.Wm)
    peaks = torch.zeros(5, 1, h2, w2)
    peaks[:, 0, 3, 3] = 1.0
    peaks[:, 0, 3, 4] = 0.6
    return model, batch, peaks


def _grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def _run(setup, lut5, peaks_on=True, **kw):
    model, batch, peaks = setup
    args = dict(k_train=K, loss_cfg=LOSS, render_px_budget=HUGE, chunks=1) | kw
    stats = accumulate_step(model, batch, peaks if peaks_on else None, lut5, 3, **args)
    return stats, _grads(model).clone()


@pytest.mark.unit
def test_single_chunk_equals_total_loss(setup, lut5):
    model, batch, peaks = setup
    out = encode(model, batch.x)
    params, gate, ok = decode_candidates(out, K, batch.crop_hw, model.cfg)
    t = batch.targets
    rec = vsf.render_windowed(params, gate, t.Hm, t.Wm, "diffusion", lut5)
    ref = total_loss(rec, t, gate, batch.areas, LOSS, 3, logits=out.logits, peak_maps=peaks, ok=ok)
    stats, _ = _run(setup, lut5)
    assert stats.loss == pytest.approx(float(ref.total), rel=1e-5)
    assert stats.recon == pytest.approx(ref.recon, rel=1e-5) and stats.heatmap == pytest.approx(ref.heatmap, rel=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("peaks_on", [True, False])
def test_chunked_loss_and_gradients_equal_full_batch(setup, lut5, peaks_on):
    full, g_full = _run(setup, lut5, peaks_on)
    chunked, g_chunk = _run(setup, lut5, peaks_on, chunks=3)
    assert chunked.n_chunks == 3
    assert chunked.loss == pytest.approx(full.loss, rel=1e-4)
    assert chunked.binarise == pytest.approx(full.binarise, rel=1e-4)
    assert chunked.k_active == pytest.approx(full.k_active)
    assert torch.allclose(g_chunk, g_full, rtol=1e-3, atol=1e-6)


@pytest.mark.unit
def test_render_over_budget_splits_until_it_fits(setup, lut5):
    full, g_full = _run(setup, lut5)
    split, g_split = _run(setup, lut5, render_px_budget=1)
    assert split.n_chunks == 5  # jusqu'à une cellule par morceau (une cellule seule passe toujours)
    assert split.loss == pytest.approx(full.loss, rel=1e-4)
    assert torch.allclose(g_split, g_full, rtol=1e-3, atol=1e-6)


@pytest.mark.unit
def test_render_cost_counts_window_pixels(lut5):
    params = torch.zeros(1, 3, 8)
    params[..., 3:5] = 1.0  # rayon physique 0.5 + softplus(1) ~ 1.8 -> portée 3.7 -> fenêtre h = 8
    gate = torch.tensor([[1.0, 1.0, 0.0]])
    assert render_cost(params, gate, 64, 64) == 2 * (2 * 8 + 1) ** 2


@pytest.mark.unit
def test_oom_once_retries_with_halved_chunks(setup, lut5, monkeypatch):
    full, g_full = _run(setup, lut5)
    real, calls = encoder_step.encode, {"n": 0}

    def flaky(model, x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return real(model, x)
    monkeypatch.setattr(encoder_step, "encode", flaky)
    stats, g = _run(setup, lut5)
    assert stats.n_chunks == 2 and stats.loss == pytest.approx(full.loss, rel=1e-4)
    assert torch.allclose(g, g_full, rtol=1e-3, atol=1e-6)


@pytest.mark.unit
def test_oom_on_single_cell_gives_clear_error(setup, lut5, monkeypatch):
    def always(model, x):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(encoder_step, "encode", always)
    with pytest.raises(RuntimeError, match="une cellule seule"):
        _run(setup, lut5)


@pytest.mark.unit
def test_non_oom_errors_are_not_swallowed(setup, lut5, monkeypatch):
    monkeypatch.setattr(encoder_step, "encode", lambda m, x: (_ for _ in ()).throw(RuntimeError("autre")))
    with pytest.raises(RuntimeError, match="autre"):
        _run(setup, lut5)


@pytest.mark.unit
def test_heads_stay_float32_under_autocast(setup):
    model, batch, _ = setup
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch.x)
    assert out.logits.dtype == torch.float32 and out.offsets.dtype == torch.float32


@pytest.mark.unit
def test_nan_params_raise_non_finite_step_before_render(setup, lut5, monkeypatch):
    real = encoder_step.encode

    def nan_logits(model, x):
        out = real(model, x)
        return dataclasses.replace(out, logits=out.logits * float("nan"))
    monkeypatch.setattr(encoder_step, "encode", nan_logits)
    monkeypatch.setattr(encoder_step.vsf, "render_windowed", lambda *a, **k: pytest.fail("rendu appelé"))
    with pytest.raises(encoder_step.NonFiniteStep):
        _run(setup, lut5)
    assert all(p.grad is None for p in setup[0].parameters())


@pytest.mark.unit
def test_non_finite_gradient_raises_and_clears_grads(setup, lut5):
    first = next(setup[0].parameters())
    handle = first.register_hook(lambda g: g * float("inf"))
    with pytest.raises(encoder_step.NonFiniteStep, match="gradients"):
        _run(setup, lut5)
    handle.remove()
    assert all(p.grad is None for p in setup[0].parameters())
