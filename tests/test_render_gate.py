"""Porte fractionnaire des sources : active[b, k] dans [0, 1] met à l'échelle la contribution de la source."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import variable_source_fit as vsf  # noqa: E402

H = W = 24
YY, XX = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
XC, YC = XX[None], YY[None]


def make_params(k=3):
    # (mx, my, amp, ra, rb, angle) : gaussienne, 6 paramètres
    rows = [[8.0, 9.0, 1.0, 0.5, 0.2, 0.3], [15.0, 12.0, 0.4, 0.1, 0.6, -0.5], [11.0, 15.0, 0.8, 0.3, 0.3, 1.0]]
    return torch.tensor([rows[:k]], dtype=torch.float32)


def both(params, active):
    dense = vsf.render(params, active, XC, YC, "gauss")
    win = vsf.render_windowed(params, active, H, W, "gauss")
    return dense, win


def test_binary_float_gate_equals_bool_mask():
    p = make_params()
    mask = torch.tensor([[True, False, True]])
    for fn in (lambda a: vsf.render(p, a, XC, YC, "gauss"), lambda a: vsf.render_windowed(p, a, H, W, "gauss")):
        assert torch.equal(fn(mask.float()), fn(mask.to(torch.int64)))
        assert torch.equal(fn(mask.float()), fn(mask.float().clone()))


@pytest.mark.parametrize("renderer", ["dense", "windowed"])
def test_half_gate_gives_half_contribution(renderer):
    p = make_params()
    def r(a):
        return both(p, torch.tensor([a]))[0 if renderer == "dense" else 1]
    full_second = r([0.0, 1.0, 0.0])
    half_second = r([0.0, 0.5, 0.0])
    assert torch.allclose(half_second, 0.5 * full_second, rtol=1e-6, atol=1e-7)
    assert half_second.abs().max() > 0
    # les autres sources restent intactes
    assert torch.allclose(r([1.0, 0.5, 1.0]), r([1.0, 0.0, 1.0]) + half_second, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("renderer", ["dense", "windowed"])
def test_gradient_flows_to_gate(renderer):
    p = make_params()
    a = torch.tensor([[0.3, 0.7, 1.0]], requires_grad=True)
    rec = vsf.render(p, a, XC, YC, "gauss") if renderer == "dense" else vsf.render_windowed(p, a, H, W, "gauss")
    rec.sum().backward()
    for k in range(3):
        alone = torch.zeros(1, 3)
        alone[0, k] = 1.0
        expected = (vsf.render(p, alone, XC, YC, "gauss")).sum()
        assert a.grad[0, k] != 0
        assert torch.allclose(a.grad[0, k], expected, rtol=1e-4)


def test_zero_gate_sources_skipped_and_silent():
    p = make_params()
    a = torch.tensor([[0.0, 1.0, 0.0]])
    # la source à porte nulle n'est pas évaluée : le résultat vaut celui de la seule source 1, même si ses paramètres sont NaN
    p_nan = p.clone()
    p_nan[0, 0] = float("nan")
    p_nan[0, 2] = float("nan")
    win = vsf.render_windowed(p_nan, a, H, W, "gauss")
    assert torch.isfinite(win).all()
    assert torch.equal(win, vsf.render_windowed(p, a, H, W, "gauss"))
    assert torch.allclose(win, vsf.render(p, a, XC, YC, "gauss"), atol=1e-4)


def test_windowed_matches_dense_with_fractional_gates():
    p = make_params()
    a = torch.tensor([[0.25, 0.5, 0.9]])
    dense, win = both(p, a)
    assert torch.allclose(dense, win, rtol=1e-3, atol=5e-3 * float(dense.max()))


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("renderer", ["dense", "windowed"])
def test_gate_outside_unit_interval_raises(renderer, bad):
    a = torch.tensor([[1.0, bad, 0.0]])
    with pytest.raises(ValueError, match="active"):
        if renderer == "dense":
            vsf.render(make_params(), a, XC, YC, "gauss")
        else:
            vsf.render_windowed(make_params(), a, H, W, "gauss")


def test_gradient_is_zero_at_zero_gate_and_positive_above():
    p = make_params()
    a = torch.tensor([[0.0, 1e-3, 1.0]], requires_grad=True)
    vsf.render_windowed(p, a, H, W, "gauss").sum().backward()
    assert a.grad[0, 0] == 0  # source sautée : pas de gradient (plancher positif requis pour apprendre)
    assert a.grad[0, 1] > 0
