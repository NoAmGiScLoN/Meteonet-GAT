"""Étape A1a : taux d'apprentissage optionnel et planning cosinus par vague dans fit_group."""
import math
import sys

import pytest
import torch

import optim_schedule as osch
import variable_source_fit as vsf

KMAX, ITERS = 4, 12
LR = 0.05


@pytest.fixture
def lut5_depth1(synthetic_lut):
    return synthetic_lut[:1][None, None]


@pytest.fixture
def lut5_full(synthetic_lut):
    return synthetic_lut[None, None]


@pytest.fixture
def cells(make_cell):
    return [make_cell(0), make_cell(1, blobs=((10.0, 9.0, 5.0, 3.0, 5.0),))]


def _lut(kind, lut5_depth1, lut5_full):
    return lut5_depth1 if kind in ("gauss", "diff7") else lut5_full


def _run(cells, kind, lut, **kw):
    return vsf.fit_group(cells, kind, KMAX, ITERS, lut, **kw)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["gauss", "diff7"])
def test_default_call_equals_explicit_default_lr_and_no_schedule(cells, lut5_depth1, kind):
    rec_a, _, (p_a, _) = _run(cells, kind, lut5_depth1)
    rec_b, _, (p_b, _) = _run(cells, kind, lut5_depth1, lr=0.05, lr_schedule=None)
    assert torch.equal(p_a, p_b)
    assert rec_a == rec_b


@pytest.mark.unit
def test_cosine_gives_different_params_from_fixed(cells, lut5_depth1):
    _, _, (p_fixed, _) = _run(cells, "diff7", lut5_depth1)
    _, _, (p_cos, _) = _run(cells, "diff7", lut5_depth1, lr_schedule="cosine")
    assert not torch.equal(p_fixed, p_cos)


@pytest.mark.unit
@pytest.mark.parametrize("kind,col", [("diff7", 6), ("diff_tau", 7), ("diff_w", 6)])
def test_frozen_columns_unchanged_under_cosine(cells, lut5_depth1, lut5_full, kind, col):
    _, _, (params, active) = _run(cells, kind, _lut(kind, lut5_depth1, lut5_full), lr_schedule="cosine")
    assert active.sum() > 0
    assert torch.all(params[..., col] == 0)


@pytest.fixture
def lr_spy(monkeypatch):
    """Adam espionné : enregistre, par vague, le lr de param_groups[0] avant chaque step."""
    waves = []
    real = torch.optim.Adam

    class SpyAdam(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            waves.append([])

        def step(self, *a, **k):
            waves[-1].append(self.param_groups[0]["lr"])
            return super().step(*a, **k)

    monkeypatch.setattr(torch.optim, "Adam", SpyAdam)
    return waves


def _cosine(lr, t, iters):
    eta_min = lr * osch.ETA_MIN_FRAC
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * t / iters)) / 2


@pytest.mark.unit
def test_cosine_restarts_each_wave_and_decays_toward_eta_min(cells, lut5_depth1, lr_spy):
    _run(cells, "diff7", lut5_depth1, lr=0.08, lr_schedule="cosine")
    n_waves = len([k for k in vsf.SCHEDULE if k <= KMAX])
    assert len(lr_spy) == n_waves >= 2
    for lrs in lr_spy:
        assert len(lrs) == ITERS
        assert lrs[0] == pytest.approx(0.08)
        assert lrs[-1] == pytest.approx(_cosine(0.08, ITERS - 1, ITERS))
        # le dernier pas est à t = ITERS-1 (le planning avance après le step) : eta_min n'est atteint
        # qu'à t = ITERS, jamais observé par un step, donc lrs[-1] reste strictement au-dessus.
        assert lrs[-1] > 0.08 * osch.ETA_MIN_FRAC
        assert all(a > b for a, b in zip(lrs, lrs[1:]))


@pytest.mark.unit
def test_no_schedule_keeps_lr_constant(cells, lut5_depth1, lr_spy):
    _run(cells, "diff7", lut5_depth1)
    assert all(lr == 0.05 for lrs in lr_spy for lr in lrs)


@pytest.mark.unit
def test_larger_lr_changes_result(cells, lut5_depth1):
    _, _, (p_a, _) = _run(cells, "diff7", lut5_depth1)
    _, _, (p_b, _) = _run(cells, "diff7", lut5_depth1, lr=0.1)
    assert not torch.equal(p_a, p_b)


@pytest.mark.unit
@pytest.mark.parametrize("kw", [dict(lr=0), dict(lr=-1), dict(lr=float("nan")), dict(lr=float("inf")),
                                dict(lr_schedule="linear")])
def test_invalid_lr_or_schedule_raises(cells, lut5_depth1, kw):
    with pytest.raises(ValueError):
        _run(cells, "diff7", lut5_depth1, **kw)


@pytest.mark.unit
def test_make_scheduler_none_returns_none():
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=0.1)
    assert osch.make_scheduler(opt, None, 10, 0.1) is None


@pytest.mark.unit
def test_parser_defaults_and_options():
    ap = vsf.build_parser()
    args = ap.parse_args([])
    assert args.lr == 0.05 and args.lr_schedule == "none"
    args = ap.parse_args(["--lr", "0.1", "--lr-schedule", "cosine"])
    assert args.lr == 0.1 and args.lr_schedule == "cosine"
    assert ap.parse_args(["--lr-schedule", "none"]).lr_schedule == "none"
    assert osch.LR_SCHEDULE_CHOICES == ("none", "cosine")


@pytest.mark.unit
@pytest.mark.parametrize("bad", [["--lr", "0"], ["--lr", "-1"], ["--lr", "nan"], ["--lr", "abc"],
                                 ["--lr-schedule", "linear"]])
def test_parser_rejects_bad_lr_options(bad):
    with pytest.raises(SystemExit):
        vsf.build_parser().parse_args(bad)


@pytest.mark.unit
def test_validate_lr_accepts_numpy_float_and_rejects_bool():
    import numpy as np
    osch.validate_lr(np.float32(0.05), None)
    with pytest.raises(ValueError):
        osch.validate_lr(True, None)


@pytest.mark.integration
@pytest.mark.parametrize("cli, expected", [("cosine", "cosine"), ("none", None)])
def test_main_threads_lr_and_schedule_to_fit_group(tmp_path, monkeypatch, make_cell, cli, expected):
    import functools
    import pickle

    import profile_lut as pl
    sample = [make_cell(0, bin_name="50-200")]
    path = tmp_path / "sample.pkl"
    path.write_bytes(pickle.dumps(sample))
    seen = []
    real = vsf.fit_group

    def spy(cells, kind, kmax, iters, *a, **k):
        seen.append({key: k[key] for key in ("lr", "lr_schedule")})
        return real(cells, kind, kmax, 1, *a, **k)

    monkeypatch.setattr(vsf, "fit_group", spy)
    monkeypatch.setattr(vsf, "LogWLutSpec", functools.partial(pl.LogWLutSpec, n=32, steps=60, n_r=32))
    monkeypatch.setattr(sys, "argv", ["vsf", "--sample", str(path), "--bins", "50-200", "--n-w", "3",
                                      "--device", "cpu", "--out", str(tmp_path / "run"),
                                      "--lr", "0.02", "--lr-schedule", cli])
    vsf.main()
    assert seen == [dict(lr=0.02, lr_schedule=expected)]
