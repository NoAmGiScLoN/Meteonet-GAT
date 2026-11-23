"""Tests du fitter a K adaptatif : planning, regles d'arret, elagage, retassage, compactage, coherence avec fit_group."""
import logging

import numpy as np
import pytest
import torch

import adaptive_fit as af
import variable_source_fit as vsf
from residual_stats import STAT_COLUMNS

ARM = "diff7"
N_PAR = vsf.ARM_NPAR[ARM]
SMALL_BLOB = ((12.0, 12.0, 5.0, 2.0, 2.0),)
P = 8
AMP_STRONG, AMP_MID, AMP_WEAK = 10.0, 5.0, 0.01


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def _cfg(**kw):
    base = dict(arm=ARM, target=0.0, k_cap=8, compression_frac=None, prune_frac=0.0, iters_override=3)
    base.update(kw)
    return af.AdaptiveConfig(**base)


@pytest.fixture
def lut5(synthetic_lut):
    return synthetic_lut[:1][None, None]


def _state(amps_by_cell, capacity=None):
    """params/active contigus ; l'amplitude (colonne 2) vaut softplus^-1(amp)."""
    capacity = capacity or max(len(a) for a in amps_by_cell)
    params = torch.zeros(len(amps_by_cell), capacity, P)
    active = torch.zeros(len(amps_by_cell), capacity)
    for b, amps in enumerate(amps_by_cell):
        for k, amp in enumerate(amps):
            params[b, k, 0] = 100 * b + k  # marqueur d'identite
            params[b, k, 2] = _inv_softplus(amp)
            active[b, k] = 1
    return params, active


# ------------------------------------------------------------------ planning
@pytest.mark.unit
def test_wave_schedule_cap_128_and_256():
    assert af.wave_schedule(128) == (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    assert af.wave_schedule(256) == af.wave_schedule(128) + (192, 256)
    assert af.wave_schedule(8) == (1, 2, 3, 4, 6, 8)


@pytest.mark.unit
@pytest.mark.parametrize("cap", [0, 5, 100, 512, -1])
def test_wave_schedule_rejects_cap_outside_schedule(cap):
    with pytest.raises(ValueError):
        af.wave_schedule(cap)


@pytest.mark.unit
@pytest.mark.parametrize("area,expected", [(50, 1), (200, 7), (10, 1), (1000, 35)])
def test_compression_k_limit(area, expected):
    assert af.compression_k_limit(area, N_PAR, 0.25) == expected


@pytest.mark.unit
def test_compression_k_limit_off_when_frac_none():
    assert af.compression_k_limit(200, N_PAR, None) is None


# ------------------------------------------------------------------ regles d'arret
@pytest.mark.unit
@pytest.mark.parametrize("args,expected", [
    (dict(nrmse=0.05, k_target=8, k_cap=8, k_limit=8, schedule_exhausted=True), "target"),
    (dict(nrmse=0.5, k_target=8, k_cap=8, k_limit=8, schedule_exhausted=False), "cap"),
    (dict(nrmse=0.5, k_target=4, k_cap=8, k_limit=None, schedule_exhausted=True), "cap"),
    (dict(nrmse=0.5, k_target=4, k_cap=8, k_limit=4, schedule_exhausted=False), "compression"),
    (dict(nrmse=0.5, k_target=3, k_cap=8, k_limit=4, schedule_exhausted=False), None),
    (dict(nrmse=0.5, k_target=3, k_cap=8, k_limit=None, schedule_exhausted=False), None),
    (dict(nrmse=0.1, k_target=3, k_cap=8, k_limit=4, schedule_exhausted=False), "target"),
])
def test_stop_reason_priority(args, expected):
    assert af.stop_reason(target=0.1, **args) == expected


@pytest.mark.unit
@pytest.mark.parametrize("k_target,k_cap,k_limit,exhausted,expected", [
    (13, 16, 13, True, "compression"),
    (16, 16, 16, True, "cap"),
    (12, 16, None, True, "cap"),
    (13, 16, 20, True, "cap"),
    (13, 16, 13, False, "compression"),
])
def test_stop_reason_compression_before_schedule_end(k_target, k_cap, k_limit, exhausted, expected):
    assert af.stop_reason(0.5, k_target, k_cap, k_limit, exhausted, 0.1) == expected


# ------------------------------------------------------------------ elagage / retassage
@pytest.mark.unit
def test_prune_removes_weak_sources_and_counts():
    params, active = _state([[AMP_STRONG, AMP_WEAK, AMP_MID], [AMP_MID, AMP_MID]])
    new_active, n_pruned = af.prune(params, active, np.array([10.0, 10.0]), 0.01)
    assert new_active.tolist() == [[1, 0, 1], [1, 1, 0]]
    assert n_pruned.tolist() == [1, 0]


@pytest.mark.unit
def test_prune_keeps_strongest_when_all_below_threshold():
    params, active = _state([[0.02, 0.05, 0.03]])
    new_active, n_pruned = af.prune(params, active, np.array([100.0]), 0.5)
    assert new_active.tolist() == [[0, 1, 0]]
    assert n_pruned.tolist() == [2]


@pytest.mark.unit
def test_prune_does_not_mutate_inputs_and_zero_frac_is_off():
    params, active = _state([[AMP_STRONG, AMP_WEAK]])
    p0, a0 = params.clone(), active.clone()
    new_active, n_pruned = af.prune(params, active, np.array([10.0]), 0.0)
    af.prune(params, active, np.array([10.0]), 0.5)
    assert torch.equal(params, p0) and torch.equal(active, a0)
    assert torch.equal(new_active, active) and n_pruned.tolist() == [0]


@pytest.mark.unit
def test_repack_is_stable_contiguous_and_new_tensors():
    params, active = _state([[AMP_MID] * 4, [AMP_MID] * 4])
    active = torch.tensor([[1.0, 0, 1, 1], [0, 0, 0, 1]])
    p0, a0 = params.clone(), active.clone()
    new_p, new_a, k_cur = af.repack(params, active)
    assert k_cur.tolist() == [3, 1]
    assert new_a.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]
    assert new_p[0, :3, 0].tolist() == [0.0, 2.0, 3.0]
    assert new_p[1, 0, 0].item() == 103.0
    assert torch.equal(params, p0) and torch.equal(active, a0)


# ------------------------------------------------------------------ config
@pytest.mark.unit
@pytest.mark.parametrize("kw", [dict(arm="nope"), dict(target=-0.1), dict(k_cap=5), dict(compression_frac=0.0),
                                dict(compression_frac=1.5), dict(prune_frac=-0.1), dict(prune_frac=1.0),
                                dict(lr=0.0), dict(iters_override=0)])
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


@pytest.mark.unit
def test_default_config_values():
    cfg = af.AdaptiveConfig()
    # D24 : compression 0.5 (0.25 bloquait les 50-200 px à K≈2)
    assert (cfg.arm, cfg.target, cfg.k_cap, cfg.compression_frac, cfg.prune_frac) == ("diff7", 0.1, 128, 0.5, 0.01)


@pytest.mark.unit
def test_missing_id_raises(make_cell, lut5):
    cell = make_cell(0)
    del cell["id"]
    with pytest.raises(ValueError, match="id"):
        af.fit_adaptive([cell], _cfg(), lut5)


# ------------------------------------------------------------------ ajustement
@pytest.mark.unit
def test_single_blob_stops_on_target_with_small_k(make_cell, lut5):
    cell = make_cell(7, blobs=SMALL_BLOB)
    results, trace, sources = af.fit_adaptive([cell], _cfg(target=0.3, iters_override=50), lut5)
    assert results[0]["stop_reason"] == "target"
    assert results[0]["K"] <= 2 and results[0]["nrmse"] <= 0.3
    assert trace[-1]["id"] == 7


@pytest.mark.unit
def test_results_columns_and_sources_shapes(make_cell, lut5):
    cells = [make_cell(0), make_cell(1, blobs=SMALL_BLOB)]
    seen = []
    results, trace, sources = af.fit_adaptive(cells, _cfg(k_cap=4), lut5, on_cell_done=lambda r, s: seen.append(r["id"]))
    assert sorted(seen) == [0, 1] and len(results) == 2
    for r in results:
        assert r["stop_reason"] == "cap" and r["K"] == 4
        assert set(STAT_COLUMNS) <= set(r)
        assert r["n_params"] == r["K"] * N_PAR
        assert r["compression_ratio"] == pytest.approx(r["n_params"] / r["area"])
        assert sources[r["id"]]["params"].shape == (r["K"], P)
        assert {"x0", "y0", "cx", "cy", "tag", "date", "bin", "area"} <= set(sources[r["id"]])
    assert {row["id"] for row in trace} == {0, 1}


@pytest.mark.unit
def test_mixed_k_limits_follow_min_of_schedule_and_limit(make_cell, lut5):
    cells = [make_cell(0, blobs=SMALL_BLOB), make_cell(1)]
    frac = 0.25
    limits = [af.compression_k_limit(c["area"], N_PAR, frac) for c in cells]
    assert limits[0] < limits[1] < 16
    results, trace, _ = af.fit_adaptive(cells, _cfg(k_cap=16, compression_frac=frac), lut5)
    sched = af.wave_schedule(16)
    for cell, limit in zip(cells, limits):
        seq = [r["k_target"] for r in trace if r["id"] == cell["id"]]
        expected = [min(k, limit) for k in sched[:len(seq)]]
        assert seq == expected and seq[-1] == limit
    by_id = {r["id"]: r for r in results}
    for cell, limit in zip(cells, limits):
        assert by_id[cell["id"]]["K"] == limit and by_id[cell["id"]]["stop_reason"] == "compression"
        assert by_id[cell["id"]]["k_limit"] == limit


@pytest.mark.unit
def test_compaction_does_not_change_remaining_cell(make_cell, lut5):
    early, late = make_cell(0, blobs=SMALL_BLOB), make_cell(1)
    cfg = _cfg(k_cap=8, compression_frac=0.25)
    assert af.compression_k_limit(early["area"], N_PAR, 0.25) < af.compression_k_limit(late["area"], N_PAR, 0.25)
    res_b, _, src_b = af.fit_adaptive([early, late], cfg, lut5)
    res_a, _, src_a = af.fit_adaptive([late], cfg, lut5)
    assert {r["id"]: r["waves"] for r in res_b}[0] < {r["id"]: r["waves"] for r in res_b}[1]
    assert np.allclose(src_b[1]["params"], src_a[1]["params"], atol=1e-6)
    nr_b = {r["id"]: r["nrmse"] for r in res_b}[1]
    assert nr_b == pytest.approx(res_a[0]["nrmse"], abs=1e-6)


@pytest.mark.unit
def test_consistent_with_fit_group(make_cell, lut5):
    cells = [make_cell(0), make_cell(1, blobs=SMALL_BLOB)]
    cfg = _cfg(k_cap=8, iters_override=5)
    records, _, (params, active) = vsf.fit_group(cells, ARM, 8, 5, lut5)
    results, trace, sources = af.fit_adaptive(cells, cfg, lut5)
    assert len(trace) == len(records)
    by_key = {(r["cell"], r["K"]): r for r in records}
    for row in trace:
        ref = by_key[(row["id"], row["K_active"])]
        for key in ("nrmse", "iou", "mass_err", "peak_err"):
            assert row[key] == pytest.approx(ref[key], abs=1e-6)
    for b, cell in enumerate(cells):
        assert np.allclose(sources[cell["id"]]["params"], params[b][active[b] > 0].numpy(), atol=1e-6)


# ------------------------------------------------------------------ elagage de bout en bout
@pytest.mark.unit
def test_pruning_end_to_end_keeps_trace_results_and_sources_consistent(make_cell, lut5):
    cell = make_cell(3)
    cfg = _cfg(k_cap=8, prune_frac=0.3, iters_override=20)
    sched = af.wave_schedule(8)
    results, trace, sources = af.fit_adaptive([cell], cfg, lut5)
    row = results[0]
    assert row["K"] < row["k_target_last"]
    assert row["waves"] <= len(sched) and len(trace) == row["waves"]
    assert trace[-1]["K_active"] == sources[3]["params"].shape[0] == row["K"]
    assert row["n_pruned"] == sum(t["n_pruned"] for t in trace) > 0


# ------------------------------------------------------------------ compactage
def _isolated(cell, cfg, lut5):
    res, _, src = af.fit_adaptive([cell], cfg, lut5)
    return res[0], src[cell["id"]]["params"]


def _assert_matches_isolated(cell, cfg, lut5, res_batch, src_batch):
    res_iso, params_iso = _isolated(cell, cfg, lut5)
    by_id = {r["id"]: r for r in res_batch}
    assert by_id[cell["id"]]["K"] == res_iso["K"]
    assert np.allclose(src_batch[cell["id"]]["params"], params_iso, atol=1e-6)
    assert by_id[cell["id"]]["nrmse"] == pytest.approx(res_iso["nrmse"], abs=1e-6)


@pytest.mark.unit
def test_compaction_when_largest_canvas_cell_stops_first(make_cell, lut5):
    big_canvas = make_cell(0, blobs=SMALL_BLOB, size=40)
    small_canvas = make_cell(1)
    cfg = _cfg(k_cap=8, compression_frac=0.25)
    res, _, src = af.fit_adaptive([big_canvas, small_canvas], cfg, lut5)
    waves = {r["id"]: r["waves"] for r in res}
    assert waves[0] < waves[1]
    _assert_matches_isolated(small_canvas, cfg, lut5, res, src)


@pytest.mark.unit
def test_compaction_when_middle_cell_stops_first(make_cell, lut5):
    first, middle, last = make_cell(0), make_cell(1, blobs=SMALL_BLOB), make_cell(2, blobs=((10.0, 14.0, 5.0, 3.0, 4.0),))
    cfg = _cfg(k_cap=8, compression_frac=0.25)
    limits = [af.compression_k_limit(c["area"], N_PAR, 0.25) for c in (first, middle, last)]
    assert limits[1] < limits[0] and limits[1] < limits[2]
    res, _, src = af.fit_adaptive([first, middle, last], cfg, lut5)
    waves = {r["id"]: r["waves"] for r in res}
    assert waves[1] < waves[0] and waves[1] < waves[2]
    for cell in (first, last):
        _assert_matches_isolated(cell, cfg, lut5, res, src)


# ------------------------------------------------------------------ divergence
@pytest.mark.unit
def test_stop_reason_diverged_on_non_finite_nrmse():
    for bad in (float("nan"), float("inf")):
        assert af.stop_reason(bad, 3, 8, None, False, 0.1) == "diverged"


@pytest.mark.unit
def test_fit_stops_with_diverged_and_logs_warning_on_nan(make_cell, lut5, monkeypatch, caplog):
    real_evaluate = vsf.evaluate

    def nan_evaluate(*args, **kw):
        rec, mets = real_evaluate(*args, **kw)
        return rec, [{**m, "nrmse": float("nan")} for m in mets]

    monkeypatch.setattr(vsf, "evaluate", nan_evaluate)
    with caplog.at_level(logging.WARNING, logger="adaptive_fit"):
        results, trace, sources = af.fit_adaptive([make_cell(0)], _cfg(k_cap=8), lut5)
    assert results[0]["stop_reason"] == "diverged" and results[0]["waves"] == 1
    assert len(trace) == 1 and 0 in sources
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ------------------------------------------------------------------ cible de vague bornee par la limite
@pytest.mark.unit
def test_k_target_sequence_ends_with_limit_exactly_once():
    seq = []
    for wave_k in af.wave_schedule(64):
        seq.append(af.wave_k_target(wave_k, 64, 35))
        if seq[-1] == 35:
            break
    assert seq == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 35]
    assert af.wave_k_target(48, 64, None) == 48 and af.wave_k_target(48, 32, None) == 32


@pytest.mark.unit
def test_fit_with_limit_not_in_schedule_hits_limit_once(make_cell, lut5, monkeypatch):
    monkeypatch.setattr(af, "compression_k_limit", lambda area, n_par, frac: 35)
    cell = make_cell(0, size=40, blobs=((20.0, 20.0, 6.0, 12.0, 12.0),))
    results, trace, _ = af.fit_adaptive([cell], _cfg(k_cap=64, compression_frac=0.25, iters_override=1), lut5)
    seq = [t["k_target"] for t in trace]
    assert seq[-3:] == [24, 32, 35] and seq.count(35) == 1
    assert results[0]["stop_reason"] == "compression" and results[0]["k_target_last"] == 35


@pytest.mark.unit
def test_fit_limit_binding_on_last_wave_is_compression(make_cell, lut5, monkeypatch):
    monkeypatch.setattr(af, "compression_k_limit", lambda area, n_par, frac: 13)
    cell = make_cell(0, size=40, blobs=((20.0, 20.0, 6.0, 12.0, 12.0),))
    results, _, _ = af.fit_adaptive([cell], _cfg(k_cap=16, compression_frac=0.25, iters_override=1), lut5)
    assert results[0]["stop_reason"] == "compression" and results[0]["k_target_last"] == 13

