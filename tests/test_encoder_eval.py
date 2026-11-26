"""Évaluation de l'encodeur (phase 6) : encodage, raffinement, benchmark, table d'adoption, débit."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import variable_source_fit as vsf
from adaptive_fit import compression_k_limit
from encoder_adoption import (CENSUS, adoption_by_year, adoption_table, check_e5_matches_samples, load_e5,
                              throughput_projection)
from encoder_eval import benchmark, check_lut_key, crop_extraction_seconds, encode_cells, load_encoder, refine
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from source_encoder import EncoderConfig, SourceEncoder

E5_ROOT = Path(__file__).resolve().parent.parent / "archive" / "E5_adaptive_k" / "results"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "results" / "03_variable_sources"
THRESHOLD = 0.1  # sous la probabilité initiale (~0.12) : l'encodeur aléatoire produit des sources
ITERS = 6
REFINE_CHECK_ITERS = 20
E5_COLUMNS = ["id", "tag", "date", "bin", "area", "K", "k_target_last", "stop_reason", "waves", "nrmse", "iou",
              "mass_err", "peak_err", "n_params", "compression_ratio", "fit_s", "n_pruned", "k_limit", "peak_bias",
              "mass_bias", "resid_std_in", "resid_std_edge", "spill_frac", "spectral_slope", "nrmse_mask"]


@pytest.fixture(scope="module")
def lut5(tmp_path_factory):
    return load_or_build_logw_lut(LogWLutSpec(n=32, steps=60, n_r=32), tmp_path_factory.mktemp("l"))["lut"][None, None]


@pytest.fixture
def tiny(tmp_path):
    cfg = EncoderConfig(base_ch=4, k_train=16, cap=16)
    torch.manual_seed(0)
    model = SourceEncoder(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "encoder": cfg, "step": 1, "best_val": 0.3}, ckpt)
    return model.eval(), cfg, ckpt


@pytest.fixture
def cells(make_cell):
    return [make_cell(0, size=24), make_cell(1, size=32), make_cell(2, size=24, bin_name="200-1000"),
            make_cell(3, blobs=((10.0, 10.0, 5.0, 3.0, 3.0),), size=24)]


def _write_sample(path, cells):
    with open(path, "wb") as fh:
        pickle.dump([{k: v for k, v in c.items() if k != "id"} for c in cells], fh)


@pytest.mark.unit
def test_load_encoder_rebuilds_model_in_eval_mode(tiny):
    model, cfg, ckpt = tiny
    loaded, loaded_cfg = load_encoder(ckpt, torch.device("cpu"))
    assert loaded_cfg == cfg and not loaded.training
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))


@pytest.mark.unit
def test_load_encoder_reads_last_checkpoint_without_config(tiny, tmp_path):
    model, cfg, _ = tiny
    last = tmp_path / "last.pt"
    torch.save({"model": model.state_dict()}, last)
    with pytest.raises(ValueError, match="encoder"):
        load_encoder(last, torch.device("cpu"))
    loaded, loaded_cfg = load_encoder(last, torch.device("cpu"), cfg=cfg)
    assert loaded_cfg == cfg


@pytest.mark.unit
def test_encode_cells_returns_raw_params_within_limit(tiny, cells):
    model, cfg, _ = tiny
    res = encode_cells(model, cfg, cells, torch.device("cpu"), threshold=THRESHOLD, batch_px=1500)
    assert set(res) == {0, 1, 2, 3}
    for c in cells:
        r = res[c["id"]]
        assert r["params"].dtype == np.float32 and r["params"].shape == (r["K"], 8)
        assert 1 <= r["K"] <= min(cfg.cap, compression_k_limit(c["area"], cfg.n_par, cfg.compression_frac))
        assert np.all(r["params"][:, 6] == 0) and np.isfinite(r["params"]).all()
        assert r["encode_s"] > 0


@pytest.mark.unit
def test_encode_cells_is_independent_of_batching(tiny, cells):
    model, cfg, _ = tiny
    one = encode_cells(model, cfg, cells[:1], torch.device("cpu"), threshold=THRESHOLD)
    many = encode_cells(model, cfg, cells, torch.device("cpu"), threshold=THRESHOLD, batch_px=10 ** 6)
    assert one[0]["K"] == many[0]["K"]


@pytest.mark.unit
def test_encode_cells_does_not_track_gradients(tiny, cells):
    model, cfg, _ = tiny
    encode_cells(model, cfg, cells, torch.device("cpu"), threshold=THRESHOLD)
    assert all(p.grad is None for p in model.parameters())


def _fit_inputs(tiny, cells, n=2):
    model, cfg, _ = tiny
    res = encode_cells(model, cfg, cells[:n], torch.device("cpu"), threshold=THRESHOLD)
    kmax = max(res[c["id"]]["K"] for c in cells[:n])
    params = torch.zeros(n, kmax, 8)
    active = torch.zeros(n, kmax)
    for b, c in enumerate(cells[:n]):
        k = res[c["id"]]["K"]
        params[b, :k] = torch.from_numpy(res[c["id"]]["params"])
        active[b, :k] = 1
    return params, active, vsf.make_targets(cells[:n])


@pytest.mark.unit
def test_refine_zero_iterations_is_identity(tiny, cells, lut5):
    params, active, targets = _fit_inputs(tiny, cells)
    out = refine(params, active, targets, 0, lut5)
    assert torch.equal(out, params) and out is not params


@pytest.mark.unit
def test_refine_lowers_the_loss_without_touching_input(tiny, cells, lut5):
    params, active, targets = _fit_inputs(tiny, cells)
    before = params.clone()
    out = refine(params, active, targets, REFINE_CHECK_ITERS, lut5)
    _, m0 = vsf.evaluate(refine(params, active, targets, 0, lut5), active, targets, "diff7", lut5)
    _, m1 = vsf.evaluate(out, active, targets, "diff7", lut5)
    assert torch.equal(params, before)
    assert sum(m["nrmse"] for m in m1) < sum(m["nrmse"] for m in m0)


@pytest.fixture
def bench(tiny, cells, lut5, tmp_path):
    model, cfg, _ = tiny
    _write_sample(tmp_path / "s16.pkl", cells)
    out = tmp_path / "out"
    df = benchmark(model, cfg, {"2016": tmp_path / "s16.pkl"}, {}, (0, ITERS), torch.device("cpu"), out,
                   lut5=lut5, threshold=THRESHOLD)
    return df, out


@pytest.mark.integration
def test_benchmark_writes_e5_schema_plus_new_columns(bench):
    df, out = bench
    for level in (0, ITERS):
        frame = pd.read_csv(out / f"refine_{level}" / "cells_50_200.csv")
        assert set(E5_COLUMNS) <= set(frame.columns)
        assert {"refine_iters", "encode_s", "refine_s", "year"} <= set(frame.columns)
        assert set(frame["refine_iters"]) == {level} and set(frame["id"]) == {0, 1, 3}
    assert set(pd.read_csv(out / f"refine_{ITERS}" / "cells_200_1000.csv")["id"]) == {2}
    assert len(df) == 8 and (df["encode_s"] > 0).all()


@pytest.mark.integration
def test_benchmark_refinement_curve_has_one_row_per_bin_and_level(bench):
    _, out = bench
    curve = pd.read_csv(out / "refinement_curve.csv")
    assert list(curve.columns) == ["bin", "refine_iters", "median_nrmse", "p90_nrmse", "median_K",
                                   "median_params_per_area", "median_seconds"]
    assert len(curve) == 4
    assert (curve["median_seconds"] > 0).all()
    a = curve[curve["bin"] == "50-200"].set_index("refine_iters")
    assert a.loc[ITERS, "median_nrmse"] < a.loc[0, "median_nrmse"]


def _frame(nrmse, ppa=0.3, ids=None, bin_="50-200", iters=None, secs=None, year="2016"):
    ids = list(range(len(nrmse))) if ids is None else ids
    d = pd.DataFrame(dict(year=year, id=ids, bin=bin_, nrmse=nrmse, compression_ratio=ppa))
    if iters is not None:
        d["refine_iters"] = iters
    if secs is not None:
        d["encode_s"], d["refine_s"] = secs
    return d


@pytest.mark.unit
@pytest.mark.parametrize("delta,iters,expected", [(0.019, 0, True), (0.021, 0, False), (0.004, 50, True),
                                                  (0.006, 50, False), (0.019, 50, False)])
def test_adoption_nrmse_thresholds(delta, iters, expected):
    e5 = _frame([0.2, 0.3, 0.4])
    e7 = _frame([0.2 + delta, 0.3 + delta, 0.4 + delta], iters=iters)
    row = adoption_table(e7, e5).iloc[0]
    assert row["paired_diff_median_info"] == pytest.approx(delta) and bool(row["passed"]) is expected


@pytest.mark.unit
@pytest.mark.parametrize("ppa,expected", [(0.329, True), (0.331, False)])
def test_adoption_params_per_area_limit(ppa, expected):
    e5 = _frame([0.2, 0.3], ppa=0.3)
    e7 = _frame([0.2, 0.3], ppa=ppa, iters=0)
    row = adoption_table(e7, e5).iloc[0]
    assert row["ppa_ratio"] == pytest.approx(ppa / 0.3) and bool(row["passed"]) is expected


@pytest.mark.unit
def test_verdict_uses_unpaired_medians_while_paired_difference_is_informative():
    e5 = _frame([0.1, 0.2, 0.9], ids=[0, 1, 2])
    e7 = _frame([0.1, 0.2, 0.2], ids=[0, 1, 2], iters=0)
    e7_other = _frame([0.12, 0.22, 0.5], ids=[0, 1, 5], iters=0)  # paires 0 et 1 : diff 0.02 ; médiane 0.22
    row = adoption_table(e7_other, e5).iloc[0]
    assert row["paired_diff_median_info"] == pytest.approx(0.02) and row["n_paired"] == 2
    assert row["e5_nrmse"] == pytest.approx(0.2) and row["e7_nrmse"] == pytest.approx(0.22)
    assert bool(row["passed"]) is True
    far = _frame([0.1, 0.2, 0.9], ids=[0, 1, 2], iters=0)  # médiane E7 0.2 identique à E5
    assert adoption_table(far, e5).iloc[0]["paired_diff_median_info"] == 0
    divergent = _frame([0.05, 0.05, 0.95], ids=[0, 1, 2], iters=0)  # diff pairée médiane -0.05 mais médiane 0.05
    assert adoption_table(divergent, e5).iloc[0]["paired_diff_median_info"] == pytest.approx(-0.05)
    assert e7.shape[0] == 3


@pytest.mark.unit
def test_pairing_ignores_unmatched_ids_in_informative_difference():
    e5 = _frame([0.2, 0.3, 0.9], ids=[0, 1, 2])
    e7 = _frame([0.25, 0.35, 0.1, 0.1], ids=[0, 1, 7, 8], iters=0)
    row = adoption_table(e7, e5).iloc[0]
    assert row["n_paired"] == 2 and row["paired_diff_median_info"] == pytest.approx(0.05)


@pytest.mark.unit
def test_every_year_must_pass():
    e5 = pd.concat([_frame([0.2, 0.3], year="2016"), _frame([0.2, 0.3], year="2017")])
    e7 = pd.concat([_frame([0.3, 0.4], year="2016"), _frame([0.2, 0.3], year="2017")]).assign(refine_iters=0)
    by_year = adoption_by_year(e7, e5).set_index("year")
    assert not by_year.loc["2016", "passed"] and by_year.loc["2017", "passed"]
    combined = adoption_table(e7, e5).iloc[0]
    assert bool(combined["passed"]) is False and combined["years_passed"] == 1 and combined["n_years"] == 2


@pytest.mark.unit
def test_non_finite_e7_nrmse_is_counted_and_fails():
    e5 = _frame([0.2, 0.3, 0.4])
    e7 = _frame([0.2, 0.3, float("nan")], iters=0)
    row = adoption_by_year(e7, e5).iloc[0]
    assert row["n_nonfinite"] == 1 and bool(row["passed"]) is False


@pytest.mark.unit
def test_adoption_one_row_per_bin_and_level():
    e5 = pd.concat([_frame([0.2, 0.3], bin_="50-200"), _frame([0.4, 0.5], ids=[2, 3], bin_=">=5000")])
    e7 = pd.concat([_frame([0.2, 0.3], bin_="50-200", iters=i) for i in (0, 25)]
                   + [_frame([0.4, 0.5], ids=[2, 3], bin_=">=5000", iters=i) for i in (0, 25)])
    t = adoption_table(e7, e5)
    assert sorted(zip(t["bin"], t["refine_iters"])) == [("50-200", 0), ("50-200", 25), (">=5000", 0), (">=5000", 25)]


@pytest.mark.unit
def test_adoption_without_pairs_fails():
    row = adoption_table(_frame([0.2], ids=[5], iters=0), _frame([0.2], ids=[6])).iloc[0]
    assert row["n_paired"] == 0 and bool(row["passed"]) is True


@pytest.mark.unit
def test_duplicate_e5_ids_raise():
    with pytest.raises(ValueError, match="doublon"):
        adoption_table(_frame([0.2], iters=0), _frame([0.2, 0.3], ids=[0, 0]))


@pytest.mark.unit
def test_e5_rows_must_match_sample_cells(make_cell):
    cells = {"2016": [make_cell(0), make_cell(1)]}
    ok = pd.DataFrame(dict(year="2016", id=[0, 1], area=[c["area"] for c in cells["2016"]], tag="syn",
                           date="2016-01-01"))
    check_e5_matches_samples(ok, cells)
    with pytest.raises(ValueError, match="aire"):
        check_e5_matches_samples(ok.assign(area=ok["area"] + 1), cells)
    with pytest.raises(ValueError, match="tag"):
        check_e5_matches_samples(ok.assign(tag="autre"), cells)
    with pytest.raises(ValueError, match="absent"):
        check_e5_matches_samples(ok.assign(id=[0, 9]), cells)


def _timing(secs_by_level):
    return pd.concat([_frame([0.1], bin_=b, iters=lv, secs=s) for lv, s in secs_by_level.items() for b in CENSUS])


@pytest.mark.unit
def test_throughput_projection_arithmetic():
    e7 = _timing({0: (0.01, 0.0), 100: (0.01, 0.02)})
    t = throughput_projection(e7, "cuda").set_index("refine_iters")
    total_cells = sum(CENSUS.values())
    assert t.loc[0, "hours"] == pytest.approx(total_cells * 0.01 / 3600)
    assert t.loc[100, "hours"] == pytest.approx(total_cells * 0.03 / 3600)
    assert t.loc[0, "within_budget"] is True or bool(t.loc[0, "within_budget"])
    assert (t["device"] == "cuda").all()
    slow = throughput_projection(e7, "cuda", budget_h=20).set_index("refine_iters")
    assert not slow.loc[100, "within_budget"] and slow.loc[0, "within_budget"]


@pytest.mark.unit
def test_throughput_projection_adds_crop_cost_per_cell():
    e7 = _timing({0: (0.01, 0.0)})
    t = throughput_projection(e7, "cuda", crop_s=0.01)
    assert t.iloc[0]["hours"] == pytest.approx(sum(CENSUS.values()) * 0.02 / 3600)


@pytest.mark.unit
def test_throughput_verdict_is_undetermined_off_cuda(caplog):
    with caplog.at_level("WARNING"):
        t = throughput_projection(_timing({0: (0.01, 0.0)}), "cpu")
    assert np.isnan(t.iloc[0]["within_budget"]) and t.iloc[0]["device"] == "cpu"
    assert "cuda" in caplog.text


@pytest.mark.unit
def test_throughput_non_strict_gives_nan_for_missing_class(caplog):
    e7 = _timing({0: (0.01, 0.0)})
    t = throughput_projection(e7[e7["bin"] != ">=5000"], "cuda", strict=False)
    assert np.isnan(t.iloc[0]["hours"]) and np.isnan(t.iloc[0]["within_budget"])


@pytest.mark.unit
def test_throughput_requires_every_census_class():
    e7 = _timing({0: (0.01, 0.0)})
    with pytest.raises(ValueError, match="classe"):
        throughput_projection(e7[e7["bin"] != ">=5000"], "cuda")


@pytest.mark.unit
def test_crop_extraction_cost_is_measured_positive():
    assert 0 < crop_extraction_seconds(repeats=1) < 10


@pytest.mark.unit
def test_lut_key_must_be_default_unless_overridden():
    check_lut_key(LogWLutSpec(), allow_override=False)
    with pytest.raises(ValueError, match="ace17c5890"):
        check_lut_key(LogWLutSpec(n=32), allow_override=False)
    check_lut_key(LogWLutSpec(n=32), allow_override=True)


@pytest.mark.integration
def test_benchmark_rejects_e5_year_missing_from_samples(tiny, tmp_path, lut5, cells):
    model, cfg, _ = tiny
    _write_sample(tmp_path / "s.pkl", cells)
    with pytest.raises(ValueError, match="2017"):
        benchmark(model, cfg, {"2016": tmp_path / "s.pkl"}, {"2017": (tmp_path, tmp_path)}, (0,),
                  torch.device("cpu"), tmp_path / "o", lut5=lut5)


@pytest.mark.unit
def test_census_matches_dataset_sizes():
    assert CENSUS == {"50-200": 2_700_000, "200-1000": 1_100_000, "1000-5000": 270_000, ">=5000": 80_000}


@pytest.mark.integration
@pytest.mark.skipif(not (E5_ROOT / "adaptive_2016_c050").exists() or not (E5_ROOT / "adaptive_2017").exists(),
                    reason="archive E5 absente")
def test_adoption_of_e5_against_itself_passes_with_zero_difference():
    dirs = {y: (E5_ROOT / f"adaptive_{y}_c050", E5_ROOT / f"adaptive_{y}") for y in ("2016", "2017")}
    e5 = load_e5(dirs)
    assert len(e5) == 320 and set(e5["bin"]) == {"50-200", "200-1000", "1000-5000", ">=5000"}
    t = adoption_table(e5.assign(refine_iters=0), e5)
    assert (t["paired_diff_median_info"] == 0).all() and (t["n_paired"] == 80).all() and t["passed"].all()
