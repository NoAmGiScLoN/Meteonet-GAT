"""Tests de src/optimizer_check.py : plan, grille appariée, table de choix, pile-up de w."""
import copy
import inspect
import logging
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import optimizer_check as oc
import variable_source_fit as vsf

TINY_FLAGS = ["--n-w", "4", "--n-r", "32", "--lut-n", "32", "--lut-steps", "60"]
TINY_SPEC_KW = dict(n_w=4, n_r=32, n=32, steps=60)
BASE = "fixed:0.05"
BASELINE = (BASE, vsf.ITERS_BY_BIN)
REAL_FIT_SIGNATURE = inspect.signature(vsf.fit_group)  # capturée avant tout monkeypatch : détecte la dérive


# ---------------------------------------------------------------- parse_plan_entry
@pytest.mark.unit
def test_parse_plan_entry_fixed_and_cosine():
    fixed = oc.parse_plan_entry("fixed:0.05@200,500")
    assert fixed == [(oc.OptSetting("fixed:0.05", 0.05, None), 200), (oc.OptSetting("fixed:0.05", 0.05, None), 500)]
    cos = oc.parse_plan_entry("cosine:0.1@200")
    assert cos == [(oc.OptSetting("cosine:0.1", 0.1, "cosine"), 200)]


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["", "fixed", "fixed:0.05", "fixed:0.05@", "fixed@200", "adam:0.05@200",
                                 "fixed:abc@200", "fixed:-1@200", "fixed:0@200", "fixed:nan@200",
                                 "fixed:0.05@0", "fixed:0.05@x", "fixed:0.05@2.5", "fixed:0.05@200@5",
                                 "fixed:0.05@200,200"])
def test_parse_plan_entry_rejects_invalid(bad):
    with pytest.raises(ValueError):
        oc.parse_plan_entry(bad)


@pytest.mark.unit
def test_plan_default_is_the_approved_grid():
    items = [x for e in oc.PLAN_DEFAULT for x in oc.parse_plan_entry(e)]
    assert [(s.name, i) for s, i in items] == [("fixed:0.05", 200), ("fixed:0.05", 500), ("cosine:0.05", 200),
                                               ("cosine:0.05", 500), ("cosine:0.1", 200)]


# ---------------------------------------------------------------- run_grid (spy)
def _spy(calls):
    def fake_fit_group(*args, **kwargs):
        bound = REAL_FIT_SIGNATURE.bind(*args, **kwargs)  # TypeError si l'appel ne colle plus à vsf.fit_group
        bound.apply_defaults()
        cells, kmax, iters = bound.arguments["cells"], bound.arguments["kmax"], bound.arguments["iters"]
        lr, lr_schedule = bound.arguments["lr"], bound.arguments["lr_schedule"]
        calls.append(dict(ids=[c["id"] for c in cells], kmax=kmax, iters=iters, lr=lr, schedule=lr_schedule))
        recs = [dict(bin=c["bin"], cell=c["id"], area=c["area"], K=K, nrmse=0.5 / K, iou=0.1, mass_err=0.0,
                     peak_err=0.0) for c in cells for K in (1, kmax)]
        prm = torch.zeros(len(cells), kmax, 8)
        act = torch.ones(len(cells), kmax)
        return recs, {}, (prm, act)
    return fake_fit_group


@pytest.mark.unit
def test_run_grid_rows_calls_and_pairing(monkeypatch, make_cell):
    calls: list[dict] = []
    monkeypatch.setattr(vsf, "fit_group", _spy(calls))
    cells = [make_cell(i) for i in range(3)]
    plan = ("fixed:0.05@2,3", "cosine:0.1@2")
    df, params = oc.run_grid(cells, plan, None, {"50-200": 4}, {"50-200": 2})
    assert len(df) == 3 * 2 * 3  # cellules x K (1 et cap) x (setting, iters)
    assert list(df.columns) == oc.RESULT_COLUMNS
    per = {k: sorted(g.cell.unique()) for k, g in df.groupby(["setting", "iters"])}
    assert len(per) == 3 and all(v == [0, 1, 2] for v in per.values())
    real = [c for c in calls if c["kmax"] == 4]
    assert {(c["lr"], c["schedule"], c["iters"]) for c in real} == {(0.05, None, 2), (0.05, None, 3),
                                                                    (0.1, "cosine", 2)}
    assert [c["ids"] for c in real if c["iters"] == 2 and c["lr"] == 0.05] == [[0, 1], [2]]
    assert set(params) == {("fixed:0.05", 2, "50-200"), ("fixed:0.05", 3, "50-200"), ("cosine:0.1", 2, "50-200")}
    assert set(params[("cosine:0.1", 2, "50-200")]) == {0, 1, 2}
    cos = df[df.setting == "cosine:0.1"]
    assert set(cos.schedule) == {"cosine"} and set(cos.lr) == {0.1}


@pytest.mark.unit
def test_run_grid_does_not_mutate_inputs_and_writes_csv(monkeypatch, make_cell, tmp_path):
    monkeypatch.setattr(vsf, "fit_group", _spy([]))
    cells = [make_cell(0)]
    plan = ["fixed:0.05@2"]
    snapshot = copy.deepcopy(cells)
    oc.run_grid(cells, plan, None, {"50-200": 3}, {"50-200": 2}, csv_path=tmp_path / "g.csv")
    assert plan == ["fixed:0.05@2"]
    for after, before in zip(cells, snapshot):
        assert after.keys() == before.keys()
        assert all(np.array_equal(after[k], before[k]) for k in before)
    assert len(pd.read_csv(tmp_path / "g.csv")) == 2


@pytest.mark.unit
def test_run_grid_validation(monkeypatch, make_cell):
    monkeypatch.setattr(vsf, "fit_group", _spy([]))
    with pytest.raises(ValueError, match="K max"):
        oc.run_grid([make_cell(0, bin_name="zzz")], ["fixed:0.05@2"], None, {"50-200": 3}, {"50-200": 2})
    with pytest.raises(ValueError, match="plan"):
        oc.run_grid([make_cell(0)], [], None, {"50-200": 3}, {"50-200": 2})
    with pytest.raises(ValueError, match="dupliqu"):
        oc.run_grid([make_cell(0)], ["fixed:0.05@2", "fixed:0.05@2,3"], None, {"50-200": 3}, {"50-200": 2})


def _failing_spy(fail_when):
    inner = _spy([])

    def fit(cells, kind, kmax, iters, *args, **kwargs):
        if fail_when(cells, kmax, iters, kwargs):
            raise RuntimeError("boom")
        return inner(cells, kind, kmax, iters, *args, **kwargs)
    return fit


@pytest.mark.unit
def test_run_grid_persists_csv_and_params_before_a_failing_second_bin(monkeypatch, make_cell, tmp_path):
    monkeypatch.setattr(vsf, "fit_group", _failing_spy(lambda cells, kmax, it, kw: cells[0]["bin"] == "200-1000"
                                                       and kmax > 1))
    cells = [make_cell(0), make_cell(1, bin_name="200-1000")]
    with pytest.raises(RuntimeError, match="boom"):
        oc.run_grid(cells, ["fixed:0.05@2"], None, {"50-200": 3, "200-1000": 4}, {"50-200": 2, "200-1000": 2},
                    csv_path=tmp_path / "g.csv", params_path=tmp_path / "p.pkl")
    assert set(pd.read_csv(tmp_path / "g.csv").bin) == {"50-200"}
    assert set(pickle.loads((tmp_path / "p.pkl").read_bytes())) == {("fixed:0.05", 2, "50-200")}


@pytest.mark.unit
def test_run_grid_rewrites_csv_after_each_fit_not_only_per_bin(monkeypatch, make_cell, tmp_path):
    monkeypatch.setattr(vsf, "fit_group", _failing_spy(lambda cells, kmax, it, kw: kmax > 1 and kw["lr"] == 0.1))
    with pytest.raises(RuntimeError, match="boom"):
        oc.run_grid([make_cell(0)], ["fixed:0.05@2", "cosine:0.1@2"], None, {"50-200": 3}, {"50-200": 2},
                    csv_path=tmp_path / "g.csv")
    assert set(pd.read_csv(tmp_path / "g.csv").setting) == {"fixed:0.05"}


@pytest.mark.unit
def test_run_grid_resets_cuda_peak_memory_per_bin(monkeypatch, make_cell):
    monkeypatch.setattr(vsf, "fit_group", _spy([]))
    resets: list[int] = []
    monkeypatch.setattr(vsf, "DEVICE", SimpleNamespace(type="cuda"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: resets.append(1))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    cells = [make_cell(0), make_cell(1, bin_name="200-1000")]
    oc.run_grid(cells, ["fixed:0.05@2"], None, {"50-200": 3, "200-1000": 4}, {"50-200": 2, "200-1000": 2})
    assert len(resets) == 2


@pytest.mark.unit
def test_warn_short_bins(make_cell, caplog):
    cells = [make_cell(0), make_cell(1, bin_name="200-1000")]
    with caplog.at_level(logging.WARNING):
        oc.warn_short_bins(cells, {"50-200": 3, "200-1000": 4}, per_bin=2)
    assert "50-200" in caplog.text and "1 cellule" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        oc.warn_short_bins(cells, {"50-200": 3}, per_bin=1)
    assert caplog.text == ""


# ---------------------------------------------------------------- choice_table
def grid_df(values: dict[tuple[str, int], list[float]], bin_name="50-200", k_cap=8) -> pd.DataFrame:
    """values : (setting, iters) -> nrmse par cellule au K cap (plus une ligne K=1 sans effet)."""
    rows = []
    for (setting, it), vals in values.items():
        for cell, v in enumerate(vals):
            for K, val in ((k_cap, v), (1, 9.0)):
                rows.append(dict(bin=bin_name, cell=cell, area=100, K=K, iters=it, setting=setting, nrmse=val))
    return pd.DataFrame(rows)


def _row(table, setting, iters, bin_name="50-200"):
    sel = table[(table.bin == bin_name) & (table.setting == setting) & (table.iters == iters)]
    assert len(sel) == 1
    return sel.iloc[0]


KW = dict(gain_min=0.005, tail_tol=0.002, cheap_tol=0.002)


@pytest.mark.unit
def test_choice_table_adopts_clear_gain():
    base = [0.30] * 5
    df = grid_df({(BASE, 200): base, ("cosine:0.05", 200): [v - 0.01 for v in base]})
    t = oc.choice_table(df, BASELINE, **KW)
    r = _row(t, "cosine:0.05", 200)
    assert r.adopt and r.median_gain == pytest.approx(0.01) and r.p10_gain == pytest.approx(0.01)
    assert _row(t, "cosine:0.05", 200).recommended and not _row(t, BASE, 200).recommended


@pytest.mark.unit
def test_choice_table_rejects_bad_tail_and_recommends_baseline():
    base = [0.30] * 10
    cand = [0.29] * 8 + [0.34] * 2  # médiane +0.01 mais 20 % des cellules à -0.04
    df = grid_df({(BASE, 200): base, ("cosine:0.1", 200): cand})
    t = oc.choice_table(df, BASELINE, **KW)
    r = _row(t, "cosine:0.1", 200)
    assert r.median_gain >= 0.005 and r.p10_gain < -0.002 and not r.adopt
    assert _row(t, BASE, 200).recommended


@pytest.mark.unit
def test_choice_table_rejects_small_median_gain():
    df = grid_df({(BASE, 200): [0.3] * 4, ("cosine:0.05", 200): [0.298] * 4})
    assert not _row(oc.choice_table(df, BASELINE, **KW), "cosine:0.05", 200).adopt


@pytest.mark.unit
def test_choice_table_cheap_match_for_large_bins_only():
    name = "1000-5000"
    vals = {(BASE, 500): [0.20] * 4, (BASE, 200): [0.22] * 4, ("cosine:0.05", 200): [0.2005] * 4}
    t = oc.choice_table(grid_df(vals, name, 48), BASELINE, **KW)
    cheap = _row(t, "cosine:0.05", 200, name)
    assert cheap.cheap_match and not cheap.adopt  # itérations différentes de la baseline : pas d'adoption
    assert not _row(t, BASE, 200, name).cheap_match  # fixed@200 est 0.02 plus loin
    small = oc.choice_table(grid_df({(BASE, 200): [0.2] * 4, ("cosine:0.05", 200): [0.2] * 4}), BASELINE, **KW)
    assert not _row(small, "cosine:0.05", 200).cheap_match


@pytest.mark.unit
def test_choice_table_recommends_best_adopted():
    base = [0.30] * 4
    df = grid_df({(BASE, 200): base, ("cosine:0.05", 200): [0.29] * 4, ("cosine:0.1", 200): [0.28] * 4})
    t = oc.choice_table(df, BASELINE, **KW)
    assert _row(t, "cosine:0.1", 200).recommended and t.recommended.sum() == 1


@pytest.mark.unit
def test_choice_table_incomplete_pairing_raises():
    df = grid_df({(BASE, 200): [0.3] * 4, ("cosine:0.05", 200): [0.29] * 3})
    with pytest.raises(ValueError, match="NaN ou cellule manquante"):
        oc.choice_table(df, BASELINE, **KW)


@pytest.mark.unit
def test_choice_table_nan_baseline_raises_but_nan_candidate_is_not_adoptable(caplog):
    df = grid_df({(BASE, 200): [0.3] * 4, ("cosine:0.05", 200): [0.2, 0.2, 0.2, np.nan]})
    with caplog.at_level(logging.WARNING):
        t = oc.choice_table(df, BASELINE, **KW)
    assert not _row(t, "cosine:0.05", 200).adopt and _row(t, BASE, 200).recommended
    assert "NaN" in caplog.text
    bad = grid_df({(BASE, 200): [0.3, 0.3, np.nan], ("cosine:0.05", 200): [0.2] * 3})
    with pytest.raises(ValueError, match="NaN ou cellule manquante"):
        oc.choice_table(bad, BASELINE, **KW)


@pytest.mark.unit
def test_choice_table_default_baseline_and_baseline_iters_column():
    df = grid_df({(BASE, vsf.ITERS_BY_BIN["50-200"]): [0.3] * 3})
    t = oc.choice_table(df, gain_min=0.005, tail_tol=0.002, cheap_tol=0.002)
    assert set(t.baseline_iters) == {vsf.ITERS_BY_BIN["50-200"]}


@pytest.mark.unit
def test_choice_table_warns_when_cheap_match_cannot_be_evaluated(caplog):
    name = "1000-5000"
    df = grid_df({(BASE, 200): [0.2] * 3, ("cosine:0.05", 200): [0.2] * 3}, name, 48)
    with caplog.at_level(logging.WARNING):
        t = oc.choice_table(df, (BASE, {name: 200}), **KW)
    assert "cheap_match" in caplog.text and name in caplog.text and not t.cheap_match.any()


# ---- seuils exacts (float64 : bases dyadiques, écarts = constantes du module)
def _gains_df(gains, base=0.25, iters=200):
    return grid_df({(BASE, iters): [base] * len(gains), ("cosine:0.05", iters): [base - g for g in gains]})


def _adopt(gains):
    t = oc.choice_table(_gains_df(gains), (BASE, {"50-200": 200}), oc.GAIN_MIN, oc.TAIL_TOL, oc.CHEAP_TOL)
    return bool(_row(t, "cosine:0.05", 200).adopt)


@pytest.mark.unit
def test_threshold_median_gain_exactly_at_limit():
    assert _adopt([oc.GAIN_MIN] * 5)
    assert not _adopt([oc.GAIN_MIN - 1e-9] * 5)


@pytest.mark.unit
def test_threshold_p10_exactly_at_limit():
    def gains(p10):  # 11 cellules : le 10e centile (interpolation linéaire) est la 2e plus petite valeur
        return [-0.01, p10] + [0.01] * 9
    assert _adopt(gains(-oc.TAIL_TOL))
    assert not _adopt(gains(-oc.TAIL_TOL - 1e-9))


@pytest.mark.unit
@pytest.mark.parametrize("diff, expected", [(oc.CHEAP_TOL, True), (-oc.CHEAP_TOL, True),
                                            (oc.CHEAP_TOL + 1e-9, False), (-oc.CHEAP_TOL - 1e-9, False)])
def test_threshold_cheap_diff_exactly_at_limit(diff, expected):
    name = "1000-5000"
    vals = {(BASE, 500): [0.25] * 3, (BASE, 200): [0.3] * 3, ("cosine:0.05", 200): [0.25 + diff] * 3}
    t = oc.choice_table(grid_df(vals, name, 48), (BASE, {name: 500}), oc.GAIN_MIN, oc.TAIL_TOL, oc.CHEAP_TOL)
    assert bool(_row(t, "cosine:0.05", 200, name).cheap_match) is expected


@pytest.mark.unit
@pytest.mark.parametrize("base_iters, adopted_iters, rejected_iters", [(500, 500, 200), (200, 200, 500)])
def test_adoption_requires_the_baseline_iteration_count(base_iters, adopted_iters, rejected_iters):
    vals = {(BASE, 200): [0.3] * 4, (BASE, 500): [0.3] * 4,
            ("cosine:0.05", 200): [0.29] * 4, ("cosine:0.05", 500): [0.29] * 4}
    t = oc.choice_table(grid_df(vals), (BASE, {"50-200": base_iters}), **KW)
    assert _row(t, "cosine:0.05", adopted_iters).adopt and not _row(t, "cosine:0.05", rejected_iters).adopt


@pytest.mark.unit
def test_choice_table_missing_baseline_raises():
    df = grid_df({("cosine:0.05", 200): [0.3] * 3})
    with pytest.raises(ValueError, match="baseline"):
        oc.choice_table(df, BASELINE, **KW)


# ---------------------------------------------------------------- pileup_table / plot
@pytest.mark.unit
def test_pileup_table_known_raw_w():
    p = np.zeros((4, 8))
    p[:, 7] = [-10.0, 10.0, 0.0, 0.0]  # 1 bas, 1 haut, 2 au milieu
    t = oc.pileup_table({("cosine:0.1", 200, "50-200"): {0: p[:2], 1: p[2:]}})
    assert list(t.columns) == ["setting", "iters", "bin", "frac_low", "frac_high", "n_sources"]
    r = t.iloc[0]
    assert (r.setting, r.iters, r.bin, r.n_sources) == ("cosine:0.1", 200, "50-200", 4)
    assert r.frac_low == pytest.approx(0.25) and r.frac_high == pytest.approx(0.25)


@pytest.mark.unit
def test_plot_grid_writes_png(tmp_path):
    df = grid_df({(BASE, 200): [0.3] * 3, ("cosine:0.05", 200): [0.2] * 3})
    oc.plot_grid(df, tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 0


def _capture_axes(monkeypatch):
    seen = []
    real_close = oc.plt.close
    monkeypatch.setattr(oc.plt, "close", lambda fig: (seen.append(fig.axes[0]), real_close(fig)))
    return seen


@pytest.mark.unit
def test_plot_grid_uses_label_index_positions_markers_only(monkeypatch, tmp_path):
    seen = _capture_axes(monkeypatch)
    full = grid_df({(BASE, 200): [0.3] * 3, (BASE, 500): [0.3] * 3, ("cosine:0.05", 200): [0.2] * 3})
    partial = grid_df({(BASE, 200): [0.4] * 3, ("cosine:0.05", 200): [0.1] * 3}, bin_name="200-1000")  # sans fixed@500
    oc.plot_grid(pd.concat([full, partial]), tmp_path / "g.png")
    ax = seen[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["cosine:0.05@200", "fixed:0.05@200", "fixed:0.05@500"]
    by_bin = {ln.get_label().split(" ")[0]: ln for ln in ax.lines}
    assert list(by_bin["50-200"].get_xdata()) == [0, 1, 2]
    assert list(by_bin["200-1000"].get_xdata()) == [0, 1]  # indices explicites, pas un décalage
    assert all(ln.get_linestyle() == "None" for ln in ax.lines)


@pytest.mark.unit
def test_plot_grid_empty_df_does_not_warn(tmp_path):
    empty = pd.DataFrame(columns=["bin", "cell", "area", "K", "iters", "setting", "nrmse"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        oc.plot_grid(empty, tmp_path / "g.png")
    assert (tmp_path / "g.png").exists()


# ---------------------------------------------------------------- réel (CPU) et main
@pytest.mark.integration
def test_run_grid_real_tiny_cpu(tmp_path, make_cell):
    from profile_lut import LogWLutSpec
    lut = vsf.lut_for_arm("diff7", tmp_path, LogWLutSpec(**TINY_SPEC_KW))
    cells = [make_cell(0), make_cell(1)]
    df, params = oc.run_grid(cells, ["fixed:0.05@2", "cosine:0.05@2"], lut, {"50-200": 2}, {"50-200": 2})
    assert len(df) == 2 * 2 * 2 and np.isfinite(df.nrmse).all()
    assert params[("cosine:0.05", 2, "50-200")][0].shape[1] == 8


def _write_sample(tmp_path, make_cell):
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps([make_cell(0), make_cell(1), make_cell(2, bin_name="200-1000")]))
    return path


@pytest.mark.integration
def test_main_writes_all_outputs_and_restores_device(tmp_path, make_cell):
    out = tmp_path / "out"
    before = vsf.DEVICE
    oc.main(["--sample", str(_write_sample(tmp_path, make_cell)), "--per-bin", "2", "--plan", "fixed:0.05@2",
             "cosine:0.05@2", "--baseline-iters", "2", "--bins", "50-200", "200-1000", "--device", "cpu", "--out", str(out), *TINY_FLAGS])
    assert vsf.DEVICE is before
    for name in ("optimizer_grid.csv", "optimizer_pileup.csv", "optimizer_choice.csv", "optimizer_grid.png",
                 "params_grid.pkl"):
        assert (out / name).exists(), name
    choice = pd.read_csv(out / "optimizer_choice.csv")
    assert set(choice.bin) == {"50-200", "200-1000"} and set(choice.baseline_iters) == {2}
    assert set(pickle.loads((out / "params_grid.pkl").read_bytes())) >= {("fixed:0.05", 2, "50-200")}


@pytest.mark.unit
def test_main_restores_device_on_failure(tmp_path, make_cell, monkeypatch):
    monkeypatch.setattr(vsf, "lut_for_arm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    before = vsf.DEVICE
    with pytest.raises(RuntimeError):
        oc.main(["--sample", str(_write_sample(tmp_path, make_cell)), "--plan", "fixed:0.05@200", "--bins",
                 "50-200", "--device", "cpu", "--out", str(tmp_path / "o")])
    assert vsf.DEVICE is before


@pytest.mark.unit
def test_main_rejects_bad_plan_and_missing_baseline_before_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(vsf, "fit_group", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fit")))
    with pytest.raises(ValueError):
        oc.main(["--sample", "x", "--plan", "bogus", "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="baseline"):
        oc.main(["--sample", "x", "--plan", "cosine:0.05@200", "--bins", "50-200", "--out", str(tmp_path)])
