"""Décisions D17 (bras par défaut = diff7) et D19/D22 (itérations par classe de taille ; D22 : 500 dès 200 px)."""
import functools
import pickle
import sys

import pytest

import profile_lut as pl
import variable_source_fit as vsf

TINY = dict(n=32, steps=60, n_r=32)
EXPECTED_ITERS = {"50-200": 200, "200-1000": 500, "1000-5000": 500, ">=5000": 500}


@pytest.mark.unit
def test_iters_by_bin_keys_match_bins_and_d19_values():
    assert set(vsf.ITERS_BY_BIN) == {b[0] for b in vsf.BINS}
    assert vsf.ITERS_BY_BIN == EXPECTED_ITERS


@pytest.mark.unit
@pytest.mark.parametrize("name", list(EXPECTED_ITERS))
def test_iters_for_bin_uses_table_without_override(name):
    assert vsf.iters_for_bin(name, None) == EXPECTED_ITERS[name]


@pytest.mark.unit
@pytest.mark.parametrize("name", list(EXPECTED_ITERS))
def test_iters_for_bin_override_applies_to_every_bin(name):
    assert vsf.iters_for_bin(name, 7) == 7


@pytest.mark.unit
@pytest.mark.parametrize("name,override", [("nope", None), ("nope", 5), ("50-200", 0), ("50-200", -3)])
def test_iters_for_bin_rejects_unknown_bin_or_bad_override(name, override):
    with pytest.raises(ValueError):
        vsf.iters_for_bin(name, override)


@pytest.mark.unit
def test_parser_defaults_arms_diff7_and_iters_none():
    args = vsf.build_parser().parse_args([])
    assert args.arms == ["diff7"]
    assert args.iters is None


def _run_main_with_spy(tmp_path, monkeypatch, make_cell, extra):
    sample = [make_cell(0, bin_name="50-200"), make_cell(1, bin_name="1000-5000")]
    sample_path = tmp_path / "sample.pkl"
    sample_path.write_bytes(pickle.dumps(sample))
    calls = []
    real = vsf.fit_group

    def spy(cells, kind, kmax, iters, *a, **k):
        calls.append(iters)
        return real(cells, kind, kmax, 1, *a, **k)  # 1 itération réelle : test rapide

    monkeypatch.setattr(vsf, "fit_group", spy)
    monkeypatch.setattr(vsf, "LogWLutSpec", functools.partial(pl.LogWLutSpec, **TINY))
    monkeypatch.setattr(sys, "argv", ["vsf", "--sample", str(sample_path), "--bins", "50-200", "1000-5000",
                                      "--n-w", "3", "--device", "cpu", "--out", str(tmp_path / "run"), *extra])
    vsf.main()
    return calls


@pytest.mark.integration
def test_main_without_iters_uses_iters_by_bin(tmp_path, monkeypatch, make_cell):
    calls = _run_main_with_spy(tmp_path, monkeypatch, make_cell, [])
    assert calls == [200, 500]  # un groupe par classe, bras diff7 par défaut


@pytest.mark.integration
def test_main_with_explicit_iters_applies_to_every_bin(tmp_path, monkeypatch, make_cell):
    calls = _run_main_with_spy(tmp_path, monkeypatch, make_cell, ["--iters", "5"])
    assert calls == [5, 5]
