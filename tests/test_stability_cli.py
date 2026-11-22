"""CLI de stability_test : bras, paires, itérations par classe, tables par bras, validation de --iters."""
import pickle

import pandas as pd
import pytest
import torch

import stability_test as st
import variable_source_fit as vsf
from profile_lut import LogWLutSpec

BIN = "50-200"
N_PAIRS = 2
TINY_LUT = ["--n-w", "3", "--n-r", "16", "--lut-n", "16", "--lut-steps", "20"]
PARAM_WIDTH = {"gauss": 6, "diffusion": 8, "diff7": 8}  # largeur réelle de fit_group (diff7 : colonnes gelées)
SENTINEL_DIFF7 = torch.full((1, 1, 1, 3, 4), 0.5)
SENTINEL_DEFAULT = torch.full((1, 1, 3, 3, 4), 0.25)


NOISE = 1e-3


def fake_params(cell_id, n_pairs, kmax, width):
    """Paramètres déterministes par cellule : base propre à la paire (id % n_pairs) + bruit infime seedé par l'id."""
    base = torch.randn(kmax, width, generator=torch.Generator().manual_seed(100 + cell_id % n_pairs))
    noise = torch.randn(kmax, width, generator=torch.Generator().manual_seed(cell_id))
    return base + NOISE * noise


def write_pairs(path, make_cell, n=N_PAIRS):
    cells = [(make_cell(0), make_cell(1), 0.8) for _ in range(n)]
    path.write_bytes(pickle.dumps(cells))


class Spies:
    def __init__(self):
        self.fit_calls, self.lut_calls = [], []
        self.n_pairs = N_PAIRS


@pytest.fixture
def spies(monkeypatch):
    """Remplace fit_group et lut_for_arm par des espions déterministes ; restaure OUT et DEVICE."""
    s = Spies()
    monkeypatch.setattr(vsf, "OUT", vsf.OUT)
    monkeypatch.setattr(vsf, "DEVICE", vsf.DEVICE)

    def fake_fit(cells, arm, kmax, iters, lut5, *a, **kw):
        s.fit_calls.append(dict(arm=arm, kmax=kmax, iters=iters, lut5=lut5, n=len(cells)))
        b = len(cells)
        params = torch.stack([fake_params(c["id"], s.n_pairs, kmax, PARAM_WIDTH[arm]) for c in cells])
        recs = [dict(cell=c["id"], K=kmax, nrmse=0.1) for c in cells]
        return recs, None, (params, torch.ones(b, kmax))

    def fake_lut(arm, out, spec):
        s.lut_calls.append(dict(arm=arm, out=out, spec=spec))
        return {"gauss": None, "diff7": SENTINEL_DIFF7}.get(arm, SENTINEL_DEFAULT)

    monkeypatch.setattr(vsf, "fit_group", fake_fit)
    monkeypatch.setattr(vsf, "lut_for_arm", fake_lut)
    return s


@pytest.fixture
def run(tmp_path, make_cell, spies):
    pairs = tmp_path / "pairs.pkl"
    write_pairs(pairs, make_cell)

    def _run(*extra):
        st.main(["--out", str(tmp_path), "--pairs", str(pairs), "--device", "cpu", *extra])
        return pd.read_csv(tmp_path / "stability.csv")
    return _run


@pytest.mark.unit
def test_default_main_fits_gauss_and_diffusion_with_200_iters_and_k_fixed(tmp_path, make_cell, spies):
    # Arrange : chemin par défaut des paires, sans --pairs
    write_pairs(tmp_path / f"stability_pairs_{st.PER_BIN_DEFAULT}.pkl", make_cell)
    # Act
    st.main(["--out", str(tmp_path), "--device", "cpu"])
    # Assert
    assert [c["arm"] for c in spies.fit_calls] == ["gauss", "diffusion"]
    assert all(c["iters"] == 200 and c["kmax"] == st.K_FIXED[BIN] for c in spies.fit_calls)
    assert [c["arm"] for c in spies.lut_calls] == ["gauss", "diffusion"]
    assert spies.fit_calls[1]["lut5"] is SENTINEL_DEFAULT
    cols = list(pd.read_csv(tmp_path / "stability.csv").columns)
    assert cols[:9] == ["bin", "arm", "K", "pair", "iou_raw", "nrmse_t", "nrmse_t1", "cost_same", "cost_other"]
    assert cols[9:] == [f"d_{n}" for n in st.NAMES["diffusion"]]


@pytest.mark.unit
def test_arms_diff7_passes_diff7_table_to_fit_group(run, spies):
    run("--arms", "diff7")
    assert [c["arm"] for c in spies.fit_calls] == ["diff7"]
    assert spies.fit_calls[0]["lut5"] is SENTINEL_DIFF7


@pytest.mark.unit
def test_gauss_receives_no_table(run, spies):
    run("--arms", "gauss")
    assert spies.fit_calls[0]["lut5"] is None


@pytest.mark.unit
def test_cost_same_below_cost_other_and_other_pairs_i_with_next(tmp_path, make_cell, spies):
    # Arrange : 3 paires, cellules d'une même paire quasi identiques, paires distinctes entre elles
    n = 3
    spies.n_pairs = n
    pairs_path = tmp_path / "pairs.pkl"
    write_pairs(pairs_path, make_cell, n=n)
    prs = pickle.loads(pairs_path.read_bytes())
    # Act
    st.main(["--out", str(tmp_path), "--pairs", str(pairs_path), "--device", "cpu", "--arms", "gauss"])
    df = pd.read_csv(tmp_path / "stability.csv").sort_values("pair")
    # Assert
    kmax, width = st.K_FIXED[BIN], PARAM_WIDTH["gauss"]
    ones = torch.ones(kmax)
    desc = [st.descriptors(fake_params(k, n, kmax, width), ones, c, "gauss")
            for k, c in enumerate([p[0] for p in prs] + [p[1] for p in prs])]
    assert (df.cost_same < df.cost_other).all()
    for i, row in enumerate(df.itertuples()):
        expected_other, _ = st.match(desc[i], desc[n + (i + 1) % n])
        wrong_other, _ = st.match(desc[i], desc[n + (i + 2) % n])
        assert row.cost_other == pytest.approx(expected_other)
        assert expected_other != pytest.approx(wrong_other)


@pytest.mark.unit
def test_missing_pairs_file_fails_before_building_luts(tmp_path, spies):
    with pytest.raises(FileNotFoundError):
        st.main(["--out", str(tmp_path), "--pairs", str(tmp_path / "absent.pkl"), "--device", "cpu"])
    assert spies.lut_calls == [] and spies.fit_calls == []


@pytest.mark.unit
def test_iters_and_iters_by_bin_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        st.build_parser().parse_args(["--iters", "5", "--iters-by-bin"])


@pytest.mark.unit
def test_duplicate_arms_rejected_with_clear_message(capsys):
    with pytest.raises(SystemExit):
        st.build_parser().parse_args(["--arms", "gauss", "gauss"])
    assert "gauss" in capsys.readouterr().err


@pytest.mark.unit
def test_unknown_arm_exits():
    with pytest.raises(SystemExit):
        st.build_parser().parse_args(["--arms", "foo"])


@pytest.mark.unit
def test_iters_by_bin_uses_iters_by_bin_table(run, spies):
    run("--arms", "gauss", "--iters-by-bin")
    assert spies.fit_calls[0]["iters"] == vsf.ITERS_BY_BIN[BIN]


@pytest.mark.unit
def test_iters_flag_applies_to_all_bins_without_iters_by_bin(run, spies):
    run("--arms", "gauss", "--iters", "7")
    assert spies.fit_calls[0]["iters"] == 7


@pytest.mark.unit
def test_pairs_path_is_used_and_diff7_rows_have_d_w_not_d_tau(run):
    df = run("--arms", "diff7")
    assert "d_w" in df.columns and "d_tau" not in df.columns
    assert set(df.arm) == {"diff7"} and len(df) == N_PAIRS


@pytest.mark.unit
def test_lut_flags_build_spec_passed_to_lut_for_arm(run, spies):
    run("--arms", "diff7", *TINY_LUT, "--w-min", "0.1", "--w-max", "0.4", "--tau7", "0.3")
    spec = spies.lut_calls[0]["spec"]
    assert (spec.n_w, spec.n_r, spec.n, spec.steps) == (3, 16, 16, 20)
    assert (spec.w_min, spec.w_max, spec.tau) == (0.1, 0.4, 0.3)


@pytest.mark.unit
def test_build_arm_luts_maps_each_arm(spies):
    luts = st.build_arm_luts(["gauss", "diff7"], vsf.OUT, LogWLutSpec())
    assert luts == {"gauss": None, "diff7": SENTINEL_DIFF7}


@pytest.mark.unit
def test_main_restores_vsf_out_and_device(run, tmp_path):
    out0, dev0 = vsf.OUT, vsf.DEVICE
    run("--arms", "gauss")
    assert vsf.OUT == out0 and vsf.DEVICE == dev0


@pytest.mark.unit
def test_build_returns_early_after_building_pairs(tmp_path, monkeypatch, spies):
    built = []
    monkeypatch.setattr(st, "build_pairs", lambda per_bin, path: built.append((per_bin, path)))
    st.main(["--build", "--out", str(tmp_path), "--per-bin", "3"])
    assert built == [(3, tmp_path / "stability_pairs_3.pkl")]
    assert spies.fit_calls == [] and not (tmp_path / "stability.csv").exists()


@pytest.mark.integration
def test_tiny_real_cpu_run_gauss_and_diff7_writes_csv(tmp_path, make_cell):
    pairs = tmp_path / "pairs.pkl"
    write_pairs(pairs, make_cell)
    st.main(["--out", str(tmp_path), "--pairs", str(pairs), "--device", "cpu", "--arms", "gauss", "diff7",
             "--iters", "2", *TINY_LUT])
    df = pd.read_csv(tmp_path / "stability.csv")
    assert set(df.arm) == {"gauss", "diff7"} and len(df) == 2 * N_PAIRS
    assert df.loc[df.arm == "diff7", "d_w"].notna().all()


# ---------------------------------------------------------------- 0c : --iters validé à l'analyse
@pytest.mark.unit
@pytest.mark.parametrize("bad", ["0", "-3", "x", "1.5"])
def test_stability_parser_rejects_bad_iters(bad):
    with pytest.raises(SystemExit):
        st.build_parser().parse_args(["--iters", bad])


@pytest.mark.unit
def test_stability_parser_iters_default_and_value():
    assert st.build_parser().parse_args([]).iters == 200
    assert st.build_parser().parse_args(["--iters", "5"]).iters == 5


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["0", "-3", "x"])
def test_vsf_parser_rejects_bad_iters(bad):
    with pytest.raises(SystemExit):
        vsf.build_parser().parse_args(["--iters", bad])


@pytest.mark.unit
def test_vsf_parser_iters_default_and_value():
    assert vsf.build_parser().parse_args([]).iters is None
    assert vsf.build_parser().parse_args(["--iters", "5"]).iters == 5
