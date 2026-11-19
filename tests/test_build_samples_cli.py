"""Arguments de scripts/build_samples.py : le comportement par défaut ne change pas, --logw est optionnel."""
import dataclasses

import pytest

import build_samples as bs
import profile_lut as pl


@pytest.mark.unit
def test_default_arguments_do_not_request_log_w_table():
    assert bs.parse_args([]).logw is False


@pytest.mark.unit
def test_logw_flags_default_to_spec_defaults():
    # Act
    args = bs.parse_args(["--logw"])
    # Assert
    d = pl.LogWLutSpec()
    assert args.logw is True
    assert (args.w_min, args.w_max, args.n_w) == (d.w_min, d.w_max, d.n_w)


@pytest.mark.unit
def test_logw_flags_are_parsed():
    args = bs.parse_args(["--logw", "--w-min", "0.05", "--w-max", "0.4", "--n-w", "9"])
    assert (args.w_min, args.w_max, args.n_w) == (0.05, 0.4, 9)


@pytest.mark.unit
def test_spec_from_args_builds_validated_spec():
    spec = bs.spec_from_args(bs.parse_args(["--logw", "--n-w", "9"]))
    assert spec == dataclasses.replace(pl.LogWLutSpec(), n_w=9)


@pytest.mark.unit
def test_spec_from_args_rejects_invalid_range():
    with pytest.raises(ValueError):
        bs.spec_from_args(bs.parse_args(["--logw", "--w-min", "0.6", "--w-max", "0.5"]))


@pytest.mark.unit
def test_build_logw_writes_table_in_given_dir(tmp_path):
    spec = pl.LogWLutSpec(n_w=3, n=32, steps=20, n_r=16)
    path = bs.build_logw(spec, tmp_path)
    assert path == pl.logw_lut_path(tmp_path, spec) and path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("flag,value", [("--w-min", "0.05"), ("--w-max", "0.4"), ("--n-w", "9")])
def test_w_flags_without_logw_are_rejected(flag, value, capsys):
    with pytest.raises(SystemExit):
        bs.parse_args([flag, value])
    assert "--logw" in capsys.readouterr().err
