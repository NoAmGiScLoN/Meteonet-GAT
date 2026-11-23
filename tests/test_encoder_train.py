"""Boucle d'entraînement de l'encodeur (phase 5 E7) : petits shards, CPU, déterministe."""
import json
import logging
import signal
from dataclasses import replace

import pandas as pd
import pytest
import torch

from encoder_loss import LossConfig
import encoder_train
from encoder_train import TrainConfig, build_parser, main, train
from profile_lut import LogWLutSpec
from shard_helpers import write_tiny_dir
from source_encoder import EncoderConfig

TINY_LUT = LogWLutSpec(n=32, steps=60, n_r=32)


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    return write_tiny_dir(tmp_path_factory.mktemp("tiny") / "s", n_train=12, n_val=6)


@pytest.fixture(scope="module")
def lut_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("lut")


@pytest.fixture
def make_cfg(shards, lut_dir, tmp_path):
    def make(**kw) -> TrainConfig:
        base = dict(shards=shards, out=tmp_path / "run", lut_dir=lut_dir, lut=TINY_LUT, px_budget=4 * 32 * 32,
                    epochs=3, warmup_epochs=1, lr=3e-3, warmup_steps=2, k_train=8, log_every=1, val_every_steps=4,
                    max_steps=6, encoder=EncoderConfig(base_ch=4, k_train=8, cap=16), loss=LossConfig(binarise_ramp_steps=2))
        return TrainConfig(**(base | kw))
    return make


def _train_csv(cfg):
    return pd.read_csv(cfg.out / "train.csv")


@pytest.mark.unit
def test_max_steps_stops_and_writes_checkpoints_and_logs(make_cfg):
    cfg = make_cfg(max_steps=5)
    result = train(cfg)
    assert result.step == 5
    log = _train_csv(cfg)
    assert log["step"].tolist() == [1, 2, 3, 4, 5]
    for col in ("epoch", "loss", "recon", "sparsity", "binarise", "heatmap", "k_active", "params_per_area",
                "px_per_s", "ms_step", "data_ms", "data_wait_ms", "render_ms", "n_chunks", "peak_mem_mb"):
        assert col in log.columns
    assert log["loss"].notna().all()
    assert torch.load(cfg.out / "last.pt", weights_only=False)["step"] == 5
    assert (cfg.out / "best.pt").exists() and (cfg.out / "run_config.json").exists()


@pytest.mark.unit
def test_training_reduces_loss_on_tiny_set(make_cfg):
    cfg = make_cfg(max_steps=24, epochs=12, warmup_epochs=0, val_every_steps=1000)
    train(cfg)
    loss = _train_csv(cfg)["loss"]
    assert loss.iloc[-4:].mean() < loss.iloc[:4].mean()


@pytest.mark.unit
def test_validation_writes_per_class_rows(make_cfg):
    cfg = make_cfg(max_steps=4, val_every_steps=2)
    train(cfg)
    val = pd.read_csv(cfg.out / "val.csv")
    assert {"step", "scope", "bin", "n", "nrmse_median", "k_median", "params_per_area_median"} <= set(val.columns)
    assert set(val["bin"]) == {"50-200", "200-1000", "all"}
    sub = val[val.scope == "sub"]
    assert sorted(set(sub["step"])) == [2, 4]
    assert 3 in set(val[val.scope == "full"]["step"])  # fin d'époque (3 lots par époque)
    row = sub[(sub.step == 4) & (sub.bin == "50-200")].iloc[0]
    assert row["n"] == 4 and row["nrmse_median"] > 0


@pytest.mark.unit
def test_resume_gives_same_next_step_loss_as_uninterrupted(make_cfg, tmp_path):
    full = make_cfg(out=tmp_path / "full", max_steps=6)
    train(full)
    part = make_cfg(out=tmp_path / "part", max_steps=3)
    train(part)
    train(replace(part, max_steps=6), resume=True)
    a, b = _train_csv(full), _train_csv(part)
    assert b["step"].tolist() == [1, 2, 3, 4, 5, 6]
    assert b["loss"].tolist() == pytest.approx(a["loss"].tolist(), rel=1e-6)
    assert (part.out / "prev.pt").exists()


@pytest.mark.unit
def test_resume_drops_log_rows_after_checkpoint(make_cfg):
    cfg = make_cfg(max_steps=3)
    train(cfg)
    with open(cfg.out / "train.csv", "a") as fh:  # ligne écrite après le dernier checkpoint (crash simulé)
        fh.write("4," + ",".join(["0"] * (len(_train_csv(cfg).columns) - 1)) + "\n")
    train(replace(cfg, max_steps=4), resume=True)
    assert _train_csv(cfg)["step"].tolist() == [1, 2, 3, 4]


@pytest.mark.unit
def test_resume_refuses_changed_config_but_allows_epochs_and_max_steps(make_cfg):
    cfg = make_cfg(max_steps=2)
    train(cfg)
    with pytest.raises(ValueError, match="reprise refusee"):
        train(replace(cfg, lr=1e-2), resume=True)
    train(replace(cfg, max_steps=3, epochs=4), resume=True)
    assert json.loads((cfg.out / "run_config.json").read_text())["lr"] == cfg.lr


@pytest.mark.unit
def test_fresh_run_refuses_existing_checkpoint(make_cfg):
    cfg = make_cfg(max_steps=1)
    train(cfg)
    with pytest.raises(ValueError, match="--resume"):
        train(cfg)


@pytest.mark.unit
def test_cli_runs_and_parser_defaults(shards, lut_dir, tmp_path):
    args = build_parser().parse_args(["--shards", str(shards), "--out", str(tmp_path / "o")])
    assert args.resume is False and args.epochs == 10
    main(["--shards", str(shards), "--out", str(tmp_path / "o"), "--lut-dir", str(lut_dir), "--px-budget", "4096",
          "--max-steps", "2", "--base-ch", "4", "--k-train", "8", "--lut-n", "32", "--lut-steps", "60",
          "--lut-n-r", "32", "--val-every-steps", "1000", "--log-every", "1"])
    assert (tmp_path / "o" / "last.pt").exists()


@pytest.mark.unit
def test_validation_subsample_per_class_and_full_at_epoch_end(make_cfg):
    cfg = make_cfg(max_steps=4, val_every_steps=2, val_max_per_class=2)
    train(cfg)
    val = pd.read_csv(cfg.out / "val.csv")
    sub = val[(val.scope == "sub") & (val.bin != "all")]
    full = val[(val.scope == "full") & (val.bin != "all")]
    assert (sub["n"] == 2).all() and sorted(full["n"].unique()) == [2, 4]


@pytest.mark.unit
def test_best_checkpoint_is_never_overwritten_by_a_worse_run(make_cfg):
    cfg = make_cfg(max_steps=4, val_every_steps=2)
    train(cfg)
    best = torch.load(cfg.out / "best.pt", weights_only=False)
    assert best["best_val"] > 0
    torch.save({**best, "best_val": 1e-9, "step": -7}, cfg.out / "best.pt")
    train(replace(cfg, max_steps=8), resume=True)
    after = torch.load(cfg.out / "best.pt", weights_only=False)
    assert after["best_val"] == 1e-9 and after["step"] == -7
    assert torch.load(cfg.out / "last.pt", weights_only=False)["best_val"] == 1e-9


@pytest.mark.unit
def test_prefetch_does_not_change_the_losses(make_cfg, tmp_path):
    a = make_cfg(out=tmp_path / "a", max_steps=5, prefetch_depth=0)
    b = make_cfg(out=tmp_path / "b", max_steps=5, prefetch_depth=2)
    train(a)
    train(b)
    assert _train_csv(b)["loss"].tolist() == pytest.approx(_train_csv(a)["loss"].tolist(), rel=1e-6)
    assert (_train_csv(b)["data_ms"] > 0).all()


@pytest.mark.unit
def test_epochs_change_on_resume_logs_warning(make_cfg, caplog):
    cfg = make_cfg(max_steps=2)
    train(cfg)
    with caplog.at_level(logging.WARNING):
        train(replace(cfg, epochs=cfg.epochs + 1, max_steps=3), resume=True)
    assert any("epochs" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_sigterm_writes_checkpoint_and_stops_cleanly(make_cfg, monkeypatch):
    cfg = make_cfg(max_steps=50)
    real = encoder_train.append_row
    before = signal.getsignal(signal.SIGTERM)

    def row_then_signal(path, row, columns):
        real(path, row, columns)
        if path.name == "train.csv" and row["step"] == 2:
            signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(encoder_train, "append_row", row_then_signal)
    result = train(cfg)
    assert result.step == 2 and result.stopped_by == signal.SIGTERM
    assert torch.load(cfg.out / "last.pt", weights_only=False)["step"] == 2
    assert signal.getsignal(signal.SIGTERM) == before
    monkeypatch.setattr(encoder_train, "append_row", real)
    assert train(replace(cfg, max_steps=3), resume=True).step == 3
