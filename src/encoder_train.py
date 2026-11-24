"""Entraînement de l'encodeur de sources (phase 5 E7) : boucle, validation périodique, reprise exacte.

Par pas : lot à budget de pixels -> encodeur (bf16 seulement sur CUDA, pour le CNN) -> candidats décodés ->
rendu fp32 -> perte totale (cartes de pics gloutonnes pendant les époques d'échauffement) -> clip -> AdamW.
La reprise (--resume) restaure modèle, optimiseur, planificateur, états aléatoires, pas, époque et position
dans l'époque : la perte du pas suivant est celle d'un entraînement ininterrompu. Le planificateur cosinus
dépend de epochs : prolonger epochs à la reprise change la fin de la courbe (c'est voulu).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import math
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cell_shard_dataset import CellShardDataset
from encoder_ckpt import (append_row, best_recorded, check_run_config, ensure_columns, has_checkpoint, load_checkpoint,
                          rng_state, save_best, save_checkpoint, set_rng_state, trim_csv)
from encoder_data import iter_items, make_lr_lambda
from encoder_loss import LossConfig
from encoder_step import RENDER_KIND, NonFiniteStep, accumulate_step, all_finite
from encoder_val import VAL_COLUMNS, validate
from prefetch import prefetch
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from train_guard import SNAPSHOT_EVERY, SkipCounter, Snapshot, is_spike
from size_batching import PixelBudgetSampler, batch_to_device, load_index
from source_encoder import EncoderConfig, SourceEncoder

logger = logging.getLogger(__name__)

DEFAULT_LUT_DIR = Path(__file__).resolve().parent.parent / "results" / "03_variable_sources"
TRAIN_COLUMNS = ["step", "epoch", "lr", "loss", "recon", "sparsity", "binarise", "heatmap", "k_active",
                 "params_per_area", "px_per_s", "ms_step", "data_ms", "data_wait_ms", "render_ms", "n_chunks", "peak_mem_mb", "bin", "n_skipped", "grad_norm", "max_abs_logit",
                 "max_raw_A", "max_raw_a", "max_raw_b", "max_raw_w"]
SKIPPED_COLUMNS = ["step", "epoch", "bin", "shard", "position", "reason"]
POST_STEP_NAN = "poids non finis après le pas (instantané restauré)"
MUTABLE_KEYS = frozenset({"epochs", "max_steps", "device", "out", "shards", "lut_dir", "checkpoint_every_min",
                          "log_every", "val_every_steps", "prefetch_depth", "render_px_budget",
                          "max_consecutive_skips", "max_skip_fraction", "spike_factor", "spike_window"})
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@dataclass(frozen=True)
class TrainConfig:
    shards: Path
    out: Path
    device: str = "cpu"
    px_budget: int = 400_000
    epochs: int = 10
    warmup_epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    clip: float = 5.0
    k_train: int = 256
    seed: int = 0
    checkpoint_every_min: float = 30.0
    val_every_steps: int = 500
    max_steps: int | None = None
    log_every: int = 50
    heat_sigma: float = 1.0
    val_max_per_class: int = 500
    render_px_budget: int = 20_000_000
    prefetch_depth: int = 2
    max_consecutive_skips: int = 20  # pas non finis consécutifs tolérés avant d'abandonner
    max_skip_fraction: float = 0.01  # part maximale de pas écartés dans une époque (seuil plancher : 1 pas)
    spike_factor: float = 100.0  # pas écarté si norme du gradient > spike_factor x médiane récente
    spike_window: int = 200  # taille de l'historique des normes (le garde n'agit qu'une fois plein)
    train_split: str = "train"
    val_split: str = "val"
    lut_dir: Path = DEFAULT_LUT_DIR
    lut: LogWLutSpec = field(default_factory=LogWLutSpec)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        for name in ("px_budget", "epochs", "warmup_steps", "k_train", "val_every_steps", "log_every", "val_max_per_class",
                     "render_px_budget", "max_consecutive_skips", "spike_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} doit être >= 1")
        if not 0 <= self.max_skip_fraction <= 1:
            raise ValueError("max_skip_fraction doit être dans [0, 1]")
        if self.warmup_epochs < 0 or self.clip <= 0 or self.lr <= 0 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("warmup_epochs >= 0, clip > 0, lr > 0 et max_steps >= 1 requis")


@dataclass(frozen=True)
class TrainResult:
    step: int
    epoch: int
    best_val: float
    stopped_by: int | None = None


def _save(cfg: TrainConfig, model, opt, sched, step: int, epoch: int, pos: int, best: float,
          skips: SkipCounter, norms) -> None:
    save_checkpoint(cfg.out, {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                              "rng": rng_state(), "step": step, "epoch": epoch, "pos": pos, "best_val": best,
                              "n_skipped": skips.total, "epoch_skipped": skips.epoch, "grad_norms": list(norms)})


def _guarded_update(cfg: TrainConfig, model, opt, snapshot: Snapshot, norms, batch, peaks, lut5, step: int):
    """(StepStats, None) si le pas est appliqué, sinon (None, raison) avec poids inchangés : valeurs non finies
    (NonFiniteStep), pic de gradient (norme > spike_factor x médiane des spike_window dernières) ou poids non finis
    après opt.step (instantané restauré)."""
    try:
        stats = accumulate_step(model, batch, peaks, lut5, step, k_train=cfg.k_train, loss_cfg=cfg.loss,
                                render_px_budget=cfg.render_px_budget)
    except NonFiniteStep as exc:
        return None, str(exc)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip))  # norme avant écrêtage
    if not math.isfinite(norm) or is_spike(norm, norms, cfg.spike_factor):
        model.zero_grad(set_to_none=True)
        return None, "spike"
    opt.step()
    if not all_finite(model.parameters()):
        snapshot.restore(model, opt)
        return None, POST_STEP_NAN
    norms.append(norm)
    return dataclasses.replace(stats, grad_norm=norm), None


def _skip_step(cfg: TrainConfig, skips: SkipCounter, reason: str, item, index, step: int, epoch: int,
               n_batches: int) -> str | None:
    """Trace un pas écarté (WARNING, skipped_batches.csv) ; retourne le message d'abandon si les limites sont dépassées.
    Le planificateur avance quand même (échéances de lr alignées sur le numéro de pas)."""
    skips.hit()
    entries = [index.entries[i] for i in item.ids]
    logger.warning("pas %d écarté (%s) : classe %s, cellules %s", step, reason, item.batch.meta[0]["bin"],
                   [(e.shard, e.position) for e in entries])
    for e in entries:
        append_row(cfg.out / "skipped_batches.csv", {"step": step, "epoch": epoch, "bin": e.bin, "shard": e.shard,
                                                      "position": e.position, "reason": reason}, SKIPPED_COLUMNS)
    where = f"voir {cfg.out / 'skipped_batches.csv'}"
    if skips.consecutive > cfg.max_consecutive_skips:
        return f"{skips.consecutive} pas consécutifs écartés (limite {cfg.max_consecutive_skips}), pas {step} : {where}"
    limit = max(1, math.floor(cfg.max_skip_fraction * n_batches))
    if skips.epoch > limit:
        return f"époque {epoch} : {skips.epoch} pas écartés sur {n_batches} (limite {limit}) : {where}"
    return None


def _validate_and_log(cfg: TrainConfig, model, index, lut5, device, step: int, best: float, full: bool) -> float:
    """Validation (sous-échantillon fixe, ou split entier en fin d'époque) ; best.pt ne suit que le sous-échantillon,
    seul comparable d'une validation à l'autre."""
    rows = validate(model, index, cfg.val_split, cfg.px_budget, lut5, device, step, RENDER_KIND, cfg.seed,
                    None if full else cfg.val_max_per_class)
    for row in rows:
        append_row(cfg.out / "val.csv", row, VAL_COLUMNS)
    score = next((r["nrmse_median"] for r in rows if r["bin"] == "all"), math.inf)
    if full or not score < best:
        return best
    save_best(cfg.out, {"model": model.state_dict(), "step": step, "best_val": score, "encoder": cfg.encoder})
    return score


class StopFlag:
    """Drapeau posé par SIGTERM / SIGINT : la boucle termine le pas en cours, écrit un point de reprise et s'arrête."""

    def __init__(self) -> None:
        self.signum: int | None = None
        self._previous: dict[int, object] = {}

    def _handle(self, signum: int, frame) -> None:
        self.signum = signum
        logger.warning("signal %d reçu : arrêt après le pas en cours", signum)

    def __enter__(self) -> "StopFlag":
        if threading.current_thread() is threading.main_thread():
            self._previous = {s: signal.signal(s, self._handle) for s in STOP_SIGNALS}
        return self

    def __exit__(self, *exc) -> None:
        for s, handler in self._previous.items():
            signal.signal(s, handler)


def _log_row(cfg, path, step, epoch, lr, stats, item, wait_ms, dt, batch, device, n_skipped: int) -> None:
    px = batch.x.shape[0] * batch.x.shape[2] * batch.x.shape[3]
    cuda = device.type == "cuda"
    append_row(path, {"step": step, "epoch": epoch, "lr": lr, "loss": stats.loss, "recon": stats.recon,
                      "sparsity": stats.sparsity, "binarise": stats.binarise, "heatmap": stats.heatmap,
                      "k_active": stats.k_active, "params_per_area": stats.params_per_area, "px_per_s": px / dt,
                      "ms_step": 1000 * dt, "data_ms": item.data_ms, "data_wait_ms": wait_ms,
                      "render_ms": stats.render_ms, "n_chunks": stats.n_chunks,
                      "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if cuda else 0.0,
                      "bin": batch.meta[0]["bin"], "n_skipped": n_skipped, "grad_norm": stats.grad_norm,
                      "max_abs_logit": stats.max_abs_logit, "max_raw_A": stats.max_raw_A, "max_raw_a": stats.max_raw_a,
                      "max_raw_b": stats.max_raw_b, "max_raw_w": stats.max_raw_w}, TRAIN_COLUMNS)  # un lot = une classe de taille
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)  # pic par fenêtre de journalisation


def _build(cfg: TrainConfig):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    index = load_index(cfg.shards)
    lut5 = load_or_build_logw_lut(cfg.lut, cfg.lut_dir)["lut"][None, None].to(device)
    model = SourceEncoder(cfg.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    per_epoch = len(PixelBudgetSampler(index, cfg.train_split, cfg.px_budget, cfg.seed, 0))
    if per_epoch == 0:
        raise ValueError(f"aucune cellule dans le split {cfg.train_split!r}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(cfg.warmup_steps, cfg.epochs * per_epoch))
    return device, index, lut5, model, opt, sched


def train(cfg: TrainConfig, resume: bool = False) -> TrainResult:
    """Entraîne (ou reprend) ; retourne pas, époque, meilleur nrmse de validation et signal d'arrêt éventuel."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    if not resume and has_checkpoint(cfg.out):
        raise ValueError(f"{cfg.out} contient déjà un point de reprise : utiliser --resume ou un autre dossier")
    check_run_config(cfg.out, {**dataclasses.asdict(cfg), "lut_key": cfg.lut.key()}, resume, MUTABLE_KEYS)
    device, index, lut5, model, opt, sched = _build(cfg)
    data = CellShardDataset(index)
    step = epoch = pos = 0
    best = best_recorded(cfg.out)  # ne jamais écraser un best.pt meilleur
    skips, norms, snapshot = SkipCounter(), collections.deque(maxlen=cfg.spike_window), Snapshot()
    resumed = resume and has_checkpoint(cfg.out)
    if resumed:
        ck = load_checkpoint(cfg.out)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        set_rng_state(ck["rng"])
        step, epoch, pos, best = ck["step"], ck["epoch"], ck["pos"], min(ck["best_val"], best)
        skips = SkipCounter(total=ck.get("n_skipped", 0), epoch=ck.get("epoch_skipped", 0))
        norms.extend(ck.get("grad_norms", ())[-cfg.spike_window:])
        for name, columns in (("train.csv", TRAIN_COLUMNS), ("val.csv", VAL_COLUMNS), ("skipped_batches.csv", SKIPPED_COLUMNS)):
            trim_csv(cfg.out / name, step)
            ensure_columns(cfg.out / name, columns, {"n_skipped": 0})  # CSV d'une version antérieure : schéma complété
        logger.info("reprise au pas %d (époque %d, lot %d)", step, epoch, pos)
    model.train()
    snapshot.take(model, opt)  # dernier état sain (renouvelé tous les SNAPSHOT_EVERY pas)
    stop = cfg.max_steps if cfg.max_steps is not None else math.inf
    last_ckpt, validated_at = time.monotonic(), step if resumed else -1
    with StopFlag() as flag:
        while epoch < cfg.epochs and step < stop and flag.signum is None:
            batches = PixelBudgetSampler(index, cfg.train_split, cfg.px_budget, cfg.seed, epoch).batches()
            items = prefetch(iter_items(data, batches[pos:], epoch < cfg.warmup_epochs, cfg.encoder, cfg.heat_sigma),
                             cfg.prefetch_depth)
            try:
                while pos < len(batches) and step < stop and flag.signum is None:
                    t_wait = time.perf_counter()
                    item = next(items)
                    wait_ms = 1000 * (time.perf_counter() - t_wait)
                    t0 = time.perf_counter()
                    batch = batch_to_device(item.batch, device)
                    peaks = None if item.peaks is None else item.peaks.to(device)
                    lr = opt.param_groups[0]["lr"]
                    stats, reason = _guarded_update(cfg, model, opt, snapshot, norms, batch, peaks, lut5, step)
                    sched.step()  # aussi après un pas écarté : échéances de lr alignées sur le numéro de pas
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    dt = time.perf_counter() - t0 + wait_ms / 1000
                    step, pos = step + 1, pos + 1
                    abort = None
                    if reason is None:
                        skips.ok()
                        if step % SNAPSHOT_EVERY == 0:
                            snapshot.take(model, opt)
                        if step % cfg.log_every == 0:
                            _log_row(cfg, cfg.out / "train.csv", step, epoch, lr, stats, item, wait_ms, dt, batch,
                                     device, skips.total)
                    else:
                        abort = _skip_step(cfg, skips, reason, item, index, step, epoch, len(batches))
                    if abort is not None:  # poids = dernier état sain : point de reprise atomique avant d'abandonner
                        _save(cfg, model, opt, sched, step, epoch, pos, best, skips, norms)
                        raise RuntimeError(abort)
                    if step % cfg.val_every_steps == 0:
                        best = _validate_and_log(cfg, model, index, lut5, device, step, best, full=False)
                        validated_at = step
                    if time.monotonic() - last_ckpt >= 60 * cfg.checkpoint_every_min:
                        _save(cfg, model, opt, sched, step, epoch, pos, best, skips, norms)
                        last_ckpt = time.monotonic()
            finally:
                items.close()
            if pos >= len(batches):
                epoch, pos, skips.epoch = epoch + 1, 0, 0
                if flag.signum is None:
                    _validate_and_log(cfg, model, index, lut5, device, step, best, full=True)
        if validated_at != step and flag.signum is None:
            best = _validate_and_log(cfg, model, index, lut5, device, step, best, full=False)
        _save(cfg, model, opt, sched, step, epoch, pos, best, skips, norms)
        stopped_by = flag.signum
    logger.info("terminé : pas %d, époque %d, meilleur nrmse val %.4f", step, epoch, best)
    return TrainResult(step, epoch, best, stopped_by)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lut-dir", type=Path, default=DEFAULT_LUT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--px-budget", type=int, default=TrainConfig.px_budget)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--k-train", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every-min", type=float, default=30.0)
    ap.add_argument("--val-every-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-max-per-class", type=int, default=500)
    ap.add_argument("--render-px-budget", type=int, default=TrainConfig.render_px_budget)
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--max-consecutive-skips", type=int, default=TrainConfig.max_consecutive_skips)
    ap.add_argument("--max-skip-fraction", type=float, default=TrainConfig.max_skip_fraction)
    ap.add_argument("--spike-factor", type=float, default=TrainConfig.spike_factor)
    ap.add_argument("--spike-window", type=int, default=TrainConfig.spike_window)
    ap.add_argument("--base-ch", type=int, default=EncoderConfig.base_ch)
    ap.add_argument("--lambda-sparsity", type=float, default=LossConfig.lambda_sparsity)
    ap.add_argument("--lambda-binarise", type=float, default=LossConfig.lambda_binarise)
    ap.add_argument("--lambda-heatmap", type=float, default=LossConfig.lambda_heatmap)
    ap.add_argument("--lut-n", type=int, default=LogWLutSpec.n)
    ap.add_argument("--lut-steps", type=int, default=LogWLutSpec.steps)
    ap.add_argument("--lut-n-r", type=int, default=LogWLutSpec.n_r)
    ap.add_argument("--resume", action="store_true")
    return ap


def config_from_args(a: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        shards=a.shards, out=a.out, lut_dir=a.lut_dir, device=a.device, px_budget=a.px_budget, epochs=a.epochs,
        warmup_epochs=a.warmup_epochs, lr=a.lr, weight_decay=a.weight_decay, warmup_steps=a.warmup_steps, clip=a.clip,
        k_train=a.k_train, seed=a.seed, checkpoint_every_min=a.checkpoint_every_min, val_every_steps=a.val_every_steps,
        max_steps=a.max_steps, log_every=a.log_every,
        val_max_per_class=a.val_max_per_class, render_px_budget=a.render_px_budget, prefetch_depth=a.prefetch_depth,
        max_consecutive_skips=a.max_consecutive_skips, max_skip_fraction=a.max_skip_fraction,
        spike_factor=a.spike_factor, spike_window=a.spike_window,
        lut=LogWLutSpec(n=a.lut_n, steps=a.lut_steps, n_r=a.lut_n_r),
        encoder=EncoderConfig(base_ch=a.base_ch, k_train=a.k_train),
        loss=LossConfig(lambda_sparsity=a.lambda_sparsity, lambda_binarise=a.lambda_binarise,
                        lambda_heatmap=a.lambda_heatmap))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    result = train(config_from_args(args), resume=args.resume)
    if result.stopped_by is not None:
        raise SystemExit(128 + result.stopped_by)


if __name__ == "__main__":
    main()
