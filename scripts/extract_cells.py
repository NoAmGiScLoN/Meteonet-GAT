"""Extrait les cellules de pluie MeteoNet (int16 brut) en shards npz pour l'encodeur E7.

Usage : python scripts/extract_cells.py --start 2016-01-01 --end 2018-12-31 --out results/e7_cells

Les jours exclus (benchmark E5 +- marge), hors plage ou sans quota (test) sont ignorés. Une carte n'est
segmentée que si son indice (pas de 5 min depuis le début du run) est un multiple du pas d'au moins
une classe de taille ; une cellule n'est candidate que si l'indice est multiple du pas de SA classe.
La sélection est un bottom-k par hachage (cell_selection) : exacte et indépendante de l'ordre.
Les shards (un par tranche de --shard-size cellules d'un (split, classe)) sont écrits à la fin.
Reprise : --resume refuse une configuration différente. Le sélecteur est sauvegardé (atomiquement,
dans <out>/_partial/selector.pkl) à chaque frontière de fichier npz : après une panne, --resume repart de
la dernière frontière et donne le même résultat qu'un run ininterrompu ; une sortie complète est conservée.
--max-ram-gb borne la mémoire des cellules retenues (erreur claire au dépassement).
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import time
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, time as dtime
from itertools import islice
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cell_delimitation import DATA_DIR  # noqa: E402
from cell_selection import (DEFAULT_FRAME_STEPS, DEFAULT_TRAIN_QUOTAS, CellSelector, bins_due,  # noqa: E402
                            default_quotas, frame_index)
from cell_shard_dataset import ShardIndex  # noqa: E402
from cell_shards import (BIN_NAMES, DEFAULT_MIN_AREA, MIN_BIN_AREA, SHARD_SUFFIX,  # noqa: E402
                         extract_frame_cells, write_shard)
from data_split import (SplitConfig, benchmark_days, default_benchmark_paths, excluded_days,  # noqa: E402
                        split_of)
from frame_stream import DEFAULT_STEP_MIN, iter_frames  # noqa: E402

log = logging.getLogger("extract_cells")

CONFIG_FILE = "run_config.json"
INDEX_FILE = "index.csv"
SUMMARY_FILE = "summary.csv"
DEFAULT_SHARD_SIZE = 5000
DEFAULT_LOG_EVERY = 2000
PARTIAL_DIR = "_partial"
CHECKPOINT_FILE = "selector.pkl"
DEFAULT_MAX_RAM_GB = 16.0
GB = 1024 ** 3
PIXEL_BYTES = 2  # int16 brut ; les masques empaquetés (2 bits/px) sont négligeables


class RamBudgetExceeded(RuntimeError):
    """La mémoire des cellules retenues dépasse --max-ram-gb."""


def parse_quota_spec(spec: str) -> dict[str, int]:
    """'50-200=5,>=5000=7' -> {'50-200': 5, '>=5000': 7} ; ValueError si classe inconnue ou entier invalide."""
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, sep, value = part.rpartition("=")
        if not sep or name not in BIN_NAMES:
            raise ValueError(f"quota invalide {part!r} : attendu <classe>=<entier>, classes {BIN_NAMES}")
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"quota invalide {part!r} : entier attendu") from exc
        if count < 0:
            raise ValueError(f"quota invalide {part!r} : entier >= 0 attendu")
        out[name] = count
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extraction des cellules en shards npz (E7).")
    ap.add_argument("--start", type=date.fromisoformat, required=True, help="premier jour (AAAA-MM-JJ)")
    ap.add_argument("--end", type=date.fromisoformat, required=True, help="dernier jour inclus")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--benchmark", type=Path, nargs="+", default=None, help="sources.pkl à exclure (défaut : E5)")
    ap.add_argument("--quota-train", type=str, default=None, help="surcharges '<classe>=<n>,...'")
    ap.add_argument("--quota-val", type=str, default=None, help="surcharges val (défaut : train // 16)")
    ap.add_argument("--salt", type=str, default="e7-cells")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    ap.add_argument("--max-ram-gb", type=float, default=DEFAULT_MAX_RAM_GB,
                    help="plafond de mémoire des cellules retenues (erreur au-delà)")
    ap.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help="trames entre deux logs")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.end < args.start:
        ap.error("--end doit être >= --start")
    if args.shard_size < 1 or args.log_every < 1:
        ap.error("--shard-size et --log-every doivent être >= 1")
    if args.min_area < MIN_BIN_AREA:
        ap.error(f"--min-area doit être >= {MIN_BIN_AREA} (les classes commencent à {MIN_BIN_AREA} px)")
    if not args.max_ram_gb > 0:
        ap.error("--max-ram-gb doit être > 0")
    return args


def build_quotas(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    train = {**DEFAULT_TRAIN_QUOTAS, **(parse_quota_spec(args.quota_train) if args.quota_train else {})}
    quotas = default_quotas(train)
    for name, count in (parse_quota_spec(args.quota_val) if args.quota_val else {}).items():
        quotas[("val", name)] = count
    return quotas


def run_config(args: argparse.Namespace, quotas: dict, cfg: SplitConfig, excluded: frozenset[date]) -> dict:
    """Configuration qui détermine le contenu de la sortie (comparée à la reprise)."""
    digest = hashlib.sha256(",".join(sorted(d.isoformat() for d in excluded)).encode()).hexdigest()
    return {"start": args.start.isoformat(), "end": args.end.isoformat(), "salt": args.salt,
            "shard_size": args.shard_size, "min_area": args.min_area, "frame_steps": DEFAULT_FRAME_STEPS,
            "quotas": {f"{s}|{b}": n for (s, b), n in sorted(quotas.items())},
            "split": {k: str(v) for k, v in dataclasses.asdict(cfg).items()}, "excluded_sha256": digest}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Écriture atomique : fichier temporaire voisin, fsync, os.replace."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode())


def check_run_config(out: Path, config: dict, resume: bool) -> None:
    """Écrit run_config.json ; à la reprise, refuse une configuration différente."""
    path = out / CONFIG_FILE
    config = json.loads(json.dumps(config))
    if resume and path.exists():
        previous = json.loads(path.read_text())
        if previous != config:
            diff = {k: (previous.get(k), config.get(k)) for k in config if previous.get(k) != config.get(k)}
            raise ValueError(f"reprise refusée : run_config.json diffère (ancien, nouveau) {diff}")
        return
    _atomic_write_bytes(path, json.dumps(config, indent=2).encode())


def prepare_output(out: Path, resume: bool, overwrite: bool) -> None:
    """Crée le dossier ; refuse d'écraser une sortie existante sans --resume/--overwrite."""
    out.mkdir(parents=True, exist_ok=True)
    stale = [p for p in out.iterdir()
             if p.suffix == SHARD_SUFFIX or p.name in (CONFIG_FILE, INDEX_FILE, SUMMARY_FILE, PARTIAL_DIR)]
    if stale and not (resume or overwrite):
        raise FileExistsError(f"{out} contient déjà une extraction : utiliser --resume ou --overwrite")
    if overwrite:
        for path in stale:
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def _is_complete(out: Path) -> bool:
    """Vrai si index.csv et summary.csv existent, que chaque shard listé existe et que les comptes concordent."""
    if not ((out / INDEX_FILE).exists() and (out / SUMMARY_FILE).exists()):
        return False
    index, summary = pd.read_csv(out / INDEX_FILE), pd.read_csv(out / SUMMARY_FILE)
    if not all((out / name).is_dir() for name in index["shard"].unique()):
        return False
    total = summary[summary["bin"] == "total"]
    if len(total) != 1 or int(total.n_cells.iloc[0]) != len(index):
        return False
    counts = index.groupby(["split", "bin"]).size().to_dict()
    listed = {(r.split, r.bin): int(r.n_cells) for r in summary[summary["bin"] != "total"].itertuples()}
    return counts == listed


def _day_split(day: date, cfg: SplitConfig, excluded: frozenset[date], quotas: dict) -> str | None:
    """Split du jour s'il a au moins un quota > 0 ; None si hors plage, exclu ou sans quota."""
    if not cfg.start <= day <= cfg.end:
        return None
    split = split_of(day, cfg, excluded)
    return split if any(n > 0 for (s, _), n in quotas.items() if s == split) else None


def save_checkpoint(out: Path, selector: CellSelector, last_t: datetime, progress: dict) -> None:
    """Sauvegarde atomique du sélecteur et de la progression (frontière de fichier)."""
    (out / PARTIAL_DIR).mkdir(exist_ok=True)
    payload = {"state": selector.state(), "last_t": last_t.isoformat(), **progress}
    _atomic_write_bytes(out / PARTIAL_DIR / CHECKPOINT_FILE, pickle.dumps(payload))


def load_checkpoint(out: Path) -> dict | None:
    """Dernier checkpoint, None s'il n'y en a pas."""
    path = out / PARTIAL_DIR / CHECKPOINT_FILE
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())  # nosec B301  (fichier local produit par ce script)


def extract(args: argparse.Namespace, quotas: dict, cfg: SplitConfig, excluded: frozenset[date],
            checkpoint: dict | None = None) -> CellSelector:
    """Parcourt les cartes, applique les pas par classe et renvoie la sélection bottom-k.

    Avec `checkpoint`, repart juste après son dernier instant (frontière de fichier).
    """
    origin = datetime.combine(args.start, dtime.min)
    end = datetime.combine(args.end, dtime(23, 59, 59))
    selector = CellSelector(quotas, args.salt)
    candidates: Counter = Counter()
    n_frames = n_segmented = 0
    first = origin
    if checkpoint is not None:
        selector = CellSelector.from_state(quotas, args.salt, checkpoint["state"])
        candidates, n_frames, n_segmented = Counter(checkpoint["candidates"]), checkpoint["n_frames"], checkpoint["n_segmented"]
        first = datetime.fromisoformat(checkpoint["last_t"]) + timedelta(minutes=DEFAULT_STEP_MIN)
        log.info("reprise après %s (%d cartes déjà traitées)", checkpoint["last_t"], n_frames)
    max_bytes = args.max_ram_gb * GB
    day_cache: dict[date, str | None] = {}
    previous_source, previous_t = None, None
    t_start = t_wait = time.perf_counter()
    waiting = 0.0
    for frame in iter_frames(first, end, args.data_dir):
        waiting += time.perf_counter() - t_wait
        if previous_source is not None and frame.source != previous_source:
            save_checkpoint(args.out, selector, previous_t, {"candidates": dict(candidates), "n_frames": n_frames,
                                                              "n_segmented": n_segmented})
        previous_source, previous_t = frame.source, frame.t
        n_frames += 1
        day = frame.t.date()
        if day not in day_cache:
            day_cache[day] = _day_split(day, cfg, excluded, quotas)
        split = day_cache[day]
        due = frozenset() if split is None else frozenset(
            b for b in bins_due(frame_index(frame.t, origin), DEFAULT_FRAME_STEPS) if quotas[(split, b)] > 0)
        if due:
            n_segmented += 1

            def keep(bin_name: str, local_id: int, split=split, due=due, t=frame.t) -> bool:
                if bin_name not in due:
                    return False
                candidates[(split, bin_name)] += 1
                return selector.would_accept(split, bin_name, t, local_id)

            for rec in extract_frame_cells(frame.R, frame.t, split, args.min_area, keep):
                selector.offer(rec)
            if selector.nbytes() > max_bytes:
                raise RamBudgetExceeded(f"cellules retenues : {selector.nbytes() / GB:.2f} Go > --max-ram-gb "
                                        f"{args.max_ram_gb} : réduire les quotas ou augmenter --max-ram-gb")
        if n_frames % args.log_every == 0:
            log.info("%d cartes (%d segmentées), candidats %s, retenues %s, mémoire %.1f Mo, %.0f s",
                     n_frames, n_segmented, dict(candidates), selector.counts(), selector.nbytes() / 1e6,
                     time.perf_counter() - t_start)
        t_wait = time.perf_counter()
    total = time.perf_counter() - t_start
    log.info("fin : %d cartes (%d segmentées) en %.1f s dont %.1f s de chargement ; candidats %s",
             n_frames, n_segmented, total, waiting, dict(candidates))
    return selector


def write_outputs(out: Path, selector: CellSelector, shard_size: int) -> None:
    """Shards par (split, classe), puis index.csv et summary.csv (atomiques, summary en dernier)."""
    for path in out.glob(f"*{SHARD_SUFFIX}"):
        shutil.rmtree(path)  # reste d'une exécution interrompue
    for (split, name) in sorted(selector.counts()):
        slug = name.replace(">=", "ge")
        records = selector.iter_selected(split, name)
        k = 0
        while chunk := list(islice(records, shard_size)):
            write_shard(out / f"{split}_{slug}_{k:04d}{SHARD_SUFFIX}", chunk)
            k += 1
    index = ShardIndex.from_dir(out).to_frame()
    _atomic_write_csv(out / INDEX_FILE, index)
    index = index.assign(raw_mb=index.h * index.w * PIXEL_BYTES / 1e6)
    per = index.groupby(["split", "bin"], as_index=False).agg(n_cells=("area", "size"), raw_mb=("raw_mb", "sum"))
    total = pd.DataFrame([{"split": "all", "bin": "total", "n_cells": int(len(index)),
                           "raw_mb": float(index.raw_mb.sum())}])
    _atomic_write_csv(out / SUMMARY_FILE, pd.concat([per, total], ignore_index=True))
    shutil.rmtree(out / PARTIAL_DIR, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    quotas, cfg = build_quotas(args), SplitConfig()
    paths = args.benchmark or default_benchmark_paths(ROOT)
    excluded = excluded_days(benchmark_days(paths), cfg)
    prepare_output(args.out, args.resume, args.overwrite)
    check_run_config(args.out, run_config(args, quotas, cfg, excluded), args.resume)
    if args.resume and _is_complete(args.out):
        log.info("extraction déjà terminée dans %s : rien à faire", args.out)
        return 0
    checkpoint = load_checkpoint(args.out) if args.resume else None
    selector = extract(args, quotas, cfg, excluded, checkpoint)
    write_outputs(args.out, selector, args.shard_size)
    log.info("résumé : %s", args.out / SUMMARY_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
