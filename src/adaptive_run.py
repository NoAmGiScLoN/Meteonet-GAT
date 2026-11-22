"""CLI de la session B : ajustement a K adaptatif (diff7) avec sorties incrementales et tables de synthese.

Ecrit dans --out : cells_<classe>.csv (une ligne par cellule terminee), waves_<classe>.csv (trace par vague, ecrite en
fin de groupe), sources/<classe>_<id>.pkl (une source par cellule, ecriture atomique), sources.pkl (fusion atomique de
tous les fichiers par cellule), memory.csv (pic memoire par groupe, colonnes timestamp et run), run_config.json,
adaptive_run.log et summary_{stop,k,error}.csv.

Reprise (--resume) : une cellule est faite si son id est dans cells_<classe>.csv ET si son fichier source existe ; les
cellules avec une ligne mais sans source sont rajustees et leurs lignes obsoletes retirees des CSV (cells et waves).
Les lignes de vagues d'un groupe interrompu sont perdues pour les cellules deja terminees (ecrites en fin de groupe).
Sans --resume, un dossier contenant deja des sorties est refuse sauf --overwrite. La configuration (run_config.json)
doit etre identique a la reprise.

Cellules "diverged" (nrmse non fini) : elles sont ecrites comme terminees et ne sont PAS rajustees a la reprise ;
leur nrmse est exclu des statistiques (mediane, p90) et compte dans n_diverged.
"""
import argparse
import logging
import itertools
import json
import math
import os
import pickle
import shutil
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import variable_source_fit as vsf
from adaptive_fit import (COMPRESSION_FRAC_DEFAULT, FLOOR_TAIL, K_CAP_DEFAULT, PRUNE_FRAC_DEFAULT, SCHEDULE_TAIL,
                          TARGET_DEFAULT, AdaptiveConfig, fit_adaptive)
from profile_lut import LogWLutSpec

ARM = "diff7"
K_CAP_CHOICES = tuple(vsf.SCHEDULE) + SCHEDULE_TAIL + FLOOR_TAIL
STOP_REASONS = ("target", "cap", "compression", "diverged")
P90 = 0.9
IQR_LOW, IQR_HIGH = 0.25, 0.75
MEMORY_COLUMNS = ["bin", "group", "n_cells", "peak_bytes", "seconds", "timestamp", "run"]
SOURCES_FILE = "sources.pkl"
SOURCES_DIR = "sources"
CONFIG_FILE = "run_config.json"
LOG_FILE = "adaptive_run.log"
# cuda.OutOfMemoryError n'existe pas dans toutes les versions de torch ; () n'attrape rien
OOM_ERRORS: tuple[type[BaseException], ...] = tuple(
    t for t in (getattr(torch.cuda, "OutOfMemoryError", None),) if t is not None)
log = logging.getLogger(__name__)


def _finite(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"reel attendu, recu {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"reel fini attendu, recu {value}")
    return value


def nonneg_float(text: str) -> float:
    value = _finite(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"doit etre >= 0, recu {value}")
    return value


def prune_float(text: str) -> float:
    value = nonneg_float(text)
    if value >= 1:
        raise argparse.ArgumentTypeError(f"doit etre dans [0, 1), recu {value}")
    return value


def compression_float(text: str) -> float | None:
    """0 -> None (compression desactivee) ; sinon dans (0, 1]."""
    value = _finite(text)
    if value == 0:
        return None
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"doit etre dans (0, 1] ou 0, recu {value}")
    return value


def positive_float(text: str) -> float:
    value = _finite(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"doit etre > 0, recu {value}")
    return value


def dedupe(items: Sequence[str]) -> list[str]:
    """Supprime les doublons en conservant l'ordre."""
    return list(dict.fromkeys(items))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True, help="echantillon pre-construit (pickle)")
    ap.add_argument("--out", required=True, help="dossier de sortie (contient aussi les tables de profils)")
    ap.add_argument("--target", type=nonneg_float, default=TARGET_DEFAULT, help="nrmse cible (>= 0)")
    ap.add_argument("--k-cap", type=int, choices=K_CAP_CHOICES,
                    default=K_CAP_DEFAULT)
    ap.add_argument("--compression-frac", type=compression_float, default=COMPRESSION_FRAC_DEFAULT,
                    help="fraction (0, 1] de l'aire ; 0 = compression desactivee")
    ap.add_argument("--prune-frac", type=prune_float, default=PRUNE_FRAC_DEFAULT)
    bin_names = [b[0] for b in vsf.BINS]
    ap.add_argument("--bins", nargs="+", default=bin_names, choices=bin_names)
    ap.add_argument("--largest", type=vsf.positive_int, default=None, help="les N plus grandes cellules (une classe)")
    ap.add_argument("--max-per-bin", type=vsf.positive_int, default=None)
    ap.add_argument("--iters", type=vsf.positive_int, default=None, help="iterations par vague (defaut : par classe)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="saute les cellules deja faites (ligne + source)")
    mode.add_argument("--overwrite", action="store_true", help="efface les sorties existantes de --out")
    d = LogWLutSpec()
    ap.add_argument("--w-min", type=positive_float, default=d.w_min)
    ap.add_argument("--w-max", type=positive_float, default=d.w_max)
    ap.add_argument("--n-w", type=vsf.positive_int, default=d.n_w)
    ap.add_argument("--tau7", type=float, default=d.tau)
    ap.add_argument("--n-r", type=vsf.positive_int, default=d.n_r, help="resolution radiale de la table (tests rapides)")
    ap.add_argument("--lut-n", type=vsf.positive_int, default=d.n, help="grille de la PDE de la table (tests rapides)")
    ap.add_argument("--lut-steps", type=vsf.positive_int, default=d.steps, help="pas de la PDE de la table (tests rapides)")
    return ap


def select_cells(sample: Sequence[dict], bins: Sequence[str] | None, max_per_bin: int | None,
                 largest: int | None) -> list[dict]:
    """Copies des cellules choisies, "id" = indice d'origine, ordre d'id croissant (deterministe)."""
    chosen = [{**c, "id": i} for i, c in enumerate(sample) if bins is None or c["bin"] in bins]
    if largest is not None:
        chosen = sorted(sorted(chosen, key=lambda c: (-c["area"], c["id"]))[:largest], key=lambda c: c["id"])
        found = sorted({c["bin"] for c in chosen})
        if len(found) > 1:
            raise ValueError(f"--largest : les cellules doivent tomber dans une seule classe, trouve {found}")
    if max_per_bin is not None:
        counts: dict[str, int] = {}
        kept = []
        for c in chosen:
            counts[c["bin"]] = counts.get(c["bin"], 0) + 1
            if counts[c["bin"]] <= max_per_bin:
                kept.append(c)
        chosen = kept
    return chosen


def seconds_per_wave(trace: Sequence[dict]) -> float:
    """Duree moyenne d'une vague : somme des secondes / nombre de vagues distinctes ; NaN si trace vide."""
    waves = {t["wave"] for t in trace}
    return sum(t["seconds"] for t in trace) / len(waves) if waves else float("nan")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Ecriture atomique : fichier temporaire voisin + fsync + os.replace."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    """Lecture tolerante : derniere ligne tronquee (pas de saut de ligne final) et lignes malformees ignorees."""
    text = path.read_text()
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    return pd.read_csv(StringIO(text), on_bad_lines="skip") if text else pd.DataFrame()


def _repair_tail(path: Path) -> None:
    """Tronque une derniere ligne incomplete (crash en cours d'ecriture) avant d'ajouter."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with open(path, "wb") as fh:
            fh.write(data[:data.rfind(b"\n") + 1])


def _append_rows(path: Path, rows: Sequence[dict]) -> None:
    """Ajoute des lignes (en-tete verifie, flush + fsync)."""
    if not rows:
        return
    frame = pd.DataFrame(list(rows))
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        _repair_tail(path)
        header = path.read_text().split("\n", 1)[0].split(",")
        if header != list(frame.columns):
            raise ValueError(f"en-tete de {path.name} incompatible : {header} != {list(frame.columns)}")
    with open(path, "a", newline="") as fh:
        frame.to_csv(fh, header=not exists, index=False)
        fh.flush()
        os.fsync(fh.fileno())


def _append_row(path: Path, row: dict) -> None:
    _append_rows(path, [row])


def _rewrite_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode())


def _source_path(out: Path, slug: str, cell_id: int) -> Path:
    return out / SOURCES_DIR / f"{slug}_{cell_id}.pkl"


def _write_source(out: Path, slug: str, cell_id: int, entry: dict) -> None:
    (out / SOURCES_DIR).mkdir(exist_ok=True)
    _atomic_write_bytes(_source_path(out, slug, cell_id), pickle.dumps(entry))


def _merge_sources(out: Path) -> dict:
    """Fusion de tous les fichiers par cellule (id -> entree)."""
    merged = {}
    for path in sorted((out / SOURCES_DIR).glob("*.pkl")):
        merged[int(path.stem.rsplit("_", 1)[1])] = pickle.loads(path.read_bytes())  # nosec B301  (produit ici)
    return merged


def _flush_sources(out: Path, sources: dict) -> None:
    _atomic_write_bytes(out / SOURCES_FILE, pickle.dumps(sources))


def _cells_frame(out: Path, name: str) -> pd.DataFrame:
    path = out / f"cells_{vsf.bin_slug(name)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["id"])
    frame = _read_csv(path)
    if "id" not in frame.columns:
        return pd.DataFrame(columns=["id"])
    return frame.dropna(subset=["id"])


def done_ids(out: Path, name: str) -> set[int]:
    """Ids faits : dans cells_<classe>.csv ET avec un fichier source."""
    slug = vsf.bin_slug(name)
    ids = set(_cells_frame(out, name)["id"].astype(int))
    return {i for i in ids if _source_path(out, slug, i).exists()}


def drop_stale_rows(out: Path, name: str, done: set[int]) -> None:
    """Retire de cells_/waves_<classe>.csv les lignes des cellules non faites (elles seront rajustees)."""
    slug = vsf.bin_slug(name)
    for prefix in ("cells", "waves"):
        path = out / f"{prefix}_{slug}.csv"
        if not path.exists():
            continue
        frame = _read_csv(path)
        if "id" not in frame.columns:
            continue
        frame = frame.dropna(subset=["id"])
        kept = frame[frame["id"].astype(int).isin(done)]
        if len(kept) != len(frame):
            log.warning("%s : %d lignes obsoletes retirees", path.name, len(frame) - len(kept))
            _rewrite_csv(path, kept)


def warm_up(cell: dict, lut5: torch.Tensor | None) -> None:
    """Ajustement jetable (1 cellule, 1 iteration, K=1) pour exclure l'initialisation paresseuse du chronometrage."""
    cfg = AdaptiveConfig(arm=ARM, k_cap=1, compression_frac=None, iters_override=1)
    fit_adaptive([cell], cfg, lut5)


def _next_run_number(out: Path) -> int:
    path = out / "memory.csv"
    if not path.exists():
        return 1
    mem = _read_csv(path)
    return int(mem["run"].max()) + 1 if "run" in mem.columns and mem["run"].notna().any() else 1


def _fit_group(name: str, group: list[dict], cfg: AdaptiveConfig, lut5, out: Path, idx: int, run: int,
               sources: dict, finished: set[int]) -> None:
    slug, cuda = vsf.bin_slug(name), vsf.DEVICE.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(vsf.DEVICE)
    t0 = time.perf_counter()

    def on_cell_done(row: dict, entry: dict) -> None:
        _write_source(out, slug, row["id"], entry)  # source d'abord : une ligne sans source est rajustee
        _append_row(out / f"cells_{slug}.csv", row)
        sources[row["id"]] = entry
        finished.add(row["id"])

    _, trace, _ = fit_adaptive(group, cfg, lut5, on_cell_done=on_cell_done)
    seconds = time.perf_counter() - t0
    peak = float(torch.cuda.max_memory_allocated(vsf.DEVICE)) if cuda else float("nan")
    _flush_sources(out, sources)
    _append_rows(out / f"waves_{slug}.csv", trace)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _append_row(out / "memory.csv", dict(zip(MEMORY_COLUMNS, [name, idx, len(group), peak, seconds, stamp, run])))
    log.info("classe %s groupe %d : %d cellules, %.1f s, %.2f s/vague, pic %s o", name, idx, len(group), seconds,
             seconds_per_wave(trace), "n/a" if math.isnan(peak) else f"{peak:.0f}")


def _fit_group_resilient(name: str, group: list[dict], cfg: AdaptiveConfig, lut5, out: Path, counter,
                         run: int, sources: dict) -> None:
    """_fit_group ; sur OOM CUDA, vide le cache, coupe le groupe en deux et reessaie (jusqu'a 1 cellule)."""
    finished: set[int] = set()
    try:
        _fit_group(name, group, cfg, lut5, out, next(counter), run, sources, finished)
    except OOM_ERRORS as exc:
        remaining = [c for c in group if c["id"] not in finished]
        if len(group) <= 1 or not remaining:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = max(1, len(remaining) // 2)
        log.warning("classe %s : memoire insuffisante (%s) pour %d cellules, groupe coupe en %d + %d", name,
                    exc, len(remaining), half, len(remaining) - half)
        for part in (remaining[:half], remaining[half:]):
            if part:
                _fit_group_resilient(name, part, cfg, lut5, out, counter, run, sources)


def run_bin(name: str, cells: Sequence[dict], cfg: AdaptiveConfig, lut5: torch.Tensor | None, out: Path,
            done: set[int], run: int = 1, sources: dict | None = None) -> None:
    """Ajuste les cellules de la classe (hors `done`) par groupes de vsf.GROUP[name], sorties apres chaque cellule/groupe."""
    pending = [c for c in cells if c["id"] not in done]
    size = vsf.GROUP[name]
    sources = _merge_sources(out) if sources is None else sources
    log.info("classe %s : %d cellules a ajuster (%d deja faites)", name, len(pending), len(cells) - len(pending))
    if pending:
        warm_up(pending[0], lut5)
    counter = itertools.count()
    for g0 in range(0, len(pending), size):
        _fit_group_resilient(name, pending[g0:g0 + size], cfg, lut5, out, counter, run, sources)


def _share(s: pd.Series, reason: str) -> float:
    return float((s == reason).mean())


def summarize(df_cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables par classe : arrets ("stop"), K ("k") et erreurs ("error").

    nrmse_median / nrmse_p90 sont calcules sur les valeurs finies uniquement ; n_diverged compte les cellules "diverged".
    """
    if "bin" not in df_cells.columns:
        raise ValueError("cellule(s) sans colonne 'bin' : lignes de cells_*.csv invalides")
    if (df_cells.area <= 0).any() or (df_cells.K <= 0).any():
        raise ValueError("area et K doivent etre > 0 (logarithme de la pente K ~ area)")
    stop, kk, err = [], [], []
    for name, g in df_cells.groupby("bin", sort=False):
        n_diverged = int((g.stop_reason == "diverged").sum())
        if n_diverged:
            log.warning("classe %s : %d cellule(s) diverged (non rajustees a la reprise)", name, n_diverged)
        finite = g.nrmse[np.isfinite(g.nrmse)]
        stop.append(dict(bin=name, n=len(g), **{f"share_{r}": _share(g.stop_reason, r) for r in STOP_REASONS}))
        slope = float(np.polyfit(np.log(g.area), np.log(g.K), 1)[0]) if len(g) >= 2 and g.area.nunique() > 1 \
            else float("nan")
        kk.append(dict(bin=name, n=len(g), K_median=float(g.K.median()), K_q25=float(g.K.quantile(IQR_LOW)),
                       K_q75=float(g.K.quantile(IQR_HIGH)), slope_logK_logarea=slope))
        err.append(dict(bin=name, n=len(g), n_diverged=n_diverged,
                        nrmse_median=float(finite.median()) if len(finite) else float("nan"),
                        nrmse_p90=float(finite.quantile(P90)) if len(finite) else float("nan"),
                        share_target_reached=_share(g.stop_reason, "target"),
                        compression_ratio_median=float(g.compression_ratio.median()),
                        mass_bias_median=float(g.mass_bias.median()), peak_bias_median=float(g.peak_bias.median())))
    return {"stop": pd.DataFrame(stop), "k": pd.DataFrame(kk), "error": pd.DataFrame(err)}


def _write_summaries(out: Path) -> None:
    frames = [f for f in (_read_csv(p) for p in sorted(out.glob("cells_*.csv"))) if len(f)]
    if not frames:
        log.warning("aucune cellule ajustee : pas de synthese")
        return
    df = pd.concat(frames, ignore_index=True)
    if "bin" in df.columns:
        df = df.drop_duplicates(["bin", "id"], keep="last")
    for key, table in summarize(df).items():
        _rewrite_csv(out / f"summary_{key}.csv", table)
        log.info("synthese %s\n%s", key, table.to_string(index=False))


def _output_paths(out: Path) -> list[Path]:
    names = ["memory.csv", SOURCES_FILE, SOURCES_DIR, CONFIG_FILE]
    return [p for pat in ("cells_*.csv", "waves_*.csv", "summary_*.csv") for p in sorted(out.glob(pat))] + \
           [out / n for n in names if (out / n).exists()]


def prepare_output(out: Path, resume: bool, overwrite: bool) -> None:
    """Sans --resume : refuse un dossier contenant deja des sorties, sauf --overwrite (les efface)."""
    existing = _output_paths(out)
    if resume or not existing:
        return
    if not overwrite:
        raise FileExistsError(f"{out} contient deja des sorties ({', '.join(p.name for p in existing)}) : "
                              "utiliser --resume pour reprendre ou --overwrite pour les effacer")
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    log.warning("--overwrite : %d sorties effacees dans %s", len(existing), out)


def run_config(args: argparse.Namespace, spec: LogWLutSpec) -> dict:
    return dict(target=args.target, k_cap=args.k_cap, compression_frac=args.compression_frac,
                prune_frac=args.prune_frac, iters=args.iters, lut=spec.key(), arm=ARM)


def check_run_config(out: Path, config: dict, resume: bool) -> None:
    """Ecrit run_config.json au premier lancement ; a la reprise, refuse une configuration differente."""
    path = out / CONFIG_FILE
    config = json.loads(json.dumps(config))
    if resume and path.exists():
        previous = json.loads(path.read_text())
        if previous != config:
            diff = {k: (previous.get(k), config.get(k)) for k in config if previous.get(k) != config.get(k)}
            raise ValueError(f"reprise refusee : run_config.json differe (ancien, nouveau) {diff}")
        return
    _atomic_write_bytes(path, json.dumps(config, indent=2).encode())


def _run(args: argparse.Namespace, out: Path, spec: LogWLutSpec) -> None:
    cfg = AdaptiveConfig(arm=ARM, target=args.target, k_cap=args.k_cap, compression_frac=args.compression_frac,
                         prune_frac=args.prune_frac, iters_override=args.iters)
    prepare_output(out, args.resume, args.overwrite)
    check_run_config(out, run_config(args, spec), args.resume)
    sample = pickle.loads(Path(args.sample).read_bytes())  # nosec B301  (echantillon local, produit par le projet)
    cells = select_cells(sample, args.bins, args.max_per_bin, args.largest)
    lut5 = vsf.lut_for_arm(ARM, out, spec)
    done = {n: done_ids(out, n) if args.resume else set() for n in args.bins}
    if args.resume:
        for n in args.bins:
            drop_stale_rows(out, n, done[n])
    todo = [c for c in cells if c["id"] not in done[c["bin"]]]
    run, sources = _next_run_number(out), _merge_sources(out)
    t0 = time.perf_counter()
    for name in args.bins:
        members = [c for c in cells if c["bin"] == name]
        if members:
            run_bin(name, members, cfg, lut5, out, done[name], run, sources)
    if (out / SOURCES_DIR).exists():
        _flush_sources(out, _merge_sources(out))
    _write_summaries(out)
    log.info("%d cellules, duree totale %.1f s", len(todo), time.perf_counter() - t0)


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> LogWLutSpec:
    """Valide les options de la table (avant tout chargement) et le device ; retourne la specification."""
    if args.w_min >= args.w_max:
        ap.error(f"--w-min doit etre < --w-max (recu {args.w_min}, {args.w_max})")
    try:
        spec = LogWLutSpec(w_min=args.w_min, w_max=args.w_max, n_w=args.n_w, tau=args.tau7, n_r=args.n_r,
                           n=args.lut_n, steps=args.lut_steps)
    except ValueError as exc:
        ap.error(str(exc))
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"--device {args.device} demande mais cuda est indisponible")
    return spec


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.bins = dedupe(args.bins)
    spec = validate_args(ap, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    handler = logging.FileHandler(out / LOG_FILE)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(min(previous_level or logging.INFO, logging.INFO))
    previous_device = vsf.DEVICE
    vsf.DEVICE = torch.device(args.device)
    try:
        _run(args, out, spec)
    finally:
        vsf.DEVICE = previous_device
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous_level)


if __name__ == "__main__":
    main()
