"""Tracker v2 mois par mois : sorties parquet <out>/<aaaa-mm>/ (detections, lineage, events,
state_end.pkl, month_summary.json, puis DONE en dernier), écritures atomiques (tmp + fsync +
os.replace). Un mois DONE est sauté à la reprise (état rechargé) ; les identifiants de piste
sont globaux et la 1re carte d'un mois est liée à la dernière du précédent comme en continu.
"""
import argparse
import json
import logging
import os
import pickle  # nosec B403  (état local produit par le projet)
import shutil
import subprocess  # nosec B404
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cell_delimitation import DATA_DIR
from frame_stream import DEFAULT_STEP_MIN, Frame, NoFramesError, check_step, iter_frames
from tracker_state import TrackerConfig, TrackerState
from tracker_v2 import CELL_COLUMNS, DET_EXTRA, EVENT_COLUMNS, LINEAGE_COLUMNS, step

logger = logging.getLogger(__name__)

STATE_SCHEMA = 1  # à incrémenter quand Track / TrackerState changent (états .pkl incompatibles)
LOAD_ERRORS = (OSError, EOFError, pickle.UnpicklingError, ValueError, KeyError, TypeError,
               AttributeError, ImportError)
DONE_FILE, CONFIG_FILE = "DONE", "run_config.json"
STATE_FILE, SUMMARY_FILE = "state_end.pkl", "month_summary.json"
TABLES = ("detections", "lineage", "events")
PROGRESS_EVERY = 1000  # cartes entre deux messages de progression
OVERRIDES = ("min_area_start", "min_area_keep", "max_coast", "young_frames", "d_max",
             "max_cost", "alpha")
COLUMNS = {"detections": CELL_COLUMNS + DET_EXTRA, "lineage": LINEAGE_COLUMNS,
           "events": EVENT_COLUMNS}
STR = "str" if int(pd.__version__.split(".")[0]) >= 3 else "object"
TS, I64, F64 = "datetime64[ns]", "int64", "float64"
DTYPES = {
    "detections": {
        "timestamp": TS, "local_id": I64, "area_px": I64, "cy": F64, "cx": F64, "cy_w": F64,
        "cx_w": F64, "intensity_sum": F64, "intensity_mean": F64, "intensity_max": "int16",
        "border_flag": bool, "missing_neighbor_flag": bool, "n_holes_bbox": I64,
        "track_id": I64, "truncated": bool, "coasted_before": I64, "v_y": F64, "v_x": F64,
        "v_src": STR, "pred_src": STR, "velocity_reset": bool},
    "lineage": {
        "t": TS, "parent_track": I64, "child_track": I64, "link_type": STR,
        "overlap_frac_parent": F64, "overlap_frac_child": F64, "share_of_child": F64,
        "share_of_parent": F64, "is_main": bool},
    "events": {"t": TS, "track_id": I64, "event": STR, "truncated": bool, "area": I64,
               "notes": STR},
}


@dataclass(frozen=True)
class MonthSummary:
    """Bilan d'un mois : cartes, cellules, cellules suivies, pistes distinctes, événements."""
    month: str
    frames: int
    cells: int
    tracked_cells: int
    tracks: int
    events: dict[str, int]
    seconds: float
    git_sha: str = "unknown"

    @property
    def frames_per_s(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def assemble(parts: Sequence[pd.DataFrame], kind: str) -> pd.DataFrame:
    """Concatène les tables d'un type (detections | lineage | events) avec des dtypes stables."""
    filled = [p for p in parts if len(p)]
    table = (pd.concat(filled, ignore_index=True) if filled
             else pd.DataFrame(columns=COLUMNS[kind]))
    return table.astype(DTYPES[kind])


def month_window(year: int, month: int, start_time: datetime | None = None,
                 end_time: datetime | None = None) -> tuple[datetime, datetime]:
    """Intervalle [début, fin] du mois, éventuellement restreint par start_time / end_time."""
    first = datetime(year, month, 1)
    nxt = datetime(year + (month == 12), month % 12 + 1, 1)
    start, end = first, nxt - timedelta(seconds=1)
    if start_time is not None:
        start = max(start, start_time)
    if end_time is not None:
        end = min(end, end_time)
    return start, end


def _fsync_path(path: Path) -> None:
    """fsync d'un fichier ou d'un répertoire (le répertoire : au mieux, selon la plateforme)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _commit(tmp: Path, path: Path) -> None:
    _fsync_path(tmp)
    os.replace(tmp, path)
    _fsync_path(path.parent)


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    _commit(tmp, path)


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False)
    _commit(tmp, path)


def _link_first_frame(frame: Frame, last_t: datetime | None, cfg: TrackerConfig) -> Frame:
    """1re carte d'un mois : pas écoulés et coupure calculés depuis la dernière carte traitée."""
    if last_t is None:
        return frame
    steps = check_step(last_t, frame.t, DEFAULT_STEP_MIN)
    gap = (frame.t - last_t).total_seconds() / 60 > cfg.max_gap_min
    return replace(frame, dt_steps=steps, gap_break=gap)


def _summarize(label: str, frames: int, parts: dict[str, list], seconds: float,
               git_sha: str) -> tuple:
    tables = {k: assemble(parts[k], k) for k in TABLES}
    det = tables["detections"]
    tracked = det[det.track_id > 0]
    counts = tables["events"].event.value_counts()
    events = {str(k): int(v) for k, v in counts.items()}
    return tables, MonthSummary(label, frames, len(det), len(tracked),
                                int(tracked.track_id.nunique()), events, seconds, git_sha)


def run_month(year: int, month: int, cfg: TrackerConfig, out_dir: Path, state: TrackerState,
              prev_R: np.ndarray | None, data_dir: Path = DATA_DIR,
              start_time: datetime | None = None, end_time: datetime | None = None,
              git_sha: str = "unknown") -> tuple[TrackerState, np.ndarray | None, MonthSummary]:
    """Traite un mois et écrit ses sorties ; retourne (état, dernière carte, bilan)."""
    label = f"{year}-{month:02d}"
    month_dir = Path(out_dir) / label
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / DONE_FILE).unlink(missing_ok=True)
    start, end = month_window(year, month, start_time, end_time)
    parts: dict[str, list] = {k: [] for k in TABLES}
    t0, n = time.perf_counter(), 0
    try:
        for frame in iter_frames(start, end, data_dir, cfg.max_gap_min):
            frame = _link_first_frame(frame, state.t, cfg) if n == 0 else frame
            state, out = step(state, frame, cfg, prev_R)
            prev_R = frame.R
            for kind in TABLES:
                parts[kind].append(getattr(out, kind))
            n += 1
            if n % PROGRESS_EVERY == 0:
                logger.info("%s : %d cartes (%.2f cartes/s)", label, n,
                            n / (time.perf_counter() - t0))
    except (FileNotFoundError, NoFramesError) as exc:
        if n > 0:
            raise
        logger.warning("%s : aucune donnée (%s) ; mois vide, état inchangé", label, exc)
    tables, summary = _summarize(label, n, parts, time.perf_counter() - t0, git_sha)
    _write_month(month_dir, tables, state, prev_R, summary)
    logger.info("%s : %d cartes, %d cellules, %d pistes en %.1f s (%.2f cartes/s)", label,
                summary.frames, summary.cells, summary.tracks, summary.seconds,
                summary.frames_per_s)
    return state, prev_R, summary


def _write_month(month_dir: Path, tables: dict[str, pd.DataFrame], state: TrackerState,
                 prev_R: np.ndarray | None, summary: MonthSummary) -> None:
    for kind, table in tables.items():
        _atomic_parquet(table, month_dir / f"{kind}.parquet")
    payload = {"state": state, "prev_R": prev_R, "t": state.t}
    _atomic_write(month_dir / STATE_FILE, pickle.dumps(payload))
    _atomic_write(month_dir / SUMMARY_FILE, json.dumps(asdict(summary), indent=2).encode())
    _atomic_write(month_dir / DONE_FILE, b"")  # en dernier : mois complet


def load_state(month_dir: Path) -> tuple[TrackerState, np.ndarray | None]:
    """Recharge (état, prev_R) d'un mois terminé."""
    payload = pickle.loads((month_dir / STATE_FILE).read_bytes())  # nosec B301
    return payload["state"], payload["prev_R"]


def months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Liste (année, mois) de start à end inclus ; ValueError si end < start."""
    if (end.year, end.month) < (start.year, start.month):
        raise ValueError(f"Mois de fin {end:%Y-%m} antérieur au début {start:%Y-%m}")
    index = range(start.year * 12 + start.month - 1, end.year * 12 + end.month)
    return [(i // 12, i % 12 + 1) for i in index]


def _load_summary(month_dir: Path) -> MonthSummary:
    return MonthSummary(**json.loads((month_dir / SUMMARY_FILE).read_text()))


def _resume_point(months: Sequence[tuple[int, int]], out_dir: Path
                  ) -> tuple[list[MonthSummary], TrackerState, np.ndarray | None]:
    """Préfixe de mois terminés (DONE) lisibles, avec l'état de fin du dernier si des mois
    restent à traiter. Un dernier mois dont le bilan ou l'état est illisible perd son DONE
    et sera refait depuis l'état du mois précédent."""
    dirs = []
    for year, month in months:
        month_dir = Path(out_dir) / f"{year}-{month:02d}"
        if not (month_dir / DONE_FILE).exists():
            break
        dirs.append(month_dir)
    while dirs:
        try:
            summaries = [_load_summary(d) for d in dirs]
            if len(dirs) == len(months):
                return summaries, TrackerState(), None  # rien à traiter : état inutile
            state, prev_R = load_state(dirs[-1])
            return summaries, state, prev_R
        except LOAD_ERRORS as exc:
            logger.warning("%s : sorties illisibles (%s), mois refait", dirs[-1].name, exc)
            (dirs.pop() / DONE_FILE).unlink(missing_ok=True)
    return [], TrackerState(), None


def run_range(months: Sequence[tuple[int, int]], cfg: TrackerConfig, out_dir: Path,
              data_dir: Path = DATA_DIR, start_time: datetime | None = None,
              end_time: datetime | None = None, git_sha: str = "unknown"
              ) -> list[MonthSummary]:
    """Enchaîne les mois en sautant ceux qui portent DONE (état du dernier rechargé)."""
    summaries, state, prev_R = _resume_point(months, out_dir)
    if summaries:
        logger.info("%d mois déjà terminé(s), sautés (reprise)", len(summaries))
    for year, month in months[len(summaries):]:
        state, prev_R, summary = run_month(year, month, cfg, out_dir, state, prev_R, data_dir,
                                           start_time, end_time, git_sha)
        summaries.append(summary)
    return summaries


def prepare_output(out: Path, resume: bool, overwrite: bool) -> None:
    """--overwrite vide la sortie ; sans --resume une sortie existante est refusée."""
    if overwrite and out.exists():
        shutil.rmtree(out)
    elif (out / CONFIG_FILE).exists() and not resume:
        raise FileExistsError(f"{out} contient déjà une exécution : utiliser --resume ou "
                              "--overwrite")
    out.mkdir(parents=True, exist_ok=True)


def check_run_config(out: Path, config: dict, resume: bool) -> None:
    """Écrit run_config.json ; à la reprise, refuse une configuration différente.

    La clé « provenance » (SHA git, répertoire de données) est enregistrée mais jamais
    comparée. Reprise sans run_config.json : refusée si des mois sont déjà terminés."""
    path = out / CONFIG_FILE
    config = json.loads(json.dumps(config))
    if resume and path.exists():
        previous = json.loads(path.read_text())
        old = {k: v for k, v in previous.items() if k != "provenance"}
        new = {k: v for k, v in config.items() if k != "provenance"}
        if old != new:
            diff = {k: (old.get(k), new.get(k)) for k in new if old.get(k) != new.get(k)}
            raise ValueError(f"reprise refusée : run_config.json diffère (ancien, nouveau) {diff}")
        return
    if resume and any(out.glob(f"*/{DONE_FILE}")):
        raise ValueError(f"reprise refusée : run_config.json absent de {out} alors que des "
                         "mois sont terminés")
    _atomic_write(path, json.dumps(config, indent=2).encode())


def git_sha() -> str:
    """SHA du commit courant, ou 'unknown' hors dépôt git."""
    try:
        run = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)  # nosec B603 B607
        sha = run.stdout.strip() if run.returncode == 0 else ""
        return sha or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def format_table(summaries: Sequence[MonthSummary]) -> str:
    """Tableau final : pistes, longueur moyenne (cellules suivies / pistes ; une piste à cheval
    sur deux mois compte dans chacun) et événements pour 1000 pistes."""
    lines = [f"{'mois':8} {'cartes':>8} {'pistes':>8} {'long.moy':>9} {'evt/1000p':>10}"]
    rows = list(summaries) + [MonthSummary(
        "total", sum(s.frames for s in summaries), sum(s.cells for s in summaries),
        sum(s.tracked_cells for s in summaries), sum(s.tracks for s in summaries),
        {"all": sum(sum(s.events.values()) for s in summaries)}, 0.0)]
    for s in rows:
        length = s.tracked_cells / s.tracks if s.tracks else 0.0
        per_1000 = 1000 * sum(s.events.values()) / s.tracks if s.tracks else 0.0
        lines.append(f"{s.month:8} {s.frames:>8} {s.tracks:>8} {length:>9.1f} {per_1000:>10.1f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tracker v2 mois par mois (parquet)")
    p.add_argument("--start", required=True, help="premier mois YYYY-MM")
    p.add_argument("--end", required=True, help="dernier mois YYYY-MM")
    p.add_argument("--out", type=Path, default=Path("results/tracker_v2"))
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--start-time", help="début ISO (restreint le premier mois)")
    p.add_argument("--end-time", help="fin ISO (restreint le dernier mois)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    defaults = TrackerConfig()
    for field in fields(defaults):
        if field.name in OVERRIDES:
            p.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name, default=None,
                           type=type(getattr(defaults, field.name)))
    return p


def _config_from(args: argparse.Namespace) -> TrackerConfig:
    changes = {n: getattr(args, n) for n in OVERRIDES if getattr(args, n) is not None}
    return replace(TrackerConfig(), **changes)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée CLI ; retourne 0 en cas de succès."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    start, end = datetime.strptime(args.start, "%Y-%m"), datetime.strptime(args.end, "%Y-%m")
    start_time = datetime.fromisoformat(args.start_time) if args.start_time else None
    end_time = datetime.fromisoformat(args.end_time) if args.end_time else None
    months, cfg = months_between(start, end), _config_from(args)
    prepare_output(args.out, args.resume, args.overwrite)
    sha = git_sha()
    check_run_config(args.out, {"tracker": asdict(cfg), "start": args.start,
                                "start_time": args.start_time, "state_schema": STATE_SCHEMA,
                                "provenance": {"git_sha": sha, "data_dir": str(args.data_dir)}},
                     args.resume)
    summaries = run_range(months, cfg, args.out, args.data_dir, start_time, end_time, sha)
    logger.info("Bilan\n%s", format_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
