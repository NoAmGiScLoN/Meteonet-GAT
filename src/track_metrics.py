"""Métriques de qualité d'une exécution du tracker v2 (sorties de track_run) : longueur des
pistes, événements, masse de pluie suivie, proxys de changement d'identité, succès des coasts,
« flicker » (pistes mortes après un coast mais retrouvées avec 2 cartes de coast) et planche de
revue visuelle. CLI : python src/track_metrics.py <out_dir> écrit <out_dir>/_report/.
"""
import argparse
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cell_delimitation import DATA_DIR
from lineage import EVENT_TYPES
from track_flicker import flicker_rates, flicker_table
from track_review import EVENTS_PER_PAGE, ReviewEvent, write_review
from track_tables import SIZE_LABELS, STEP, RunTables, load_run, size_class

logger = logging.getLogger(__name__)

__all__ = ["EVENTS_PER_PAGE", "RunTables", "load_run", "compute_metrics", "flicker_table",
           "flicker_rates", "review_sheet", "main"]

SHORT_MAX, LONG_MIN, MASS_MIN_AREA, MASS_MIN_FRAMES = 2, 12, 50, 3  # seuils en cartes / px
AREA_RATIO_MAX = 3.0
REPORT_DIR = "_report"
END_EVENTS = ("death", "exit_edge", "exit_missing", "merge_end", "gap_break")


def _tracks(det: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par piste : length, max_area, size_class, censored.

    Censurée = 1re détection à la 1re carte de l'exécution, ou à une carte de gap_break (piste
    née à la reprise), ou dernière détection à la dernière carte, ou piste coupée par un
    gap_break. Bornes de l'exécution = extrêmes des détections et des événements ; les
    gap_break ne sont connus que s'il existait des pistes à couper."""
    tracked = det[det.track_id > 0]
    g = tracked.groupby("track_id")
    out = g.agg(length=("timestamp", "size"), max_area=("area_px", "max"),
                first=("timestamp", "min"), last=("timestamp", "max"))
    bounds = [x for x in (det.timestamp.min(), ev.t.min()) if pd.notna(x)]
    ends = [x for x in (det.timestamp.max(), ev.t.max()) if pd.notna(x)]
    gaps = ev[ev.event == "gap_break"]
    out["censored"] = ((out["first"] == min(bounds)) | (out["last"] == max(ends))
                       | out["first"].isin(gaps.t) | out.index.isin(gaps.track_id))
    out["size_class"] = size_class(out.max_area)
    return out.drop(columns=["first", "last"])


def _length_stats(lengths: pd.Series) -> dict:
    if not len(lengths):
        nan = float("nan")
        return {"n_tracks": 0, "median": nan, "p90": nan, "share_le_2": nan, "share_ge_12": nan}
    return {"n_tracks": int(len(lengths)), "median": float(lengths.median()),
            "p90": float(lengths.quantile(0.9)),
            "share_le_2": float((lengths <= SHORT_MAX).mean()),
            "share_ge_12": float((lengths >= LONG_MIN).mean())}


def _pairs(det: pd.DataFrame) -> pd.DataFrame:
    """Paires de détections successives d'une piste : rapport d'aires, reset, |Δv|, à un pas."""
    d = det[det.track_id > 0].sort_values(["track_id", "timestamp"])
    g = d.groupby("track_id")
    prev_area, prev_vy, prev_vx = g.area_px.shift(), g.v_y.shift(), g.v_x.shift()
    pairs = pd.DataFrame({"ratio": d.area_px / prev_area, "reset": d.velocity_reset,
                          "dv": np.hypot(d.v_y - prev_vy, d.v_x - prev_vx),
                          "one_step": (d.timestamp - g.timestamp.shift()) == STEP})
    return pairs[prev_area.notna()]


def _switch_stats(det: pd.DataFrame) -> dict:
    """Proxys de changement d'identité sur les paires à un pas ; paires à cheval sur un trou
    comptées à part (n_pairs_across_gap) et exclues des proxys."""
    allp = _pairs(det)
    pairs = allp[allp.one_step]
    stats = {"n_pairs": int(len(pairs)), "n_pairs_across_gap": int(len(allp) - len(pairs))}
    if not len(pairs):
        nan = float("nan")
        return {**stats, "share_area_ratio": nan, "share_velocity_reset": nan,
                "median_abs_dv": nan}
    jump = (pairs.ratio > AREA_RATIO_MAX) | (pairs.ratio < 1 / AREA_RATIO_MAX)
    return {**stats, "share_area_ratio": float(jump.mean()),
            "share_velocity_reset": float(pairs.reset.astype(bool).mean()),
            "median_abs_dv": float(pairs.dv.median())}


def _next_after(coasts: pd.DataFrame, times: pd.DataFrame, name: str) -> pd.DataFrame:
    """Date (colonne `name`) du prochain enregistrement de `times` de la même piste, après t."""
    times = times.rename(columns={"t": name}).sort_values(name)
    return pd.merge_asof(coasts.sort_values("t"), times, left_on="t", right_on=name,
                         by="track_id", direction="forward", allow_exact_matches=False)


def _coast_stats(tables: RunTables) -> dict:
    """Succès d'un coast : prochain enregistrement de la piste = détection (succès) ou fin
    (échec). Exclus : coast sans suite (fin de l'exécution) ou suivi d'une carte absente (la
    suite n'arrive pas au pas attendu, trou dans les données)."""
    ev, det = tables.events, tables.detections
    c = ev.loc[ev.event == "coast", ["track_id", "t"]].sort_values(["track_id", "t"])
    c = c.assign(cum=c.groupby("track_id").cumcount()).reset_index(drop=True)
    if not len(c):
        return {"n": 0, "n_continued": 0, "n_excluded": 0, "success_rate": float("nan")}
    dets = det.loc[det.track_id > 0, ["track_id", "timestamp"]].rename(
        columns={"timestamp": "t"}).drop_duplicates()
    ends = ev.loc[ev.event.isin(END_EVENTS), ["track_id", "t"]]
    c = _next_after(_next_after(c, dets, "t_det"), ends, "t_end")
    c["found"] = c.t_det.notna() & (c.t_end.isna() | (c.t_det <= c.t_end))
    c["t_next"] = c.t_det.where(c.found, c.t_end)
    c = c.sort_values("t_next").reset_index(drop=True)
    known = c[c.t_next.notna()]
    last = pd.merge_asof(known[["track_id", "t_next"]], c[["track_id", "t", "cum"]].sort_values(
        "t"), left_on="t_next", right_on="t", by="track_id", allow_exact_matches=False)
    between = (last.cum - known.cum.to_numpy()).clip(lower=0).to_numpy()
    on_time = ((known.t_next - known.t) / STEP).round().to_numpy() == between + 1
    kept = known[on_time]
    n_ok, n = int(kept.found.sum()), len(kept)
    return {"n": n, "n_continued": n_ok, "n_excluded": int(len(c) - n),
            "success_rate": n_ok / n if n else float("nan")}


def _mass_share(det: pd.DataFrame, tracks: pd.DataFrame) -> float:
    """Part de la masse (cellules >= 50 px) des pistes de plus de 3 cartes. Une piste censurée
    d'au plus 3 cartes observées est indécidable : exclue du numérateur et du dénominateur."""
    big = det[det.area_px >= MASS_MIN_AREA]
    undecided = tracks.index[tracks.censored & (tracks.length <= MASS_MIN_FRAMES)]
    big = big[~big.track_id.isin(undecided)]
    total = big.intensity_sum.sum()
    if total <= 0:
        return float("nan")
    long_ids = tracks.index[tracks.length > MASS_MIN_FRAMES]
    return float(big[big.track_id.isin(long_ids)].intensity_sum.sum() / total)


def _by_class(tracks: pd.DataFrame) -> dict:
    return {c: _length_stats(tracks.length[tracks.size_class == c]) for c in SIZE_LABELS}


def compute_metrics(tables: RunTables) -> dict:
    """Ensemble des métriques de l'exécution (dict sérialisable en JSON).

    length / by_class : pistes non censurées (référence) ; length_all / by_class_all : toutes."""
    det, ev = tables.detections, tables.events
    tracks = _tracks(det, ev)
    n = len(tracks)
    sure = tracks[~tracks.censored]
    counts = ev.event.value_counts() if len(ev) else pd.Series(dtype=int)
    per_1000 = {k: (1000 * int(counts.get(k, 0)) / n if n else float("nan"))
                for k in EVENT_TYPES}
    return {"n_tracks": n, "n_censored": int(tracks.censored.sum()),
            "length": _length_stats(sure.length), "length_all": _length_stats(tracks.length),
            "by_class": _by_class(sure), "by_class_all": _by_class(tracks),
            "events_per_1000_tracks": per_1000,
            "mass_share_long_tracks": _mass_share(det, tracks),
            "switch": _switch_stats(det), "coast": _coast_stats(tables)}


def review_sheet(out_dir: Path, data_dir: Path = DATA_DIR, n: int = 40, seed: int = 0,
                 png_path: Path | None = None, months: Sequence[str] | None = None
                 ) -> list[ReviewEvent]:
    """Échantillon stratifié d'événements en planches PNG + review_events.csv."""
    tables = load_run(out_dir, months)
    png = Path(png_path) if png_path else Path(out_dir) / REPORT_DIR / "review.png"
    return write_review(tables.events, tables.detections, Path(data_dir), n, seed, png)


def _jsonable(obj: object) -> object:
    """NaN -> None et types numpy -> natifs, pour un JSON valide."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Métriques et planches de revue du tracker v2")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--months", nargs="+", default=None, help="mois aaaa-mm (défaut : tous)")
    p.add_argument("--review-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Écrit metrics.json, metrics_by_class.csv, flicker.csv et la revue dans _report/."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    report = args.out_dir / REPORT_DIR
    report.mkdir(parents=True, exist_ok=True)
    tables = load_run(args.out_dir, args.months)
    metrics = compute_metrics(tables)
    flicker = flicker_table(tables)
    metrics["flicker"] = flicker_rates(flicker)
    (report / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    by_class = pd.concat({"uncensored": pd.DataFrame(metrics["by_class"]).T,
                          "all": pd.DataFrame(metrics["by_class_all"]).T}, names=["scope"])
    by_class.to_csv(report / "metrics_by_class.csv", index_label=["scope", "size_class"])
    flicker.to_csv(report / "flicker.csv", index=False)
    logger.info("%d pistes ; métriques écrites dans %s", metrics["n_tracks"], report)
    try:
        write_review(tables.events, tables.detections, args.data_dir, args.review_n, args.seed,
                     report / "review.png")
    except Exception:  # noqa: BLE001  (les métriques sont déjà écrites : ne pas tout perdre)
        logger.exception("planches de revue non produites")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
