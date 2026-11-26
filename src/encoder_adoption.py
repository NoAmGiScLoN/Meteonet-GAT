"""Règle d'adoption de l'encodeur (phase 6 E7) : comparaison à E5 sur les mêmes cellules et projection de débit.

Règle approuvée, évaluée par (année, classe, niveau) : sans raffinement, médiane(nrmse E7) <= médiane(nrmse E5) + 0.02 ;
avec raffinement, <= + 0.005 ; dans les deux cas params/aire médian <= 1.1 x E5 (le raffinement ne change pas K).
Les médianes portent sur l'ensemble des cellules de chaque côté ; toutes les années doivent passer. Un nrmse E7 non
fini est compté (n_nonfinite) et fait échouer la classe. La différence médiane appariée (même id) n'est qu'une
information (`paired_diff_median_info`). Le temps projeté sur le jeu complet (recensement par classe) est comparé au
budget de 36 h seulement pour une mesure sur cuda ; il exclut l'E/S et le calcul des statistiques résiduelles, et
n'inclut l'extraction des recadrages que si `crop_s` est fourni.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import variable_source_fit as vsf

logger = logging.getLogger(__name__)

CENSUS = {"50-200": 2_700_000, "200-1000": 1_100_000, "1000-5000": 270_000, ">=5000": 80_000}  # cellules par classe
BUDGET_H = 36.0
TOL_NO_REFINE = 0.02
TOL_REFINED = 0.005
PPA_FACTOR = 1.1
SECONDS_PER_HOUR = 3600.0
SMALL_BINS = ("50-200", "200-1000", "1000-5000")  # E5 adopté : runs *_c050 ; >= 5000 px : runs sans compression
LARGE_BIN = ">=5000"
BY_YEAR_COLUMNS = ["year", "bin", "refine_iters", "n_e5", "n_e7", "n_paired", "n_nonfinite", "e5_nrmse", "e7_nrmse",
                   "paired_diff_median_info", "e5_ppa", "e7_ppa", "ppa_ratio", "nrmse_ok", "ppa_ok", "passed"]
ADOPTION_COLUMNS = ["bin", "refine_iters", "n_years", "years_passed", "n_paired", "n_nonfinite", "e5_nrmse",
                    "e7_nrmse", "paired_diff_median_info", "ppa_ratio", "passed"]
THROUGHPUT_COLUMNS = ["refine_iters", "device", "crop_s", "hours", "within_budget"]


def load_e5(e5_dirs: Mapping[str, tuple[Path, Path]]) -> pd.DataFrame:
    """Cellules E5 adoptées : {année: (dossier classes < 5000 px, dossier classes >= 5000 px)} -> trame avec `year`."""
    frames = []
    for year, (small_dir, large_dir) in e5_dirs.items():
        for name, *_ in vsf.BINS:
            folder = Path(large_dir if name == LARGE_BIN else small_dir)
            frames.append(pd.read_csv(folder / f"cells_{vsf.bin_slug(name)}.csv").assign(year=str(year)))
    return pd.concat(frames, ignore_index=True)


def _with_year(frame: pd.DataFrame) -> pd.DataFrame:
    return frame if "year" in frame else frame.assign(year="")


def check_no_duplicates(e5: pd.DataFrame) -> None:
    dup = _with_year(e5).duplicated(["year", "id"])
    if dup.any():
        raise ValueError(f"E5 : {int(dup.sum())} doublon(s) (year, id)")


def check_e5_matches_samples(e5: pd.DataFrame, cells_by_year: Mapping[str, Sequence[dict]]) -> None:
    """Chaque ligne E5 doit désigner la cellule de même id de l'échantillon (aire, et tag/date si présents)."""
    for row in _with_year(e5).itertuples(index=False):
        cells = cells_by_year.get(str(row.year))
        if cells is None or not 0 <= row.id < len(cells):
            raise ValueError(f"E5 : cellule {row.year}/{row.id} absente de l'échantillon")
        cell = cells[row.id]
        if int(row.area) != int(cell["area"]):
            raise ValueError(f"E5 {row.year}/{row.id} : aire {row.area} != {cell['area']} (mauvais échantillon ?)")
        for name in ("tag", "date"):
            if hasattr(row, name) and str(getattr(row, name)) != str(cell[name]):
                raise ValueError(f"E5 {row.year}/{row.id} : {name} {getattr(row, name)} != {cell[name]}")


def _finite_median(values: pd.Series) -> float:
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else float("nan")


def _year_row(year: str, bin_name: str, level: int, e7: pd.DataFrame, e5: pd.DataFrame) -> dict:
    pair = e7.merge(e5, on="id", suffixes=("_e7", "_e5"))
    n_bad = int((~np.isfinite(e7["nrmse"])).sum())
    e5_nrmse, e7_nrmse = _finite_median(e5["nrmse"]), _finite_median(e7["nrmse"])
    e5_ppa, e7_ppa = float(e5["compression_ratio"].median()), float(e7["compression_ratio"].median())
    tol = TOL_NO_REFINE if level == 0 else TOL_REFINED
    nrmse_ok = bool(n_bad == 0 and e7_nrmse <= e5_nrmse + tol)
    ppa_ok = bool(e7_ppa <= PPA_FACTOR * e5_ppa)
    diff = (pair["nrmse_e7"] - pair["nrmse_e5"])
    return dict(year=year, bin=bin_name, refine_iters=level, n_e5=len(e5), n_e7=len(e7), n_paired=len(pair),
                n_nonfinite=n_bad, e5_nrmse=e5_nrmse, e7_nrmse=e7_nrmse,
                paired_diff_median_info=_finite_median(diff) if len(pair) else float("nan"),
                e5_ppa=e5_ppa, e7_ppa=e7_ppa, ppa_ratio=e7_ppa / e5_ppa, nrmse_ok=nrmse_ok, ppa_ok=ppa_ok,
                passed=nrmse_ok and ppa_ok)


def adoption_by_year(e7: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (année, classe, niveau de raffinement) selon la règle ci-dessus."""
    check_no_duplicates(e5)
    e7, e5 = _with_year(e7), _with_year(e5)
    cols = ["id", "nrmse", "compression_ratio"]
    rows = []
    for (year, bin_name, level), group in e7.groupby(["year", "bin", "refine_iters"], sort=True):
        reference = e5[(e5["year"] == year) & (e5["bin"] == bin_name)][cols]
        rows.append(_year_row(year, bin_name, int(level), group[cols], reference))
    return pd.DataFrame(rows, columns=BY_YEAR_COLUMNS)


def adoption_table(e7: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
    """Verdict combiné par (classe, niveau) : réussi seulement si toutes les années réussissent."""
    by_year = adoption_by_year(e7, e5)
    rows = []
    for (bin_name, level), g in by_year.groupby(["bin", "refine_iters"], sort=True):
        rows.append(dict(bin=bin_name, refine_iters=int(level), n_years=len(g), years_passed=int(g["passed"].sum()),
                         n_paired=int(g["n_paired"].sum()), n_nonfinite=int(g["n_nonfinite"].sum()),
                         e5_nrmse=float(g["e5_nrmse"].mean()), e7_nrmse=float(g["e7_nrmse"].mean()),
                         paired_diff_median_info=float(g["paired_diff_median_info"].mean()),
                         ppa_ratio=float(g["ppa_ratio"].mean()), passed=bool(g["passed"].all())))
    return pd.DataFrame(rows, columns=ADOPTION_COLUMNS)


def throughput_projection(e7: pd.DataFrame, device: str, census: Mapping[str, int] = CENSUS,
                          budget_h: float = BUDGET_H, crop_s: float = 0.0, strict: bool = True) -> pd.DataFrame:
    """Heures projetées sur le jeu complet par niveau : somme sur les classes du recensement de cellules x temps moyen
    par cellule (encode_s + refine_s + crop_s). Toutes les classes du recensement doivent être mesurées (sinon
    ValueError ; avec strict=False : NaN et avertissement). within_budget n'est déterminé que pour device == "cuda" (sinon NaN et avertissement)."""
    seconds = e7["encode_s"] + e7["refine_s"] + crop_s
    on_cuda = str(device).startswith("cuda")
    if not on_cuda:
        logger.warning("mesure sur %s : verdict de budget indéterminé (NaN), seul cuda est représentatif", device)
    rows = []
    for level, group in e7.assign(_s=seconds).groupby("refine_iters", sort=True):
        per_bin = group.groupby("bin")["_s"].mean()
        missing = [b for b in census if b not in per_bin]
        message = f"projection : classe(s) du recensement sans mesure au niveau {level} : {missing}"
        if missing and strict:
            raise ValueError(message)
        if missing:
            logger.warning(message)
        hours = float("nan") if missing else float(sum(census[b] * per_bin[b] for b in census) / SECONDS_PER_HOUR)
        within = bool(hours <= budget_h) if on_cuda and not missing else float("nan")
        rows.append(dict(refine_iters=int(level), device=str(device), crop_s=crop_s, hours=hours, within_budget=within))
        logger.info("projection raffinement %d : %.1f h (budget %.0f h)", level, hours, budget_h)
    return pd.DataFrame(rows, columns=THROUGHPUT_COLUMNS)
