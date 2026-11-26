"""Balayage de la taille des groupes de raffinement (E7, débit Jetson).

Pour chaque classe, les mêmes cellules (échantillons de benchmark, répliquées pour remplir les grands groupes) sont
raffinées avec des groupes de vsf.GROUP[classe] × multiplicateur. Sorties dans --out :
  sweep.csv       temps par cellule, accélération vs ×1, mémoire de pointe, écart max de nrmse vs ×1, OOM
  projection.csv  heures projetées sur le recensement (4,1 M cellules) avec le meilleur groupe par classe et
                  les itérations retenues (temps ∝ itérations ; insertion et encodeur non inclus)

  PYTHONPATH=src python scripts/refine_batch_sweep.py --ckpt results/10_e7/train_v3/best.pt \
      --samples results/03_variable_sources/sample_40.pkl results/03_variable_sources/sample_40_2017.pkl \
      --device cuda --out results/10_e7/batch_sweep
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

import variable_source_fit as vsf
from encoder_adoption import BUDGET_H, CENSUS
from encoder_eval import _load_sample, encode_cells, load_encoder
from encoder_step import _free_memory, is_oom
from profile_lut import LogWLutSpec, load_or_build_logw_lut
from refine_batching import refine_grouped, replicate

logger = logging.getLogger("refine_batch_sweep")
DEFAULT_MULTIPLIERS = (1, 4, 16, 64)
DEFAULT_ITERS = 50
PIPELINE_ITERS = {"50-200": 100, "200-1000": 200, "1000-5000": 300, ">=5000": 400}  # sondes d'insertion E7


def _load_all(paths) -> list[dict]:
    cells = [c for p in paths for c in _load_sample(Path(p))]
    return [{**c, "id": i} for i, c in enumerate(cells)]  # ids uniques entre années


def sweep_bin(name, cells, encoded, multipliers, iters, lut5, device) -> list[dict]:
    base = vsf.GROUP[name]
    pool = replicate(cells, max(len(cells), base * max(multipliers)))
    enc = {c["id"]: encoded[c["source_id"]] for c in pool}
    rows, ref = [], None
    for m in sorted(multipliers):
        try:
            run = refine_grouped(pool, enc, base * m, iters, lut5, device)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            _free_memory()
            logger.warning("%s ×%d : manque de mémoire, multiplicateurs suivants ignorés", name, m)
            rows.append(dict(bin=name, multiplier=m, group_size=base * m, n_cells=len(pool), oom=True))
            break
        ref = ref or run
        diff = max(abs(run.nrmse[i] - ref.nrmse[i]) for i in run.nrmse)
        rows.append(dict(bin=name, multiplier=m, group_size=base * m, n_cells=len(pool), n_groups=run.n_groups,
                         seconds_per_cell=run.seconds_per_cell, speedup=ref.seconds_per_cell / run.seconds_per_cell,
                         peak_mem_mb=run.peak_mem_mb, max_abs_nrmse_diff=diff,
                         median_nrmse=float(pd.Series(run.nrmse).median()), oom=False))
        logger.info("%s ×%d (groupe %d) : %.4f s/cellule, ×%.1f, %.0f Mo", name, m, base * m,
                    run.seconds_per_cell, rows[-1]["speedup"], run.peak_mem_mb)
    return rows


def projection(sweep: pd.DataFrame, iters: int) -> pd.DataFrame:
    ok = sweep[~sweep.oom.astype(bool)]
    best = ok.loc[ok.groupby("bin").seconds_per_cell.idxmin()]
    rows = []
    for _, r in best.iterrows():
        for label, rel in (("x1", ok[(ok.bin == r.bin) & (ok.multiplier == 1)].seconds_per_cell.iloc[0]),
                           ("best", r.seconds_per_cell)):
            per_cell = rel * PIPELINE_ITERS[r.bin] / iters
            rows.append(dict(bin=r.bin, groups=label, multiplier=1 if label == "x1" else r.multiplier,
                             pipeline_iters=PIPELINE_ITERS[r.bin], seconds_per_cell=per_cell,
                             hours=CENSUS[r.bin] * per_cell / 3600))
    out = pd.DataFrame(rows)
    totals = out.groupby("groups").hours.sum()
    logger.info("projection raffinement (hors encodeur ~1,8 h) : ×1 %.0f h, meilleur %.0f h, budget %.0f h",
                totals.get("x1", float("nan")), totals.get("best", float("nan")), BUDGET_H)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--samples", type=Path, nargs="+", required=True)
    ap.add_argument("--multipliers", type=int, nargs="+", default=list(DEFAULT_MULTIPLIERS))
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--bins", nargs="+", choices=[b[0] for b in vsf.BINS], default=[b[0] for b in vsf.BINS])
    ap.add_argument("--lut-dir", type=Path, default=Path("results/03_variable_sources"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.iters < 1 or min(a.multipliers) < 1 or 1 not in a.multipliers:
        raise SystemExit("--iters >= 1, multiplicateurs >= 1 et ×1 (référence) requis")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = torch.device(a.device)
    vsf.DEVICE = device
    model, cfg = load_encoder(a.ckpt, device)
    lut5 = load_or_build_logw_lut(LogWLutSpec(), a.lut_dir)["lut"][None, None].to(device)
    cells = _load_all(a.samples)
    with torch.no_grad():
        encoded = encode_cells(model, cfg, cells, device)
    first = [c for c in cells if c["bin"] == a.bins[0]][:1]
    refine_grouped(first, encoded, 1, 2, lut5, device)  # échauffement jetable
    rows = []
    for name in a.bins:
        rows += sweep_bin(name, [c for c in cells if c["bin"] == name], encoded, a.multipliers, a.iters, lut5, device)
    a.out.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(a.out / "sweep.csv", index=False)
    proj = projection(sweep, a.iters)
    proj.to_csv(a.out / "projection.csv", index=False)
    print(sweep.round(4).to_string(index=False))
    print(proj.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
