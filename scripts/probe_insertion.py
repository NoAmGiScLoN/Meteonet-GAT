"""Sonde E7 : insertion de sources aux pics du résidu avant raffinement, face à E5 sur les mêmes cellules.

Variantes : none (encodeur seul), x1.5 (jusqu'à 1.5 × k_enc), limit (jusqu'à la limite de compression D24) ;
chaque variante est raffinée à chaque niveau d'itérations (Adam neuf par niveau, comme le benchmark).
Sorties : <out>/cells.csv (une ligne par cellule × variante × niveau) et <out>/summary.csv (médianes par classe,
écart apparié à E5, params/aire rapporté à E5).
L'écart à E5 n'est pas à calcul égal : E5 ajuste par vagues adaptatives (temps e5_seconds = fit_s, mesuré sur Jetson).

  PYTHONPATH=src python scripts/probe_insertion.py --ckpt best.pt --sample results/03_variable_sources/sample_40.pkl \
      --e5 archive/E5_adaptive_k/results/adaptive_2016_c050 archive/E5_adaptive_k/results/adaptive_2016 --out out/
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import variable_source_fit as vsf
from encoder_adoption import check_e5_matches_samples, check_no_duplicates, load_e5
from encoder_eval import _load_sample, _pack, _sync, _targets, encode_cells, load_encoder, refine
from encoder_insertion import insert_sources, insertion_targets
from profile_lut import LogWLutSpec, load_or_build_logw_lut

logger = logging.getLogger("probe_insertion")


@dataclass(frozen=True)
class Variant:
    insert: bool
    factor: float | None = None  # None avec insert : remplir jusqu'à la limite de compression


VARIANTS = {"none": Variant(insert=False), "x1.5": Variant(insert=True, factor=1.5), "limit": Variant(insert=True)}
DEFAULT_ITERS = (0, 50, 100, 200)
SEED_STRIDE = 1000  # graines distinctes par classe : seed = indice de classe × SEED_STRIDE + rang du groupe


def _group_rows(group, encoded, variant, level, lut5, device, cfg, seed) -> list[dict]:
    params, active = _pack(group, encoded, device)
    targets = _targets(group, device)
    _sync(device)
    t0 = time.perf_counter()
    spec = VARIANTS[variant]
    if spec.insert:
        k_to = insertion_targets([encoded[c["id"]]["K"] for c in group], [c["area"] for c in group], cfg,
                                 spec.factor)
        params, active = insert_sources(params, active, targets, group, k_to, lut5, np.random.default_rng(seed))
    fitted = refine(params, active, targets, level, lut5)
    _sync(device)
    seconds = (time.perf_counter() - t0) / len(group)
    _, mets = vsf.evaluate(fitted, active, targets, "diff7", lut5)
    k = (active > 0).sum(1).tolist()
    return [dict(id=c["id"], bin=c["bin"], area=c["area"], variant=variant, refine_iters=level, K=int(k[b]),
                 k_enc=encoded[c["id"]]["K"], ppa=k[b] * cfg.n_par / c["area"], nrmse=mets[b]["nrmse"],
                 seconds=seconds + encoded[c["id"]]["encode_s"]) for b, c in enumerate(group)]


def _groups(cells):
    for b, (name, *_) in enumerate(vsf.BINS):
        members = sorted((c for c in cells if c["bin"] == name), key=lambda c: c["rate"].shape)
        for i in range(0, len(members), vsf.GROUP[name]):
            yield members[i:i + vsf.GROUP[name]], b * SEED_STRIDE + i


def run(cells, encoded, iters, lut5, device, cfg) -> pd.DataFrame:
    first, seed = next(_groups(cells))
    for variant in VARIANTS:  # échauffement jetable : coûts de premier appel hors mesure, pour chaque variante
        _group_rows(first[:1], encoded, variant, 1, lut5, device, cfg, seed)
    rows: list[dict] = []
    for level in iters:  # niveaux à l'extérieur : les variantes d'un niveau tournent dans le même état machine
        for variant in VARIANTS:
            for group, seed in _groups(cells):
                rows += _group_rows(group, encoded, variant, level, lut5, device, cfg, seed)
            logger.info("raffinement %d, variante %s terminé", level, variant)
    return pd.DataFrame(rows)


def check_pairing(e5: pd.DataFrame, cells) -> None:
    """E5 issu du même échantillon : pas de doublon, mêmes (aire, tag, date) par id, mêmes ids et classes."""
    check_no_duplicates(e5)
    check_e5_matches_samples(e5, {"probe": cells})
    want = {c["id"]: c["bin"] for c in cells}
    got = dict(zip(e5["id"], e5["bin"]))
    if got != want:
        raise ValueError(f"ids/classes E5 différents de l'échantillon ({len(got)} E5, {len(want)} cellules)")


def summarize(df: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
    ref = e5[["id", "bin", "nrmse", "compression_ratio", "fit_s"]].rename(
        columns={"nrmse": "e5_nrmse", "compression_ratio": "e5_ppa", "fit_s": "e5_seconds"})
    m = df.merge(ref, on=["id", "bin"], how="left", validate="m:1")
    if m["e5_nrmse"].isna().any():
        raise ValueError(f"{int(m['e5_nrmse'].isna().sum())} lignes sans cellule E5 correspondante")
    m = m.assign(delta=m.nrmse - m.e5_nrmse, nonfinite=~np.isfinite(m.nrmse))
    nonfinite = int(m.nonfinite.sum())
    if nonfinite:
        logger.warning("%d lignes à nrmse non fini (exclues des médianes, comptées dans n_nonfinite)", nonfinite)
    g = m.groupby(["bin", "variant", "refine_iters"], sort=True)
    out = g.agg(n=("id", "size"), n_nonfinite=("nonfinite", "sum"), nrmse=("nrmse", "median"),
                e5_nrmse=("e5_nrmse", "median"), paired_delta=("delta", "median"), K=("K", "median"),
                k_enc=("k_enc", "median"), ppa=("ppa", "median"), e5_ppa=("e5_ppa", "median"),
                seconds=("seconds", "median"), e5_seconds=("e5_seconds", "median")).reset_index()
    return out.assign(ppa_ratio=out.ppa / out.e5_ppa)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--e5", type=Path, nargs=2, required=True, metavar=("DOSSIER_C050", "DOSSIER_GE5000"))
    ap.add_argument("--iters", type=int, nargs="+", default=list(DEFAULT_ITERS))
    ap.add_argument("--lut-dir", type=Path, default=Path("results/03_variable_sources"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = torch.device(a.device)
    vsf.DEVICE = device
    model, cfg = load_encoder(a.ckpt, device)
    lut5 = load_or_build_logw_lut(LogWLutSpec(), a.lut_dir)["lut"][None, None].to(device)
    cells = _load_sample(a.sample)
    e5 = load_e5({"probe": tuple(a.e5)})
    check_pairing(e5, cells)
    with torch.no_grad():
        encoded = encode_cells(model, cfg, cells, device)
    logger.info("%d cellules encodées", len(cells))
    df = run(cells, encoded, a.iters, lut5, device, cfg)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "cells.csv", index=False)
    summary = summarize(df, e5)
    summary.to_csv(a.out / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
