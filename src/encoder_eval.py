"""Évaluation de l'encodeur de sources (phase 6 E7) : encodage, raffinement optionnel, benchmark face à E5.

Chronométrage : temps mural par cellule d'encodage (complétion et sélection incluses) et de raffinement (cibles et
Adam), avec un lot d'échauffement jetable avant mesure et une synchronisation cuda autour des zones chronométrées ; il
exclut l'extraction des recadrages (mesurée à part par crop_extraction_seconds), les statistiques résiduelles et l'E/S.

Pour chaque niveau de raffinement (nombre d'itérations Adam après l'encodeur) le benchmark écrit
<out>/refine_<n>/cells_<classe>.csv au format E5 (+ year, refine_iters, encode_s, refine_s), puis refinement_curve.csv,
adoption_table.csv et throughput.csv (voir encoder_adoption). Les ids sont les indices dans la liste d'échantillon ;
ils se recoupent entre années, d'où la colonne `year`.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle
import time
from datetime import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import variable_source_fit as vsf
from adaptive_fit import compression_k_limit
from cell_shards import extract_frame_cells
from encoder_adoption import (adoption_by_year, adoption_table, check_e5_matches_samples, load_e5, throughput_projection)
from encoder_step import encode
from profile_lut import load_or_build_logw_lut, LogWLutSpec
from residual_stats import cell_mask, compute_residual_stats
from source_encoder import EncoderConfig, SourceEncoder, pad_batch, select_sources

logger = logging.getLogger(__name__)

KIND = "diff7"
REFINE_LR = 0.05
BATCH_PX = 400_000  # pixels complétés par lot d'encodage
DEFAULT_ITERS = (0, 25, 50, 100, 200)
THRESHOLD = 0.5
P90 = 0.9
DEFAULT_LUT_KEY = "ace17c5890"
CROP_FRAME_SIDE = 512
CROP_BLOCK = 16  # côté (px) des carrés synthétiques ; chacun donne une cellule de 256 px
CURVE_COLUMNS = ["bin", "refine_iters", "median_nrmse", "p90_nrmse", "median_K", "median_params_per_area",
                 "median_seconds"]


def load_encoder(ckpt_path: Path, device: torch.device, cfg: EncoderConfig | None = None) -> tuple[SourceEncoder, EncoderConfig]:
    """Reconstruit l'encodeur depuis best.pt (qui contient sa config) ou last.pt (config à fournir) ; mode eval."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg or ck.get("encoder")
    if cfg is None:
        raise ValueError(f"{ckpt_path} ne contient pas de clé 'encoder' : fournir cfg (cas de last.pt)")
    model = SourceEncoder(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batches(cells: Sequence[dict], batch_px: int, multiple: int) -> list[list[dict]]:
    """Lots de cellules de tailles voisines dont le tenseur complété ne dépasse pas batch_px (au moins une cellule)."""
    pad = lambda n: -(-n // multiple) * multiple
    ordered = sorted(cells, key=lambda c: (pad(c["rate"].shape[0]), pad(c["rate"].shape[1])))
    batches: list[list[dict]] = []
    for c in ordered:
        h, w = c["rate"].shape
        if batches:
            hp = max(pad(h), *(pad(o["rate"].shape[0]) for o in batches[-1]))
            wp = max(pad(w), *(pad(o["rate"].shape[1]) for o in batches[-1]))
            if hp * wp * (len(batches[-1]) + 1) <= batch_px:
                batches[-1].append(c)
                continue
        batches.append([c])
    return batches


@torch.no_grad()
def encode_cells(model: SourceEncoder, cfg: EncoderConfig, cells: Sequence[dict], device: torch.device,
                 threshold: float = THRESHOLD, batch_px: int = BATCH_PX) -> dict[int, dict]:
    """{id: dict(params (K, 8) float32 brut, K, encode_s)} ; encode_s = temps mural par cellule (lot / taille du lot,
    complétion et select_sources inclus)."""
    results: dict[int, dict] = {}
    for batch in _batches(cells, batch_px, cfg.pad_multiple):
        _sync(device)
        t0 = time.perf_counter()
        crops = [(c["rate"], cell_mask(c["rate"]), np.ones(c["rate"].shape, dtype=bool)) for c in batch]
        x, _ = pad_batch(crops, cfg.pad_multiple)
        out = encode(model, x.to(device))
        params, active = select_sources(out, [c["area"] for c in batch], [c["rate"].shape for c in batch], cfg,
                                        threshold)
        params, active = params.cpu().numpy(), active.cpu().numpy() > 0
        _sync(device)
        seconds = (time.perf_counter() - t0) / len(batch)
        for b, c in enumerate(batch):
            kept = params[b][active[b]].astype(np.float32)
            results[c["id"]] = dict(params=kept, K=len(kept), encode_s=seconds)
    return results


def refine(params: torch.Tensor, active: torch.Tensor, targets: vsf.Targets, iters: int,
           lut5: torch.Tensor | None, lr: float = REFINE_LR) -> torch.Tensor:
    """Raffinement Adam du fitter (diff7) ; iters == 0 : copie identique, sans appel au fitter."""
    if iters <= 0:
        return params.detach().clone()
    return vsf.optimise(params, active, targets, KIND, lut5, iters, lr=lr, lr_schedule=None)


def _targets(cells: Sequence[dict], device: torch.device) -> vsf.Targets:
    t = vsf.make_targets(cells)
    return dataclasses.replace(t, T=t.T.to(device), sT=t.sT.to(device), valid=t.valid.to(device),
                               npix=t.npix.to(device))


def _pack(cells: Sequence[dict], encoded: Mapping[int, dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kmax = max(1, max(encoded[c["id"]]["K"] for c in cells))
    params = torch.zeros(len(cells), kmax, 8)
    active = torch.zeros(len(cells), kmax)
    for b, c in enumerate(cells):
        e = encoded[c["id"]]
        params[b, :e["K"]] = torch.from_numpy(e["params"])
        active[b, :e["K"]] = 1
    return params.to(device), active.to(device)


def _rows(group, encoded, params, active, targets, lut5, level, refine_s, year, cfg) -> list[dict]:
    rec, mets = vsf.evaluate(params, active, targets, KIND, lut5)
    rows = []
    for b, c in enumerate(group):
        h, w = c["rate"].shape
        k, enc_s = encoded[c["id"]]["K"], encoded[c["id"]]["encode_s"]
        n_params = k * vsf.ARM_NPAR[KIND]
        stats = compute_residual_stats(rec[b, :h, :w].cpu().numpy(), c["rate"]).as_dict()
        limit = min(cfg.cap, compression_k_limit(c["area"], cfg.n_par, cfg.compression_frac))
        rows.append(dict(id=c["id"], tag=c["tag"], date=c["date"], bin=c["bin"], area=c["area"], K=k,
                         k_target_last=k, stop_reason="encoder", waves=0, **mets[b], n_params=n_params,
                         compression_ratio=n_params / c["area"], fit_s=enc_s + refine_s, n_pruned=0,
                         k_limit=limit, **stats, year=year, refine_iters=level, encode_s=enc_s, refine_s=refine_s))
    return rows


def _refine_level(cells, encoded, level, lut5, device, year, cfg) -> list[dict]:
    rows: list[dict] = []
    for name, *_ in vsf.BINS:
        members = sorted((c for c in cells if c["bin"] == name), key=lambda c: c["rate"].shape)
        for i in range(0, len(members), vsf.GROUP[name]):
            group = members[i:i + vsf.GROUP[name]]
            params, active = _pack(group, encoded, device)
            _sync(device)
            t0 = time.perf_counter()
            targets = _targets(group, device)
            fitted = refine(params, active, targets, level, lut5)
            _sync(device)
            seconds = (time.perf_counter() - t0) / len(group) if level > 0 else 0.0
            rows += _rows(group, encoded, fitted, active, targets, lut5, level, seconds, year, cfg)
    return rows


def refinement_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Courbe par (classe, niveau) : nrmse médian et p90, K médian, params/aire médian, secondes médianes par cellule."""
    g = df.assign(seconds=df["encode_s"] + df["refine_s"]).groupby(["bin", "refine_iters"], sort=True)
    out = g.agg(median_nrmse=("nrmse", "median"), p90_nrmse=("nrmse", lambda s: s.quantile(P90)),
                median_K=("K", "median"), median_params_per_area=("compression_ratio", "median"),
                median_seconds=("seconds", "median")).reset_index()
    return out[CURVE_COLUMNS]


def _load_sample(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        return [{**c, "id": i} for i, c in enumerate(pickle.load(fh))]


def check_lut_key(spec: LogWLutSpec, allow_override: bool) -> None:
    """La table de profils doit être celle de l'entraînement (clé par défaut) sauf dérogation explicite."""
    if spec.key() != DEFAULT_LUT_KEY and not allow_override:
        raise ValueError(f"clé de table {spec.key()} != {DEFAULT_LUT_KEY} (utiliser --allow-lut-override)")


def crop_extraction_seconds(repeats: int = 3) -> float:
    """Coût mesuré (s) d'extraction d'un recadrage par cell_shards.extract_frame_cells sur une carte synthétique."""
    frame = np.zeros((CROP_FRAME_SIDE, CROP_FRAME_SIDE), np.int16)
    for i in range(0, CROP_FRAME_SIDE - CROP_BLOCK, 2 * CROP_BLOCK):
        for j in range(0, CROP_FRAME_SIDE - CROP_BLOCK, 2 * CROP_BLOCK):
            frame[i:i + CROP_BLOCK, j:j + CROP_BLOCK] = 40 + (i + j) % 20
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        records = extract_frame_cells(frame, datetime(2016, 1, 1), "train")
        best = min(best, (time.perf_counter() - t0) / max(1, len(records)))
    return best


def _warm_up(model, cfg, cells, lut5, device, threshold, with_refine: bool) -> None:
    """Un lot d'encodage et un groupe de raffinement jetables, synchronisés (kernels/allocations hors mesure)."""
    first = _batches(cells, BATCH_PX, cfg.pad_multiple)[0]
    encoded = encode_cells(model, cfg, first, device, threshold)
    if with_refine:
        group = first[:1]
        params, active = _pack(group, encoded, device)
        refine(params, active, _targets(group, device), 1, lut5)
    _sync(device)


def benchmark(model: SourceEncoder, cfg: EncoderConfig, sample_path: Mapping[str, Path],
              e5_dirs: Mapping[str, tuple[Path, Path]], iters_list: Sequence[int], device: torch.device, out_dir: Path,
              lut5: torch.Tensor | None = None, threshold: float = THRESHOLD,
              crop_s: float = 0.0) -> pd.DataFrame:
    """Encode (une fois) puis raffine à chaque niveau ; écrit les CSV dans out_dir et retourne toutes les lignes.
    Si e5_dirs est non vide, écrit aussi adoption_table.csv et throughput.csv."""
    out_dir = Path(out_dir)
    absent = sorted(set(map(str, e5_dirs)) - set(map(str, sample_path)))
    if absent:
        raise ValueError(f"--e5 : année(s) {absent} absente(s) de --samples")
    if lut5 is not None:
        vsf.check_lut_depth(KIND, lut5)
        lut5 = lut5.to(device)
    samples = {str(y): _load_sample(Path(p)) for y, p in sample_path.items()}
    e5 = load_e5(e5_dirs) if e5_dirs else None
    if e5 is not None:
        check_e5_matches_samples(e5, samples)
    rows: list[dict] = []
    for year, cells in samples.items():
        logger.info("année %s : %d cellules", year, len(cells))
        _warm_up(model, cfg, cells, lut5, device, threshold, any(i > 0 for i in iters_list))
        encoded = encode_cells(model, cfg, cells, device, threshold)
        for level in iters_list:
            rows += _refine_level(cells, encoded, level, lut5, device, year, cfg)
            logger.info("année %s, raffinement %d terminé", year, level)
    df = pd.DataFrame(rows)
    for (level, name), part in df.groupby(["refine_iters", "bin"]):
        folder = out_dir / f"refine_{level}"
        folder.mkdir(parents=True, exist_ok=True)
        part.to_csv(folder / f"cells_{vsf.bin_slug(name)}.csv", index=False)
    refinement_curve(df).to_csv(out_dir / "refinement_curve.csv", index=False)
    throughput_projection(df, device.type, crop_s=crop_s, strict=False).to_csv(out_dir / "throughput.csv", index=False)
    if e5 is not None:
        adoption_by_year(df, e5).to_csv(out_dir / "adoption_by_year.csv", index=False)
        adoption_table(df, e5).to_csv(out_dir / "adoption_table.csv", index=False)
    return df


def _year_pairs(items: Sequence[str]) -> dict[str, str]:
    pairs = dict(item.split("=", 1) for item in items)
    if len(pairs) != len(items):
        raise ValueError("années dupliquées")
    return pairs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help="best.pt (ou last.pt avec run_config.json voisin)")
    ap.add_argument("--samples", nargs="+", required=True, metavar="ANNEE=PKL")
    ap.add_argument("--e5", nargs="*", default=[], metavar="ANNEE=DOSSIER_C050,DOSSIER_GE5000")
    ap.add_argument("--iters", nargs="+", type=int, default=list(DEFAULT_ITERS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-lut-override", action="store_true", help="accepter une table différente de l'entraînement")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--lut-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "03_variable_sources")
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    a = build_parser().parse_args(argv)
    device = torch.device(a.device)
    model, cfg = load_encoder(a.ckpt, device)
    samples = {y: Path(p) for y, p in _year_pairs(a.samples).items()}
    e5 = {y: tuple(Path(d) for d in v.split(",")) for y, v in _year_pairs(a.e5).items()}
    absent = sorted(set(e5) - set(samples))
    if absent:
        raise SystemExit(f"--e5 : année(s) {absent} absente(s) de --samples")
    spec = LogWLutSpec()
    check_lut_key(spec, a.allow_lut_override)
    lut5 = load_or_build_logw_lut(spec, a.lut_dir)["lut"][None, None]
    benchmark(model, cfg, samples, e5, a.iters, device, a.out, lut5=lut5, threshold=a.threshold,
              crop_s=crop_extraction_seconds())


if __name__ == "__main__":
    main()
