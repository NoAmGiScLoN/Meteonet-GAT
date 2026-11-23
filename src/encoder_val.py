"""Validation de l'encodeur : nrmse par classe de taille des sources sélectionnées (inférence 0/1), K et paramètres/aire."""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

import variable_source_fit as vsf
from cell_shard_dataset import CellShardDataset, ShardIndex
from encoder_step import encode
from size_batching import PixelBudgetSampler, batch_to_device, collate
from source_encoder import SourceEncoder, select_sources

logger = logging.getLogger(__name__)

VAL_COLUMNS = ["step", "scope", "bin", "n", "nrmse_median", "k_median", "params_per_area_median"]
ALL_BIN = "all"


@torch.no_grad()
def validate(model: SourceEncoder, index: ShardIndex, split: str, px_budget: int, lut5: torch.Tensor,
             device: torch.device, step: int, kind: str = "diffusion", seed: int = 0,
             max_per_class: int | None = None) -> list[dict]:
    """Une ligne par classe (médianes) plus une ligne « all » dont nrmse_median est la moyenne des médianes de classe.

    max_per_class : sous-échantillon fixe de n cellules par classe (scope « sub »), sinon tout le split (« full »)."""
    was_training = model.training
    model.eval()
    try:
        per_bin = _collect(model, index, split, px_budget, lut5, device, kind, seed, max_per_class)
    finally:
        model.train(was_training)
    scope = "full" if max_per_class is None else "sub"
    cfg = model.cfg
    rows = [{"step": step, "scope": scope, "bin": name, "n": len(v["nrmse"]), "nrmse_median": float(np.nanmedian(v["nrmse"])),
             "k_median": float(np.median(v["k"])), "params_per_area_median": float(np.median(v["ppa"]))}
            for name, v in sorted(per_bin.items())]
    if rows:
        rows.append({"step": step, "scope": scope, "bin": ALL_BIN, "n": sum(r["n"] for r in rows),
                     "nrmse_median": float(np.mean([r["nrmse_median"] for r in rows])),
                     "k_median": float(np.mean([r["k_median"] for r in rows])),
                     "params_per_area_median": float(np.mean([r["params_per_area_median"] for r in rows]))})
    logger.info("validation %s pas %d : %s", scope, step, {r["bin"]: round(r["nrmse_median"], 4) for r in rows})
    return rows


def _collect(model, index, split, px_budget, lut5, device, kind, seed, max_per_class) -> dict[str, dict[str, list[float]]]:
    data = CellShardDataset(index)
    per_bin: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cfg = model.cfg
    for ids in PixelBudgetSampler(index, split, px_budget, seed, 0, max_per_bin=max_per_class):
        batch = batch_to_device(collate([data[i] for i in ids]), device)
        out = encode(model, batch.x)
        params, active = select_sources(out, batch.areas.tolist(), batch.crop_hw, cfg)
        t = batch.targets
        rec = vsf.render_windowed(params, active, t.Hm, t.Wm, kind, lut5)
        mets = vsf.metrics(rec, t.T, t.valid)
        for m, meta, k in zip(mets, batch.meta, active.sum(1).tolist()):
            per_bin[meta["bin"]]["nrmse"].append(m["nrmse"])
            per_bin[meta["bin"]]["k"].append(k)
            per_bin[meta["bin"]]["ppa"].append(k * cfg.n_par / meta["area"])
    return per_bin
