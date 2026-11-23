"""Données d'entraînement : cartes de pics d'échauffement et production des lots collés (CPU, préchargeables)."""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from adaptive_fit import compression_k_limit
from cell_shard_dataset import CellShardDataset
from cell_shards import CellRecord, rate_mm_h
from encoder_loss import greedy_peak_map
from size_batching import Batch, collate
from source_encoder import EncoderConfig, head_grid_hw

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class BatchItem:
    """Lot collé (CPU), cartes de pics (échauffement) ou None, et temps de production en ms."""
    batch: Batch
    peaks: torch.Tensor | None
    data_ms: float


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Facteur de lr : montée linéaire sur warmup_steps puis cosinus jusqu'à 0 à total_steps."""
    def factor(s: int) -> float:
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = min(1.0, (s - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def peak_maps_for(records: Sequence[CellRecord], hm: int, wm: int, enc: EncoderConfig, sigma: float) -> torch.Tensor:
    """Cartes cibles (B, 1, h2, w2) des pics gloutons ; k = limite d'inférence (cap, compression)."""
    grid = head_grid_hw(hm, wm)
    maps = [greedy_peak_map(rate_mm_h(r), r.cell_mask,
                            min(enc.cap, compression_k_limit(r.area, enc.n_par, enc.compression_frac)),
                            sigma, grid, valid=~r.invalid) for r in records]
    return torch.from_numpy(np.stack(maps))[:, None]


def iter_items(data: CellShardDataset, batches: Sequence[Sequence[int]], with_peaks: bool, enc: EncoderConfig,
               sigma: float) -> Iterator[BatchItem]:
    """Lots dans l'ordre donné ; le chronomètre inclut lecture des shards, collation et cartes de pics."""
    for ids in batches:
        t0 = time.perf_counter()
        records = [data[i] for i in ids]
        batch = collate(records)
        peaks = peak_maps_for(records, batch.targets.Hm, batch.targets.Wm, enc, sigma) if with_peaks else None
        yield BatchItem(batch, peaks, 1000 * (time.perf_counter() - t0))
