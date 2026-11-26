"""Lots à budget de pixels par classe de taille et collation des recadrages (phase 5 E7).

Un lot regroupe des cellules d'une même classe de taille, complétées au maximum du lot (multiple de 32) ;
la somme des pixels complétés reste sous `px_budget` (une cellule seule plus grande forme son propre lot).
Les cibles sont celles du fitter : taux en mm/h sur le cadre du recadrage, `valid` = pixels du recadrage
non manquants (les pixels de cellule hors masque sont des cibles de taux nul).
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cell_shard_dataset import INDEX_COLUMNS, IndexEntry, ShardIndex
from cell_shards import CellRecord, rate_mm_h
from source_encoder import pad_batch
from variable_source_fit import Targets

logger = logging.getLogger(__name__)

PAD_MULTIPLE = 32
SORT_WINDOW = 64  # cellules triées par taille dans une fenêtre aléatoire : moins de complétion perdue
SUBSAMPLE_SALT = 7919
INDEX_FILE = "index.csv"


def _ceil_to(n: int, m: int = PAD_MULTIPLE) -> int:
    return -(-n // m) * m


def padded_pixels(h: int, w: int, multiple: int = PAD_MULTIPLE) -> int:
    """Pixels d'un recadrage h x w complété à des multiples de `multiple`."""
    return _ceil_to(h, multiple) * _ceil_to(w, multiple)


def load_index(directory: Path) -> ShardIndex:
    """Index de <dir>/index.csv (écrit par l'extracteur) s'il existe, sinon lecture des métadonnées des shards."""
    path = Path(directory) / INDEX_FILE
    if not path.exists():
        return ShardIndex.from_dir(directory)
    frame = pd.read_csv(path)
    missing = [c for c in INDEX_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} : colonnes manquantes {missing}")
    entries = tuple(IndexEntry(str(r.shard), int(r.position), str(r.split), str(r.bin), int(r.area), int(r.h), int(r.w))
                    for r in frame.itertuples(index=False))
    return ShardIndex(Path(directory), entries)


class PixelBudgetSampler:
    """Itère des listes d'indices (positions dans `index`) : chaque cellule du split une fois par époque."""

    def __init__(self, index: ShardIndex, split: str, px_budget: int, seed: int, epoch: int = 0,
                 bins: Sequence[str] | None = None, max_per_bin: int | None = None) -> None:
        """max_per_bin : sous-échantillon fixe (indépendant de l'époque, fonction de seed) de n cellules par classe."""
        if px_budget <= 0 or (max_per_bin is not None and max_per_bin < 1):
            raise ValueError("px_budget et max_per_bin doivent être > 0")
        self._entries = index.entries
        ids = [i for i, e in enumerate(index.entries) if e.split == split and (bins is None or e.bin in bins)]
        self._ids = tuple(ids if max_per_bin is None else self._subsample(ids, max_per_bin, seed))
        self.px_budget, self.seed, self.epoch = px_budget, seed, epoch
        self._cache: list[list[int]] | None = None

    def _subsample(self, ids: Sequence[int], n: int, seed: int) -> list[int]:
        rng = np.random.default_rng([seed, SUBSAMPLE_SALT])
        by_bin: dict[str, list[int]] = {}
        for i in ids:
            by_bin.setdefault(self._entries[i].bin, []).append(i)
        kept: list[int] = []
        for name in sorted(by_bin):
            chosen = rng.permutation(len(by_bin[name]))[:n]
            kept += sorted(by_bin[name][j] for j in chosen)
        return sorted(kept)

    def _pack(self, ids: Sequence[int]) -> list[list[int]]:
        """Remplissage glouton : un lot est fermé quand l'ajout dépasserait le budget (taille complétée au max)."""
        batches: list[list[int]] = []
        current: list[int] = []
        hm = wm = 0
        for i in ids:
            e = self._entries[i]
            nh, nw = max(hm, e.h), max(wm, e.w)
            if current and (len(current) + 1) * padded_pixels(nh, nw) > self.px_budget:
                batches.append(current)
                current, nh, nw = [], e.h, e.w
            current.append(i)
            hm, wm = nh, nw
        if current:
            batches.append(current)
        return batches

    def batches(self) -> list[list[int]]:
        """Tous les lots de l'époque (déterministes pour (seed, epoch))."""
        if self._cache is None:
            rng = np.random.default_rng([self.seed, self.epoch])
            by_bin: dict[str, list[int]] = {}
            for i in self._ids:
                by_bin.setdefault(self._entries[i].bin, []).append(i)
            out: list[list[int]] = []
            for name in sorted(by_bin):
                ids = [by_bin[name][j] for j in rng.permutation(len(by_bin[name]))]
                ordered: list[int] = []
                for s in range(0, len(ids), SORT_WINDOW):
                    window = ids[s:s + SORT_WINDOW]
                    ordered += sorted(window, key=lambda i: (self._entries[i].h, self._entries[i].w))
                out += self._pack(ordered)
            self._cache = [out[j] for j in rng.permutation(len(out))]
        return [list(b) for b in self._cache]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())


@dataclass(frozen=True, eq=False)
class Batch:
    """Lot collé : entrées (B, 3, Hp, Wp), cibles du fitter (Hm x Wm = taille max des recadrages), aires, tailles."""
    x: torch.Tensor
    targets: Targets
    areas: torch.Tensor
    crop_hw: tuple[tuple[int, int], ...]
    meta: tuple[dict, ...]


def collate(records: Sequence[CellRecord]) -> Batch:
    """Lot depuis des CellRecord ; ne modifie pas les enregistrements. valid = pixels du recadrage non manquants."""
    if not records:
        raise ValueError("lot vide")
    rates = [rate_mm_h(r) for r in records]
    valids = [~np.asarray(r.invalid, dtype=bool) for r in records]
    x, _ = pad_batch([(rt, r.cell_mask, v) for rt, r, v in zip(rates, records, valids)], PAD_MULTIPLE)
    hm, wm = max(r.shape[0] for r in rates), max(r.shape[1] for r in rates)
    T = torch.zeros(len(records), hm, wm)
    valid = torch.zeros(len(records), hm, wm, dtype=torch.bool)
    for b, (rt, v) in enumerate(zip(rates, valids)):
        T[b, :rt.shape[0], :rt.shape[1]] = torch.from_numpy(rt)
        valid[b, :rt.shape[0], :rt.shape[1]] = torch.from_numpy(v)
    targets = Targets(T=T, sT=T.sqrt(), valid=valid, npix=valid.sum((1, 2)).clamp(min=1), Hm=hm, Wm=wm)
    meta = tuple({"bin": r.bin, "area": r.area, "t": r.t, "local_id": r.local_id} for r in records)
    return Batch(x=x, targets=targets, areas=torch.tensor([float(r.area) for r in records]),
                 crop_hw=tuple((int(r.raw.shape[0]), int(r.raw.shape[1])) for r in records), meta=meta)


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Copie du lot dont les tenseurs sont sur `device`."""
    t = batch.targets
    targets = dataclasses.replace(t, T=t.T.to(device), sT=t.sT.to(device), valid=t.valid.to(device),
                                  npix=t.npix.to(device))
    return dataclasses.replace(batch, x=batch.x.to(device), targets=targets, areas=batch.areas.to(device))
