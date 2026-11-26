"""Petits shards de cellules synthétiques (écrits comme l'extracteur : shards + index.csv) pour les tests d'entraînement."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import cell_shards as cs
from cell_shard_dataset import ShardIndex

T0 = datetime(2016, 2, 3)
SIDES = (9, 12, 20)  # aires 81, 144, 400 : classes 50-200 et 200-1000


def make_records(n: int, split: str, sides: tuple[int, ...] = SIDES, missing: bool = True) -> list[cs.CellRecord]:
    """n cellules carrées (taux croissant en pente) ; un pixel manquant voisin de la cellule si `missing`."""
    out = []
    for i in range(n):
        side = sides[i % len(sides)]
        R = np.zeros((60, 60), np.int16)
        yy, xx = np.mgrid[0:side, 0:side]
        R[5:5 + side, 5:5 + side] = (30 + 5 * ((xx + yy) % 7) + 2 * i).astype(np.int16)
        if missing:
            R[5 + side + 1, 5] = cs.MISSING
        recs = cs.extract_frame_cells(R, T0 + timedelta(minutes=5 * i), split)
        out.append(max(recs, key=lambda r: r.area))
    return out


def write_tiny_dir(directory: Path, n_train: int = 6, n_val: int = 3, sides: tuple[int, ...] = SIDES) -> Path:
    """Shards train/val et index.csv (colonnes de l'extracteur)."""
    directory.mkdir(parents=True, exist_ok=True)
    cs.write_shard(directory / "train_0000.shard", make_records(n_train, "train", sides))
    cs.write_shard(directory / "val_0000.shard", make_records(n_val, "val", sides))
    ShardIndex.from_dir(directory).to_frame().to_csv(directory / "index.csv", index=False)
    return directory
