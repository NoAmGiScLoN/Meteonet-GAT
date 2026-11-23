"""Extraction d'un jeu de cellules (segmentation 1 mm/h, 8-connexité, aire >= MIN_AREA) pour le benchmark
de compression. Chaque cellule -> patch 64x64 (champ de taux en mm/h masqué par la cellule, rééchantillonné
sur un carré centré) + vecteur de moments géométriques calculé sur la cellule d'origine.

Usage : python cell_dataset.py   (écrit results/02_fixed_vector_benchmark/cells.npz)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from cell_delimitation import DATA_DIR, MISSING, STRUCT_8, THRESHOLD_RAW, segment

OUT = Path(__file__).resolve().parent.parent / "results" / "02_fixed_vector_benchmark"
MIN_AREA = 50
PATCH = 64
FRAME_STEP = 12  # une carte par heure
FILES = [  # (fichier mensuel, étiquette)
    ("NW_rainfall_2016/rainfall-NW-2016-01/rainfall_NW_2016_01.1.npz", "2016-01"),
    ("NW_rainfall_2016/rainfall-NW-2016-07/rainfall_NW_2016_07.1.npz", "2016-07"),
]
MOMENT_NAMES = ["log_area", "mean_rate", "max_rate", "log_sum_rate", "log_elong", "cos2th", "sin2th",
                "fill_ratio", "hole_frac", "border"]


def raw_to_rate(R):
    """Brut MeteoNet (int16) -> mm/h ; int32 car R * 12 déborde int16 au-delà de 2730."""
    return R.astype(np.int32) * 12 / 100


def to_patch(field, mask):
    """Recadre sur la boîte englobante, complète en carré (centré), rééchantillonne en PATCH x PATCH."""
    h, w = field.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq[oy:oy + h, ox:ox + w] = field
    t = torch.from_numpy(sq)[None, None]
    mode = dict(mode="area") if s >= PATCH else dict(mode="bilinear", align_corners=False)
    return F.interpolate(t, size=(PATCH, PATCH), **mode)[0, 0].numpy()


def geometric_moments(rate, mask, border):
    ys, xs = np.nonzero(mask)
    area = mask.sum()
    cov = np.cov(np.stack([xs, ys]), bias=True) if area > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    l2, l1 = np.maximum(evals, 1e-6)
    th = np.arctan2(evecs[1, 1], evecs[0, 1])
    filled = ndi.binary_fill_holes(mask).sum()
    r = rate[mask]
    h, w = mask.shape
    return [np.log(area), r.mean(), r.max(), np.log(r.sum()), 0.5 * np.log(l1 / l2), np.cos(2 * th),
            np.sin(2 * th), area / (h * w), (filled - area) / filled, float(border)]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    patches, moments, meta = [], [], []
    for rel, tag in FILES:
        with np.load(DATA_DIR / rel, allow_pickle=True) as f:
            dates, data = f["dates"], f["data"]
        for i in range(0, len(dates), FRAME_STEP):
            R = data[i]
            labels, n, valid = segment(R, STRUCT_8, THRESHOLD_RAW)
            if n == 0:
                continue
            area = np.bincount(labels.ravel(), minlength=n + 1)
            near_missing = ndi.binary_dilation(~valid, structure=STRUCT_8)
            for lid, sl in enumerate(ndi.find_objects(labels), start=1):
                if area[lid] < MIN_AREA:
                    continue
                mask = labels[sl] == lid
                rate = np.where(mask, raw_to_rate(R[sl]), 0).astype(np.float32)  # mm/h
                border = (sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == R.shape[0]
                          or sl[1].stop == R.shape[1] or near_missing[sl][mask].any())
                patches.append(to_patch(rate, mask))
                moments.append(geometric_moments(rate, mask, border))
                meta.append((tag, dates[i].day, str(dates[i]), int(area[lid])))
        print(tag, "cellules cumulées :", len(patches), flush=True)
        del data
    meta = np.array(meta, dtype=object)
    np.savez_compressed(OUT / "cells.npz", rate=np.stack(patches).astype(np.float32),
                        moments=np.array(moments, np.float32), tag=meta[:, 0].astype(str),
                        day=meta[:, 1].astype(int), date=meta[:, 2].astype(str), area=meta[:, 3].astype(int),
                        moment_names=np.array(MOMENT_NAMES))
    print("écrit", OUT / "cells.npz")


if __name__ == "__main__":
    build()
