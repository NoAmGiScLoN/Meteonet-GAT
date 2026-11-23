"""Cellules de pluie au format brut int16 pour l'encodeur E7 : extraction par carte et shards npz sans perte.

Un shard est un dossier `<nom>.shard` de tableaux .npy NON compressés (raw int16 + masques empaquetés par
enregistrement, octets alignés) : np.load(mmap_mode="r") permet de lire un enregistrement par tranche, sans
décompresser le shard entier (accès aléatoire O(taille de la cellule)).

Le recadrage est celui de variable_source_fit.crop_cell (boîte englobante + MARGIN, borné au bord de la
carte), mais les valeurs restent en entiers bruts MeteoNet pour un stockage sans perte ; le taux (mm/h)
se recalcule à la demande avec rate_mm_h. Ce module n'importe pas torch.
"""
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from cell_delimitation import MISSING, STRUCT_8, segment

logger = logging.getLogger(__name__)

MARGIN = 3  # identique à variable_source_fit.MARGIN (vérifié par un test)
DEFAULT_MIN_AREA = 50
# (nom, aire min incluse, aire max exclue) : identique à variable_source_fit.BINS (vérifié par un test)
BINS: list[tuple[str, int, int]] = [("50-200", 50, 200), ("200-1000", 200, 1000),
                                    ("1000-5000", 1000, 5000), (">=5000", 5000, 10 ** 9)]
BIN_NAMES = tuple(name for name, _, _ in BINS)
MIN_BIN_AREA = BINS[0][1]
RAW_TO_MM_H = 12 / 100  # brut = centièmes de mm sur 5 min
SHARD_FORMAT = 2
SHARD_SUFFIX = ".shard"
META_FILE = "meta.npz"
KeepFn = Callable[[str, int], bool]


@dataclass(frozen=True, eq=False)  # eq=False : champs ndarray
class CellRecord:
    """Une cellule : métadonnées et recadrage brut (int16, zéro hors cellule)."""
    t: datetime
    local_id: int
    split: str
    bin: str
    area: int
    y0: int
    x0: int
    cy: float
    cx: float
    border_flag: bool
    missing_flag: bool
    raw: np.ndarray  # int16 (h, w) : valeurs brutes dans le masque, 0 ailleurs
    cell_mask: np.ndarray  # bool (h, w)
    invalid: np.ndarray  # bool (h, w) : pixels == -1 du recadrage


def bin_of(area: int) -> str | None:
    """Nom de la classe de taille de l'aire, None si l'aire est sous le minimum."""
    for name, lo, hi in BINS:
        if lo <= area < hi:
            return name
    return None


def rate_mm_h(rec: CellRecord) -> np.ndarray:
    """Taux (mm/h) float32 du recadrage, égal à vsf.crop_cell (calcul en int32 : pas de débordement int16)."""
    return np.where(rec.cell_mask, rec.raw.astype(np.int32) * 12 / 100, 0).astype(np.float32)


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _crop(R: np.ndarray, labels: np.ndarray, lid: int, sl: tuple[slice, slice]
          ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    y0, y1 = max(sl[0].start - MARGIN, 0), min(sl[0].stop + MARGIN, R.shape[0])
    x0, x1 = max(sl[1].start - MARGIN, 0), min(sl[1].stop + MARGIN, R.shape[1])
    window = R[y0:y1, x0:x1]
    mask = labels[y0:y1, x0:x1] == lid
    raw = np.where(mask, window, 0).astype(np.int16)
    return y0, x0, _readonly(raw), _readonly(mask), _readonly(window == MISSING)


def cell_stats(labels: np.ndarray, n: int, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Aire, centroïde et drapeaux bord / voisin manquant de chaque cellule (indice k = étiquette k+1).

    Mêmes valeurs que cell_delimitation.describe mais ~30x plus rapide (describe calcule aussi des
    statistiques d'intensité et de trous inutiles ici).
    """
    flat = labels.ravel()
    yy, xx = np.indices(labels.shape)
    area = np.bincount(flat, minlength=n + 1)[1:]
    safe = np.maximum(area, 1)
    cy = np.bincount(flat, weights=yy.ravel(), minlength=n + 1)[1:] / safe
    cx = np.bincount(flat, weights=xx.ravel(), minlength=n + 1)[1:] / safe
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    near_missing = ndi.binary_dilation(~valid, structure=STRUCT_8)

    def flag(ids: np.ndarray) -> np.ndarray:
        out = np.zeros(n + 1, dtype=bool)
        out[ids] = True
        return out[1:]

    return {"area": area, "cy": cy, "cx": cx, "border": flag(edge), "missing": flag(labels[near_missing])}


def extract_frame_cells(R: np.ndarray, t: datetime, split: str, min_area: int = DEFAULT_MIN_AREA,
                        keep: KeepFn | None = None) -> list[CellRecord]:
    """Cellules d'aire >= min_area d'une carte ; `keep(bin, local_id)` évite de recadrer les cellules écartées."""
    if min_area < MIN_BIN_AREA:
        raise ValueError(f"min_area doit être >= {MIN_BIN_AREA} (les classes commencent à {MIN_BIN_AREA} px), reçu {min_area}")
    labels, n, valid = segment(R)
    if n == 0:
        return []
    stats = cell_stats(labels, n, valid)
    objects = ndi.find_objects(labels)
    records: list[CellRecord] = []
    for k in np.flatnonzero(stats["area"] >= min_area):
        area, lid = int(stats["area"][k]), int(k) + 1
        name = bin_of(area)
        if name is None or (keep is not None and not keep(name, lid)):
            continue
        y0, x0, raw, mask, invalid = _crop(R, labels, lid, objects[lid - 1])
        records.append(CellRecord(
            t=t, local_id=lid, split=split, bin=name, area=area, y0=y0, x0=x0,
            cy=float(stats["cy"][k]), cx=float(stats["cx"][k]),
            border_flag=bool(stats["border"][k]), missing_flag=bool(stats["missing"][k]),
            raw=raw, cell_mask=mask, invalid=invalid))
    return records


# ------------------------------------------------------------ enregistrement empaqueté
@dataclass(frozen=True, eq=False)
class PackedCell:
    """CellRecord compact (masques empaquetés en bits) pour garder beaucoup de cellules en mémoire."""
    t: datetime
    local_id: int
    split: str
    bin: str
    area: int
    y0: int
    x0: int
    cy: float
    cx: float
    border_flag: bool
    missing_flag: bool
    raw: np.ndarray  # int16 (h, w)
    mask_bits: np.ndarray  # uint8, np.packbits du masque aplati
    invalid_bits: np.ndarray  # uint8

    @property
    def nbytes(self) -> int:
        return int(self.raw.nbytes + self.mask_bits.nbytes + self.invalid_bits.nbytes)


_SCALARS = ("t", "local_id", "split", "bin", "area", "y0", "x0", "cy", "cx", "border_flag", "missing_flag")


def pack_record(rec: CellRecord) -> PackedCell:
    """Version compacte d'un enregistrement (masques en bits)."""
    return PackedCell(**{k: getattr(rec, k) for k in _SCALARS}, raw=rec.raw,
                      mask_bits=np.packbits(rec.cell_mask.ravel()), invalid_bits=np.packbits(rec.invalid.ravel()))


def unpack_record(packed: PackedCell) -> CellRecord:
    """Enregistrement complet (masques booléens, lecture seule) depuis sa version compacte."""
    shape, size = packed.raw.shape, packed.raw.size

    def bits(b: np.ndarray) -> np.ndarray:
        return _readonly(np.unpackbits(b, count=size).astype(bool).reshape(shape))

    return CellRecord(**{k: getattr(packed, k) for k in _SCALARS}, raw=_readonly(packed.raw.copy()),
                      cell_mask=bits(packed.mask_bits), invalid=bits(packed.invalid_bits))


# ------------------------------------------------------------------ shards
def _to_ns(t: datetime) -> int:
    return int(np.datetime64(t, "ns").astype(np.int64))


def _from_ns(ns: int) -> datetime:
    return np.datetime64(int(ns), "ns").astype("datetime64[us]").item()


def _shard_arrays(packed: Sequence[PackedCell]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(métadonnées, gros tableaux) d'un shard ; offsets en éléments (raw) et en octets (bits)."""
    def col(name: str, dtype: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in packed], dtype=dtype)

    def offsets(sizes: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def cat(chunks: list[np.ndarray], dtype: type) -> np.ndarray:
        return np.concatenate(chunks).astype(dtype, copy=False) if chunks else np.empty(0, dtype=dtype)

    meta = {
        "format": np.array(SHARD_FORMAT), "t_ns": np.array([_to_ns(r.t) for r in packed], dtype=np.int64),
        "local_id": col("local_id", "int64"), "split": col("split", "U16"), "bin": col("bin", "U16"),
        "area": col("area", "int64"), "y0": col("y0", "int32"), "x0": col("x0", "int32"),
        "cy": col("cy", "float64"), "cx": col("cx", "float64"),
        "border_flag": col("border_flag", "bool"), "missing_flag": col("missing_flag", "bool"),
        "h": np.array([r.raw.shape[0] for r in packed], dtype=np.int32),
        "w": np.array([r.raw.shape[1] for r in packed], dtype=np.int32),
        "offsets": offsets([r.raw.size for r in packed]),
        "bit_offsets": offsets([r.mask_bits.size for r in packed]),
    }
    big = {"raw": cat([r.raw.ravel() for r in packed], np.int16),
           "mask_bits": cat([r.mask_bits for r in packed], np.uint8),
           "invalid_bits": cat([r.invalid_bits for r in packed], np.uint8)}
    return meta, big


def write_shard(path: Path, records: Sequence[CellRecord]) -> None:
    """Écrit un shard (dossier de .npy) de façon atomique : dossier temporaire voisin puis os.replace."""
    path = Path(path)
    meta, big = _shard_arrays([pack_record(r) for r in records])
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.old.tmp")
    try:
        tmp.mkdir()
        np.savez(tmp / META_FILE, **meta)
        for name, array in big.items():
            np.save(tmp / f"{name}.npy", array)
        if path.exists():
            os.replace(path, backup)
        os.replace(tmp, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


_META_KEYS = ("t_ns", "local_id", "split", "bin", "area", "y0", "x0", "cy", "cx", "border_flag",
              "missing_flag", "h", "w", "offsets", "bit_offsets")


def read_shard_meta(path: Path) -> dict[str, np.ndarray]:
    """Métadonnées par enregistrement seules (sans les pixels), pour indexer sans tout charger."""
    with np.load(Path(path) / META_FILE) as npz:
        if int(npz["format"]) != SHARD_FORMAT:
            raise ValueError(f"{path} : format de shard {int(npz['format'])} non géré")
        return {key: npz[key] for key in _META_KEYS}


def _load_mapped(path: Path) -> np.ndarray:
    """Tableau .npy en mmap lecture seule (chargement normal si le fichier est vide, non mappable)."""
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        return np.load(path)


@dataclass(frozen=True, eq=False)
class ShardHandle:
    """Shard ouvert : métadonnées en mémoire, pixels en mmap ; record(i) ne lit que la cellule i."""
    meta: dict[str, np.ndarray]
    raw: np.ndarray
    mask_bits: np.ndarray
    invalid_bits: np.ndarray

    def __len__(self) -> int:
        return len(self.meta["h"])

    def record(self, i: int) -> CellRecord:
        if not 0 <= i < len(self):
            raise IndexError(f"position {i} hors de [0, {len(self)})")
        m = self.meta
        a, b = int(m["offsets"][i]), int(m["offsets"][i + 1])
        ba, bb = int(m["bit_offsets"][i]), int(m["bit_offsets"][i + 1])
        packed = PackedCell(
            t=_from_ns(m["t_ns"][i]), local_id=int(m["local_id"][i]), split=str(m["split"][i]),
            bin=str(m["bin"][i]), area=int(m["area"][i]), y0=int(m["y0"][i]), x0=int(m["x0"][i]),
            cy=float(m["cy"][i]), cx=float(m["cx"][i]), border_flag=bool(m["border_flag"][i]),
            missing_flag=bool(m["missing_flag"][i]),
            raw=np.array(self.raw[a:b]).reshape(int(m["h"][i]), int(m["w"][i])),
            mask_bits=np.array(self.mask_bits[ba:bb]), invalid_bits=np.array(self.invalid_bits[ba:bb]))
        return unpack_record(packed)


def open_shard(path: Path) -> ShardHandle:
    """Ouvre un shard sans lire ses pixels."""
    path = Path(path)
    return ShardHandle(meta=read_shard_meta(path), raw=_load_mapped(path / "raw.npy"),
                       mask_bits=_load_mapped(path / "mask_bits.npy"),
                       invalid_bits=_load_mapped(path / "invalid_bits.npy"))


def read_shard(path: Path) -> list[CellRecord]:
    """Relit tous les enregistrements d'un shard sans perte."""
    handle = open_shard(path)
    return [handle.record(i) for i in range(len(handle))]
