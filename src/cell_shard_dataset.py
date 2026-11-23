"""Index des shards de cellules et jeu de données paresseux (sans torch : __len__ / __getitem__)."""
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cell_shards import SHARD_SUFFIX, CellRecord, ShardHandle, open_shard, read_shard_meta

logger = logging.getLogger(__name__)

SHARD_GLOB = f"*{SHARD_SUFFIX}"
INDEX_COLUMNS = ["shard", "position", "split", "bin", "area", "h", "w"]


@dataclass(frozen=True)
class IndexEntry:
    """Position d'un enregistrement : shard, rang dans le shard, et colonnes utiles aux échantillonneurs."""
    shard: str
    position: int
    split: str
    bin: str
    area: int
    h: int
    w: int


@dataclass(frozen=True)
class ShardIndex:
    """Liste ordonnée des enregistrements d'un dossier de shards (ordre : nom de shard, position)."""
    directory: Path
    entries: tuple[IndexEntry, ...]

    @classmethod
    def from_dir(cls, directory: Path) -> "ShardIndex":
        """Lit seulement les métadonnées de chaque shard du dossier."""
        entries: list[IndexEntry] = []
        for path in sorted(Path(directory).glob(SHARD_GLOB)):
            meta = read_shard_meta(path)
            for i in range(len(meta["h"])):
                entries.append(IndexEntry(path.name, i, str(meta["split"][i]), str(meta["bin"][i]),
                                          int(meta["area"][i]), int(meta["h"][i]), int(meta["w"][i])))
        return cls(Path(directory), tuple(entries))

    def __len__(self) -> int:
        return len(self.entries)

    def filter(self, split: str | None = None, bin: str | None = None) -> "ShardIndex":  # noqa: A002
        """Sous-index restreint à un split et/ou une classe."""
        kept = tuple(e for e in self.entries if (split is None or e.split == split) and (bin is None or e.bin == bin))
        return ShardIndex(self.directory, kept)

    def to_frame(self) -> pd.DataFrame:
        """Tableau (shard, position, split, bin, area, h, w)."""
        return pd.DataFrame([[getattr(e, c) for c in INDEX_COLUMNS] for e in self.entries], columns=INDEX_COLUMNS)


class CellShardDataset:
    """Accès par indice à un CellRecord ; un petit cache LRU garde les derniers shards ouverts.

Les shards sont des tableaux .npy non compressés ouverts en mmap : un accès aléatoire ne lit que la
cellule demandée (pas de décompression d'un shard entier), donc aucun échantillonneur groupé n'est requis.
"""

    def __init__(self, index: ShardIndex, cache_size: int = 2) -> None:
        if cache_size < 1:
            raise ValueError("cache_size doit être >= 1")
        self._index = index
        self._cache_size = cache_size
        self._cache: OrderedDict[str, ShardHandle] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _shard(self, name: str) -> ShardHandle:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        handle = open_shard(self._index.directory / name)
        self._cache[name] = handle
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return handle

    def __getitem__(self, i: int) -> CellRecord:
        if not -len(self) <= i < len(self):
            raise IndexError(f"indice {i} hors de [0, {len(self)})")
        entry = self._index.entries[i]
        return self._shard(entry.shard).record(entry.position)
