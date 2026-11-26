"""Sélection déterministe des cellules d'entraînement : bottom-k par hachage et pas de trames par classe.

Pour chaque (split, classe de taille), on garde les `quota` cellules de plus petit sha256(sel, t ISO,
local_id) avec un tas borné : le résultat est exact et indépendant de l'ordre du flux.
"""
import hashlib
import heapq
from collections.abc import Iterator, Mapping
from datetime import datetime
from functools import lru_cache

from cell_shards import BIN_NAMES, CellRecord, PackedCell, pack_record, unpack_record

STEP_MIN = 5
VAL_DIVISOR = 16  # val = 1/16 du train par classe
HASH_BYTES = 16
DEFAULT_TRAIN_QUOTAS = {"50-200": 80000, "200-1000": 60000, "1000-5000": 40000, ">=5000": 20000}
DEFAULT_FRAME_STEPS = {"50-200": 6, "200-1000": 6, "1000-5000": 3, ">=5000": 2}  # en cartes de 5 min
Quotas = dict[tuple[str, str], int]


def default_quotas(train: Mapping[str, int] | None = None) -> Quotas:
    """Quotas par (split, classe) : train donné, val = train // 16, test = 0 (extrait plus tard)."""
    train = dict(DEFAULT_TRAIN_QUOTAS if train is None else train)
    quotas: Quotas = {}
    for name in BIN_NAMES:
        quotas[("train", name)] = train[name]
        quotas[("val", name)] = train[name] // VAL_DIVISOR
        quotas[("test", name)] = 0
    return quotas


@lru_cache(maxsize=4096)  # would_accept puis offer demandent la même clé : un seul hachage par cellule
def cell_key(salt: str, t: datetime, local_id: int) -> int:
    """Clé de hachage entière d'une cellule (indépendante du split et de l'ordre du flux)."""
    digest = hashlib.sha256(f"{salt}|{t.isoformat()}|{local_id}".encode()).digest()
    return int.from_bytes(digest[:HASH_BYTES], "big")


def frame_index(t: datetime, origin: datetime, step_min: int = STEP_MIN) -> int:
    """Indice de la carte sur la grille de 5 min depuis `origin` (calculé sur t : les trous ne le décalent pas)."""
    seconds = (t - origin).total_seconds()
    if seconds % (step_min * 60) != 0:
        raise ValueError(f"{t} n'est pas sur la grille de {step_min} min depuis {origin}")
    return int(seconds // (step_min * 60))


def bins_due(idx: int, steps: Mapping[str, int]) -> frozenset[str]:
    """Classes dont le pas de trames divise l'indice : la carte n'est segmentée que si l'ensemble est non vide."""
    return frozenset(name for name, step in steps.items() if idx % step == 0)


Heap = list[tuple[int, str, int, PackedCell]]


class CellSelector:
    """Garde, par (split, classe), les `quota` cellules de plus petite clé (tas max borné). État mutable.

    Les cellules sont gardées empaquetées (masques en bits) ; `nbytes` suit la mémoire occupée.
    """

    def __init__(self, quotas: Mapping[tuple[str, str], int], salt: str) -> None:
        self._quotas = dict(quotas)
        self._salt = salt
        # tas de (-clé, t ISO, local_id, cellule) : (clé, t, id) est unique, jamais de comparaison de cellules
        self._heaps: dict[tuple[str, str], Heap] = {}
        self._nbytes = 0

    @classmethod
    def from_state(cls, quotas: Mapping[tuple[str, str], int], salt: str, state: Mapping) -> "CellSelector":
        """Reconstruit un sélecteur depuis `state()` (reprise après interruption)."""
        selector = cls(quotas, salt)
        selector._heaps = {key: list(heap) for key, heap in state.items()}
        selector._nbytes = sum(item[3].nbytes for heap in selector._heaps.values() for item in heap)
        return selector

    def state(self) -> dict[tuple[str, str], Heap]:
        """Copie de l'état (tas de cellules empaquetées), sérialisable avec pickle."""
        return {key: list(heap) for key, heap in self._heaps.items()}

    def nbytes(self) -> int:
        """Mémoire des cellules retenues (pixels et masques empaquetés)."""
        return self._nbytes

    def would_accept(self, split: str, bin_name: str, t: datetime, local_id: int) -> bool:
        """Vrai si la cellule entrerait dans la sélection courante (sans la construire)."""
        quota = self._quotas.get((split, bin_name), 0)
        if quota <= 0:
            return False
        heap = self._heaps.get((split, bin_name), [])
        return len(heap) < quota or cell_key(self._salt, t, local_id) < -heap[0][0]

    def offer(self, rec: CellRecord) -> bool:
        """Propose une cellule ; vrai si elle est retenue (éventuellement en évinçant la plus grande clé)."""
        if not self.would_accept(rec.split, rec.bin, rec.t, rec.local_id):
            return False
        heap = self._heaps.setdefault((rec.split, rec.bin), [])
        packed = pack_record(rec)
        item = (-cell_key(self._salt, rec.t, rec.local_id), rec.t.isoformat(), rec.local_id, packed)
        if len(heap) < self._quotas[(rec.split, rec.bin)]:
            heapq.heappush(heap, item)
        else:
            self._nbytes -= heapq.heapreplace(heap, item)[3].nbytes
        self._nbytes += packed.nbytes
        return True

    def iter_selected(self, split: str, bin_name: str) -> Iterator[CellRecord]:
        """Cellules retenues, dépaquetées une à une, triées par (t, local_id)."""
        items = self._heaps.get((split, bin_name), [])
        for it in sorted(items, key=lambda it: (it[1], it[2])):
            yield unpack_record(it[3])

    def selected(self, split: str, bin_name: str) -> list[CellRecord]:
        """Cellules retenues (dépaquetées), triées par (t, local_id)."""
        return list(self.iter_selected(split, bin_name))

    def counts(self) -> dict[tuple[str, str], int]:
        """Nombre de cellules retenues par (split, classe) non vide."""
        return {key: len(heap) for key, heap in self._heaps.items() if heap}
