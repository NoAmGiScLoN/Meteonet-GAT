"""Découpage train/val/test par jours pour l'encodeur E7, sans fuite vers les cellules du benchmark E5.

L'unité d'affectation est un bloc de jours consécutifs (3 par défaut), pour qu'un orage à cheval sur
minuit ne soit pas coupé entre train et test. Les blocs sont définis depuis une époque fixe.

Affectation par quotas stratifiés : la strate d'un bloc est (année, saison météo) de son premier jour ;
dans chaque strate, les blocs sont triés par sha256(sel, indice) (reproductible entre machines), les
round(val_frac·n) premiers vont en val, les round(test_frac·n) suivants en test, le reste en train
(au moins 1 bloc val et 1 test si n >= 3). Les quotas portent sur les blocs, y compris ceux qui
contiennent des jours exclus ; l'exclusion du benchmark, par jour, prime ensuite sur le split du bloc.

Les dates sont des horodatages radar naïfs (MeteoNet, UTC) ; les jours des vignettes d'entraînement
doivent suivre la même convention. Les dates avec fuseau sont refusées.
"""
import math
from functools import lru_cache
import hashlib
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

SPLITS = ("train", "val", "test")
Split = Literal["train", "val", "test", "excluded"]
_E5_SOURCES = ("adaptive_2016", "adaptive_2017")
_SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
_HASH_BYTES = 8
_EPOCH = date(2016, 1, 1)  # origine fixe des blocs (division entière valable avant l'époque)


@dataclass(frozen=True)
class SplitConfig:
    """Fractions de jours val/test (le reste va en train), sel de hachage, marge d'exclusion."""
    val_frac: float = 0.1
    test_frac: float = 0.1
    salt: str = "e7"
    exclude_margin_days: int = 1
    block_days: int = 3
    start: date = date(2016, 1, 1)
    end: date = date(2018, 12, 31)

    def __post_init__(self) -> None:
        for name in ("val_frac", "test_frac"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0 < value < 1):
                raise ValueError(f"{name} doit être fini et dans ]0, 1[ (reçu {value})")
        if self.val_frac + self.test_frac >= 1:
            raise ValueError("val_frac + test_frac doit être < 1")
        if self.block_days < 1:
            raise ValueError("block_days doit être >= 1")
        if self.end < self.start:
            raise ValueError("end doit être >= start")
        if self.exclude_margin_days < 0:
            raise ValueError("exclude_margin_days doit être >= 0")


def _unit_hash(salt: str, block: int) -> float:
    """Valeur déterministe dans [0, 1) issue de (sel, indice de bloc)."""
    digest = hashlib.sha256(f"{salt}|{block}".encode()).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "big") / 2 ** (8 * _HASH_BYTES)


def _quota(frac: float, n: int) -> int:
    return max(1, round(frac * n)) if n >= 3 else round(frac * n)


@lru_cache(maxsize=None)
def _block_splits(cfg: SplitConfig) -> dict[int, str]:
    """Table indice de bloc -> split (train/val/test), calculée une fois par configuration."""
    first = (cfg.start - _EPOCH).days // cfg.block_days
    last = (cfg.end - _EPOCH).days // cfg.block_days
    strata: dict[tuple[int, str], list[int]] = {}
    for idx in range(first, last + 1):
        day0 = _EPOCH + timedelta(days=idx * cfg.block_days)
        strata.setdefault((day0.year, _SEASONS[day0.month]), []).append(idx)
    table: dict[int, str] = {}
    for blocks in strata.values():
        ordered = sorted(blocks, key=lambda i: _unit_hash(cfg.salt, i))
        n_val, n_test = _quota(cfg.val_frac, len(ordered)), _quota(cfg.test_frac, len(ordered))
        for rank, idx in enumerate(ordered):
            table[idx] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
    return table


def split_of(day: date, cfg: SplitConfig, excluded_days: frozenset[date]) -> Split:
    """Sous-ensemble du jour ; 'excluded' s'il appartient à excluded_days ; ValueError hors [start, end]."""
    if not cfg.start <= day <= cfg.end:
        raise ValueError(f"jour {day} hors de la plage [{cfg.start}, {cfg.end}]")
    if day in excluded_days:
        return "excluded"
    return _block_splits(cfg)[(day - _EPOCH).days // cfg.block_days]  # type: ignore[return-value]


def _days_of_pickle(path: Path) -> set[date]:
    # pickle.load : acceptable, fichiers d'archive locaux de confiance
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} : un dict {{id: dict}} est attendu, reçu {type(data).__name__}")
    days: set[date] = set()
    for key, src in data.items():
        if not isinstance(src, dict) or "date" not in src:
            raise ValueError(f"{path} : la source {key!r} n'est pas un dict avec un champ 'date'")
        try:
            stamp = datetime.fromisoformat(str(src["date"]))
        except ValueError as exc:
            raise ValueError(f"{path} : date invalide {src['date']!r} (source {key!r})") from exc
        if stamp.tzinfo is not None:
            raise ValueError(f"{path} : date avec fuseau refusée {src['date']!r} (horodatages naïfs attendus)")
        days.add(stamp.date())
    return days


def benchmark_days(paths: Sequence[Path]) -> frozenset[date]:
    """Jours couverts par les sources.pkl du benchmark (jamais vide : sinon ValueError)."""
    if not paths:
        raise ValueError("aucun fichier de benchmark fourni : l'exclusion serait vide")
    days: set[date] = set()
    for path in paths:
        days |= _days_of_pickle(Path(path))
    if not days:
        raise ValueError("aucune date trouvée dans le benchmark : l'exclusion serait vide")
    return frozenset(days)


def excluded_days(benchmark: frozenset[date], cfg: SplitConfig) -> frozenset[date]:
    """Jours du benchmark élargis de ± cfg.exclude_margin_days (priment sur le split du bloc)."""
    if not benchmark:
        raise ValueError("benchmark vide : l'exclusion serait vide")
    margin = cfg.exclude_margin_days
    return frozenset(d + timedelta(days=k) for d in benchmark for k in range(-margin, margin + 1))


def default_benchmark_paths(root: Path) -> list[Path]:
    """Chemins des sources.pkl E5 (2016 et 2017) sous la racine du dépôt."""
    return [root / "archive" / "E5_adaptive_k" / "results" / name / "sources.pkl" for name in _E5_SOURCES]


def split_table(days: Iterable[date], cfg: SplitConfig, excluded: frozenset[date]) -> pd.DataFrame:
    """Tableau (day, split, month, season, year) pour le rapport."""
    rows = [(d, split_of(d, cfg, excluded), d.month, _SEASONS[d.month], d.year) for d in days]
    return pd.DataFrame(rows, columns=["day", "split", "month", "season", "year"])
