"""État du tracker v2 : configuration validée, pistes (Track) et état global, tous immuables.

Représentation des pixels d'une cellule : deux tableaux d'indices (ys, xs) int32 en lecture seule.
Choix pour les grandes cellules : une cellule de 10^5 px coûte ~0,8 Mo, négligeable face à la
carte (quelques Mo) et seules les pistes actives sont conservées ; en contrepartie l'advection
et le bincount opèrent directement sur les coordonnées, sans réexpansion d'un masque bbox
(un masque bool serait plus petit pour une cellule compacte, mais très creux pour une
cellule étirée en diagonale, et obligerait à np.nonzero à chaque pas).
"""
import math
from dataclasses import dataclass, replace
from datetime import datetime
from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np

PIXEL_DTYPE = np.int32
INT_FIELDS = ("min_area_start", "min_area_keep", "max_coast", "max_gap_min", "young_frames",
              "coarse")
FLOAT_FIELDS = ("d_max", "w_pos", "max_cost", "f_min", "alpha", "min_peak_ratio")


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class TrackerConfig:
    """Paramètres du tracker v2 (validés à la construction)."""
    min_area_start: int = 50      # aire minimale pour créer une piste
    min_area_keep: int = 35       # aire minimale pour continuer une piste (hystérésis)
    max_coast: int = 1            # cartes consécutives manquées tolérées
    max_gap_min: int = 15         # écart temporel maximal (min) avant coupure
    d_max: float = 25.0           # distance de porte (px)
    w_pos: float = 0.5            # poids de la position dans le coût
    max_cost: float = 0.75        # coût maximal d'un lien accepté
    f_min: float = 0.1            # fraction de recouvrement minimale (fusion / scission)
    alpha: float = 0.5            # lissage exponentiel de la vitesse
    young_frames: int = 2         # en dessous : vitesse issue du champ de mouvement
    coarse: int = 4               # facteur de grossissement de la corrélation de phase
    min_peak_ratio: float = 2.0   # rapport de pic minimal pour un mouvement fiable

    def __post_init__(self) -> None:
        for name in INT_FIELDS:
            if not _is_int(getattr(self, name)):
                raise ValueError(f"{name} doit être un entier (reçu {getattr(self, name)!r})")
        for name in FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} doit être un réel fini (reçu {value!r})")
        if not 0 < self.min_area_keep <= self.min_area_start:
            raise ValueError("Il faut 0 < min_area_keep <= min_area_start "
                             f"(reçu {self.min_area_keep}, {self.min_area_start})")
        if self.max_coast < 0:
            raise ValueError(f"max_coast doit être >= 0 (reçu {self.max_coast})")
        if self.max_gap_min < 1:
            raise ValueError(f"max_gap_min doit être >= 1 (reçu {self.max_gap_min})")
        if not self.d_max > 0:
            raise ValueError(f"d_max doit être > 0 (reçu {self.d_max})")
        for name in ("w_pos", "max_cost", "f_min", "alpha"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} doit être dans ]0, 1] (reçu {getattr(self, name)})")
        if self.young_frames < 1:
            raise ValueError(f"young_frames doit être >= 1 (reçu {self.young_frames})")
        if self.coarse < 1:
            raise ValueError(f"coarse doit être >= 1 (reçu {self.coarse})")
        if not self.min_peak_ratio > 0:
            raise ValueError(f"min_peak_ratio doit être > 0 (reçu {self.min_peak_ratio})")


def _pixels(values: np.ndarray, name: str) -> np.ndarray:
    """Copie int32 en lecture seule (isole la piste du tableau de l'appelant) ; entiers >= 0."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} doit être 1-D (forme {raw.shape})")
    is_float = raw.dtype.kind == "f"
    integral = raw.dtype.kind in "iu" or (is_float and np.array_equal(raw, np.round(raw)))
    if not integral:
        raise ValueError(f"{name} doit contenir des entiers")
    if raw.size and raw.min() < 0:
        raise ValueError(f"{name} doit être >= 0")
    out = raw.astype(PIXEL_DTYPE)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)  # eq=False : champs ndarray
class Track:
    """Une piste : dernière observation de la cellule, vitesse lissée par pas et compteurs."""
    track_id: int
    local_id: int
    cy: float
    cx: float
    v: tuple[float, float]   # (vy, vx) px par pas, vy > 0 vers le bas
    area: int
    mass: float              # intensity_sum
    ys: np.ndarray           # lignes des pixels du masque (int32, lecture seule)
    xs: np.ndarray           # colonnes des pixels du masque (int32, lecture seule)
    age: int                 # cartes observées
    coasted: int             # cartes consécutives manquées
    truncated: bool
    gap_steps: int = 0       # pas depuis la dernière observation
    border_flag: bool = False   # dernière observation au contact du bord de la grille
    missing_flag: bool = False  # dernière observation au contact de pixels manquants

    def __post_init__(self) -> None:
        ys, xs = _pixels(self.ys, "ys"), _pixels(self.xs, "xs")
        if ys.shape != xs.shape:
            raise ValueError(f"ys et xs de longueurs différentes : {ys.shape} vs {xs.shape}")
        if not _is_int(self.area) or self.area <= 0 or self.area != ys.size:
            raise ValueError(f"area doit être un entier > 0 égal à ys.size "
                             f"(reçu {self.area}, {ys.size})")
        if not _is_int(self.age) or self.age < 0:
            raise ValueError(f"age doit être un entier >= 0 (reçu {self.age!r})")
        if not _is_int(self.coasted) or self.coasted < 0:
            raise ValueError(f"coasted doit être un entier >= 0 (reçu {self.coasted!r})")
        if not _is_int(self.gap_steps) or self.gap_steps < 0:
            raise ValueError(f"gap_steps doit être un entier >= 0 (reçu {self.gap_steps!r})")
        for name in ("border_flag", "missing_flag", "truncated"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} doit être un booléen (reçu {getattr(self, name)!r})")
        if len(self.v) != 2 or not all(math.isfinite(c) for c in self.v):
            raise ValueError(f"v doit être (vy, vx) fini (reçu {self.v!r})")
        if not (math.isfinite(self.cy) and math.isfinite(self.cx)):
            raise ValueError(f"Centre non fini : ({self.cy}, {self.cx})")
        if not math.isfinite(self.mass) or self.mass < 0:
            raise ValueError(f"mass doit être fini et >= 0 (reçu {self.mass})")
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "xs", xs)


@dataclass(frozen=True)
class TrackerState:
    """Pistes actives, prochain identifiant et instant de la dernière carte traitée."""
    tracks: tuple[Track, ...] = ()
    next_id: int = 1
    t: datetime | None = None

    def allocate_id(self) -> tuple[int, "TrackerState"]:
        """Retourne (nouvel identifiant, nouvel état au compteur avancé) ; l'original est inchangé."""
        return self.next_id, replace(self, next_id=self.next_id + 1)

    def with_tracks(self, tracks: Iterable[Track]) -> "TrackerState":
        """Nouvel état dont les pistes sont `tracks` (en tuple) ; l'original est inchangé."""
        return replace(self, tracks=tuple(tracks))

    def with_time(self, t: datetime | None) -> "TrackerState":
        """Nouvel état daté de `t` ; l'original est inchangé."""
        return replace(self, t=t)
