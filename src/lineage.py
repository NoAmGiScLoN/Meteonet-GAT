"""Généalogie du tracker v2 : liens de recouvrement, fusions / scissions, événements de bord.

Règle d'identité (validée) : dans une fusion, le parent de plus grande MASSE DE PLUIE garde
l'identifiant ; dans une scission, c'est l'enfant de plus grande masse. Égalité : plus grand
recouvrement, puis identifiant le plus bas. Chaque lien est consigné dans la table de lignée.

Règle plusieurs <-> plusieurs (déterministe, sans effet d'ordre des entrées) :
  1. chaque cellule choisit son parent principal parmi les pistes qui la recouvrent
     (masse de la piste, puis recouvrement, puis identifiant de piste le plus bas) ;
  2. chaque piste ne garde que son enfant principal parmi les cellules qui l'ont choisie
     (masse de la cellule, puis recouvrement, puis local_id le plus bas) et lui transmet son id ;
  3. les autres cellules démarrent une piste neuve (split_start, parent = leur parent principal) ;
  4. les pistes qui n'ont été choisies par aucune cellule se terminent (merge_end).
Les cas 1-1, fusion (n pistes -> 1 cellule) et scission (1 piste -> n cellules) en sont des
cas particuliers. Tous les liens du graphe de recouvrement sont consignés.

Hystérésis : l'éligibilité à la continuation (aire >= min_area_keep) est à la charge de
l'appelant lorsqu'il construit `cells` ; celle à la création (aire >= min_area_start) passe
par `eligible_new`. Partition : chaque piste est continuée, terminée ou non appariée ; chaque
cellule est continuée, nouvelle piste ou non appariée (l'appelant décide coast / mort /
naissance / entrée pour les non appariées).
"""
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NamedTuple, get_args

import numpy as np
import pandas as pd

from tracker_state import Track, TrackerConfig

EventType = Literal["birth", "death", "entry_edge", "exit_edge", "entry_missing", "exit_missing",
                    "merge_end", "split_start", "coast", "gap_break"]
LinkType = Literal["continuation", "merge", "split"]
EVENT_TYPES: tuple[str, ...] = get_args(EventType)
LINK_TYPES: tuple[str, ...] = get_args(LinkType)
EXIT_FRAC_OUT = 0.5   # fraction prédite hors grille à partir de laquelle la fin est une sortie


class Candidate(NamedTuple):
    """Paire (piste, cellule) recouverte : indice de piste, local_id, pixels communs, fractions."""
    track_index: int
    local_id: int
    inter: int
    frac_parent: float   # inter / aire piste
    frac_child: float    # inter / aire cellule


@dataclass(frozen=True)
class Link:
    """Ligne de la table de lignée : parent -> cellule enfant.

    share_of_child  : masse du parent / somme des masses de tous les parents liés à cet enfant.
    share_of_parent : masse de l'enfant / somme des masses de tous les enfants liés à ce parent.
    Les premières somment à 1 par enfant, les secondes à 1 par parent, pour tout graphe.
    """
    parent_track: int
    child_local_id: int
    link_type: LinkType
    overlap_frac_parent: float
    overlap_frac_child: float
    share_of_child: float
    share_of_parent: float
    is_main: bool

    def __post_init__(self) -> None:
        if self.link_type not in LINK_TYPES:
            raise ValueError(f"link_type inconnu : {self.link_type!r}")
        for name in ("overlap_frac_parent", "overlap_frac_child", "share_of_child",
                     "share_of_parent"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} doit être dans [0, 1] (reçu {value!r})")


@dataclass(frozen=True)
class GroupResult:
    """Décision d'identité : partition des pistes et des cellules, plus la lignée."""
    continuations: Mapping[int, int]                          # {track_id: local_id}
    ended: Mapping[int, EventType]                            # {track_id: merge_end}
    new_children: tuple[tuple[int, int | None, EventType], ...]  # (local_id, parent, event)
    links: tuple[Link, ...]                                   # triés (track_id, local_id)
    unmatched_tracks: tuple[int, ...] = ()                    # track_id ni continuées ni finies
    unmatched_cells: tuple[int, ...] = ()                     # local_id ni continuées ni nouvelles

    def __post_init__(self) -> None:
        object.__setattr__(self, "continuations", MappingProxyType(dict(self.continuations)))
        object.__setattr__(self, "ended", MappingProxyType(dict(self.ended)))
        for name in ("new_children", "links", "unmatched_tracks", "unmatched_cells"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def overlap_links(overlaps: np.ndarray, tracks: Sequence[Track], cells: pd.DataFrame,
                  cfg: TrackerConfig) -> list[Candidate]:
    """Liens (piste, cellule) : inter / aire piste >= f_min ET inter / aire cellule >= f_min.

    Le recouvrement doit donc valoir au moins f_min de l'objet le plus grand : un petit
    fragment qui touche une grande cellule n'est pas lié.

    `overlaps` : sortie de association.overlap_counts (colonne = local_id, colonne 0 = fond).
    Retourne les Candidate triés par (indice de piste, local_id).
    """
    overlaps = np.asarray(overlaps)
    if overlaps.ndim != 2 or overlaps.shape[0] != len(tracks):
        raise ValueError(f"overlaps doit être (n_pistes={len(tracks)}, n_labels) "
                         f"(forme {overlaps.shape})")
    lids = cells.local_id.to_numpy(dtype=int)
    if lids.size == 0 or not tracks:
        return []
    if lids.min() < 1 or lids.max() >= overlaps.shape[1]:
        raise ValueError(f"local_id hors de la matrice ({overlaps.shape[1]} colonnes) : "
                         f"{lids.min()}..{lids.max()}")
    inter = overlaps[:, lids]
    frac_t = inter / np.array([t.area for t in tracks], dtype=float)[:, None]
    frac_c = inter / cells.area_px.to_numpy(dtype=float)[None, :]
    keep = (inter > 0) & ((frac_t >= cfg.f_min) & (frac_c >= cfg.f_min))
    rows, cols = np.nonzero(keep)
    return sorted(Candidate(int(r), int(lids[c]), int(inter[r, c]), float(frac_t[r, c]),
                            float(frac_c[r, c])) for r, c in zip(rows, cols))


def _validate(candidates: Sequence[Candidate], tracks: Sequence[Track], cells: pd.DataFrame,
              pairs: Sequence[tuple[int, int]]) -> None:
    if len({t.track_id for t in tracks}) != len(tracks):
        raise ValueError("track_id dupliqués")
    for t in tracks:
        if not (math.isfinite(t.mass) and t.mass >= 0 and t.area > 0):
            raise ValueError(f"piste {t.track_id} : masse / aire invalides")
    lids = [int(v) for v in cells.local_id]
    if len(set(lids)) != len(lids):
        raise ValueError("local_id dupliqués")
    for row in cells.itertuples():
        mass, area = float(row.intensity_sum), float(row.area_px)
        if not (math.isfinite(mass) and mass >= 0):
            raise ValueError(f"cellule {row.local_id} : intensity_sum doit être fini >= 0")
        if not (math.isfinite(area) and area > 0):
            raise ValueError(f"cellule {row.local_id} : area_px doit être > 0")
    seen: set[tuple[int, int]] = set()
    for t, lid, inter, fp, fc in candidates:
        if not 0 <= t < len(tracks) or lid not in lids:
            raise ValueError(f"candidat hors bornes : piste {t}, local_id {lid}")
        if (t, lid) in seen:
            raise ValueError(f"candidat dupliqué : ({t}, {lid})")
        seen.add((t, lid))
        if inter <= 0 or not (0.0 <= fp <= 1.0 and 0.0 <= fc <= 1.0):
            raise ValueError(f"candidat incohérent : {(t, lid, inter, fp, fc)}")
    if len({p[0] for p in pairs}) != len(pairs) or len({p[1] for p in pairs}) != len(pairs):
        raise ValueError("paires hongroises dupliquées")
    for t, c in pairs:
        if not (0 <= t < len(tracks) and 0 <= c < len(cells)):
            raise ValueError(f"paire hongroise hors bornes : ({t}, {c})")


def _share(value: float, group: Sequence[float]) -> float:
    total = float(sum(group))
    return value / total if total > 0 else 1.0 / len(group)


def classify_groups(candidates: Sequence[Candidate], tracks: Sequence[Track],
                    cells: pd.DataFrame, hungarian_pairs: Sequence[tuple[int, int]],
                    eligible_new: Collection[int] | None = None) -> GroupResult:
    """Décide continuations, fusions et scissions (règle du module) et construit la lignée.

    `hungarian_pairs` : (indice de piste, position de la cellule dans `cells`), comme assign().
    Une paire hongroise dont la piste ET la cellule n'ont aucun recouvrement retenu reste une
    continuation (appariement par distance) ; sinon le recouvrement décide et la piste ou la
    cellule sans lien reste non appariée.
    `eligible_new` : local_id autorisés à DÉMARRER une piste (aire >= min_area_start, décidé par
    l'appelant) ; None = tous. Un enfant non principal non éligible n'est pas démarré (non
    apparié) mais son lien reste consigné.
    """
    _validate(candidates, tracks, cells, hungarian_pairs)
    cands = sorted(Candidate(*c) for c in candidates)
    eligible = None if eligible_new is None else frozenset(eligible_new)
    cell_mass = {int(r.local_id): float(r.intensity_sum) for r in cells.itertuples()}
    inter = {(c.track_index, c.local_id): c.inter for c in cands}
    parents: dict[int, list[int]] = defaultdict(list)
    children: dict[int, list[int]] = defaultdict(list)
    for c in cands:
        parents[c.local_id].append(c.track_index)
        children[c.track_index].append(c.local_id)
    # 1. parent principal de chaque cellule ; 2. enfant principal de chaque piste
    chosen = {lid: min(ts, key=lambda t: (-tracks[t].mass, -inter[t, lid], tracks[t].track_id))
              for lid, ts in parents.items()}
    chosen_by: dict[int, list[int]] = defaultdict(list)
    for lid, t in chosen.items():
        chosen_by[t].append(lid)
    kept = {t: min(mine, key=lambda lid: (-cell_mass[lid], -inter[t, lid], lid))
            for t, mine in chosen_by.items()}
    continuations = {tracks[t].track_id: lid for t, lid in kept.items()}
    ended: dict[int, EventType] = {tracks[t].track_id: "merge_end"
                                   for t in children if t not in kept}
    new_children: list[tuple[int, int | None, EventType]] = []
    unmatched_cells = {lid for lid in cell_mass if lid not in parents}
    for lid, p in chosen.items():
        if kept[p] == lid:
            continue
        if eligible is None or lid in eligible:
            new_children.append((lid, tracks[p].track_id, "split_start"))
        else:
            unmatched_cells.add(lid)
    for t, c in hungarian_pairs:
        lid = int(cells.local_id.iloc[c])
        if t not in children and lid not in parents:
            continuations[tracks[t].track_id] = lid
            unmatched_cells.discard(lid)
    links = []
    for c in cands:
        t, lid = c.track_index, c.local_id
        share_c = _share(tracks[t].mass, [tracks[p].mass for p in parents[lid]])
        share_p = _share(cell_mass[lid], [cell_mass[k] for k in children[t]])
        kind: LinkType = ("merge" if len(parents[lid]) > 1
                          else "split" if len(children[t]) > 1 else "continuation")
        links.append(Link(tracks[t].track_id, lid, kind, c.frac_parent, c.frac_child,
                          share_c, share_p, kept.get(t) == lid))
    unmatched_tracks = sorted(tr.track_id for tr in tracks if tr.track_id not in continuations
                              and tr.track_id not in ended)
    return GroupResult(continuations, ended, tuple(sorted(new_children)),
                       tuple(sorted(links, key=lambda lk: (lk.parent_track, lk.child_local_id))),
                       tuple(unmatched_tracks), tuple(sorted(unmatched_cells)))


def _flag(flags: Mapping[str, object] | pd.Series, name: str) -> bool:
    if name not in flags:
        raise ValueError(f"drapeau manquant : {name}")
    return bool(flags[name])


def is_truncated(cell_row: Mapping[str, object] | pd.Series) -> bool:
    """Cellule tronquée : au contact du bord de la grille ou de pixels manquants."""
    return _flag(cell_row, "border_flag") or _flag(cell_row, "missing_neighbor_flag")


def edge_event(flags: Mapping[str, object] | pd.Series, kind: Literal["start", "end"],
               predicted_out_frac: float = 0.0) -> EventType:
    """Événement de début ("start") ou de fin ("end") d'une piste ; bord > manquant.

    Fin : bord ou fraction prédite hors grille >= 0,5 -> exit_edge ; manquant -> exit_missing ;
    sinon death. Début : bord -> entry_edge ; manquant -> entry_missing ; sinon birth.
    """
    if kind not in ("start", "end"):
        raise ValueError(f"kind doit être 'start' ou 'end' (reçu {kind!r})")
    if not (math.isfinite(predicted_out_frac) and 0.0 <= predicted_out_frac <= 1.0):
        raise ValueError(f"predicted_out_frac doit être dans [0, 1] (reçu {predicted_out_frac})")
    border, missing = _flag(flags, "border_flag"), _flag(flags, "missing_neighbor_flag")
    if kind == "start":
        return "entry_edge" if border else "entry_missing" if missing else "birth"
    if border or predicted_out_frac >= EXIT_FRAC_OUT:
        return "exit_edge"
    return "exit_missing" if missing else "death"
