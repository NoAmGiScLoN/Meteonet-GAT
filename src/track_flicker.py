"""Métrique de « flicker » : pistes mortes juste après un coast qui auraient été retrouvées
avec un coast plus long (aide à décider de coaster 2 cartes)."""
import numpy as np
import pandas as pd

from tracker_state import TrackerConfig
from track_tables import STEP, RunTables, SIZE_LABELS, size_class

MIN_AREA = TrackerConfig().min_area_keep   # 35 px : aire minimale d'une cellule candidate
START_EVENTS = ("birth", "entry_edge", "entry_missing")
COLUMNS = ["track_id", "t_death", "max_area", "size_class", "steps", "n_candidates",
           "recovered"]


def _dead_after_coast(tables: RunTables) -> pd.DataFrame:
    """Pistes dont la mort suit un coast : position et vitesse de la dernière observation."""
    ev, det = tables.events, tables.detections
    deaths = ev.loc[ev.event == "death", ["track_id", "t"]]
    coasts = ev.loc[ev.event == "coast", ["track_id", "t"]].assign(after=1)
    deaths = deaths.assign(t_prev=deaths.t - STEP).merge(
        coasts.rename(columns={"t": "t_prev"}), on=["track_id", "t_prev"])
    tracked = det[det.track_id > 0].sort_values("timestamp")
    last = tracked.groupby("track_id").agg(
        t_last=("timestamp", "last"), cy=("cy", "last"), cx=("cx", "last"),
        v_y=("v_y", "last"), v_x=("v_x", "last"), max_area=("area_px", "max")).reset_index()
    dead = deaths.merge(last, on="track_id")
    dead["steps"] = ((dead.t - dead.t_last) / STEP).round().astype(int)
    dead["py"] = dead.cy + dead.steps * dead.v_y
    dead["px"] = dead.cx + dead.steps * dead.v_x
    return dead


def _candidates(tables: RunTables) -> pd.DataFrame:
    """Détections >= 35 px non suivies ou nées à leur carte (clé : cid)."""
    ev, det = tables.events, tables.detections
    births = ev.loc[ev.event.isin(START_EVENTS), ["track_id", "t"]].assign(born=True)
    cand = det[det.area_px >= MIN_AREA].merge(
        births, left_on=["track_id", "timestamp"], right_on=["track_id", "t"], how="left")
    cand = cand[(cand.track_id == 0) | cand.born.notna()]
    return cand[["timestamp", "cy", "cx"]].rename(columns={"timestamp": "t"}).reset_index(
        drop=True).rename_axis("cid").reset_index()


def _match(pairs: pd.DataFrame) -> set:
    """Appariement glouton par distance croissante : une candidate pour une seule piste.
    Une paire acceptée est la meilleure de sa piste et de sa candidate (équivaut au glouton)."""
    accepted = []
    pairs = pairs.sort_values("dist")
    while len(pairs):
        best_dead = pairs.drop_duplicates("track_id")
        best = best_dead.drop_duplicates("cid")
        accepted.append(best[["track_id", "cid"]])
        pairs = pairs[~pairs.track_id.isin(best.track_id) & ~pairs.cid.isin(best.cid)]
    return set(pd.concat(accepted).track_id) if accepted else set()


def flicker_table(tables: RunTables, d_max: float | None = None) -> pd.DataFrame:
    """Une ligne par piste morte juste après un coast : retrouvée (recovered) ou non.

    Approximation : depuis la dernière observation, `steps` = (mort - dernière obs.) / 5 min
    (2 pour max_coast = 1) ; la position prédite est dernier centroïde + v x steps. La piste est
    dite retrouvée si, à la carte de sa mort, une détection d'au moins 35 px, non suivie ou née
    à cette carte (cellules de 35-49 px incluses), se trouve à moins de d_max (25 px) de cette
    position. Une candidate ne retrouve qu'une piste (glouton par distance). Ni association
    complète ni recouvrement ; la vitesse est celle de la dernière détection."""
    d_max = TrackerConfig().d_max if d_max is None else d_max
    dead = _dead_after_coast(tables)
    if not len(dead):
        return pd.DataFrame(columns=COLUMNS)
    pairs = dead[["track_id", "t", "py", "px"]].merge(_candidates(tables), on="t")
    pairs["dist"] = np.hypot(pairs.cy - pairs.py, pairs.cx - pairs.px)
    pairs = pairs[pairs.dist <= d_max]
    n_cand = pairs.groupby("track_id").size()
    dead["n_candidates"] = dead.track_id.map(n_cand).fillna(0).astype(int)
    dead["recovered"] = dead.track_id.isin(_match(pairs))
    dead["size_class"] = size_class(dead.max_area).astype(str)
    return dead.rename(columns={"t": "t_death"})[COLUMNS].reset_index(drop=True)


def flicker_rates(table: pd.DataFrame) -> dict:
    """Taux de retrouvailles (overall + par classe de taille) ; NaN si aucune piste."""
    def rate(sub: pd.DataFrame) -> float:
        return float(sub.recovered.mean()) if len(sub) else float("nan")
    return {"overall": rate(table), "n": int(len(table)),
            "by_class": {c: rate(table[table.size_class == c]) for c in SIZE_LABELS}}
