"""Points de reprise atomiques, états aléatoires, run_config.json et journaux CSV de l'entraînement de l'encodeur."""
from __future__ import annotations

import json
import logging
import os
import random
import uuid
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

LAST, PREV, BEST = "last.pt", "prev.pt", "best.pt"
CONFIG_FILE = "run_config.json"


def rng_state() -> dict[str, Any]:
    """États des générateurs (python, numpy, torch CPU, torch CUDA si disponible)."""
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": cuda}


def set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(out: Path, payload: Mapping[str, Any]) -> None:
    """Écrit last.pt de façon atomique ; l'ancien last.pt devient prev.pt (jamais d'état perdu en cas de crash)."""
    out = Path(out)
    tmp = out / f".{LAST}.{uuid.uuid4().hex}.new"
    try:
        _atomic_save(dict(payload), tmp)
        if (out / LAST).exists():
            os.replace(out / LAST, out / PREV)
        os.replace(tmp, out / LAST)
    finally:
        tmp.unlink(missing_ok=True)


def save_best(out: Path, payload: Mapping[str, Any]) -> None:
    _atomic_save(dict(payload), Path(out) / BEST)


def best_recorded(out: Path) -> float:
    """Meilleur nrmse de validation inscrit dans best.pt (inf s'il manque ou est illisible)."""
    path = Path(out) / BEST
    if not path.exists():
        return float("inf")
    try:
        return float(torch.load(path, map_location="cpu", weights_only=False)["best_val"])
    except Exception:
        logger.exception("best.pt illisible : %s", path)
        return float("inf")


def has_checkpoint(out: Path) -> bool:
    return any((Path(out) / n).exists() for n in (LAST, PREV))


def load_checkpoint(out: Path) -> dict[str, Any]:
    """last.pt, ou prev.pt si last.pt manque ou est illisible (crash entre deux remplacements)."""
    for name in (LAST, PREV):
        path = Path(out) / name
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:  # fichier tronqué : on tente le précédent
                logger.exception("point de reprise illisible : %s", path)
    raise FileNotFoundError(f"aucun point de reprise lisible dans {out}")


def check_run_config(out: Path, config: Mapping[str, Any], resume: bool, mutable: Collection[str]) -> None:
    """Écrit run_config.json au premier lancement ; à la reprise refuse tout changement hors `mutable`
    (le fichier est alors mis à jour avec les valeurs mutables courantes)."""
    path = Path(out) / CONFIG_FILE
    config = json.loads(json.dumps(config, default=str))
    if resume and path.exists():
        previous = json.loads(path.read_text())
        diff = {k: (previous.get(k), config.get(k)) for k in set(config) | set(previous)
                if k not in mutable and previous.get(k) != config.get(k)}
        if diff:
            raise ValueError(f"reprise refusee : run_config.json differe (ancien, nouveau) {diff}")
        if previous.get("epochs") != config.get("epochs"):
            logger.warning("epochs change à la reprise : %s -> %s (la fin de la courbe cosinus change)",
                           previous.get("epochs"), config.get("epochs"))
        if previous == config:
            return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(config, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_row(path: Path, row: Mapping[str, Any], columns: Sequence[str]) -> None:
    """Ajoute une ligne (en-tête au premier appel)."""
    frame = pd.DataFrame([{c: row[c] for c in columns}], columns=list(columns))
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def trim_csv(path: Path, last_step: int) -> None:
    """Supprime les lignes dont step > last_step (écrites après le dernier point de reprise)."""
    if not path.exists():
        return
    frame = pd.read_csv(path)
    kept = frame[frame["step"] <= last_step]
    if len(kept) != len(frame):
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        kept.to_csv(tmp, index=False)
        os.replace(tmp, path)


def ensure_columns(path: Path, columns: Sequence[str], fill: Mapping[str, Any] | None = None) -> None:
    """Réécrit (atomiquement) un CSV dont l'en-tête diffère de `columns` : colonnes manquantes ajoutées
    (valeur de `fill`, sinon NaN), colonnes inconnues supprimées, ordre de `columns`. Sans effet si identique."""
    if not path.exists() or list(pd.read_csv(path, nrows=0).columns) == list(columns):
        return
    frame = pd.read_csv(path)
    for name, value in (fill or {}).items():
        if name in columns and name not in frame.columns:
            frame[name] = value
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.reindex(columns=list(columns)).to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
