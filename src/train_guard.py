"""Garde-fous de l'entraînement (phase 5 E7) : compteur de pas écartés, détection de pics de gradient, instantané.

Un pas écarté laisse les poids inchangés. Coût mémoire de l'instantané : poids + 2 moments d'AdamW en fp32,
soit ~1,1 M paramètres x 3 x 4 octets ~ 13 Mo (copie en mémoire, pas sur disque).
"""
from __future__ import annotations

import copy
import statistics
from collections import deque
from dataclasses import dataclass

import torch

SNAPSHOT_EVERY = 100  # pas entre deux instantanés du dernier état sain


@dataclass
class SkipCounter:
    """Pas écartés : cumul du run, série en cours et cumul de l'époque (état mutable de la boucle)."""
    total: int = 0
    consecutive: int = 0
    epoch: int = 0

    def hit(self) -> None:
        self.total, self.consecutive, self.epoch = self.total + 1, self.consecutive + 1, self.epoch + 1

    def ok(self) -> None:
        self.consecutive = 0


def is_spike(norm: float, history: deque, factor: float) -> bool:
    """Vrai si la norme dépasse factor x médiane de l'historique, une fois l'historique plein."""
    return len(history) == history.maxlen and norm > factor * statistics.median(history)


class Snapshot:
    """Copie en mémoire de modèle + optimiseur au dernier état sain, pour annuler un pas qui a produit des poids non finis."""

    def __init__(self) -> None:
        self._model: dict | None = None
        self._opt: dict | None = None

    def take(self, model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
        self._model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._opt = copy.deepcopy(opt.state_dict())

    def restore(self, model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
        if self._model is None:
            raise RuntimeError("aucun instantané à restaurer")
        model.load_state_dict(self._model)
        opt.load_state_dict(copy.deepcopy(self._opt))
