"""Insertion de sources aux pics du résidu après l'encodeur, avant raffinement (E7).

L'encodeur propose souvent moins de sources que la limite de compression (D24) ; le raffinement ne fait que déplacer
les sources existantes. On complète donc chaque cellule jusqu'à une cible (k_enc × facteur, ou la limite) en réutilisant
le placement des vagues E5 (vsf.place_sources : maxima du résidu en racine, rayon initial selon aire / K).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

import variable_source_fit as vsf
from adaptive_fit import compression_k_limit
from source_encoder import EncoderConfig

KIND = "diff7"
ROUND_DIGITS = 9  # 1.1 × 10 = 11.000000000000002 ne doit pas donner 12


def insertion_targets(k_enc: Sequence[int], areas: Sequence[int], cfg: EncoderConfig,
                      factor: float | None) -> np.ndarray:
    """Cible de sources par cellule : min(limite, ceil(facteur · k_enc)), jamais sous k_enc, au moins 1.
    factor None : remplir jusqu'à la limite min(cap, ⌊frac · aire / n_par⌋)."""
    if len(k_enc) != len(areas):
        raise ValueError(f"k_enc ({len(k_enc)}) et areas ({len(areas)}) de longueurs différentes")
    if factor is not None and not (math.isfinite(factor) and factor >= 1.0):
        raise ValueError(f"factor doit être fini et >= 1 (reçu {factor})")
    out = []
    for k, area in zip(k_enc, areas):
        limit = min(cfg.cap, compression_k_limit(int(area), cfg.n_par, cfg.compression_frac))
        wanted = limit if factor is None else min(limit, math.ceil(round(factor * int(k), ROUND_DIGITS)))
        out.append(max(int(k), wanted, 1))
    return np.asarray(out, dtype=np.int64)


def insert_sources(params: torch.Tensor, active: torch.Tensor, targets: vsf.Targets, cells: Sequence[dict],
                   k_to: np.ndarray, lut5: torch.Tensor | None,
                   rng: np.random.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Nouvelles copies (params, active) de capacité max(k_to, capacité actuelle), les k_enc premières sources
    intactes et k_enc..k_to-1 placées aux pics du résidu. Entrées non modifiées. active doit être contigu par cellule.
    Les sources placées sont créées sur vsf.DEVICE : params doit s'y trouver."""
    if params.device != vsf.DEVICE:
        raise ValueError(f"params sur {params.device}, vsf.DEVICE = {vsf.DEVICE}")
    k_from = (active > 0).sum(1).cpu().numpy().astype(np.int64)
    k_to = np.asarray(k_to, dtype=np.int64)
    if k_to.shape != k_from.shape:
        raise ValueError(f"k_to de forme {k_to.shape}, attendu {k_from.shape}")
    if (k_to < k_from).any():
        raise ValueError("k_to doit être >= au nombre de sources actives de chaque cellule")
    capacity = max(int(k_to.max(initial=0)), params.shape[1])
    new_p = params.new_zeros(params.shape[0], capacity, params.shape[2])
    new_a = active.new_zeros(active.shape[0], capacity)
    new_p[:, :params.shape[1]] = params
    new_a[:, :active.shape[1]] = active
    vsf.place_sources(new_p, new_a, targets, cells, k_from, k_to, KIND, rng, lut5)  # écrit dans les copies
    return new_p, new_a
