"""Raffinement par groupes de taille réglable (E7, débit).

Le benchmark raffine par groupes E5 (vsf.GROUP : 40 / 20 / 10 / 4 cellules), trop petits pour occuper le GPU du
Jetson : le temps par itération y est dominé par les lancements de noyaux. La perte est une somme sur des cellules
indépendantes (Adam agit par paramètre), donc la taille du groupe ne change pas le résultat d'une cellule ; ce module
mesure le temps par cellule et la mémoire de pointe selon la taille du groupe.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

import variable_source_fit as vsf
from encoder_eval import KIND, _pack, _sync, _targets, refine


@dataclass(frozen=True)
class GroupedRun:
    nrmse: dict[int, float]  # id -> nrmse après raffinement
    seconds_per_cell: float  # temps mural (cibles + Adam), synchronisé, divisé par le nombre de cellules
    n_groups: int
    peak_mem_mb: float  # pointe d'allocation cuda (0 hors cuda)


def make_groups(cells: Sequence[dict], group_size: int) -> list[list[dict]]:
    """Cellules triées par forme de recadrage puis découpées en groupes de group_size (le dernier peut être plus court)."""
    if group_size < 1:
        raise ValueError(f"group_size doit être >= 1 (reçu {group_size})")
    ordered = sorted(cells, key=lambda c: c["rate"].shape)
    return [ordered[i:i + group_size] for i in range(0, len(ordered), group_size)]


def replicate(cells: Sequence[dict], n: int) -> list[dict]:
    """n cellules prises en boucle dans cells, copies avec id = 0..n-1 et source_id = id d'origine."""
    if not cells:
        raise ValueError("replicate : liste de cellules vide")
    return [{**cells[i % len(cells)], "id": i, "source_id": cells[i % len(cells)]["id"]} for i in range(n)]


def refine_grouped(cells: Sequence[dict], encoded: Mapping[int, dict], group_size: int, iters: int,
                   lut5: torch.Tensor | None, device: torch.device) -> GroupedRun:
    """Raffine toutes les cellules par groupes de group_size ; encoded est indexé par id de cellule."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    nrmse: dict[int, float] = {}
    total = 0.0
    groups = make_groups(cells, group_size)
    for group in groups:
        params, active = _pack(group, encoded, device)
        _sync(device)
        t0 = time.perf_counter()
        targets = _targets(group, device)
        fitted = refine(params, active, targets, iters, lut5)
        _sync(device)
        total += time.perf_counter() - t0
        _, mets = vsf.evaluate(fitted, active, targets, KIND, lut5)
        nrmse.update({c["id"]: float(m["nrmse"]) for c, m in zip(group, mets)})
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0
    return GroupedRun(nrmse=nrmse, seconds_per_cell=total / max(1, len(cells)), n_groups=len(groups),
                      peak_mem_mb=peak)
