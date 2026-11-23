"""Pertes de l'encodeur de sources (phase 4 E7) : reconstruction, parcimonie, binarisation, échauffement par cartes de pics."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from variable_source_fit import EPS, Targets

logger = logging.getLogger(__name__)

HEAD_STRIDE = 2
RHO_MIN, RHO_MAX, RHO_SCALE = 2.0, 40.0, 0.7  # mêmes constantes que vsf.place_sources
FOCAL_ALPHA, FOCAL_BETA = 2.0, 4.0  # focale de type CornerNet


def recon_loss(rec: torch.Tensor, targets: Targets) -> torch.Tensor:
    """Perte du fitter (racine du taux, EPS) par cellule : (B,). Formule identique à `variable_source_fit.optimise`."""
    err = (rec.clamp(min=0) + EPS).sqrt() - targets.sT
    return ((err ** 2) * targets.valid).sum((1, 2)) / targets.npix


def sparsity_loss(gate: torch.Tensor, areas, n_par: int = 7, ok: torch.Tensor | None = None) -> torch.Tensor:
    """Moyenne sur les cellules de Σ_k porte · n_par / aire : paramètres rapportés à l'aire.
    ok (B, K) : places valides ; les autres sont exclues de la somme."""
    a = torch.as_tensor(areas, dtype=gate.dtype, device=gate.device)
    g = gate if ok is None else gate * ok.to(gate.dtype)
    return (g.sum(1) * n_par / a).mean()


def binarise_loss(gate: torch.Tensor, ok: torch.Tensor | None = None) -> torch.Tensor:
    """Moyenne de porte · (1 − porte) sur les places valides : nulle à 0 et 1, maximale (0.25) à 0.5."""
    term = gate * (1 - gate)
    if ok is None:
        return term.mean()
    m = ok.to(gate.dtype)
    return (term * m).sum() / m.sum().clamp(min=1.0)


def warmup_heatmap_loss(logits: torch.Tensor, peak_maps: torch.Tensor) -> torch.Tensor:
    """Perte focale sur la grille stride 2 contre des cibles gaussiennes (pic = 1), normalisée par le nombre de pics."""
    pos = (peak_maps >= 1.0).to(logits.dtype)
    log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    pos_term = pos * (1 - p) ** FOCAL_ALPHA * log_p
    neg_term = (1 - pos) * (1 - peak_maps) ** FOCAL_BETA * p ** FOCAL_ALPHA * log_q
    return -(pos_term + neg_term).sum() / pos.sum().clamp(min=1.0)


def greedy_peak_map(rate: np.ndarray, cell_mask: np.ndarray, k: int, sigma: float,
                    grid_hw: tuple[int, int], valid: np.ndarray | None = None) -> np.ndarray:
    """Carte cible (h2, w2) float32 : jusqu'à k pics placés gloutonnement sur le résidu racine, comme
    `vsf.place_sources` (maximum, puis suppression d'un carré de demi-côté sup = max(int(rho), 2),
    rho = 0.7·sqrt(aire / (π k))).

    Le résidu vaut −inf hors du masque de cellule, sur les pixels invalides et non finis : aucun pic n'y tombe.
    On s'arrête quand il ne reste aucun résidu positif (moins de k pics possible).
    Chaque pic est une gaussienne d'écart-type `sigma` (px) centrée au centre de cellule 2j + 0.5 (convention des
    centres de pixels du rendu), mise à exactement 1 dans la cellule du pixel du pic.
    grid_hw : taille de la grille, obligatoire (voir `source_encoder.head_grid_hw`)."""
    if k <= 0 or sigma <= 0:
        raise ValueError("k et sigma doivent être > 0")
    h2, w2 = grid_hw
    ok = np.isfinite(rate) & np.asarray(cell_mask, dtype=bool)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    res = np.where(ok, np.sqrt(np.clip(np.nan_to_num(rate), 0, None)) - math.sqrt(EPS), -np.inf)
    rho = float(np.clip(RHO_SCALE * math.sqrt(max(int(ok.sum()), 1) / (math.pi * k)), RHO_MIN, RHO_MAX))
    sup = max(int(rho), 2)
    centre = (HEAD_STRIDE - 1) / 2
    cy = HEAD_STRIDE * np.arange(h2)[:, None] + centre
    cx = HEAD_STRIDE * np.arange(w2)[None, :] + centre
    heat = np.zeros((h2, w2), dtype=np.float32)
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(res), res.shape)
        if not res[y, x] > 0:
            break
        blob = np.exp(-((cx - x) ** 2 + (cy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        heat = np.maximum(heat, blob)
        heat[min(y // HEAD_STRIDE, h2 - 1), min(x // HEAD_STRIDE, w2 - 1)] = 1.0
        res[max(y - sup, 0):y + sup + 1, max(x - sup, 0):x + sup + 1] = -np.inf
    return heat


@dataclass(frozen=True)
class LossConfig:
    """Poids des termes ; lambda_binarise monte linéairement sur binarise_ramp_steps pas."""

    lambda_sparsity: float = 0.01
    lambda_binarise: float = 0.05
    binarise_ramp_steps: int = 500
    lambda_heatmap: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lambda_sparsity", "lambda_binarise", "lambda_heatmap"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} doit être >= 0")
        if self.binarise_ramp_steps < 1:
            raise ValueError("binarise_ramp_steps doit être >= 1")

    def binarise_weight(self, step: int) -> float:
        return self.lambda_binarise * min(1.0, max(step, 0) / self.binarise_ramp_steps)


@dataclass(frozen=True, eq=False)
class LossParts:
    """Perte totale (tenseur, pour backward) et termes bruts (flottants, pour le journal)."""

    total: torch.Tensor
    recon: float
    sparsity: float
    binarise: float
    heatmap: float
    binarise_weight: float


def total_loss(rec: torch.Tensor, targets: Targets, gate: torch.Tensor, areas, cfg: LossConfig, step: int,
               logits: torch.Tensor | None = None, peak_maps: torch.Tensor | None = None,
               n_par: int = 7, ok: torch.Tensor | None = None) -> LossParts:
    """recon + λ_sp · parcimonie + λ_bin(step) · binarisation + λ_hm · carte de pics (si logits et peak_maps fournis)."""
    recon = recon_loss(rec, targets).mean()
    sparse = sparsity_loss(gate, areas, n_par, ok)
    binar = binarise_loss(gate, ok)
    use_heat = logits is not None and peak_maps is not None and cfg.lambda_heatmap > 0
    heat = warmup_heatmap_loss(logits, peak_maps) if use_heat else torch.zeros((), dtype=recon.dtype, device=recon.device)
    w_bin = cfg.binarise_weight(step)
    total = recon + cfg.lambda_sparsity * sparse + w_bin * binar + cfg.lambda_heatmap * heat
    return LossParts(total=total, recon=float(recon.detach()), sparsity=float(sparse.detach()), binarise=float(binar.detach()),
                     heatmap=float(heat.detach()), binarise_weight=w_bin)
