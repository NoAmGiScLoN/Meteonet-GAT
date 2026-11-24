"""Un pas d'entraînement par morceaux (accumulation de gradient) avec garde-fous mémoire (phase 5 E7).

Le lot est découpé en `chunks` morceaux le long de la dimension batch (même complétion que le lot entier,
donc mêmes activations) ; chaque terme de perte est normalisé par son dénominateur du lot entier : la somme
des pertes et des gradients des morceaux égale celle du lot entier. Un morceau dont le rendu dépasserait
`render_px_budget`, ou qui provoque un manque de mémoire, fait relancer le pas avec deux fois plus de morceaux ;
seule une cellule seule qui échoue encore lève une erreur.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import variable_source_fit as vsf
from encoder_loss import binarise_loss, recon_loss, sparsity_loss, warmup_heatmap_loss, LossConfig
from profile_lut import L_LUT
from size_batching import Batch
from source_encoder import EncoderConfig, HeadOut, SourceEncoder, decode_candidates, decode_candidates_hard, hard_gate_st  # noqa: F401

logger = logging.getLogger(__name__)

RENDER_KIND = vsf.render_kind("diff7")
ACTIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class StepStats:
    """Termes du lot entier (sommes des contributions des morceaux) et indicateurs du pas."""
    loss: float
    recon: float
    sparsity: float
    binarise: float
    heatmap: float
    k_active: float
    params_per_area: float
    render_ms: float
    n_chunks: int
    k_active_hard: float = 0.0  # Σ 1[p >= seuil] par cellule : ce que garde l'inférence
    gate_median: float = 0.0  # médiane des probabilités des places valides (moyenne pondérée des médianes des morceaux)
    max_abs_logit: float = 0.0  # extrêmes des sorties brutes de têtes (alerte précoce d'une divergence)
    max_raw_A: float = 0.0
    max_raw_a: float = 0.0
    max_raw_b: float = 0.0
    max_raw_w: float = 0.0
    grad_norm: float = 0.0  # norme avant écrêtage, renseignée par la boucle d'entraînement


class NonFiniteStep(RuntimeError):
    """Paramètres, portes, perte ou gradient non finis : le pas est à écarter (poids inchangés)."""


def all_finite(tensors) -> bool:
    """Tous les tenseurs sont finis ; une seule synchronisation hôte-device pour l'ensemble."""
    flags = [torch.isfinite(t).all() for t in tensors]
    return not flags or bool(torch.stack(flags).all())


def _require_finite(name: str, *tensors: torch.Tensor) -> None:
    if not all_finite(tensors):
        raise NonFiniteStep(f"{name} non finis")


def encode(model: SourceEncoder, x: torch.Tensor) -> HeadOut:
    """Passage de l'encodeur : bf16 pour le CNN sur CUDA seulement ; têtes en float32."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = model(x)
    return dataclasses.replace(out, **{f.name: getattr(out, f.name).float() for f in dataclasses.fields(out)})


def render_cost(params: torch.Tensor, gate: torch.Tensor, h: int, w: int) -> int:
    """Pixels de fenêtres de rendu : Σ (2·demi-taille + 1)² sur les candidats de porte > 0 (seaux de render_windowed)."""
    with torch.no_grad():
        p = params[gate > 0]
        if p.shape[0] == 0:
            return 0
        reach = L_LUT * torch.maximum(0.5 + F.softplus(p[:, 3]), 0.5 + F.softplus(p[:, 4]))
        sizes = torch.tensor(vsf.HALF_SIZES, dtype=reach.dtype, device=reach.device)
        bucket = torch.searchsorted(sizes, reach.ceil() + 1).clamp(max=len(vsf.HALF_SIZES) - 1)
        half = torch.clamp(sizes[bucket], max=float(max(h, w)))
        return int(((2 * half + 1) ** 2).sum())


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _slice(batch: Batch, a: int, b: int) -> Batch:
    t = batch.targets
    targets = dataclasses.replace(t, T=t.T[a:b], sT=t.sT[a:b], valid=t.valid[a:b], npix=t.npix[a:b])
    return dataclasses.replace(batch, x=batch.x[a:b], targets=targets, areas=batch.areas[a:b],
                               crop_hw=batch.crop_hw[a:b], meta=batch.meta[a:b])


def _ok_total(batch: Batch, k_train: int) -> int:
    """Places valides du lot : min(k, positions de grille dans le recadrage) par cellule (comme _decode)."""
    grid = (batch.x.shape[2] // 2) * (batch.x.shape[3] // 2)
    return sum(min(min(k_train, grid), math.ceil(h / 2) * math.ceil(w / 2)) for h, w in batch.crop_hw)


def _decode_for_training(out: HeadOut, k_train: int, batch: Batch, cfg: EncoderConfig, threshold: float | None):
    """(params, porte rendue, porte molle, ok, evaluate_zero). Échauffement (threshold None) : porte molle fractionnaire.
    Ensuite : porte dure à gradient direct ; les sources à porte 0 restent évaluées (gradient vers leur logit)."""
    if threshold is None:
        params, gate, ok = decode_candidates(out, k_train, batch.crop_hw, cfg)
        return params, gate, gate, ok, None
    params, gate, prob, ok = decode_candidates_hard(out, k_train, batch.crop_hw, cfg, threshold, batch.areas)
    return params, gate, prob, ok, ok


def _chunk(model: SourceEncoder, batch: Batch, peaks, lut5, step: int, k_train: int, cfg: LossConfig,
           budget: int, n_total: int, m_total: int, p_total: float, threshold: float | None = None,
           heat_scale: float = 1.0):
    """Perte d'un morceau (normalisée par le lot entier) et rétropropagation ; None si le rendu dépasse le budget."""
    out = encode(model, batch.x)
    params, gate, soft, ok, eval_zero = _decode_for_training(out, k_train, batch, model.cfg, threshold)
    _require_finite("paramètres ou portes", params, gate)  # avant le rendu (_check_gate reste le filet)
    t, n = batch.targets, batch.x.shape[0]
    if n > 1 and render_cost(params, ok.to(gate.dtype), t.Hm, t.Wm) > budget:
        return None
    t0 = time.perf_counter()
    rec = vsf.render_windowed(params, gate, t.Hm, t.Wm, RENDER_KIND, lut5, evaluate_zero=eval_zero)
    if rec.device.type == "cuda":
        torch.cuda.synchronize(rec.device)
    render_ms = 1000 * (time.perf_counter() - t0)
    recon = recon_loss(rec, t).sum() / n_total
    sparse = sparsity_loss(soft, batch.areas, model.cfg.n_par, ok, model.cfg.cap, model.cfg.compression_frac) * (n / n_total)
    m_c = ok.to(gate.dtype).sum()
    binar = binarise_loss(soft, ok) * m_c.clamp(min=1.0) / max(m_total, 1)
    use_heat = peaks is not None and cfg.lambda_heatmap > 0
    if use_heat:
        pos_c = float((peaks >= 1.0).sum())
        heat = warmup_heatmap_loss(out.logits, peaks) * max(pos_c, 1.0) / max(p_total, 1.0)
    else:
        heat = torch.zeros((), device=rec.device)
    total = recon + cfg.lambda_sparsity * sparse + cfg.binarise_weight(step) * binar + cfg.lambda_heatmap * heat_scale * heat
    _require_finite("perte", total)
    total.backward()
    active = ((soft.detach() >= ACTIVE_THRESHOLD) & ok).sum(1).float()
    hard = ((soft.detach() >= ACTIVE_THRESHOLD) & ok).sum(1).float() if threshold is None else (gate.detach() == 1.0).sum(1).float()
    median = float(soft.detach()[ok].median()) if bool(ok.any()) else 0.0
    return {"loss": float(total.detach()), "recon": float(recon.detach()), "sparsity": float(sparse.detach()),
            "binarise": float(binar.detach()), "heatmap": float(heat.detach()), "k": float(active.sum()),
            "k_hard": float(hard.sum()), "gate_median": median * n / n_total,
            "ppa": float((active * model.cfg.n_par / batch.areas).sum()), "render_ms": render_ms,
            **_head_extremes(out)}


MAX_KEYS = ("max_abs_logit", "max_raw_A", "max_raw_a", "max_raw_b", "max_raw_w")


def _head_extremes(out: HeadOut) -> dict[str, float]:
    """Plus grandes valeurs absolues des sorties brutes (un seul transfert vers l'hôte)."""
    heads = (out.logits, out.raw_A, out.raw_a, out.raw_b, out.raw_w)
    values = torch.stack([h.detach().abs().max() for h in heads]).tolist()
    return dict(zip(MAX_KEYS, values))


def _free_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attempt(model, batch: Batch, peaks, lut5, step, k_train, cfg, budget, chunks: int, threshold, heat_scale):
    """StepStats si tout passe, sinon "oom" ou "render" (les gradients partiels sont alors à jeter)."""
    n = batch.x.shape[0]
    m_total = _ok_total(batch, k_train)
    p_total = float((peaks >= 1.0).sum()) if peaks is not None else 0.0
    acc: dict[str, float] = {}
    for idx in np.array_split(np.arange(n), chunks):
        a, b = int(idx[0]), int(idx[-1]) + 1
        failed = False
        try:
            res = _chunk(model, _slice(batch, a, b), None if peaks is None else peaks[a:b], lut5, step, k_train, cfg,
                         budget, n, m_total, p_total, threshold, heat_scale)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            failed = True
        if failed:
            _free_memory()
            return "oom"
        if res is None:
            return "render"
        for key, value in res.items():
            acc[key] = max(acc.get(key, 0.0), value) if key in MAX_KEYS else acc.get(key, 0.0) + value
    _require_finite("gradients", *(p.grad for p in model.parameters() if p.grad is not None))
    return StepStats(loss=acc["loss"], recon=acc["recon"], sparsity=acc["sparsity"], binarise=acc["binarise"],
                     heatmap=acc["heatmap"], k_active=acc["k"] / n, k_active_hard=acc["k_hard"] / n,
                     gate_median=acc["gate_median"], params_per_area=acc["ppa"] / n,
                     render_ms=acc["render_ms"], n_chunks=chunks, **{k: acc[k] for k in MAX_KEYS})


def accumulate_step(model: SourceEncoder, batch: Batch, peaks: torch.Tensor | None, lut5: torch.Tensor, step: int, *,
                    k_train: int, loss_cfg: LossConfig, render_px_budget: int, chunks: int = 1,
                    gate_threshold: float | None = None, heat_scale: float = 1.0) -> StepStats:
    """Calcule la perte du lot et accumule les gradients dans model (remis à zéro au début) ; relance en
    doublant les morceaux si le rendu dépasse render_px_budget ou si la mémoire manque.
    gate_threshold None : porte molle (échauffement) ; sinon porte dure 1[p >= seuil] à gradient direct, comme
    l'inférence. heat_scale : facteur du poids de la carte de pics (décroissance en fin d'échauffement).
    Lève NonFiniteStep (gradients remis à zéro) si paramètres, portes, perte ou gradients ne sont pas finis."""
    n = batch.x.shape[0]
    chunks = max(1, min(chunks, n))
    while True:
        model.zero_grad(set_to_none=True)
        try:
            result = _attempt(model, batch, peaks, lut5, step, k_train, loss_cfg, render_px_budget, chunks, gate_threshold, heat_scale)
        except NonFiniteStep:
            model.zero_grad(set_to_none=True)  # aucun gradient partiel ne doit survivre au pas écarté
            raise
        if isinstance(result, StepStats):
            return result
        if chunks >= n:
            model.zero_grad(set_to_none=True)
            raise RuntimeError(f"manque de mémoire : une cellule seule ne tient pas (aire {batch.meta[0]['area']}, "
                               f"recadrage {batch.crop_hw[0]}) ; réduire k_train ou la taille de l'encodeur")
        chunks = min(n, chunks * 2)
        logger.warning("pas %d : %s -> relance en %d morceaux", step, result, chunks)
