"""Ajustement a K adaptatif : vagues de sources, arret sur cible / plafond / compression, elagage et retassage.

Regles d'arret (testees sur la cible de vague bornee) : divergence (nrmse non fini) > cible nrmse > plafond
k_cap (ou planning epuise) > limite de compression K_limit = max(1, floor(frac * aire / n_par)).
Le motif "cap" peut etre rapporte avec K < k_cap apres elagage : k_target_last (cible de la derniere vague)
et K (sources actives) permettent de le distinguer.
Compactage : une cellule terminee est retiree du lot par selection d'indices (params, active, cibles) sans
recadrage. Les resultats restent equivalents en virgule flottante a un ajustement isole : la perte est
separable par cellule et Adam est elementaire (pas de couplage entre cellules du lot). Cette equivalence
n'est valable qu'avec rng=None : avec un Generator, les tirages dependent de la composition du lot.
"""
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import variable_source_fit as vsf
from optim_schedule import LR_DEFAULT, validate_lr
from residual_stats import compute_residual_stats

log = logging.getLogger(__name__)

SCHEDULE_TAIL = (96, 128)
FLOOR_TAIL = (192, 256)
PRUNE_FRAC_DEFAULT = 0.01
COMPRESSION_FRAC_DEFAULT = 0.5   # D24 (session B) : 0.25 bloquait les 50-200 px à K≈2
TARGET_DEFAULT = 0.1
K_CAP_DEFAULT = 128
AMP_COL = 2
N_PAR_GAUSS_RENDER = 6
N_PAR_DIFFUSION_RENDER = 8
METRIC_KEYS = ("nrmse", "iou", "mass_err", "peak_err")


def _full_schedule() -> tuple[int, ...]:
    return tuple(vsf.SCHEDULE) + SCHEDULE_TAIL + FLOOR_TAIL


@dataclass(frozen=True)
class AdaptiveConfig:
    arm: str = "diff7"
    target: float = TARGET_DEFAULT
    k_cap: int = K_CAP_DEFAULT
    compression_frac: float | None = COMPRESSION_FRAC_DEFAULT
    prune_frac: float = PRUNE_FRAC_DEFAULT
    lr: float = LR_DEFAULT
    iters_override: int | None = None

    def __post_init__(self) -> None:
        if self.arm not in vsf.ARM_NPAR:
            raise ValueError(f"bras inconnu : {self.arm!r}")
        if not self.target >= 0:
            raise ValueError(f"target doit etre >= 0, recu {self.target!r}")
        if self.k_cap not in _full_schedule():
            raise ValueError(f"k_cap doit appartenir au planning {_full_schedule()}, recu {self.k_cap!r}")
        if self.compression_frac is not None and not 0 < self.compression_frac <= 1:
            raise ValueError(f"compression_frac doit etre dans (0, 1] ou None, recu {self.compression_frac!r}")
        if not 0 <= self.prune_frac < 1:
            raise ValueError(f"prune_frac doit etre dans [0, 1), recu {self.prune_frac!r}")
        validate_lr(self.lr, None)
        if self.iters_override is not None and self.iters_override < 1:
            raise ValueError(f"iters_override doit etre >= 1, recu {self.iters_override!r}")


def wave_schedule(cap: int) -> tuple[int, ...]:
    """Tete SCHEDULE puis queues, tronquees a `cap` (qui doit etre une valeur du planning)."""
    full = _full_schedule()
    if cap not in full:
        raise ValueError(f"cap doit appartenir au planning {full}, recu {cap!r}")
    return tuple(k for k in full if k <= cap)


def compression_k_limit(area: int, n_par: int, frac: float | None) -> int | None:
    """max(1, floor(frac * aire / n_par)) ; None si la compression est desactivee."""
    if frac is None:
        return None
    return max(1, math.floor(frac * area / n_par))


def stop_reason(nrmse: float, k_target: int, k_cap: int, k_limit: int | None,
                schedule_exhausted: bool, target: float) -> str | None:
    """Priorite : diverged (nrmse non fini) > target > cap (cible = plafond) > compression > cap (calendrier epuise).

    k_limit == k_cap donne "cap" ; None = continuer."""
    if not math.isfinite(nrmse):
        return "diverged"
    if nrmse <= target:
        return "target"
    if k_target == k_cap:
        return "cap"
    if k_limit is not None and k_target == k_limit:
        return "compression"
    return "cap" if schedule_exhausted else None


def prune(params: torch.Tensor, active: torch.Tensor, peak_rate: np.ndarray,
          frac: float) -> tuple[torch.Tensor, np.ndarray]:
    """Desactive les sources d'amplitude softplus < frac * pic de la cellule (garde la plus forte). Nouveau masque."""
    amp = F.softplus(params[..., AMP_COL]).detach()
    is_active = active > 0
    thresh = torch.as_tensor(frac * np.asarray(peak_rate), dtype=amp.dtype, device=amp.device)[:, None]
    keep = is_active & (amp >= thresh)
    strongest = torch.where(is_active, amp, torch.full_like(amp, -1.0)).argmax(1)
    keep[torch.arange(len(keep)), strongest] |= is_active.any(1)
    new_active = keep.to(active.dtype)
    n_pruned = (is_active.sum(1) - keep.sum(1)).cpu().numpy().astype(int)
    return new_active, n_pruned


def repack(params: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Ramene les lignes actives en tete (ordre stable) ; nouveaux tenseurs, K actif par cellule."""
    is_active = active > 0
    order = torch.argsort((~is_active).to(torch.int8), dim=1, stable=True)
    new_params = torch.gather(params, 1, order[..., None].expand(-1, -1, params.shape[-1]))
    k_cur = is_active.sum(1)
    mask = torch.arange(active.shape[1], device=active.device)[None, :] < k_cur[:, None]
    new_params = torch.where(mask[..., None], new_params, torch.zeros_like(new_params))
    return new_params, mask.to(active.dtype), k_cur.cpu().numpy().astype(int)


@dataclass(frozen=True, eq=False)
class _Live:
    """Etat du lot de cellules encore actives (index le long du lot)."""
    cells: list[dict]
    params: torch.Tensor
    active: torch.Tensor
    targets: vsf.Targets
    k_cur: np.ndarray
    k_limit: list[int | None]
    peak: np.ndarray
    n_pruned: np.ndarray
    fit_s: np.ndarray
    waves: np.ndarray

    def select(self, keep: Sequence[int]) -> "_Live":
        idx = torch.as_tensor(list(keep), dtype=torch.long, device=self.params.device)
        t = self.targets
        targets = vsf.Targets(T=t.T[idx], sT=t.sT[idx], valid=t.valid[idx], npix=t.npix[idx], Hm=t.Hm, Wm=t.Wm)
        return _Live(cells=[self.cells[i] for i in keep], params=self.params[idx], active=self.active[idx],
                     targets=targets, k_cur=self.k_cur[list(keep)], k_limit=[self.k_limit[i] for i in keep],
                     peak=self.peak[list(keep)], n_pruned=self.n_pruned[list(keep)],
                     fit_s=self.fit_s[list(keep)], waves=self.waves[list(keep)])


def _init_live(cells: list[dict], cfg: AdaptiveConfig) -> _Live:
    n_par = vsf.ARM_NPAR[cfg.arm]
    p_cols = N_PAR_GAUSS_RENDER if vsf.render_kind(cfg.arm) == "gauss" else N_PAR_DIFFUSION_RENDER
    n = len(cells)
    return _Live(cells=list(cells), params=torch.zeros(n, cfg.k_cap, p_cols, device=vsf.DEVICE),
                 active=torch.zeros(n, cfg.k_cap, device=vsf.DEVICE), targets=vsf.make_targets(cells),
                 k_cur=np.zeros(n, dtype=int),
                 k_limit=[compression_k_limit(c["area"], n_par, cfg.compression_frac) for c in cells],
                 peak=np.array([float(c["rate"].max()) for c in cells]), n_pruned=np.zeros(n, dtype=int),
                 fit_s=np.zeros(n), waves=np.zeros(n, dtype=int))


def _validate_cells(cells: list[dict]) -> None:
    if not cells:
        raise ValueError("aucune cellule")
    if any("id" not in c for c in cells):
        raise ValueError("chaque cellule doit avoir une cle 'id'")
    if len({c["bin"] for c in cells}) != 1:
        raise ValueError("les cellules doivent appartenir a une seule classe de taille")


def wave_k_target(wave_k: int, k_cap: int, k_limit: int | None) -> int:
    """Cible de la vague : valeur du planning bornee par le plafond puis, si actif, par la limite de compression."""
    target = min(wave_k, k_cap)
    return target if k_limit is None else min(target, k_limit)


class _WaveOutput(NamedTuple):
    live: _Live
    mets: list[dict]
    k_target: list[int]
    seconds: float
    n_pruned: np.ndarray
    rec: torch.Tensor


def _run_wave(live: _Live, wave_k: int, cfg: AdaptiveConfig, iters: int, lut5, rng) -> _WaveOutput:
    """Place -> optimise -> elague -> retasse -> evalue.

    Bien que _Live soit gele, place_sources ecrit en place dans live.params / live.active (les tenseurs de l'etat
    precedent sont donc modifies) ; l'etat retourne est un nouveau _Live.
    """
    t0 = time.perf_counter()
    k_target = [wave_k_target(wave_k, cfg.k_cap, lim) for lim in live.k_limit]
    vsf.place_sources(live.params, live.active, live.targets, live.cells, live.k_cur, np.array(k_target), cfg.arm,
                      rng, lut5)
    params = vsf.optimise(live.params, live.active, live.targets, cfg.arm, lut5, iters, lr=cfg.lr, lr_schedule=None)
    active, n_pruned = prune(params, live.active, live.peak, cfg.prune_frac)
    params, active, k_cur = repack(params, active)
    rec, mets = vsf.evaluate(params, active, live.targets, cfg.arm, lut5)
    seconds = time.perf_counter() - t0
    live = _Live(**{**live.__dict__, "params": params, "active": active, "k_cur": k_cur,
                    "n_pruned": live.n_pruned + n_pruned, "fit_s": live.fit_s + seconds / len(live.cells),
                    "waves": live.waves + 1})
    return _WaveOutput(live, mets, k_target, seconds, n_pruned, rec)


def _result_row(live: _Live, i: int, cfg: AdaptiveConfig, met: dict, reason: str, k_target: int, rec: torch.Tensor) -> dict:
    c, k = live.cells[i], int(live.k_cur[i])
    h, w = c["rate"].shape
    stats = compute_residual_stats(rec[i, :h, :w].cpu().numpy(), c["rate"]).as_dict()
    n_params = k * vsf.ARM_NPAR[cfg.arm]
    return dict(id=c["id"], tag=c["tag"], date=c["date"], bin=c["bin"], area=c["area"], K=k, k_target_last=k_target,
                stop_reason=reason, waves=int(live.waves[i]), **met, n_params=n_params,
                compression_ratio=n_params / c["area"], fit_s=float(live.fit_s[i]),
                n_pruned=int(live.n_pruned[i]), k_limit=live.k_limit[i], **stats)


def _source_entry(live: _Live, i: int) -> dict:
    c, k = live.cells[i], int(live.k_cur[i])
    return dict(params=live.params[i, :k].detach().cpu().numpy().copy(), x0=c["x0"], y0=c["y0"], cx=c["cx"],
                cy=c["cy"], tag=c["tag"], date=c["date"], bin=c["bin"], area=c["area"])


def fit_adaptive(cells: list[dict], cfg: AdaptiveConfig, lut5: torch.Tensor | None,
                 rng: np.random.Generator | None = None,
                 on_cell_done: Callable[[dict, dict], None] | None = None) -> tuple[list[dict], list[dict], dict]:
    """Ajuste un lot de cellules d'une meme classe ; retourne (resultats, trace par vague, sources par id).

    Motifs d'arret : target, cap, compression, diverged (nrmse non fini, avertissement journalise).
    Dans la trace, `seconds` = duree de la vague pour le lot / nombre de cellules vivantes (par cellule et vague).
    L'equivalence avec un ajustement isole n'est garantie qu'avec rng=None.
    """
    _validate_cells(cells)
    vsf.check_lut_depth(cfg.arm, lut5)
    iters = cfg.iters_override or vsf.iters_for_bin(cells[0]["bin"], None)
    sched = wave_schedule(cfg.k_cap)
    live, results, trace, sources = _init_live(cells, cfg), [], [], {}
    for wave, wave_k in enumerate(sched):
        live, mets, k_target, seconds, n_pruned, rec = _run_wave(live, wave_k, cfg, iters, lut5, rng)
        keep = []
        for i, met in enumerate(mets):
            trace.append(dict(id=live.cells[i]["id"], wave=wave, k_target=k_target[i], K_active=int(live.k_cur[i]),
                              n_pruned=int(n_pruned[i]), **met, seconds=seconds / len(mets)))
            reason = stop_reason(met["nrmse"], k_target[i], cfg.k_cap, live.k_limit[i],
                                 wave == len(sched) - 1, cfg.target)
            if reason is None:
                keep.append(i)
                continue
            if reason == "diverged":
                log.warning("cellule %s : nrmse non fini a la vague %d (K=%d)", live.cells[i]["id"], wave,
                            int(live.k_cur[i]))
            row, entry = _result_row(live, i, cfg, met, reason, k_target[i], rec), _source_entry(live, i)
            results.append(row)
            sources[row["id"]] = entry
            log.info("cellule %s : arret %s a K=%d (nrmse %.3f)", row["id"], reason, row["K"], met["nrmse"])
            if on_cell_done is not None:
                on_cell_done(row, entry)
        if not keep:
            break
        if len(keep) < len(mets):
            live = live.select(keep)
    return results, trace, sources
