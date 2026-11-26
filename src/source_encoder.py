"""Encodeur de sources (phase 4 E7) : détecteur entièrement convolutif (UNet) à stride 2.

Entrée : 3 canaux (racine du taux, masque de cellule, masque de validité), complétés à un multiple de 32.
Sortie : cartes de têtes à H/2 x W/2 décodées en paramètres bruts diff7 (B, K, 8) du fitter
[x, y, A, a, b, theta, tau (=0, gelé), w] et en portes fractionnaires (plancher > 0 pour garder un gradient).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_fit import compression_k_limit

logger = logging.getLogger(__name__)

N_PARAMS_RAW = 8  # colonnes du format brut diff7
LOGIT_BIAS_INIT = -2.0  # probabilité initiale faible : peu de sources actives au départ
AMPLITUDE_INIT = 0.5  # amplitude initiale (mm/h) : petite
HEAD_WEIGHT_STD = 0.01  # poids de tête petits : l'initialisation est portée par les biais
LOGIT_MASKED = -1e4  # logit des positions hors zone valide (zone de padding)
MIN_RADIUS_LIMIT = 1.0  # plancher de la limite de rayon (px), pour les très petits recadrages
GROUPS = 4
SOFT_CLAMP_FRAC = 0.1  # largeur de la zone de saturation douce du rayon, en fraction de la limite
ANGLE_EPS = 1e-6  # décalage de cos2t : gradient fini de atan2 en (0, 0)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


@dataclass(frozen=True)
class EncoderConfig:
    """Hyperparamètres de l'encodeur et du décodage."""

    in_ch: int = 3
    base_ch: int = 24
    depth: int = 4
    head_stride: int = 2
    radius_clamp_frac: float = 0.5
    gate_floor: float = 1e-3
    k_train: int = 256
    cap: int = 128
    compression_frac: float = 0.5
    n_par: int = 7

    def __post_init__(self) -> None:
        for name in ("in_ch", "base_ch", "depth", "k_train", "cap", "n_par"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} doit être un entier > 0, reçu {value!r}")
        if self.base_ch % GROUPS:
            raise ValueError(f"base_ch doit être un multiple de {GROUPS}")
        if self.head_stride != 2:
            raise ValueError("seul head_stride=2 est supporté")
        if not 0 < self.gate_floor < 1:
            raise ValueError("gate_floor doit être dans ]0, 1[")
        if self.radius_clamp_frac <= 0 or self.compression_frac <= 0:
            raise ValueError("radius_clamp_frac et compression_frac doivent être > 0")

    @property
    def pad_multiple(self) -> int:
        """Stride au goulot : 2 (stem) * 2**depth = 32 pour depth=4."""
        return 2 ** (self.depth + 1)


@dataclass(frozen=True)
class PadInfo:
    """Taille d'origine (h, w) d'un recadrage avant complétion."""

    h: int
    w: int


@dataclass(frozen=True)
class HeadOut:
    """Cartes de têtes à H/2 x W/2 : toutes (B, 1, h, w) sauf offsets (B, 2, h, w).

    cos2t / sin2t sont les sorties brutes (non normalisées) : theta = 0.5 * atan2(sin2t, cos2t)."""

    logits: torch.Tensor
    offsets: torch.Tensor
    raw_A: torch.Tensor
    raw_a: torch.Tensor
    raw_b: torch.Tensor
    cos2t: torch.Tensor
    sin2t: torch.Tensor
    raw_w: torch.Tensor


RADIUS_SOFT_CLAMP = 250.0  # borne douce des rayons bruts (couvre la plus grande limite de rayon, ~195)
HEAD_SOFT_CLAMP = 50.0  # borne douce des sorties brutes non bornées (logit, A, w) : c·tanh(x/c)


def soft_clamp(x: torch.Tensor, c: float = HEAD_SOFT_CLAMP) -> torch.Tensor:
    """c·tanh(x/c) : quasi identité près de 0 (écart < 1e-3 pour |x| < 1, ~0.02 à |x| = 3), bornée par ±c ;
    empêche une sortie de tête d'exploser en inf/NaN."""
    return c * torch.tanh(x / c)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(GROUPS, c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(GROUPS, c_out), nn.ReLU(inplace=True))


class SourceEncoder(nn.Module):
    """UNet : stem stride 2, `depth` descentes (stride 32), contexte global, remontée jusqu'au stride 2."""

    N_HEAD = 9  # logit, 2 offsets, A, a, b, cos2t, sin2t, w

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = [min(cfg.base_ch * 2 ** i, 4 * cfg.base_ch) for i in range(cfg.depth + 1)]
        self.stem = nn.Sequential(nn.Conv2d(cfg.in_ch, ch[0], 3, stride=2, padding=1),
                                  nn.GroupNorm(GROUPS, ch[0]), nn.ReLU(inplace=True), _block(ch[0], ch[0]))
        self.down = nn.ModuleList(_block(ch[i - 1], ch[i]) for i in range(1, cfg.depth + 1))
        self.context = nn.Conv2d(2 * ch[-1], ch[-1], 1)
        self.up = nn.ModuleList(_block(ch[i] + ch[i - 1], ch[i - 1]) for i in range(cfg.depth, 0, -1))
        self.head = nn.Conv2d(ch[0], self.N_HEAD, 1)
        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.head.weight, std=HEAD_WEIGHT_STD)
        radius = _inv_softplus(self.cfg.head_stride * 2 - 0.5)  # rayon initial = stride * 2 px
        bias = [LOGIT_BIAS_INIT, 0.0, 0.0, _inv_softplus(AMPLITUDE_INIT), radius, radius, 1.0, 0.0, 0.0]
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> HeadOut:
        m = self.cfg.pad_multiple
        if x.dim() != 4 or x.shape[1] != self.cfg.in_ch or x.shape[2] % m or x.shape[3] % m:
            raise ValueError(f"entrée (B, {self.cfg.in_ch}, H, W) avec H, W multiples de {m} attendue, reçu {tuple(x.shape)}")
        skips = [self.stem(x)]
        for block in self.down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        z = skips[-1]
        z = F.relu(self.context(torch.cat([z, z.mean((2, 3), keepdim=True).expand_as(z)], 1)))
        for block, skip in zip(self.up, reversed(skips[:-1])):
            z = block(torch.cat([F.interpolate(z, scale_factor=2, mode="nearest"), skip], 1))
        with torch.autocast(device_type=z.device.type, enabled=False):  # positions et rayons en fp32
            h = self.head(z.float())
        return HeadOut(logits=soft_clamp(h[:, 0:1]), offsets=torch.tanh(h[:, 1:3]), raw_A=soft_clamp(h[:, 3:4]),
                       raw_a=soft_clamp(h[:, 4:5], RADIUS_SOFT_CLAMP), raw_b=soft_clamp(h[:, 5:6], RADIUS_SOFT_CLAMP), cos2t=h[:, 6:7], sin2t=h[:, 7:8],
                       raw_w=soft_clamp(h[:, 8:9]))


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ entrées
def _to_padded(rate: np.ndarray, cell_mask: np.ndarray, valid: np.ndarray, hp: int, wp: int) -> torch.Tensor:
    h, w = rate.shape
    x = torch.zeros(3, hp, wp)
    x[0, :h, :w] = torch.from_numpy(np.sqrt(np.clip(rate, 0, None)).astype(np.float32))
    x[1, :h, :w] = torch.from_numpy(np.asarray(cell_mask, dtype=np.float32))
    x[2, :h, :w] = torch.from_numpy(np.asarray(valid, dtype=np.float32))
    return x


def _ceil_to(n: int, m: int) -> int:
    return -(-n // m) * m


def head_grid_hw(h: int, w: int, multiple: int = 32, stride: int = 2) -> tuple[int, int]:
    """Taille (h2, w2) de la grille des têtes pour un recadrage (ou le maximum d'un lot) h x w après complétion."""
    return _ceil_to(h, multiple) // stride, _ceil_to(w, multiple) // stride


def pad_inputs(rate: np.ndarray, cell_mask: np.ndarray, valid: np.ndarray,
               multiple: int = 32) -> tuple[torch.Tensor, PadInfo]:
    """Construit les 3 canaux (racine du taux, masque de cellule, validité), complétés par des zéros."""
    h, w = rate.shape
    return _to_padded(rate, cell_mask, valid, _ceil_to(h, multiple), _ceil_to(w, multiple)), PadInfo(h, w)


def pad_batch(crops: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
              multiple: int = 32) -> tuple[torch.Tensor, list[PadInfo]]:
    """Lot de recadrages de tailles différentes, complétés au maximum du lot (multiple de `multiple`)."""
    if not crops:
        raise ValueError("lot vide")
    hp = _ceil_to(max(c[0].shape[0] for c in crops), multiple)
    wp = _ceil_to(max(c[0].shape[1] for c in crops), multiple)
    x = torch.stack([_to_padded(*c, hp, wp) for c in crops])
    return x, [PadInfo(*c[0].shape) for c in crops]


# ------------------------------------------------------------------ décodage
def _crop_sizes(crop_hw: Sequence, batch: int) -> list[tuple[int, int]]:
    if len(crop_hw) == 2 and all(isinstance(v, (int, np.integer)) for v in crop_hw):
        return [(int(crop_hw[0]), int(crop_hw[1]))] * batch
    if len(crop_hw) != batch:
        raise ValueError(f"crop_hw : {batch} tailles attendues, reçu {len(crop_hw)}")
    return [(int(h), int(w)) for h, w in crop_hw]


def _inv_softplus_stable(s: torch.Tensor) -> torch.Tensor:
    """log(expm1(s)) sans débordement fp32 (expm1 dépasse pour s > ~88.7) : s + log(1 − e^(−s)), s > 0."""
    return s + torch.log(-torch.expm1(-s))


def _soft_clamp_raw(raw: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Borne douce du rayon physique r = 0.5 + softplus(raw) : r' = limit − softplus_τ(limit − r) < limit, avec
    τ = SOFT_CLAMP_FRAC * limit (identité loin de la limite, gradient non nul près de celle-ci). Retourne le brut associé."""
    s_max = (limit - 0.5)[:, None]  # on travaille sur s = softplus(raw) = r − 0.5 >= 0
    beta = 1.0 / (SOFT_CLAMP_FRAC * s_max)
    s_soft = s_max - F.softplus((s_max - F.softplus(raw)) * beta) / beta
    return _inv_softplus_stable(s_soft.clamp(min=1e-6))


def _nms_logits(out: HeadOut) -> tuple[HeadOut, torch.Tensor]:
    """NMS 3x3 sur la grille des têtes : (têtes dont les logits non maxima locaux de p sont masqués, carte keep)."""
    p = torch.sigmoid(out.logits)
    keep = p == F.max_pool2d(p, 3, stride=1, padding=1)
    return dataclasses.replace(out, logits=torch.where(keep, out.logits, torch.full_like(out.logits, LOGIT_MASKED))), keep


def _k_limits(areas, cfg: EncoderConfig, device: torch.device) -> torch.Tensor:
    """(B,) min(cap, limite de compression D24) par cellule."""
    return torch.tensor([min(cfg.cap, compression_k_limit(int(a), cfg.n_par, cfg.compression_frac)) for a in areas],
                        device=device)


def _on_mask(prob: torch.Tensor, ok: torch.Tensor, threshold: float, limits: torch.Tensor) -> torch.Tensor:
    """Sources gardées : p >= seuil, place valide, rang (candidats triés par logit) < limite. Partagé entre
    l'inférence (`select_sources`) et l'entraînement à porte dure, qui ne peuvent donc pas diverger."""
    rank = torch.arange(prob.shape[1], device=prob.device)[None, :]
    return (prob >= threshold) & ok & (rank < limits[:, None])


def _decode(out: HeadOut, k: int, crop_hw: Sequence, cfg: EncoderConfig, keep: torch.Tensor | None = None):
    """Retourne (params (B,k,8), prob (B,k), ok (B,k)) : top-k des logits dans la zone non complétée.

    Politique de position : un candidat peut se trouver n'importe où dans le recadrage (non complété), y compris
    juste hors du masque de cellule ou sur des pixels manquants : une source peut déborder du masque et la perte
    ignore les pixels invalides. Seul le padding est exclu ; les places sans position valide ont ok = False."""
    B, _, h2, w2 = out.logits.shape
    sizes = _crop_sizes(crop_hw, B)
    hh = torch.tensor([s[0] for s in sizes], device=out.logits.device)
    ww = torch.tensor([s[1] for s in sizes], device=out.logits.device)
    ii = torch.arange(h2, device=hh.device)[None, :, None]
    jj = torch.arange(w2, device=hh.device)[None, None, :]
    stride = cfg.head_stride
    valid = ((ii * stride < hh[:, None, None]) & (jj * stride < ww[:, None, None])).reshape(B, -1)
    flat = out.logits.reshape(B, -1)
    k = min(k, flat.shape[1])
    idx = torch.where(valid, flat, torch.full_like(flat, LOGIT_MASKED)).topk(k, dim=1).indices

    def pick(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, -1).gather(1, idx)

    row, col = torch.div(idx, w2, rounding_mode="floor"), idx % w2
    offx, offy = pick(out.offsets[:, 0:1]), pick(out.offsets[:, 1:2])
    centre = (stride - 1) / 2  # centre de la cellule j : stride * j + 0.5 (pixels entiers = centres de pixels)
    x = (stride * col + centre + stride * offx).clamp(min=0)
    y = (stride * row + centre + stride * offy).clamp(min=0)
    x, y = torch.minimum(x, (ww - 1)[:, None].to(x.dtype)), torch.minimum(y, (hh - 1)[:, None].to(y.dtype))
    limit = torch.clamp(0.5 * cfg.radius_clamp_frac * torch.maximum(hh, ww).to(x.dtype), min=MIN_RADIUS_LIMIT)
    theta = 0.5 * torch.atan2(pick(out.sin2t), pick(out.cos2t) + ANGLE_EPS)
    params = torch.stack([x, y, pick(out.raw_A), _soft_clamp_raw(pick(out.raw_a), limit),
                          _soft_clamp_raw(pick(out.raw_b), limit), theta, torch.zeros_like(x), pick(out.raw_w)], -1)
    ok = valid.gather(1, idx)
    if keep is not None:  # candidats non maxima exclus (inexistants à l'inférence)
        ok = ok & keep.reshape(B, -1).gather(1, idx)
    return params, torch.sigmoid(pick(out.logits)), ok


def decode_candidates(out: HeadOut, k_train: int, crop_hw: Sequence,
                      cfg: EncoderConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entraînement : (params (B,k,8) bruts, porte (B,k), ok (B,k) booléen).

    Porte dans [gate_floor, 1] pour les places ok ; exactement 0 ailleurs (le rendu les saute, gradient nul).
    Rayon borné en douceur : 2 * max(a, b) < radius_clamp_frac * max(H, W) (plancher MIN_RADIUS_LIMIT sur la limite)."""
    params, prob, ok = _decode(out, k_train, crop_hw, cfg)
    gate = cfg.gate_floor + (1 - cfg.gate_floor) * prob
    return params, torch.where(ok, gate, torch.zeros_like(gate)), ok


def hard_gate_st(prob: torch.Tensor, ok: torch.Tensor, threshold: float, limits: torch.Tensor | None = None) -> torch.Tensor:
    """Porte dure à gradient direct : 1 pour les sources que garde `select_sources` en avant (p >= seuil et, si
    `limits` (B,) est donné, rang < limite), gradient de p en arrière (hard + (p − p.detach()), exact en avant).
    Places non ok exactement à 0."""
    lim = torch.full((prob.shape[0],), prob.shape[1], device=prob.device) if limits is None else limits
    hard = _on_mask(prob.detach(), ok, threshold, lim).to(prob.dtype)
    return torch.where(ok, hard + (prob - prob.detach()), torch.zeros_like(prob))


def decode_candidates_hard(out: HeadOut, k_train: int, crop_hw: Sequence, cfg: EncoderConfig, threshold: float,
                           areas) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entraînement après l'échauffement : (params (B,k,8), porte dure à gradient (B,k), prob (B,k), ok (B,k)).

    La porte rendue vaut exactement la sélection de `select_sources` : NMS 3x3, p >= seuil, au plus
    min(cap, limite de compression) sources par cellule. ok = places valides ET maxima locaux (les non maxima
    n'existent pas à l'inférence : ni rendus, ni comptés). prob = probabilité molle (0 hors ok), pour la parcimonie
    et la binarisation. Le rendu doit évaluer les sources à porte 0 (`evaluate_zero=ok`) pour que leur logit reçoive
    un gradient (une source éteinte forte peut en évincer une plus faible)."""
    masked, keep = _nms_logits(out)
    params, prob, ok = _decode(masked, k_train, crop_hw, cfg, keep)
    prob = torch.where(ok, prob, torch.zeros_like(prob))
    return params, hard_gate_st(prob, ok, threshold, _k_limits(areas, cfg, prob.device)), prob, ok


@torch.no_grad()
def select_sources(out: HeadOut, areas, crop_hw: Sequence, cfg: EncoderConfig,
                   threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Inférence : sources de probabilité >= threshold, au plus min(cap, limite de compression) par cellule.

    NMS 3x3 sur la grille des têtes : seuls les maxima locaux de p sont gardés (une source par blob).
    Retourne (params (B, Kmax, 8), active (B, Kmax) 0/1), actives en tête (triées par probabilité)."""
    out, _ = _nms_logits(out)
    limits = _k_limits(areas, cfg, out.logits.device)
    params, prob, ok = _decode(out, int(limits.max()), crop_hw, cfg)
    return params, _on_mask(prob, ok, threshold, limits).to(params.dtype)
