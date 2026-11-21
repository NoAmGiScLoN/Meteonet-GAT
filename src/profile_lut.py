"""Diffusion profile tables: shared PDE core and the log-spaced-w table used by the "diff7" arm.

The radial profile of a source is the solution of an explicit conservative diffusion with
D(r) = dmax * (tau * sigmoid((1 - r) / w) + floor), starting from a point-like Gaussian, in the
source's own elliptical frame. It only depends on (tau, w), hence can be tabulated.
"""
import hashlib
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

logger = logging.getLogger(__name__)

L_LUT = 2.0  # profile defined for r in [0, L_LUT] (in source radii), zero beyond
N_ANGLES = 32  # directions averaged for the radial profile
KEY_LENGTH = 10  # hex chars of the sha1 used in the table file name
LOGW_PREFIX = "diffusion_profiles_logw_"
DEFAULT_W_MIN = 0.02  # w axis of the default (tau, w) table and default bounds of the log-w table
DEFAULT_W_MAX = 0.5
DEFAULT_N_W = 16
FLOAT_FIELDS = ("w_min", "w_max", "tau", "sigma0", "floor", "dmax")


def default_w_nodes(n_w: int = DEFAULT_N_W) -> torch.Tensor:
    """Linearly spaced w nodes of the default (tau, w) table."""
    return torch.linspace(DEFAULT_W_MIN, DEFAULT_W_MAX, n_w)


def diffuse_profiles(taus: torch.Tensor, ws: torch.Tensor, n_r: int, n: int, steps: int,
                     sigma0: float, floor: float, dmax: float, l_lut: float = L_LUT) -> torch.Tensor:
    """Radial profiles for every (tau, w) pair, shape (n_tau * n_w, n_r), tau-major, max-normalised.

    Explicit conservative diffusion on an n x n grid over [-l_lut, l_lut]^2 (no-flux border);
    the last radial bin is forced to 0.
    """
    ax = (torch.arange(n) + 0.5) / n * 2 * l_lut - l_lut
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    R = torch.sqrt(X ** 2 + Y ** 2)
    D = dmax * (taus[:, None, None, None] * torch.sigmoid((1 - R) / ws[None, :, None, None]) + floor)
    D = D.reshape(-1, 1, n, n)  # D en unités de grille (dt/h^2 absorbé), stable car D <= 0.25
    u = torch.exp(-R ** 2 / (2 * sigma0 ** 2)).expand(D.shape[0], 1, n, n).clone()
    Dx, Dy = 0.5 * (D[..., :, 1:] + D[..., :, :-1]), 0.5 * (D[..., 1:, :] + D[..., :-1, :])
    for _ in range(steps):
        fx = Dx * (u[..., :, 1:] - u[..., :, :-1])
        fy = Dy * (u[..., 1:, :] - u[..., :-1, :])
        u = u + F.pad(fx, (0, 1)) - F.pad(fx, (1, 0)) + F.pad(fy, (0, 0, 0, 1)) - F.pad(fy, (0, 0, 1, 0))
    r = torch.linspace(0, l_lut, n_r)
    ang = torch.linspace(0, 2 * math.pi, N_ANGLES + 1)[:-1]
    gx = (r[None, :] * torch.cos(ang[:, None])) / l_lut
    gy = (r[None, :] * torch.sin(ang[:, None])) / l_lut
    grid = torch.stack([gx, gy], -1)[None].expand(u.shape[0], -1, -1, -1)
    prof = F.grid_sample(u, grid, align_corners=False).mean(2)[:, 0]
    prof = prof / prof.max(1, keepdim=True).values
    prof[:, -1] = 0
    return prof


@dataclass(frozen=True)
class LogWLutSpec:
    """Parameters of the log-w table (single tau slice, w nodes geometrically spaced)."""

    w_min: float = DEFAULT_W_MIN
    w_max: float = DEFAULT_W_MAX
    n_w: int = DEFAULT_N_W
    tau: float = 0.5
    n_r: int = 128
    n: int = 96
    steps: int = 1200
    sigma0: float = 0.15
    floor: float = 0.005
    dmax: float = 0.24

    def __post_init__(self) -> None:
        for name in FLOAT_FIELDS:  # tau=1 and tau=1.0 must give the same key
            object.__setattr__(self, name, float(getattr(self, name)))
        if not 0 < self.w_min < self.w_max:
            raise ValueError(f"need 0 < w_min < w_max, got {self.w_min}, {self.w_max}")
        if not 0 <= self.tau <= 1:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        for name in ("n_w", "n_r", "n", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.n_w < 2:
            raise ValueError(f"n_w must be >= 2, got {self.n_w}")

    def key(self) -> str:
        """Short stable hash of every field, used in the table file name."""
        payload = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha1(payload).hexdigest()[:KEY_LENGTH]


def logw_lut_path(out: Path, spec: LogWLutSpec) -> Path:
    return Path(out) / f"{LOGW_PREFIX}{spec.key()}.pt"


def build_logw_lut(spec: LogWLutSpec, out: Path) -> dict:
    """Solve the PDE for tau = spec.tau and log-spaced w, save and return the table.

    Returns {"lut": (1, n_w, n_r), "ws": (n_w,), "spec": dict}.
    """
    ws = torch.logspace(math.log10(spec.w_min), math.log10(spec.w_max), spec.n_w)
    taus = torch.tensor([spec.tau], dtype=torch.float32)
    prof = diffuse_profiles(taus, ws, spec.n_r, spec.n, spec.steps, spec.sigma0, spec.floor, spec.dmax, L_LUT)
    table = {"lut": prof.reshape(1, spec.n_w, spec.n_r), "ws": ws, "spec": asdict(spec)}
    path = logw_lut_path(out, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, path)
    logger.info("log-w table built: %s (key %s)", path, spec.key())
    return table


def load_or_build_logw_lut(spec: LogWLutSpec, out: Path) -> dict:
    """Load the table for `spec` if present (checking the stored spec), else build it."""
    path = logw_lut_path(out, spec)
    if not path.exists():
        return build_logw_lut(spec, out)
    table = torch.load(path, map_location="cpu")
    if table["spec"] != asdict(spec):
        raise ValueError(f"stored spec in {path} differs from the requested spec: {table['spec']} != {asdict(spec)}")
    logger.info("log-w table loaded: %s (key %s)", path, spec.key())
    return table


def w_physical(raw_w: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """Physical edge sharpness w for raw parameters, as read by the renderer.

    sigmoid(raw) is mapped to a fractional node index over [0, n_w - 1] and interpolated linearly on `ws`.
    raw -> -inf gives ws[0], raw -> +inf gives ws[-1]; raw = 0 gives the middle node (the geometric mean
    of w_min and w_max when nodes are log-spaced and n_w is odd; for even n_w, the arithmetic mean of the
    two middle nodes).
    """
    ws = np.asarray(ws, dtype=np.float64)
    if ws.ndim != 1 or len(ws) < 2:
        raise ValueError("ws must be a 1-D array with at least 2 nodes")
    frac = expit(np.asarray(raw_w, dtype=np.float64))
    return np.interp(frac * (len(ws) - 1), np.arange(len(ws)), ws)
