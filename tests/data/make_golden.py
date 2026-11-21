"""Génère les données de référence (golden) à partir du code NON modifié.

    python tests/data/make_golden.py

Écrit fit_group_golden.pt (fit_group sur 2 cellules synthétiques) et lut_tiny_ref.pt (petite table PDE).
Données générées avec torch 2.15.0 (CPU, macOS) ; la version est aussi stockée dans fit_group_golden.pt["_meta"].
Le test de non-régression compare avec une tolérance de 1e-5 pour absorber le bruit flottant entre versions.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from conftest import build_cell, build_synthetic_lut  # noqa: E402
import variable_source_fit as vsf  # noqa: E402

KINDS = ["gauss", "diffusion", "diff_tau", "diff_w"]
KMAX, ITERS, RNG_SEED = 4, 5, 1
TINY_LUT_ARGS = dict(n=32, steps=60, n_tau=3, n_w=4, n_r=32)
GOLDEN_CELL_BLOBS = [((12.0, 12.0, 6.0, 4.0, 3.0), (6.0, 15.0, 3.0, 2.5, 2.5)),
                     ((10.0, 9.0, 5.0, 3.0, 5.0), (16.0, 16.0, 4.0, 3.0, 3.0))]


def golden_cells():
    return [build_cell(i, blobs) for i, blobs in enumerate(GOLDEN_CELL_BLOBS)]


def run_case(kind, use_rng, lut):
    rng = np.random.default_rng(RNG_SEED) if use_rng else None
    records, _, (params, active) = vsf.fit_group(golden_cells(), kind, KMAX, ITERS, lut[None, None], (), rng)
    return dict(records=records, params=params, active=active)


def build_tiny_pde_lut():
    with tempfile.TemporaryDirectory() as tmp:
        old = vsf.OUT
        vsf.OUT = Path(tmp)
        try:
            return vsf.build_profile_lut(**TINY_LUT_ARGS)
        finally:
            vsf.OUT = old


if __name__ == "__main__":
    torch.manual_seed(0)
    lut = build_synthetic_lut()
    golden = {f"{k}|{int(r)}": run_case(k, r, lut) for k in KINDS for r in (False, True)}
    golden["_meta"] = {"torch": str(torch.__version__)}
    torch.save(golden, HERE / "fit_group_golden.pt")
    torch.save(build_tiny_pde_lut(), HERE / "lut_tiny_ref.pt")
    print("golden written:", sorted(golden))
