"""Rebuild every intermediate input that is NOT stored in git (too large / derived from MeteoNet).

Needs the raw MeteoNet data in DATA/ (see README). Run from the repository root:
    python scripts/build_samples.py

Writes:
  results/02_fixed_vector_benchmark/cells.npz          (11 874 cells, 64x64 patches, for compression_benchmark.py)
  results/03_variable_sources/sample_40.pkl            (160 native-resolution cells, Jan + Jul 2016 test days)
  results/03_variable_sources/sample_40_2017.pkl       (160 cells, Apr + Sep 2017, all days)
  results/03_variable_sources/stability_pairs_25.pkl   (25 pairs per size bin, same cell at t and t + 5 min)
  results/03_variable_sources/diffusion_profiles.pt    (pre-computed diffusion profile table)
Existing files are kept (delete them to force a rebuild).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cell_dataset  # noqa: E402
import stability_test  # noqa: E402
import variable_source_fit as vsf  # noqa: E402

FILES_2017 = [("NW_rainfall_2017/rainfall-NW-2017-04/rainfall_NW_2017_04.3.npz", "2017-04"),
              ("NW_rainfall_2017/rainfall-NW-2017-09/rainfall_NW_2017_09.2.npz", "2017-09")]

if __name__ == "__main__":
    if not (cell_dataset.OUT / "cells.npz").exists():
        cell_dataset.build()
    vsf.prepare_sample(40)
    vsf.prepare_sample(40, seed=0, files=FILES_2017, test_only=False, name="sample_40_2017.pkl")
    pairs = vsf.OUT / "stability_pairs_25.pkl"
    if not pairs.exists():
        stability_test.build_pairs(25, pairs)
    vsf.build_profile_lut()
    print("inputs ready in", cell_dataset.OUT, "and", vsf.OUT)
