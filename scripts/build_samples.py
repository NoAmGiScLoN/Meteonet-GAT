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

With --logw, ONLY the log-spaced-w table of the "diff7" arm is built (flags --w-min --w-max --n-w), as
results/03_variable_sources/diffusion_profiles_logw_<key>.pt; the other inputs are left untouched.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cell_dataset  # noqa: E402
import stability_test  # noqa: E402
import variable_source_fit as vsf  # noqa: E402
from profile_lut import LogWLutSpec, load_or_build_logw_lut, logw_lut_path  # noqa: E402

FILES_2017 = [("NW_rainfall_2017/rainfall-NW-2017-04/rainfall_NW_2017_04.3.npz", "2017-04"),
              ("NW_rainfall_2017/rainfall-NW-2017-09/rainfall_NW_2017_09.2.npz", "2017-09")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = LogWLutSpec()
    ap = argparse.ArgumentParser(description="Rebuild the inputs not stored in git.")
    ap.add_argument("--logw", action="store_true", help="build only the log-w table of the diff7 arm")
    ap.add_argument("--w-min", type=float, default=None, help=f"with --logw (default {d.w_min})")
    ap.add_argument("--w-max", type=float, default=None, help=f"with --logw (default {d.w_max})")
    ap.add_argument("--n-w", type=int, default=None, help=f"with --logw (default {d.n_w})")
    args = ap.parse_args(argv)
    given = [f for f, v in (("--w-min", args.w_min), ("--w-max", args.w_max), ("--n-w", args.n_w)) if v is not None]
    if given and not args.logw:
        ap.error(f"{', '.join(given)} only apply to the log-w table: add --logw")
    return argparse.Namespace(logw=args.logw, w_min=d.w_min if args.w_min is None else args.w_min,
                              w_max=d.w_max if args.w_max is None else args.w_max,
                              n_w=d.n_w if args.n_w is None else args.n_w)


def spec_from_args(args: argparse.Namespace) -> LogWLutSpec:
    return LogWLutSpec(w_min=args.w_min, w_max=args.w_max, n_w=args.n_w)


def build_logw(spec: LogWLutSpec, out: Path) -> Path:
    """Build (or reuse) the log-w table in `out`; returns its path."""
    load_or_build_logw_lut(spec, out)
    return logw_lut_path(out, spec)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.logw:
        print("log-w table:", build_logw(spec_from_args(args), vsf.OUT))
        sys.exit(0)
    if not (cell_dataset.OUT / "cells.npz").exists():
        cell_dataset.build()
    vsf.prepare_sample(40)
    vsf.prepare_sample(40, seed=0, files=FILES_2017, test_only=False, name="sample_40_2017.pkl")
    pairs = vsf.OUT / "stability_pairs_25.pkl"
    if not pairs.exists():
        stability_test.build_pairs(25, pairs)
    vsf.build_profile_lut()
    print("inputs ready in", cell_dataset.OUT, "and", vsf.OUT)
