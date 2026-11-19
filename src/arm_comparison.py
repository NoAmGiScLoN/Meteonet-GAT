"""Compare fitting arms (e.g. diff7 vs diffusion) from the fits_*.csv / params_*.pkl of one or more runs.

Usage: python src/arm_comparison.py --runs DIR [DIR ...] --out DIR [--logw-key KEY] [--archived DIR]
Writes, per run: <run>_paired_delta.csv, <run>_equivalence.csv and, when params files exist,
<run>_w_pileup.csv and <run>_w_hist.png.  A run missing the diff7 or diffusion arm is skipped with a warning.
--logw-key picks the diffusion_profiles_logw_<key>.pt table when a run directory holds several.
--archived DIR compares the diffusion arm of each run with the archived fits in DIR (e.g.
archive/E2_variable_sources/02_jetson_baseline/results/main) and writes reproduction.csv (all runs, "run" column).
The comparison is on per-(bin, K) medians of nrmse: with --max-per-bin the cell ids differ from the archive,
so paired per-cell differences are not possible.
"""
import argparse
import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.special import expit  # noqa: E402

from profile_lut import LOGW_PREFIX, default_w_nodes, w_physical  # noqa: E402

logger = logging.getLogger(__name__)

EQUIV_TOL = 0.005  # nrmse: |median difference| under which two arms are deemed equivalent
PILEUP_EDGE = 0.03  # sigmoid(w_raw) within this distance of 0 or 1 counts as saturated
PILEUP_FRAC = 0.10  # more than this fraction of saturated sources flags a pile-up
MIN_K_COMPARE = 6  # smaller K are dominated by the initialisation, not compared
W_COL = 7  # raw w column in the 8-column parameter layout
N_BINS_HIST = 30
KEYS = ["bin", "arm", "K"]
DELTA_COLS = ["bin", "K", "med_a", "med_b", "delta", "paired_delta", "n"]


def load_fits(run_dir: Path) -> pd.DataFrame:
    """Concatenate every fits_*.csv of a run directory."""
    files = sorted(Path(run_dir).glob("fits_*.csv"))
    if not files:
        raise ValueError(f"no fits_*.csv in {run_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired_delta(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    """Per (bin, K): medians of nrmse for both arms on the cells fitted by both, and their differences."""
    for arm in (arm_a, arm_b):
        if arm not in set(df.arm):
            raise ValueError(f"arm {arm!r} not found in fits (available: {sorted(set(df.arm))})")
    a = df[df.arm == arm_a][["bin", "K", "cell", "nrmse"]]
    b = df[df.arm == arm_b][["bin", "K", "cell", "nrmse"]]
    both = a.merge(b, on=["bin", "K", "cell"], suffixes=("_a", "_b"))
    both = both.assign(diff=both.nrmse_a - both.nrmse_b)
    grouped = both.groupby(["bin", "K"]).agg(med_a=("nrmse_a", "median"), med_b=("nrmse_b", "median"),
                                             paired_delta=("diff", "median"), n=("cell", "count")).reset_index()
    grouped["delta"] = grouped.med_a - grouped.med_b
    return grouped[DELTA_COLS]


EQUIV_COLUMNS = ("paired_delta", "delta")


def equivalence_table(delta: pd.DataFrame, tol: float, min_k: int,
                      column: str = "paired_delta") -> pd.DataFrame:
    """Per bin: max |column| over K >= min_k (up to the bin cap, i.e. its largest K) and verdict.

    Default column = paired_delta, the median over cells of (arm_a - arm_b) on the same cell (decision of
    checkpoint 1d): it removes cell-to-cell variation, whereas the difference of medians ("delta") moves by up
    to ~0.009 between two runs of the same arm (GPU vs CPU summation order), above the 0.005 tolerance.

    A bin without any K >= min_k gets NaN and passes=False.
    """
    if column not in EQUIV_COLUMNS:
        raise ValueError(f"unknown equivalence column {column!r}; expected one of {EQUIV_COLUMNS}")
    rows = []
    for name, g in delta.groupby("bin", sort=False):
        k_cap = int(g.K.max())
        big = g[g.K >= min_k]
        max_abs = float(big[column].abs().max()) if len(big) else float("nan")
        at_cap = float(g[g.K == k_cap][column].iloc[0])
        rows.append(dict(bin=name, k_cap=k_cap, max_abs_delta=max_abs, delta_at_cap=at_cap,
                         passes=bool(len(big) and max_abs <= tol), criterion=column))
    return pd.DataFrame(rows, columns=["bin", "k_cap", "max_abs_delta", "delta_at_cap", "passes", "criterion"])


def stack_active_params(params: dict[int, np.ndarray], min_cols: int) -> np.ndarray:
    """Concatenate the per-cell (K, P) arrays; validates that they are non-empty and wide enough."""
    rows = [np.asarray(p) for p in params.values() if len(p)]
    if not rows:
        raise ValueError("no source parameters provided")
    stacked = np.concatenate(rows, axis=0)
    if stacked.ndim != 2 or stacked.shape[1] < min_cols:
        raise ValueError(f"parameters need at least {min_cols} columns, got shape {stacked.shape}")
    return stacked


def w_pileup(params: dict[int, np.ndarray], edge: float) -> dict[str, float]:
    """Fractions of sources whose sigmoid(raw w) lies below `edge` / above 1 - edge."""
    if not 0 < edge < 0.5:
        raise ValueError(f"edge must be in (0, 0.5), got {edge}")
    s = expit(stack_active_params(params, W_COL + 1)[:, W_COL])
    return {"frac_low": float((s < edge).mean()), "frac_high": float((s > 1 - edge).mean())}


def baseline_reproduction(df_new: pd.DataFrame, df_archived: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Median nrmse per (bin, arm, K) of a new run vs an archived run; passes = |diff| <= tol."""
    med_new = df_new.groupby(KEYS).nrmse.median().rename("med_new")
    med_arch = df_archived.groupby(KEYS).nrmse.median().rename("med_archived")
    out = pd.concat([med_new, med_arch], axis=1, join="inner").reset_index()
    if out.empty:
        raise ValueError("no common (bin, arm, K) between the two runs")
    out["diff"] = out.med_new - out.med_archived
    out["passes"] = out["diff"].abs() <= tol
    return out


def plot_w_histograms(params_by_arm: dict[str, dict[int, np.ndarray]], ws_by_arm: dict[str, np.ndarray],
                      path: Path) -> None:
    """One histogram per arm of the physical w (log x-axis), read through each arm's table nodes."""
    if not params_by_arm:
        raise ValueError("no arm to plot")
    fig, axes = plt.subplots(1, len(params_by_arm), figsize=(5 * len(params_by_arm), 3.5), squeeze=False)
    for ax, (arm, params) in zip(axes[0], params_by_arm.items()):
        ws = ws_by_arm[arm]
        w = w_physical(stack_active_params(params, W_COL + 1)[:, W_COL], ws)
        ax.hist(w, bins=np.geomspace(ws.min(), ws.max(), N_BINS_HIST))
        ax.set_xscale("log")
        ax.set_xlabel("w")
        ax.set_title(f"{arm} (n={len(w)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_run_params(run_dir: Path, arm: str) -> dict[int, np.ndarray]:
    """Merge the params_<bin>_<arm>.pkl files of a run (empty dict if none)."""
    merged: dict[int, np.ndarray] = {}
    for f in sorted(Path(run_dir).glob(f"params_*_{arm}.pkl")):
        merged.update(pickle.loads(f.read_bytes()))
    return merged


def w_nodes(run_dir: Path, arm: str, logw_key: str | None = None) -> np.ndarray:
    """w nodes read by an arm: log-w table of the run for diff7, default linear axis otherwise.

    Several log-w tables in one run directory are ambiguous: pass `logw_key` to pick one.
    """
    if arm != "diff7":
        return default_w_nodes().numpy()
    tables = sorted(Path(run_dir).glob(f"{LOGW_PREFIX}*.pt"))
    if logw_key is not None:
        tables = [t for t in tables if t.stem == f"{LOGW_PREFIX}{logw_key}"]
        if not tables:
            raise ValueError(f"no {LOGW_PREFIX}{logw_key}.pt in {run_dir}")
    if not tables:
        raise ValueError(f"no {LOGW_PREFIX}*.pt in {run_dir}")
    if len(tables) > 1:
        raise ValueError(f"several log-w tables in {run_dir}, pass --logw-key: {[t.name for t in tables]}")
    return torch.load(tables[0], map_location="cpu")["ws"].numpy()


def has_arms(df: pd.DataFrame, run_dir: Path, arms: tuple[str, ...]) -> bool:
    """True if every arm is in the fits; otherwise logs a warning naming the missing ones."""
    missing = [a for a in arms if a not in set(df.arm)]
    if missing:
        logger.warning("%s: arm(s) %s missing from fits, run skipped", Path(run_dir).name, missing)
    return not missing


def compare_run(run_dir: Path, out: Path, logw_key: str | None = None) -> None:
    """Write the comparison tables (and w diagnostics when params exist) of one run."""
    name = Path(run_dir).name
    df = load_fits(run_dir)
    if not has_arms(df, run_dir, ("diff7", "diffusion")):
        return
    delta = paired_delta(df, "diff7", "diffusion")
    equiv = equivalence_table(delta, EQUIV_TOL, MIN_K_COMPARE)
    delta.to_csv(out / f"{name}_paired_delta.csv", index=False)
    equiv.to_csv(out / f"{name}_equivalence.csv", index=False)
    logger.info("%s\n%s", name, equiv.to_string(index=False))
    params = {arm: p for arm in ("diffusion", "diff7") if (p := load_run_params(run_dir, arm))}
    if not params:
        return
    pile = pd.DataFrame([dict(arm=a, **w_pileup(p, PILEUP_EDGE)) for a, p in params.items()])
    pile["flag"] = (pile.frac_low > PILEUP_FRAC) | (pile.frac_high > PILEUP_FRAC)
    pile.to_csv(out / f"{name}_w_pileup.csv", index=False)
    ws = {a: w_nodes(run_dir, a, logw_key) for a in params}
    plot_w_histograms(params, ws, out / f"{name}_w_hist.png")


def reproduction_table(runs: list[Path], archived_dir: Path) -> pd.DataFrame:
    """Diffusion arm of each run vs the archived fits (per-bin/K medians); one row per (run, bin, K)."""
    archived = load_fits(archived_dir)
    archived = archived[archived.arm == "diffusion"]
    tables = []
    for run in runs:
        df = load_fits(run)
        if not has_arms(df, run, ("diffusion",)):
            continue
        tables.append(baseline_reproduction(df[df.arm == "diffusion"], archived, EQUIV_TOL).assign(run=Path(run).name))
    if not tables:
        raise ValueError("no run with a diffusion arm to compare with the archive")
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (fits_*.csv, params_*.pkl)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--logw-key", default=None, help="key of the log-w table when a run holds several")
    ap.add_argument("--archived", default=None, help="archived fits directory: writes reproduction.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        compare_run(Path(run), out, args.logw_key)
    if args.archived:
        reproduction_table([Path(r) for r in args.runs], Path(args.archived)).to_csv(out / "reproduction.csv", index=False)


if __name__ == "__main__":
    main()
