"""Planchers de bruit de la nrmse en domaine racine utilisee par l'ajusteur.

Un seul plancher est "irreductible" : la quantification du taux (R entier, pas de 0.12 mm/h).
Les autres sont des erreurs limitees par la lissite : borne superieure de ce qu'un modele lisse ne
peut pas representer, pas de l'erreur irreductible.
 - proxies de texture sur l'INTERIEUR erode de la cellule (colonnes *_int{n}) : mesurent la texture ;
 - versions sur le crop entier (colonnes *_full) : "bord + texture", bornes superieures larges
   (le flou dans la marge a zero et le median sur les coins de la cellule y comptent).
"""
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

RATE_STEP = 0.12  # mm/h : R * 12 / 100, R entier
JITTER_HALF = RATE_STEP / 2
SMOOTH_SIGMAS = (0.5, 1.0)
EROSION_ITERS = (1, 2)
MEDIAN_SIZE = 3
FILTER_MODE = "nearest"  # mode unique de tous les filtres
N_MC = 20
SEED = 0
P90 = 0.9
MIN_CELLS_PER_GROUP = 10
ANALYTIC_VAR_DIVISOR = 48.0  # var(sqrt(t+u)) ~ q^2 / (48 t), u ~ U(-q/2, q/2)
# La methode delta surestime pour t petit ; valide pour t >= 9q (tous les pixels de cellule reels).
ANALYTIC_MIN_RATE = 9 * RATE_STEP
REQUIRED_CELL_KEYS = ("tag", "bin", "area", "rate")
CELLS_CSV = "noise_floor_cells.csv"
SUMMARY_CSV = "noise_floor_summary.csv"


def _smooth_col(s: float) -> str:
    return f"smooth_s{s}"


FLOOR_COLUMNS = (
    ["quant_mc", "quant_analytic"]
    + [f"{_smooth_col(s)}_int{n}" for s in SMOOTH_SIGMAS for n in EROSION_ITERS]
    + [f"speckle_int{n}" for n in EROSION_ITERS]
    + [f"{_smooth_col(s)}_full" for s in SMOOTH_SIGMAS]
    + ["speckle_full"]
)


def _as_rate(rate: np.ndarray) -> np.ndarray:
    """Copie float64 validee : ni NaN/inf ni valeur negative."""
    arr = np.asarray(rate, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("rate contains NaN or inf")
    if np.any(arr < 0):
        raise ValueError("rate contains negative values")
    return arr


def nrmse_sqrt(target: np.ndarray, recon: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||sqrt(max(recon,0)) - sqrt(target)||_2 / ||sqrt(target)||_2 (comme vsf.metrics, mask optionnel).

    Pas de EPS ici, comme vsf.metrics ; la loss d'entrainement, elle, utilise EPS = 1e-3."""
    t = _as_rate(target)
    r = np.asarray(recon, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("recon contains NaN or inf")
    if mask is not None:
        t, r = t[mask], r[mask]
    st = np.sqrt(t)
    denom = np.linalg.norm(st)
    if not denom > 0:
        raise ValueError("target has zero norm")
    return float(np.linalg.norm(np.sqrt(np.clip(r, 0, None)) - st) / denom)


def quantization_floor_mc(rate: np.ndarray, rng: np.random.Generator | None = None,
                          n: int = N_MC) -> float:
    """Plancher irreductible : mediane sur n tirages de l'erreur entre le champ quantifie et un
    champ continu plausible (rng par defaut : graine SEED)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(SEED) if rng is None else rng
    rate = _as_rate(rate)
    inside = rate > 0
    vals = []
    for _ in range(n):
        jitter = rng.uniform(-JITTER_HALF, JITTER_HALF, size=rate.shape)
        vals.append(nrmse_sqrt(rate, np.clip(np.where(inside, rate + jitter, 0.0), 0, None)))
    return float(np.median(vals))


def quantization_floor_analytic(rate: np.ndarray, min_rate: float = ANALYTIC_MIN_RATE) -> float:
    """Methode delta : sqrt(sum q^2/(48 t) / sum t) sur les pixels t >= min_rate.

    Hypothese : bruit de quantification uniforme independant, petit devant t. Pour t petit
    (< 9q = ANALYTIC_MIN_RATE) la methode surestime le Monte-Carlo, donc ces pixels sont exclus
    (aucun pixel de cellule reel n'est concerne). min_rate=0 : tous les pixels t > 0."""
    t = _as_rate(rate)
    t = t[(t > 0) & (t >= min_rate)]
    if t.size == 0:
        raise ValueError("rate has no pixel above min_rate")
    return float(np.sqrt(np.sum(RATE_STEP ** 2 / (ANALYTIC_VAR_DIVISOR * t)) / np.sum(t)))


def smoothing_floor_full(rate: np.ndarray, sigma: float) -> float:
    """nrmse entre le crop entier et sa version lissee (mode 'nearest').

    Borne superieure "bord + texture" : le flou dans la marge nulle est compte. Erreur limitee par
    la lissite, pas irreductible."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    rate = _as_rate(rate)
    return nrmse_sqrt(rate, ndi.gaussian_filter(rate, sigma, mode=FILTER_MODE))


def speckle_floor_full(rate: np.ndarray, size: int = MEDIAN_SIZE) -> float:
    """nrmse entre le crop entier et son filtre median (mode 'nearest').

    Borne superieure "bord + texture" (le median rogne les coins de la cellule), pas irreductible."""
    if size < 1:
        raise ValueError("size must be >= 1")
    rate = _as_rate(rate)
    return nrmse_sqrt(rate, ndi.median_filter(rate, size=size, mode=FILTER_MODE))


def interior_mask(rate: np.ndarray, iterations: int) -> np.ndarray:
    """Cellule (rate > 0) erodee de `iterations` pixels."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    return ndi.binary_erosion(_as_rate(rate) > 0, iterations=iterations)


def smoothing_floor_interior(rate: np.ndarray, sigma: float, iterations: int) -> float:
    """nrmse sur l'interieur erode entre le champ et sa version lissee (proxy de texture).

    Lissage par convolution normalisee sur la cellule : smooth(rate*cell)/smooth(cell), de sorte que
    la marge a zero ne fuit pas dans la cellule. Erreur limitee par la lissite (borne superieure de
    ce qu'un modele lisse ne capture pas), pas irreductible. NaN si l'interieur est vide."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    rate = _as_rate(rate)
    interior = interior_mask(rate, iterations)
    if not interior.any():
        return float("nan")
    cell = rate > 0
    num = ndi.gaussian_filter(np.where(cell, rate, 0.0), sigma, mode=FILTER_MODE)
    den = ndi.gaussian_filter(cell.astype(np.float64), sigma, mode=FILTER_MODE)
    smooth = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return nrmse_sqrt(rate, smooth, mask=interior)


def speckle_floor_interior(rate: np.ndarray, iterations: int, size: int = MEDIAN_SIZE) -> float:
    """nrmse sur l'interieur erode entre le champ et son filtre median (proxy de texture).

    Erreur limitee par la lissite, pas irreductible. NaN si l'interieur est vide."""
    if size < 1:
        raise ValueError("size must be >= 1")
    rate = _as_rate(rate)
    interior = interior_mask(rate, iterations)
    if not interior.any():
        return float("nan")
    return nrmse_sqrt(rate, ndi.median_filter(rate, size=size, mode=FILTER_MODE), mask=interior)


def _check_cell(cell: dict) -> None:
    missing = [k for k in REQUIRED_CELL_KEYS if k not in cell]
    if missing:
        raise ValueError(f"cell missing keys {missing}; required {list(REQUIRED_CELL_KEYS)}")


def _cell_floors(rate: np.ndarray, rng: np.random.Generator) -> dict:
    out = dict(quant_mc=quantization_floor_mc(rate, rng),
               quant_analytic=quantization_floor_analytic(rate))
    for n in EROSION_ITERS:
        for s in SMOOTH_SIGMAS:
            out[f"{_smooth_col(s)}_int{n}"] = smoothing_floor_interior(rate, s, n)
        out[f"speckle_int{n}"] = speckle_floor_interior(rate, n)
    for s in SMOOTH_SIGMAS:
        out[f"{_smooth_col(s)}_full"] = smoothing_floor_full(rate, s)
    out["speckle_full"] = speckle_floor_full(rate)
    return out


def floors_table(sample: list[dict], seed: int = SEED, name: str = "sample") -> pd.DataFrame:
    """Un enregistrement par cellule avec tous les planchers (NaN si l'interieur erode est vide)."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in sample:
        _check_cell(cell)
        rows.append(dict(sample=name, tag=cell["tag"], bin=cell["bin"], area=cell["area"],
                         **_cell_floors(cell["rate"], rng)))
    df = pd.DataFrame(rows)
    for n in EROSION_ITERS:
        n_empty = int(df[f"speckle_int{n}"].isna().sum())
        if n_empty:
            log.warning("%s: %d cells with empty interior after erosion %d (NaN floors)",
                        name, n_empty, n)
    return df


def summarize_floors(df: pd.DataFrame) -> pd.DataFrame:
    """Medianes et p90 (NaN ignores) de chaque plancher par (sample, bin)."""
    keys = ["sample", "bin"]
    grouped = df.groupby(keys)[FLOOR_COLUMNS]
    med = grouped.median().add_suffix("_median")
    p90 = grouped.quantile(P90).add_suffix("_p90")
    out = pd.concat([med, p90], axis=1)
    sizes = df.groupby(keys).size()
    out.insert(0, "n_cells", sizes)
    for key, n in sizes[sizes < MIN_CELLS_PER_GROUP].items():
        log.warning("group %s has %d cells (fewer than %d): medians/p90 unreliable",
                    key, n, MIN_CELLS_PER_GROUP)
    return out.reset_index()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/dev/noise_floor"))
    args = ap.parse_args(argv)
    frames = []
    for path in args.samples:
        # Fichiers d'echantillon produits par nos propres scripts (de confiance), jamais externes.
        sample = pickle.loads(path.read_bytes())  # nosec B301
        frames.append(floors_table(sample, SEED, name=path.stem))
    cells = pd.concat(frames, ignore_index=True)
    summary = summarize_floors(cells)
    args.out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out / CELLS_CSV, index=False)
    summary.to_csv(args.out / SUMMARY_CSV, index=False)
    log.info("noise floor summary:\n%s", summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
