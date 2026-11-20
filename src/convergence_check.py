"""Convergence en itérations Adam du bras diff7 (ou diffusion) : 200 itérations par vague suffisent-elles ?

Pour chaque classe de taille, les mêmes cellules (tirage reproductible) sont ajustées avec K = cap de la classe
et la planification complète des vagues, à plusieurs niveaux d'itérations. Règle de choix par classe : le plus petit
niveau dont le gain médian apparié (par cellule) de nrmse au K cap vers le niveau suivant est < CONV_TOL ;
sinon le plus grand niveau, avec converged=False.

Écrit convergence.csv, convergence_choice.csv et convergence.png dans --out.
"""
import argparse
import logging
import pickle
import time
import zlib
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import variable_source_fit as vsf  # noqa: E402
from profile_lut import LogWLutSpec  # noqa: E402

ITERS_DEFAULT = (200, 500, 2000)
CONV_TOL = 0.005
MIN_LEVELS = 2
P90 = 0.9
FIG_SIZE = (6, 4)
FIG_DPI = 120
PER_BIN_DEFAULT = 10
SEED_DEFAULT = 0
METRIC_COLUMNS = ["nrmse", "iou", "mass_err", "peak_err"]
RESULT_COLUMNS = ["bin", "cell", "area", "K", "iters"] + METRIC_COLUMNS + ["fit_time_s"]
CHOICE_BASE_COLUMNS = ["bin", "k_cap", "chosen_iters", "converged"]
logger = logging.getLogger(__name__)


def select_cells(sample: list[dict], per_bin: int, seed: int) -> list[dict]:
    """Choisit `per_bin` cellules par classe sans remise.

    Le générateur de chaque classe dépend de (seed, crc32(nom)) : modifier les cellules d'une classe ne change pas
    le tirage des autres. Résultat ordonné par classe (ordre de première apparition) puis par indice d'origine ;
    toutes les cellules sont prises si la classe en compte moins. "id" = indice d'origine dans `sample`.
    Retourne des copies ; `sample` n'est pas modifié.
    """
    if per_bin < 1:
        raise ValueError(f"per_bin doit être >= 1, reçu {per_bin}")
    by_bin: dict[str, list[int]] = {}
    for idx, cell in enumerate(sample):
        by_bin.setdefault(cell["bin"], []).append(idx)
    chosen: list[dict] = []
    for name, indices in by_bin.items():
        if len(indices) <= per_bin:
            picked = indices
        else:
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            picked = sorted(rng.choice(indices, per_bin, replace=False).tolist())
        chosen += [{**sample[i], "id": i} for i in picked]
    return chosen


def validate_levels(iters_list: Sequence[int]) -> list[int]:
    """Niveaux triés et dédoublonnés ; ValueError si moins de 2 niveaux distincts ou un niveau < 1."""
    levels = sorted(set(int(i) for i in iters_list))
    if len(levels) < MIN_LEVELS or levels[0] < 1:
        raise ValueError(f"iters_list doit contenir au moins {MIN_LEVELS} entiers distincts >= 1, reçu "
                         f"{list(iters_list)}")
    return levels


def _fit_levels(group: list[dict], arm: str, kmax: int, levels: Sequence[int],
                lut5: torch.Tensor | None) -> list[dict]:
    """Ajuste `group` (une fois par niveau d'itérations) et retourne une ligne par cellule x K x niveau.

    fit_time_s = durée de l'appel fit_group (tout le groupe, toutes les vagues) divisée à parts égales entre
    les cellules du groupe.
    """
    rows: list[dict] = []
    for iters in levels:
        t0 = time.perf_counter()
        records, _, _ = vsf.fit_group(group, arm, kmax, iters, lut5)
        elapsed = time.perf_counter() - t0
        logger.info("  groupe de %d cellules, %d itérations : %.1f s", len(group), iters, elapsed)
        rows += [{**{k: r[k] for k in ["bin", "cell", "area", "K"] + METRIC_COLUMNS}, "iters": iters,
                  "fit_time_s": elapsed / len(group)} for r in records]
    return rows


def _warm_up(cells: list[dict], arm: str, lut5: torch.Tensor | None) -> None:
    """Ajustement jetable (1 cellule, iters=1, kmax=1) pour exclure l'initialisation paresseuse du chronométrage."""
    if cells:
        vsf.fit_group(cells[:1], arm, 1, 1, lut5)


def _check_bins(cells: list[dict], kmax_by_bin: dict[str, int], group_by_bin: dict[str, int]) -> None:
    unknown = {c["bin"] for c in cells} - set(kmax_by_bin)
    if unknown:
        raise ValueError(f"classes (bin) sans K max : {sorted(unknown)}")
    no_group = set(kmax_by_bin) - set(group_by_bin)
    if no_group:
        raise ValueError(f"classes (bin) sans taille de group_by_bin : {sorted(no_group)}")


def run_convergence(cells: list[dict], arm: str, iters_list: Sequence[int], lut5: torch.Tensor | None,
                    kmax_by_bin: dict[str, int], group_by_bin: dict[str, int],
                    csv_path: Path | str | None = None) -> pd.DataFrame:
    """Ajuste chaque classe à chaque niveau d'itérations (mêmes cellules, mêmes groupes : comparaison appariée).

    Une ligne par cellule x K de la planification (<= cap) x niveau. Si `csv_path` est donné, le CSV complet
    (lignes déjà obtenues) est réécrit après chaque classe. Les classes sans cellule sont signalées et ignorées.
    """
    levels = validate_levels(iters_list)
    _check_bins(cells, kmax_by_bin, group_by_bin)
    _warm_up(cells, arm, lut5)
    if vsf.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict] = []
    t_start = time.perf_counter()
    for name, kmax in kmax_by_bin.items():
        members = [c for c in cells if c["bin"] == name]
        if not members:
            logger.warning("classe %s : aucune cellule dans l'échantillon, ignorée", name)
            continue
        size = group_by_bin[name]
        for g0 in range(0, len(members), size):
            rows += _fit_levels(members[g0:g0 + size], arm, kmax, levels, lut5)
        logger.info("classe %s terminée (%d cellules), %.1f s écoulées", name, len(members),
                    time.perf_counter() - t_start)
        if vsf.DEVICE.type == "cuda":
            logger.info("  pic mémoire CUDA : %.2f Go", torch.cuda.max_memory_allocated() / 1e9)
        if csv_path is not None:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(csv_path, index=False)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _paired_table(sub: pd.DataFrame, name: str, k_cap: int) -> pd.DataFrame:
    table = sub[sub.K == k_cap].pivot(index="cell", columns="iters", values="nrmse")
    missing = table.isna().any(axis=1)
    if missing.any() or table.shape[1] < 2:
        raise ValueError(f"classe {name} : chaque cell doit avoir au moins 2 niveaux d'itérations au K cap "
                         f"{k_cap} (cell incomplètes : {table.index[missing].tolist()})")
    return table


def choose_iters(df: pd.DataFrame, tol: float = CONV_TOL) -> pd.DataFrame:
    """Une ligne par classe : k_cap, chosen_iters, gain_<a>_<b> (médiane par cellule de nrmse(a) - nrmse(b)),
    gain_p90_<a>_<b> (p90 des mêmes gains), converged."""
    rows = []
    for name, sub in df.groupby("bin", sort=False):
        k_cap = int(sub.K.max())
        table = _paired_table(sub, name, k_cap)
        levels = sorted(table.columns)
        pairs = list(zip(levels[:-1], levels[1:]))
        per_cell = {(a, b): table[a] - table[b] for a, b in pairs}
        gains = {f"gain_{a}_{b}": float(d.median()) for (a, b), d in per_cell.items()}
        gains.update({f"gain_p90_{a}_{b}": float(d.quantile(P90)) for (a, b), d in per_cell.items()})
        chosen = next((a for (a, b) in pairs if gains[f"gain_{a}_{b}"] < tol), None)
        converged = chosen is not None
        if not converged:
            chosen = levels[-1]
            logger.warning("classe %s : pas de convergence (gain >= %g à chaque niveau), niveau retenu %d",
                           name, tol, chosen)
        rows.append({"bin": name, "k_cap": k_cap, "chosen_iters": int(chosen), "converged": converged, **gains})
    return pd.DataFrame(rows)


def plot_convergence(df: pd.DataFrame, path: Path | str) -> None:
    """nrmse médian au K cap en fonction du nombre d'itérations, une courbe par classe (axe x log)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, sub in df.groupby("bin", sort=False):
        curve = sub[sub.K == sub.K.max()].groupby("iters").nrmse.median()
        ax.plot(curve.index, curve.values, marker="o", label=f"{name} (K={int(sub.K.max())})")
    ax.set_xscale("log")
    ax.set_xlabel("itérations Adam par vague")
    ax.set_ylabel("nrmse médian au K cap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True, help="échantillon pré-construit (pickle)")
    ap.add_argument("--arm", default="diff7", choices=["diff7", "diffusion"])
    ap.add_argument("--iters-list", type=int, nargs="+", default=list(ITERS_DEFAULT))
    ap.add_argument("--per-bin", type=int, default=PER_BIN_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="graine du tirage des cellules")
    bin_names = [b[0] for b in vsf.BINS]
    ap.add_argument("--bins", nargs="+", default=bin_names, choices=bin_names)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="dossier de sortie (contient aussi les tables de profils)")
    d = LogWLutSpec()
    ap.add_argument("--w-min", type=float, default=d.w_min)
    ap.add_argument("--w-max", type=float, default=d.w_max)
    ap.add_argument("--n-w", type=int, default=d.n_w)
    ap.add_argument("--tau7", type=float, default=d.tau)
    ap.add_argument("--n-r", type=int, default=d.n_r, help="résolution radiale de la table (tests rapides)")
    ap.add_argument("--lut-n", type=int, default=d.n, help="grille de la PDE de la table (tests rapides)")
    ap.add_argument("--lut-steps", type=int, default=d.steps, help="pas de la PDE de la table (tests rapides)")
    return ap


def _run(args: argparse.Namespace, out: Path) -> None:
    levels = validate_levels(args.iters_list)
    spec = LogWLutSpec(w_min=args.w_min, w_max=args.w_max, n_w=args.n_w, tau=args.tau7, n_r=args.n_r,
                       n=args.lut_n, steps=args.lut_steps)
    sample = pickle.loads(Path(args.sample).read_bytes())  # trusted local file (own sample pickle)
    cells = [c for c in select_cells(sample, args.per_bin, args.seed) if c["bin"] in args.bins]
    kmax = {name: k for name, _, _, k in vsf.BINS if name in args.bins}
    if args.arm == "diffusion" and not (out / "diffusion_profiles.pt").exists():
        logger.info("table 8 paramètres absente de %s : construite par la PDE", out)
    lut5 = vsf.lut_for_arm(args.arm, out, spec)
    logger.info("%d cellules, bras %s, niveaux %s, clé de table %s", len(cells), args.arm, levels, spec.key())
    df = run_convergence(cells, args.arm, levels, lut5, kmax, vsf.GROUP, csv_path=out / "convergence.csv")
    df.to_csv(out / "convergence.csv", index=False)  # avant choose_iters : ne jamais perdre les ajustements
    choice = choose_iters(df)
    choice.to_csv(out / "convergence_choice.csv", index=False)
    plot_convergence(df, out / "convergence.png")
    logger.info("\n%s", choice.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_levels(args.iters_list)  # échoue avant tout calcul
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous_device = vsf.DEVICE
    vsf.DEVICE = torch.device(args.device)
    try:
        _run(args, out)
    finally:
        vsf.DEVICE = previous_device


if __name__ == "__main__":
    main()
