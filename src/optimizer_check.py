"""Réglages de l'optimiseur pour le fit diff7 : lr constant ou cosinus, 200 ou 500 itérations ?

Les mêmes cellules (tirage reproductible) et les mêmes groupes sont ajustés avec K = cap de la classe, pour chaque
(réglage, itérations) du plan. Règle d'adoption par classe, au K cap, gain apparié par cellule = nrmse(baseline) -
nrmse(réglage) avec baseline = lr constant 0.05 à ITERS_BY_BIN[classe] : ADOPT si, à nombre d'itérations égal à la
baseline, médiane >= GAIN_MIN et 10e centile >= -TAIL_TOL ; cheap_match si, à 200 itérations, un réglage est à
+-CHEAP_TOL (médiane) de la baseline à 500 itérations (classes >= 1000 px). Le pile-up de w est rapporté, pas un critère.
Le 10e centile est calculé par interpolation linéaire entre statistiques d'ordre (pandas quantile, méthode "linear").

Écrit optimizer_grid.csv, optimizer_pileup.csv, optimizer_choice.csv, optimizer_grid.png et params_grid.pkl.
"""
import argparse
import logging
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import arm_comparison  # noqa: E402
import variable_source_fit as vsf  # noqa: E402
from convergence_check import METRIC_COLUMNS, PER_BIN_DEFAULT, SEED_DEFAULT, select_cells  # noqa: E402
from convergence_check import _check_bins as check_bins  # noqa: E402  (convergence_check n'expose pas d'alias public)
from optim_schedule import validate_lr  # noqa: E402
from profile_lut import LogWLutSpec  # noqa: E402

ARM = "diff7"
PLAN_DEFAULT = ("fixed:0.05@200,500", "cosine:0.05@200,500", "cosine:0.1@200")
SCHEDULE_BY_KIND = {"fixed": None, "cosine": "cosine"}
BASELINE_SETTING = "fixed:0.05"
GAIN_MIN = 0.005
TAIL_TOL = 0.002
CHEAP_TOL = 0.002
CHEAP_ITERS = 200
CHEAP_REF_ITERS = 500
CHEAP_MIN_AREA = 1000
P10 = 0.1
EPS = 1e-12
FIG_SIZE = (8, 4.5)
FIG_DPI = 120
RESULT_COLUMNS = ["bin", "cell", "area", "K", "iters", "setting", "lr", "schedule"] + METRIC_COLUMNS + ["fit_time_s"]
PILEUP_COLUMNS = ["setting", "iters", "bin", "frac_low", "frac_high", "n_sources"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptSetting:
    name: str
    lr: float
    schedule: str | None


# ------------------------------------------------------------------ plan
def parse_plan_entry(entry: str) -> list[tuple[OptSetting, int]]:
    """'cosine:0.05@200,500' -> [(OptSetting('cosine:0.05', 0.05, 'cosine'), 200), (..., 500)]."""
    head, sep, tail = entry.partition("@")
    kind, colon, lr_text = head.partition(":")
    if not sep or not colon or kind not in SCHEDULE_BY_KIND:
        raise ValueError(f"entrée de plan invalide {entry!r} (attendu <{'|'.join(SCHEDULE_BY_KIND)}>:<lr>@<it>[,<it>])")
    try:
        lr = float(lr_text)
        iters = [int(t) for t in tail.split(",")]
    except ValueError:
        raise ValueError(f"entrée de plan invalide {entry!r} : lr réel et itérations entières attendus") from None
    validate_lr(lr, SCHEDULE_BY_KIND[kind])
    if min(iters) < 1 or len(set(iters)) != len(iters):
        raise ValueError(f"entrée de plan {entry!r} : itérations >= 1 et distinctes attendues, reçu {iters}")
    setting = OptSetting(f"{kind}:{lr:g}", lr, SCHEDULE_BY_KIND[kind])
    return [(setting, it) for it in iters]


def parse_plan(plan: Sequence[str]) -> list[tuple[OptSetting, int]]:
    """Plan complet ; ValueError si vide ou si un (réglage, itérations) apparaît deux fois."""
    items = [item for entry in plan for item in parse_plan_entry(entry)]
    if not items:
        raise ValueError("plan vide")
    keys = [(s.name, it) for s, it in items]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"plan : (réglage, itérations) dupliqués {dup}")
    return items


def check_baseline_in_plan(items: Sequence[tuple[OptSetting, int]], baseline: tuple[str, dict[str, int]],
                           bins: Sequence[str]) -> None:
    """ValueError si la baseline (réglage, itérations de chaque classe) n'est pas dans le plan."""
    present = {(s.name, it) for s, it in items}
    missing = sorted({(baseline[0], baseline[1][b]) for b in bins} - present)
    if missing:
        raise ValueError(f"baseline absente du plan : {missing}")


# ------------------------------------------------------------------ ajustement
def _fit_setting(group: list[dict], setting: OptSetting, iters: int, kmax: int,
                 lut5: torch.Tensor | None) -> tuple[list[dict], dict[int, np.ndarray]]:
    t0 = time.perf_counter()
    records, _, (prm, act) = vsf.fit_group(group, ARM, kmax, iters, lut5, lr=setting.lr,
                                           lr_schedule=setting.schedule)
    elapsed = time.perf_counter() - t0
    logger.info("  %s @%d, groupe de %d cellules : %.1f s", setting.name, iters, len(group), elapsed)
    rows = [{**{k: r[k] for k in ["bin", "cell", "area", "K"] + METRIC_COLUMNS}, "iters": iters,
             "setting": setting.name, "lr": setting.lr, "schedule": setting.schedule or "fixed",
             "fit_time_s": elapsed / len(group)} for r in records]
    return rows, vsf.params_by_cell(group, prm, act)


def run_grid(cells: list[dict], plan: Sequence[str], lut5: torch.Tensor | None, kmax_by_bin: dict[str, int],
             group_by_bin: dict[str, int], csv_path: Path | str | None = None,
             params_path: Path | str | None = None
             ) -> tuple[pd.DataFrame, dict[tuple[str, int, str], dict[int, np.ndarray]]]:
    """Ajuste chaque classe pour chaque (réglage, itérations) : mêmes cellules, mêmes groupes (appariement).

    Retourne (lignes par cellule x K x réglage, params[(réglage, itérations, classe)] = {cellule : (K, P)}).
    Si `csv_path` est donné, le CSV complet est réécrit après chaque ajustement (groupe, réglage, itérations) ;
    si `params_path` est donné, les paramètres sont sérialisés (pickle) après chaque classe.
    """
    items = parse_plan(plan)
    check_bins(cells, kmax_by_bin, group_by_bin)
    if cells:
        vsf.fit_group(cells[:1], ARM, 1, 1, lut5)  # chauffe : exclut l'initialisation paresseuse du chronométrage
    rows: list[dict] = []
    params: dict[tuple[str, int, str], dict[int, np.ndarray]] = {}
    t_start = time.perf_counter()
    for name, kmax in kmax_by_bin.items():
        members = [c for c in cells if c["bin"] == name]
        if not members:
            logger.warning("classe %s : aucune cellule dans l'échantillon, ignorée", name)
            continue
        if vsf.DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()  # le pic loggé est celui de la classe
        for g0 in range(0, len(members), group_by_bin[name]):
            group = members[g0:g0 + group_by_bin[name]]
            for setting, iters in items:
                new_rows, saved = _fit_setting(group, setting, iters, kmax, lut5)
                rows += new_rows
                params.setdefault((setting.name, iters, name), {}).update(saved)
                if csv_path is not None:
                    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(csv_path, index=False)
        logger.info("classe %s terminée (%d cellules), %.1f s écoulées", name, len(members),
                    time.perf_counter() - t_start)
        if vsf.DEVICE.type == "cuda":
            logger.info("  pic mémoire CUDA : %.2f Go", torch.cuda.max_memory_allocated() / 1e9)
        if params_path is not None:
            Path(params_path).write_bytes(pickle.dumps(params))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), params


# ------------------------------------------------------------------ tables
def pileup_table(params: dict[tuple[str, int, str], dict[int, np.ndarray]]) -> pd.DataFrame:
    """Fraction de sources w saturées (bas / haut) par (réglage, itérations, classe)."""
    rows = []
    for (setting, iters, name), by_cell in params.items():
        frac = arm_comparison.w_pileup(by_cell, arm_comparison.PILEUP_EDGE)
        rows.append({"setting": setting, "iters": iters, "bin": name, **frac,
                     "n_sources": int(sum(len(p) for p in by_cell.values()))})
    return pd.DataFrame(rows, columns=PILEUP_COLUMNS)


def _cap_table(sub: pd.DataFrame, name: str, baseline_key: tuple[str, int]) -> pd.DataFrame:
    """nrmse au K cap, une ligne par cellule, une colonne par (réglage, itérations).

    ValueError ("NaN ou cellule manquante") si une cellule manque dans une colonne ou si la baseline est NaN ;
    un NaN dans une autre colonne est conservé (réglage non adoptable, voir choice_table).
    """
    cap = sub[sub.K == sub.K.max()]
    index = ["cell", "setting", "iters"]
    table = cap.set_index(index).nrmse.unstack(["setting", "iters"])
    present = cap.assign(_p=1).set_index(index)["_p"].unstack(["setting", "iters"]).notna()
    bad = ~present.all(axis=1)
    if baseline_key in table.columns:
        bad = bad | table[baseline_key].isna()
    if bad.any():
        raise ValueError(f"classe {name} : NaN ou cellule manquante, cell sans toutes les colonnes valides : "
                         f"{table.index[bad].tolist()}")
    return table


def _bin_min_areas() -> dict[str, int]:
    return {n: lo for n, lo, _, _ in vsf.BINS}


def choice_table(df: pd.DataFrame, baseline: tuple[str, dict[str, int]] | None = None,
                 gain_min: float = GAIN_MIN, tail_tol: float = TAIL_TOL,
                 cheap_tol: float = CHEAP_TOL) -> pd.DataFrame:
    """Une ligne par classe x (réglage, itérations) : median_gain, p10_gain (baseline - réglage, par cellule au K
    cap), adopt, cheap_match, baseline_iters, recommended (un seul par classe : meilleure adoption, sinon la
    baseline). Baseline par défaut : (BASELINE_SETTING, vsf.ITERS_BY_BIN). Un réglage avec un NaN au K cap n'est
    jamais adopté (journalisé)."""
    base_name, base_iters = baseline if baseline is not None else (BASELINE_SETTING, vsf.ITERS_BY_BIN)
    min_area = _bin_min_areas()
    rows = []
    for name, sub in df.groupby("bin", sort=False):
        key = (base_name, base_iters[name])
        table = _cap_table(sub, name, key)
        if key not in table.columns:
            raise ValueError(f"classe {name} : baseline {key} absente de la grille")
        ref500 = (base_name, CHEAP_REF_ITERS)
        is_large = min_area.get(name, 0) >= CHEAP_MIN_AREA
        large = is_large and ref500 in table.columns and not table[ref500].isna().any()
        if is_large and not large:
            logger.warning("classe %s : cheap_match non évaluable, %s@%d absent (ou NaN) du plan", name, base_name,
                           CHEAP_REF_ITERS)
        bin_rows = []
        for col in table.columns:
            has_nan = bool(table[col].isna().any())
            if has_nan:
                logger.warning("classe %s, %s@%d : NaN au K cap, réglage non adoptable", name, *col)
            gain = table[key] - table[col]
            med, p10 = float(gain.median()), float(gain.quantile(P10))
            adopt = (not has_nan and col != key and col[1] == key[1]
                     and med >= gain_min - EPS and p10 >= -tail_tol - EPS)
            cheap = False
            if large and not has_nan and col[1] == CHEAP_ITERS:
                cheap = abs(float((table[ref500] - table[col]).median())) <= cheap_tol + EPS
            bin_rows.append({"bin": name, "k_cap": int(sub.K.max()), "setting": col[0], "iters": int(col[1]),
                             "median_gain": med, "p10_gain": p10, "adopt": bool(adopt),
                             "cheap_match": bool(cheap), "baseline_iters": int(key[1])})
        adopted = [r for r in bin_rows if r["adopt"]]
        best = max(adopted, key=lambda r: r["median_gain"]) if adopted else \
            next(r for r in bin_rows if (r["setting"], r["iters"]) == key)
        best_id = (best["setting"], best["iters"])
        rows += [{**r, "recommended": (r["setting"], r["iters"]) == best_id} for r in bin_rows]
    return pd.DataFrame(rows)


def plot_grid(df: pd.DataFrame, path: Path | str) -> None:
    """nrmse médian au K cap, par classe, en fonction du (réglage, itérations) : marqueurs seuls, abscisses = rang
    du libellé trié (une classe sans une combinaison garde les bonnes positions)."""
    labels = sorted({f"{s}@{i}" for s, i in zip(df.setting, df.iters)})
    position = {label: idx for idx, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, sub in df.groupby("bin", sort=False):
        cap = sub[sub.K == sub.K.max()].groupby(["setting", "iters"]).nrmse.median()
        ax.plot([position[f"{s}@{i}"] for s, i in cap.index], cap.values, linestyle="none", marker="o",
                label=f"{name} (K={int(sub.K.max())})")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("nrmse médian au K cap")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def warn_short_bins(cells: list[dict], kmax_by_bin: dict[str, int], per_bin: int) -> None:
    """Avertit si une classe demandée a moins de `per_bin` cellules dans l'échantillon."""
    for name in kmax_by_bin:
        n = sum(1 for c in cells if c["bin"] == name)
        if n < per_bin:
            logger.warning("classe %s : %d cellule(s) au lieu de %d demandées", name, n, per_bin)


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True, help="échantillon pré-construit (pickle)")
    ap.add_argument("--plan", nargs="+", default=list(PLAN_DEFAULT), help="entrées <fixed|cosine>:<lr>@<it>[,<it>]")
    ap.add_argument("--baseline-iters", type=int, default=None,
                    help="itérations de la baseline pour toutes les classes (défaut : ITERS_BY_BIN ; tests rapides)")
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


def _baseline(args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    if args.baseline_iters is None:
        return BASELINE_SETTING, vsf.ITERS_BY_BIN
    return BASELINE_SETTING, {b: args.baseline_iters for b in vsf.ITERS_BY_BIN}


def _run(args: argparse.Namespace, out: Path, baseline: tuple[str, dict[str, int]]) -> None:
    spec = LogWLutSpec(w_min=args.w_min, w_max=args.w_max, n_w=args.n_w, tau=args.tau7, n_r=args.n_r,
                       n=args.lut_n, steps=args.lut_steps)
    sample = pickle.loads(Path(args.sample).read_bytes())  # nosec B301  trusted local file (own sample pickle)
    cells = [c for c in select_cells(sample, args.per_bin, args.seed) if c["bin"] in args.bins]
    kmax = {name: k for name, _, _, k in vsf.BINS if name in args.bins}
    warn_short_bins(cells, kmax, args.per_bin)
    lut5 = vsf.lut_for_arm(ARM, out, spec)
    logger.info("%d cellules, plan %s, clé de table %s", len(cells), list(args.plan), spec.key())
    df, params = run_grid(cells, args.plan, lut5, kmax, vsf.GROUP, csv_path=out / "optimizer_grid.csv",
                       params_path=out / "params_grid.pkl")
    df.to_csv(out / "optimizer_grid.csv", index=False)  # avant les analyses : ne jamais perdre les ajustements
    (out / "params_grid.pkl").write_bytes(pickle.dumps(params))
    pileup = pileup_table(params)
    pileup.to_csv(out / "optimizer_pileup.csv", index=False)
    choice = choice_table(df, baseline)
    choice.to_csv(out / "optimizer_choice.csv", index=False)
    plot_grid(df, out / "optimizer_grid.png")
    logger.info("\n%s\n\n%s", choice.to_string(index=False), pileup.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.baseline_iters is not None and args.baseline_iters < 1:
        raise ValueError(f"--baseline-iters doit être >= 1, reçu {args.baseline_iters}")
    baseline = _baseline(args)
    check_baseline_in_plan(parse_plan(args.plan), baseline, args.bins)  # échoue avant tout calcul
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous_device = vsf.DEVICE
    vsf.DEVICE = torch.device(args.device)
    try:
        _run(args, out, baseline)
    finally:
        vsf.DEVICE = previous_device


if __name__ == "__main__":
    main()
