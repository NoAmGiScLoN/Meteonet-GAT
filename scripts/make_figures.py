"""Generate every figure used in README.md from the result files committed in archive/*/results/.

New runs write to results/; copy their outputs into archive/ (with a report.md) before regenerating figures.

Usage (from the repository root):  python scripts/make_figures.py
Needs numpy, pandas, matplotlib. The raw MeteoNet data in DATA/ is only needed for the radar-map figure
(skipped with a message if the data is absent).
"""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
# Sorties committées des expériences passées, archivées (voir archive/)
E0_RES = ROOT / "archive" / "E0_segmentation_tracking" / "results"
E1_RES = ROOT / "archive" / "E1_fixed_vectors" / "results"
E2_JETSON_RES = ROOT / "archive" / "E2_variable_sources" / "02_jetson_baseline" / "results"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BINS = ["50-200", "200-1000", "1000-5000", ">=5000"]
BIN_LABEL = {"50-200": "50–200 px", "200-1000": "200–1 000 px", "1000-5000": "1 000–5 000 px", ">=5000": "≥ 5 000 px"}
# fixed categorical order (colour follows the entity in every figure)
COL = {"gauss": "#2a78d6", "diffusion": "#eb6834", "ae": "#1baf7a", "pca": "#eda100", "zernike": "#e87ba4",
       "diff_tau": "#6250d6", "diff_w": "#008300", "mean": "#8a8984"}
NAME = {"gauss": "Gaussian sources (6 par.)", "diffusion": "Diffusion sources (8 par.)",
        "diff_tau": "Diffusion, only τ free (7 par.)", "diff_w": "Diffusion, only w free (7 par.)",
        "ae": "CNN autoencoder", "pca": "PCA", "zernike": "Zernike", "gmm": "Gaussian decoder (CNN)",
        "mean": "Mean patch"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def load_fits(run):
    return pd.concat([pd.read_csv(p) for p in sorted((E2_JETSON_RES / run).glob("fits_*.csv"))])


def median_curve(df, arm, b, x="n_params", y="nrmse"):
    d = df[(df.arm == arm) & (df.bin == b)]
    return d.groupby(x)[y].median()


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


# ---------------------------------------------------------------- 1. radar map + 1 mm/h cells
def fig_radar_map():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from datetime import datetime
    try:
        from cell_delimitation import load_frames, segment, MISSING
        data, dates = load_frames(datetime(2016, 1, 1, 12, 0), 1)
    except Exception as e:  # raw data absent
        print("skip radar map:", e)
        return
    R = data[0].astype(float)
    rate = np.where(R > 0, R * 12 / 100, np.nan)
    labels, n, valid = segment(data[0])
    area = np.bincount(labels.ravel())
    keep = np.isin(labels, np.nonzero(area >= 50)[0][1:])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax in axs:
        ax.set_facecolor("white")
        ax.imshow(np.where(valid, np.nan, 1), cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    im = axs[0].imshow(rate, cmap="Blues", norm=LogNorm(0.1, 30), interpolation="nearest")
    axs[0].set_title("MeteoNet NW rain rate, 2016-01-01 12:00 (grey = no radar coverage)")
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="rain rate (mm/h)")
    axs[1].imshow(np.where(rate >= 1, rate, np.nan), cmap="Blues", norm=LogNorm(0.1, 30), interpolation="nearest")
    axs[1].contour(keep, levels=[0.5], colors=[COL["diffusion"]], linewidths=0.8)
    axs[1].set_title(f"Pixels ≥ 1 mm/h; orange outline = kept cells (8-connected, ≥ 50 px)")
    save(fig, "01_radar_and_cells.png")


# ---------------------------------------------------------------- 2. fixed-vector benchmark
def fig_fixed_benchmark():
    r = json.loads((E1_RES / "results.json").read_text())
    rj = json.loads((E1_RES / "results_jetson.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    arms = [("pca", "pca"), ("zernike", "zernike"), ("gmm", "gauss"), ("diffusion", "diffusion"), ("ae", "ae")]
    for key, metric, ax in [("nrmse", "nrmse", axs[0]), ("iou", "iou", axs[1])]:
        for arm, c in arms:
            pts = sorted((v["dim"], v["all"][metric]) for k, v in r.items() if v["arm"] == arm and
                         not (arm == "gmm" and v["dim"] == 48))
            lab = NAME[arm] if arm != "diffusion" else "Diffusion decoder (CNN)"
            ax.plot(*zip(*pts), "o-", color=COL[c], label=lab)
        ae60 = rj["ae_48"]["all"][metric]
        ax.plot([48], [ae60], marker="*", ms=11, ls="none", color=COL["ae"], label="CNN autoencoder, 60 epochs (Jetson)")
        ax.axhline(r["mean_0"]["all"][metric], color=COL["mean"], lw=1, ls="--", label="Mean patch (d = 0)")
        ax.set_xlabel("vector dimension d"); ax.set_xticks([6, 12, 18, 30, 48])
    axs[0].set_ylabel("nrmse on √rate (lower is better)"); axs[0].set_title("Reconstruction error")
    axs[1].set_ylabel("footprint IoU at 1 mm/h (higher is better)"); axs[1].set_title("Footprint overlap")
    axs[0].annotate("Gaussian d = 48 collapsed\n(output ≡ 0), not shown", xy=(47, 0.62), ha="right", fontsize=8, color=MUTED)
    axs[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Step 1 — fixed-length vectors (32×32 patches, 2 888 test cells)", fontweight="bold")
    save(fig, "02_fixed_vector_benchmark.png")


# ---------------------------------------------------------------- 3. variable sources: rate-distortion
def fig_rate_distortion(run="main", name="03_variable_sources_rate_distortion.png", title=None):
    df = load_fits(run)
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.8), constrained_layout=True, sharex="col")
    for j, b in enumerate(BINS):
        for i, y in enumerate(["nrmse", "iou"]):
            ax = axs[i, j]
            for arm in ["gauss", "diffusion"]:
                s = median_curve(df, arm, b, y=y)
                ax.plot(s.index, s.values, "o-", color=COL[arm], label=NAME[arm])
            ax.set_xscale("log", base=2)
            if y == "nrmse":
                ax.axhline(0.15, color=MUTED, lw=0.8, ls=":")
                ax.set_title(BIN_LABEL[b])
            else:
                ax.set_xlabel("parameters per cell")
        axs[0, j].set_ylim(0, None)
    axs[0, 0].set_ylabel("median nrmse (√rate)"); axs[1, 0].set_ylabel("median IoU at 1 mm/h")
    axs[0, 0].text(axs[0, 0].get_xlim()[0] * 1.1, 0.16, "target 0.15", fontsize=7, color=MUTED)
    axs[1, 3].legend(loc="lower right", fontsize=8)
    fig.suptitle(title or "Step 2 — variable number of sources at native resolution (160 cells, Jan + Jul 2016)",
                 fontweight="bold")
    save(fig, name)


# ---------------------------------------------------------------- 4. reconstructions
def fig_snapshots():
    s = pickle.loads((E2_JETSON_RES / "main" / "snapshots.pkl").read_bytes())
    pick = {"50-200": (8, 6), "200-1000": (16, 12), "1000-5000": (16, 12), ">=5000": (16, 12)}
    fig, axs = plt.subplots(4, 3, figsize=(10, 11), constrained_layout=True)
    for i, b in enumerate(BINS):
        tgt = s[(b, "target")]
        kg, kd = pick[b]
        imgs = [(tgt, "radar (target)"), (s[(b, "gauss", kg)], f"{kg} Gaussian sources = {6 * kg} par."),
                (s[(b, "diffusion", kd)], f"{kd} diffusion sources = {8 * kd} par.")]
        vmax = np.nanmax(tgt)
        for j, (img, t) in enumerate(imgs):
            ax = axs[i, j]
            ax.imshow(img, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(tgt >= 1, levels=[0.5], colors=[COL["diffusion"] if j else INK], linewidths=0.7)
            if j:
                ax.contour(img >= 1, levels=[0.5], colors=[INK], linewidths=0.7, linestyles="--")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{BIN_LABEL[b]} — {t}", fontsize=8, fontweight="normal")
    fig.suptitle("Same parameter budget, Gaussian vs diffusion sources\n"
                 "(solid = true 1 mm/h edge, dashed = reconstructed edge)", fontweight="bold")
    save(fig, "04_reconstructions.png")


# ---------------------------------------------------------------- 5. ablation
def fig_ablation():
    main, abl = load_fits("main"), load_fits("ablation")
    df = pd.concat([main, abl])
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, b in zip(axs, BINS):
        for arm in ["gauss", "diff_tau", "diff_w", "diffusion"]:
            s = median_curve(df, arm, b, x="K")
            ax.plot(s.index, s.values, "o-", color=COL[arm], label=NAME[arm], lw=1.6)
        ax.set_xscale("log", base=2); ax.set_title(BIN_LABEL[b]); ax.set_xlabel("number of sources K")
        ax.set_ylim(0, None)
    axs[0].set_ylabel("median nrmse (√rate)")
    axs[3].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Step 3 — ablation: which diffusion parameter carries the gain? (same K)", fontweight="bold")
    save(fig, "05_ablation_tau_w.png")


# ---------------------------------------------------------------- 6. season 2017 vs 2016
def fig_season():
    a, b17 = load_fits("main"), load_fits("season2017")
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, b in zip(axs, BINS):
        for df, ls, yr in [(a, "-", "2016 (Jan+Jul)"), (b17, "--", "2017 (Apr+Sep)")]:
            for arm in ["gauss", "diffusion"]:
                s = median_curve(df, arm, b)
                ax.plot(s.index, s.values, ls, marker="o" if ls == "-" else "s", color=COL[arm],
                        label=f"{NAME[arm].split(' (')[0]}, {yr}", lw=1.6, alpha=1 if ls == "-" else 0.75)
        ax.set_xscale("log", base=2); ax.set_title(BIN_LABEL[b]); ax.set_xlabel("parameters per cell")
        ax.set_ylim(0, None)
    axs[0].set_ylabel("median nrmse (√rate)")
    axs[3].legend(fontsize=7.5)
    fig.suptitle("Step 4 — another season and year: the ranking holds, margins shrink", fontweight="bold")
    save(fig, "06_season_2017.png")


# ---------------------------------------------------------------- 7. restarts
def fig_restart():
    a, r = load_fits("main"), load_fits("restart_seed1")
    m = a.merge(r, on=["cell", "bin", "arm", "K"], suffixes=("_s0", "_s1"))
    m = m[m.K == m.groupby(["bin", "arm"]).K.transform("max")]
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    for arm in ["gauss", "diffusion"]:
        d = m[m.arm == arm]
        ax.scatter(d.nrmse_s0, d.nrmse_s1, s=14, color=COL[arm], label=NAME[arm], edgecolor="white", lw=0.4)
    lim = [0, max(m.nrmse_s0.max(), m.nrmse_s1.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("nrmse, original initialisation"); ax.set_ylabel("nrmse, perturbed initialisation (seed 1)")
    ax.set_title("Step 5 — restarts (each cell at its max K)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "07_restarts.png")


# ---------------------------------------------------------------- 8. stability
def fig_stability():
    st = pd.read_csv(E2_JETSON_RES / "stability" / "stability.csv")
    st["ratio"] = st.cost_same / st.cost_other
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, b in enumerate(BINS):
        for k, arm in enumerate(["gauss", "diffusion"]):
            v = st[(st.bin == b) & (st.arm == arm)].ratio.values
            x = i + (k - 0.5) * 0.32
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=10, color=COL[arm], alpha=0.55, edgecolor="none",
                       label=NAME[arm] if i == 0 else None)
            ax.plot([x - 0.12, x + 0.12], [np.median(v)] * 2, color=INK, lw=2)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(3.45, 1.02, "same cell no closer than another cell", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(range(4), [f"{BIN_LABEL[b]}\nK = {k}" for b, k in zip(BINS, [3, 6, 12, 24])])
    ax.set_ylabel("cost(same cell, t→t+5 min)\n÷ cost(different cell)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Step 6 — temporal stability of the source sets (black bar = median, lower = more stable)")
    save(fig, "08_stability.png")


# ---------------------------------------------------------------- 9. params needed
def fig_params_needed():
    df = load_fits("main")
    rows = []
    for b in BINS:
        for arm in ["gauss", "diffusion"]:
            d = df[(df.bin == b) & (df.arm == arm)]
            ok = d[d.nrmse <= 0.15].groupby("cell").n_params.min()
            n = d.cell.nunique()
            rows.append((b, arm, ok.median() if len(ok) else np.nan, len(ok) / n))
    t = pd.DataFrame(rows, columns=["bin", "arm", "median_params", "frac"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
    w = 0.36
    for k, arm in enumerate(["gauss", "diffusion"]):
        d = t[t.arm == arm]
        x = np.arange(4) + (k - 0.5) * w * 1.08
        bars = ax.bar(x, d.median_params, w, color=COL[arm], label=NAME[arm])
        for xi, (p, f) in zip(x, zip(d.median_params, d.frac)):
            ax.text(xi, (0 if np.isnan(p) else p) + 8, f"{f:.0%}\nof cells", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(4), [BIN_LABEL[b] for b in BINS]); ax.set_ylabel("median parameters")
    ax.set_title("Parameters needed to reach nrmse ≤ 0.15 (among cells that reach it)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, t.median_params.max() * 1.3)
    save(fig, "09_params_needed.png")
    return t


def copy_existing():
    for src, dst in [(E0_RES / "S2_overlay.png", "01b_segmentation_overlay.png"),
                     (E0_RES / "T1_tracks.png", "01c_tracking.png"),
                     (E1_RES / "cnn_encoder_schematic.png", "02b_cnn_encoder.png")]:
        shutil.copy(src, FIG / dst)
        print("copied", FIG / dst)


if __name__ == "__main__":
    fig_radar_map()
    copy_existing()
    fig_fixed_benchmark()
    fig_rate_distortion()
    fig_snapshots()
    fig_ablation()
    fig_season()
    fig_restart()
    fig_stability()
    print(fig_params_needed())
