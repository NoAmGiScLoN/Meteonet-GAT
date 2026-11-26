"""Overview figure of the pipeline, built from real data: radar map -> 1 mm/h cells -> one cell at native
resolution -> fitted diffusion sources -> node tensor -> graph (cells at t and t + 5 min).

Usage (from the repository root):  python scripts/pipeline_figure.py
Needs DATA/ (MeteoNet), results/03_variable_sources/diffusion_profiles.pt (scripts/build_samples.py) and
archive/E0_segmentation_tracking/results/T1_detections.csv. Writes docs/figures/00_pipeline.png.
"""
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import variable_source_fit as vsf  # noqa: E402
from cell_delimitation import load_frames, segment, MIN_RATE_MM_H  # noqa: E402
from cell_dataset import MIN_AREA, raw_to_rate  # noqa: E402

START = datetime(2016, 1, 1, 12, 0)
CELL_AREA_RANGE = (1000, 5000)  # a mid-sized cell: complex enough to need several sources
K_MAX = 12                      # sources fitted for the example cell
FIT_ITERS = 200
INFLUENCE_RADIUS_PX = 200       # illustration: candidate influence edge if centroids are closer than this
PX_KM2 = 0.8
DETECTIONS = ROOT / "archive" / "E0_segmentation_tracking" / "results" / "T1_detections.csv"
OUT = ROOT / "docs" / "figures" / "00_pipeline.png"
INK, MUTED, ACCENT, ACCENT2, NODE = "#0b0b0b", "#52514e", "#eb6834", "#2a78d6", "#9fb8d9"


def rate_map(R):
    """Taux (mm/h), NaN hors pluie ; conversion int32 (int16 déborde au-delà de 2730)."""
    return np.where(R > 0, raw_to_rate(R), np.nan)


def softplus(v):
    return np.log1p(np.exp(v))


def pick_cell(labels: np.ndarray, n: int) -> int:
    area = np.bincount(labels.ravel(), minlength=n + 1)
    ok = np.nonzero((area >= CELL_AREA_RANGE[0]) & (area < CELL_AREA_RANGE[1]))[0]
    if len(ok) == 0:
        raise RuntimeError(f"no cell with area in {CELL_AREA_RANGE} px on {START}")
    return int(ok[np.argmax(area[ok])])


def fit_cell(cell: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit diffusion sources to one cell on CPU -> (reconstruction, raw parameters (K, 8), metrics)."""
    vsf.DEVICE = torch.device("cpu")
    lut5 = vsf.build_profile_lut()[None, None]
    records, snaps, (params, active) = vsf.fit_group([cell], "diffusion", K_MAX, FIT_ITERS, lut5, snap_k=(K_MAX,))
    return snaps[K_MAX], params[0][active[0] > 0].numpy(), records[-1]


def source_table(p: np.ndarray, cy: float, cx: float) -> pd.DataFrame:
    """Physical source parameters (same transforms as vsf.render); positions relative to the centroid."""
    return pd.DataFrame({
        "x": p[:, 0] - cx, "y": p[:, 1] - cy, "A": softplus(p[:, 2]),
        "a": 0.5 + softplus(p[:, 3]), "b": 0.5 + softplus(p[:, 4]),
        "theta": np.degrees(p[:, 5]) % 180, "w": 1 / (1 + np.exp(-p[:, 7])),
    }).sort_values("A", ascending=False, ignore_index=True)


def panel_image(ax, img, title, norm=None, cmap="Blues"):
    ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")


def shade_missing(ax, valid):
    ax.imshow(np.where(valid, np.nan, 1), cmap=ListedColormap(["#d4d3cf"]), interpolation="nearest")


def draw_sources(ax, p: np.ndarray):
    for r in p:
        ra, rb = 0.5 + softplus(r[3]), 0.5 + softplus(r[4])
        ax.add_patch(Ellipse((r[0], r[1]), 2 * ra, 2 * rb, angle=math.degrees(r[5]), fill=False, ec=ACCENT, lw=1))
        ax.plot(r[0], r[1], "+", color=ACCENT, ms=5, mew=1)


def draw_node(ax, glob: dict, src: pd.DataFrame):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("④ Node tensor of this cell", fontsize=9.5, fontweight="bold", loc="left")
    ax.add_patch(Rectangle((0.0, 0.63), 1, 0.35, fc="#f1f0ec", ec=MUTED, lw=0.6))
    ax.text(0.02, 0.95, "Global block (cell-level values)", fontsize=8.5, fontweight="bold", va="top")
    ax.text(0.02, 0.88, "\n".join(f"{k:<26}{v}" for k, v in glob.items()), fontsize=7.6,
            family="monospace", va="top", color=INK)
    ax.add_patch(Rectangle((0.0, 0.0), 1, 0.59, fc="white", ec=ACCENT, lw=1.2))
    ax.text(0.02, 0.565, f"Source set: K × 7 table (here K = {len(src)}; rows unordered, padded rows masked)",
            fontsize=8.5, fontweight="bold", va="top")
    hdr = f"{'x':>7}{'y':>7}{'A':>7}{'a':>7}{'b':>7}{'θ (°)':>7}{'w':>7}"
    rows = [f"{r.x:7.1f}{r.y:7.1f}{r.A:7.1f}{r.a:7.1f}{r.b:7.1f}{r.theta:7.0f}{r.w:7.2f}"
            for r in src.head(6).itertuples()]
    more = f"   … {len(src) - len(rows)} more rows"
    ax.text(0.02, 0.49, "\n".join([hdr, *rows, more]), fontsize=7.6, family="monospace",
            va="top", color=INK)
    ax.text(0.02, 0.03, "x, y: offset from cell centroid (px)   A: peak (mm/h)   a, b: ellipse radii (px)\n"
            "θ: orientation   w: edge sharpness (small = flat core with sharp edge, large = smooth bump)",
            fontsize=6.9, color=MUTED, va="bottom")


MAP_H, MAP_W = 565, 784
SHEAR, SQUASH, LAYER_GAP = 0.45, 0.32, 300


def project(cx: float, cy: float, layer: int) -> tuple[float, float]:
    """Oblique projection of map coordinates so the two time layers can be stacked as planes."""
    return cx + SHEAR * (MAP_H - cy), SQUASH * (MAP_H - cy) + layer * LAYER_GAP


def graph_positions(det: pd.DataFrame, times: list[str]) -> dict:
    pos = {}
    for layer, t in enumerate(times):
        for r in det[(det.timestamp == t) & (det.track_id > 0)].itertuples():
            pos[(t, r.track_id)] = (*project(r.cx, r.cy, layer), r.area_px, r.cx, r.cy)
    return pos


def draw_influence_edges(ax, pos: dict, t: str, focus_track: int):
    """Candidate influence edges within one frame; edges into the example cell are highlighted as arrows."""
    keys = [k for k in pos if k[0] == t]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if math.hypot(pos[a][3] - pos[b][3], pos[a][4] - pos[b][4]) >= INFLUENCE_RADIUS_PX:
                continue
            if focus_track in (a[1], b[1]):
                src, dst = (b, a) if a[1] == focus_track else (a, b)
                ax.add_patch(FancyArrowPatch(pos[src][:2], pos[dst][:2], arrowstyle="-|>", mutation_scale=8,
                                             color=ACCENT2, lw=1.1, shrinkA=3, shrinkB=7, zorder=2))
            else:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=ACCENT2, lw=0.5, alpha=0.25, zorder=1)


def draw_graph(ax, det: pd.DataFrame, times: list[str], focus_track: int):
    ax.set_title("⑤ Forecasting model: GAT on each frame (influence between cells) + LSTM along each track",
                 fontsize=9.5, fontweight="bold", loc="left")
    ax.axis("off")
    pos = graph_positions(det, times)
    for layer, t in enumerate(times):
        corners = [project(x, y, layer) for x, y in [(0, MAP_H), (MAP_W, MAP_H), (MAP_W, 0), (0, 0)]]
        ax.add_patch(plt.Polygon(corners, closed=True, fc="#f4f3ef", ec="#d4d3cf", lw=0.8, zorder=0))
        ax.text(corners[0][0] - 15, corners[0][1] + 10, f"graph at\nt = {t[11:16]}", fontsize=8.5,
                color=MUTED, ha="right")
        draw_influence_edges(ax, pos, t, focus_track)
    t0, t1 = times
    for (t, tid), (x, y, *_) in pos.items():
        if t != t0 or (t1, tid) not in pos:
            continue
        hl = tid == focus_track
        ax.add_patch(FancyArrowPatch((x, y), pos[(t1, tid)][:2], arrowstyle="-|>", mutation_scale=9 if hl else 6,
                                     color=ACCENT if hl else "#b9b8b2", lw=1.6 if hl else 0.6, shrinkA=6, shrinkB=6,
                                     zorder=1))
        if hl:
            xm, ym = (x + pos[(t1, tid)][0]) / 2, (y + pos[(t1, tid)][1]) / 2
            ax.text(xm + 12, ym, "LSTM\nh(t) → h(t+1)", fontsize=8, color=ACCENT, fontweight="bold", va="center")
    for (t, tid), (x, y, area, *_) in pos.items():
        hl = tid == focus_track
        ax.scatter(x, y, s=8 + math.sqrt(area) * 2.2, color=ACCENT if hl else NODE,
                   edgecolor=INK if hl else "white", lw=1.0 if hl else 0.4, zorder=3)
    ax.plot([], [], color=ACCENT2, lw=0.8, alpha=0.5, label="influence edge (GAT, within a frame)")
    ax.plot([], [], color=ACCENT2, lw=1.4, marker=">", ms=4, label="influences on the example cell")
    ax.plot([], [], color=ACCENT, lw=1.6, marker="^", ms=4, label="LSTM step along a track (t → t+1)")
    ax.scatter([], [], color=ACCENT, edgecolor=INK, s=40, label="the cell shown in ①–④")
    ax.set_xlim(-90, MAP_W + SHEAR * MAP_H + 10)
    ax.set_ylim(-10, LAYER_GAP + SQUASH * MAP_H + 10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=7, frameon=False,
              handlelength=1.6, columnspacing=1.0)


def arrow(fig, a, b):
    pa, pb = a.get_position(), b.get_position()
    y = (pa.y0 + pa.y1) / 2
    fig.add_artist(FancyArrowPatch((pa.x1 + 0.003, y), (pb.x0 - 0.003, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.2))


def main():
    data, dates = load_frames(START, 1)
    R = data[0]
    labels, n, valid = segment(R)
    lid = pick_cell(labels, n)
    sl = ndi.find_objects(labels)[lid - 1]
    area = int((labels == lid).sum())
    cell = {**vsf.crop_cell(R, labels, lid, sl, area, "2016-01", dates[0]), "id": 0, "bin": "1000-5000"}
    recon, p, met = fit_cell(cell)
    ys, xs = np.nonzero(cell["rate"] > 0)
    src = source_table(p, ys.mean(), xs.mean())

    det = pd.read_csv(DETECTIONS)
    times = sorted(det.timestamp.unique())[:2]
    d0 = det[det.timestamp == times[0]]
    focus = int(d0.iloc[((d0.cy - cell["cy"]) ** 2 + (d0.cx - cell["cx"]) ** 2).argmin()].track_id)

    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.22, wspace=0.12,
                          left=0.02, right=0.98, top=0.92, bottom=0.02)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[0, i]) for i in range(4))
    box = dict(xy=(sl[1].start - 6, sl[0].start - 6), width=sl[1].stop - sl[1].start + 12,
               height=sl[0].stop - sl[0].start + 12, fill=False, ec=ACCENT, lw=1.6)

    panel_image(ax1, rate_map(R), "① Radar rain rate (MeteoNet, 5 min, ~1 km)",
                LogNorm(0.1, 30))
    shade_missing(ax1, valid)
    ax1.add_patch(Rectangle(**box))

    area_all = np.bincount(labels.ravel())
    kept = np.isin(labels, np.nonzero(area_all >= MIN_AREA)[0][1:])
    colours = plt.cm.tab20(np.random.default_rng(3).permutation(20))
    panel_image(ax2, np.where(kept, labels % 20 + 1, 0),
                f"② Cells: pixels ≥ {MIN_RATE_MM_H:g} mm/h, connected, ≥ {MIN_AREA} px",
                cmap=ListedColormap(np.vstack([[1, 1, 1, 1], colours])))
    shade_missing(ax2, valid)
    ax2.add_patch(Rectangle(**box))

    vnorm = plt.Normalize(0, float(cell["rate"].max()))
    panel_image(ax3, np.where(cell["rate"] > 0, cell["rate"], np.nan),
                f"One cell at native resolution\n{area} px ≈ {area * PX_KM2:,.0f} km²", vnorm)
    panel_image(ax4, recon, f"③ Rebuilt from {len(src)} diffusion sources (ellipses)\n"
                f"nrmse {met['nrmse']:.2f} · IoU {met['iou']:.2f}", vnorm)
    draw_sources(ax4, p)
    for a in (ax3, ax4):
        a.contour(cell["rate"] >= 1, levels=[0.5], colors=[INK], linewidths=0.6)
    for a, b in [(ax1, ax2), (ax2, ax3), (ax3, ax4)]:
        arrow(fig, a, b)

    rate_in = cell["rate"][cell["rate"] > 0]
    glob = {"centroid (row, col)": f"({cell['cy']:.0f}, {cell['cx']:.0f}) px", "time": str(dates[0])[:16],
            "area": f"{area} px", "peak / mean rate": f"{rate_in.max():.1f} / {rate_in.mean():.1f} mm/h",
            "elongation, orientation": "from second moments", "velocity, edge flags": "from the track's past"}
    draw_node(fig.add_subplot(gs[1, 0:2]), glob, src)
    draw_graph(fig.add_subplot(gs[1, 2:4]), det, times, focus)

    fig.suptitle("From radar maps to a spatio-temporal graph model: each rain cell becomes a node described by a set of sources",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.savefig(OUT, dpi=140, facecolor="white")
    print(f"wrote {OUT} | cell {lid}, {area} px, K = {len(src)}, nrmse {met['nrmse']:.3f}, IoU {met['iou']:.3f}")


if __name__ == "__main__":
    main()
