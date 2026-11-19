"""Expérience 1 : S0 (R > 0, 8-connexité) sur MeteoNet NW, 2016-01-01 12:00,
puis vérification de la continuité sur les cartes suivantes (12:05, 12:10, ...).

Usage : python cell_delimitation.py [--start "2016-01-01 12:00"] [--n-frames 3]
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "01_segmentation_tracking"
MISSING = -1  # code MeteoNet « hors couverture / donnée manquante » (≠ 0 pluie)
# R = pluie cumulée en centièmes de mm (MeteoNet), supposée sur 5 min -> taux (mm/h) = R * 12 / 100
MIN_RATE_MM_H = 1.0
THRESHOLD_RAW = MIN_RATE_MM_H * 100 / 12  # 8.33 : pixel pluvieux si R > 8.33 (soit R >= 9)
STRUCT_8 = np.ones((3, 3), dtype=bool)
STRUCT_4 = ndi.generate_binary_structure(2, 1)


# ---------------------------------------------------------------- lecture
def load_frames(start: datetime, n_frames: int):
    """Charge n_frames cartes consécutives à partir de `start` (fichier mensuel npz)."""
    month_dir = DATA_DIR / f"NW_rainfall_{start.year}" / f"rainfall-NW-{start.year}-{start.month:02d}"
    for path in sorted(month_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as f:
            dates = list(f["dates"])
            if start in dates:
                i = dates.index(start)
                if i + n_frames > len(dates):
                    raise RuntimeError("Séquence à cheval sur deux fichiers : non géré.")
                data = f["data"][i:i + n_frames].copy()
                return data, dates[i:i + n_frames]
    raise FileNotFoundError(f"{start} introuvable dans {month_dir}")


# ------------------------------------------------------------ segmentation
def segment(R, structure=STRUCT_8, threshold=THRESHOLD_RAW):
    """Masque R > threshold (>= 1 mm/h par défaut) sur pixels valides ; composantes connexes."""
    valid = R != MISSING
    rain = valid & (R > threshold)
    labels, n = ndi.label(rain, structure=structure)
    return labels, n, valid


def describe(R, labels, n, valid, t):
    """Tableau de caractéristiques (une ligne par cellule)."""
    if n == 0:
        return pd.DataFrame()
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(R), labels, idx).astype(int)
    com = np.array(ndi.center_of_mass(np.ones_like(R, dtype=float), labels, idx))  # (y, x)
    wcom = np.array(ndi.center_of_mass(np.where(valid, R, 0).astype(float), labels, idx))
    s = ndi.sum(R, labels, idx)
    mx = ndi.maximum(R, labels, idx)
    # contact avec le bord de la grille / avec des pixels manquants
    border = np.zeros(labels.shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    touch_border = ndi.maximum(border, labels, idx).astype(bool)
    near_missing = ndi.binary_dilation(~valid, structure=STRUCT_8)
    touch_missing = ndi.maximum(near_missing, labels, idx).astype(bool)
    n_holes = np.zeros(n, int)
    for k, sl in enumerate(ndi.find_objects(labels)):
        m = labels[sl] == k + 1
        _, nh = ndi.label(~m, structure=STRUCT_4)  # fonds 4-connexes (dual de 8)
        n_holes[k] = max(nh - 1, 0)  # moins le fond extérieur (approx. : bbox)
    return pd.DataFrame({
        "timestamp": t, "local_id": idx, "area_px": area,
        "cy": com[:, 0], "cx": com[:, 1], "cy_w": wcom[:, 0], "cx_w": wcom[:, 1],
        "intensity_sum": s, "intensity_mean": s / area, "intensity_max": mx,
        "border_flag": touch_border, "missing_neighbor_flag": touch_missing,
        "n_holes_bbox": n_holes,
    })


# --------------------------------------------------------------- continuité
def link_stats(L0, L1, df0, df1):
    """Pour chaque cellule de t : recouvrement brut avec les cellules de t+1
    (aucun déplacement compensé -> simple test de continuité, pas du suivi)."""
    both = (L0 > 0) & (L1 > 0)
    pairs, counts = np.unique(np.stack([L0[both], L1[both]]), axis=1, return_counts=True)
    a0 = dict(zip(df0.local_id, df0.area_px))
    a1 = dict(zip(df1.local_id, df1.area_px))
    rows = [(i, j, c, c / (a0[i] + a1[j] - c)) for (i, j), c in zip(pairs.T, counts)]
    return pd.DataFrame(rows, columns=["id_t", "id_t1", "inter_px", "iou"])


def summarize_continuity(links, df0, df1, big=50):
    rows = []
    for _, c in df0[df0.area_px >= big].iterrows():
        l = links[links.id_t == c.local_id].sort_values("iou", ascending=False)
        if l.empty:
            rows.append((int(c.local_id), int(c.area_px), None, 0, 0.0, np.nan))
            continue
        b = l.iloc[0]
        d = np.hypot(*(df1.set_index("local_id").loc[b.id_t1, ["cx", "cy"]].values
                       - c[["cx", "cy"]].values.astype(float)))
        rows.append((int(c.local_id), int(c.area_px), int(b.id_t1), len(l), b.iou, d))
    return pd.DataFrame(rows, columns=["id_t", "area_t", "best_id_t1", "n_overlaps", "best_iou", "centroid_shift_px"])


# ------------------------------------------------------------------ figures
def plot_labels(R, labels, valid, title, path, big=200):
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.25, 1, (labels.max() + 1, 3))
    colors[0] = (1, 1, 1)
    rgb = colors[labels]
    rgb[~valid] = (0.6, 0.6, 0.6)  # gris = manquant
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(rgb, interpolation="nearest")
    for k, sl in enumerate(ndi.find_objects(labels), start=1):
        if (labels[sl] == k).sum() >= big:
            cy, cx = ndi.center_of_mass(labels == k)
            ax.text(cx, cy, str(k), fontsize=6, ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overlay(frames, labels_list, valid, dates, path):
    """Contours des cellules à t, t+1, t+2 superposés (rouge, vert, bleu)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(np.where(valid, 0.0, 0.5), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    for L, col, d in zip(labels_list, ["red", "limegreen", "blue"], dates):
        ax.contour(L > 0, levels=[0.5], colors=col, linewidths=0.5)
        ax.plot([], [], color=col, label=str(d))
    ax.legend(loc="lower right")
    ax.set_title("Contours des zones pluvieuses (S0) — gris = données manquantes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01 12:00")
    ap.add_argument("--n-frames", type=int, default=3)
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, dates = load_frames(start, args.n_frames)
    steps = {(b - a).total_seconds() for a, b in zip(dates, dates[1:])}
    print(f"Cartes : {dates[0]} -> {dates[-1]}, forme {data.shape}, pas de temps (s) : {steps}")

    results = {}
    for name, struct in [("S2_8conn", STRUCT_8), ("S1_4conn", STRUCT_4)]:
        res = []
        for R, t in zip(data, dates):
            L, n, valid = segment(R, struct)
            res.append((L, n, valid, describe(R, L, n, valid, t)))
        results[name] = res

    print("\n=== Segmentation (R >= 1 mm/h sur pixels valides) ===")
    for name, res in results.items():
        for (L, n, valid, df), t in zip(res, dates):
            a = df.area_px
            print(f"{name} {t:%H:%M}: {n:5d} cellules | aire moy {a.mean():6.1f} med {a.median():4.0f} "
                  f"max {a.max():6d} ({a.max() / (L > 0).sum():.0%} des pixels pluvieux) | "
                  f"pixels pluie {(L > 0).mean():.1%} | bord {df.border_flag.mean():.0%} | "
                  f"près de manquant {df.missing_neighbor_flag.mean():.0%}")

    # sauvegardes S0
    res0 = results["S2_8conn"]
    pd.concat([r[3] for r in res0]).to_csv(OUT_DIR / "S2_cells.csv", index=False)
    np.savez_compressed(OUT_DIR / "S2_labels.npz", labels=np.stack([r[0] for r in res0]).astype(np.int32),
                        dates=np.array([str(d) for d in dates]))

    # continuité t -> t+1 (S0)
    print("\n=== Continuité (recouvrement brut, sans compensation du déplacement) ===")
    for k in range(len(dates) - 1):
        L0, _, _, d0 = res0[k]
        L1, _, _, d1 = res0[k + 1]
        links = link_stats(L0, L1, d0, d1)
        summ = summarize_continuity(links, d0, d1)
        summ.to_csv(OUT_DIR / f"S2_continuity_{k}.csv", index=False)
        big0 = d0[d0.area_px >= 50]
        print(f"{dates[k]:%H:%M}->{dates[k+1]:%H:%M}: {len(big0)} cellules >=50 px ; "
              f"IoU brut médian {summ.best_iou.median():.2f} ; sans recouvrement {(summ.best_iou == 0).sum()} ; "
              f"chevauchent >=2 cellules suivantes {(summ.n_overlaps >= 2).sum()} ; "
              f"décalage centroïde médian {summ.centroid_shift_px.median():.1f} px")
        print(summ.sort_values("area_t", ascending=False).head(8).round(2).to_string(index=False))
        # séparations / fusions géométriques simples
        n_to = links.groupby("id_t").size()
        n_from = links.groupby("id_t1").size()
        print(f"  liens avec recouvrement : 1->plusieurs (séparation géom.) {(n_to >= 2).sum()}, "
              f"plusieurs->1 (fusion géom.) {(n_from >= 2).sum()}\n")

    # figures
    for (L, n, valid, df), t in zip(res0, dates):
        plot_labels(data[0], L, valid, f"S0 (8-conn.) {t} — {n} composantes",
                    OUT_DIR / f"S2_labels_{t:%H%M}.png")
    plot_overlay(data, [r[0] for r in res0[:3]], res0[0][2], dates[:3], OUT_DIR / "S2_overlay.png")
    print(f"Sorties dans {OUT_DIR}")


if __name__ == "__main__":
    main()
