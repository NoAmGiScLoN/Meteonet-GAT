"""Suivi temporel (T1) : track_id stable au-dessus de la segmentation S0.

Prédiction à vitesse constante -> coût = position (vs position prédite) + (1 - IoU du masque translaté)
-> affectation hongroise -> rejet des liens trop coûteux. Pas de tolérance de disparition (g = 0).
Les cellules < min_area ne sont pas suivies (track_id = 0) ; elles restent dans S0_cells.

Usage : python track_cells.py [--start "2016-01-01 12:00"] [--n-frames 6]
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cell_delimitation import OUT_DIR, load_frames, segment, describe

BIG = 1e6


def _coords(labels, objs, lid):
    sl = objs[lid - 1]
    ys, xs = np.nonzero(labels[sl] == lid)
    return ys + sl[0].start, xs + sl[1].start


def match(L0, df0, L1, df1, state, d_max, w_pos, max_cost):
    """Associe les cellules actives (state: track_id -> dict) à celles de L1.
    Retourne [(track_id, local_id_t1, cost, dist, iou)]."""
    tids = list(state)
    cands = df1.set_index("local_id")
    cids = list(cands.index)
    if not tids or not cids:
        return []
    objs0 = ndi.find_objects(L0)
    H, W = L1.shape
    a1 = cands.area_px.reindex(cids).values
    C = np.full((len(tids), len(cids)), BIG)
    info = {}
    for r, tid in enumerate(tids):
        s = state[tid]
        ys, xs = _coords(L0, objs0, s["local_id"])
        dy, dx = int(round(s["vy"])), int(round(s["vx"]))
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        inter = np.bincount(L1[yy[ok], xx[ok]], minlength=L1.max() + 1)
        for c, cid in enumerate(cids):
            dist = np.hypot(cands.at[cid, "cy"] - (s["cy"] + s["vy"]), cands.at[cid, "cx"] - (s["cx"] + s["vx"]))
            i = inter[cid]
            iou = i / (s["area"] + a1[c] - i)
            if dist > d_max and iou < 0.1:
                continue  # gating
            C[r, c] = w_pos * dist / d_max + (1 - w_pos) * (1 - iou)
            info[(r, c)] = (dist, iou)
    rows, cols = linear_sum_assignment(C)
    out = []
    for r, c in zip(rows, cols):
        if C[r, c] <= max_cost:
            out.append((tids[r], cids[c], C[r, c], *info[(r, c)]))
    return out


def run_tracking(labels_list, dfs, min_area=50, d_max=25.0, w_pos=0.5, max_cost=0.75):
    """Retourne (table des détections avec track_id, table des associations, événements)."""
    next_id = 1
    state, dets, assoc, events = {}, [], [], []
    for k, (L, df) in enumerate(zip(labels_list, dfs)):
        big = df[df.area_px >= min_area]
        new_state, tid_of = {}, {}
        if k > 0:
            m = match(labels_list[k - 1], dfs[k - 1], L, big, state, d_max, w_pos, max_cost)
            for tid, lid, cost, dist, iou in m:
                tid_of[lid] = tid
                assoc.append((dfs[k - 1].timestamp.iloc[0], state[tid]["local_id"], df.timestamp.iloc[0], lid,
                              tid, cost, dist, iou, "continuation"))
            for tid in set(state) - set(tid_of.values()):
                events.append((df.timestamp.iloc[0], "death", tid, ""))
        for _, c in big.iterrows():
            lid = int(c.local_id)
            if lid not in tid_of:
                tid_of[lid] = next_id
                if k > 0:
                    events.append((c.timestamp, "birth", next_id, ""))
                next_id += 1
            tid = tid_of[lid]
            if tid in state:  # vitesse = dernier déplacement observé (Δt constant = 1 pas)
                vy, vx = c.cy - state[tid]["cy"], c.cx - state[tid]["cx"]
            else:
                vy = vx = 0.0
            new_state[tid] = dict(local_id=lid, cy=c.cy, cx=c.cx, vy=vy, vx=vx, area=c.area_px)
        state = new_state
        d = df.copy()
        d["track_id"] = d.local_id.map(tid_of).fillna(0).astype(int)
        dets.append(d)
    assoc = pd.DataFrame(assoc, columns=["timestamp_from", "local_id_from", "timestamp_to", "local_id_to",
                                         "track_id", "cost", "distance", "translated_iou", "association_type"])
    events = pd.DataFrame(events, columns=["timestamp", "event_type", "track_id", "notes"])
    return pd.concat(dets, ignore_index=True), assoc, events


def plot_tracks(data_labels, dets, dates, valid, path, big=200):
    n = len(dates)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5 * rows), squeeze=False)
    rng = np.random.default_rng(1)
    palette = rng.uniform(0.2, 0.95, (dets.track_id.max() + 1, 3))
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        L = data_labels[k]
        d = dets[(dets.timestamp == dates[k]) & (dets.track_id > 0)]
        lut = np.ones((L.max() + 1, 3))
        lut[d.local_id.values] = palette[d.track_id.values]
        rgb = lut[L]
        rgb[(L > 0) & ~np.isin(L, d.local_id.values)] = (0.82, 0.82, 0.82)  # cellules non suivies
        rgb[~valid] = (0.5, 0.5, 0.5)
        ax.imshow(rgb, interpolation="nearest")
        for _, c in d[d.area_px >= big].iterrows():
            ax.text(c.cx, c.cy, str(int(c.track_id)), fontsize=7, ha="center", va="center", weight="bold")
        ax.set_title(f"{dates[k]:%H:%M} — id = track_id", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01 12:00")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--d-max", type=float, default=25.0)
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, dates = load_frames(start, args.n_frames)
    labels_list, dfs = [], []
    for R, t in zip(data, dates):
        L, n, valid = segment(R)
        labels_list.append(L)
        dfs.append(describe(R, L, n, valid, t))

    dets, assoc, events = run_tracking(labels_list, dfs, min_area=args.min_area, d_max=args.d_max)
    dets.to_csv(OUT_DIR / "T1_detections.csv", index=False)
    assoc.to_csv(OUT_DIR / "T1_associations.csv", index=False)
    events.to_csv(OUT_DIR / "T1_events.csv", index=False)

    tracked = dets[dets.track_id > 0]
    # contrôle de cohérence : un track_id ne doit apparaître qu'une fois par carte
    assert not tracked.duplicated(["timestamp", "track_id"]).any()
    length = tracked.groupby("track_id").size()
    print(f"{len(dates)} cartes, cellules suivies (aire >= {args.min_area}) : {tracked.groupby('timestamp').size().tolist()}")
    print(f"trajectoires : {len(length)} | présentes sur toutes les cartes : {(length == len(dates)).sum()} | "
          f"longueur médiane {length.median():.0f}")
    print(events.groupby(["timestamp", "event_type"]).size().unstack(fill_value=0).to_string())
    print("\nGrandes cellules (aire max) — track_id / local_id par carte :")
    top = tracked.sort_values("area_px", ascending=False).drop_duplicates("track_id").head(8).track_id
    for tid in top:
        s = tracked[tracked.track_id == tid].sort_values("timestamp")
        print(f"  track {tid:3d}: local_id " + " ".join(f"{t:%H:%M}={l}({a})" for t, l, a in
                                                       zip(s.timestamp, s.local_id, s.area_px)))
    print("\nAssociations les plus faibles (à vérifier visuellement) :")
    print(assoc.sort_values("cost", ascending=False).head(6).round(2).to_string(index=False))

    valid = data[0] != -1
    plot_tracks(labels_list, dets, dates, valid, OUT_DIR / "T1_tracks.png")
    print(f"\nSorties dans {OUT_DIR}")


if __name__ == "__main__":
    main()
