"""Stabilité temporelle de la représentation par sources.

La même cellule à t et t + 5 min (appariement mutuel, IoU brut >= 0,5) est ajustée avec le même nombre de
sources K (fixé par classe de taille). Les deux ensembles de sources sont appariés (algorithme hongrois) et
on mesure leur écart. Référence : le même écart entre deux cellules DIFFÉRENTES de la même classe.
Un rapport (même cellule / cellules différentes) proche de 0 = représentation stable ; proche de 1 = instable.

Descripteurs d'une source (invariants à la translation et à l'échelle de la cellule) :
  position relative au centroïde / rayon équivalent, log amplitude, log rayons / rayon équivalent,
  angle (écart mesuré par (1 - cos 2Δθ) / 2), et pour la diffusion tau, w.

Usage :
  python stability_test.py --build                 (sur le Mac : construit les paires depuis les données brutes)
  python stability_test.py --device cuda           (ajuste et compare)
"""
import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

import variable_source_fit as vsf
from cell_delimitation import DATA_DIR, STRUCT_8, THRESHOLD_RAW, segment
from cell_dataset import FILES, FRAME_STEP, MIN_AREA

K_FIXED = {"50-200": 3, "200-1000": 6, "1000-5000": 12, ">=5000": 24}


# ------------------------------------------------------------------ construction des paires (données brutes)
def build_pairs(per_bin, path, seed=0):
    pairs = []
    for rel, tag in FILES:
        with np.load(DATA_DIR / rel, allow_pickle=True) as f:
            dates, data = f["dates"], f["data"]
        for i in range(0, len(dates) - 1, FRAME_STEP):
            if (dates[i + 1] - dates[i]).total_seconds() != 300:
                continue
            R0, R1 = data[i], data[i + 1]
            L0, n0, _ = segment(R0, STRUCT_8, THRESHOLD_RAW)
            L1, n1, _ = segment(R1, STRUCT_8, THRESHOLD_RAW)
            if n0 == 0 or n1 == 0:
                continue
            a0 = np.bincount(L0.ravel(), minlength=n0 + 1)
            a1 = np.bincount(L1.ravel(), minlength=n1 + 1)
            both = (L0 > 0) & (L1 > 0)
            (ii, jj), cnt = np.unique(np.stack([L0[both], L1[both]]), axis=1, return_counts=True)
            iou = cnt / (a0[ii] + a1[jj] - cnt)
            best0, best1 = {}, {}
            for i0, j1, v in zip(ii, jj, iou):
                if v > best0.get(i0, (0, 0))[0]:
                    best0[i0] = (v, j1)
                if v > best1.get(j1, (0, 0))[0]:
                    best1[j1] = (v, i0)
            sl0, sl1 = ndi.find_objects(L0), ndi.find_objects(L1)
            for i0, (v, j1) in best0.items():
                if v >= 0.5 and best1[j1][1] == i0 and a0[i0] >= MIN_AREA and a1[j1] >= MIN_AREA:
                    pairs.append((vsf.crop_cell(R0, L0, i0, sl0[i0 - 1], a0[i0], tag, dates[i]),
                                  vsf.crop_cell(R1, L1, j1, sl1[j1 - 1], a1[j1], tag, dates[i + 1]), float(v)))
        del data
        print(tag, "paires cumulées :", len(pairs), flush=True)
    rng = np.random.default_rng(seed)
    sample = []
    for name, lo, hi, _ in vsf.BINS:
        idx = [k for k, p in enumerate(pairs) if lo <= p[0]["area"] < hi]
        for k in rng.choice(idx, size=min(per_bin, len(idx)), replace=False):
            c0, c1, v = pairs[k]
            c0["bin"] = c1["bin"] = name  # classe définie par la cellule à t
            sample.append((c0, c1, v))
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(pickle.dumps(sample))
    print("écrit", path, "-", pd.Series([s[0]["bin"] for s in sample]).value_counts().to_dict())


# ------------------------------------------------------------------ comparaison des ensembles de sources
def descriptors(params, active, cell, kind):
    p = params[active > 0]
    r_eq = math.sqrt(cell["area"] / math.pi)
    lcx, lcy = cell["cx"] - cell["x0"], cell["cy"] - cell["y0"]
    ra, rb = 0.5 + F.softplus(p[:, 3]), 0.5 + F.softplus(p[:, 4])
    th = p[:, 5].clone()
    swap = rb > ra  # forme canonique : a >= b (l'ellipse (a, b, θ) = (b, a, θ + π/2))
    ra, rb = torch.where(swap, rb, ra), torch.where(swap, ra, rb)
    th = torch.where(swap, th + math.pi / 2, th)
    cols = [(p[:, 0] - lcx) / r_eq, (p[:, 1] - lcy) / r_eq, torch.log(F.softplus(p[:, 2])),
            torch.log(ra / r_eq), torch.log(rb / r_eq), th]
    if kind != "gauss":
        cols += [torch.sigmoid(p[:, 6]), torch.sigmoid(p[:, 7])]
    return torch.stack(cols, 1).numpy()


NAMES = {"gauss": ["x", "y", "logA", "log_a", "log_b", "angle"],
         "diffusion": ["x", "y", "logA", "log_a", "log_b", "angle", "tau", "w"]}


def match(D0, D1):
    """Appariement hongrois ; retourne le coût moyen et les écarts absolus par paramètre."""
    diff = D0[:, None, :] - D1[None, :, :]
    ang = (1 - np.cos(2 * diff[..., 5])) / 2
    sq = diff ** 2
    sq[..., 5] = ang
    C = sq.sum(-1)
    r, c = linear_sum_assignment(C)
    absd = np.abs(diff[r, c])
    absd[:, 5] = ang[r, c]
    return float(C[r, c].mean()), absd.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="construire les paires depuis les données brutes")
    ap.add_argument("--per-bin", type=int, default=25)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(vsf.OUT))
    args = ap.parse_args()
    out = Path(args.out)
    pairs_path = out / f"stability_pairs_{args.per_bin}.pkl"
    if args.build:
        build_pairs(args.per_bin, pairs_path)
        return

    vsf.OUT = out
    vsf.DEVICE = torch.device(args.device)
    print("device :", vsf.DEVICE, flush=True)
    lut5 = vsf.build_profile_lut()[None, None].to(vsf.DEVICE)
    sample = pickle.loads(pairs_path.read_bytes())
    rows = []
    for name, *_ in vsf.BINS:
        prs = [s for s in sample if s[0]["bin"] == name]
        if not prs:
            continue
        cells = [p[0] for p in prs] + [p[1] for p in prs]
        for k, c in enumerate(cells):
            c["id"] = k
        n = len(prs)
        for arm in ["gauss", "diffusion"]:
            t0 = time.time()
            desc, err = [None] * len(cells), {}
            for g0 in range(0, len(cells), vsf.GROUP[name]):
                gr = cells[g0:g0 + vsf.GROUP[name]]
                recs, _, (params, active) = vsf.fit_group(gr, arm, K_FIXED[name], args.iters, lut5)
                for b, c in enumerate(gr):
                    desc[c["id"]] = descriptors(params[b], active[b], c, arm)
                for r in recs:
                    if r["K"] == K_FIXED[name]:
                        err[r["cell"]] = r["nrmse"]
            for i in range(n):
                same, absd = match(desc[i], desc[n + i])
                other, _ = match(desc[i], desc[n + (i + 1) % n])  # référence : une autre cellule à t + 5 min
                rows.append(dict(bin=name, arm=arm, K=K_FIXED[name], pair=i, iou_raw=prs[i][2],
                                 nrmse_t=err[i], nrmse_t1=err[n + i], cost_same=same, cost_other=other,
                                 **{f"d_{nm}": v for nm, v in zip(NAMES[arm], absd)}))
            print(f"  {name:10s} {arm:9s} terminé ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stability.csv", index=False)
    summ = df.groupby(["bin", "arm"]).agg(K=("K", "first"), n=("pair", "count"), nrmse=("nrmse_t", "median"),
                                          cost_same=("cost_same", "median"), cost_other=("cost_other", "median"))
    summ["ratio"] = summ.cost_same / summ.cost_other
    print("\n=== Stabilité : écart médian entre ensembles de sources (même cellule à 5 min vs cellules différentes) ===")
    print(summ.round(3).to_string())
    print("\n=== Écart absolu moyen par paramètre, même cellule (médiane sur les paires) ===")
    dcols = [c for c in df.columns if c.startswith("d_")]
    print(df.groupby(["bin", "arm"])[dcols].median().round(3).to_string())


if __name__ == "__main__":
    main()
