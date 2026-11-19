"""Représentation à nombre variable de sources, à résolution native (pas de redimensionnement).

Chaque cellule est approchée par K sources (K croissant jusqu'à un plafond dépendant de la taille) :
  - bras "gauss"     : 6 paramètres / source (x, y, amplitude, rayon a, rayon b, angle), profil exp(-r^2/2)
  - bras "diffusion" : 8 paramètres / source = les 6 mêmes + temps de diffusion tau + netteté du bord w.
    Profil = solution d'une diffusion à diffusivité variable relative à la source, dans le repère elliptique
    de la source : D(r) = Dmax * tau * sigmoid((1 - r) / w) (+ plancher), à partir d'une source ponctuelle.
    Dans ce repère, l'EDP ne dépend que de (tau, w) : on la résout une fois sur une grille (tau, w) et on
    interpole le profil radial (table pré-calculée = même résultat que résoudre l'EDP pour chaque source).
Ajustement direct par cellule (Adam), sans encodeur : on mesure ce que la représentation peut décrire.
Sources ajoutées par vagues aux maxima du résidu ; métriques enregistrées pour chaque K.

Bras supplémentaires (ablation) : "diff_tau" (seul tau libre, w fixé) et "diff_w" (seul w libre, tau fixé),
7 paramètres / source.  --seed > 0 : perturbe l'initialisation (test de robustesse aux redémarrages).
--sample : échantillon de cellules pré-construit (pickle), par ex. une autre saison.

Usage : python variable_source_fit.py [--per-bin 40] [--iters 200] [--arms gauss diffusion] [--device cuda]
"""
import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from cell_delimitation import DATA_DIR, STRUCT_8, THRESHOLD_RAW, segment
from cell_dataset import FILES, FRAME_STEP, MIN_AREA

OUT = Path(__file__).resolve().parent.parent / "results" / "03_variable_sources"
MARGIN = 3  # pixels autour de la boîte englobante (cible = 0) : pénalise le débordement
BINS = [("50-200", 50, 200, 8), ("200-1000", 200, 1000, 24), ("1000-5000", 1000, 5000, 48),
        (">=5000", 5000, 10 ** 9, 64)]  # (nom, aire min, aire max, K max)
GROUP = {"50-200": 40, "200-1000": 20, "1000-5000": 10, ">=5000": 4}  # cellules optimisées ensemble
SCHEDULE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
EPS = 1e-3
ARM_NPAR = {"gauss": 6, "diffusion": 8, "diff_tau": 7, "diff_w": 7}
ARM_FIXED = {"diff_tau": [7], "diff_w": [6]}  # colonnes gelées : 6 = tau, 7 = w (valeur fixe : sigmoid(0))
DEVICE = torch.device("cpu")  # surchargé par --device
L_LUT = 2.0  # le profil est défini pour r dans [0, L_LUT] (en rayons de la source), nul au-delà


# ------------------------------------------------------------------ échantillon de cellules natives
def crop_cell(R, labels, lid, sl, area, tag, date):
    """Découpe native d'une cellule : boîte englobante + MARGIN, taux (mm/h) de la cellule seule."""
    y0, y1 = max(sl[0].start - MARGIN, 0), min(sl[0].stop + MARGIN, R.shape[0])
    x0, x1 = max(sl[1].start - MARGIN, 0), min(sl[1].stop + MARGIN, R.shape[1])
    m = labels[y0:y1, x0:x1] == lid
    rate = np.where(m, R[y0:y1, x0:x1] * 12 / 100, 0).astype(np.float32)
    ys, xs = np.nonzero(m)
    return dict(tag=tag, date=str(date), area=int(area), rate=rate, y0=y0, x0=x0,
                cy=float(ys.mean() + y0), cx=float(xs.mean() + x0))


def prepare_sample(per_bin, seed=0, files=None, test_only=True, name=None):
    """Échantillon stratifié par taille. Par défaut : fichiers du benchmark, jours de test (> 7)."""
    path = OUT / (name or f"sample_{per_bin}.pkl")
    if path.exists():
        return pickle.loads(path.read_bytes())
    cells = []
    for rel, tag in (files or FILES):
        with np.load(DATA_DIR / rel, allow_pickle=True) as f:
            dates, data = f["dates"], f["data"]
        for i in range(0, len(dates), FRAME_STEP):
            if test_only and dates[i].day <= 7:  # jours de test seulement (comme le benchmark)
                continue
            R = data[i]
            labels, n, _ = segment(R, STRUCT_8, THRESHOLD_RAW)
            area = np.bincount(labels.ravel(), minlength=n + 1)
            for lid, sl in enumerate(ndi.find_objects(labels), start=1):
                if area[lid] >= MIN_AREA:
                    cells.append(crop_cell(R, labels, lid, sl, area[lid], tag, dates[i]))
        del data
    rng = np.random.default_rng(seed)
    sample = []
    for bname, lo, hi, _ in BINS:
        idx = [k for k, c in enumerate(cells) if lo <= c["area"] < hi]
        pick = rng.choice(idx, size=min(per_bin, len(idx)), replace=False)
        for k in pick:
            cells[k]["bin"] = bname
            sample.append(cells[k])
    OUT.mkdir(exist_ok=True, parents=True)
    path.write_bytes(pickle.dumps(sample))
    return sample


# ------------------------------------------------------------------ table des profils de diffusion
def build_profile_lut(n_tau=24, n_w=16, n_r=128, n=96, steps=1200, sigma0=0.15, floor=0.005, dmax=0.24):
    path = OUT / "diffusion_profiles.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    ax = (torch.arange(n) + 0.5) / n * 2 * L_LUT - L_LUT
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    R = torch.sqrt(X ** 2 + Y ** 2)
    taus = torch.linspace(0, 1, n_tau)
    ws = torch.linspace(0.02, 0.5, n_w)
    D = dmax * (taus[:, None, None, None] * torch.sigmoid((1 - R) / ws[None, :, None, None]) + floor)
    D = D.reshape(-1, 1, n, n)  # D en unités de grille (dt/h^2 absorbé), stable car D <= 0.25
    u = torch.exp(-R ** 2 / (2 * sigma0 ** 2)).expand(D.shape[0], 1, n, n).clone()
    Dx, Dy = 0.5 * (D[..., :, 1:] + D[..., :, :-1]), 0.5 * (D[..., 1:, :] + D[..., :-1, :])
    for _ in range(steps):  # diffusion explicite conservative, flux nul au bord
        fx = Dx * (u[..., :, 1:] - u[..., :, :-1])
        fy = Dy * (u[..., 1:, :] - u[..., :-1, :])
        u = u + F.pad(fx, (0, 1)) - F.pad(fx, (1, 0)) + F.pad(fy, (0, 0, 0, 1)) - F.pad(fy, (0, 0, 1, 0))
    # profil radial moyenné sur 32 directions
    r = torch.linspace(0, L_LUT, n_r)
    ang = torch.linspace(0, 2 * math.pi, 33)[:-1]
    gx = (r[None, :] * torch.cos(ang[:, None])) / L_LUT
    gy = (r[None, :] * torch.sin(ang[:, None])) / L_LUT
    grid = torch.stack([gx, gy], -1)[None].expand(u.shape[0], -1, -1, -1)
    prof = F.grid_sample(u, grid, align_corners=False).mean(2)[:, 0]  # (n_tau*n_w, n_r)
    prof = prof / prof.max(1, keepdim=True).values
    prof[:, -1] = 0
    lut = prof.reshape(n_tau, n_w, n_r)
    torch.save(lut, path)
    return lut


# ------------------------------------------------------------------ rendu
def render(params, active, Xc, Yc, kind, lut5=None):
    """params (B,K,P), active (B,K), Xc/Yc (B,H,W) -> champ de taux (B,H,W) en mm/h."""
    B, K, _ = params.shape
    mx, my = params[..., 0, None, None], params[..., 1, None, None]
    amp = F.softplus(params[..., 2]) * active
    ra = 0.5 + F.softplus(params[..., 3, None, None])
    rb = 0.5 + F.softplus(params[..., 4, None, None])
    c, s = torch.cos(params[..., 5, None, None]), torch.sin(params[..., 5, None, None])
    dx, dy = Xc[:, None] - mx, Yc[:, None] - my
    u = (c * dx + s * dy) / ra
    v = (-s * dx + c * dy) / rb
    r2 = u ** 2 + v ** 2
    if kind == "gauss":
        prof = torch.exp(-0.5 * r2)
    else:
        H, W = Xc.shape[1:]
        r = torch.sqrt(r2 + 1e-9)
        gx = r / L_LUT * 2 - 1  # coordonnée r (> 1 -> hors table -> 0)
        gy = (torch.sigmoid(params[..., 7, None, None]) * 2 - 1).expand_as(r)  # netteté w
        gz = (torch.sigmoid(params[..., 6, None, None]) * 2 - 1).expand_as(r)  # temps tau
        grid = torch.stack([gx, gy, gz], -1).reshape(1, B * K, H * W, 1, 3)
        prof = F.grid_sample(lut5, grid, align_corners=True, padding_mode="zeros").reshape(B, K, H, W)
    return (amp[..., None, None] * prof).sum(1)


HALF_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]


def render_windowed(params, active, H, W, kind, lut5=None):
    """Même résultat que `render`, mais chaque source n'est évaluée que dans une fenêtre autour d'elle
    (profil nul au-delà de L_LUT rayons pour la diffusion, < 0.2 % au-delà de 3.5 rayons pour la gaussienne),
    puis ajoutée sur la grille (B, H, W). Fenêtres regroupées par taille."""
    B, K, P = params.shape
    bidx, kidx = torch.nonzero(active > 0, as_tuple=True)
    p = params[bidx, kidx]  # (N, P)
    ra = 0.5 + F.softplus(p[:, 3])
    rb = 0.5 + F.softplus(p[:, 4])
    reach = (L_LUT if kind == "diffusion" else 3.5) * torch.maximum(ra, rb).detach()
    canvas = torch.zeros(B * H * W, dtype=params.dtype, device=params.device)
    hsz = torch.tensor(HALF_SIZES, dtype=reach.dtype, device=reach.device)
    bucket = torch.searchsorted(hsz, reach.ceil() + 1).clamp(max=len(HALF_SIZES) - 1)
    for bi in torch.unique(bucket).tolist():
        sel = torch.nonzero(bucket == bi, as_tuple=True)[0]
        h = min(HALF_SIZES[bi], max(H, W))
        q = p[sel]
        off = torch.arange(-h, h + 1, dtype=params.dtype, device=params.device)
        cx, cy = q[:, 0].detach().round(), q[:, 1].detach().round()
        X = (cx[:, None, None] + off[None, None, :]).expand(-1, 2 * h + 1, -1)  # (n, S, S)
        Y = (cy[:, None, None] + off[None, :, None]).expand(-1, -1, 2 * h + 1)
        dx, dy = X - q[:, 0, None, None], Y - q[:, 1, None, None]
        c, s = torch.cos(q[:, 5, None, None]), torch.sin(q[:, 5, None, None])
        u = (c * dx + s * dy) / ra[sel, None, None]
        v = (-s * dx + c * dy) / rb[sel, None, None]
        r2 = u ** 2 + v ** 2
        if kind == "gauss":
            prof = torch.exp(-0.5 * r2)
        else:
            r = torch.sqrt(r2 + 1e-9)
            gx = r / L_LUT * 2 - 1
            gy = (torch.sigmoid(q[:, 7, None, None]) * 2 - 1).expand_as(r)
            gz = (torch.sigmoid(q[:, 6, None, None]) * 2 - 1).expand_as(r)
            n_, S = r.shape[0], r.shape[1]
            grid = torch.stack([gx, gy, gz], -1).reshape(1, n_, S * S, 1, 3)
            prof = F.grid_sample(lut5, grid, align_corners=True, padding_mode="zeros").reshape(n_, S, S)
        vals = F.softplus(q[:, 2, None, None]) * prof
        inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        flat = bidx[sel, None, None] * (H * W) + Y.clamp(0, H - 1).long() * W + X.clamp(0, W - 1).long()
        canvas = canvas.index_add(0, flat[inb], vals[inb])
    return canvas.view(B, H, W)


def inv_softplus(y):
    return math.log(math.expm1(y))


def metrics(rec, T, valid):
    out = []
    for b in range(T.shape[0]):
        m = valid[b]
        t, h = T[b][m], rec[b][m].clamp(min=0)
        st, sh = t.sqrt(), h.sqrt()
        bt, bh = t >= 1, h >= 1
        out.append(dict(nrmse=float((sh - st).norm() / st.norm()), iou=float((bt & bh).sum() / (bt | bh).sum().clamp(min=1)),
                        mass_err=float((h.sum() - t.sum()).abs() / t.sum()),
                        peak_err=float((h.max() - t.max()).abs() / t.max())))
    return out


# ------------------------------------------------------------------ ajustement d'un groupe de cellules
def fit_group(cells, kind, kmax, iters, lut5, snap_k=(), rng=None):
    """kind : gauss | diffusion | diff_tau | diff_w.  rng : perturbation de l'initialisation (ou None).
    Retourne (enregistrements par K, images, paramètres finaux (B, kmax, P) et masque actif, sur CPU)."""
    rkind = "gauss" if kind == "gauss" else "diffusion"
    frozen = ARM_FIXED.get(kind, [])
    B = len(cells)
    Hm, Wm = max(c["rate"].shape[0] for c in cells), max(c["rate"].shape[1] for c in cells)
    T = torch.zeros(B, Hm, Wm, device=DEVICE)
    valid = torch.zeros(B, Hm, Wm, dtype=torch.bool, device=DEVICE)
    for b, c in enumerate(cells):
        h, w = c["rate"].shape
        T[b, :h, :w] = torch.from_numpy(c["rate"])
        valid[b, :h, :w] = True
    sT = T.sqrt()
    P = 6 if rkind == "gauss" else 8
    params = torch.zeros(B, kmax, P, device=DEVICE)
    active = torch.zeros(B, kmax, device=DEVICE)
    k_cur, records, snaps = 0, [], {}
    for K in [k for k in SCHEDULE if k <= kmax]:
        with torch.no_grad():
            rec = render_windowed(params, active, Hm, Wm, rkind, lut5)
            resid = ((sT - (rec + EPS).sqrt()) * valid).cpu().numpy()
            pos = (T - rec).clamp(min=0).cpu().numpy()
        for b in range(B):  # nouvelles sources aux maxima du résidu
            res = resid[b].copy()
            # rayon initial : disque d'aire = aire de la cellule / K (même règle pour les deux bras ;
            # sigma gaussien = moitié du rayon du disque)
            rho = float(np.clip(0.7 * math.sqrt(cells[b]["area"] / (math.pi * K)), 2.0, 40.0))
            if rng is not None:
                rho *= float(rng.uniform(0.8, 1.2))
            r0 = inv_softplus(max((0.5 * rho if kind == "gauss" else rho) - 0.5, 0.1))
            sup = max(int(rho), 2)  # pas deux nouvelles sources à moins de rho pixels
            for k in range(k_cur, K):
                y, x = np.unravel_index(np.argmax(res), res.shape)
                init = [x, y, inv_softplus(max(float(pos[b, y, x]), 0.5)), r0, r0, 0, 0, 0][:P]
                if rng is not None:
                    init[0] += float(rng.uniform(-1.5, 1.5))
                    init[1] += float(rng.uniform(-1.5, 1.5))
                    init[5] = float(rng.uniform(0, math.pi))
                    for col in (6, 7)[:P - 6]:
                        if col not in frozen:
                            init[col] = float(rng.normal(0, 0.5))
                params[b, k] = torch.tensor(init, device=DEVICE)
                active[b, k] = 1
                res[max(y - sup, 0):y + sup + 1, max(x - sup, 0):x + sup + 1] = -1e9
        k_cur = K
        params.requires_grad_(True)
        opt = torch.optim.Adam([params], lr=0.05)
        npix = valid.sum((1, 2))
        for _ in range(iters):
            rec = render_windowed(params, active, Hm, Wm, rkind, lut5)
            loss = ((((rec.clamp(min=0) + EPS).sqrt() - sT) ** 2) * valid).sum((1, 2)) / npix
            opt.zero_grad()
            loss.sum().backward()
            for col in frozen:
                params.grad[..., col] = 0
            opt.step()
        params = params.detach()
        with torch.no_grad():
            rec = render_windowed(params, active, Hm, Wm, rkind, lut5)
        for b, mt in enumerate(metrics(rec, T, valid)):
            records.append(dict(cell=cells[b]["id"], bin=cells[b]["bin"], area=cells[b]["area"], arm=kind,
                                K=K, n_params=K * ARM_NPAR[kind], **mt))
        if K in snap_k:
            snaps[K] = rec[0, :cells[0]["rate"].shape[0], :cells[0]["rate"].shape[1]].cpu().numpy()
    return records, snaps, (params.cpu(), active.cpu())


def bin_slug(name):
    return name.replace(">=", "ge").replace("-", "_")


def summarize(df):
    budgets = [24, 48, 96, 192, 384]
    print("\n=== Erreur médiane (nrmse sur sqrt(taux)) à nombre de paramètres égal ===")
    rows = []
    for name, *_ in BINS:
        d = df[df.bin == name]
        for n in budgets:
            g = d[(d.arm == "gauss") & (d.n_params == n)]
            f = d[(d.arm == "diffusion") & (d.n_params == n)]
            if len(g) and len(f):
                rows.append(dict(bin=name, n_params=n, gauss_K=n // 6, gauss_nrmse=g.nrmse.median(), gauss_iou=g.iou.median(),
                                 diff_K=n // 8, diff_nrmse=f.nrmse.median(), diff_iou=f.iou.median(),
                                 diff_better_frac=float((f.set_index("cell").nrmse < g.set_index("cell").nrmse).mean())))
    t1 = pd.DataFrame(rows)
    print(t1.round(3).to_string(index=False))
    print("\n=== Paramètres nécessaires pour atteindre une erreur cible (médiane sur les cellules ; % atteint) ===")
    rows = []
    for name, *_ in BINS:
        for arm in [a for a in ARM_NPAR if a in set(df.arm)]:
            d = df[(df.bin == name) & (df.arm == arm)]
            row = dict(bin=name, arm=arm)
            for tol in [0.3, 0.2, 0.15]:
                need = d[d.nrmse <= tol].groupby("cell").n_params.min()
                ncell = d.cell.nunique()
                row[f"<= {tol}"] = f"{need.median():.0f} ({len(need) / ncell:.0%})" if len(need) else f"- (0%)"
            rows.append(row)
    t2 = pd.DataFrame(rows)
    print(t2.to_string(index=False))
    print("\n=== Erreur médiane par nombre de sources K (tous les bras, y compris l'ablation) ===")
    t3 = df[df.K.isin([3, 6, 12, 24, 48])].pivot_table(index=["bin", "K"], columns="arm", values="nrmse", aggfunc="median")
    print(t3.round(3).to_string())
    return t1, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=40)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--arms", nargs="+", default=["gauss", "diffusion"])
    ap.add_argument("--bins", nargs="+", default=[b[0] for b in BINS], help="classes de taille à traiter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT), help="dossier des résultats (contient l'échantillon en cache)")
    ap.add_argument("--sample", default=None, help="échantillon pré-construit (pickle) au lieu de --per-bin")
    ap.add_argument("--seed", type=int, default=0, help="> 0 : initialisation perturbée (redémarrage)")
    args = ap.parse_args()
    globals()["OUT"] = Path(args.out)
    globals()["DEVICE"] = torch.device(args.device)
    print("device :", DEVICE, flush=True)
    OUT.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(0)
    sample = pickle.loads(Path(args.sample).read_bytes()) if args.sample else prepare_sample(args.per_bin)
    rng = np.random.default_rng(args.seed) if args.seed > 0 else None
    for k, c in enumerate(sample):
        c["id"] = k
    print(f"{len(sample)} cellules :", pd.Series([c['bin'] for c in sample]).value_counts().to_dict(), flush=True)
    t0 = time.time()
    lut = build_profile_lut()
    lut5 = lut[None, None].to(DEVICE)  # (1, 1, tau, w, r)
    print(f"table des profils de diffusion {tuple(lut.shape)} ({time.time() - t0:.0f}s)", flush=True)

    snap_path = OUT / "snapshots.pkl"
    snaps = pickle.loads(snap_path.read_bytes()) if snap_path.exists() else {}
    for name, lo, hi, kmax in BINS:
        if name not in args.bins:
            continue
        records = []
        group = [c for c in sample if c["bin"] == name]
        for arm in args.arms:
            t0 = time.time()
            for g0 in range(0, len(group), GROUP[name]):
                gr = group[g0:g0 + GROUP[name]]
                snap_k = (6, 8, 12, 16) if g0 == 0 else ()
                rec, sn, _ = fit_group(gr, arm, kmax, args.iters, lut5, snap_k, rng)
                records += rec
                for K, img in sn.items():
                    snaps[(name, arm, K)] = img
                if g0 == 0:
                    snaps[(name, "target")] = gr[0]["rate"]
            print(f"  {name:10s} {arm:9s} terminé ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(records).to_csv(OUT / f"fits_{bin_slug(name)}.csv", index=False)
        snap_path.write_bytes(pickle.dumps(snaps))
    # résumé sur toutes les classes déjà calculées (fichiers fits_<classe>.csv)
    df = pd.concat([pd.read_csv(OUT / f"fits_{bin_slug(b[0])}.csv") for b in BINS
                    if (OUT / f"fits_{bin_slug(b[0])}.csv").exists()], ignore_index=True)
    t1, t2 = summarize(df)
    t1.to_csv(OUT / "summary_equal_params.csv", index=False)
    t2.to_csv(OUT / "summary_params_needed.csv", index=False)


if __name__ == "__main__":
    main()
