"""Benchmark de compression des cellules : chaque cellule (patch PxP, P=32 par défaut du champ sqrt(taux mm/h)) -> vecteur de
dimension d -> reconstruction. Évalué sur des jours de test jamais vus.

Bras : mean (d=0), pca, zernike, ae (décodeur CNN appris), gmm (décodeur = K noyaux de chaleur anisotropes),
       diffusion (décodeur = K sources + diffusion explicite avec D(x,y) variable, 3 paramètres de champ D).
Tous les bras reconstruisent à partir du seul vecteur de dimension d (les moments géométriques, communs à tous
les bras, sont stockés à part et ne participent pas à la reconstruction).

Usage : python compression_benchmark.py --arms pca zernike ae gmm diffusion --dims 6 12 18 30 48
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent.parent / "results" / "02_fixed_vector_benchmark"
BINS = [("50-200", 50, 200), ("200-1000", 200, 1000), ("1000-5000", 1000, 5000), (">=5000", 5000, 10 ** 9)]
P = 32  # résolution des patchs (surchargée par --res)


# ------------------------------------------------------------------ données
def load():
    """Patchs sqrt(taux) à la résolution P x P (P = 32 par défaut), jours <= 7 = entraînement."""
    d = np.load(OUT / "cells.npz")
    s = np.sqrt(d["rate"]).astype(np.float32)  # sqrt(mm/h) : compresse la queue lourde
    f = 64 // P
    if f > 1:  # moyenne par blocs -> résolution P x P
        s = s.reshape(len(s), P, f, P, f).mean((2, 4))
    train = d["day"] <= 7
    return s, d["area"], train


# ------------------------------------------------------------------ métriques
def evaluate(s_true, s_hat, area):
    s_hat = np.maximum(s_hat, 0)
    flat = lambda a: a.reshape(len(a), -1)
    t, h = flat(s_true), flat(s_hat)
    nrmse = np.linalg.norm(h - t, axis=1) / np.linalg.norm(t, axis=1)
    mt, mh = (t ** 2).sum(1), (h ** 2).sum(1)
    mass = np.abs(mh - mt) / mt
    peak = np.abs(h.max(1) - t.max(1)) / t.max(1)
    bt, bh = t >= 1, h >= 1  # seuil 1 mm/h
    iou = (bt & bh).sum(1) / np.maximum((bt | bh).sum(1), 1)
    res = {}
    for name, lo, hi in [("all", 0, 10 ** 9)] + BINS:
        m = (area >= lo) & (area < hi)
        if m.sum() == 0:
            continue
        res[name] = dict(n=int(m.sum()), nrmse=float(nrmse[m].mean()), nrmse_med=float(np.median(nrmse[m])),
                         mass_err=float(mass[m].mean()), iou=float(iou[m].mean()), peak_err=float(peak[m].mean()))
    return res


# ------------------------------------------------------------- bras linéaires
def zernike_basis(d, size=P):
    ax = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(ax, ax)
    rho = np.sqrt(X ** 2 + Y ** 2) / np.sqrt(2)  # disque circonscrit au carré
    th = np.arctan2(Y, X)
    mods, n = [], 0
    while len(mods) < d:
        for m in range(n % 2, n + 1, 2):
            mods += [(n, 0, "c")] if m == 0 else [(n, m, "c"), (n, m, "s")]
        n += 1
    cols = []
    for n, m, kind in mods[:d]:
        R = sum((-1) ** k * math.factorial(n - k) / (math.factorial(k) * math.factorial((n + m) // 2 - k)
                                                     * math.factorial((n - m) // 2 - k)) * rho ** (n - 2 * k)
                for k in range((n - m) // 2 + 1))
        ang = np.cos(m * th) if kind == "c" else np.sin(m * th)
        c = (R * ang).ravel()
        cols.append(c / np.linalg.norm(c))
    return np.stack(cols, 1)


def run_linear(arm, d, s_tr, s_te):
    Xtr, Xte = s_tr.reshape(len(s_tr), -1), s_te.reshape(len(s_te), -1)
    if arm == "mean":
        return np.tile(Xtr.mean(0), (len(Xte), 1))
    if arm == "pca":
        mu = Xtr.mean(0)
        _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        V = Vt[:d].T
        return (Xte - mu) @ V @ V.T + mu
    if arm == "zernike":
        B = zernike_basis(d)
        return Xte @ np.linalg.pinv(B).T @ B.T
    raise ValueError(arm)


# ------------------------------------------------------------- bras neuronaux
class Encoder(nn.Module):
    def __init__(self, out):
        super().__init__()
        L = int(math.log2(P // 4))
        ch = [1, 32, 64, 128, 128][:L + 1]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)  # P -> 4
        self.ch_last = ch[-1]
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(ch[-1] * 16, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        return self.fc(self.conv(x))


class CNNDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        L = int(math.log2(P // 4))
        ch = ([128, 128, 64, 32])[:L] + [1] if L < 4 else [128, 128, 64, 32, 1]
        if L == 3:
            ch = [128, 64, 32, 1]
        self.c0 = ch[0]
        self.fc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, ch[0] * 16), nn.ReLU())
        layers = []
        for i, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1)] + ([nn.ReLU()] if i < len(ch) - 2 else [])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return F.softplus(self.deconv(self.fc(z).view(-1, self.c0, 4, 4)))


def grid(n):
    ax = (torch.arange(n) + 0.5) / n * 2 - 1
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    return X, Y


class GaussDecoder(nn.Module):
    """K noyaux de chaleur anisotropes (x, y, amplitude, log sigma_a, log sigma_b, angle) = 6 paramètres chacun."""
    def __init__(self, d):
        super().__init__()
        self.K = d // 6
        X, Y = grid(P)
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

    def forward(self, p):
        p = p.view(-1, self.K, 6, 1, 1)
        mx, my = torch.tanh(p[:, :, 0]), torch.tanh(p[:, :, 1])
        amp = F.softplus(p[:, :, 2]) * 3
        sa, sb = torch.exp(p[:, :, 3].clamp(-4.5, 0.7)), torch.exp(p[:, :, 4].clamp(-4.5, 0.7))
        th = p[:, :, 5]
        dx, dy = self.X - mx, self.Y - my
        u = torch.cos(th) * dx + torch.sin(th) * dy
        v = -torch.sin(th) * dx + torch.cos(th) * dy
        return (amp * torch.exp(-0.5 * (u ** 2 / sa ** 2 + v ** 2 / sb ** 2))).sum(1, keepdim=True)


class DiffusionDecoder(nn.Module):
    """K sources ponctuelles (x, y, masse) + champ de diffusivité log D = a0 + a1 x + a2 y (3 paramètres),
    diffusion explicite conservative (flux nul au bord) pendant `steps` pas sur grille 32x32, puis sur-échantillonnage."""
    def __init__(self, d, steps=64, n=None, dmax=0.24):
        super().__init__()
        n = n or P
        self.K, self.steps, self.n, self.dmax = (d - 3) // 3, steps, n, dmax
        X, Y = grid(n)
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

    def forward(self, p):
        B = p.shape[0]
        src = p[:, :3 * self.K].view(B, self.K, 3, 1, 1)
        mx, my = torch.tanh(src[:, :, 0]), torch.tanh(src[:, :, 1])
        mass = F.softplus(src[:, :, 2]) * 40
        dist2 = ((self.X - mx) ** 2 + (self.Y - my) ** 2) * (self.n / 2) ** 2  # en pixels de grille
        u = (mass * torch.exp(-0.5 * dist2) / (2 * math.pi)).sum(1, keepdim=True)
        a = p[:, 3 * self.K:].view(B, 3, 1, 1)
        D = (self.dmax * torch.sigmoid(a[:, 0:1] + a[:, 1:2] * self.X + a[:, 2:3] * self.Y)).unsqueeze(1).squeeze(2)
        Dx, Dy = 0.5 * (D[..., :, 1:] + D[..., :, :-1]), 0.5 * (D[..., 1:, :] + D[..., :-1, :])
        for _ in range(self.steps):
            fx = Dx * (u[..., :, 1:] - u[..., :, :-1])
            fy = Dy * (u[..., 1:, :] - u[..., :-1, :])
            u = u + F.pad(fx, (0, 1, 0, 0)) - F.pad(fx, (1, 0, 0, 0)) + F.pad(fy, (0, 0, 0, 1)) - F.pad(fy, (0, 0, 1, 0))
        return F.interpolate(u, size=(P, P), mode="bilinear", align_corners=False)


def init_physical(enc, per, K, base, extra=()):
    """Sorties initiales raisonnables : petites poids, sources réparties sur un cercle, largeurs moyennes."""
    last = enc.fc[-1]
    last.weight.data *= 0.1
    b = torch.zeros(last.out_features)
    for k in range(K):
        v = torch.tensor(base, dtype=torch.float32)
        if K > 1:
            a = 2 * math.pi * k / K
            v[0], v[1] = 0.55 * math.cos(a), 0.55 * math.sin(a)  # atanh(0.5)
        b[k * per:(k + 1) * per] = v
    for i, e in enumerate(extra):
        b[K * per + i] = e
    last.bias.data = b


def make_model(arm, d):
    if arm == "ae":
        return Encoder(d), CNNDecoder(d), d
    if arm == "gmm":
        K = d // 6
        enc = Encoder(K * 6)
        init_physical(enc, 6, K, [0, 0, 0, -1.2, -1.2, 0])
        return enc, GaussDecoder(d), K * 6
    if arm == "diffusion":
        K = (d - 3) // 3
        enc = Encoder(3 * K + 3)
        init_physical(enc, 3, K, [0, 0, 0], extra=(0, 0, 0))
        return enc, DiffusionDecoder(3 * K + 3), 3 * K + 3
    raise ValueError(arm)


def run_neural(arm, d, s_tr, s_te, epochs, seed=0, bs=128, lr=2e-3, device="cpu"):
    torch.manual_seed(seed)
    enc, dec, real_d = make_model(arm, d)
    model = nn.Sequential(enc, dec).to(device)
    xtr = torch.from_numpy(s_tr)[:, None]
    xte = torch.from_numpy(s_te)[:, None]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(xtr) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), bs):
            x = xtr[perm[i:i + bs]].to(device)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    {arm} d={real_d} ep {ep + 1}/{epochs} train mse {tot / len(xtr):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        rec = torch.cat([model(xte[i:i + 256].to(device)).cpu() for i in range(0, len(xte), 256)])
    return rec[:, 0].numpy(), real_d


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["mean", "pca", "zernike", "ae", "gmm", "diffusion"])
    ap.add_argument("--dims", nargs="+", type=int, default=[6, 12, 18, 30, 48])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-train", type=int, default=0, help="sous-échantillonne l'entraînement (tests rapides)")
    ap.add_argument("--tag", default="results")
    ap.add_argument("--res", type=int, default=32, choices=[32, 64], help="résolution des patchs")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print("device :", args.device, flush=True)
    globals()["P"] = args.res

    s, area, train = load()
    s_tr, s_te, a_te = s[train], s[~train], area[~train]
    if args.max_train:
        s_tr = s_tr[:args.max_train]
    print(f"train {len(s_tr)} cellules, test {len(s_te)} cellules (jours > 7)", flush=True)
    path = OUT / f"{args.tag}.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    for arm in args.arms:
        for d in ([0] if arm == "mean" else args.dims):
            t0 = time.time()
            if arm in ("mean", "pca", "zernike"):
                rec, real_d = run_linear(arm, d, s_tr, s_te).reshape(-1, P, P), d
            else:
                rec, real_d = run_neural(arm, d, s_tr, s_te, args.epochs, lr=args.lr, device=args.device)
            res = evaluate(s_te, rec, a_te)
            res["time_s"] = time.time() - t0
            results[f"{arm}_{real_d}"] = dict(arm=arm, dim=real_d, **res)
            r = res["all"]
            print(f"{arm:10s} d={real_d:3d}  nrmse {r['nrmse']:.3f}  mass_err {r['mass_err']:.3f}  "
                  f"IoU {r['iou']:.3f}  peak_err {r['peak_err']:.3f}  ({res['time_s']:.0f}s)", flush=True)
            path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
