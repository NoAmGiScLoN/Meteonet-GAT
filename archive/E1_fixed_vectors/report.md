# E1 — Fixed-length vectors

**Status:** archived outputs of the tagged code (`v0-jetson-baseline`). Part of the [README](../../README.md) study (§6).

## Question

With every cell forced into the same d-dimensional vector, which encoding reconstructs a cell best: PCA, Zernike,
a Gaussian decoder, a diffusion decoder or a CNN autoencoder? The answer supports dropping fixed-length vectors
(D8). Decisions D5–D8 and D12.

## Setup

- **Data:** 11,874 cells (≥ 50 px), one map per hour, first 10 days of January and July 2016. Train: days 1–7
  (8,986 cells). Test: days 8–10 (2,888 cells).
- **Input:** 64 × 64 patches in `results/02_fixed_vector_benchmark/cells.npz` (not archived); scored at 32 × 32
  (`--res` default), on √rate.
- **Arms:** mean (d = 0), PCA, Zernike, `gmm` (the README's "Gaussian decoder"), `diffusion` (shared-diffusivity
  decoder), `ae` (CNN autoencoder).
- **Training:** neural arms 25 epochs in the CPU run. Jetson re-runs: gmm d = 48 at 25 epochs, lr 5e-4; ae d = 48 at 60 epochs.
- **Hardware:** CPU run on the development machine (macOS 12, CPU only; README §12). Re-runs on the Jetson Orin Nano (CUDA).

## Reproduce

```bash
git worktree add ../gat-v0 v0-jetson-baseline
cd ../gat-v0
pip install -r requirements.txt
python scripts/build_samples.py   # writes results/02_fixed_vector_benchmark/cells.npz (needs MeteoNet in DATA/)
python src/compression_benchmark.py --arms mean pca zernike ae gmm diffusion --dims 6 12 18 30 48   # tagged README command; see Caveats
# Jetson re-runs (steps 6-7 of scripts/run_jetson.sh)
python src/compression_benchmark.py --arms gmm --dims 48 --epochs 25 --lr 5e-4 --device cuda --tag results_jetson
python src/compression_benchmark.py --arms ae --dims 48 --epochs 60 --device cuda --tag results_jetson
```

Outputs: `../gat-v0/results/02_fixed_vector_benchmark/results.json` and `results_jetson.json`. Redirect stdout to
`benchmark.log` to get the training log. The gmm arm was removed after the tag (commit `b6ff2e5`, D12).

## Key results

nrmse on √rate, all 2,888 test cells ([`results/results.json`](results/results.json), [`results/results_jetson.json`](results/results_jetson.json)):

| Arm | d = 6 | d = 18 | d = 48 |
|---|---|---|---|
| PCA | 0.432 | 0.302 | 0.200 |
| Zernike | 0.614 | 0.466 | 0.312 |
| Gaussian decoder (`gmm`) | 0.405 | 0.296 | collapsed (1.000) |
| Diffusion decoder (shared diffusivity) | 0.611 | 0.331 | 0.252 |
| CNN autoencoder | **0.350** | **0.230** | 0.214 (25 ep.) → **0.191** (60 ep.) |

- Mean baseline (d = 0): 0.692. PCA and Zernike also at d = 12 and 30 (PCA 0.347 and 0.246; Zernike 0.526 and 0.379).
- AE at d = 48 (60 epochs), by size bin: 0.149 / 0.221 / 0.324 / 0.381 (50–200 / 200–1,000 / 1,000–5,000 / ≥ 5,000 px).
  Mass error 0.034, IoU 0.844, peak error 0.076 ([`results/results_jetson.json`](results/results_jetson.json)).
- gmm at d = 48 collapses: the output is zero (mass error 1.000, IoU 0.000). Training MSE stays at 0.6224 from epoch 6
  on, both at lr 2e-3 ([`results/benchmark.log`](results/benchmark.log)) and at lr 5e-4 on the Jetson
  ([`../E2_variable_sources/02_jetson_baseline/results/logs/6_gmm48.log`](../E2_variable_sources/02_jetson_baseline/results/logs/6_gmm48.log)).
- Wall-clock per arm in the CPU log sums to 2.06 h (gmm d = 48: 1,716 s; diffusion d = 18: 1,777 s).

## Decisions

- [D5](../../README.md#9-decision-log) compression, not prediction ([§1.3](../../README.md#13-scope-of-this-repository-the-representation-and-why-it-must-be-accurate))
- [D6](../../README.md#9-decision-log) evaluate by reconstruction ([§1.3](../../README.md#13-scope-of-this-repository-the-representation-and-why-it-must-be-accurate))
- [D7](../../README.md#9-decision-log) √rate errors, size bins, split by day, equal budget ([§5](../../README.md#5-evaluation-protocol))
- [D8](../../README.md#9-decision-log) drop the fixed-length vectors ([§6.5](../../README.md#65-why-fixed-length-vectors-were-abandoned))
- [D12](../../README.md#9-decision-log) drop the Gaussian fixed-vector decoder (gmm collapsed at d = 48)

## Caveats

- **The invocation is not fully recorded.** `results.json` has gmm, diffusion and ae only at d = 6, 18 and 48, while the
  tagged README command lists dims 6 12 18 30 48 for all arms. The archived files come from more than one invocation.
  The neural arms used 25 epochs, while the tagged default is 40.
- **The "diffusion decoder" here is the shared-diffusivity decoder of Experiment 1** (README §6.2). It is not the
  per-source diffusion of Experiment 2 (D10).
- **Not like-for-like with Experiment 2:** the AE is scored on 32 × 32 patches, the sources at native resolution
  (README §8).
- **One seed** (`seed=0`) per configuration; no repeats.
- **The README's "≈ 1.5 h CPU" estimate does not match the log** (2.06 h).
