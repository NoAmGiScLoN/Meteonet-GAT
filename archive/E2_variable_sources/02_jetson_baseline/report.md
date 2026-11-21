# E2b — Variable-size sources, Jetson baseline (full programme)

**Status:** archived outputs of the tagged code (`v0-jetson-baseline`). Part of the [README](../../../README.md) study (§7).
This run supports D10 and D11, and the stability evidence for D16.

## Question

Seven steps in one session (`scripts/run_jetson.sh`):

1. Main: Gaussian vs diffusion sources at equal parameter budget.
2. Stability: is the same cell described the same way 5 min later?
3. Ablation: which of τ and w matters?
4. Generalisation: does the result hold in April and September 2017?
5. Restarts: does it depend on the initialisation (seed 1)?
6–7. Re-runs of Experiment 1: gmm d = 48, and ae d = 48 at 60 epochs.

## Setup

- **Data:** 160 cells (40 per bin) from January and July 2016 (`sample_40.pkl`); 160 cells from April and September 2017
  (`sample_40_2017.pkl`); 25 stability pairs per bin (same cell at t and t + 5 min, mutual best IoU ≥ 0.5).
  Bins: 50–200, 200–1,000, 1,000–5,000 and ≥ 5,000 px.
- **Arms:** `gauss` (6 parameters per source), `diffusion` (8: τ and w), `diff_tau` and `diff_w` (7 each, one extra parameter free).
- **Budgets:** progressive K ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}, capped at 8 / 24 / 48 / 64 by bin; 200 iterations per fit.
  Stability uses K = 3 / 6 / 12 / 24.
- **Hardware:** NVIDIA Jetson Orin Nano (JetPack 7.2 per README §12); torch 2.15.0+cu132, numpy 2.5.5 (`logs/run_jetson.log` header).

## Reproduce

```bash
git worktree add ../gat-v0 v0-jetson-baseline
cd ../gat-v0
pip install -r requirements.txt   # JetPack 7.x: pip install torch --index-url https://download.pytorch.org/whl/cu132
python scripts/build_samples.py   # inputs to results/02_fixed_vector_benchmark and results/03_variable_sources
bash scripts/run_jetson.sh        # about 7.5 h; PY and DEV overridable
```

Outputs: `../gat-v0/results/04_jetson_run/{main,stability,ablation,season2017,restart_seed1}/`, logs in
`results/04_jetson_run/logs/`, and `results/02_fixed_vector_benchmark/results_jetson.json`.

## Session timing

From the `début` / `fin` lines of [`logs/run_jetson.log`](results/logs/run_jetson.log) and the per-fit timers of each step log:

| Step | Output | Clock | Timer sum |
|---|---|---|---|
| 1 main | `main/` | 10:15–11:43 | 88.5 min |
| 2 stability | `stability/` | 11:43–13:01 | 78.1 min |
| 3 ablation | `ablation/` | 13:01–14:36 | 94.7 min |
| 4 season 2017 | `season2017/` | 14:36–16:13 | 96.2 min |
| 5 restart | `restart_seed1/` | 16:13–17:46 | 93.5 min |
| 6 gmm d = 48 | `results_jetson.json` | 17:46–17:47 | 43 s |
| 7 ae d = 48 | `results_jetson.json` | 17:47–17:48 | 68 s |

Wall-clock for the whole session is 7 h 33 min. The per-fit timers sum to 7.5 h, matching the README's estimate.

## Key results

### Step 1: main, equal budget (median nrmse on √rate; share of cells where diffusion is better)

Source: [`results/main/summary_equal_params.csv`](results/main/summary_equal_params.csv).

| Bin | Params | Gauss | Diffusion | Diffusion better |
|---|---|---|---|---|
| 50–200 | 24 | 0.296 | 0.218 | 0.950 |
| 50–200 | 48 | 0.239 | 0.137 | 1.000 |
| 200–1,000 | 24 | 0.316 | 0.292 | 0.700 |
| 200–1,000 | 48 | 0.255 | 0.213 | 0.925 |
| 200–1,000 | 96 | 0.204 | 0.146 | 0.975 |
| 1,000–5,000 | 24 | 0.397 | 0.436 | 0.350 |
| 1,000–5,000 | 48 | 0.319 | 0.321 | 0.425 |
| 1,000–5,000 | 96 | 0.247 | 0.233 | 0.700 |
| 1,000–5,000 | 192 | 0.196 | 0.173 | 0.925 |
| ≥ 5,000 | 24 | 0.514 | 0.621 | 0.175 |
| ≥ 5,000 | 48 | 0.417 | 0.450 | 0.175 |
| ≥ 5,000 | 96 | 0.325 | 0.343 | 0.225 |
| ≥ 5,000 | 192 | 0.247 | 0.242 | 0.400 |
| ≥ 5,000 | 384 | 0.194 | 0.182 | 0.625 |

Parameters needed for nrmse ≤ 0.15 (median; share of cells reaching it), from `main/summary_params_needed.csv`:

| Bin | Gauss | Diffusion |
|---|---|---|
| 50–200 | 48 (5 %) | 48 (85 %) |
| 200–1,000 | 120 (15 %) | 96 (100 %) |
| 1,000–5,000 | 288 (30 %) | 256 (80 %) |
| ≥ 5,000 | 384 (2 %) | 512 (42 %) |

### Step 2: stability (25 pairs per bin)

Source: [`stability/stability.csv`](results/stability/stability.csv). Ratio = matching cost of the same cell at t and t + 5 min
divided by the cost against a different cell of the same bin. Values are the median of per-pair ratios (the README §7.6
definition; the ratio of mean costs gives other values).

| Bin | K | Median ratio, gauss | Median ratio, diffusion | Same cell closer, gauss / diffusion |
|---|---|---|---|---|
| 50–200 | 3 | 0.291 | 0.344 | 84 % / 88 % |
| 200–1,000 | 6 | 0.527 | 0.558 | 88 % / 92 % |
| 1,000–5,000 | 12 | 0.432 | 0.483 | 100 % / 96 % |
| ≥ 5,000 | 24 | 0.296 | 0.255 | 100 % / 100 % |

### Step 3: ablation (median nrmse at the largest K of each bin)

Sources: `main/fits_*.csv` (gauss, diffusion) and `ablation/fits_*.csv` (τ-only, w-only).

| Bin | K | Gauss (main) | Only τ free | Only w free | τ and w (main diffusion) |
|---|---|---|---|---|---|
| 50–200 | 8 | 0.239 | 0.140 | 0.099 | 0.103 |
| 200–1,000 | 24 | 0.172 | 0.108 | 0.089 | 0.085 |
| 1,000–5,000 | 48 | 0.167 | 0.132 | 0.120 | 0.120 |
| ≥ 5,000 | 64 | 0.194 | 0.164 | 0.158 | 0.159 |

The ablation arms have 7 parameters per source; the last column is the main diffusion arm (8 per source) at the same K.
Parameters needed for ≤ 0.15: `diff_w` 42 / 84 / 224 / 448 (85 / 95 / 78 / 38 %), `diff_tau` 42 / 112 / 224 / 448 (57 / 80 / 75 / 25 %).

### Steps 4 and 5: another season, and restarts, at one budget per bin

Sources: `season2017/summary_equal_params.csv`, `restart_seed1/summary_equal_params.csv`, and the main table above.

| Bin | Budget | Main g / d | Season 2017 g / d | Restart (seed 1) g / d |
|---|---|---|---|---|
| 50–200 | 48 | 0.239 / 0.137 | 0.241 / 0.181 | 0.232 / 0.136 |
| 200–1,000 | 96 | 0.204 / 0.146 | 0.223 / 0.178 | 0.203 / 0.143 |
| 1,000–5,000 | 192 | 0.196 / 0.173 | 0.227 / 0.214 | 0.198 / 0.177 |
| ≥ 5,000 | 384 | 0.194 / 0.182 | 0.215 / 0.217 | 0.193 / 0.187 |

Restart reproducibility, recomputed per cell at the largest K (main vs restart): correlation 0.89–1.00 (gauss 0.970–0.998,
diffusion 0.887–0.988), median absolute difference 0.002–0.010.

### Steps 6 and 7: Experiment 1 re-runs at d = 48

- gmm (Gaussian decoder), lr 5e-4, 25 epochs: nrmse 1.000, mass error 1.000, IoU 0.000. Training MSE is 0.6224 from
  epoch 6 onward ([`logs/6_gmm48.log`](results/logs/6_gmm48.log)). Collapsed.
- ae, 60 epochs: nrmse 0.191, mass error 0.034, IoU 0.844, peak error 0.076 ([`logs/7_ae48.log`](results/logs/7_ae48.log)).

## Decisions

- [D10](../../../README.md#9-decision-log) per-source diffusion profile instead of Gaussian ([§7.5](../../../README.md#75-results))
- [D11](../../../README.md#9-decision-log) drop τ, keep w ([§7.5 step 3](../../../README.md#step-3--ablation-of-τ-and-w))
- [D16](../../../README.md#9-decision-log) field-space loss, because sources are non-identifiable ([§7.6](../../../README.md#76-step-2--temporal-stability))
- The Jetson choice itself (D13) is supported by the CPU run in [E2a](../01_cpu_prelim/report.md).

## Caveats

- **Sample size:** 40 cells per bin and 25 stability pairs per bin. Medians only (README §10).
- **The ablation summary is empty.** `ablation/summary_equal_params.csv` has no rows (the step's log prints "Empty DataFrame").
  The ablation numbers above come from `fits_*.csv`.
- **The closing line of the archived log differs from the tagged script.** `logs/run_jetson.log` ends by naming
  `outputs_varsrc_jetson/` and `outputs_bench/`, while the tagged `run_jetson.sh` names `results/04_jetson_run/`.
  The archived run therefore used another revision of the script. Step names and outputs match; the commands were not compared line by line.
- **Not archived:** `stability_pairs_25.pkl` and `diffusion_profiles.pt` (git-ignored inputs, rebuilt by `scripts/build_samples.py`).
- **Not re-derived here:** README §7.5 claims of a median gain of 0 from keeping the best of two fits, and of larger 2017 cells
  (median 16,100 px against 8,900 px in 2016).
