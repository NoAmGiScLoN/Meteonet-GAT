# E5 — Adaptive number of sources per cell (session B)

**Status:** archived outputs of the code at tag `v3-sessionB` (commit `f67b41e`). Part of the
[README](../../README.md) study. This run supports decisions D23–D25.

## Question

1. **D23.** Until E4 every size class used a fixed number of sources (K = 8 / 24 / 48 / 64), and the error was
   still falling at that cap. Does growing K per cell until it reaches the target nrmse ≤ 0.1 (D21) do better than
   fixed K per class?
2. **D24.** A compression limit keeps the parameter count of a cell below a fraction of its pixel count. Which
   fraction is right?
3. **D25.** Is a cap of 128 sources enough for the largest cells, or does 256 pay off?
4. Does 2017 confirm 2016?

## Setup

- **Fitter.** `src/adaptive_fit.py` with the CLI `src/adaptive_run.py`, arm diff7 (D17), default log-w table (D18),
  iterations per wave of D22, fixed learning rate 0.05 (D20).
  - Waves follow the schedule 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 (then 192, 256 for the floor test).
  - **Stop rules,** checked after every wave in this order: diverged > target (whole-crop nrmse ≤ 0.1) > cap >
    compression > schedule exhausted.
  - **Compression limit:** K_limit = max(1, ⌊frac · area / 7⌋); every wave is clipped to it.
  - **Pruning:** after each wave, sources with peak softplus(A) below 1 % of the cell's peak rate are dropped
    (at least one is kept).
  - **Residual statistics** per cell (`src/residual_stats.py`): peak and mass bias, residual spread inside the cell
    and on a 2-px edge band, spill outside the cell, spectral slope, nrmse on the mask.
- **Samples.** `sample_40` (2016) and `sample_40_2017`, 40 cells per class, the same cells as E3.
- **Runs** (`scripts/run_jetson_sessionB.sh`):

  | Stage | Folder | Sample | Classes | Compression | Cap |
  |---|---|---|---|---|---|
  | 1 | `adaptive_2016` | 2016 | all | 0.25 | 128 |
  | 2 | `adaptive_2017` | 2017 | all | 0.25 | 128 |
  | 3 | `floor256` | 10 largest 2016 cells | ≥ 5000 | 0.25 | 256 |
  | 5 | `adaptive_2016_c050` | 2016 | < 5000 px | 0.5 | 128 |
  | 6 | `adaptive_2017_c050` | 2017 | < 5000 px | 0.5 | 128 |

  Stages 5–6 were added after stages 1–3 (see Key results). The ≥ 5000 px class was not rerun: its compression
  limit is at least 178 at 0.25 and 357 at 0.5, always above the cap, so it never binds.
- **Hardware:** Jetson Orin Nano.
  - Stages 1–3: 6 h 09 min, 2026-11-22 20:27 – 2026-11-23 02:36, commit `6193c1b`.
  - Stages 5–6: 1 h 41 min, 2026-11-23 11:49–13:30, commit `f67b41e`.
  - GPU memory peak 499 MB (2017, ≥ 5000 px); no out-of-memory retry and no stage retry.

## Reproduce

```bash
git worktree add ../gat-v3 v3-sessionB && cd ../gat-v3
# git-ignored inputs in results/03_variable_sources/: sample_40.pkl, sample_40_2017.pkl,
#   diffusion_profiles.pt (scripts/build_samples.py); the log-w table is built on first use
mkdir -p results/07_sessionB/logs
STAGES="0 1 3 2 4" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_sessionB.sh \
    > results/07_sessionB/logs/run.log 2>&1 &
STAGES="5 6 4" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_sessionB.sh \
    > results/07_sessionB/logs/run_c050.log 2>&1 &
```

At `v3-sessionB` the default compression is still 0.25, so stages 0–3 reproduce as run. From D24 the default is
0.5, and the script on `main` passes `--compression-frac 0.25` to stages 0–3 explicitly.

## Key results

### First run: compression 0.25 hurt the cells under 1000 px

Median nrmse of the same cells, adaptive (stages 1–2) against the E3 fixed-K fits at the class cap
([`adaptive_2016/summary_error.csv`](results/adaptive_2016/summary_error.csv),
[`adaptive_2017/summary_error.csv`](results/adaptive_2017/summary_error.csv)). Values are 2016 / 2017:

| Class | Adaptive K | Adaptive nrmse | Fixed K (E3) | Fixed nrmse (E3) | Main stop |
|---|---|---|---|---|---|
| 50–200 | 2 / 2 | 0.255 / 0.280 | 8 | 0.093 / 0.141 | compression 100 % / 100 % |
| 200–1000 | 12 / 12.5 | 0.132 / 0.172 | 24 | 0.085 / 0.112 | compression 83 % / 95 % |
| 1000–5000 | 55.5 / 48 | 0.095 / 0.131 | 48 | 0.117 / 0.144 | target 65 % / compression 80 % |
| ≥ 5000 | 127 / 127 | 0.110 / 0.137 | 64 | 0.159 / 0.196 | cap 70 % / 80 % |

- **Large cells gain:** −0.052 / −0.057 paired median on ≥ 5000 px, no cell worse.
- **Small cells lose:** every 50–200 px cell is worse, by +0.178 paired median in 2016.
- **The compression limit is the cause.** At 0.25 a 76 px cell may use only K = 2. The E3 fixed-K curves show that
  50–200 px cells need 0.5–0.95 parameters per pixel to reach 0.1, and 200–1000 px cells about 0.5.

### Choosing the compression limit from the E3 curves

Best nrmse per cell within n_params ≤ frac · area, read on the E3 fixed-K curves (median, 2016 / 2017). The curves
stop at the E3 class caps, so the values for 0.75 and 1.0 are partly limited by the cap:

| Class | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| 50–200 | 0.255 / 0.286 | 0.168 / 0.193 | 0.117 / 0.159 | 0.096 / 0.148 |
| 200–1000 | 0.156 / 0.187 | 0.109 / 0.134 | 0.094 / 0.120 | 0.085 / 0.112 |
| 1000–5000 | 0.125 / 0.166 | 0.117 / 0.144 | 0.117 / 0.144 | 0.117 / 0.144 |

0.5 was chosen for every class. 50–200 px cells keep a visible error at 0.5, which was accepted because small cells
weigh little in the global movement of the cell field that the GAT is meant to learn.

### Rerun at compression 0.5 (stages 5–6)

Same cells, 2016 / 2017 ([`adaptive_2016_c050/summary_error.csv`](results/adaptive_2016_c050/summary_error.csv),
[`adaptive_2017_c050/summary_error.csv`](results/adaptive_2017_c050/summary_error.csv)):

| Class | Median K | Median nrmse | p90 nrmse | Paired change vs 0.25 | Paired change vs fixed K | Target reached | Compression stop |
|---|---|---|---|---|---|---|---|
| 50–200 | 5 / 5 | 0.164 / 0.174 | 0.224 / 0.290 | −0.105 / −0.090 | +0.049 / +0.047 | 10 % / 8 % | 90 % / 93 % |
| 200–1000 | 22 / 24 | 0.089 / 0.098 | 0.118 / 0.159 | −0.048 / −0.066 | 0.000 / −0.005 | 80 % / 53 % | 20 % / 48 % |
| 1000–5000 | 64 / 85.5 | 0.090 / 0.092 | 0.099 / 0.123 | −0.003 / −0.038 | −0.032 / −0.061 | 93 % / 70 % | 0 % / 8 % |

- **200–1000 px matches fixed K** and **1000–5000 px beats it**, in both years.
- **50–200 px stays 0.05 behind fixed K=8,** with 35 parameters against 56. This is the accepted trade.
- **2016 and 2017 now agree:** the medians of the classes from 200 to 5000 px differ by at most 0.009, against
  0.036–0.040 at 0.25.
- **Biases are small:** median mass bias −0.8 % to −1.0 % above 200 px and −2.7 % / −3.1 % on 50–200 px; IoU
  0.94–0.95 above 200 px; spill outside the cell ≤ 1.1 %. Both biases stay negative: the reconstruction slightly
  underestimates the rain.
- **Pruning** dropped at least one source in 17 % / 27 % of cells.
- **Run time** was set by the number of waves, not by K: one group wave costs about the same whatever K.

### Floor test: cap 256 on the 10 largest cells

[`floor256/waves_ge5000.csv`](results/floor256/waves_ge5000.csv), median nrmse after each wave:

| K | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|
| nrmse | 0.187 | 0.153 | 0.136 | 0.113 | 0.102 |

- At the end, the median is 0.098 and 5 of 10 cells reach 0.1 (1 of 10 at cap 128 in stage 1).
- Cost: about 40 % more time per cell than cap 128; GPU memory peak 324 MB.
- Not adopted (D25), so the ≥ 5000 px class stays at 0.110 / 0.137.

### Adopted settings, all classes (D23–D25)

Stages 5–6 below 5000 px, stages 1–2 for ≥ 5000 px. Over the 160 cells per year: median nrmse 0.098 / 0.120;
53 % / 38 % of cells reach 0.1 and 81 % / 70 % reach 0.15.

## Decisions

- [D23](../../README.md#9-decision-log) **Adaptive K is the default fitter.** It replaces fixed K per class:
  target nrmse ≤ 0.1 (D21), pruning at 1 % of the cell peak, residual statistics saved for every cell.
- [D24](../../README.md#9-decision-log) **Compression limit 0.5 in every class** (n_params ≤ 0.5 · area). 0.25 held
  50–200 px cells at K ≈ 2. 0.5 lets 200–1000 px cells match fixed K, and small cells were judged less important
  for the global movement.
- [D25](../../README.md#9-decision-log) **Keep the cap at 128 sources.** Cap 256 brings the ten largest cells from
  0.136 to 0.098 but was not adopted; 70–80 % of ≥ 5000 px cells stop at the cap.

## Caveats

- **The ≥ 5000 px class is the weak point.** It is the class that drives cell movement, and it stays about 0.01–0.04
  above the target with cap 128. Cap 256 is the measured fix if this matters for the GAT.
- **50–200 px results depend on whole-crop nrmse.** Their error is mostly at the sharp 1 mm/h edge (nrmse on the
  mask 0.12 / 0.15 against 0.16 / 0.17 on the crop).
- **The compression choice was made from E3 curves capped at K = 8 / 24 / 48,** then checked by the rerun.
- **≥ 5000 px was not rerun at 0.5.** Its results come from stages 1–2, where compression could not bind.
- **The floor test used 10 cells**, the largest of 2016 only.
- **`smoke/`** holds the 1-cell check of stage 0. **Not archived:** the profile tables (`*.pt`, git-ignored).
