# E3 — 7-parameter diffusion sources (diff7) and iterations per wave

**Status:** archived outputs of the code at tag `v1-diff7-convergence` (commit `1ff7c40`). Part of the
[README](../../README.md) study. This run supports decisions D17–D19.

## Question

1. **D17:** can `τ` be dropped from the diffusion source without losing accuracy? diff7 = 7 parameters per source
   `(x, y, A, a, b, θ, w)`, with `τ` fixed at 0.5. It is compared with the 8-parameter `diffusion` source (D11
   proposed this from the E2b ablation).
2. **D18:** which `w` table should diff7 use?
3. **D19:** are 200 Adam iterations per wave enough?

## Setup

- **Data:** the same 160 cells as E2b (40 per size class, January and July 2016, `sample_40.pkl`) and the 160
  cells of April and September 2017 (`sample_40_2017.pkl`).
- **diff7 profile table:** the same PDE as the default table, with τ = 0.5 exactly and log-spaced `w` nodes
  (`src/profile_lut.py`; the file name carries a hash of its settings).
  - Default table: `w` 0.02–0.5, 16 nodes, key `ace17c5890`.
  - Wide table: `w` 0.005–0.5, 24 nodes, key `1e980880f3`.
- **Fits:** as in E2b. Progressive K up to 8 / 24 / 48 / 64 per class, 200 iterations per wave, Adam with lr 0.05.
- **Pass test for diff7 (checkpoint 1d):**
  - the median over cells of (diff7 − diffusion) nrmse on the same cell;
  - at every K ≥ 6;
  - one-sided: diff7 may be better, but never worse by more than 0.005.
- **Convergence check (checkpoint 2):**
  - 10 cells per class, seed 0, diff7, each class at its cap;
  - 200 / 500 / 2000 iterations per wave on the same cells;
  - choose the smallest level whose paired median gain to the next level is below 0.005.
- **Hardware:** Jetson Orin Nano, torch 2.15.0+cu132. 6 h 47 min, 2026-11-20 14:10–20:57.

## Reproduce

```bash
git worktree add ../gat-v1 v1-diff7-convergence && cd ../gat-v1
# copy the git-ignored inputs into results/03_variable_sources/:
#   sample_40.pkl, sample_40_2017.pkl, diffusion_profiles.pt (built by scripts/build_samples.py)
mkdir -p results/05_diff7_convergence/logs
STAGES="0 1 2 3 4 5" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_diff7.sh \
    > results/05_diff7_convergence/logs/run.log 2>&1 &
```

The script builds the log-w tables if they are missing. Outputs land in `results/05_diff7_convergence/`; this
folder is that output, moved here.

## Key results

### The baseline reproduces

The fresh 8-parameter run matches the archived E2b run in every class and at every K: 38 of 38 rows are within
tolerance, and the largest difference in medians is 0.004
([`results/compare_main/reproduction.csv`](results/compare_main/reproduction.csv)).

### D17 — diff7 against the 8-parameter source

Median per-cell difference, diff7 − diffusion (positive = diff7 worse). Sources:
[`compare_main/main_paired_delta.csv`](results/compare_main/main_paired_delta.csv) and
[`compare_2017/season2017_diff7_paired_delta.csv`](results/compare_2017/season2017_diff7_paired_delta.csv).
For 2017 the diffusion arm is the archived E2b `season2017` run, with the same cell ids.

| K | 50–200 | 200–1000 | 1000–5000 | ≥ 5000 (2016) | ≥ 5000 (2017) |
|---|---|---|---|---|---|
| 6 | +0.001 | +0.001 | +0.005 | +0.020 | +0.013 |
| 8 | −0.004 | +0.005 | +0.001 | +0.018 | +0.007 |
| 12 | | +0.003 | −0.000 | +0.011 | +0.009 |
| 24 | | +0.002 | +0.003 | +0.007 | +0.008 |
| 48 | | | +0.003 | +0.003 | +0.005 |
| 64 | | | | +0.002 | +0.003 |

Verdict of the one-sided test
([`compare_main/main_equivalence.csv`](results/compare_main/main_equivalence.csv),
[`compare_2017/season2017_diff7_equivalence.csv`](results/compare_2017/season2017_diff7_equivalence.csv)):

| Class | 2016 worst | 2016 | 2017 worst | 2017 |
|---|---|---|---|---|
| 50–200 | +0.0007 | pass | +0.0028 | pass |
| 200–1000 | +0.0050 | fail (by 0.00003) | +0.0044 | pass |
| 1000–5000 | +0.0051 | fail (by 0.00005) | +0.0055 | fail |
| ≥ 5000 | +0.0200 | fail | +0.0127 | fail |

Parameters needed for nrmse ≤ 0.15 (median; share of cells reaching it), 2016, from
[`logs/1_main.log`](results/logs/1_main.log):

| Class | diffusion (8 per source) | diff7 (7 per source) |
|---|---|---|
| 50–200 | 48 (85 %) | 42 (78 %) |
| 200–1000 | 96 (100 %) | 84 (92 %) |
| 1000–5000 | 256 (80 %) | 224 (75 %) |
| ≥ 5000 | 512 (40 %) | 448 (40 %) |

### D18 — wide w table

Using the wide table (0.005–0.5, 24 nodes) instead of the default ([`compare_wext/`](results/compare_wext/)):

- 200–1000 px gets worse: worst per-cell difference +0.008, against +0.005 with the default table.
- Other classes are about the same.
- The share of sources at the sharp end of the range falls from 23 % to 13 %
  ([`main_w_pileup.csv`](results/compare_main/main_w_pileup.csv),
  [`main_wext_w_pileup.csv`](results/compare_wext/main_wext_w_pileup.csv)), but the error does not improve.

A local CPU test on three classes gave the same result before the run.

### D19 — iterations per wave

Median nrmse at each class cap, 10 cells per class
([`convergence/convergence.csv`](results/convergence/convergence.csv),
[`convergence/convergence_choice.csv`](results/convergence/convergence_choice.csv)):

| Class | 200 | 500 | 2000 | Median gain 200→500 | p90 gain 200→500 | Chosen |
|---|---|---|---|---|---|---|
| 50–200 | 0.134 | 0.111 | 0.119 | 0.0034 | 0.084 | 200 |
| 200–1000 | 0.075 | 0.070 | 0.075 | 0.0050 (just below the threshold) | 0.022 | 200 |
| 1000–5000 | 0.111 | 0.107 | 0.112 | 0.0073 | 0.019 | 500 |
| ≥ 5000 | 0.173 | 0.164 | 0.165 | 0.0068 | 0.019 | 500 |

- **2000 iterations is never better than 500.** 20–50 % of cells are worse by more than 0.002 at 2000 than at
  500. With a fixed learning rate Adam keeps moving the sources, and later waves then place new sources
  differently.
- **Fit time grows with the iteration count.** At 500 iterations the fits take 2.5–2.8 × longer than at 200, and
  4–13 × longer at 2000.

## Decisions

- [D17](../../README.md#9-decision-log) **Adopt diff7.** Judged at equal parameter budget, which is the question
  behind D11: it reaches the same error with 12.5 % fewer parameters. The strict per-K test fails:
  - by a hair on 200–1000 and 1000–5000 px;
  - clearly on ≥ 5000 px at K ≤ 24.
  
  This is accepted. Large cells are represented at K ≥ 48, where the gap is at most 0.005.
- [D18](../../README.md#9-decision-log) **Keep the default diff7 table:** `w` 0.02–0.5, 16 log nodes.
- [D19](../../README.md#9-decision-log) **Use 200 iterations per wave below 1000 px and 500 at 1000 px and above.**

## Caveats

- **Convergence rests on 10 cells per class.** The 200–1000 px choice sits on the threshold (0.00499), and the
  median hides large tail gains (p90 0.02–0.08).
- **The learning rate was not varied.** Because 2000 iterations is worse than 500, a decaying learning rate is the
  likely lever, rather than more iterations. This is not tested.
- **Pile-up remains.** 23 % of diff7 sources sit at the sharpest `w`. Widening the range did not help, which points
  to an optimisation effect at the edge of the sigmoid rather than a missing shape.
- **The comparison is at equal K.** At equal K diff7 has 12.5 % fewer parameters. D17 relies on the
  equal-parameter view, from the parameters-needed table.
- **The `summary_equal_params.csv` files are empty.** `summarize` only pairs gauss with diffusion; use the
  `compare_*` folders instead.
- **Not archived:** the profile tables (`*.pt`, git-ignored; rebuilt by `scripts/build_samples.py --logw`).
  The `smoke*` folders are the 1-cell check of stage 0.
