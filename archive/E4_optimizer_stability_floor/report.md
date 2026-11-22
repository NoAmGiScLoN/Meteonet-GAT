# E4 — Optimizer schedule, diff7 stability baseline and noise floor

**Status:** archived outputs of the code at tag `v2-sessionA` (commit `12bfc85`). Part of the
[README](../../README.md) study. This run supports decisions D20–D22.

## Question

1. **D20.** E3 found that 2000 iterations per wave is worse than 500. Does a decaying learning rate (cosine,
   restarted each wave) do better than the fixed rate 0.05?
2. **D22.** Is the 200 vs 500 iteration choice of D19 confirmed? This is checked on the same cells.
3. **Stability.** How stable is diff7 from one frame to the next, compared with the archived Gaussian and
   8-parameter sources? This is the baseline for the warm-start comparison planned next.
4. **D21.** What error does the data itself allow, and is the adaptive target nrmse ≤ 0.1 sensible?

## Setup

- **Optimizer grid.** `src/optimizer_check.py`:
  - 10 cells per size class (seed 0, same selection as the E3 convergence check), diff7, K = class cap;
  - settings fixed 0.05 at {200, 500}, cosine 0.05 at {200, 500} and cosine 0.1 at {200} iterations per wave;
  - cosine floor at 1 % of the starting rate;
  - every setting fitted on identical cells and groups.
- **Adoption rule.** Applied per class, against the D19 setting (fixed 0.05 at 200 below 1000 px, 500 above). A
  setting is adopted if either holds:
  - its paired median gain is ≥ 0.005 at equal iterations **and** its 10th-percentile gain is ≥ −0.002;
  - or it matches fixed@500 within ±0.002 at 200 iterations, on cells of 1000 px and above.
- **Stability.** `src/stability_test.py`:
  - `--arms diff7 --iters-by-bin`, i.e. the D19 iteration counts;
  - the same 25 pairs per class as E2b (same cell at t and t + 5 min), at K = 3 / 6 / 12 / 24.
- **Noise floor.** `src/noise_floor.py`, run on the Mac on the 2016 and 2017 samples (40 cells per class):
  - quantization (rates in steps of 0.12 mm/h), by Monte Carlo and a delta-method formula;
  - texture proxies on the eroded cell interior: normalised Gaussian smoothing at σ 0.5 and 1 px, and a 3×3
    median;
  - the full-crop versions of the same proxies, kept as edge-plus-texture upper bounds.
- **Hardware:** Jetson Orin Nano. 4 h 23 min, 2026-11-22 13:33–17:57. The noise floor ran on the Mac CPU in
  under a minute.

## Reproduce

```bash
git worktree add ../gat-v2 v2-sessionA && cd ../gat-v2
# git-ignored inputs in results/03_variable_sources/: sample_40.pkl, sample_40_2017.pkl,
#   stability_pairs_25.pkl, diffusion_profiles.pt (scripts/build_samples.py)
mkdir -p results/06_sessionA/logs
STAGES="0 1 2" PY=~/gatenv/bin/python nohup bash scripts/run_jetson_sessionA.sh \
    > results/06_sessionA/logs/run.log 2>&1 &
python src/noise_floor.py --samples results/03_variable_sources/sample_40.pkl \
    results/03_variable_sources/sample_40_2017.pkl --out results/06_sessionA/noise_floor
```

## Key results

### D20 — optimizer

Median nrmse at each class cap
([`optimizer/optimizer_grid.csv`](results/optimizer/optimizer_grid.csv)). Bold marks the D19 setting:

| Class | fixed 200 | fixed 500 | cosine 0.05 / 200 | cosine 0.05 / 500 | cosine 0.1 / 200 |
|---|---|---|---|---|---|
| 50–200 | **0.134** | 0.111 | 0.145 | 0.131 | 0.121 |
| 200–1000 | **0.071** | 0.066 | 0.087 | 0.072 | 0.093 |
| 1000–5000 | 0.115 | **0.107** | 0.119 | 0.108 | 0.114 |
| ≥ 5000 | 0.174 | **0.162** | 0.173 | 0.172 | 0.178 |

Choice table ([`optimizer/optimizer_choice.csv`](results/optimizer/optimizer_choice.csv)):

- **No cosine setting is adopted in any class,** and none matches fixed@500 at 200 iterations.
- **At equal cost, cosine is worse by 0.004–0.017** in paired median.
- **The one positive median fails the tail check.** Cosine 0.1 at 200 on 50–200 px gains +0.009, but its
  10th-percentile cell loses 0.019.
- **Fit times are the same** for cosine and fixed.

**w pile-up** ([`optimizer/optimizer_pileup.csv`](results/optimizer/optimizer_pileup.csv)): the share of sources at
the sharpest `w` grows with optimisation effort.

| Class | fixed 200 | fixed 500 | cosine 0.05 / 200 |
|---|---|---|---|
| 50–200 | 45 % | 50 % | 25 % |
| 200–1000 | 33 % | 46 % | 22 % |
| 1000–5000 | 29 % | 39 % | 16 % |
| ≥ 5000 | 11 % | 25 % | 7 % |

The settings that fit better have more pile-up. So the pile-up is not a symptom of poor convergence: sources that
are optimised longer move towards sharper edges.

### D22 — iterations on 200–1000 px, revisited

Paired median gain of fixed@500 over fixed@200, measured on the same 10 cells in both runs:

| Class | E3 | E4 | E4 p10 |
|---|---|---|---|
| 50–200 | 0.0034 | 0.0034 | −0.022 |
| 200–1000 | 0.0050 | **0.0068** | +0.0003 |

On 200–1000 px the fixed@200 median itself differs between the two runs: 0.0754 in E3, 0.0706 here. The code and
cells are the same, so this is GPU non-determinism (`index_add`, `grid_sample` backward). On 10 cells this noise
(~0.005) is as large as the decision threshold. Averaged over both runs, the gain is about 0.006.

### Stability — diff7 against the archived arms

[`stability/stability.csv`](results/stability/stability.csv); the gauss and 8-parameter rows are from
[E2b](../E2_variable_sources/02_jetson_baseline/results/stability/stability.csv).

| Class | Median per-pair ratio, gauss / diffusion / diff7 | Same cell closer (diff7) | Position drift d_x, diffusion / diff7 |
|---|---|---|---|
| 50–200 (K 3) | 0.29 / 0.34 / 0.39 | 88 % | 0.32 / 0.31 |
| 200–1000 (K 6) | 0.53 / 0.56 / 0.44 | 100 % | 0.43 / 0.37 |
| 1000–5000 (K 12) | 0.43 / 0.48 / 0.51 | 92 % | 0.32 / 0.37 |
| ≥ 5000 (K 24) | 0.30 / 0.26 / 0.31 | 100 % | 0.30 / 0.30 |

The ratio is cost_same / cost_other; lower is more stable. Drift is in units of the cell's equivalent radius.

- diff7 is about as stable as the 8-parameter source.
- Its error on these pairs is slightly lower: median nrmse at t of 0.238 / 0.215 / 0.213 / 0.269, against
  0.246 / 0.214 / 0.222 / 0.283 for the 8-parameter source.
- Individual sources still move about 0.3 radii in 5 minutes.
- The run summary also prints the ratio of medians, which is the statistic `stability_test` logs. The table
  above uses the median of per-pair ratios, as in the E2b report.

### D21 — noise floor

Per-class medians, 2016 / 2017, from
[`noise_floor/noise_floor_summary.csv`](results/noise_floor/noise_floor_summary.csv):

| Class | Quantization | Interior σ 0.5 px | Interior σ 1 px | Interior 3×3 median | Full-crop σ 1 px (upper bound) |
|---|---|---|---|---|---|
| 50–200 | 0.009 / 0.009 | 0.027 / 0.037 | 0.072 / 0.095 | 0.073 / 0.107 | 0.45 / 0.45 |
| 200–1000 | 0.008 / 0.007 | 0.022 / 0.035 | 0.059 / 0.093 | 0.054 / 0.089 | 0.29 / 0.30 |
| 1000–5000 | 0.008 / 0.006 | 0.018 / 0.032 | 0.049 / 0.086 | 0.041 / 0.082 | 0.21 / 0.25 |
| ≥ 5000 | 0.006 / 0.007 | 0.017 / 0.016 | 0.049 / 0.045 | 0.039 / 0.037 | 0.16 / 0.15 |

The interior columns use 1 erosion step. Their 90th percentiles reach 0.16–0.19 on the 2017 small classes.

- **Quantization is the only irreducible floor,** and it is at most 0.013.
- **The full-crop proxies are not floors.** They mostly measure the blur smearing the sharp 1 mm/h cell edge into
  the zero margin. diff7 already reaches 0.085 on 50–200 px cells in E3, where the full-crop σ 1 value is 0.45.

## Decisions

- [D20](../../README.md#9-decision-log) **Keep the fixed learning rate 0.05.** No cosine setting passes the
  adoption rule. Cosine stays available as an option, but is not the default.
- [D21](../../README.md#9-decision-log) **The adaptive target is nrmse ≤ 0.1 in every class.** That is about
  10 × the quantization floor and above the interior texture floor. The hardest small cells will stop on the cap
  or compression rule.
- [D22](../../README.md#9-decision-log) **Use 500 iterations per wave from 200 px upward and 200 below; this
  revises D19.** The gain on 200–1000 px averages about 0.006 over E3 and E4, and the extra cost is small.

## Caveats

- **10 cells per class.** The run-to-run GPU noise of about 0.005 on medians is as large as the decision
  thresholds. Decisions close to a threshold (D22) should be re-checked on a larger sample when one is fitted
  anyway.
- **The stability arms used different iteration counts.** diff7 ran with the D19 counts (500 on ≥ 1000 px) while
  the archived arms used 200.
- **The cause of the w pile-up is still open.** Sharper edges than the table allows may be wanted. The wider table
  tried in E3 did not lower the error, though.
- **A higher fixed rate (e.g. 0.1) was not tested.**
- **`smoke_*` folders** hold the 1-cell checks of stage 0.
- **Not archived:** the profile tables (`*.pt`, git-ignored).
