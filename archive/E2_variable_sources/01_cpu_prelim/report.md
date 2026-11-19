# E2a — Variable-size sources, CPU preliminary run

**Status:** archived outputs of the tagged code (`v0-jetson-baseline`). Part of the [README](../../../README.md) study (§7.4).

## Question

Before the full programme: on the two smallest size bins, do per-source diffusion sources beat Gaussian sources at
equal parameter budget? The answer motivated the move to the Jetson (D13) and supports the node design (D9).

## Setup

- **Data:** 40 cells per bin for 50–200 px and 200–1,000 px, from January and July 2016. Input sample
  `results/03_variable_sources/sample_40.pkl` (not archived); its first rows match the Jetson main run exactly.
- **Arms:** `gauss` (6 parameters per source) and `diffusion` (8 per source: τ and w). K schedule as in the code,
  capped at 8 (50–200 px) and 24 (200–1,000 px). 200 iterations (tagged default; not logged).
- **Hardware:** CPU only (macOS 12 development machine, README §12). No GPU run.

## Reproduce

```bash
git worktree add ../gat-v0 v0-jetson-baseline
cd ../gat-v0
pip install -r requirements.txt
python scripts/build_samples.py   # writes sample_40.pkl, stability pairs and diffusion_profiles.pt to results/03_variable_sources/
python src/variable_source_fit.py --per-bin 40 --bins 50-200 200-1000 --arms gauss diffusion --iters 200 --device cpu
```

The invocation was not recorded. This is the tagged argparse with the two bins named in the README. Outputs land in
`../gat-v0/results/03_variable_sources/`.

## Key results

Median nrmse on √rate, 40 cells per bin, recomputed from [`results/fits.csv`](results/fits.csv). The Jetson main run
(same cells) is shown for comparison:

| Bin | Params | Gauss (CPU) | Diffusion (CPU) | Diffusion better (CPU) | Gauss (Jetson) | Diffusion (Jetson) |
|---|---|---|---|---|---|---|
| 50–200 px | 24 | 0.296 | 0.222 | 0.950 | 0.296 | 0.218 |
| 50–200 px | 48 | 0.237 | 0.145 | 1.000 | 0.239 | 0.137 |
| 200–1,000 px | 24 | 0.316 | 0.291 | 0.700 | 0.316 | 0.292 |
| 200–1,000 px | 48 | 0.255 | 0.213 | 0.925 | 0.255 | 0.213 |
| 200–1,000 px | 96 | 0.204 | 0.143 | 0.975 | 0.204 | 0.146 |

- The ranking matches the full Jetson run. The medians agree within 0.01 at every shared budget.
- Timings ([`results/full_run.log`](results/full_run.log)): 50–200 px gauss 10 s, diffusion 16 s; 200–1,000 px gauss 44 s,
  diffusion 94 s; 1,000–5,000 px gauss 220 s.
- The `summary_*.csv` files in `results/` come from the 16-cell test run ([`results/test_run.log`](results/test_run.log):
  4 cells per bin, all four bins). They are not results of this run. For example, at 24 parameters on 50–200 px
  the test run gives diffusion 0.276 against 0.222 on the 40 cells.

## Decisions

- [D9](../../../README.md#9-decision-log) node = variable-size source set at native resolution ([§7.1](../../../README.md#71-representation))
- [D13](../../../README.md#9-decision-log) heavy experiments on the Jetson Orin Nano ([§7.4](../../../README.md#74-experimental-programme))

## Caveats

- **The run is incomplete.** Only 50–200 px and 200–1,000 px are complete (40 cells each). The log ends after the
  1,000–5,000 px Gaussian arm, with no completion line and no recorded reason. The README (§7.4) presents it as a two-bin run.
- **The summary tables in this folder are from the 16-cell test run** (see Key results). Use `results/fits.csv` for the
  40-cell numbers.
- **Diffusion here has τ and w (8 parameters).** D11 (drop τ) was decided later, from the Jetson ablation (E2b).
- **One run per cell,** no restarts. The restart test is in E2b.
