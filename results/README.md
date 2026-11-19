# results/

This folder holds only:

- **Rebuildable inputs**, git-ignored and written by `scripts/build_samples.py`: `02_fixed_vector_benchmark/cells.npz`,
  `03_variable_sources/` (sample pickles, stability pairs, diffusion profile table) and the copies of the profile
  table in `04_jetson_run/`.
- **New runs**, written by the scripts in `src/` and `scripts/`.

The outputs of the completed experiments are in [`../archive/`](../archive/README.md), not here. Nothing in
`results/` is versioned by git.
