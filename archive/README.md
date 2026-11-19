# Archive

Index of the completed experiments. Each folder holds the archived outputs of one experiment in `results/` and a
`report.md` with the question, set-up, reproduction command, key results, decisions and caveats.

| Experiment | Question | Decisions | Report |
|---|---|---|---|
| E0 — segmentation and tracking | Which pixels form rain cells, and can the cells be tracked over time? | [D1–D4](../README.md#9-decision-log) | [E0 report](E0_segmentation_tracking/report.md) |
| E1 — fixed-length vectors | Which fixed-size encoding reconstructs a cell best at equal size? | [D5–D8, D12](../README.md#9-decision-log) | [E1 report](E1_fixed_vectors/report.md) |
| E2a — variable sources, CPU preliminary | Do per-source diffusion sources beat Gaussians on the two smallest size bins? | [D9, D13](../README.md#9-decision-log) | [E2a report](E2_variable_sources/01_cpu_prelim/report.md) |
| E2b — variable sources, Jetson baseline | Full programme: equal budget, stability, ablation, another season, restarts | [D10, D11, D16](../README.md#9-decision-log) | [E2b report](E2_variable_sources/02_jetson_baseline/report.md) |

Nothing in this folder has been deleted: the outputs were moved here from `results/` with `git mv`, so the rename
history is kept. The code that produced them was not moved. It is pinned by the git tag `v0-jetson-baseline`
(commit `5f76618`); the one later code change, the removal of the collapsed Gaussian arm (commit `b6ff2e5`, D12),
is noted in the E1 report. To re-run an experiment exactly as archived, check out the tag in a separate
worktree (`git worktree add ../gat-v0 v0-jetson-baseline`) and follow the Reproduce section of its report. The
git-ignored inputs (sample pickles, stability pairs, the diffusion profile table, `cells.npz`) are not archived.
They are rebuilt with `scripts/build_samples.py` and stay in [`results/`](../results/README.md).
