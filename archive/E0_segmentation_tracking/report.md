# E0 — Segmentation and tracking

**Status:** archived outputs of the tagged code (`v0-jetson-baseline`). Part of the [README](../../README.md) study.

## Question

Which pixels count as rain, so that connected components give usable rain cells (decisions D1–D4)? And can
the cells be linked over time into tracks, which the LSTM and the stability test need?

## Setup

- **Data:** MeteoNet NW radar, 5-min maps from 2016-01-01 12:00. Segmentation on 3 maps (12:00, 12:05, 12:10);
  tracking on 6 maps (12:00 to 12:25). Missing data (`-1`) is never treated as rain.
- **Segmentation (S2):** rainy pixel = rate ≥ 1 mm/h (`R > 8.33`, i.e. `R ≥ 9`), 8-connectivity. Cells under
  50 px are not tracked.
- **Tracker (T1):** constant-velocity prediction; link cost `0.5 × distance / 25 px + 0.5 × (1 − IoU)`;
  Hungarian assignment; links above 0.75 rejected (births and deaths instead). Tracked cells: ≥ 50 px.
- **Hardware:** not recorded in the outputs. The README (§12) gives the development machine: macOS 12, CPU only.

## Reproduce

```bash
git worktree add ../gat-v0 v0-jetson-baseline
cd ../gat-v0
pip install -r requirements.txt
# MeteoNet raw data must be in ../gat-v0/DATA/ (git-ignored; link or copy it from the main checkout)
python src/cell_delimitation.py --start "2016-01-01 12:00" --n-frames 3
python src/track_cells.py --start "2016-01-01 12:00" --n-frames 6
```

Outputs land in `../gat-v0/results/01_segmentation_tracking/` with the same names as the archived files
(`S2_*`, `T1_*`). Compare against `archive/E0_segmentation_tracking/results/`.

## Key results

Segmentation, from [`results/S2_cells.csv`](results/S2_cells.csv), grouped by timestamp:

| Map | Cells | Cells ≥ 50 px | Largest cell (px) | Rainy px | Border-flagged |
|---|---|---|---|---|---|
| 12:00 | 52 | 16 | 10,904 | 21,561 | 4 |
| 12:05 | 48 | 16 | 10,386 | 21,434 | 4 |
| 12:10 | 42 | 19 | 6,364 | 21,076 | 3 |

Tracking, from [`results/T1_detections.csv`](results/T1_detections.csv), [`results/T1_events.csv`](results/T1_events.csv)
and [`results/T1_associations.csv`](results/T1_associations.csv):

| Map | 12:00 | 12:05 | 12:10 | 12:15 | 12:20 | 12:25 | Total |
|---|---|---|---|---|---|---|---|
| Detections (all cells) | 52 | 48 | 42 | 67 | 77 | 70 | 356 |
| Tracked (≥ 50 px) | 16 | 16 | 19 | 19 | 24 | 25 | 119 |

- **47 tracks:** 28 span two or more maps, 19 appear on a single map.
- **Links:** 72 continuations; 31 births and 22 deaths. No merge or split is recorded (not modelled, README §4).
- **Largest cell at 12:00:** 10,904 px, about 51 % of the 21,561 rainy pixels (matches README §3.2).

## Decisions

- [D1](../../README.md#9-decision-log) missing data is never dry (`valid` mask in the segmentation)
- [D2](../../README.md#9-decision-log) rainy = rate ≥ 1 mm/h, 8-connectivity ([§3.2](../../README.md#32-threshold-1-mmh-retained))
- [D3](../../README.md#9-decision-log) large cells are kept whole ([§3.3](../../README.md#33-additional-rules))
- [D4](../../README.md#9-decision-log) cells under 50 px are ignored ([§3.3](../../README.md#33-additional-rules))

Tracking rules are described in [§4](../../README.md#4-tracking); the tracker is not a decision in the log.

## Caveats

- **The R > 0 comparison is not reproducible from this archive.** README §3.1 (166 / 167 / 167 cells, giant object
  of 51,900 px, 4-connectivity check) has no output file here. The tagged code can take a threshold argument, but
  its main only runs the ≥ 1 mm/h variants (`S2_8conn` saved; `S1_4conn` printed, not saved). Its docstring still says "S0".
- **Tracking is validated visually only** (README §4). The counts above describe the output, not its accuracy.
- Border and missing-neighbour flags mark possibly truncated cells; the counts above are informational.
- Only 3 maps (segmentation) and 6 maps (tracking) were processed. This is a small sample, not an evaluation.
