# E6 — Tracker v2: lineage, hysteresis and gap handling over 2016–2018

**Status:** archived outputs of the code at commit `1bbcd15` (branch `feat/tracker-v2`). Part of the
[README](../../README.md) study. This run supports decisions D26–D29.

## Question

The E0 tracker (README §4) linked cells by a constant-velocity cost and a one-to-one assignment. It had no merges
or splits, did not separate edge exits from decay, gave a new id after any missed detection, and was checked only
visually on 6 maps. Each LSTM sequence of the forecasting model is a track, so:

1. Can a tracker with an overlap-based lineage (merges, splits, parent table), size hysteresis and a one-frame gap
   tolerance run over all 315,597 maps of 2016–2018?
2. What are its track lengths, event rates and identity-switch proxies, by size class?
3. Do sampled events look right?

## Setup

- **Code.** `src/tracker_v2.py` (one step: segmentation, motion, association, lineage, events),
  `src/tracker_state.py` (frozen state and `TrackerConfig`), `src/motion.py`, `src/lineage.py`,
  `src/frame_stream.py` (maps across files with time-gap flags), driver `src/track_run.py`, metrics
  `src/track_metrics.py` (+ `track_flicker.py`, `track_review.py`, `track_tables.py`).
- **Rules** (`results/run_config.json`):
  - **Segmentation** unchanged (D1, D2): rate ≥ 1 mm/h, 8-connectivity, missing data never dry.
  - **Hysteresis:** a track starts at ≥ 50 px (D4) and continues down to 35 px.
  - **Motion:** per-track velocity smoothed by an exponential average (α = 0.5). Tracks younger than 2 maps use the
    global motion of the map from phase correlation (4× coarsened, accepted when the peak ratio ≥ 2), else zero.
  - **Association:** overlap of the advected track with the cells, gate 25 px, cost as E0 (weight 0.5 on position,
    links above 0.75 rejected).
  - **Lineage:** a link needs ≥ 10 % overlap of both objects. In a merge the parent with the largest rain mass keeps
    the id (the others end with `merge_end`); in a split the child with the largest mass keeps it (the others start
    with `split_start`). Every link is written to `lineage.parquet`, many-to-many cases included.
  - **Gaps:** a missed track coasts 1 map on its prediction, then ends. A time gap > 15 min ends all tracks
    (`gap_break`).
  - **Truncated cells** (touching the grid edge or missing data) are tracked and flagged; entries and exits through
    the edge or missing data are separate events.
- **Outputs** per month: `detections.parquet` (every connected component, with `track_id` 0 when untracked),
  `lineage.parquet`, `events.parquet`, `state_end.pkl`, `month_summary.json`, `DONE`. Track ids are global; the
  first map of a month is linked to the last map of the previous one, so the run equals one continuous pass.
- **Checks.** January 2016 first (`results/jan2016/`), then the full run. Metrics and a 40-event review sheet
  stratified by event type (`track_metrics.py`).
- **Hardware:** Mac (CPU), 2026-11-24 00:58–09:39; 7.47 h of processing, 593–886 s per month, ~10 maps/s.

## Reproduce

```bash
git checkout 1bbcd15
PYTHONPATH=src python src/track_run.py --start 2016-01 --end 2018-12 --out results/09_tracking_v2
PYTHONPATH=src python src/track_metrics.py results/09_tracking_v2            # writes results/09_tracking_v2/_report/
python archive/E6_tracker_v2/end_pattern.py results/09_tracking_v2 archive/E6_tracker_v2/results/end_pattern.csv
```

`--resume` skips months with a `DONE` file. Raw data in `DATA/` (see README §2).

## Key results

### Volume (`results/months.csv`)

315,597 maps, 20.7 M connected components, **4.53 M tracked cells** in **614,595 tracks** (59 censored by the end
of the run).

### Track length (`results/metrics_by_class.csv`, uncensored tracks, by the track's largest size)

| Class | Tracks | Median length (maps) | 90th percentile | ≤ 2 maps | ≥ 12 maps (1 h) |
|---|---|---|---|---|---|
| 50–200 | 451,964 | 3 | 11 | 48 % | 9 % |
| 200–1000 | 134,644 | 8 | 30 | 18 % | 39 % |
| 1000–5000 | 23,969 | 15 | 55 | 16 % | 57 % |
| ≥ 5000 | 3,959 | 31 | 112 | 16 % | 67 % |

Tracks longer than 3 maps carry **97.8 %** of the rain mass of cells ≥ 50 px. Short tracks are mostly small,
short-lived cells, as expected from D4.

### Events (per 1,000 tracks)

| birth | death | merge_end | split_start | entry_edge | exit_edge | entry_missing | exit_missing | coast | gap_break |
|---|---|---|---|---|---|---|---|---|---|
| 703 | 669 | 250 | 216 | 73 | 75 | 8 | 6 | 867 | 0.1 |

### Identity-switch proxies and gaps (`results/metrics.json`)

- **Area jumps:** 1.1 % of the 3.85 M successive detections of a track change area by more than ×3.
- **Velocity resets:** 0.7 %; median velocity change 0.81 px per map.
- **Coasts:** 12 % of coasting tracks are found again on the next map.
- **Flicker:** 8.8 % of the tracks that die after their coast would have been found with a 2-map coast
  (9.2 % / 7.5 % / 6.3 % / 5.1 % by class). One map was kept (D28).
- **January 2016 alone** (`results/jan2016/`): switch, coast and flicker figures within 0.6 points of the full run;
  winter tracks of the large classes are shorter (median 13 / 22 maps for 1000–5000 / ≥ 5000 px, against 15 / 31).

### Review sheet (`results/review/`, `results/review_events.csv`)

40 events, 4 for each of the 10 event types, drawn over the three years. **35 correct, 5 doubtful;** track identities
are consistent in all 40. The doubtful ones are labels or missing links, not wrong identities:

- **#13, #16** (exit_edge): a track ends next to a large field after coasting, with no merge link — probably
  absorbed.
- **#2** (birth): a one-map track born against a large cell's edge with no split link — probably a fragment.
- **#20** (entry_missing): probably the end of a neighbouring track breaking off next to missing data; the later
  merge is recorded.
- **#14** (exit_edge): a truncated cell dissipating at the edge; exit and death cannot be told apart.

Two events that looked wrong on the sheet were checked in the tables and are correct: #33 coasts through a fully
missing map and is then recorded as merging into its neighbour; #39's red outline at t is another track drawn in a
similar red.

### Probably missed merges (`results/end_pattern.csv`)

Measured after the review, over 461,103 track ends (100 ends whose last detection is in the previous month, 0.02 %,
are skipped). With a 1-map coast every ending track coasts first, and the merge test is not applied to the coasted
map. Share of ends whose last centroid (10 min before the end event) lies inside the equivalent disk of a cell at
least 5× larger on the map of the end event:

| End | Tracks | Inside a 5× larger cell | Within contact distance (loose) | Last detection truncated |
|---|---|---|---|---|
| death | 411,128 | 10.2 % | 26.1 % | 0 % |
| exit_edge | 46,207 | 3.0 % | 14.7 % | 94 % |
| exit_missing | 3,768 | 4.1 % | 15.8 % | 100 % |

This is an **upper bound** on missed merges: up to about 42,000 deaths, next to 153,356 recorded merges. No chance
baseline was computed (a small cell dying near a large rain area meets the criterion by coincidence too), and the
centroid is 10 min older than the large cell it is tested against. Edge exits are mostly labelled right.

## Decisions

- [D26](../../README.md#9-decision-log) **Tracker v2 replaces the E0 tracker.** Overlap-based association with a
  lineage table; in merges and splits the largest rain mass keeps the id.
- [D27](../../README.md#9-decision-log) **Size hysteresis:** tracks start at 50 px and continue down to 35 px.
- [D28](../../README.md#9-decision-log) **Gap handling:** coast 1 map; a time gap > 15 min ends every track.
  A 2-map coast would recover 8.8 % of the tracks that die after coasting.
- [D29](../../README.md#9-decision-log) **Truncated cells are tracked and flagged;** entries and exits through the
  grid edge or missing data are separate events.

## Caveats

- **Missed merges.** Up to ~42,000 deaths (10 % of deaths, an upper bound) may be absorptions by a larger neighbour
  that the lineage does not record. Identities are not affected; the GAT would see these cells die next to a large one. A
  merge test against the larger neighbour when a coasting track ends would fix it, in the tracker (rerun 7.5 h) or
  as an extra edge type at GAT load time. Not done (kept as a caveat).
- **The switch proxies are proxies.** Area jumps and velocity resets flag possible switches; there is no ground
  truth beyond the 40 reviewed events.
- **The review is a first pass by the assistant,** checked in the tables for the doubtful cases and accepted by the
  author.
- **Small cells dominate the counts.** 74 % of tracks are 50–200 px and half of them last ≤ 2 maps.
- **Not archived:** the per-month parquet outputs (862 MB, `results/09_tracking_v2/`) and `flicker.csv` (19 MB,
  rebuilt by `track_metrics.py`).
