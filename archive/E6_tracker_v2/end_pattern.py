"""E6 : fins de piste situées dans une cellule bien plus grande (fusions probablement manquées, estimation).

Pour chaque fin (death, exit_edge, exit_missing), on regarde à l'instant de la fin si le dernier centroïde de la
piste tombe dans une cellule >= RATIO fois plus grande : critère strict = dans le disque équivalent de la grande
cellule ; critère large = à distance de contact (somme des rayons équivalents + GAP_PX).
  python archive/E6_tracker_v2/end_pattern.py results/09_tracking_v2 archive/E6_tracker_v2/results/end_pattern.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENDS = {"death", "exit_edge", "exit_missing"}
RATIO = 5.0
GAP_PX = 10.0


def month_rows(m: Path) -> list[tuple]:
    d = pd.read_parquet(m / "detections.parquet", columns=["timestamp", "track_id", "area_px", "cx", "cy", "truncated"])
    e = pd.read_parquet(m / "events.parquet").sort_values("t")
    last = d[d.track_id > 0].sort_values("timestamp").groupby("track_id").tail(1).set_index("track_id")
    by_t = {t: g for t, g in d[d.area_px >= 50].groupby("timestamp")}
    rows = []
    for r in e[e.event.isin(ENDS)].itertuples():
        if r.track_id not in last.index:
            continue
        cell = last.loc[r.track_id]
        g = by_t.get(r.t)
        strict = loose = False
        if g is not None:
            big = g[g.area_px >= RATIO * cell.area_px]
            if len(big):
                dist = np.hypot(big.cx - cell.cx, big.cy - cell.cy)
                r_big = np.sqrt(big.area_px / np.pi)
                strict = bool((dist <= r_big).any())
                loose = bool((dist <= r_big + np.sqrt(cell.area_px / np.pi) + GAP_PX).any())
        rows.append((m.name, r.event, bool(cell.truncated), strict, loose))
    return rows


def main(root: str, out: str) -> None:
    months = sorted(p for p in Path(root).iterdir() if (p / "DONE").exists())
    df = pd.DataFrame([row for m in months for row in month_rows(m)],
                      columns=["month", "event", "last_truncated", "inside_big_strict", "near_big_loose"])
    table = df.groupby("event").agg(n=("event", "size"), inside_big_strict=("inside_big_strict", "mean"),
                                    near_big_loose=("near_big_loose", "mean"),
                                    last_truncated=("last_truncated", "mean")).reset_index()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage : python end_pattern.py <dossier_tracker> <sortie.csv>")
    main(sys.argv[1], sys.argv[2])
