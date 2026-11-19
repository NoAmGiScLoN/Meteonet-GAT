from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
OUT_DIR = Path(__file__).resolve().parent / "samples"

target = datetime(2016, 1, 1, 12, 0)

path = DATA_DIR / "NW_rainfall_2016" / "rainfall-NW-2016-01" / "rainfall_NW_2016_01.1.npz"

with np.load(path, allow_pickle=True) as npz:
    dates = npz["dates"]
    idx = np.where(dates == target)[0]
    if len(idx) == 0:
        raise ValueError(f"{target} absent de ce fichier ({dates[0]} -> {dates[-1]})")
    grid = npz["data"][idx[0]]

df = pd.DataFrame(grid)
out = OUT_DIR / f"rainfall_{target:%Y-%m-%d_%H%M}.xlsx"
df.to_excel(out, sheet_name=f"{target:%Y-%m-%d %H%M}")
print(f"Écrit : {out}  ({grid.shape[0]} lignes x {grid.shape[1]} colonnes)")
