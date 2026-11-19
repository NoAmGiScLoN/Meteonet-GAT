from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"

# Example file: first chunk of January 2016
path = DATA_DIR / "NW_rainfall_2016" / "rainfall-NW-2016-01" / "rainfall_NW_2016_01.1.npz"

# allow_pickle=True is needed because 'dates' is stored as an array of datetime objects
with np.load(path, allow_pickle=True) as npz:
    print("keys:", npz.files)
    data = npz["data"]              
    dates = npz["dates"]            
    miss_dates = npz["miss_dates"]  

print("data:", data.shape, data.dtype)
print("dates:", dates[0], "->", dates[-1], f"({len(dates)} steps)")
print("missing dates:", len(miss_dates))
print("value range:", data.min(), "to", data.max())
