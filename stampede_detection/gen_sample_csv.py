"""
gen_sample_csv.py
-----------------
Generates `sample_data/mobile_pings.csv` — a realistic file the user
can upload into the 📡 Mobile Network module. This ONLY produces the
sample CSV; the module itself always runs REAL DBSCAN on whatever CSV
you upload.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_data", "mobile_pings.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main(n: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Base sparse population on 100x100 grid
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    # Two bottlenecks -- large dense clusters near gates
    n_c1 = 180
    x[:n_c1] = rng.normal(30, 3.0, n_c1)
    y[:n_c1] = rng.normal(65, 3.0, n_c1)
    n_c2 = 140
    x[n_c1:n_c1 + n_c2] = rng.normal(72, 2.5, n_c2)
    y[n_c1:n_c1 + n_c2] = rng.normal(35, 2.5, n_c2)
    x = np.clip(x, 0, 100); y = np.clip(y, 0, 100)

    df = pd.DataFrame({
        "device_id": [f"IMEI{100000 + i}" for i in range(n)],
        "x": np.round(x, 2),
        "y": np.round(y, 2),
        "signal_dbm": np.round(rng.normal(-72, 6, n), 1),
        "carrier": rng.choice(["Airtel", "Jio", "Vi", "BSNL"], size=n),
    })
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
