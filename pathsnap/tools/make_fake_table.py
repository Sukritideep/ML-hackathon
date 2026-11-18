#Generating a fake candidate table with the contract columns.
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
for t in range(30):
    for obs in range(200):
        true_k = rng.integers(0, 5)
        dists = [abs(rng.normal(8 if k == true_k else 30, 10)) for k in range(5)]
        order = np.argsort(dists)
        best = min(dists)
        for k in range(5):
            rank = int(np.where(order == k)[0][0]) + 1
            rows.append((f"t{t:02d}", obs, t * 1000 + k, dists[k], rank,
                         dists[k] - best, rng.uniform(0, 1),
                         rng.uniform(0, 90), rng.uniform(0, 20), int(k == true_k)))

df = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "f_dist", "f_rank",
                                 "f_dist_gap", "f_proj", "f_bearing_diff",
                                 "f_speed", "label"])
os.makedirs("results", exist_ok=True)
df.to_parquet("results/fake_table.parquet")
print(df.shape, "positive rate:", df["label"].mean())