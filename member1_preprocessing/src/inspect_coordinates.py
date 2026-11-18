
from pathlib import Path
import json
import pandas as pd

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

all_longitudes = []
all_latitudes = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        # Read GPS coordinates
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["longitude", "latitude"]
        )

        # Read road node coordinates
        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        all_longitudes.extend(track["longitude"].tolist())
        all_latitudes.extend(track["latitude"].tolist())

        all_longitudes.extend(nodes["longitude"].tolist())
        all_latitudes.extend(nodes["latitude"].tolist())

# Calculate geographic extent
min_lon = min(all_longitudes)
max_lon = max(all_longitudes)
min_lat = min(all_latitudes)
max_lat = max(all_latitudes)

center_lon = (min_lon + max_lon) / 2
center_lat = (min_lat + max_lat) / 2

# Calculate UTM zone from the geographic center
utm_zone = int((center_lon + 180) // 6) + 1

hemisphere = "North" if center_lat >= 0 else "South"

epsg = (
    32600 + utm_zone
    if hemisphere == "North"
    else 32700 + utm_zone
)

print("\nGEOGRAPHIC DATASET EXTENT")
print("=========================")

print("Minimum longitude:", min_lon)
print("Maximum longitude:", max_lon)
print("Minimum latitude:", min_lat)
print("Maximum latitude:", max_lat)

print("\nDATASET CENTER")
print("==============")

print("Center longitude:", center_lon)
print("Center latitude:", center_lat)

print("\nSUGGESTED UTM PROJECTION")
print("========================")

print("UTM zone:", utm_zone)
print("Hemisphere:", hemisphere)
print("Suggested EPSG:", epsg)

# Check whether all coordinates fall within the selected zone
zones = {
    int((lon + 180) // 6) + 1
    for lon in all_longitudes
}

print("\nUTM zones represented:", sorted(zones))

if len(zones) == 1:
    print("All coordinates fall within one UTM zone.")
else:
    print("WARNING: Coordinates span multiple UTM zones.")
