
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

records = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        # Read GPS coordinates
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        # Read road nodes
        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        # Use GPS median as the representative trip location
        center_lon = track["longitude"].median()
        center_lat = track["latitude"].median()

        # Calculate UTM zone
        utm_zone = int((center_lon + 180) // 6) + 1

        # Select EPSG according to hemisphere
        if center_lat >= 0:
            epsg = 32600 + utm_zone
        else:
            epsg = 32700 + utm_zone

        # Determine zones represented by GPS and road nodes
        all_lon = pd.concat([
            track["longitude"],
            nodes["longitude"]
        ])

        all_lat = pd.concat([
            track["latitude"],
            nodes["latitude"]
        ])

        zones = sorted(
            set(
                ((all_lon + 180) // 6 + 1)
                .astype(int)
                .tolist()
            )
        )

        records.append({
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "center_longitude": center_lon,
            "center_latitude": center_lat,
            "min_longitude": all_lon.min(),
            "max_longitude": all_lon.max(),
            "min_latitude": all_lat.min(),
            "max_latitude": all_lat.max(),
            "selected_utm_zone": utm_zone,
            "suggested_epsg": epsg,
            "zones_covered": ",".join(map(str, zones)),
            "number_of_zones": len(zones)
        })

# Create report
df = pd.DataFrame(records)

output_dir = Path(
    "member1_preprocessing/outputs/reports"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "per_trip_crs_audit.csv"

df.to_csv(output_file, index=False)

print("\nPER-TRIP CRS AUDIT")
print("==================")

print("Trips inspected:", len(df))

print("\nSuggested CRS distribution:")
print(df["suggested_epsg"].value_counts().to_string())

print("\nTrips spanning multiple UTM zones:")
print(
    df[df["number_of_zones"] > 1][
        ["trip_id", "suggested_epsg",
         "number_of_zones", "zones_covered"]
    ].to_string(index=False)
)

print("\nLargest geographic extents:")
df["longitude_span"] = (
    df["max_longitude"] - df["min_longitude"]
)

print(
    df.nlargest(10, "longitude_span")[
        ["trip_id", "min_longitude",
         "max_longitude", "longitude_span",
         "suggested_epsg"]
    ].to_string(index=False)
)

print("\nReport saved to:", output_file)
