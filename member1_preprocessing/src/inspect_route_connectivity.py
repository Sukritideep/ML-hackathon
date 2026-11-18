
from pathlib import Path
from collections import deque
import pandas as pd
from pyproj import Geod

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)

# Preserve trip IDs, including leading zeros
disconnected = pd.read_csv(
    REPORT_DIR / "disconnected_route_pairs.csv",
    dtype={"trip_id": str}
)

geod = Geod(ellps="WGS84")


def shortest_path_hops(adjacency, start, target, max_hops=5):
    """Find a directed path up to max_hops."""

    if start == target:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, depth = queue.popleft()

        if depth >= max_hops:
            continue

        for next_node in adjacency.get(node, []):

            if next_node == target:
                return depth + 1

            if next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, depth + 1))

    return None


results = []

# Cache each trip's network so repeated pairs are efficient
trip_cache = {}

for _, row in disconnected.iterrows():

    trip_id = row["trip_id"]
    partition = row["partition"]

    if trip_id not in trip_cache:

        trip_dir = DATA_ROOT / partition / trip_id

        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None
        ).to_numpy(dtype=float)

        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None
        ).to_numpy(dtype=int)

        adjacency = {}
        reverse_adjacency = {}

        for start, end in arcs:
            adjacency.setdefault(start, []).append(end)
            reverse_adjacency.setdefault(end, []).append(start)

        trip_cache[trip_id] = (
            nodes, adjacency, reverse_adjacency
        )

    nodes, adjacency, reverse_adjacency = trip_cache[trip_id]

    previous_end = int(row["previous_end_node"])
    next_start = int(row["next_start_node"])

    # Geographic distance between the two disconnected nodes
    lon1, lat1 = nodes[previous_end]
    lon2, lat2 = nodes[next_start]

    _, _, distance = geod.inv(
        lon1, lat1, lon2, lat2
    )

    # Check direct reverse arc
    reverse_exists = (
        previous_end in reverse_adjacency.get(next_start, [])
    )

    # Check directed paths through intermediate nodes
    forward_hops = shortest_path_hops(
        adjacency, previous_end, next_start, max_hops=5
    )

    backward_hops = shortest_path_hops(
        adjacency, next_start, previous_end, max_hops=5
    )

    results.append({
        "trip_id": trip_id,
        "previous_arc_id": row["previous_arc_id"],
        "next_arc_id": row["next_arc_id"],
        "previous_end_node": previous_end,
        "next_start_node": next_start,
        "endpoint_distance_m": distance,
        "direct_reverse_arc_exists": reverse_exists,
        "forward_path_hops_max5": forward_hops,
        "backward_path_hops_max5": backward_hops
    })


result_df = pd.DataFrame(results)

output_file = REPORT_DIR / "route_connectivity_analysis.csv"
result_df.to_csv(output_file, index=False)

print("\nROUTE CONNECTIVITY ANALYSIS")
print("===========================")

print(
    result_df.to_string(index=False)
)

print("\nSUMMARY")
print("=======")

print(
    "Pairs with direct reverse arc:",
    result_df["direct_reverse_arc_exists"].sum()
)

print(
    "Pairs with forward path within 5 hops:",
    result_df["forward_path_hops_max5"].notna().sum()
)

print(
    "Pairs with backward path within 5 hops:",
    result_df["backward_path_hops_max5"].notna().sum()
)

print("\nSaved:", output_file)
