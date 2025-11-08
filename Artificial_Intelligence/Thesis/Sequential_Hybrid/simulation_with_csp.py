#!/usr/bin/env python3
"""
Linux-safe Last-Mile Simulation
- Numba + multiprocessing
- Dynamic method selection: montecarlo / mdp_value_iteration / mdp_policy_iteration
- Live vehicle map + dynamic Gantt chart handled in animation.py
"""

import sys, json, numpy as np
from utility import build_distance_matrix, compute_travel_matrix
import multiprocessing as mp
import animation as anim

# -------------------------
# Load cluster data
# -------------------------
with open("cluster_data.json","r") as f:
    data = json.load(f)

coords = {int(k): tuple(v) for k,v in data["coords"].items()}
clusters = {int(k): v for k,v in data["clusters"].items()}
depots = [int(d) for d in data["depots"]]

nodes = np.array(list(coords.keys()))
coords_arr = np.array([coords[n] for n in nodes], dtype=np.float32)
n_nodes = len(nodes)

# -------------------------
# Parameters
# -------------------------
START_TIME = 8*60
END_TIME = 18*60
SERVICE_MIN = 5
VEHICLE_SPEED = 20.0
ACCIDENT_PROB = 0.2
EDGE_ALPHA = 0.2
FPS = 60

# -------------------------
# Travel matrix
# -------------------------
travel_matrix = compute_travel_matrix(coords_arr, VEHICLE_SPEED)
edge_prob = np.ones((n_nodes, n_nodes), dtype=np.float32)

# -------------------------
# Main simulation
# -------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn')

    # Determine method
    method_name = sys.argv[1] if len(sys.argv) > 1 else "montecarlo"
    print(f"Running simulation with method: {method_name}")

    # Dynamically import the method
    if method_name == "montecarlo":
        import montecarlo as method
    elif method_name == "mdp_value_iteration":
        import mdp_value_iteration as method
    elif method_name == "mdp_policy_iteration":
        import mdp_policy_iteration as method
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Build distance matrix
    travel_minutes_mat, node2idx, idx2node = build_distance_matrix(coords, nodes)

    # Run simulation
    all_histories = method.simulate(
        clusters, node2idx, idx2node, depots, travel_matrix, edge_prob,
        ACCIDENT_PROB, EDGE_ALPHA, START_TIME, END_TIME, SERVICE_MIN
    )

    # Animate
    ani = anim.animate_simulation(
        coords_arr, clusters, node2idx, idx2node, depots, all_histories,
        VEHICLE_SPEED=VEHICLE_SPEED, ACCIDENT_PROB=ACCIDENT_PROB
    )
