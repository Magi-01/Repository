#!/usr/bin/env python3
"""
Linux-safe Last-Mile Simulation
- Numba + multiprocessing
- Fully precomputed Gantt + fading map animation
"""

import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import multiprocessing as mp

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
node2idx = {node:i for i,node in enumerate(nodes)}
idx2node = {i:node for node,i in node2idx.items()}

# -------------------------
# Parameters
# -------------------------
START_TIME = 8*60
END_TIME = 18*60
SERVICE_MIN = 5
VEHICLE_SPEED = 20.0
ACCIDENT_PROB = 0.2
EDGE_ALPHA = 0.2
FADE_STEPS = 3
FPS = 60

# -------------------------
# Vectorized distances
# -------------------------
dx = coords_arr[:,0,None] - coords_arr[None,:,0]
dy = coords_arr[:,1,None] - coords_arr[None,:,1]
dist_matrix = np.hypot(dx, dy)
travel_matrix = ((dist_matrix / 1000) / VEHICLE_SPEED) * 60
np.fill_diagonal(travel_matrix, 0.0)
edge_prob = np.ones((n_nodes,n_nodes), dtype=np.float32)

# -------------------------
# Worker function for multiprocessing
# -------------------------
def simulate_vehicle_wrapper(args):
    vid, cluster_nodes, depot = args
    import numpy as np
    from numba import njit
    import heapq

    cluster_idx = np.array([node2idx[n] for n in cluster_nodes], dtype=np.int32)
    depot_idx = node2idx[depot]
    current_time = START_TIME
    remaining = cluster_idx.copy()
    current_pos = depot_idx
    day = 1
    hist = []

    @njit
    def update_edge(prob, accident, alpha):
        if accident:
            prob *= 1 - alpha
        else:
            prob += (1 - prob) * alpha
        if prob < 0.01: prob = 0.01
        elif prob > 1.0: prob = 1.0
        return prob

    @njit
    def astar_numba(start, goals, travel_mat, prob_mat):
        open_set = [(0.0, start, [start])]
        goal_set = set(goals)
        visited = set()
        while open_set:
            cost, current, path = heapq.heappop(open_set)
            if current in goal_set:
                return path
            if current in visited:
                continue
            visited.add(current)
            for j in range(travel_mat.shape[0]):
                if j == current:
                    continue
                new_cost = cost + travel_mat[current,j]/prob_mat[current,j]
                heapq.heappush(open_set, (new_cost, j, path + [j]))
        return path

    while remaining.size > 0:
        path = astar_numba(current_pos, remaining.tolist(), travel_matrix, edge_prob)
        if path[0] != current_pos:
            path = [current_pos] + path

        for i in range(1,len(path)):
            frm, to = path[i-1], path[i]
            accident = np.random.rand() < ACCIDENT_PROB
            edge_prob[frm,to] = update_edge(edge_prob[frm,to], accident, EDGE_ALPHA)
            travel = travel_matrix[frm,to]
            current_time += travel

            if accident:
                hist.append((day, idx2node[frm], current_time, True))
                current_pos = frm
                break
            else:
                current_pos = to
                hist.append((day, idx2node[to], current_time, False))

            mask = remaining != to
            remaining = remaining[mask]

            if current_time > END_TIME:
                current_time = START_TIME
                current_pos = depot_idx
                day += 1
                break

    # Return to depot
    path_back = astar_numba(current_pos, [depot_idx], travel_matrix, edge_prob)
    for i in range(1,len(path_back)):
        frm,to = path_back[i-1], path_back[i]
        travel = travel_matrix[frm,to]
        current_time += travel
        hist.append((day, idx2node[to], current_time, False))

    return hist

# -------------------------
# Run parallel simulation
# -------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn')
    args_list = [(vid, clusters[vid], depots[vid%len(depots)]) for vid in clusters]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(simulate_vehicle_wrapper, args_list)
    all_histories = {vid: hist for vid,hist in zip(clusters.keys(), results)}

    # -------------------------
    # Animation setup
    # -------------------------
    plt.switch_backend('Qt5Agg')
    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2,2, height_ratios=[2,1])
    ax_map = fig.add_subplot(gs[:,0])
    ax_gantt = fig.add_subplot(gs[0,1])
    ax_gantt.set_title("Gantt Chart")
    ax_map.set_title("Map")

    colors = plt.cm.tab20(np.arange(len(clusters)))
    ax_map.set_xlim(coords_arr[:,0].min()-50, coords_arr[:,0].max()+50)
    ax_map.set_ylim(coords_arr[:,1].min()-50, coords_arr[:,1].max()+50)
    ax_map.grid(True)

    # Depots and deliveries
    ax_map.scatter(coords_arr[[node2idx[d] for d in depots],0],
                   coords_arr[[node2idx[d] for d in depots],1],
                   c='k', s=100, marker='X', zorder=5)
    for vid, cluster_nodes in clusters.items():
        idxs = [node2idx[n] for n in cluster_nodes]
        ax_map.scatter(coords_arr[idxs,0], coords_arr[idxs,1],
                       c=[colors[vid]]*len(idxs), marker='x', s=60, zorder=4)

    vehicle_scatters = [ax_map.scatter([],[],c=[colors[vid]],s=50,zorder=6) for vid in clusters]

    # Routes
    all_routes = []
    for vid in clusters:
        hist = all_histories[vid]
        coords_seq = np.array([coords[node] for day,node,t,acc in hist])
        if len(coords_seq) > 0:
            all_routes.append(coords_seq)

    fade_buffers = [ [] for _ in clusters ]
    fade_lines = [LineCollection([], colors=[colors[vid]], linewidths=2, alpha=0.5) for vid in clusters]
    for ln in fade_lines:
        ax_map.add_collection(ln)

    frame_text = ax_map.text(0.02,0.95,'', transform=ax_map.transAxes, fontsize=10)

    # Precompute Gantt bars
    for vid, hist in all_histories.items():
        for day,node,t,acc in hist:
            if node in clusters[vid]:
                color = 'red' if acc else 'tab:blue'
                ax_gantt.broken_barh([(t,SERVICE_MIN)], (vid*10,8), facecolors=color)
    ax_gantt.set_xlim(START_TIME, max(t for hist in all_histories.values() for day,node,t,acc in hist)+10)
    ax_gantt.set_ylim(0,len(clusters)*10)
    ax_gantt.set_yticks([i*10+4 for i in range(len(clusters))])
    ax_gantt.set_yticklabels([f"Vehicle {i}" for i in range(len(clusters))])

    # Animation update
    def update(frame_idx):
        frame_text.set_text(f"Frame: {frame_idx}")
        for vid, route_coords in enumerate(all_routes):
            if frame_idx < len(route_coords):
                fade_buffers[vid].append(route_coords[frame_idx])
            if len(fade_buffers[vid]) > FADE_STEPS:
                fade_buffers[vid].pop(0)
            if len(fade_buffers[vid]) > 1:
                segments = [fade_buffers[vid][i:i+2] for i in range(len(fade_buffers[vid])-1)]
                fade_lines[vid].set_segments(segments)
                segment_colors = [(*colors[vid][:3], (i+1)/FADE_STEPS) for i in range(len(segments))]
                fade_lines[vid].set_colors(segment_colors)
            if fade_buffers[vid]:
                vehicle_scatters[vid].set_offsets(fade_buffers[vid][-1])

    max_len = max(len(r) for r in all_routes if len(r) > 0)
    ani = FuncAnimation(fig, update, frames=max_len, interval=1000/FPS)
    plt.tight_layout()
    plt.show()