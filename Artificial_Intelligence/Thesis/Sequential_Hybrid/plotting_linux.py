#!/usr/bin/env python3
"""
animate_clusters_routes_and_gantt.py
------------------------------------
Animate cluster routes and visualize Gantt chart from JSON output.
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- Load JSON ---
def load_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# --- Animate cluster routes ---
def animate_routes(data, interval=50, fade_steps=10, points_per_step=5):
    """
    Smooth cluster route animation with fading trails.
    - interval: milliseconds between frames
    - fade_steps: number of previous segments to keep for fading
    - points_per_step: interpolation steps between consecutive points
    """
    coords_dict = data["coords"]
    clusters_dict = data["clusters"]
    cluster_routes = data.get("cluster_routes", {})
    depots_list = data["depots"]

    fig, ax = plt.subplots(figsize=(6, 6))
    n_colors = max(len(clusters_dict), 10)
    colors = plt.cm.tab20(np.arange(n_colors))

    cluster_info = []
    for i, (cluster_id, cluster_indices) in enumerate(clusters_dict.items()):
        # Pick depot in cluster or first point
        depot_in_cluster = next((d for d in depots_list if d in cluster_indices), cluster_indices[0])
        start_idx = depot_in_cluster
        route_idx = cluster_routes.get(cluster_id, cluster_indices)
        route_idx_ordered = [start_idx] + [idx for idx in route_idx if idx != start_idx]
        route_coords = np.array([coords_dict[str(idx)] for idx in route_idx_ordered])
        cluster_info.append(route_coords)

        # Draw depot
        ax.scatter(*coords_dict[str(start_idx)], c=colors[i], s=120, marker='o', edgecolor='k', zorder=6)
        # Draw deliveries
        deliveries_coords = np.array([coords_dict[str(idx)] for idx in cluster_indices])
        ax.scatter(deliveries_coords[:, 0], deliveries_coords[:, 1], c=[colors[i]], marker='x', label=f"Cluster {cluster_id}")

    ax.legend()
    ax.set_title("Smooth Cluster Route Animation")
    all_coords = np.array(list(coords_dict.values()))
    ax.set_xlim(all_coords[:, 0].min() - 10, all_coords[:, 0].max() + 10)
    ax.set_ylim(all_coords[:, 1].min() - 10, all_coords[:, 1].max() + 10)
    ax.grid(True)

    # Prepare lines storage for fading
    cluster_lines = [[] for _ in cluster_info]

    # Interpolate points for smooth animation
    interp_routes = []
    for route in cluster_info:
        smooth = []
        for i in range(len(route) - 1):
            segment = np.linspace(route[i], route[i+1], points_per_step, endpoint=False)
            smooth.extend(segment)
        smooth.append(route[-1])
        interp_routes.append(np.array(smooth))

    max_len = max(len(r) for r in interp_routes)

    def init():
        return []

    def update(frame):
        artists = []
        for i, route_coords in enumerate(interp_routes):
            if frame > 0 and frame < len(route_coords):
                line, = ax.plot(route_coords[frame-1:frame+1, 0],
                                route_coords[frame-1:frame+1, 1],
                                lw=2, c=colors[i], alpha=1.0)
                cluster_lines[i].append(line)

            # Fade previous segments
            for age, line in enumerate(reversed(cluster_lines[i])):
                alpha = max(0, (fade_steps - age) / fade_steps)
                line.set_alpha(alpha)

            # Remove old lines beyond fade_steps
            while len(cluster_lines[i]) > fade_steps:
                old_line = cluster_lines[i].pop(0)
                old_line.remove()

            artists.extend(cluster_lines[i])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_len, init_func=init,
                                  blit=True, repeat=False, interval=interval)
    plt.show()


# --- Gantt chart ---
def plot_gantt(data):
    clusters_dict = data["clusters"]
    cluster_schedules = data.get("cluster_schedules", {})  # optional

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.arange(len(clusters_dict)))

    for i, cluster_id in enumerate(clusters_dict):
        schedule = cluster_schedules.get(cluster_id, [])
        for j, (start, end) in enumerate(schedule):
            ax.barh(i, end - start, left=start, color=colors[i])
            ax.text(start + (end - start) / 2, i, f"T{j}", color='white',
                    ha='center', va='center', fontsize=8)

    ax.set_yticks(np.arange(len(clusters_dict)))
    ax.set_yticklabels([f"Cluster {cid}" for cid in clusters_dict])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Schedule by Cluster")
    plt.tight_layout()
    plt.show()

# --- Entry point ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python animate_clusters_routes_and_gantt.py results.json")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_results(json_path)

    print(f"Loaded {len(data['clusters'])} clusters | Runtime: {data.get('runtime', 0):.2f}s")
    animate_routes(data)
    plot_gantt(data)

if __name__ == "__main__":
    main()
