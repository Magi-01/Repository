#!/usr/bin/env python3
"""
Animation utilities for Last-Mile Simulation
- Separate vehicle map animation and Gantt chart animation
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

FADE_STEPS = 3
FPS = 60

# -----------------------------
# Vehicle Map Animation
# -----------------------------
def update_map(frame_idx, all_routes, vehicle_scatters, fade_buffers, fade_lines, colors, fade_steps=3):
    """
    Update vehicle positions and fade trails on the map.
    all_routes: list of np.arrays of shape (T,2) per vehicle
    vehicle_scatters: scatter objects for vehicles
    fade_buffers: list of lists storing past positions for fade effect
    fade_lines: LineCollection objects for vehicle trails
    fade_steps: number of previous positions to show in trail
    """
    for vid, route_coords in enumerate(all_routes):
        # Only update if we have a valid frame
        if frame_idx < len(route_coords):
            pos = route_coords[frame_idx]
            fade_buffers[vid].append(pos)
        
        # Keep only the last fade_steps positions
        if len(fade_buffers[vid]) > fade_steps:
            fade_buffers[vid] = fade_buffers[vid][-fade_steps:]
        
        # Update line segments for fade trail
        if len(fade_buffers[vid]) > 1:
            segments = [fade_buffers[vid][i:i+2] for i in range(len(fade_buffers[vid])-1)]
            fade_lines[vid].set_segments(segments)
            segment_colors = [(*colors[vid][:3], (i+1)/len(segments)) for i in range(len(segments))]
            fade_lines[vid].set_colors(segment_colors)
        else:
            fade_lines[vid].set_segments([])
        
        # Update vehicle marker
        if fade_buffers[vid]:
            vehicle_scatters[vid].set_offsets(fade_buffers[vid][-1])

# -----------------------------
# Gantt Bar Animation
# -----------------------------
def update_gantt(frame_idx, all_histories, gantt_bars, speed_texts, vehicle_speeds, start_times, VEHICLE_SPEED):
    for vid, hist in enumerate(all_histories.values()):
        if not hist: 
            continue
        first_time = start_times[vid]
        elapsed = [max(0, t - first_time) for day,node,t,acc in hist]
        if frame_idx < len(elapsed):
            gantt_bars[vid].set_width(elapsed[frame_idx])
            speed_texts[vid].set_x(elapsed[frame_idx])
            last_idx = min(frame_idx, len(vehicle_speeds[vid])-1)
            speed_texts[vid].set_text(f"{vehicle_speeds[vid][last_idx]:.1f}")

# -----------------------------
# Animate Simulation
# -----------------------------

def animate_simulation(coords_arr, clusters, node2idx, idx2node, depots, all_histories, VEHICLE_SPEED=20.0, ACCIDENT_PROB=0.2, FPS=60):
    n_clusters = len(clusters)
    colors = plt.cm.tab20(np.arange(n_clusters))

    # Prepare routes
    all_routes = []
    for vid in clusters:
        hist = all_histories[vid]
        coords_seq = np.array([coords_arr[node2idx[node]] for day,node,t,acc in hist])
        all_routes.append(coords_seq)

    # Figure + axes
    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2,2, height_ratios=[2,1])
    ax_map = fig.add_subplot(gs[:,0])
    ax_gantt = fig.add_subplot(gs[0,1])
    ax_gantt.set_title("Elapsed Time Gantt")
    ax_map.set_title("Map")

    ax_map.set_xlim(coords_arr[:,0].min()-50, coords_arr[:,0].max()+50)
    ax_map.set_ylim(coords_arr[:,1].min()-50, coords_arr[:,1].max()+50)
    ax_map.grid(True)

    # Depots
    ax_map.scatter(coords_arr[[d for d in depots],0],
                   coords_arr[[d for d in depots],1],
                   c='k', s=100, marker='X', zorder=5)

    # Delivery nodes
    for vid, cluster_nodes in clusters.items():
        idxs = [node2idx[n] for n in cluster_nodes]
        ax_map.scatter(coords_arr[idxs,0], coords_arr[idxs,1],
               c=[colors[vid]]*len(idxs), marker='x', s=60, zorder=4)

    # Vehicle markers
    vehicle_scatters = [ax_map.scatter([],[],c=[colors[vid]],s=50,zorder=6) for vid in clusters]
    fade_buffers = [[] for _ in clusters]
    fade_lines = [LineCollection([], colors=[colors[vid]], linewidths=2, alpha=0.5) for vid in clusters]
    for ln in fade_lines:
        ax_map.add_collection(ln)
    frame_text = ax_map.text(0.02,0.95,'', transform=ax_map.transAxes, fontsize=10)

    # Gantt bars and speed texts
    gantt_bars, speed_texts, vehicle_speeds, start_times = [], [], [], []
    max_time = 0
    for vid, hist in all_histories.items():
        if not hist:
            continue
        first_time = hist[0][2]
        start_times.append(first_time)
        speeds = []
        for day,node,t,acc in hist:
            if node in clusters[vid]:
                speeds.append(VEHICLE_SPEED if not acc else VEHICLE_SPEED*0.3)
        vehicle_speeds.append(np.array(speeds))
        bar = ax_gantt.barh(vid, 0, color='red', height=0.6, align='center')
        gantt_bars.append(bar[0])
        txt = ax_gantt.text(0, vid, f"{vehicle_speeds[-1][0]:.1f}", va='center', ha='left', color='black', fontsize=9)
        speed_texts.append(txt)
        max_time = max(max_time, max(t - first_time for day,node,t,acc in hist))

    ax_gantt.set_xlim(0, max_time + 10)
    ax_gantt.set_ylim(-0.5, n_clusters - 0.5)
    ax_gantt.set_xlabel("Elapsed Time (minutes)")
    ax_gantt.set_yticks(range(n_clusters))
    ax_gantt.set_yticklabels([f"Vehicle {i}" for i in range(n_clusters)])

    # Animation update function
    def update(frame_idx):
        frame_text.set_text(f"Frame: {frame_idx}")
        update_map(frame_idx, all_routes, vehicle_scatters, fade_buffers, fade_lines, colors)
        update_gantt(frame_idx, all_histories, gantt_bars, speed_texts, vehicle_speeds, start_times, VEHICLE_SPEED)

    max_len = max(len(r) for r in all_routes if len(r) > 0)
    ani = FuncAnimation(fig, update, frames=max_len, interval=1000/FPS)
    plt.tight_layout()
    plt.show()
    return ani