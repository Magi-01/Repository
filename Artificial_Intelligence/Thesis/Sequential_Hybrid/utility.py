#!/usr/bin/env python3
"""
Utility.py
Utility functions for last-mile delivery MDP/Monte Carlo simulations.
Shared by Monte Carlo, MDP Value Iteration, and MDP Policy Iteration.
"""

import numpy as np
import heapq
import math
import json
from numba import njit

# -------------------------
# Vectorized distance computation
# -------------------------
def compute_travel_matrix(coords_arr, vehicle_speed=20.0):
    dx = coords_arr[:,0,None] - coords_arr[None,:,0]
    dy = coords_arr[:,1,None] - coords_arr[None,:,1]
    dist_matrix = np.hypot(dx, dy)
    travel_matrix = ((dist_matrix / 1000) / vehicle_speed) * 60  # in minutes
    np.fill_diagonal(travel_matrix, 0.0)
    return travel_matrix

# -------------------------
# Update edge probability after accident / success
# -------------------------
@njit
def update_edge(prob, accident, alpha=0.2):
    """Update probability of safe traversal after an accident event"""
    if accident:
        prob *= 1 - alpha
    else:
        prob += (1 - prob) * alpha
    if prob < 0.01: prob = 0.01
    elif prob > 1.0: prob = 1.0
    return prob

# -------------------------
# A* search (Numba optimized)
# -------------------------
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

# -------------------------
# Flatten vehicle histories for Gantt
# -------------------------
def flatten_histories(all_histories):
    times = [t for hist in all_histories.values() for day,node,t,acc in hist]
    return max(times) if times else 0

# -----------------------------
# Distance & mapping utilities
# -----------------------------
def build_distance_matrix(coords, nodes, vehicle_speed=20.0):
    """
    Build travel time matrix (minutes) between nodes based on coordinates.
    coords: dict[node] -> (x,y)
    nodes: list of node IDs
    """
    pts = np.array([coords[n] for n in nodes], dtype=float)
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    D = np.sqrt(dx**2 + dy**2)
    travel_minutes_mat = ((D / 1000.0) / vehicle_speed) * 60.0
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    idx2node = {i: n for i, n in enumerate(nodes)}
    return travel_minutes_mat, node2idx, idx2node
