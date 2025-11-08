#!/usr/bin/env python3
"""
MDP Value Iteration for Last-Mile Simulation
"""

import numpy as np
from utility import update_edge, astar_numba
import multiprocessing as mp

def value_iteration(n_nodes, travel_matrix, edge_prob, gamma=0.9, tol=1e-3, max_iter=500):
    V = np.zeros(n_nodes, dtype=np.float32)
    for _ in range(max_iter):
        V_prev = V.copy()
        for s in range(n_nodes):
            vals = [-travel_matrix[s,a]/edge_prob[s,a] + gamma*V_prev[a]
                    for a in range(n_nodes) if a != s]
            if vals:
                V[s] = max(vals)
        if np.max(np.abs(V - V_prev)) < tol:
            break
    return V

def compute_policy(V, travel_matrix, edge_prob):
    n_nodes = len(V)
    policy = np.zeros(n_nodes, dtype=np.int32)
    for s in range(n_nodes):
        best_a = np.argmax([-travel_matrix[s,a]/edge_prob[s,a] + V[a]
                            if a != s else -np.inf for a in range(n_nodes)])
        policy[s] = best_a
    return policy

def simulate_vehicle(vid, cluster_nodes, node2idx, idx2node, depot, travel_matrix, edge_prob,
                     ACCIDENT_PROB, EDGE_ALPHA, START_TIME, END_TIME, SERVICE_MIN):
    cluster_idx = np.array([node2idx[n] for n in cluster_nodes], dtype=np.int32)
    depot_idx = node2idx[depot]
    n_nodes = travel_matrix.shape[0]

    V = value_iteration(n_nodes, travel_matrix, edge_prob)
    policy = compute_policy(V, travel_matrix, edge_prob)

    current_time, current_pos, day, hist = START_TIME, depot_idx, 1, []
    remaining = cluster_idx.copy()

    while remaining.size > 0:
        target = remaining[0]
        path = astar_numba(current_pos, [target], travel_matrix, edge_prob)
        if path[0] != current_pos: path = [current_pos]+path

        for i in range(1, len(path)):
            frm,to = path[i-1], path[i]
            accident = np.random.rand() < ACCIDENT_PROB
            edge_prob[frm,to] = update_edge(edge_prob[frm,to], accident, EDGE_ALPHA)
            current_time += travel_matrix[frm,to]
            hist.append((day, idx2node[to], current_time, accident))
            if accident: current_pos = frm; break
            current_pos = to
            remaining = remaining[remaining != to]

            if current_time > END_TIME:
                current_time, current_pos, day = START_TIME, depot_idx, day+1
                break

    return hist

def simulate(clusters, node2idx, idx2node, depots, travel_matrix, edge_prob,
             ACCIDENT_PROB, EDGE_ALPHA, START_TIME, END_TIME, SERVICE_MIN):
    args = [(vid, clusters[vid], node2idx, idx2node,
             depots[vid % len(depots)], travel_matrix, edge_prob.copy(),
             ACCIDENT_PROB, EDGE_ALPHA, START_TIME, END_TIME, SERVICE_MIN)
            for vid in clusters]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(simulate_vehicle, args)
    return {vid: hist for vid, hist in zip(clusters.keys(), results)}
