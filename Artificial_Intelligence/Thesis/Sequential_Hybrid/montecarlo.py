#!/usr/bin/env python3
"""
Montecarlo.py
Monte Carlo Simulation for Last-Mile Delivery
"""

import numpy as np
from utility import update_edge, astar_numba
import multiprocessing as mp

def simulate_vehicle(vid, cluster_nodes, node2idx, idx2node, depot, travel_matrix, edge_prob,
                     ACCIDENT_PROB, EDGE_ALPHA, START_TIME, END_TIME, SERVICE_MIN):

    cluster_idx = np.array([node2idx[n] for n in cluster_nodes], dtype=np.int32)
    depot_idx = node2idx[depot]
    current_time = START_TIME
    remaining = cluster_idx.copy()
    current_pos = depot_idx
    day = 1
    hist = []

    while remaining.size > 0:
        target = np.random.choice(remaining)
        path = astar_numba(current_pos, [target], travel_matrix, edge_prob)
        if path[0] != current_pos:
            path = [current_pos] + path

        for i in range(1, len(path)):
            frm, to = path[i-1], path[i]
            accident = np.random.rand() < ACCIDENT_PROB
            edge_prob[frm,to] = update_edge(edge_prob[frm,to], accident, EDGE_ALPHA)
            travel = travel_matrix[frm,to]
            current_time += travel

            hist.append((day, idx2node[to], current_time, accident))
            if accident:
                current_pos = frm
                break
            else:
                current_pos = to
                remaining = remaining[remaining != to]

            if current_time > END_TIME:
                current_time = START_TIME
                current_pos = depot_idx
                day += 1
                break

    # Return to depot
    path_back = astar_numba(current_pos, [depot_idx], travel_matrix, edge_prob)
    for i in range(1,len(path_back)):
        frm,to = path_back[i-1], path_back[i]
        current_time += travel_matrix[frm,to]
        hist.append((day, idx2node[to], current_time, False))

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
