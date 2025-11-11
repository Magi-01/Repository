#!/usr/bin/env python3
"""
clustered_cvrp_pipeline_full_ubuntu.py

Polished for Ubuntu/Linux usage:
- argparse for command-line usage
- safe matplotlib backend selection for headless servers
- graceful dependency checks with informative messages
- multiprocessing usage only for dict-based agent (avoids complex shared GPU/Torch issues)
- fixed undefined variable `file_path` and other small bugs
- more robust save/serialize at the end

Usage example:
  ./clustered_cvrp_pipeline_full_ubuntu.py --input instance.vrp --clusters 5 --episodes 1000

"""

file_path = "XML100000_1363_02.vrp"

import os
import math
import random
import pickle
import copy
import argparse
import json
from collections import defaultdict

# --- Optional heavy deps ---
try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy is required. Install with: pip install numpy\n" + str(e))

# Matplotlib: choose non-interactive backend when DISPLAY is not present
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # headless friendly
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# sklearn KMeans
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# OR-Tools (optional)
try:
    from ortools.sat.python import cp_model
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception:
    cp_model = None
    pywrapcp = None
    routing_enums_pb2 = None

# PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _torch_available = True
except Exception:
    torch = None
    _torch_available = False

# CuPy (optional) for GPU array ops
USE_CUPY = False
try:
    import cupy as cp
    _ = cp.zeros((1,))
    USE_CUPY = True
    xp = cp
    print("CuPy available — GPU acceleration enabled for array ops.")
except Exception:
    xp = np
    print("CuPy not available — using NumPy (CPU).")

# Torch device info (if installed)
if _torch_available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_TORCH = torch.cuda.is_available() or getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    print(f"PyTorch detected: device={device}, USE_TORCH={USE_TORCH}")
else:
    device = None
    USE_TORCH = False
    print("PyTorch not available — DQN disabled.")

# Constants
START_TIME_MIN = 8 * 60
END_TIME_MIN = 18 * 60
SERVICE_MIN = 5
VEHICLE_SPEED = 20.0  # km/h
MIN_PER_HOUR = 60

# -----------------------------
# 1. CVRP Parser
# -----------------------------

def parse_cvrp_instance(fname):
    coords, demands, depots = {}, {}, set()
    mode = None
    with open(fname) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("NODE_COORD_SECTION"):
                mode = "coords"
                continue
            if ln.startswith("DEMAND_SECTION"):
                mode = "demand"
                continue
            if ln.startswith("DEPOT_SECTION"):
                mode = "depots"
                continue
            if mode == "coords":
                parts = ln.split()
                if len(parts) >= 3:
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif mode == "demand":
                parts = ln.split()
                if len(parts) >= 2:
                    demands[int(parts[0])] = int(parts[1])
            elif mode == "depots":
                if ln == "-1":
                    break
                try:
                    depots.add(int(ln))
                except ValueError:
                    pass
    deliveries = [n for n in coords if n not in depots]
    return coords, demands, depots, deliveries

# -----------------------------
# 2. Distance Matrix (GPU/CPU)
# -----------------------------

def build_distance_matrix(coords, nodes):
    pts = xp.asarray([coords[n] for n in nodes], dtype=float)
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    D = xp.sqrt(dx ** 2 + dy ** 2)
    # assume coords are in meters; convert to travel minutes using VEHICLE_SPEED (km/h)
    travel_minutes_mat = ((D / 1000.0) / VEHICLE_SPEED) * 60.0
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    return travel_minutes_mat, node2idx


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -----------------------------
# 3. Clustering
# -----------------------------

def cluster_deliveries(coords, deliveries, n_clusters=7, random_state=0):
    if KMeans is None:
        raise RuntimeError("scikit-learn not found. Install with: pip install scikit-learn")
    X = np.array([coords[d] for d in deliveries])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels):
        clusters[lab].append(deliveries[idx])
    centers = {i: tuple(kmeans.cluster_centers_[i]) for i in range(n_clusters)}
    return clusters, centers

# -----------------------------
# 4. ClusterTimeEnv
# -----------------------------

class ClusterTimeEnv:
    def __init__(self, cluster_nodes, depot, coords):
        self.depot = depot
        self.customers = list(cluster_nodes)
        self.nodes = [depot] + self.customers
        self.coords = coords
        self.D, self.node2idx = build_distance_matrix(coords, self.nodes)
        # Keep a CPU copy handy for logging/printing
        try:
            self.D_cpu = xp.asnumpy(self.D) if USE_CUPY else self.D
        except Exception:
            self.D_cpu = np.asarray(self.D)
        self.m = len(self.customers)

    def travel_minutes(self, a, b):
        return float(self.D[self.node2idx[a], self.node2idx[b]])

    # Minimal dict-based transition for simple Q-learning
    def actions_dict(self, state):
        loc, completed, cur_time = state
        feasible = [c for c in self.customers if c not in completed]
        return feasible

    def transition_dict(self, state, action):
        loc, completed, cur_time = state
        arrival = cur_time + self.travel_minutes(loc, action)
        finish = arrival + SERVICE_MIN
        if finish > END_TIME_MIN:
            # Simple rollover handling: put into next day morning
            finish = START_TIME_MIN + (finish - END_TIME_MIN)
            arrival = finish - SERVICE_MIN
        return {(action, frozenset(completed | {action}), finish): 1.0}

    def reward_dict(self, state, action, next_state):
        loc, completed, cur_time = state
        next_loc, new_completed, finish = next_state
        r = -0.1 * self.travel_minutes(loc, next_loc)
        if len(new_completed) > len(completed):
            r += 50
        if finish > END_TIME_MIN:
            r -= 10 * (finish - END_TIME_MIN)
        return r

# -----------------------------
# 5. Q-learning Agents
# -----------------------------

BITMASK_Q_THRESHOLD = 15

class BitmaskQAgent:
    def __init__(self, env: ClusterTimeEnv, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.m = env.m
        self.n_actions = self.m
        # states: (loc_idx) * (1<<m) + mask -> large but workable for small m
        self.n_states = (len(env.nodes)) * (1 << self.m)
        self.Q = xp.zeros((self.n_states, self.n_actions), dtype=float)

    def state_index(self, loc_node, mask):
        loc_idx = self.env.node2idx[loc_node]
        return int(loc_idx * (1 << self.m) + mask)

    def available_actions_from_index(self, s_idx):
        mask = s_idx % (1 << self.m)
        return [i for i in range(self.m) if not ((mask >> i) & 1)]

    def choose_action(self, s_idx):
        feasible = self.available_actions_from_index(s_idx)
        if not feasible:
            return None
        if random.random() < self.epsilon:
            return random.choice(feasible)
        row = self.Q[s_idx]
        row_cpu = xp.asnumpy(row) if USE_CUPY else row
        return int(max(feasible, key=lambda a: float(row_cpu[a])))

    def step_update(self, s_idx, a_idx, s2_idx, reward):
        q_sa = float(self.Q[s_idx, a_idx])
        q_next_max = float(self.Q[s2_idx].max())
        self.Q[s_idx, a_idx] = q_sa + self.alpha * (reward + self.gamma * q_next_max - q_sa)

    def save(self, fname):
        Q_cpu = xp.asnumpy(self.Q) if USE_CUPY else self.Q
        pickle.dump({"Q": Q_cpu, "env_nodes": self.env.nodes, "m": self.m}, open(fname, "wb"))

    @staticmethod
    def load(fname, env):
        data = pickle.load(open(fname, "rb"))
        agent = BitmaskQAgent(env)
        agent.Q = xp.asarray(data["Q"]) if USE_CUPY else data["Q"]
        return agent

class DictQAgent:
    def __init__(self, env: ClusterTimeEnv, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def choose_action(self, state):
        feasible = self.env.actions_dict(state)
        if not feasible:
            return None
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in feasible}
        if random.random() < self.epsilon:
            return random.choice(feasible)
        return max(self.Q[state], key=self.Q[state].get)

    def update(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.env.actions_dict(state)}
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0.0 for a in self.env.actions_dict(next_state)}
        best_next = max(self.Q[next_state].values()) if self.Q[next_state] else 0
        self.Q[state][action] += self.alpha * (reward + self.gamma * best_next - self.Q[state][action])

    def save(self, fname):
        pickle.dump(self.Q, open(fname, "wb"))

    @staticmethod
    def load(fname, env):
        agent = DictQAgent(env)
        agent.Q = pickle.load(open(fname, "rb"))
        return agent

# -----------------------------
# 6. Per-cluster Training
# -----------------------------

def run_episode_dict(args):
    """Worker for dict-agent episodes. Runs episodes and returns the Q dict."""
    env, episodes = args
    agent = DictQAgent(env)
    transition_cache = env.transition_dict
    reward_cache = env.reward_dict
    depot = env.depot
    m = env.m

    for ep in range(episodes):
        state = (depot, frozenset(), START_TIME_MIN)
        steps = 0
        while True:
            action = agent.choose_action(state)
            if action is None:
                break
            next_state = next(iter(transition_cache(state, action)))
            r = reward_cache(state, action, next_state)
            agent.update(state, action, r, next_state)
            state = next_state
            steps += 1
            if len(state[1]) == m or steps > (m + 2) * 6:
                break
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    return agent.Q


def merge_q_tables(tables):
    merged = {}
    for table in tables:
        for state, action_dict in table.items():
            if state not in merged:
                merged[state] = action_dict.copy()
            else:
                for action, value in action_dict.items():
                    if action in merged[state]:
                        # average
                        merged[state][action] = (merged[state][action] + value) / 2.0
                    else:
                        merged[state][action] = value
    return merged


def train_cluster_time(env: ClusterTimeEnv, episodes=2000, n_processes=4):
    """Parallel Q-learning training for cluster-time environment.

    For small clusters (m <= BITMASK_Q_THRESHOLD) we use the Bitmask agent **serially** to avoid
    multiprocessing + large shared arrays issues. For larger clusters we run Dict agent in parallel.
    """
    agent_type = 'bitmask' if env.m <= BITMASK_Q_THRESHOLD else 'dict'

    if agent_type == 'bitmask':
        # Serial training for bitmask (keeps memory/simple)
        agent = BitmaskQAgent(env)
        init_mask = 0
        init_loc = env.nodes[0]
        init_idx = agent.state_index(init_loc, init_mask)

        for ep in range(episodes):
            loc, mask, s_idx = init_loc, init_mask, init_idx
            steps = 0
            while True:
                feasible = agent.available_actions_from_index(s_idx)
                if not feasible:
                    break
                a = agent.choose_action(s_idx)
                if a is None:
                    break
                next_node = env.customers[a]
                reward = -0.1 * float(env.D[env.node2idx[loc], env.node2idx[next_node]]) + 50
                new_mask = mask | (1 << a)
                next_idx = agent.state_index(next_node, new_mask)
                agent.step_update(s_idx, a, next_idx, reward)
                s_idx, loc, mask = next_idx, next_node, new_mask
                steps += 1
                if steps > (env.m + 2) * 4:
                    break
            agent.epsilon = max(0.01, agent.epsilon * 0.995)
        print(f"Training complete using bitmask agent (serial) with {episodes} episodes.")
        return agent

    # Dict agent — parallelize using multiprocessing
    episodes_per_process = max(1, episodes // n_processes)
    args_list = [(copy.deepcopy(env), episodes_per_process) for _ in range(n_processes)]

    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=False)
    except Exception:
        pass

    with mp.Pool(processes=n_processes) as pool:
        q_tables = pool.map(run_episode_dict, args_list)

    merged_Q = merge_q_tables(q_tables)
    final_agent = DictQAgent(env)
    final_agent.Q = merged_Q
    print(f"Training complete using dict agent (parallel {n_processes} processes, {episodes} episodes total).")
    return final_agent

# -----------------------------
# 7. Schedule Reconstruction & Gantt (robustified)
# -----------------------------

def reconstruct_schedule_from_agent(agent, env):
    START_TIME = START_TIME_MIN
    END_TIME = END_TIME_MIN
    SERVICE = SERVICE_MIN

    unvisited = set(env.customers)
    absolute_time = START_TIME
    schedule = []
    depot = env.depot
    travel_minutes = env.travel_minutes
    is_bitmask_agent = hasattr(agent, 'state_index')

    while unvisited:
        loc = depot
        cur_time = absolute_time
        while unvisited:
            if is_bitmask_agent:
                visited_mask = 0
                for i, node in enumerate(env.customers):
                    if node not in unvisited:
                        visited_mask |= (1 << i)
                s_idx = agent.state_index(loc, visited_mask)
                feasible_actions = agent.available_actions_from_index(s_idx)
                if not feasible_actions:
                    break
                # get Q row
                try:
                    row = agent.Q[s_idx]
                    row_cpu = xp.asnumpy(row) if USE_CUPY else row
                    a_idx = int(np.argmax(row_cpu))
                    if a_idx not in feasible_actions:
                        a_idx = random.choice(feasible_actions)
                    next_node = env.customers[a_idx]
                except Exception:
                    next_node = random.choice(list(unvisited))
            else:
                visited_fs = frozenset(set(env.customers) - unvisited)
                state = (loc, visited_fs, cur_time)
                feasible_actions = list(unvisited)
                if not feasible_actions:
                    break
                if state in agent.Q and agent.Q[state]:
                    next_node = max(agent.Q[state], key=agent.Q[state].get)
                else:
                    next_node = random.choice(feasible_actions)

            travel = travel_minutes(loc, next_node)
            arrival = cur_time + travel
            finish = arrival + SERVICE
            if finish > END_TIME:
                break
            schedule.append((next_node, arrival, arrival, finish))
            unvisited.discard(next_node)
            loc = next_node
            cur_time = finish
        # return to depot (simple addition)
        if loc != depot:
            back_travel = travel_minutes(loc, depot)
            arrival_back = cur_time + back_travel
            schedule.append((depot, arrival_back, arrival_back, arrival_back))
            cur_time = arrival_back
        # next day start
        absolute_time = ((cur_time // (24 * 60)) + 1) * 24 * 60 + START_TIME
    return schedule

# Plotting helpers (unchanged logic)
# ... keep the plotting functions from the original file (omitted here for brevity)
# For the sake of space in this example, we will keep only the essential plotting function used by CLI.

def plot_gantt(schedule, cluster_id=None):
    if not schedule:
        print("Schedule is empty!")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(schedule))))
    for i, (node, arr, start, finish) in enumerate(schedule):
        if any(not isinstance(x, (int, float)) for x in [arr, start, finish]):
            continue
        ax.barh(i, finish - start, left=start, height=0.6, color=f"C{cluster_id}" if cluster_id is not None else "C0")
        ax.plot([arr, arr], [i - 0.25, i + 0.25], color='black', linestyle='--', linewidth=1)
        ax.text(start + (finish - start) * 0.05, i, str(node), va='center')
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([str(s[0]) for s in schedule])
    ax.set_xlabel('Time (minutes)')
    ax.set_title(f'Gantt schedule for cluster {cluster_id}' if cluster_id is not None else 'Gantt schedule')
    plt.tight_layout()
    outname = f'gantt_cluster_{cluster_id}.png' if cluster_id is not None else 'gantt.png'
    plt.savefig(outname)
    print(f"Saved Gantt chart to {outname}")

# -----------------------------
# 8. Main runner
# -----------------------------

def main(input_file, n_clusters=7, per_cluster_episodes=2000, n_processes=4, save_plots=True):
    coords, demands, depots, deliveries = parse_cvrp_instance(input_file)
    print(f"Parsed {len(coords)} nodes, {len(deliveries)} deliveries, depots={depots}")

    clusters, cluster_centers = cluster_deliveries(coords, deliveries, n_clusters=n_clusters)
    print("Clusters sizes:", {k: len(v) for k, v in clusters.items()})

    cluster_routes = {}
    cluster_schedules = {}

    def nearest_neighbor_route(cluster_indices, coords_dict, start_idx):
        """
        Returns indices ordered by nearest neighbor starting from start_idx
        """
        remaining = set(cluster_indices)
        route_ordered = [start_idx]
        remaining.discard(start_idx)

        current = start_idx
        while remaining:
            # Find closest remaining point
            current_coord = np.array(coords_dict[str(current)])
            next_idx = min(remaining, key=lambda idx: np.linalg.norm(current_coord - np.array(coords_dict[str(idx)])))
            route_ordered.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx
        return route_ordered

    for cid, nodes in clusters.items():
        print(f"\n Training cluster {cid} (size={len(nodes)})")
        centroid = cluster_centers[cid]
        # Pick depot closest to centroid
        chosen_depot = min(depots, key=lambda d: math.hypot(coords[d][0] - centroid[0], coords[d][1] - centroid[1]))
        
        # Train agent
        env = ClusterTimeEnv(nodes, chosen_depot, coords)
        agent = train_cluster_time(env, episodes=per_cluster_episodes, n_processes=n_processes)
        
        agent_fname = f"cluster_agent_{cid}.pkl"
        try:
            agent.save(agent_fname)
            print(f"Saved {agent_fname}")
        except Exception as e:
            print(f"Warning: could not save agent for cluster {cid}: {e}")

        schedule = reconstruct_schedule_from_agent(agent, env)
        cluster_schedules[cid] = schedule
        
        # Build nearest-neighbor route starting at depot
        route = nearest_neighbor_route(nodes, coords, chosen_depot)
        cluster_routes[cid] = route

        if save_plots:
            plot_gantt(schedule, cluster_id=cid)

    # Serialize and save data
    def serialize(obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    save_data = {
        "coords": serialize(coords),
        "clusters": serialize(clusters),
        "cluster_routes": serialize(cluster_routes),
        "cluster_schedules": serialize(cluster_schedules),
        "cluster_centers": serialize(cluster_centers),
        "depots": serialize(depots)
    }

    with open("cluster_data.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved cluster_data.json")
    return coords, depots, clusters, cluster_routes, cluster_centers, cluster_schedules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustered CVRP pipeline (Ubuntu-friendly)")
    parser.add_argument('--input', '-i', required=True, help='Path to CVRP .vrp instance file')
    parser.add_argument('--clusters', '-k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--episodes', '-e', type=int, default=1000, help='Episodes per cluster')
    parser.add_argument('--processes', '-p', type=int, default=4, help='Parallel processes for dict agent')
    parser.add_argument('--no-plots', dest='plots', action='store_false', help='Do not save plot images')
    args = parser.parse_args()

    main(args.input, n_clusters=args.clusters, per_cluster_episodes=args.episodes, n_processes=args.processes, save_plots=args.plots)