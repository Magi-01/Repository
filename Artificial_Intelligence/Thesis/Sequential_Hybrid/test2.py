#!/usr/bin/env python3
"""
clustered_cvrp_pipeline_full.py

- Parse CVRPLIB instances
- Cluster deliveries (KMeans)
- Per-cluster Q-learning or DQN (GPU via CuPy or PyTorch)
- Save/load cluster agents
- CP-SAT sequencing of clusters -> timeline + Gantt
- Optional OR-Tools Routing (multi-vehicle)
"""

import os, math, random, pickle, copy
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import json

file_path = "C:\\Users\\mutua\\Documents\\Repository\\Repository\\Artificial_Intelligence\\Thesis\\Sequential_Hybrid\\XML1000_1143_01.vrp"

START_TIME_MIN = 8*60
END_TIME_MIN = 18*60
SERVICE_MIN = 5
VEHICLE_SPEED = 20
MIN_PER_HOUR = 60

# -----------------------------
# 0. GPU Detection
# -----------------------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TORCH = torch.cuda.is_available() or torch.backends.mps.is_available()
print(f"PyTorch DQN enabled: {USE_TORCH}, device={device}")

# -----------------------------
# 1. CVRP Parser
# -----------------------------
def parse_cvrp_instance(fname):
    coords, demands, depots = {}, {}, set()
    mode = None
    with open(fname) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith("NODE_COORD_SECTION"): mode = "coords"; continue
            if ln.startswith("DEMAND_SECTION"): mode = "demand"; continue
            if ln.startswith("DEPOT_SECTION"): mode = "depots"; continue
            if mode == "coords":
                parts = ln.split()
                if len(parts) >= 3: coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif mode == "demand":
                parts = ln.split()
                if len(parts) >= 2: demands[int(parts[0])] = int(parts[1])
            elif mode == "depots":
                if ln == "-1": break
                depots.add(int(ln))
    deliveries = [n for n in coords if n not in depots]
    return coords, demands, depots, deliveries

# -----------------------------
# 2. Distance Matrix (GPU/CPU)
# -----------------------------
def build_distance_matrix(coords, nodes):
    pts = xp.asarray([coords[n] for n in nodes], dtype=float)
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    D = xp.sqrt(dx**2 + dy**2)
    travel_minutes = ((D/1000) / VEHICLE_SPEED) * 60
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    return travel_minutes, node2idx

def euclid(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def travel_minutes(a, b, speed=1.0): return euclid(a, b)/speed

# -----------------------------
# 3. Clustering
# -----------------------------
def cluster_deliveries(coords, deliveries, n_clusters=7, random_state=0):
    X = np.array([coords[d] for d in deliveries])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels): clusters[lab].append(deliveries[idx])
    centers = {i: tuple(kmeans.cluster_centers_[i]) for i in range(n_clusters)}
    return clusters, centers

# -----------------------------
# 4. ClusterTimeEnv
# -----------------------------

class ClusterTimeEnv:
    def __init__(self, cluster_nodes, depot, coords):
        self.depot = depot
        self.customers = list(cluster_nodes)
        self.nodes = [depot]+self.customers
        self.coords = coords
        self.D, self.node2idx = build_distance_matrix(coords, self.nodes)
        self.D_cpu = xp.asnumpy(self.D) if USE_CUPY else self.D
        self.m = len(self.customers)

    def travel_minutes(self, a,b):
        return float(self.D[self.node2idx[a], self.node2idx[b]])

    # Dict-based for Q-learning
    def actions_dict(self, state):
        loc, completed, cur_time = state
        feasible = [c for c in self.customers if c not in completed]
        return feasible

    def transition_dict(self, state, action):
        loc, completed, cur_time = state
        arrival = cur_time + self.travel_minutes(loc, action)
        finish = arrival + SERVICE_MIN
        # If finish exceeds END_TIME_MIN, roll over to next day
        if finish > END_TIME_MIN:
            days_later = int((finish - START_TIME_MIN) // (END_TIME_MIN - START_TIME_MIN) + 1)
            cur_time_next_day = START_TIME_MIN + (finish - END_TIME_MIN)  # time into next day
            finish = cur_time_next_day
            arrival = cur_time_next_day - SERVICE_MIN
        return {(action, frozenset(completed | {action}), finish): 1.0}

    def reward_dict(self, state, action, next_state):
        loc, completed, cur_time = state
        next_loc, new_completed, finish = next_state
        r = -0.1*self.travel_minutes(loc,next_loc)  # travel cost
        if len(new_completed) > len(completed): r += 50
        # Penalize if delivery goes to next day
        if finish > END_TIME_MIN:
            r -= 10*(finish - END_TIME_MIN)
        return r

# -----------------------------
# 5. Q-learning Agents
# -----------------------------
BITMASK_Q_THRESHOLD = 15

class BitmaskQAgent:
    def __init__(self, env:ClusterTimeEnv, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.env = env; self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        self.m = env.m; self.n_actions = self.m
        self.n_states = (self.m+1)*(1<<self.m)
        self.Q = xp.zeros((self.n_states,self.n_actions),dtype=float)

    def state_index(self, loc_node, mask):
        loc_idx = self.env.node2idx[loc_node]
        return loc_idx*(1<<self.m) + mask

    def available_actions_from_index(self, s_idx):
        mask = s_idx % (1<<self.m)
        return [i for i in range(self.m) if not ((mask>>i)&1)]

    def choose_action(self, s_idx):
        feasible = self.available_actions_from_index(s_idx)
        if not feasible: return None
        if random.random()<self.epsilon: return random.choice(feasible)
        row = self.Q[s_idx]
        row_cpu = xp.asnumpy(row) if USE_CUPY else row
        return max(feasible,key=lambda a: row_cpu[a])

    def step_update(self, s_idx, a_idx, s2_idx, reward):
        q_sa = self.Q[s_idx,a_idx]
        q_next_max = float(self.Q[s2_idx].max())
        self.Q[s_idx,a_idx] = q_sa + self.alpha*(reward + self.gamma*q_next_max - q_sa)

    def save(self, fname):
        Q_cpu = xp.asnumpy(self.Q) if USE_CUPY else self.Q
        pickle.dump({"Q":Q_cpu,"env_nodes":self.env.nodes,"m":self.m}, open(fname,"wb"))

    @staticmethod
    def load(fname, env):
        data = pickle.load(open(fname,"rb"))
        agent = BitmaskQAgent(env)
        agent.Q = xp.asarray(data["Q"]) if USE_CUPY else data["Q"]
        return agent

class DictQAgent:
    def __init__(self, env:ClusterTimeEnv, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.env = env; self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        self.Q = {}

    def choose_action(self,state):
        feasible = self.env.actions_dict(state)
        if not feasible: return None
        if state not in self.Q: self.Q[state]={a:0.0 for a in feasible}
        if random.random()<self.epsilon: return random.choice(feasible)
        return max(self.Q[state],key=self.Q[state].get)

    def update(self,state,action,reward,next_state):
        if state not in self.Q: self.Q[state]={a:0.0 for a in self.env.actions_dict(state)}
        if next_state not in self.Q: self.Q[next_state]={a:0.0 for a in self.env.actions_dict(next_state)}
        best_next = max(self.Q[next_state].values()) if self.Q[next_state] else 0
        self.Q[state][action] += self.alpha*(reward+self.gamma*best_next - self.Q[state][action])

    def save(self,fname): pickle.dump(self.Q,open(fname,"wb"))
    @staticmethod
    def load(fname, env):
        agent = DictQAgent(env); agent.Q = pickle.load(open(fname,"rb")); return agent
    
'''def reconstruct_schedule_from_bitmask_agent(agent:BitmaskQAgent, env:ClusterTimeEnv, start_time=START_TIME_MIN):
    mask = 0
    loc = env.depot
    s_idx = agent.state_index(loc, mask)
    schedule = []
    
    steps = 0
    while True:
        feasible = agent.available_actions_from_index(s_idx)
        if not feasible: break
        a = agent.choose_action(s_idx)
        next_node = env.customers[a]
        arrival = start_time + env.travel_minutes(loc, next_node)
        start_service = arrival
        finish = start_service + SERVICE_MIN
        schedule.append((next_node, arrival, start_service, finish))
        # update for next step
        mask |= (1<<a)
        loc = next_node
        s_idx = agent.state_index(loc, mask)
        steps += 1
        if steps > env.m*2: break  # safety
    return schedule
'''

# -----------------------------
# 6. Per-cluster Training
# -----------------------------
def run_episode(args):
    """Worker function for one process — handles both agent types."""
    env, agent_type, episodes, START_TIME_MIN = args

    if agent_type == 'bitmask':
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

        return agent.Q

    else:
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
    """Merge multiple Q-tables (nested dicts) by averaging shared entries."""
    merged = {}

    for table in tables:
        for state, action_dict in table.items():
            if state not in merged:
                merged[state] = action_dict.copy()
            else:
                for action, value in action_dict.items():
                    if action in merged[state]:
                        merged[state][action] = (merged[state][action] + value) / 2.0
                    else:
                        merged[state][action] = value
    return merged

def train_cluster_time(env: ClusterTimeEnv, episodes=2000, n_processes=4):
    """Parallel Q-learning training for cluster-time environment."""
    BITMASK_Q_THRESHOLD = 10  # adjust as appropriate
    START_TIME_MIN = 480

    # Choose agent type
    agent_type = 'bitmask' if env.m <= BITMASK_Q_THRESHOLD else 'dict'

    # Split episodes per process
    episodes_per_process = episodes // n_processes

    # Prepare arguments for each worker
    args_list = [
        (copy.deepcopy(env), agent_type, episodes_per_process, START_TIME_MIN)
        for _ in range(n_processes)
    ]

    # Run training in parallel
    with mp.Pool(processes=n_processes) as pool:
        q_tables = pool.map(run_episode, args_list)

    # Merge results
    merged_Q = merge_q_tables(q_tables)

    # Initialize final agent and assign merged Q
    final_agent = BitmaskQAgent(env) if agent_type == 'bitmask' else DictQAgent(env)
    final_agent.Q = merged_Q

    print(f"Training complete using {agent_type} agent ({n_processes} processes, {episodes} episodes total).")
    return final_agent

# -----------------------------
# 7. Schedule Reconstruction & Gantt
# -----------------------------
def reconstruct_schedule_from_agent(agent, env):
    """
    Reconstruct schedule from a trained agent (Dict or Bitmask),
    ensuring daily depot returns and respecting service/end time constraints.

    Returns a list of tuples:
        (node_id, arrival_time, start_service_time, finish_time)
    """
    # --- Constants ---
    START_TIME_MIN = getattr(env, "START_TIME_MIN", 480)  # default 8:00
    END_TIME_MIN = getattr(env, "END_TIME_MIN", 1080)     # default 18:00
    SERVICE_MIN = getattr(env, "SERVICE_MIN", 10)

    # --- Initialization ---
    unvisited = set(env.customers)
    absolute_time = START_TIME_MIN
    schedule = []
    depot = env.depot

    # Cache lookups for performance
    travel_minutes = env.travel_minutes
    is_bitmask_agent = hasattr(agent, "state_index")

    while unvisited:
        loc = depot
        cur_time = absolute_time

        while unvisited:
            # Construct the state based on agent type
            if is_bitmask_agent:
                # For bitmask agents, reconstruct mask
                visited_mask = 0
                for i, node in enumerate(env.customers):
                    if node not in unvisited:
                        visited_mask |= (1 << i)
                s_idx = agent.state_index(loc, visited_mask)
                feasible_actions = agent.available_actions_from_index(s_idx)
                if not feasible_actions:
                    break
                # Choose best action from Q-values
                q_vals = agent.Q.get(s_idx, {})
                if q_vals:
                    a_idx = max(q_vals, key=q_vals.get)
                    next_node = env.customers[a_idx]
                else:
                    next_node = random.choice(env.customers)
            else:
                # Dict agent
                visited_fs = frozenset(set(env.customers) - unvisited)
                state = (loc, visited_fs, cur_time)
                feasible_actions = list(unvisited)
                if not feasible_actions:
                    break

                if state in agent.Q and agent.Q[state]:
                    next_node = max(agent.Q[state], key=agent.Q[state].get)
                else:
                    next_node = random.choice(feasible_actions)

            # --- Compute travel & timing ---
            travel = travel_minutes(loc, next_node)
            arrival = cur_time + travel
            finish = arrival + SERVICE_MIN

            if finish > END_TIME_MIN:
                # Stop daily route — return to depot
                break

            # Record schedule
            schedule.append((next_node, arrival, arrival, finish))
            unvisited.discard(next_node)
            loc = next_node
            cur_time = finish

        # Return to depot and start next day
        cur_time += travel_minutes(loc, depot)

        #Return to depot before next day
        if loc != depot:
            arrival_back = cur_time + travel_minutes(loc, depot)
            finish_back = arrival_back  # no service
            schedule.append((depot, arrival_back, arrival_back, finish_back))
            loc = depot
            cur_time = finish_back

        absolute_time = ((cur_time // (24 * 60)) + 1) * 24 * 60 + START_TIME_MIN

    return schedule


def plot_gantt(schedule, cluster_id=None):
    if not schedule: 
        print("Schedule is empty!")
        return

    print(f"Plotting Gantt for cluster {cluster_id}, {len(schedule)} tasks")

    # Debug: print all schedule entries
    for idx, (node, arr, start, finish) in enumerate(schedule):
        print(f"{idx}: Node={node}, Arrival={arr}, Start={start}, Finish={finish}")
        if finish < start:
            print(f"  WARNING: finish < start at node {node}!")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(schedule))))

    for i, (node, arr, start, finish) in enumerate(schedule):
        # Validate numeric values
        if any(not isinstance(x, (int, float)) for x in [arr, start, finish]):
            print(f"  WARNING: Non-numeric time at node {node}: arr={arr}, start={start}, finish={finish}")
            continue

        # Draw task bar
        ax.barh(i, finish - start, left=start, height=0.6, color=f"C{cluster_id}" if cluster_id is not None else "C0")
        # Arrival line
        ax.plot([arr, arr], [i - 0.25, i + 0.25], color='black', linestyle='--', linewidth=1)
        # Node label
        ax.text(start + (finish - start) * 0.05, i, str(node), va='center')

    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([str(s[0]) for s in schedule])
    ax.set_xlabel('Time (minutes)')
    try:
        ax.set_xlim(START_TIME_MIN - 30, END_TIME_MIN + 60)
    except NameError:
        print("START_TIME_MIN or END_TIME_MIN not defined, using automatic limits")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_title(f'Gantt schedule for cluster {cluster_id}' if cluster_id is not None else 'Gantt schedule')
    
    plt.show()

def plot_full_gantt(cluster_schedules, cluster_routes, service_time=15):
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5*sum(len(s) for s in cluster_schedules.values()))))
    y_offset = 0
    yticks = []
    ylabels = []

    for cid, schedule in cluster_schedules.items():
        for task in schedule:
            node, arr, start, finish = task
            ax.barh(y_offset, finish-start, left=start, height=0.6, color=f"C{cid}")
            ax.plot([arr, arr], [y_offset-0.3, y_offset+0.3], color='black', linestyle='--', linewidth=1)
            ax.text(start+1, y_offset, str(node), va='center')
            yticks.append(y_offset)
            ylabels.append(f"C{cid}-N{node}")
            y_offset += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_title("Full Gantt for all clusters")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.show()

def plot_full_gantt_per_vehicle(cluster_schedules, cluster_routes, service_time=15):
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5*len(cluster_schedules))))
    yticks = []
    ylabels = []

    for vehicle_id, (cid, schedule) in enumerate(cluster_schedules.items()):
        start_times = []
        for task in schedule:
            node, arr, start, finish = task
            duration = finish - start
            ax.broken_barh(
                [(start, duration)],
                (vehicle_id - 0.4, 0.8),
                facecolors=f"C{cid}"
            )
            ax.plot([arr, arr], [vehicle_id - 0.4, vehicle_id + 0.4],
                    color='black', linestyle='--', linewidth=1)
            start_times.append(start)
        yticks.append(vehicle_id)
        ylabels.append(f"Vehicle {vehicle_id} (Cluster {cid})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_title("Full Gantt by Vehicle/Cluster")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.show()

def plot_clusters_and_routes(coords, clusters, cluster_routes, cluster_centers, depots=None):
    """
    coords: dict {node_id: (x,y)}
    clusters: dict {cluster_id: [node_ids]}
    cluster_routes: dict {cluster_id: [node_ids]} (ordered route)
    cluster_centers: dict {cluster_id: (x,y)}
    depots: set of depot node_ids (optional)
    """
    fig, ax = plt.subplots(figsize=(10,8))
    
    colors = plt.cm.get_cmap("tab10", len(clusters))
    
    # Plot nodes per cluster
    for cid, nodes in clusters.items():
        xs = [coords[n][0] for n in nodes]
        ys = [coords[n][1] for n in nodes]
        ax.scatter(xs, ys, color=colors(cid), label=f"Cluster {cid}", alpha=0.6)
        
        # Plot cluster center
        cx, cy = cluster_centers[cid]
        ax.scatter([cx],[cy], color=colors(cid), marker='X', s=150, edgecolor='k')

        # Draw route lines if available
        if cid in cluster_routes:
            route = cluster_routes[cid]
            route_coords = [coords[n] for n in route]
            rx, ry = zip(*route_coords)
            ax.plot(rx, ry, color=colors(cid), linestyle='-', linewidth=2, alpha=0.7)

    # Depots
    if depots:
        dx, dy = zip(*[coords[d] for d in depots])
        ax.scatter(dx, dy, color='black', marker='D', s=100, label="Depot")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Clusters, Cluster Centers, and Routes")
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_clusters_routes_with_depots(coords, clusters, cluster_routes, cluster_centers, depots):
    """
    coords: dict {node_id: (x,y)}
    clusters: dict {cluster_id: [node_ids]}
    cluster_routes: dict {cluster_id: [node_ids]} (ordered route)
    cluster_centers: dict {cluster_id: (x,y)}
    depots: set of depot node_ids
    """
    fig, ax = plt.subplots(figsize=(12,10))
    colors = plt.cm.get_cmap("tab10", len(clusters))

    # Plot nodes per cluster
    for cid, nodes in clusters.items():
        xs = [coords[n][0] for n in nodes]
        ys = [coords[n][1] for n in nodes]
        ax.scatter(xs, ys, color=colors(cid), label=f"Cluster {cid}", alpha=0.6)

        # Cluster center
        cx, cy = cluster_centers[cid]
        ax.scatter([cx],[cy], color=colors(cid), marker='X', s=150, edgecolor='k')

    # Plot depots
    dx, dy = zip(*[coords[d] for d in depots])
    ax.scatter(dx, dy, color='black', marker='D', s=120, label="Depot")

    # Draw routes for each cluster
    for cid, route in cluster_routes.items():
        if not route: continue
        # Choose closest depot to cluster center
        cx, cy = cluster_centers[cid]
        start_depot = min(depots, key=lambda d: math.hypot(coords[d][0]-cx, coords[d][1]-cy))
        start_coord = coords[start_depot]

        # Route coordinates: start depot -> route nodes -> return depot (closest to last node)
        path_nodes = [start_depot] + route
        last_node_coord = coords[route[-1]]
        return_depot = min(depots, key=lambda d: math.hypot(coords[d][0]-last_node_coord[0], coords[d][1]-last_node_coord[1]))
        path_nodes.append(return_depot)

        path_coords = [coords[n] for n in path_nodes]
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, color=colors(cid), linestyle='-', linewidth=2, alpha=0.7, label=f"Route Cluster {cid}")

        # Mark route nodes with numbers
        for idx, n in enumerate(route):
            ax.text(coords[n][0], coords[n][1]+0.5, str(n), fontsize=9, color=colors(cid))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Clusters, Depots, and Delivery Routes")
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_clusters_routes_and_gantt(coords, clusters, cluster_routes, cluster_schedules, cluster_centers, depots):
    """
    coords: dict {node_id: (x,y)}
    clusters: dict {cluster_id: [node_ids]}
    cluster_routes: dict {cluster_id: [node_ids]} ordered route
    cluster_schedules: dict {cluster_id: [(node, arrival, start, finish)]}
    cluster_centers: dict {cluster_id: (x,y)}
    depots: set of depot node_ids
    """
    import matplotlib.gridspec as gridspec
    colors = plt.cm.get_cmap("tab10", len(clusters))
    
    fig = plt.figure(figsize=(14,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
    
    # ------------------- Spatial Map -------------------
    ax_map = fig.add_subplot(gs[0])
    
    # Plot nodes per cluster
    for cid, nodes in clusters.items():
        xs = [coords[n][0] for n in nodes]
        ys = [coords[n][1] for n in nodes]
        ax_map.scatter(xs, ys, color=colors(cid), label=f"Cluster {cid}", alpha=0.6)
        # Cluster center
        cx, cy = cluster_centers[cid]
        ax_map.scatter([cx],[cy], color=colors(cid), marker='X', s=150, edgecolor='k')
    
    # Plot depots
    dx, dy = zip(*[coords[d] for d in depots])
    ax_map.scatter(dx, dy, color='black', marker='D', s=120, label="Depot")

    # Draw routes
    for cid, route in cluster_routes.items():
        if not route: continue
        # Start depot
        cx, cy = cluster_centers[cid]
        start_depot = min(depots, key=lambda d: math.hypot(coords[d][0]-cx, coords[d][1]-cy))
        path_nodes = [start_depot] + route
        # Return depot: closest to last node
        last_node_coord = coords[route[-1]]
        return_depot = min(depots, key=lambda d: math.hypot(coords[d][0]-last_node_coord[0], coords[d][1]-last_node_coord[1]))
        path_nodes.append(return_depot)
        xs, ys = zip(*[coords[n] for n in path_nodes])
        ax_map.plot(xs, ys, color=colors(cid), linestyle='-', linewidth=2, alpha=0.7, label=f"Route Cluster {cid}")
        # Node labels
        for n in route:
            ax_map.text(coords[n][0], coords[n][1]+0.5, str(n), fontsize=9, color=colors(cid))
    
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    ax_map.set_title("Clusters, Depots, and Routes")
    ax_map.legend()
    ax_map.grid(True)

    # ------------------- Gantt Chart -------------------
    ax_gantt = fig.add_subplot(gs[1])
    y_offset = 0
    yticks = []
    ylabels = []
    
    for vehicle_id, (cid, schedule) in enumerate(cluster_schedules.items()):
        for task in schedule:
            node, arr, start, finish = task
            duration = finish - start
            ax_gantt.broken_barh(
                [(start, duration)],
                (y_offset-0.4, 0.8),
                facecolors=f"C{cid}"
            )
            ax_gantt.plot([arr, arr], [y_offset-0.4, y_offset+0.4],
                          color='black', linestyle='--', linewidth=1)
        yticks.append(y_offset)
        ylabels.append(f"Cluster {cid}")
        y_offset += 1

    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels)
    ax_gantt.set_xlabel("Time (minutes)")
    ax_gantt.set_title("Delivery Schedule (Gantt) per Cluster")
    ax_gantt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def animate_clusters_routes_and_gantt(coords, clusters, cluster_routes, cluster_schedules, cluster_centers, depots, interval=200):
    """
    Animated version of plot_clusters_routes_and_gantt:
    - Vehicles move along their routes in the map view.
    - Gantt chart shows time progress with a moving time bar.
    """

    colors = plt.cm.get_cmap("tab10", len(clusters))

    # --- Helper to safely get coordinates ---
    def get_xy(node):
        """Return coordinate (x, y) whether node is an ID or already a coordinate."""
        if isinstance(node, tuple) and len(node) == 2:
            return node
        return coords[node]

    # --- Precompute per-cluster route and time info ---
    cluster_data = {}
    global_min_t, global_max_t = float("inf"), 0
    for cid, schedule in cluster_schedules.items():
        if not schedule:
            continue
        times = [t[1] for t in schedule] + [t[3] for t in schedule]
        min_t, max_t = min(times), max(times)
        global_min_t = min(global_min_t, min_t)
        global_max_t = max(global_max_t, max_t)
        cluster_data[cid] = {
            "schedule": schedule,
            "route": cluster_routes.get(cid, []),
            "current_index": 0,
            "current_pos": cluster_centers[cid],
            "min_t": min_t,
            "max_t": max_t
        }

    # --- Figure and layout ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_gantt = fig.add_subplot(gs[1])

    # --- Static map setup ---
    for cid, nodes in clusters.items():
        xs = [coords[n][0] for n in nodes]
        ys = [coords[n][1] for n in nodes]
        ax_map.scatter(xs, ys, color=colors(cid), alpha=0.5, label=f"Cluster {cid}")
        cx, cy = cluster_centers[cid]
        ax_map.scatter([cx], [cy], color=colors(cid), marker='X', s=150, edgecolor='k')

    # Plot depots
    dx, dy = zip(*[coords[d] for d in depots])
    ax_map.scatter(dx, dy, color='black', marker='D', s=120, label="Depot")

    # --- Vehicle markers ---
    vehicle_markers = {}
    for cid in clusters:
        c = colors(cid)
        (veh_marker,) = ax_map.plot([], [], 'o', color=c, markersize=10, label=f"Vehicle {cid}")
        vehicle_markers[cid] = veh_marker

    ax_map.legend()
    ax_map.set_title("Animated Cluster Routes")
    ax_map.grid(True)

    # --- Static Gantt chart setup ---
    y_offset = 0
    yticks, ylabels = [], []
    for cid, schedule in cluster_schedules.items():
        for node, arr, start, finish in schedule:
            duration = finish - start
            ax_gantt.broken_barh(
                [(start, duration)],
                (y_offset - 0.4, 0.8),
                facecolors=f"C{cid}"
            )
        yticks.append(y_offset)
        ylabels.append(f"Cluster {cid}")
        y_offset += 1

    # Time line on Gantt
    (time_line,) = ax_gantt.plot([global_min_t, global_min_t], [-0.5, len(clusters)], 'r-', linewidth=2)
    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels)
    ax_gantt.set_xlim(global_min_t, global_max_t)
    ax_gantt.set_xlabel("Time (minutes)")
    ax_gantt.set_title("Animated Delivery Gantt Chart")
    ax_gantt.grid(True, axis='x', linestyle='--', alpha=0.5)

    # --- Animation function ---
    def update(frame_time):
        # Update moving vertical time line on Gantt
        time_line.set_xdata([frame_time, frame_time])

        for cid, data in cluster_data.items():
            sched = data["schedule"]
            # Find current segment
            for i, (node, arr, start, finish) in enumerate(sched):
                if arr <= frame_time <= finish:
                    prev_node = cluster_centers[cid] if i == 0 else sched[i - 1][0]
                    next_node = node
                    t_ratio = (frame_time - arr) / max(1e-5, (finish - arr))
                    px, py = get_xy(prev_node)
                    nx, ny = get_xy(next_node)
                    x = px + (nx - px) * t_ratio
                    y = py + (ny - py) * t_ratio
                    vehicle_markers[cid].set_data([x], [y])
                    break
                elif frame_time < sched[0][1]:
                    # Before route starts — stay at depot (nearest to cluster center)
                    cx, cy = cluster_centers[cid]
                    depot_node = min(
                        depots,
                        key=lambda d: math.hypot(coords[d][0] - cx, coords[d][1] - cy)
                    )
                    dx, dy = get_xy(depot_node)
                    vehicle_markers[cid].set_data([dx], [dy])
        return list(vehicle_markers.values()) + [time_line]

    # --- Animate ---
    frames = np.linspace(global_min_t, global_max_t, int((global_max_t - global_min_t) / 2))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, repeat=False)

    plt.tight_layout()
    from IPython.display import HTML, display
    display(HTML(ani.to_jshtml()))
    return ani


# -----------------------------
# 8. Main runner
# -----------------------------
def main(file_path, n_clusters=7, per_cluster_episodes=2000):
    coords, demands, depots, deliveries = parse_cvrp_instance(file_path)
    print(f"Parsed {len(coords)} nodes, {len(deliveries)} deliveries, depots={depots}")
    clusters, cluster_centers = cluster_deliveries(coords, deliveries, n_clusters=n_clusters)
    print("Clusters sizes:", {k:len(v) for k,v in clusters.items()})

    cluster_routes={}
    cluster_schedules = {}
    for cid, nodes in clusters.items():
        print(f"\n Training cluster {cid} (size={len(nodes)})")

        # --- Select depot nearest to cluster centroid ---
        centroid = cluster_centers[cid]
        chosen_depot = min(
            depots,
            key=lambda d: math.hypot(coords[d][0] - centroid[0], coords[d][1] - centroid[1])
        )

        # --- Initialize environment ---
        env = ClusterTimeEnv(nodes, chosen_depot, coords)

        # --- Train the agent (parallelized inside function) ---
        agent = train_cluster_time(env, episodes=per_cluster_episodes)

        # --- Save trained agent ---
        agent_fname = f"cluster_agent_{cid}.pkl"
        agent.save(agent_fname)
        print(f"Saved {agent_fname}")

        # --- Reconstruct daily schedule ---
        schedule = reconstruct_schedule_from_agent(agent, env)

        # --- Plot Gantt chart or timeline ---
        #plot_gantt(schedule, cluster_id=cid)

        # --- Store results for later use ---
        cluster_routes[cid] = [node for node, *_ in schedule]
        cluster_schedules[cid] = schedule

    #nodes = list(coords.keys())          # all node IDs
    #distances, node2idx = build_distance_matrix(coords, nodes)
    #plot_full_gantt(cluster_schedules, cluster_routes, service_time=15)
    #plot_full_gantt_per_vehicle(cluster_schedules, cluster_routes, service_time=15)
    #plot_clusters_and_routes(coords,clusters,cluster_routes,cluster_centers)
    #plot_clusters_routes_with_depots(coords,clusters,cluster_routes,cluster_centers,depots)
    #plot_clusters_routes_and_gantt(coords,clusters,cluster_routes,cluster_schedules,cluster_centers,depots)
    #animate_clusters_routes_and_gantt(coords, clusters, cluster_routes, cluster_schedules, cluster_centers, depots, interval=200)


    print("\nCluster routes trained and saved.")
    return coords, depots, clusters, cluster_routes, cluster_centers, cluster_schedules

if __name__ == "__main__":
    coords, depots, clusters, cluster_routes, cluster_centers, cluster_schedules = main(
        file_path,
        n_clusters=5,
        per_cluster_episodes=1000
    )
    def serialize(obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
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
