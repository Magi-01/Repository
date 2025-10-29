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

import os, math, random, pickle
from collections import defaultdict
import numpy as np_cpu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import torch
import torch.nn as nn
import torch.optim as optim

file_path = "C:\\Users\\mutua\\Documents\\Repository\\Repository\\Artificial_Intelligence\\Thesis\\Sequential_Hybrid\\XML100_1144_01.vrp"

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
    xp = np_cpu
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
    node2idx = {nodes[i]: i for i in range(len(nodes))}
    return D, node2idx

def euclid(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def travel_minutes(a, b, speed=1.0): return euclid(a, b)/speed

# -----------------------------
# 3. Clustering
# -----------------------------
def cluster_deliveries(coords, deliveries, n_clusters=7, random_state=0):
    X = np_cpu.array([coords[d] for d in deliveries])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels): clusters[lab].append(deliveries[idx])
    centers = {i: tuple(kmeans.cluster_centers_[i]) for i in range(n_clusters)}
    return clusters, centers

# -----------------------------
# 4. ClusterTimeEnv
# -----------------------------
START_TIME_MIN = 8*60
END_TIME_MIN = 18*60
SERVICE_MIN = 15
VEHICLE_SPEED = 60

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
    
def reconstruct_schedule_from_bitmask_agent(agent:BitmaskQAgent, env:ClusterTimeEnv, start_time=START_TIME_MIN):
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

# -----------------------------
# 6. Per-cluster Training
# -----------------------------
def train_cluster_time(env:ClusterTimeEnv, episodes=2000):
    agent_type = 'bitmask' if env.m<=BITMASK_Q_THRESHOLD else 'dict'
    if agent_type=='bitmask':
        agent = BitmaskQAgent(env); init_mask=0; init_loc=env.nodes[0]; init_idx=agent.state_index(init_loc,init_mask)
        for ep in range(episodes):
            loc, mask, s_idx = init_loc, init_mask, init_idx
            total_r=0
            steps=0
            while True:
                feasible = agent.available_actions_from_index(s_idx)
                if not feasible: break
                a = agent.choose_action(s_idx); next_node=env.customers[a]
                reward = -0.1*float(env.D[env.node2idx[loc],env.node2idx[next_node]])+50
                new_mask = mask|(1<<a); next_idx = agent.state_index(next_node,new_mask)
                agent.step_update(s_idx,a,next_idx,reward); total_r+=reward
                s_idx, loc, mask = next_idx, next_node, new_mask; steps+=1
                if steps>(env.m+2)*4: break
            agent.epsilon = max(0.01,agent.epsilon*0.995)
        return agent
    else:
        agent = DictQAgent(env)
        for ep in range(episodes):
            state = (env.depot,frozenset(),START_TIME_MIN); steps=0
            while True:
                action = agent.choose_action(state)
                if action is None: break
                next_state = list(env.transition_dict(state,action).keys())[0]
                r = env.reward_dict(state,action,next_state)
                agent.update(state,action,r,next_state)
                state = next_state; steps+=1
                if len(state[1])==env.m or steps>(env.m+2)*6: break
            agent.epsilon = max(0.01,agent.epsilon*0.995)
        return agent

# -----------------------------
# 7. Schedule Reconstruction & Gantt
# -----------------------------
def reconstruct_schedule_from_dict_agent(agent: DictQAgent, env: ClusterTimeEnv):
    """
    Reconstruct schedule ensuring depot return each day.
    Returns list of tuples:
    (node_id, absolute_time_minutes, start_service_time, finish_time)
    """
    unvisited = set(env.customers)
    absolute_time = START_TIME_MIN
    schedule = []

    while unvisited:
        loc = env.depot  # start at depot
        cur_time = absolute_time

        # Visit as many as possible within the day
        while True:
            state = (loc, frozenset(set(env.customers) - unvisited), cur_time)
            feasible = list(unvisited)
            if not feasible:
                break

            # choose best action according to agent
            if state in agent.Q and agent.Q[state]:
                action = max(agent.Q[state], key=agent.Q[state].get)
            else:
                action = random.choice(feasible)

            travel = env.travel_minutes(loc, action)
            arrival = cur_time + travel
            finish = arrival + SERVICE_MIN

            if finish > END_TIME_MIN:
                # cannot complete today, schedule next day at START_TIME_MIN
                break

            schedule.append((action, arrival, arrival, finish))
            cur_time = finish
            loc = action
            unvisited.remove(action)

        # End of day: return to depot
        travel_back = env.travel_minutes(loc, env.depot)
        cur_time += travel_back
        # Start next day at 8:00
        absolute_time = ((cur_time // (24*60)) * 24*60) + START_TIME_MIN

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
        print(f"\nTraining cluster {cid}, size={len(nodes)}")
        centroid = cluster_centers[cid]
        chosen_depot = min(depots,key=lambda d: math.hypot(coords[d][0]-centroid[0],coords[d][1]-centroid[1]))
        env = ClusterTimeEnv(nodes,chosen_depot,coords)
        agent = train_cluster_time(env,episodes=per_cluster_episodes)
        agent_fname=f"cluster_agent_{cid}.pkl"; agent.save(agent_fname)
        if isinstance(agent,DictQAgent):
            schedule = reconstruct_schedule_from_dict_agent(agent,env)
            plot_gantt(schedule, cluster_id=cid)
        else:
            schedule = reconstruct_schedule_from_bitmask_agent(agent, env)

        cluster_routes[cid] = [x[0] for x in schedule]
        cluster_schedules[cid] = schedule

    #nodes = list(coords.keys())          # all node IDs
    #distances, node2idx = build_distance_matrix(coords, nodes)
    plot_full_gantt(cluster_schedules, cluster_routes, service_time=15)
    plot_full_gantt_per_vehicle(cluster_schedules, cluster_routes, service_time=15)
    plot_clusters_and_routes(coords,clusters,cluster_routes,cluster_centers)
    plot_clusters_routes_with_depots(coords,clusters,cluster_routes,cluster_centers,depots)
    plot_clusters_routes_and_gantt(coords,clusters,cluster_routes,cluster_schedules,cluster_centers,depots)


    print("\nCluster routes trained and saved.")
    return coords, depots, clusters, cluster_routes, cluster_centers

if __name__ == "__main__":
    coords, depots, clusters, cluster_routes, cluster_centers = main(
        file_path,
        n_clusters=5,
        per_cluster_episodes=1000
    )
