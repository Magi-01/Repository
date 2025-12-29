#!/usr/bin/env python3
"""
clustered_cvrp_refactor.py
Refactored CVRP training pipeline focused on memory efficiency and LRU-bounded Q.
- Avoids deep-copying large objects to worker processes by using a Pool initializer.
- Uses on-the-fly distance computation to avoid storing large dense matrices.
- Uses float32 where large arrays appear.
- Discretizes time to reduce state cardinality.
- Uses a bounded LRU Q-table to cap memory growth.
"""

import os, math, random, argparse, json, gc, time, utility
from pathlib import Path
from collections import defaultdict, OrderedDict
try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy required. Install: pip install numpy\n"+str(e))

# Optional: scikit-learn for clustering
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# Multiprocessing
import multiprocessing as mp
import sys

# ---------------- Config ----------------
START_TIME_MIN = 8 * 60
END_TIME_MIN = 18 * 60
SERVICE_MIN = 5
VEHICLE_SPEED = 20.0  # km/h
TIME_BUCKET = 5  # minutes for time discretization
Q_MAX_STATES = 200_000  # max number of state entries across Q
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
DEFAULT_FLOAT = np.float32

# global read-only coords for workers (populated by Pool initializer)
GLOBAL_COORDS = None

# ---------------- Utilities ----------------
def parse_cvrp_instance(fname):
    coords, demands, depots = {}, {}, set()
    mode = None
    with open(fname) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("NODE_COORD_SECTION"):
                mode = "coords"; continue
            if ln.startswith("DEMAND_SECTION"):
                mode = "demand"; continue
            if ln.startswith("DEPOT_SECTION"):
                mode = "depots"; continue
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
                except:
                    pass
    deliveries = [n for n in coords if n not in depots]
    return coords, demands, depots, deliveries

def worker_init(coords):
    """
    Pool initializer: set global coords (read-only) for workers.
    Pass coords as a plain dict (small overhead for the initializer call).
    """
    global GLOBAL_COORDS
    # convert to simple floats tuple for memory predictability
    GLOBAL_COORDS = {int(k): (float(v[0]), float(v[1])) for k, v in coords.items()}

def travel_minutes_on_the_fly(a, b):
    """
    Compute travel minutes between nodes a and b using GLOBAL_COORDS.
    Called frequently but cheap (two tuple lookups + hypot).
    """
    xa, ya = GLOBAL_COORDS[a]
    xb, yb = GLOBAL_COORDS[b]
    d = math.hypot(xa - xb, ya - yb)  # same units as coords
    return ((d / 1000.0) / VEHICLE_SPEED) * 60.0

def make_state_key(loc, visited_count, cur_time):
    """
    Compact state key: (loc, num_visited, time_bucket)
    - This intentionally discards WHICH customers were visited to control state growth.
    - If you need more fidelity, consider using a bitmask for small clusters.
    """
    return (int(loc), int(visited_count), int(cur_time // TIME_BUCKET))

def nearest_k_depots(cluster_centroid, depots, coords, k=3):
    """
    Pure distance-based top-k selection.
    cluster_centroid: (x, y)
    depots: set of depot node IDs
    coords: {node_id: (x,y)}
    returns: [depot_id1, depot_id2, ...] sorted by distance
    """
    cx, cy = cluster_centroid
    dist_list = []

    for d in depots:
        dx, dy = coords[d]
        dist = math.hypot(cx - dx, cy - dy)
        dist_list.append((dist, d))

    dist_list.sort(key=lambda x: x[0])
    return [d for (_, d) in dist_list[:k]]

# ---------------- Bounded LRU Q ----------------
class BoundedLRUQ:
    """
    Bounded LRU-backed Q-table.
    - top-level mapping: state_key -> action_dict (dict of action -> value)
    - When max_states exceeded, evict oldest state_key entries.
    """
    def __init__(self, max_states=Q_MAX_STATES):
        self.max_states = int(max_states)
        self._od = OrderedDict()  # state_key -> action_dict

    def get_actions(self, state_key):
        # move-to-end for recency
        if state_key in self._od:
            self._od.move_to_end(state_key)
            return self._od[state_key]
        return None

    def ensure_state(self, state_key, feasible_actions):
        if state_key not in self._od:
            # evict oldest if needed
            if len(self._od) >= self.max_states:
                evicted = next(iter(self._od))
                self._od.pop(evicted, None)
            self._od[state_key] = {a: 0.0 for a in feasible_actions}
        else:
            # update recency
            self._od.move_to_end(state_key)
        return self._od[state_key]

    def update_value(self, state_key, action, new_value):
        actions = self.ensure_state(state_key, [action])
        actions[action] = new_value
        self._od.move_to_end(state_key)

    def items(self):
        return self._od.items()

    def __len__(self):
        return len(self._od)

    def dump_light(self):
        """Return a small serializable mapping for saving or merging (shallow copy)."""
        return {k: v.copy() for k, v in self._od.items()}

# ---------------- Agent ----------------
class DictQAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.2, q_max_states=Q_MAX_STATES):
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.Q = BoundedLRUQ(max_states=q_max_states)

    def choose_action(self, state_tuple):
        """
        state_tuple: (loc, visited_set, cur_time)
        choose among feasible (unvisited) customers
        """
        loc, visited_set, cur_time = state_tuple
        feasible = [c for c in self.env.customers if c not in visited_set]
        if not feasible:
            return None
        state_key = make_state_key(loc, len(visited_set), cur_time)
        actions = self.Q.get_actions(state_key)
        if actions is None:
            # initialize with feasible actions
            self.Q.ensure_state(state_key, feasible)
            actions = self.Q.get_actions(state_key)
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(feasible)
        # pick best among feasible — handle if actions has stale keys
        best = None
        best_v = -float('inf')
        for a in feasible:
            v = actions.get(a, 0.0)
            if v > best_v:
                best_v = v
                best = a
        return best

    def update(self, state_tuple, action, reward, next_state_tuple):
        s_loc, s_vis, s_time = state_tuple
        ns_loc, ns_vis, ns_time = next_state_tuple
        s_key = make_state_key(s_loc, len(s_vis), s_time)
        ns_key = make_state_key(ns_loc, len(ns_vis), ns_time)

        feasible_s = [c for c in self.env.customers if c not in s_vis]
        feasible_ns = [c for c in self.env.customers if c not in ns_vis]

        # ensure both states present (LRU will evict if needed)
        self.Q.ensure_state(s_key, feasible_s)
        self.Q.ensure_state(ns_key, feasible_ns)

        s_actions = self.Q.get_actions(s_key)
        ns_actions = self.Q.get_actions(ns_key)

        current_q = s_actions.get(action, 0.0)
        best_next = max(ns_actions.values()) if ns_actions else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * best_next - current_q)
        s_actions[action] = new_q
        # update recency
        self.Q._od.move_to_end(s_key)

# ---------------- Cluster Env (lightweight) ----------------
class ClusterTimeEnv:
    def __init__(self, cluster_nodes, depots_k, coords):
        """
        Lightweight environment:
        - does NOT copy global coords
        - only stores node lists (IDs)
        - uses travel_minutes_on_the_fly for distances
        depots_k: list of depot ids (nearest k depots)
        """
        self.depots = depots_k
        self.customers = list(cluster_nodes)

        # For simplicity, choose the FIRST depot as start
        # (but Q-agent is aware other depots exist)
        self.start_depot = depots_k[0]

        # Node list must include all depots being considered
        self.nodes = depots_k + self.customers

        self.coords = coords

        self.D, self.node2idx, self.idx2node = utility.build_distance_matrix(
                                                            coords,
                                                            self.nodes,
                                                            depots_k=self.depots)

        self.m = len(self.customers)

    def travel_minutes(self, a, b):
        return float(self.D[self.node2idx[a], self.node2idx[b]])

# ---------------- Training worker ----------------
def run_episode_worker(args):
    """
    Worker entrypoint used in Pool.map
    args: (cluster_nodes, depot, episodes_per_worker, agent_params)
    Returns: light q-table dict (serializable)
    """
    cluster_nodes, depot, coords, episodes, agent_kwargs = args
    env = ClusterTimeEnv(cluster_nodes, depot, coords)
    agent = DictQAgent(env, **agent_kwargs)

    for ep in range(episodes):
        state = (env.start_depot, frozenset(), START_TIME_MIN)
        steps = 0
        while True:
            action = agent.choose_action(state)
            if action is None:
                break
            travel = env.travel_minutes(state[0], action)
            finish = state[2] + travel + SERVICE_MIN
            # reward shaping: encourage servicing quickly and shorter travel
            reward = 50.0 - 0.1 * travel
            next_state = (action, frozenset(set(state[1]) | {action}), finish)
            agent.update(state, action, reward, next_state)
            state = next_state
            steps += 1
            if steps > len(env.customers) * 4:
                break
        # anneal epsilon per episode
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
    # Return a light serializable dump
    ret = agent.Q.dump_light()
    # small cleanup hints to GC
    del agent
    gc.collect()
    return ret

# ---------------- Merge Q tables incremental ----------------
def merge_q_tables_iter(tables_iterable):
    merged = {}
    count = 0
    for table in tables_iterable:
        for state_key, action_dict in table.items():
            # state_key came from make_state_key -> tuple
            if state_key not in merged:
                merged[state_key] = action_dict.copy()
            else:
                # average merging (could be replaced by more advanced aggregator)
                for a, v in action_dict.items():
                    merged[state_key][a] = (merged[state_key].get(a, 0.0) + v) / 2.0
        count += 1
        # free memory and hint GC after each table
        del table
        gc.collect()
    return merged

# ---------------- Schedule Reconstruction ----------------
def reconstruct_schedule_from_agent_q(merged_q, env):
    """
    Reconstruct schedule from merged Q (a light dict state_key->action_dict).
    Uses greedy policy from the merged Q with fallback random.
    Note: merged_q keys are compact state_keys; reconstructing per-cluster
    loses which specific customers were visited — so we still use "feasible" logic
    to choose next actions present in action dict, else random.
    """
    unvisited = set(env.customers)
    cur_time = START_TIME_MIN
    loc = env.start_depot  # use env, not merged_q

    schedule = []

    while unvisited:
        state = (loc, frozenset(set(env.customers) - unvisited), cur_time)
        feasible = [n for n in unvisited]

        if state in merged_q and merged_q[state]:
            next_node = max(merged_q[state], key=merged_q[state].get)
        else:
            next_node = random.choice(list(unvisited))

        travel = env.travel_minutes(loc, next_node)
        arrival = cur_time + travel
        finish = arrival + SERVICE_MIN

        schedule.append((next_node, arrival, arrival, finish))
        unvisited.remove(next_node)
        loc = next_node
        cur_time = finish

    return schedule


# ---------------- Top-level cluster training ----------------
def train_cluster_time(coords, cluster_nodes, depot, episodes=2000, n_processes=4, agent_kwargs=None):
    if agent_kwargs is None:
        agent_kwargs = {}
    # build args list for pool
    ep_per_worker = max(1, episodes // n_processes)
    args_list = [(cluster_nodes, depot, coords, ep_per_worker, agent_kwargs) for _ in range(n_processes)]
    # Use pool with initializer to populate GLOBAL_COORDS in workers
    with mp.Pool(processes=n_processes, initializer=worker_init, initargs=(coords,)) as pool:
        q_tables = pool.map(run_episode_worker, args_list)
    merged = merge_q_tables_iter(q_tables)
    return merged

# ---------------- Main ----------------
def main(input_file, n_clusters=7, episodes=1500, n_processes=4, q_max_states=Q_MAX_STATES):
    print(f"[DEBUG] Parsing CVRP instance: {input_file}")
    coords, demands, depots, deliveries = parse_cvrp_instance(input_file)
    print(f"[DEBUG] Found {len(coords)} nodes, {len(deliveries)} deliveries, {len(depots)} depots")

    if KMeans is None:
        raise RuntimeError("scikit-learn required for clustering. Install: pip install scikit-learn")

    print(f"[DEBUG] Clustering into {n_clusters} clusters")
    X = np.array([coords[d] for d in deliveries], dtype=DEFAULT_FLOAT)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels):
        clusters[lab].append(deliveries[idx])
    centers = {i: tuple(map(float, kmeans.cluster_centers_[i])) for i in range(n_clusters)}

    cluster_routes, cluster_schedules = {}, {}
    agent_kwargs = {"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2, "q_max_states": q_max_states}

    start_all = time.time()
    for cid, nodes in clusters.items():
        if not nodes:
            continue
        # choose nearest depot by euclidean distance to cluster center
        depots_k = nearest_k_depots(centers[cid], depots, coords, k=3)
        print(f"[DEBUG] Training cluster {cid} nearest depots: {depots_k}")

        env = ClusterTimeEnv(nodes, depots_k, coords)
        merged_q = train_cluster_time(coords, nodes, depots_k, episodes=episodes, n_processes=n_processes, agent_kwargs=agent_kwargs)
        # reconstruct a schedule for this cluster
        schedule = reconstruct_schedule_from_agent_q(merged_q, env)
        cluster_schedules[cid] = schedule
        cluster_routes[cid] = nodes
        print(f"[DEBUG] Cluster {cid} schedule length: {len(schedule)}")

    elapsed = time.time() - start_all
    print(f"[DEBUG] All clusters processed in {elapsed:.2f}s")

    # Save JSON with integer keys preserved
    save_data = {
        "coords": {int(k): v for k, v in coords.items()},
        "demands": {int(k): v for k, v in demands.items()},
        "clusters": {int(k): v for k, v in clusters.items()},
        "cluster_routes": {int(k): v for k, v in cluster_routes.items()},
        "cluster_schedules": {int(k): v for k, v in cluster_schedules.items()},
        "cluster_centers": {int(k): v for k, v in centers.items()},
        "depots": list(depots)
    }
    with open("cluster_data_refactored.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print("[DEBUG] Finished saving cluster data")
    return coords, demands, depots, clusters, cluster_routes, centers, cluster_schedules

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    default_file = BASE_DIR / "italy_auto_cities.vrp"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(default_file), help="Path to VRP file")
    parser.add_argument("--clusters", type=int, default=100, help="Number of clusters")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes per cluster (total)")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--qmax", type=int, default=Q_MAX_STATES, help="Max Q states (LRU capacity)")
    args = parser.parse_args()

    main(args.input, n_clusters=args.clusters, episodes=args.episodes, n_processes=args.processes, q_max_states=args.qmax)
