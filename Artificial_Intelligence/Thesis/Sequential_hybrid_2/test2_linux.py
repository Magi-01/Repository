#!/usr/bin/env python3
"""
clustered_cvrp_pipeline_full_ubuntu.py
- Handles large CVRP instances efficiently.
- Saves per-cluster schedules, routes, agents, and demands to JSON.
- Fixed integer keys for compatibility with Simulation_with_csp.py
"""
import os, math, random, pickle, copy, argparse, json
from collections import defaultdict
try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy required. Install: pip install numpy\n"+str(e))

# Optional dependencies
try: from sklearn.cluster import KMeans
except Exception: KMeans = None

USE_CUPY = False
try:
    import cupy as cp
    _ = cp.zeros((1,))
    xp = cp
    USE_CUPY = True
except Exception:
    xp = np

# CVRP constants
START_TIME_MIN = 8*60
END_TIME_MIN = 18*60
SERVICE_MIN = 5
VEHICLE_SPEED = 20.0  # km/h

# ---------------- CVRP Parser ----------------
def parse_cvrp_instance(fname):
    coords, demands, depots = {}, {}, set()
    mode = None
    with open(fname) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith("NODE_COORD_SECTION"): mode="coords"; continue
            if ln.startswith("DEMAND_SECTION"): mode="demand"; continue
            if ln.startswith("DEPOT_SECTION"): mode="depots"; continue
            if mode=="coords":
                parts = ln.split()
                if len(parts)>=3: coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif mode=="demand":
                parts = ln.split()
                if len(parts)>=2: demands[int(parts[0])] = int(parts[1])
            elif mode=="depots":
                if ln=="-1": break
                try: depots.add(int(ln))
                except: pass
    deliveries = [n for n in coords if n not in depots]
    return coords, demands, depots, deliveries

# ---------------- Distance Matrix ----------------
def build_distance_matrix(coords, nodes):
    pts = xp.asarray([coords[n] for n in nodes], dtype=float)
    dx = pts[:,None,0] - pts[None,:,0]
    dy = pts[:,None,1] - pts[None,:,1]
    D = xp.sqrt(dx**2 + dy**2)
    travel_minutes_mat = ((D/1000.0)/VEHICLE_SPEED)*60.0
    node2idx = {nodes[i]:i for i in range(len(nodes))}  # integer keys
    return travel_minutes_mat, node2idx

# ---------------- Clustering ----------------
def cluster_deliveries(coords, deliveries, n_clusters=7, random_state=0):
    if KMeans is None: raise RuntimeError("scikit-learn required")
    X = np.array([coords[d] for d in deliveries])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels): clusters[lab].append(deliveries[idx])
    centers = {i: tuple(kmeans.cluster_centers_[i]) for i in range(n_clusters)}
    return clusters, centers

# ---------------- Cluster Environment ----------------
class ClusterTimeEnv:
    def __init__(self, cluster_nodes, depot, coords):
        self.depot = depot
        self.customers = list(cluster_nodes)
        self.nodes = [depot]+self.customers
        self.coords = coords
        self.D, self.node2idx = build_distance_matrix(coords, self.nodes)
        self.m = len(self.customers)

    def travel_minutes(self,a,b):
        return float(self.D[self.node2idx[a], self.node2idx[b]])

# ---------------- Dict Q Agent ----------------
class DictQAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.env = env
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = {}

    def choose_action(self, state):
        feasible = [c for c in self.env.customers if c not in state[1]]
        if not feasible: return None
        if state not in self.Q: self.Q[state] = {a:0.0 for a in feasible}
        if random.random()<self.epsilon: return random.choice(feasible)
        return max(self.Q[state], key=self.Q[state].get)

    def update(self, state, action, reward, next_state):
        if state not in self.Q: self.Q[state] = {a:0.0 for a in self.env.customers if a not in state[1]}
        if next_state not in self.Q: self.Q[next_state] = {a:0.0 for a in self.env.customers if a not in next_state[1]}
        best_next = max(self.Q[next_state].values()) if self.Q[next_state] else 0
        self.Q[state][action] += self.alpha*(reward + self.gamma*best_next - self.Q[state][action])

    #def save(self,fname): pickle.dump(self.Q, open(fname,"wb"))
    #@staticmethod
    #def load(fname,env):
    #    agent = DictQAgent(env)
    #    agent.Q = pickle.load(open(fname,"rb"))
    #    return agent

# ---------------- Cluster Training ----------------
def run_episode_dict(args):
    env, episodes = args
    agent = DictQAgent(env)
    for ep in range(episodes):
        state = (env.depot, frozenset(), START_TIME_MIN)
        steps = 0
        while True:
            action = agent.choose_action(state)
            if action is None: break
            travel = env.travel_minutes(state[0],action)
            finish = state[2]+travel+SERVICE_MIN
            reward = 50 - 0.1*travel  # simple reward
            next_state = (action, frozenset(set(state[1])|{action}), finish)
            agent.update(state, action, reward, next_state)
            state = next_state
            steps +=1
            if steps>len(env.customers)*4: break
        agent.epsilon = max(0.01, agent.epsilon*0.995)
    return agent.Q

def merge_q_tables(tables):
    merged = {}
    for table in tables:
        for state, action_dict in table.items():
            if state not in merged: merged[state] = action_dict.copy()
            else:
                for a,v in action_dict.items():
                    merged[state][a] = (merged[state].get(a,0)+v)/2.0
    return merged

def train_cluster_time(env, episodes=2000, n_processes=4):
    import multiprocessing as mp
    args_list = [(copy.deepcopy(env), episodes//n_processes) for _ in range(n_processes)]
    with mp.Pool(processes=n_processes) as pool:
        q_tables = pool.map(run_episode_dict, args_list)
    final_agent = DictQAgent(env)
    final_agent.Q = merge_q_tables(q_tables)
    return final_agent

# ---------------- Schedule Reconstruction ----------------
def reconstruct_schedule_from_agent(agent, env):
    unvisited = set(env.customers)
    cur_time = START_TIME_MIN
    depot = env.depot
    schedule = []
    loc = depot
    while unvisited:
        state = (loc, frozenset(set(env.customers)-unvisited), cur_time)
        feasible = [n for n in unvisited]
        if state in agent.Q and agent.Q[state]:
            next_node = max(agent.Q[state], key=agent.Q[state].get)
        else: next_node = random.choice(list(unvisited))
        travel = env.travel_minutes(loc,next_node)
        arrival = cur_time+travel
        finish = arrival + SERVICE_MIN
        schedule.append((next_node,arrival,arrival,finish))
        unvisited.remove(next_node)
        loc = next_node
        cur_time = finish
    return schedule

# ---------------- Main ----------------
# ---------------- Main ----------------
def main(input_file,n_clusters=7,episodes=2000,n_processes=4):
    print(f"[DEBUG] Parsing CVRP instance: {input_file}")
    coords, demands, depots, deliveries = parse_cvrp_instance(input_file)
    print(f"[DEBUG] Found {len(coords)} nodes, {len(deliveries)} deliveries, {len(depots)} depots")

    print(f"[DEBUG] Clustering into {n_clusters} clusters")
    clusters, cluster_centers = cluster_deliveries(coords, deliveries, n_clusters=n_clusters)
    for cid, nodes in clusters.items():
        print(f"[DEBUG] Cluster {cid}: {len(nodes)} nodes, center at {cluster_centers[cid]}")

    cluster_routes, cluster_schedules = {}, {}
    for cid, nodes in clusters.items():
        depot = min(depots,key=lambda d: math.hypot(coords[d][0]-cluster_centers[cid][0],coords[d][1]-cluster_centers[cid][1]))
        print(f"[DEBUG] Training Q-agent for cluster {cid}, depot {depot}")
        env = ClusterTimeEnv(nodes,depot,coords)
        agent = train_cluster_time(env,episodes=episodes,n_processes=n_processes)
        #agent.save(f"cluster_agent_{cid}.pkl")
        print(f"[DEBUG] Reconstructing schedule for cluster {cid}")
        schedule = reconstruct_schedule_from_agent(agent,env)
        cluster_schedules[cid] = schedule
        cluster_routes[cid] = nodes  # placeholder
        print(f"[DEBUG] Cluster {cid} schedule: {schedule}")

    # Save JSON with integer keys preserved
    print("[DEBUG] Saving cluster_data.json")
    save_data = {
        "coords": {int(k):v for k,v in coords.items()},
        "demands": {int(k):v for k,v in demands.items()},
        "clusters": {int(k):v for k,v in clusters.items()},
        "cluster_routes": {int(k):v for k,v in cluster_routes.items()},
        "cluster_schedules": {int(k):v for k,v in cluster_schedules.items()},
        "cluster_centers": {int(k):v for k,v in cluster_centers.items()},
        "depots": list(depots)
    }
    with open("cluster_data.json","w") as f: 
        json.dump(save_data,f,indent=2)
    print("[DEBUG] Finished saving cluster data")

    return coords,demands,depots,clusters,cluster_routes,cluster_centers,cluster_schedules


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',required=True)
    parser.add_argument('--clusters','-k',type=int,default=5)
    parser.add_argument('--episodes','-e',type=int,default=1000)
    parser.add_argument('--processes','-p',type=int,default=4)
    args = parser.parse_args()
    main(args.input,n_clusters=args.clusters,episodes=args.episodes,n_processes=args.processes)
