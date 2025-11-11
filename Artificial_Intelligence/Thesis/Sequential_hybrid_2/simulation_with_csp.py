#!/usr/bin/env python3
"""
Simulation_with_csp.py
- Hybrid delivery simulation using:
  * Precomputed baseline routes (A* paths)
  * Event-triggered MDP/Monte Carlo adaptation
  * Global CSP constraint enforcement
- Fully supports demand per customer.
"""

import os, json, math, random, heapq
from collections import defaultdict
from mdp_value_iteration import MDPValueIteration
from mdp_policy_iteration import MDPPolicyIteration
from montecarlo import MonteCarloDelivery
from csp_solver import CPSSolver

# ---------------- Load cluster data ----------------
def load_cluster_data(json_file="cluster_data.json"):
    with open(json_file,"r") as f:
        data = json.load(f)
    coords = {int(k):tuple(v) for k,v in data["coords"].items()}
    demands = {int(k):v for k,v in data["demands"].items()}
    depots = set(data["depots"])
    clusters = {int(k):v for k,v in data["clusters"].items()}
    cluster_routes = {int(k):v for k,v in data["cluster_routes"].items()}
    cluster_schedules = {int(k):v for k,v in data["cluster_schedules"].items()}
    cluster_centers = {int(k):tuple(v) for k,v in data["cluster_centers"].items()}
    return coords, demands, depots, clusters, cluster_routes, cluster_schedules, cluster_centers

# ---------------- Vehicle Class ----------------
class Vehicle:
    def __init__(self, vid, depot, capacity):
        self.id = vid
        self.pos = depot
        self.capacity = capacity
        self.load = 0
        self.route = []
        self.next_idx = 0
        self.arrival_time = 0
        self.events = []

    def assign_route(self, route, start_time=480):
        self.route = route
        self.next_idx = 0
        self.arrival_time = start_time

# ---------------- A* Pathfinding ----------------
def a_star_path(graph, start, goal, weight="distance"):
    # Simple Dijkstra/A* placeholder
    frontier = [(0,start)]
    came_from = {start:None}
    cost_so_far = {start:0}
    while frontier:
        cost, current = heapq.heappop(frontier)
        if current==goal: break
        for neighbor, w in graph.get(current,[]):
            new_cost = cost_so_far[current]+w
            if neighbor not in cost_so_far or new_cost<cost_so_far[neighbor]:
                cost_so_far[neighbor]=new_cost
                priority = new_cost
                heapq.heappush(frontier,(priority,neighbor))
                came_from[neighbor]=current
    # reconstruct path
    path=[]
    cur=goal
    while cur!=start:
        path.append(cur)
        cur=came_from[cur]
        if cur is None: break
    path.append(start)
    path.reverse()
    return path

# ---------------- Simulation ----------------
class HybridSimulation:
    def __init__(self, cluster_json="cluster_data.json", vehicle_capacity=100, adaptive_method="montecarlo"):
        self.coords, self.demands, self.depots, self.clusters, self.cluster_routes, self.cluster_schedules, self.cluster_centers = load_cluster_data(cluster_json)
        self.vehicles = []
        self.vehicle_capacity = vehicle_capacity
        self.graph = self.build_graph()
        self.time = 480  # 8:00 start in minutes
        self.adaptive_method = adaptive_method  # "value", "policy", "montecarlo"

    def build_graph(self):
        graph = {}
        nodes = list(self.coords.keys())
        for i in nodes:
            neighbors = []
            for j in nodes:
                if i==j: continue
                dist = math.hypot(self.coords[i][0]-self.coords[j][0], self.coords[i][1]-self.coords[j][1])
                travel_min = (dist/1000)/20*60  # km/h to minutes
                neighbors.append((j, travel_min))
            graph[i]=neighbors
        return graph

    def init_vehicles(self):
        for cid, cluster in self.clusters.items():
            depot = min(self.depots, key=lambda d: math.hypot(self.cluster_centers[cid][0]-self.coords[d][0],self.cluster_centers[cid][1]-self.coords[d][1]))
            v = Vehicle(cid,depot,self.vehicle_capacity)
            baseline_route = self.cluster_routes[cid]
            v.assign_route(baseline_route)
            self.vehicles.append(v)

    def step(self, dt=1):
        for v in self.vehicles:
            if v.next_idx>=len(v.route): continue
            next_node = v.route[v.next_idx]
            dist = math.hypot(self.coords[v.pos][0]-self.coords[next_node][0], self.coords[v.pos][1]-self.coords[next_node][1])
            travel_time = (dist/1000)/20*60
            if self.time+dt>=v.arrival_time+travel_time:
                # reached node
                demand_here = self.demands.get(next_node,0)
                if v.load+demand_here<=v.capacity:
                    v.load+=demand_here
                    self.demands[next_node]=0
                else:
                    # trigger reroute due to capacity violation
                    self.trigger_adaptive_layer(v,next_node)
                v.pos=next_node
                v.arrival_time=self.time+travel_time
                v.next_idx+=1

    def trigger_adaptive_layer(self, vehicle, node):
        state = (vehicle.pos, {n:d for n,d in self.demands.items() if d>0}, self.time)
        
        if self.adaptive_method=="value":
            solver = MDPValueIteration(self.graph, state, vehicle.capacity)
        elif self.adaptive_method=="policy":
            solver = MDPPolicyIteration(self.graph, state, vehicle.capacity)
        else:  # default montecarlo
            solver = MonteCarloDelivery(self.graph, state, vehicle.capacity)
        
        new_route = solver.solve()
        vehicle.route = new_route
        vehicle.next_idx = 0

        # CSP enforcement
        csp = CPSSolver(self.vehicles, self.demands, self.coords)
        csp.resolve_conflicts()

    def run(self, max_time=1080):
        self.init_vehicles()
        while self.time<max_time:
            self.step()
            self.time+=1

if __name__ == "__main__":
    # Example usage
    sim = HybridSimulation(cluster_json="cluster_data.json", vehicle_capacity=100)
    sim.run(max_time=1080)  # Run from 8:00 to 18:00 (in minutes)
    
    # After simulation, print summary
    for v in sim.vehicles:
        print(f"Vehicle {v.id} final position: {v.pos}, load: {v.load}")
