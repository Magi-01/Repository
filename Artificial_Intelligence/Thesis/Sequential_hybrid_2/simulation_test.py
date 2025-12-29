#!/usr/bin/env python3
"""
HybridSimulationGPU_Logged_Full.py

- Fully GPU-accelerated VRP simulation
- Monte Carlo adaptive rerouting (black-box MonteCarloDeliveryGPUBatch)
- CSP enforcement
- Logs simulation steps to JSON for post-processing
"""

import torch
import json
from collections import defaultdict
from pathlib import Path
from montecarlo import MonteCarloDeliveryGPUBatch  # assumed available

device = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME_MIN = 8 * 60
VEHICLE_SPEED = 20.0  # km/h

# ---------------- Load Cluster Data ----------------
def load_cluster_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    all_nodes = sorted(int(k) for k in data["coords"].keys())
    node2idx = {nid: i for i, nid in enumerate(all_nodes)}
    coords_list = [data["coords"][str(nid)] for nid in all_nodes]
    demands_list = [data["demands"][str(nid)] for nid in all_nodes]
    depots = [node2idx[int(d)] for d in data["depots"]]
    clusters = {int(k): [node2idx[n] for n in v] for k, v in data["clusters"].items()}
    cluster_routes = {int(k): [node2idx[n] for n in v] for k, v in data["cluster_routes"].items()}
    cluster_centers = [tuple(data["cluster_centers"][str(k)]) for k in sorted(data["cluster_centers"].keys())]
    return coords_list, demands_list, depots, clusters, cluster_routes, cluster_centers, node2idx

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    deg2rad = torch.pi / 180.0
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1) * deg2rad

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c  # km

# ---------------- Vehicle ----------------
class VehicleGPU:
    def __init__(self, vid, depot_idx, capacity):
        self.id = vid
        self.pos_idx = depot_idx
        self.capacity = capacity
        self.load = 0.0
        self.route = []
        self.next_idx = 0
        self.arrival_time = START_TIME_MIN

    def assign_route(self, route):
        self.route = route
        self.next_idx = 0
        self.load = 0.0

# ---------------- CSP Solver ----------------
class CPSSolverGPU:
    def __init__(self, vehicles):
        self.vehicles = vehicles

    def resolve_conflicts(self):
        node_to_vehicle = defaultdict(list)
        for v in self.vehicles:
            for node in v.route[v.next_idx:]:
                node_to_vehicle[node].append(v)
        for node, vs in node_to_vehicle.items():
            if len(vs) > 1:
                keeper = vs[0]
                for other in vs[1:]:
                    if node in other.route:
                        other.route.remove(node)

# ---------------- Simulation ----------------
class HybridSimulationGPU:
    def __init__(self, cluster_json, vehicle_capacity=100, k=20):
        coords_list, demands_list, depots, clusters, cluster_routes, cluster_centers, node2idx = load_cluster_data(cluster_json)
        self.N = len(coords_list)
        self.coords_tensor = torch.tensor(coords_list, dtype=torch.float32, device=device)
        self.demands_tensor = torch.tensor(demands_list, dtype=torch.float16, device=device).squeeze()
        self.depots = depots
        self.clusters = clusters
        self.cluster_routes = cluster_routes
        self.vehicles = []
        self.vehicle_capacity = vehicle_capacity
        self.time = START_TIME_MIN
        self.k = k
        self.node2idx = node2idx

        # Sparse neighbors
        self.neighbor_idx, self.neighbor_time = self.build_sparse_neighbors()

        # Monte Carlo batch solver (black-box)
        self.mc_solver = MonteCarloDeliveryGPUBatch(
            neighbor_idx=self.neighbor_idx,
            neighbor_time=self.neighbor_time,
            vehicle_capacity=self.vehicle_capacity,
            gamma=0.95,
            n_rollouts=16,
            max_depth=50,
            device=device
        )

        # Logs
        self.log = {"time": [], "vehicles": []}

    def build_sparse_neighbors(self):
        """
        Builds k-nearest neighbors for all nodes using Haversine distance.
        Returns:
            neighbor_idx: (N, k) int32 tensor of neighbor indices
            neighbor_time: (N, k) float16 tensor of travel times in minutes
        """
        N, k = self.N, self.k
        neighbor_idx = torch.zeros((N, k), dtype=torch.int32, device=device)
        neighbor_time = torch.zeros((N, k), dtype=torch.float16, device=device)

        batch_size = 2000  # can tune based on GPU memory
        R = 6371.0  # Earth radius in km

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_coords = self.coords_tensor[i:end]  # (B, 2)
            deg2rad = torch.pi / 180.0

            lat1 = batch_coords[:, None, 0] * deg2rad
            lon1 = batch_coords[:, None, 1] * deg2rad

            lat2 = self.coords_tensor[None, :, 0] * deg2rad
            lon2 = self.coords_tensor[None, :, 1] * deg2rad

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
            dist_km = 6371.0 * c

            travel_time_batch = dist_km / VEHICLE_SPEED * 60

            # k nearest neighbors (smallest travel times)
            vals, idxs = torch.topk(travel_time_batch, k=k, largest=False)
            neighbor_idx[i:end, :] = idxs
            neighbor_time[i:end, :] = vals.half()

        return neighbor_idx, neighbor_time

    def init_vehicles(self):
        for cid, cluster in self.clusters.items():
            depot_idx = min(
                self.depots,
                key=lambda d: ((self.coords_tensor[cluster[0], 0] - self.coords_tensor[d, 0]).item() ** 2 +
                               (self.coords_tensor[cluster[0], 1] - self.coords_tensor[d, 1]).item() ** 2)
            )
            v = VehicleGPU(cid, depot_idx, self.vehicle_capacity)
            print(f"Vehicle {cid} initialized: load={v.load}, capacity={v.capacity}")
            baseline_route = self.cluster_routes.get(cid, [])
            v.assign_route(baseline_route)
            self.vehicles.append(v)

        for v in self.vehicles:
            first_node = v.route[0] if v.route else None
            print(f"Vehicle {v.id} pos_idx: {v.pos_idx}, first route node: {first_node}, demand there: {self.demands_tensor[first_node].item() if first_node is not None else 'N/A'}")


    # Inside HybridSimulationGPU class

    def step(self, dt=1.0):
        for v in self.vehicles:
            if v.next_idx >= len(v.route):
                continue

            next_node_idx = v.route[v.next_idx]
            lat1, lon1 = self.coords_tensor[v.pos_idx]
            lat2, lon2 = self.coords_tensor[next_node_idx]
            travel_time = haversine(lat1, lon1, lat2, lon2) / VEHICLE_SPEED * 60

            if self.time + dt >= v.arrival_time + travel_time.item():
                demand_here = min(self.vehicle_capacity - v.load, self.demands_tensor[next_node_idx].item())
                if demand_here <= 0:
                    print(f"Vehicle {v.id} cannot serve node {next_node_idx}: load={v.load}, node demand={self.demands_tensor[next_node_idx].item()}")
                    continue
                v.load += demand_here
                self.demands_tensor[next_node_idx] -= demand_here

                print(f"Vehicle {v.id} moved to {next_node_idx}, served {demand_here:.1f}, "
                    f"load {v.load:.1f}/{v.capacity}, remaining demand {self.demands_tensor[next_node_idx].item():.1f}")

                if v.load >= self.vehicle_capacity:
                    self.trigger_adaptive_layer(v)

                v.pos_idx = next_node_idx
                v.arrival_time = self.time + travel_time.item()
                v.next_idx += 1

        # ----------- LOGGING -----------
        snapshot = []
        for v in self.vehicles:
            snapshot.append({
                "id": v.id,
                "pos_idx": v.pos_idx,
                "load": v.load,
                "next_idx": v.next_idx,
                "route_remaining": v.route[v.next_idx:]
            })
        self.log["time"].append(self.time)
        self.log["vehicles"].append(snapshot)


    def is_constraint_violated(self, vehicle, expected_arrival, node_idx):
        """
        Checks hard constraints: capacity, delivery window, other rules
        For simplicity, only checks if we exceed a hypothetical end-of-day (18*60)
        Can be extended with node-specific windows and traffic delays
        """
        # Example: end-of-day constraint
        if expected_arrival > 18 * 60:
            return True

        # Could add more checks here:
        # - delivery window violation
        # - dynamic congestion / traffic
        # - conflict with other vehicles (CSP already resolves)
        return False


    def trigger_adaptive_layer(self, vehicle, max_depth=20):
        """
        Efficient, vectorized adaptive layer for a single vehicle.
        - Stops if vehicle is full.
        - Only visits nodes with positive demand.
        - Avoids repeated nodes.
        - Uses GPU vectorization.
        """
        if vehicle.load >= vehicle.capacity:
            print(f"Vehicle {vehicle.id} is already full, skipping adaptive layer.")
            return

        pos_idx = vehicle.pos_idx
        new_route = []
        visited = set()
        remaining_capacity = vehicle.capacity - vehicle.load

        print(f"\n--- Adaptive layer triggered for Vehicle {vehicle.id} at node {pos_idx} ---")
        print(f"Current load: {vehicle.load}, Remaining total demand: {self.demands_tensor.sum().item():.1f}")

        for step in range(max_depth):
            pos_tensor = torch.tensor([pos_idx], device=self.demands_tensor.device, dtype=torch.int32)

            # Get best action using MC batch solver
            best_action_tensor = self.mc_solver.best_action_batch(pos_tensor, self.demands_tensor, current_time=self.time)
            next_node = best_action_tensor[0].item()

            # Stop if no feasible next node
            if next_node is None or next_node in visited or self.demands_tensor[next_node] <= 0:
                break

            # Compute served amount
            served = min(remaining_capacity, self.demands_tensor[next_node].item())
            remaining_capacity -= served
            self.demands_tensor[next_node] -= served

            # Append to new route and mark visited
            new_route.append(next_node)
            visited.add(next_node)
            pos_idx = next_node

            # Print step summary (aggregate)
            unserved = self.demands_tensor.sum().item()
            print(f"Step {step+1}: move to {next_node}, served {served}, remaining capacity {remaining_capacity}, total unserved {unserved}")

            self.log["time"].append(self.time + step*0.01)  # tiny offset so adaptive steps are sequential
            self.log["vehicles"].append([
                {
                    "id": vehicle.id,
                    "pos_idx": pos_idx,
                    "load": vehicle.load,
                    "next_idx": vehicle.next_idx,
                    "route_remaining": vehicle.route[vehicle.next_idx:]
                } for vehicle in self.vehicles
            ])

            if remaining_capacity <= 0:
                break

        vehicle.route = new_route
        vehicle.next_idx = 0

        # Enforce CSP constraints across all vehicles
        CPSSolverGPU(self.vehicles).resolve_conflicts()

        print(f"New adaptive route for Vehicle {vehicle.id}: {vehicle.route}")



    def run(self, max_time=18*60):
        self.init_vehicles()
        while self.time < max_time:
            self.step()
            self.time += 1.0

    def save_log(self, filename="simulation_log.json"):
        with open(filename, "w") as f:
            json.dump(self.log, f, indent=2)

# ---------------- Main ----------------
if __name__=="__main__":
    BASE_DIR = Path(__file__).parent
    sim = HybridSimulationGPU(cluster_json=BASE_DIR/"cluster_data_refactored.json", vehicle_capacity=100)
    sim.run(max_time=18*60)
    sim.save_log(BASE_DIR/"simulation_log.json")
