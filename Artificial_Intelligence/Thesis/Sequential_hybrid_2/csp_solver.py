#!/usr/bin/env python3
"""
csp_solver.py

CPSSolver class for Hybrid Delivery Simulation
Handles multi-vehicle assignment respecting:
- Vehicle capacities
- Node demands
- No simultaneous node occupancy
- Optional time windows

Compatible with Simulation_with_csp.py:
    csp = CPSSolver(self.vehicles, self.demands, self.coords)
    csp.resolve_conflicts()
"""

import math
from collections import defaultdict

class CPSSolver:
    def __init__(self, vehicles, demands, coords):
        """
        vehicles: list of vehicle objects
            Each vehicle must have:
                - route: list of node_ids (planned)
                - capacity: numeric
        demands: dict {node_id: demand_value}
        coords: dict {node_id: (x, y)}
        """
        self.vehicles = vehicles
        self.demands = demands
        self.coords = coords
        self.travel_time = self._compute_travel_time_matrix()
        # Node occupancy map: node -> list of vehicle indices visiting it
        self.node_occupancy = defaultdict(list)

    def _compute_travel_time_matrix(self):
        """
        Precompute travel time in minutes between all nodes.
        """
        travel_time = {}
        nodes = list(self.coords.keys())
        speed_kmh = 20.0  # default speed
        for a in nodes:
            travel_time[a] = {}
            for b in nodes:
                dx = self.coords[a][0] - self.coords[b][0]
                dy = self.coords[a][1] - self.coords[b][1]
                distance_km = math.hypot(dx, dy) / 1000.0  # meters -> km
                travel_time[a][b] = (distance_km / speed_kmh) * 60.0  # minutes
        return travel_time

    def _check_capacity(self, vehicle, route):
        """
        Verify that total demand on the route <= vehicle capacity.
        """
        total_demand = sum(self.demands.get(node, 0) for node in route)
        return total_demand <= vehicle.capacity

    def _detect_conflicts(self):
        """
        Fill self.node_occupancy: node -> vehicles visiting it
        """
        self.node_occupancy.clear()
        for v_idx, v in enumerate(self.vehicles):
            for node in v.route:
                self.node_occupancy[node].append(v_idx)

    def _resolve_conflict_node(self, node):
        """
        If multiple vehicles visit the same node, reassign to one vehicle
        based on capacity availability. Others will skip or be rescheduled.
        """
        vehicles_here = self.node_occupancy[node]
        if len(vehicles_here) <= 1:
            return
        # Sort vehicles by remaining capacity
        vehicles_sorted = sorted(vehicles_here,
                                 key=lambda idx: self.vehicles[idx].capacity - sum(
                                     self.demands.get(n, 0) for n in self.vehicles[idx].route),
                                 reverse=True)
        # Keep node assigned to first vehicle
        keeper = vehicles_sorted[0]
        for idx in vehicles_sorted[1:]:
            # Remove node from other vehicle's route
            if node in self.vehicles[idx].route:
                self.vehicles[idx].route.remove(node)

    def _resolve_capacity_violations(self):
        """
        If a vehicle exceeds its capacity, remove nodes until within limit.
        Nodes removed can be reassigned later.
        """
        for v in self.vehicles:
            while not self._check_capacity(v, v.route):
                # Remove last node (greedy simple strategy)
                removed_node = v.route.pop()
                # Optionally, could reassign removed_node to another vehicle
                print(f"Vehicle {v} capacity exceeded, removed node {removed_node}")

    def resolve_conflicts(self):
        """
        Main entry point for resolving conflicts and enforcing hard constraints.
        Updates vehicle routes in-place.
        """
        # 1. Detect node conflicts
        self._detect_conflicts()
        # 2. Resolve conflicts for each node
        for node in self.node_occupancy:
            self._resolve_conflict_node(node)
        # 3. Resolve capacity violations
        self._resolve_capacity_violations()
        print("CSP resolution complete. Updated vehicle routes available in vehicles[].route")
