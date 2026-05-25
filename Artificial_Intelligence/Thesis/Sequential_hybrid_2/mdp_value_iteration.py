#!/usr/bin/env python3
"""
mdp_value_iteration.py

Value Iteration for vehicle routing with demand-awareness.
- State: (current_node, remaining_nodes, load)
- Actions: visit feasible next node
- Rewards penalize distance, over-capacity, late delivery
"""

import math
import numpy as np

class MDPValueIteration:
    def __init__(self, nodes, depot, demands, vehicle_capacity, travel_time_matrix,
                 service_time=5, end_time=18*60, gamma=0.95, epsilon=1e-3):
        """
        nodes: list of delivery node_ids
        depot: starting node_id
        demands: dict {node_id: demand}
        vehicle_capacity: numeric
        travel_time_matrix: dict[node_a][node_b] = minutes
        """
        self.nodes = nodes
        self.depot = depot
        self.demands = demands
        self.capacity = vehicle_capacity
        self.D = travel_time_matrix
        self.service_time = service_time
        self.end_time = end_time
        self.gamma = gamma
        self.epsilon = epsilon

        # States: (current_node, frozenset(remaining_nodes), load)
        self.V = {}
        self.policy = {}

    def available_actions(self, state):
        loc, remaining, load = state
        feasible = []
        for n in remaining:
            if self.demands.get(n, 0) + load <= self.capacity:
                feasible.append(n)
        return feasible

    def reward(self, state, action):
        loc, remaining, load = state
        travel = self.D[loc][action]
        new_load = load + self.demands.get(action, 0)
        penalty = 0
        if new_load > self.capacity:
            penalty -= 100  # overcapacity
        return -travel + penalty

    def step(self, state, action):
        loc, remaining, load = state
        next_remaining = set(remaining)
        next_remaining.discard(action)
        next_load = load + self.demands.get(action, 0)
        next_state = (action, frozenset(next_remaining), next_load)
        return next_state

    def run_value_iteration(self, max_iter=1000):
        # Initialize V
        all_states = [(self.depot, frozenset(self.nodes), 0)]
        self.V = {all_states[0]: 0}
        iteration = 0
        while iteration < max_iter:
            delta = 0
            states_to_update = list(self.V.keys())
            for state in states_to_update:
                actions = self.available_actions(state)
                if not actions:
                    self.V[state] = 0
                    continue
                q_values = []
                for a in actions:
                    next_state = self.step(state, a)
                    v_next = self.V.get(next_state, 0)
                    q = self.reward(state, a) + self.gamma * v_next
                    q_values.append((q, a))
                max_q, best_action = max(q_values, key=lambda x: x[0])
                delta = max(delta, abs(self.V.get(state, 0) - max_q))
                self.V[state] = max_q
                self.policy[state] = best_action
            if delta < self.epsilon:
                break
            iteration += 1
        print(f"Value Iteration converged in {iteration} iterations")
        return self.V, self.policy

    def get_action(self, state):
        return self.policy.get(state, None)
