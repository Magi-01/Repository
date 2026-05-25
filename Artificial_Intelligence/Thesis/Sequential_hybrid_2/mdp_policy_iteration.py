#!/usr/bin/env python3
"""
MDP Policy Iteration module
- State: (current_node, remaining_demands_dict, current_time)
- Actions: move to feasible neighboring node
- Rewards: fulfill demand while minimizing travel and respecting vehicle capacity
"""

import math
import copy

class MDPPolicyIteration:
    def __init__(self, graph, initial_state, vehicle_capacity, gamma=0.95, max_iterations=1000, epsilon=1e-3):
        """
        graph: dict[node] = list of (neighbor, travel_time)
        initial_state: (current_node, remaining_demands_dict, current_time)
        vehicle_capacity: max capacity of vehicle
        """
        self.graph = graph
        self.state = initial_state
        self.capacity = vehicle_capacity
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.V = {}         # state_key -> value
        self.policy = {}    # state_key -> action

    def feasible_actions(self, state):
        loc, demands, t = state
        actions = []
        for neighbor, travel_time in self.graph.get(loc, []):
            if demands.get(neighbor, 0) > 0:
                actions.append(neighbor)
        if not actions:
            actions = [n for n,_ in self.graph.get(loc, [])]  # allow moves even if demand=0
        return actions

    def transition(self, state, action):
        loc, demands, t = state
        travel_time = next((tt for n, tt in self.graph.get(loc, []) if n == action), 1)
        new_time = t + travel_time
        new_demands = demands.copy()
        if action in new_demands:
            load = min(self.capacity, new_demands[action])
            new_demands[action] -= load
            if new_demands[action] <= 0:
                del new_demands[action]
        next_state = (action, new_demands, new_time)
        return [(next_state, 1.0)]

    def reward(self, state, action, next_state):
        loc, demands, t = state
        next_loc, next_demands, next_t = next_state
        served = demands.get(action, 0)
        travel = next_t - t
        return served * 10 - travel * 0.1  # demand fulfilled vs travel cost

    def state_key(self, state):
        loc, demands, t = state
        return (loc, tuple(sorted(demands.items())), round(t))

    def policy_evaluation(self):
        for it in range(self.max_iterations):
            delta = 0
            V_new = {}
            for state_key in list(self.policy.keys()) + [self.state_key(self.state)]:
                loc, demands_items, t = state_key
                demands = dict(demands_items)
                state = (loc, demands, t)
                action = self.policy.get(state_key, None)
                if action is None:
                    action = self.feasible_actions(state)[0] if self.feasible_actions(state) else None
                    self.policy[state_key] = action
                if action is None:
                    V_new[state_key] = 0
                    continue
                val = 0
                for ns, prob in self.transition(state, action):
                    val += prob * (self.reward(state, action, ns) + self.gamma * self.V.get(self.state_key(ns), 0))
                V_new[state_key] = val
                delta = max(delta, abs(self.V.get(state_key, 0) - val))
            self.V = V_new
            if delta < self.epsilon:
                break

    def policy_improvement(self):
        policy_stable = True
        for state_key in list(self.V.keys()) + [self.state_key(self.state)]:
            loc, demands_items, t = state_key
            demands = dict(deb for deb in demands_items)
            state = (loc, demands, t)
            old_action = self.policy.get(state_key, None)
            best_val = -math.inf
            best_action = None
            for action in self.feasible_actions(state):
                val = 0
                for ns, prob in self.transition(state, action):
                    val += prob * (self.reward(state, action, ns) + self.gamma * self.V.get(self.state_key(ns), 0))
                if val > best_val:
                    best_val = val
                    best_action = action
            if best_action != old_action:
                policy_stable = False
            self.policy[state_key] = best_action
        return policy_stable

    def solve(self):
        """Run full policy iteration and return greedy route from initial state"""
        for it in range(self.max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                break

        # Construct route from initial state
        route = [self.state[0]]
        state = self.state
        for _ in range(100):
            key = self.state_key(state)
            action = self.policy.get(key, None)
            if action is None or not self.feasible_actions(state):
                break
            route.append(action)
            next_state = self.transition(state, action)[0][0]
            state = next_state
        return route
