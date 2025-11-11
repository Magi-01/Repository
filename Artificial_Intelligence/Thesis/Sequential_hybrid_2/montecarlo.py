#!/usr/bin/env python3
"""
Monte Carlo Tree Search / Rollout module for CVRP
- State: (current_node, remaining_demands_dict, current_time)
- Actions: move to feasible neighboring node
- Rewards: fulfill demand while minimizing travel and respecting vehicle capacity
"""

import random
import copy

class MonteCarloDelivery:
    def __init__(self, graph, initial_state, vehicle_capacity, gamma=0.95, n_rollouts=50, max_depth=50):
        """
        graph: dict[node] = list of (neighbor, travel_time)
        initial_state: (current_node, remaining_demands_dict, current_time)
        vehicle_capacity: max capacity per vehicle
        """
        self.graph = graph
        self.state = initial_state
        self.capacity = vehicle_capacity
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth

    def feasible_actions(self, state):
        loc, demands, t = state
        actions = []
        for neighbor, _ in self.graph.get(loc, []):
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
        return next_state

    def reward(self, state, action, next_state):
        loc, demands, t = state
        next_loc, next_demands, next_t = next_state
        served = demands.get(action, 0)
        travel = next_t - t
        return served * 10 - travel * 0.1

    def rollout(self, state):
        """Random rollout from state, discounted rewards"""
        total_reward = 0
        gamma = 1.0
        current_state = copy.deepcopy(state)
        for depth in range(self.max_depth):
            actions = self.feasible_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            next_state = self.transition(current_state, action)
            r = self.reward(current_state, action, next_state)
            total_reward += gamma * r
            gamma *= self.gamma
            current_state = next_state
        return total_reward

    def best_action(self):
        """Return the best action from initial state using multiple rollouts"""
        actions = self.feasible_actions(self.state)
        if not actions:
            return None
        action_values = {}
        for action in actions:
            total = 0
            for _ in range(self.n_rollouts):
                next_state = self.transition(self.state, action)
                r = self.reward(self.state, action, next_state)
                total += r + self.gamma * self.rollout(next_state)
            action_values[action] = total / self.n_rollouts
        best_action = max(action_values, key=action_values.get)
        return best_action

    def plan_route(self):
        """Generate a greedy route from initial state using rollouts"""
        route = [self.state[0]]
        state = copy.deepcopy(self.state)
        for _ in range(100):  # max steps
            action = self.best_action()
            if action is None:
                break
            route.append(action)
            state = self.transition(state, action)
            self.state = state
        return route
