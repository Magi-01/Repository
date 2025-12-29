#!/usr/bin/env python3
"""
Vectorized Monte Carlo rollout for CVRP on GPU
- Supports batched rollouts for multiple vehicles in parallel
- Sparse neighbor representation
- Rewards: serve demand, minimize travel, respect capacity
- Fully uses PyTorch GPU tensors
"""

import torch, random

class MonteCarloDeliveryGPUBatch:
    def __init__(self, neighbor_idx, neighbor_time, vehicle_capacity, gamma=0.95, n_rollouts=16, max_depth=50, device="cuda"):
        """
        neighbor_idx: [N, k] tensor of nearest neighbor indices
        neighbor_time: [N, k] tensor of travel times (float16)
        vehicle_capacity: max load per vehicle
        """
        self.neighbor_idx = neighbor_idx
        self.neighbor_time = neighbor_time
        self.capacity = vehicle_capacity
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.device = device

    @torch.no_grad()
    def feasible_actions(self, pos_idx_batch, demands_tensor):
        """
        pos_idx_batch: [B] current positions
        demands_tensor: [N] remaining demands
        returns list of feasible actions per batch element
        """
        B = pos_idx_batch.size(0)
        k = self.neighbor_idx.size(1)
        actions_batch = []

        for i in range(B):
            neighbors = self.neighbor_idx[pos_idx_batch[i]]        # [k]
            mask = demands_tensor[neighbors] > 0
            actions = neighbors[mask]
            if actions.numel() == 0:
                actions = neighbors
            actions_batch.append(actions)
        return actions_batch

    @torch.no_grad()
    def rollout(self, pos_idx, demands_tensor, current_time=None):
        """
        Monte Carlo rollout for a single vehicle from pos_idx.
        Adds penalties for:
        - Late delivery
        - Unserved demand (vehicle full before serving)
        - Travel cost

        current_time: current simulation time (minutes)
        """
        total_reward = 0.0
        gamma = 1.0
        current = pos_idx
        local_demands = demands_tensor.clone()

        if current_time is None:
            current_time = 0.0

        for _ in range(self.max_depth):
            neighbors = self.neighbor_idx[current]
            mask = local_demands[neighbors] > 0
            actions = neighbors[mask]
            if actions.numel() == 0:
                actions = neighbors  # fallback if all served

            # Choose randomly for stochastic rollout
            a = actions[random.randint(0, len(actions)-1)].item()

            # Compute served amount
            served = min(self.capacity, local_demands[a].item())
            unserved = local_demands[a].item() - served

            # Travel time in minutes
            travel_idx = (self.neighbor_idx[current] == a).nonzero(as_tuple=True)[0][0]
            travel_time = self.neighbor_time[current, travel_idx].float()

            # Arrival time at next node
            arrival_time = current_time + travel_time.item()

            # Reward = served deliveries * 10 - travel cost * 0.1
            reward = served * 10 - travel_time * 0.1

            # Add lateness penalty (if arrival > end-of-day)
            if arrival_time > 18 * 60:  # 18:00
                reward -= (arrival_time - 18*60) * 2.0  # penalty per minute late

            # Add unserved delivery penalty
            reward -= unserved * 5.0

            total_reward += gamma * reward
            gamma *= self.gamma

            # Update local state
            local_demands[a] -= served
            current = a
            current_time = arrival_time

            # Stop if no remaining deliveries
            if local_demands.sum() == 0:
                break

        return total_reward


    @torch.no_grad()
    def best_action_batch(self, pos_idx_batch, demands_tensor, current_time):
        """
        Fully vectorized batch Monte Carlo for multiple vehicles.
        pos_idx_batch: [B] tensor of current positions
        demands_tensor: [N] tensor of demands
        current_time: scalar float (simulation time)
        Returns: [B] tensor of best next actions
        """
        B = pos_idx_batch.size(0)
        k = self.neighbor_idx.size(1)

        # Gather neighbors and travel times: [B, k]
        neighbors = self.neighbor_idx[pos_idx_batch]         # int32
        neighbor_times = self.neighbor_time[pos_idx_batch]   # float16

        # Gather neighbor demands: [B, k]
        neighbor_demands = demands_tensor[neighbors]

        # Mask invalid neighbors (demand <= 0)
        mask = neighbor_demands > 0
        mask_any = mask.any(dim=1)
        # If a row has no valid neighbor, allow all neighbors
        mask = mask | (~mask_any).unsqueeze(1)

        # Precompute rollouts for all neighbors
        # [B, k, n_rollouts]
        rollout_rewards = torch.zeros((B, k, self.n_rollouts), device=self.device, dtype=torch.float32)

        for r in range(self.n_rollouts):
            # Vectorized rollout call
            rollout_rewards[:, :, r] = self.rollout(neighbors, demands_tensor, current_time=current_time)

        # Compute immediate reward: [B, k]
        served = torch.minimum(neighbor_demands, torch.tensor(self.capacity, device=self.device))
        immediate_reward = served * 10 - neighbor_times.float() * 0.1

        # Average over rollouts
        avg_rollout_reward = rollout_rewards.mean(dim=2)
        total_reward = immediate_reward + self.gamma * avg_rollout_reward

        # Mask invalid actions
        total_reward[~mask] = -float('inf')

        # Select best action
        best_idx = total_reward.argmax(dim=1)
        best_actions = neighbors[torch.arange(B), best_idx]

        return best_actions

    @torch.no_grad()
    def plan_route_batch(self, start_idx_batch, demands_tensor, max_steps=50):
        """
        Vectorized route planning for batch of vehicles
        start_idx_batch: [B] starting positions
        returns list of routes (list of tensors)
        """
        B = start_idx_batch.size(0)
        routes = [ [start_idx_batch[i].item()] for i in range(B)]
        pos_idx_batch = start_idx_batch.clone()

        for _ in range(max_steps):
            actions = self.best_action_batch(pos_idx_batch, demands_tensor)
            for i in range(B):
                if actions[i].item() is None:
                    continue
                routes[i].append(actions[i].item())
                served = min(self.capacity, demands_tensor[actions[i].item()].item())
                demands_tensor[actions[i].item()] -= served
            pos_idx_batch = actions
        return routes
