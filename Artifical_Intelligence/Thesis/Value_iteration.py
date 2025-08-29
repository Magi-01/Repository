import numpy as np
import random

ACTIONS = ['up', 'down', 'left', 'right', 'idle']
P_SUCCESS = 0.8  # Probability intended move succeeds
GAMMA = 0.75      # Discount factor
THRESHOLD = 0.01 # Convergence threshold

def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def value_iteration(grid, agent, rewards, max_iter=100):
    """Compute optimal value function and policy for agent using Value Iteration."""
    states = [(x,y) for x in range(grid.width) for y in range(grid.height) if (x,y) not in grid.obstacles]
    V = {s: 0 for s in states}
    policy = {s: 'idle' for s in states}

    for _ in range(max_iter):
        delta = 0
        for s in states:
            best_value = -np.inf
            best_action = 'idle'
            for a in ACTIONS:
                # Compute expected value for this action
                if a == 'idle':
                    next_states = [(s, 1.0)]
                else:
                    dx, dy = 0, 0
                    if a == 'up': dy = -1
                    elif a == 'down': dy = 1
                    elif a == 'left': dx = -1
                    elif a == 'right': dx = 1
                    intended = (s[0]+dx, s[1]+dy)
                    if intended not in states:
                        intended = s
                    next_states = [(intended, P_SUCCESS), (s, 1-P_SUCCESS)]
                
                expected = 0
                for ns, prob in next_states:
                    reward = rewards.get(ns, 0)
                    expected += prob * (reward + GAMMA * V[ns])
                
                if expected > best_value:
                    best_value = expected
                    best_action = a
            delta = max(delta, abs(V[s]-best_value))
            V[s] = best_value
            policy[s] = best_action
        if delta < THRESHOLD:
            break
    return V, policy

def compute_rewards(grid, agent):
    rewards = {}
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) in grid.obstacles:
                continue
            reward = 0

            # Reward for delivering an order
            for order in agent.inventory:
                if order.dropoff == (x, y):
                    # Check if within deadline
                    if grid.time <= order.deadline:
                        reward += 10
                    else:
                        reward -= 20  # penalty for late delivery

            # Reward for picking up ready order
            for order in grid.orders:
                if order.ready and not order.picked_up and order.pickup == (x, y):
                    reward += 5

            # Idle penalty if no actions available
            reward -= 1

            rewards[(x, y)] = reward
    return rewards


def policy_iteration(grid, agent, rewards, max_iter=100):
    states = [(x, y) for x in range(grid.width) for y in range(grid.height) if (x, y) not in grid.obstacles]
    policy = {s: random.choice(ACTIONS) for s in states}
    V = {s: 0 for s in states}

    for _ in range(max_iter):
        # Policy Evaluation
        while True:
            delta = 0
            for s in states:
                a = policy[s]
                if a == 'idle':
                    next_states = [(s, 1.0)]
                else:
                    dx, dy = 0, 0
                    if a == 'up': dy = -1
                    elif a == 'down': dy = 1
                    elif a == 'left': dx = -1
                    elif a == 'right': dx = 1
                    intended = (s[0]+dx, s[1]+dy)
                    if intended not in states:
                        intended = s
                    next_states = [(intended, P_SUCCESS), (s, 1-P_SUCCESS)]
                v = 0
                for ns, prob in next_states:
                    v += prob * (rewards.get(ns, 0) + GAMMA * V[ns])
                delta = max(delta, abs(V[s]-v))
                V[s] = v
            if delta < THRESHOLD:
                break

        # Policy Improvement
        policy_stable = True
        for s in states:
            old_action = policy[s]
            best_action = old_action
            best_value = -np.inf
            for a in ACTIONS:
                if a == 'idle':
                    next_states = [(s, 1.0)]
                else:
                    dx, dy = 0, 0
                    if a == 'up': dy = -1
                    elif a == 'down': dy = 1
                    elif a == 'left': dx = -1
                    elif a == 'right': dx = 1
                    intended = (s[0]+dx, s[1]+dy)
                    if intended not in states:
                        intended = s
                    next_states = [(intended, P_SUCCESS), (s, 1-P_SUCCESS)]
                expected = sum(prob * (rewards.get(ns, 0) + GAMMA * V[ns]) for ns, prob in next_states)
                if expected > best_value:
                    best_value = expected
                    best_action = a
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        if policy_stable:
            break
    return V, policy