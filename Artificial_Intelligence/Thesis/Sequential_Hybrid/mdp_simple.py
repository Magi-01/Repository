import random
from collections import defaultdict
import numpy as np

class MDP:
    def __init__(self, actions_fn, transition_fn, reward_fn, gamma=0.9):
        """
        actions_fn: function mapping state -> iterable of feasible actions
        transition_fn: function (state, action) -> dict of {next_state: probability}
        reward_fn: function (state, action, next_state) -> reward (float)
        gamma: discount factor
        """
        self.actions_fn = actions_fn
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.gamma = gamma

    def value_iteration(self, initial_states, theta=1e-6, max_iter=2):
        """
        initial_states: iterable of starting states (expands dynamically from there)
        theta: convergence threshold
        max_iter: safety stop
        """
        V = {s: 0.0 for s in initial_states}
        frontier = set(initial_states)  # reachable states

        for _ in range(max_iter):
            delta = 0
            new_states = set()

            for s in list(frontier):
                actions = self.actions_fn(s)
                if not actions:
                    continue

                old_v = V[s]
                # Bellman update
                V[s] = max(
                    sum(
                        p * (self.reward_fn(s, a, s_next) + self.gamma * V.get(s_next, 0.0))
                        for s_next, p in self.transition_fn(s, a).items()
                    )
                    for a in actions
                )

                delta = max(delta, abs(old_v - V[s]))

                # Expand reachable states
                for a in actions:
                    new_states.update(self.transition_fn(s, a).keys())

            # Add newly discovered states
            for ns in new_states:
                if ns not in V:
                    V[ns] = 0.0
            frontier |= new_states

            if delta < theta:
                break

        # derive greedy policy
        policy = {}
        for s in V.keys():
            actions = self.actions_fn(s)
            if not actions:
                policy[s] = None
                continue
            policy[s] = max(
                actions,
                key=lambda a: sum(
                    p * (self.reward_fn(s, a, s_next) + self.gamma * V.get(s_next, 0.0))
                    for s_next, p in self.transition_fn(s, a).items()
                )
            )
        return V, policy

class QLearningAgent:
    def __init__(self, actions_fn, reward_fn, transition_fn, state_hash,
                 gamma=0.9, alpha=0.1, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.995):
        self.actions_fn = actions_fn
        self.reward_fn = reward_fn
        self.transition_fn = transition_fn
        self.state_hash = state_hash

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(float)

    def choose_action(self, state):
        feasible = self.actions_fn(state)
        if not feasible:
            return None
        h = self.state_hash(state)
        if random.random() < self.epsilon:
            return random.choice(feasible)
        return max(feasible, key=lambda a: self.Q[(h, a)])

    def update(self, s, a, r, s_next):
        h = self.state_hash(s)
        h_next = self.state_hash(s_next)
        feasible_next = self.actions_fn(s_next)
        best_next = max((self.Q[(h_next, a_next)] for a_next in feasible_next), default=0)
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[(h, a)]
        self.Q[(h, a)] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
