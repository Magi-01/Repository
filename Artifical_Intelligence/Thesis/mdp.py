# mdp.py
import numpy as np
import random
from csp import feasible_actions
from config import Config
from utils import manhattan
import math

# -------------------
# Helpers for value/policy iteration
# -------------------
class ValueHelper:
    def next_value(self, V, P, R, gamma):
        V_new = np.copy(V)
        for s in range(len(V)):
            q_vals = []
            for a in range(P.shape[1]):
                q = 0
                for sp in range(P.shape[2]):
                    q += P[s,a,sp]*(R[s,a,sp] + gamma*V[sp])
                q_vals.append(q)
            V_new[s] = max(q_vals)
        return V_new

class PolicyHelper:
    def next_policy(self, V, P, R, gamma):
        policy = np.zeros(P.shape[0], dtype=int)
        for s in range(P.shape[0]):
            q_vals = []
            for a in range(P.shape[1]):
                q = 0
                for sp in range(P.shape[2]):
                    q += P[s,a,sp]*(R[s,a,sp] + gamma*V[sp])
                q_vals.append(q)
            policy[s] = np.argmax(q_vals)
        return policy

# -------------------
# Action helper for Q-learning
# -------------------
class ActionHelper:
    def __init__(self, mdp):
        self.mdp = mdp
        self.directions = mdp.actions
        self.dir_map = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}

    def next_state(self, agent_pos, action):
        dx, dy = self.dir_map[action]
        return (agent_pos[0] + dx, agent_pos[1] + dy)

# -------------------
# MDP class
# -------------------
def build_state_index(states):
    return {s: i for i, s in enumerate(states)}

class MDP:
    def __init__(self, states, actions=None, gamma=0.9, terminal_states=None):
        self.states = states
        self.actions = actions or ['up','down','left','right']
        self.gamma = gamma
        self.terminal_states = terminal_states or []

        # Q-table
        self.Q = np.zeros((len(states), len(self.actions)))

        # Helpers
        self.value_helper = ValueHelper()
        self.policy_helper = PolicyHelper()
        self.action_helper = ActionHelper(self)

        # State → index mapping
        self.state_to_idx = build_state_index(states)

        # For value/policy iteration
        self.V = None
        self.policy = None

    # -------------------
    # Value iteration
    # -------------------
    def value_iteration(self, l=1e-4):
        n_states = len(self.states)
        n_actions = len(self.actions)
        V = np.zeros(n_states)
        P = np.array(getattr(self, "P", np.zeros((n_states, n_actions, n_states))))
        R = np.array(getattr(self, "R", np.zeros((n_states, n_actions, n_states))))

        terminal_mask = np.zeros(n_states, dtype=bool)
        for t in self.terminal_states:
            terminal_mask[t] = True

        while True:
            Q = np.sum(P * (R + self.gamma * V[None,None,:]), axis=2)
            V_new = np.max(Q, axis=1)
            V_new[terminal_mask] = 0
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < l:
                break
        self.V = V
        return V

    # -------------------
    # Policy iteration
    # -------------------
    def policy_iteration(self):
        n_states = len(self.states)
        n_actions = len(self.actions)
        policy = np.random.randint(n_actions, size=n_states)
        V = np.zeros(n_states)

        P = np.array(getattr(self, "P", np.zeros((n_states, n_actions, n_states))))
        R = np.array(getattr(self, "R", np.zeros((n_states, n_actions, n_states))))
        terminal_mask = np.zeros(n_states, dtype=bool)
        for t in self.terminal_states:
            terminal_mask[t] = True

        stable = False
        while not stable:
            # Policy evaluation
            while True:
                Q = np.zeros((n_states, n_actions))
                for a in range(n_actions):
                    Q[:,a] = np.sum(P[:,a,:] * (R[:,a,:] + self.gamma*V[None,:]), axis=1)
                V_new = Q[np.arange(n_states), policy]
                V_new[terminal_mask] = 0
                delta = np.max(np.abs(V_new - V))
                V = V_new
                if delta < 1e-4:
                    break

            # Policy improvement
            Q_full = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                Q_full[:,a] = np.sum(P[:,a,:] * (R[:,a,:] + self.gamma*V[None,:]), axis=1)
            best_actions = np.argmax(Q_full, axis=1)
            stable = np.all(policy == best_actions)
            policy = best_actions

        self.policy = policy
        self.V = V
        return policy, V

    def optimal_policy(self):
        return self.policy

    def optimal_value(self):
        return self.V

    # -------------------
    # Reward shaping
    # -------------------
    def shaped_reward(self, next_pos, agent_pos, terminal_state, cop_pos, step_penalty):
        d_goal = manhattan(next_pos, terminal_state)
        d_cop = manhattan(next_pos, cop_pos)

        # Reward for being closer to goal
        r_goal = Config.GOAL_WEIGHT / (d_goal + 1)

        # Penalize being too close to cop
        if d_cop <= Config.SAFE_DIST:
            r_cop = -(Config.COP_WEIGHT / (d_cop + 1))
        else:
            r_cop = 0

        # Stronger step penalty proportional to steps taken
        r_step = -1.5  # per step
        r_step2 = -0.05 * step_penalty  # cumulative penalty, grows with steps

        # Terminal rewards
        if next_pos == terminal_state:
            return Config.WIN
        if next_pos == cop_pos:
            return Config.ARREST

        # Combine all
        return r_goal + r_cop + r_step + r_step2

    # -------------------
    # Q-learning step
    # -------------------
    def learn_q(self, agent_pos, terminal_state, cop_pos, obstacles=set(), alpha=0.1, epsilon=0.1, steps=0):
        # 1. Feasible actions
        feasible = feasible_actions(agent_pos, obstacles, cop_pos, terminal_state, Config.GRID_SIZE)
        if not feasible:
            next_pos = agent_pos
            reward = self.shaped_reward(next_pos, agent_pos, terminal_state, cop_pos, step_penalty=steps)
            done = next_pos == terminal_state or next_pos == cop_pos
            return next_pos, reward, done

        # 2. Epsilon-greedy
        state_idx = self.state_to_idx[(agent_pos, cop_pos)]
        if random.random() < epsilon:
            action = random.choice(feasible)
        else:
            q_vals = [self.Q[state_idx, self.actions.index(a)] for a in feasible]
            max_val = max(q_vals)
            best_actions = [a for i,a in enumerate(feasible) if q_vals[i]==max_val]
            action = random.choice(best_actions)

        # 3. Next position
        next_pos = self.action_helper.next_state(agent_pos, action)
        next_state = (next_pos, cop_pos)
        next_state_idx = self.state_to_idx[next_state]

        # 4. Reward shaping with step penalty = steps so far
        reward = self.shaped_reward(next_pos, agent_pos, terminal_state, cop_pos, step_penalty=steps)
        total_reward = 0
        total_reward += reward / (steps + 1)
        done = next_pos == terminal_state or next_pos == cop_pos

        # 5. Q-learning update
        self.Q[state_idx, self.actions.index(action)] += alpha * (
            reward + self.gamma * np.max(self.Q[next_state_idx]) - self.Q[state_idx, self.actions.index(action)]
        )

        return next_pos, reward, done
