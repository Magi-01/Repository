# mdp.py
import numpy as np
import random
from csp import neighbors, feasible_actions
from config import Config

class ValueHelper:
    def __init__(self):
        pass

    def next_value(self, V, P, R, gamma):
        """
        Compute next value for all states given policy/transition/rewards
        """
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
    def __init__(self):
        pass

    def next_policy(self, V, P, R, gamma):
        """
        Greedy policy improvement
        """
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

class ActionHelper:
    def __init__(self):
        pass

    def next_action(self, state, Q, epsilon):
        """
        Epsilon-greedy action selection
        """
        if random.random() < epsilon:
            return random.randint(0, Q.shape[2]-1)
        return np.argmax(Q[state[0], state[1], :])
    

class MDP:
    def __init__(self, states, actions=None, P=None, R=None, gamma=0.9, terminal_states=None):
        self.states = states
        self.actions = actions or ['up','down','left','right']
        self.gamma = gamma

        n_states = len(states)
        n_actions = len(self.actions)
        if P is None:
            self.P = np.zeros((n_states, n_actions, n_states), dtype=float)
        else:
            self.P = np.array(P, dtype=float)

        if R is None:
            self.R = np.zeros((n_states, n_actions, n_states), dtype=float)
        else:
            self.R = np.array(R, dtype=float)


        self.terminal_states = terminal_states or []
        # Q-table for Q-learning
        self.Q = np.zeros((len(states), len(self.actions)))
        self.value_helper = ValueHelper()
        self.policy_helper = PolicyHelper()
        self.action_helper = ActionHelper()
        self.policy = None
        self.V = None

    # -------------------
    # Value iteration
    # -------------------
    def value_iteration(self, l=1e-4):
        """
        Optimized vectorized value iteration
        """
        n_states = len(self.states)
        n_actions = len(self.actions)
        V = np.zeros(n_states)
        
        P = np.array(self.P)  # shape: [states, actions, states]
        R = np.array(self.R)
        terminal_mask = np.zeros(n_states, dtype=bool)
        for t in self.terminal_states:
            terminal_mask[t] = True
        
        while True:
            # Vectorized Q-values for all states and actions
            Q = np.sum(P * (R + self.gamma * V[None, None, :]), axis=2)
            V_new = np.max(Q, axis=1)
            V_new[terminal_mask] = 0  # Do not update terminal states
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
        """
        Optimized policy iteration using vectorized policy evaluation
        """
        n_states = len(self.states)
        n_actions = len(self.actions)
        policy = np.random.randint(n_actions, size=n_states)
        V = np.zeros(n_states)
        
        P = np.array(self.P)
        R = np.array(self.R)
        terminal_mask = np.zeros(n_states, dtype=bool)
        for t in self.terminal_states:
            terminal_mask[t] = True
        
        stable = False
        while not stable:
            # Policy evaluation (vectorized)
            while True:
                Q = np.zeros((n_states, n_actions))
                for a in range(n_actions):
                    Q[:,a] = np.sum(P[:,a,:] * (R[:,a,:] + self.gamma * V[None,:]), axis=1)
                V_new = Q[np.arange(n_states), policy]
                V_new[terminal_mask] = 0
                delta = np.max(np.abs(V_new - V))
                V = V_new
                if delta < 1e-4:
                    break
            
            # Policy improvement
            Q_full = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                Q_full[:,a] = np.sum(P[:,a,:] * (R[:,a,:] + self.gamma * V[None,:]), axis=1)
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
    # Q-learning
    # -------------------
    def learn_q(self, agent_pos, terminal_state=None, cop_pos=None, obstacles=set(),
            alpha=0.5, epsilon=0.2, single_step=False, episode=None, grid_size=10):
        """
        Perform a single step of Q-learning.
        Returns updated agent_pos, reward, done flag.
        """
        directions = self.actions
        dir_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        
        # Current state index (agent + terminal)
        state_idx = self.states.index((agent_pos, terminal_state))
        
        # Feasible actions
        feasible = feasible_actions(agent_pos, obstacles, cop_pos, terminal_state, grid_size)
        if not feasible:
            # Agent is trapped
            next_pos = agent_pos
            reward = -10
            done = False
            return next_pos, reward, done

        # Map feasible actions to indices
        feasible_idx = [directions.index(a) for a in feasible]

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            a_idx = random.choice(feasible_idx)
        else:
            q_vals = [self.Q[state_idx, i] for i in feasible_idx]
            max_val = max(q_vals)
            best_actions = [feasible_idx[i] for i, v in enumerate(q_vals) if v == max_val]
            a_idx = random.choice(best_actions)

        # Move
        dx, dy = dir_map[directions[a_idx]]
        next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

        # Boundaries and obstacles
        if next_pos[0] < 0 or next_pos[0] >= grid_size or next_pos[1] < 0 or next_pos[1] >= grid_size:
            next_pos = agent_pos
        if next_pos in obstacles or (cop_pos and next_pos == cop_pos):
            next_pos = agent_pos

        # Reward
        reward = Config.STEP
        done = False
        if terminal_state and next_pos == terminal_state:
            reward = Config.WIN
            done = True
        if cop_pos and next_pos == cop_pos:
            reward = Config.ARREST
            done = True

        # Q-learning update
        next_state_idx = self.states.index((next_pos, terminal_state))
        max_next = max([self.Q[next_state_idx, i] for i in range(len(directions))])
        self.Q[state_idx, a_idx] += alpha * (reward + self.gamma * max_next - self.Q[state_idx, a_idx])

        if single_step:
            return next_pos, reward, done
        return next_pos, reward, done
