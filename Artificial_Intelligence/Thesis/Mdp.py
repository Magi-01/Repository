# mdp.py
class MDP:
    def __init__(self, states, actions, P, R):
        """
        states: list of all states
        actions: function mapping state -> feasible actions
        P: transition model P[s][a] -> list of (prob, next_state)
        R: reward function R[s][a]
        """
        self.states = states
        self.actions = actions
        self.P = P
        self.R = R

    def value_iteration(self, gamma=0.95, theta=1e-6):
        V = {s: 0 for s in self.states}

        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                feasible = self.actions(s)
                if not feasible:
                    V[s] = 0
                    self.P[s] = None
                    continue
                V[s] = max(
                    self.R[s][a] + gamma * sum(p * V[s_next] for p, s_next in self.P[s][a])
                    for a in feasible
                )
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # extract policy
        policy = {}
        for s in self.states:
            feasible = self.actions(s)
            if feasible:
                policy[s] = max(feasible, key=lambda a:
                    self.R[s][a] + gamma * sum(p * V[s_next] for p, s_next in self.P[s][a])
                )
            else:
                policy[s] = None
        return V, policy

# For testing purposes
if __name__ == '__main__':
    # Define states
      # B is terminal
    states = ['A', 'B']

    # Define actions per state
    def actions(s):
        if s == 'A':
            return ['deliver', 'wait']
        else:
            return []  # terminal
    
    # Transition model P[s][a] -> list of (prob, next_state)
    P = {
        'A': {
            'deliver': [(1.0, 'B')],
            'wait': [(1.0, 'A')]
        },
        'B': {}
    }
    
    # Rewards R[s][a]
    R = {
        'A': {
            'deliver': 10,  # successful delivery
            'wait': -1      # penalty for idling
        },
        'B': {}
    }

    mdp = MDP(states, actions, P, R,)
    
    
    V, policy = mdp.value_iteration(gamma=0.9)
    print("Optimal Values:", V)
    print("Optimal Policy:", policy)