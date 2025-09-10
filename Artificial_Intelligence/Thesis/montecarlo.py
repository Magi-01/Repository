# montecarlo.py
import random

class MonteCarlo:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.returns = {}   # (state, action) -> list of returns
        self.Q = {}         # (state, action) -> value
        self.policy = {}    # state -> action

    def generate_episode(self, env, start_state, max_steps=100):
        episode = []
        state = start_state
        for t in range(max_steps):
            feasible = env.actions(state)
            if not feasible:
                break
            # ε-greedy
            if state not in self.policy or random.random() < 0.2:
                action = random.choice(feasible)
            else:
                action = self.policy[state]
            # sample next state + reward
            prob_next = env.P.get(state, {}).get(action, [])
            if not prob_next:
                break
            next_state = random.choices(
                [ns for pr, ns in prob_next],
                weights=[pr for pr, ns in prob_next]
            )[0]
            reward = env.R[state][action]
            episode.append((state, action, reward))
            state = next_state
        return episode

    def first_visit_mc(self, env, start_state, episodes=5000):
        for _ in range(episodes):
            episode = self.generate_episode(env, start_state)
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns.setdefault((state, action), []).append(G)
                    self.Q[(state, action)] = sum(self.returns[(state, action)]) / len(self.returns[(state, action)])

                    # greedy policy improvement
                    feasible = env.actions(state)
                    if feasible:
                        # initialize missing Q-values
                        for a in feasible:
                            self.Q.setdefault((state, a), 0)
                        # prefer "go_X" actions if any
                        productive = [a for a in feasible if a.startswith("go_")]
                        if productive:
                            best_a = max(productive, key=lambda a: self.Q[(state, a)])
                        else:
                            best_a = max(feasible, key=lambda a: self.Q[(state, a)])
                        self.policy[state] = best_a
        return self.Q, self.policy

