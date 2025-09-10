import random
from Mdp import MDP
from montecarlo import MonteCarlo
from Csp import DeliveryCSP


class DeliveryMDP:
    def __init__(self, orders=None):
        if orders is None:
            orders = ['Depot', 'D1', 'D2']

        self.locations = orders
        self.deliveries = [loc for loc in orders if loc != 'Depot']
        self.state = self.generate_states()
        self.action = lambda s: self.actions(s)

        # P and R will be populated per vehicle after CSP
        self.P = {}
        self.R = {}

    def generate_states(self):
        """Generate all states: (current_location, pending_orders)"""
        states = []
        n = len(self.deliveries)
        subsets = [[]]
        # generate all subsets of deliveries
        for i in range(1, 1 << n):
            subset = [self.deliveries[j] for j in range(n) if (i >> j) & 1]
            subsets.append(subset)
        for loc in self.locations:
            for subset in subsets:
                states.append((loc, tuple(sorted(subset))))
        return states

    def actions(self, state):
        loc, pending = state
        if not pending:
            return []  # terminal
        acts = []
        for d in pending:
            if loc != d:
                acts.append(f"go_{d}")
        if loc != 'Depot':
            acts.append("wait")
        return acts

    def transitions(self, state, action, assigned_orders):
        loc, pending = state
        pending = list(pending)
        if not pending:
            return [(1.0, state)]

        if action.startswith("go_"):
            target = action.split("_")[1]
            if target in pending and target in assigned_orders:
                next_pending = tuple(o for o in pending if o != target)
                return [(1.0, (target, next_pending))]
        elif action == "wait":
            return [(1.0, state)]

        return [(1.0, state)]

    def reward(self, state, action, next_state, assigned_orders):
        loc, pending = state
        next_loc, next_pending = next_state
        if not pending:
            return 0
        completed = len(pending) - len(next_pending)
        reward = 10 * completed
        if action == "wait" and completed == 0:
            reward -= 1
        return reward


def simulate_policy(env, policy, start_state, assigned_orders, max_steps=100):
    state = start_state
    total_reward = 0
    for _ in range(max_steps):
        acts = env.actions(state)
        if not acts:
            break
        action = policy.get(state)
        if not action:
            break
        transitions = env.P.get(state, {}).get(action, [])
        if not transitions:
            break
        next_state = random.choices(
            [ns for pr, ns in transitions],
            weights=[pr for pr, ns in transitions]
        )[0]
        total_reward += env.reward(state, action, next_state, assigned_orders)
        state = next_state
    return total_reward


if __name__ == "__main__":
    # --- CSP / Delivery Setup ---
    vehicles = ['V1', 'V2']
    orders = ['Depot', 'D1', 'D2', 'D3']
    demand = {'D1': 1, 'D2': 2, 'D3': 1}
    capacity = {'V1': 3, 'V2': 2}

    # Solve assignment CSP
    csp = DeliveryCSP(vehicles, ['D1', 'D2', 'D3'], demand, capacity)
    solution = csp.solve()
    csp.print_solution()

    # Build per-vehicle MDPs and compute policies
    delivery = DeliveryMDP(orders)
    assigned_orders_per_vehicle = {v: [] for v in vehicles}
    for (vehicle, order), assigned in solution.items():
        if assigned:  # if this order is assigned to this vehicle
            assigned_orders_per_vehicle[vehicle].append(order)

    for vehicle in vehicles:
        assigned_orders = assigned_orders_per_vehicle[vehicle]

        # Populate P and R for this vehicle
        delivery.P = {}
        delivery.R = {}
        for s in delivery.state:
            feasible = delivery.actions(s)
            delivery.P[s] = {}
            delivery.R[s] = {}
            if not feasible:
                continue
            for a in feasible:
                delivery.P[s][a] = delivery.transitions(s, a, assigned_orders)
                delivery.R[s][a] = sum(
                    pr * delivery.reward(s, a, ns, assigned_orders)
                    for pr, ns in delivery.P[s][a]
                )

        mdp = MDP(delivery.state, delivery.action, delivery.P, delivery.R)

        # Value iteration
        V, policy_vi = mdp.value_iteration(gamma=0.95)

        print(f"\nVehicle {vehicle} - Value Iteration Policy:")
        for s, a in sorted(policy_vi.items()):
            print(s, "->", a)

        # Monte Carlo
        mc = MonteCarlo(gamma=0.95)
        for s in delivery.state:
            mc.first_visit_mc(env=mdp, start_state=s, episodes=2000)

        print(f"\nVehicle {vehicle} - Monte Carlo Policy:")
        for s, a in sorted(mc.policy.items()):
            print(s, "->", a)

        # Simulation
        start_state = ('Depot', tuple(assigned_orders))
        vi_rewards = [simulate_policy(delivery, policy_vi, start_state, assigned_orders) for _ in range(100)]
        mc_rewards = [simulate_policy(delivery, mc.policy, start_state, assigned_orders) for _ in range(100)]

        print(f"\nVehicle {vehicle} - Average cumulative reward:")
        print("Value Iteration:", sum(vi_rewards)/len(vi_rewards))
        print("Monte Carlo   :", sum(mc_rewards)/len(mc_rewards))
