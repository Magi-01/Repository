import random
from collections import defaultdict

class DeliveryMDP:
    def __init__(self, orders, vehicles, travel_time=None, service_time=None, capacities=None, gamma=0.95, rollout_steps=5, num_samples=5):
        self.orders = orders
        self.vehicles = vehicles
        self.travel_time = travel_time or {}
        self.service_time = service_time or {}
        self.capacities = capacities or {v: len(orders) for v in vehicles}
        self.gamma = gamma
        self.Q = defaultdict(float)
        self.rollout_steps = rollout_steps
        self.num_samples = num_samples

    def initial_state(self, plan=None):
        state = {
            'vehicle_positions': {v: {'location': 0, 'time': 0} for v in self.vehicles},
            'pending_orders': set(self.orders),
            'done_orders': set(),
            'time': 0,
            'csp_plan': {v: list(plan.get(v, [])) if plan else [] for v in self.vehicles}
        }
        return state

    def candidate_actions(self, state):
        actions = ['continue']
        for v in self.vehicles:
            actions.append(f'reroute_{v}')
        for o in state['pending_orders']:
            for v in self.vehicles:
                actions.append(f'reassign_{o}_to_{v}')
        return actions

    def step(self, state, action):
        # Copy state
        new_state = {
            'vehicle_positions': {v: dict(state['vehicle_positions'][v]) for v in self.vehicles},
            'pending_orders': set(state['pending_orders']),
            'done_orders': set(state['done_orders']),
            'time': state['time'] + 1,
            'csp_plan': {v: list(state['csp_plan'].get(v, [])) for v in self.vehicles}
        }

        for v in self.vehicles:
            pos = new_state['vehicle_positions'][v]
            route = new_state['csp_plan'].get(v, [])

            if route:
                next_order, _ = route[0]
                order = self.orders[next_order]
                travel = self.travel_time.get((pos['location'], next_order), 0)
                service = order['s']

                arrival = pos['time'] + travel
                if arrival < order['a']:
                    arrival = order['a']
                if arrival + service > order['b']:
                    continue  # cannot deliver on time

                pos['time'] = arrival + service
                pos['location'] = next_order
                new_state['pending_orders'].discard(next_order)
                new_state['done_orders'].add(next_order)
                route.pop(0)

        reward = self.reward(new_state, action)
        self.Q[(self.state_to_key(state), action)] = reward
        return new_state

    def reward(self, state, action):
        lateness = 0
        for o in state['done_orders']:
            order = self.orders[o]
            delivered_times = [v['time'] for v in state['vehicle_positions'].values() if v['location']==o]
            if delivered_times:
                t_delivered = min(delivered_times)
                lateness += max(0, t_delivered - order['b'])
        travel_cost = sum(self.travel_time.get((0,o),0) for o in state['pending_orders'])
        reroute_penalty = 5 if 'reroute' in action or 'reassign' in action else 0
        return - (lateness + travel_cost + reroute_penalty)

    def state_to_key(self, state):
        return (frozenset(state['pending_orders']),
                frozenset(state['done_orders']),
                tuple((v, state['vehicle_positions'][v]['location']) for v in self.vehicles),
                state['time'])

    def safe_action(self, state):
        return 'continue'

    # -----------------------------
    # Monte Carlo rollout
    # -----------------------------
    def simulate_future(self, state, steps=None):
        steps = steps or self.rollout_steps
        future = state
        for _ in range(steps):
            actions = self.candidate_actions(future)
            action = random.choice(actions)
            future = self.step(future, action)
        return future

    def evaluate_action(self, state, action):
        total_reward = 0
        for _ in range(self.num_samples):
            future_state = self.step(state, action)
            future_state = self.simulate_future(future_state, steps=self.rollout_steps)
            # Reward: negative of remaining pending orders + penalties
            r = self.reward(future_state, action)
            total_reward += r
        return total_reward / self.num_samples

    def select_best_action(self, state, csp=None):
        candidates = self.candidate_actions(state)
        best_action = self.safe_action(state)
        best_score = float('-inf')
        for a in candidates:
            if csp and not csp.is_feasible(state['csp_plan']):
                continue
            score = self.evaluate_action(state, a)
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def detect_disruption(self, state, csp=None):
        if not state['pending_orders']:
            return False
        for a in self.candidate_actions(state):
            next_state = self.step(state, a)
            if len(next_state['pending_orders']) < len(state['pending_orders']):
                return False
        return True

# -----------------------------
# HybridDelivery run method (time windows enforced)
# -----------------------------
class HybridDelivery:
    def __init__(self, orders, vehicles, travel_time, capacities, service_time):
        self.orders = orders
        self.vehicles = vehicles
        self.travel_time = travel_time
        self.capacities = capacities
        self.service_time = service_time

        self.csp_plan = {v: [(o,0) for o in list(orders.keys())[i::len(vehicles)]] 
                         for i,v in enumerate(vehicles)}
        self.mdp = DeliveryMDP(orders, vehicles, travel_time, service_time, capacities)
        self.state = self.mdp.initial_state(plan=self.csp_plan)
        self.completed_plan = {v: [] for v in vehicles}

    def is_feasible(self, plan):
        for v, route in plan.items():
            total_q = sum(self.orders[o]['q'] for o,_ in route)
            if total_q > self.capacities.get(v, float('inf')):
                return False
        assigned = [o for r in plan.values() for o,_ in r]
        if set(assigned) != set(self.orders) or len(assigned) != len(set(assigned)):
            return False
        return True

    def run(self, max_steps=100):
        history = []
        for t in range(max_steps):
            pending_orders = set(self.state['pending_orders'])
            print(f"Step {t} | Pending: {pending_orders}")

            # Detect disruptions due to time windows
            disruption = self.mdp.detect_disruption(self.state, csp=self)
            if disruption:
                action = self.mdp.select_best_action(self.state, csp=self)
                print(f"Disruption detected! MDP chooses action: {action}")
            else:
                action = 'continue'
                print("No disruption, following CSP plan.")

            self.state = self.mdp.step(self.state, action)

            for v in self.vehicles:
                route_done = set(o for o,_ in self.completed_plan[v])
                for o in self.state['done_orders']:
                    if o not in route_done and self.state['vehicle_positions'][v]['location'] == o:
                        self.completed_plan[v].append((o, self.state['vehicle_positions'][v]['time']))

            reward = self.mdp.Q.get((self.mdp.state_to_key(self.state), action), 0)
            history.append((set(self.state['pending_orders']), reward))

            if not self.state['pending_orders']:
                print("All orders completed.")
                break
        return history, self.state, self.completed_plan

# -----------------------------
# Example real-life scenario
# -----------------------------
if __name__ == "__main__":
    orders = {
        1: {'a':8, 'b':10, 's':0.5, 'q':1},
        2: {'a':9, 'b':11, 's':0.5, 'q':1},
        3: {'a':8.5, 'b':12, 's':0.5, 'q':2},
        4: {'a':10, 'b':13, 's':0.5, 'q':1},
    }
    vehicles = {'V1':3, 'V2':3}  # capacities

    travel_time = {
        (0,1):0.3, (0,2):0.5, (0,3):0.4, (0,4):0.6,
        (1,0):0.3, (2,0):0.5, (3,0):0.4, (4,0):0.6,
        (1,2):0.2, (1,3):0.3, (2,3):0.2, (3,4):0.4
    }
    service_time = {1:0.5,2:0.5,3:0.5,4:0.5}

    hybrid = HybridDelivery(orders, vehicles, travel_time, vehicles, service_time)
    history, final_state, completed_plan = hybrid.run(max_steps=50)

    print("\nFinal CSP Plan:")
    for v, route in hybrid.csp_plan.items():
        print(f"{v}: {[o for o,_ in route]}")

    print("\nCompleted Plan:")
    for v, route in completed_plan.items():
        print(f"{v}: {route}")

    print("\nStep-by-step Pending Orders and Rewards:")
    for step, (pending, reward) in enumerate(history):
        print(f"Step {step}: Pending = {pending}, Reward = {reward}")