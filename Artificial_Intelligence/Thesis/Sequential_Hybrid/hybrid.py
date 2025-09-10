import random
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt

# -----------------------------
# DeliveryMDP: handles local disruptions
# -----------------------------
class DeliveryMDP:
    def __init__(self, orders, vehicles, travel_time=None, service_time=None, capacities=None, gamma=0.95):
        self.orders = orders
        self.vehicles = vehicles
        self.travel_time = travel_time or {}
        self.service_time = service_time or {}
        self.capacities = capacities or {v: len(orders) for v in vehicles}
        self.gamma = gamma
        self.Q = defaultdict(float)

    def initial_state(self, plan=None):
        state = {
            'vehicle_positions': {
                v: {
                    'location': 0,          # depot at 0
                    'time': 0,
                    'remaining_travel': 0,
                    'next_order': None,
                    'service_remaining': 0
                } for v in self.vehicles
            },
            'pending_orders': set(self.orders),
            'done_orders': set(),
            'time': 0,
            'csp_plan': {v: list(plan.get(v, [])) if plan else [] for v in self.vehicles}
        }
        return state

    def step(self, state):
        new_state = deepcopy(state)
        new_state['time'] += 1

        for v in self.vehicles:
            pos = new_state['vehicle_positions'][v]
            route = new_state['csp_plan'][v]

            # traveling
            if pos['remaining_travel'] > 0:
                pos['remaining_travel'] -= 1
                continue

            # servicing
            if pos['service_remaining'] > 0:
                pos['service_remaining'] -= 1
                continue

            # finished service
            if pos['next_order'] is not None:
                order = pos['next_order']
                new_state['pending_orders'].discard(order)
                new_state['done_orders'].add(order)
                pos['location'] = order
                pos['next_order'] = None

            # assign next order
            if pos['next_order'] is None and route:
                next_order, _ = route.pop(0)
                pos['next_order'] = next_order
                pos['service_remaining'] = self.service_time.get(next_order, 1)
                pos['remaining_travel'] = self.travel_time.get((pos['location'], next_order), 0)

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

    def candidate_actions(self, state):
        actions = ['return']
        for v in self.vehicles:
            actions.append(f'reroute_{v}')
        for o in state['pending_orders']:
            for v in self.vehicles:
                actions.append(f'reassign_{o}_to_{v}')
        return actions

    def state_to_key(self, state):
        return (frozenset(state['pending_orders']),
                frozenset(state['done_orders']),
                tuple((v, state['vehicle_positions'][v]['location']) for v in self.vehicles),
                state['time'])

    def safe_action(self, state):
        return 'continue'

    def top_k_actions(self, state, k=5):
        candidates = self.candidate_actions(state)
        scores = []
        for a in candidates:
            s = self.step(state)
            score = -len(s['pending_orders'])
            scores.append((score, a))
        scores.sort(reverse=True)
        return [a for _, a in scores[:k]]

    def select_best_action(self, state, csp=None):
        top_actions = self.top_k_actions(state, k=10)
        for action in top_actions:
            if csp and csp.is_feasible(state['csp_plan']):
                return action
        return self.safe_action(state)

    def detect_disruption(self, state, csp=None):
        if not state['pending_orders']:
            return False
        for action in self.top_k_actions(state, k=10):
            next_state = self.step(state)
            if len(next_state['pending_orders']) < len(state['pending_orders']):
                return False
        return True

# -----------------------------
# Helper: make travel times symmetric
# -----------------------------
def make_symmetric(travel_time):
    new_tt = dict(travel_time)
    for (i,j), t in travel_time.items():
        if (j,i) not in new_tt:
            new_tt[(j,i)] = t
    return new_tt

# -----------------------------
# HybridDelivery: integrates CSP + MDP
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

    def log_vehicle_states(self, state):
        print("Vehicle states:")
        for v, pos in state['vehicle_positions'].items():
            cap = self.capacities[v]
            loc = pos['location']
            t   = round(pos['time'],2)
            print(f"  {v}: loc={loc}, time={t}, cap={cap}")

    def run(self, max_steps=100):
        history = []
        for step in range(max_steps):
            print(f"\nStep {step} | Pending: {self.state['pending_orders']}")
            self.log_vehicle_states(self.state)

            disruption = self.mdp.detect_disruption(self.state, csp=self)
            if disruption:
                action = self.mdp.select_best_action(self.state, csp=self) if hasattr(self.mdp, 'select_best_action') else self.mdp.safe_action(self.state)
                print(f"Disruption detected! MDP chooses action: {action}")
            else:
                action = 'continue'
                print("No disruption, following CSP plan.")

            self.state = self.mdp.step(self.state)

            for v in self.vehicles:
                route_done = set(o for o,_ in self.completed_plan[v])
                for o in self.state['done_orders']:
                    if o not in route_done and self.state['vehicle_positions'][v]['location'] == o:
                        self.completed_plan[v].append((o, self.state['vehicle_positions'][v]['time']))

            reward = self.mdp.Q.get((self.mdp.state_to_key(self.state), action),0)
            history.append((set(self.state['pending_orders']), reward))

            if not self.state['pending_orders']:
                print("All orders completed.")
                break

        return history, self.state, self.completed_plan

# -----------------------------
# Gantt plotting
# -----------------------------
def plot_gantt(delivery_log):
    fig, ax = plt.subplots(figsize=(12,6))
    cmap = plt.get_cmap("tab20")
    for i, (vehicle, deliveries) in enumerate(delivery_log.items()):
        for j, (order, start) in enumerate(deliveries):
            finish = start + 0.5
            ax.barh(vehicle, finish-start, left=start, color=cmap(i%20))
            ax.text(start + (finish-start)/2, i, f"O{order}", va='center', ha='center', fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Vehicles")
    ax.set_title("Delivery Schedule (Gantt Chart)")
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Generate 50 orders
    orders = {}
    num_orders = 50
    start_time = 8
    window_size = 4
    for i in range(1, num_orders+1):
        a = round(start_time + (i%10)*0.5,1)
        b = a + window_size
        s = 0.5
        q = random.randint(1,4)
        orders[i] = {'a':a, 'b':b, 's':s, 'q':q}

    vehicles = {'V1':25,'V2':25} # two vehicles with capacity

    # symmetric travel times for 10 nodes
    nodes = list(range(10))
    travel_time = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                travel_time[(i,j)] = round(random.uniform(0.2,1.0),2)
    travel_time = make_symmetric(travel_time)

    # service time
    service_time = {i:0.5 for i in range(1,num_orders+1)}

    hybrid = HybridDelivery(orders, vehicles, travel_time, vehicles, service_time)
    history, final_state, completed_plan = hybrid.run(max_steps=100)

    print("\nFinal CSP Plan:")
    for v, route in hybrid.csp_plan.items():
        print(f"{v}: {[o for o,_ in route]}")

    print("\nCompleted Plan:")
    for v, route in completed_plan.items():
        print(f"{v}: {route}")

    print("\nStep-by-step Pending Orders and Rewards:")
    for step, (pending, reward) in enumerate(history):
        print(f"Step {step}: Pending = {pending}, Reward = {reward}")

    # Plot Gantt chart
    plot_gantt(completed_plan)
