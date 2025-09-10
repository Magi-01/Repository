import random
from collections import defaultdict

# -----------------------------
# DeliveryMDP (handles local disruptions)
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

        # Process vehicles sequentially
        for v in self.vehicles:
            pos = new_state['vehicle_positions'][v]
            route = new_state['csp_plan'].get(v, [])

            if route:
                next_order, sched = route[0]

                # travel: current -> depot -> order
                travel = self.travel_time.get((pos['location'], 0), 0) + self.travel_time.get((0, next_order), 0)
                service = self.service_time.get(next_order, 0)

                # Update vehicle time
                pos['time'] += travel + service

                # Complete the order
                new_state['pending_orders'].discard(next_order)
                new_state['done_orders'].add(next_order)
                pos['location'] = next_order
                route.pop(0)

        # Update reward after state
        reward = self.reward(new_state, action)
        self.Q[(self.state_to_key(state), action)] = reward
        return new_state

    def reward(self, state, action):
        travel_cost = sum(self.travel_time.get((0, o), 0) for o in state['pending_orders'])
        reroute_penalty = 5 if 'reroute' in action or 'reassign' in action else 0
        return - (travel_cost + reroute_penalty)

    def state_to_key(self, state):
        return (frozenset(state['pending_orders']), frozenset(state['done_orders']),
                tuple((v, state['vehicle_positions'][v]['location']) for v in self.vehicles), state['time'])

    def safe_action(self, state):
        return 'continue'

    def top_k_actions(self, state, k=5):
        candidates = self.candidate_actions(state)
        scores = []
        for a in candidates:
            s = self.step(state, a)
            score = -len(s['pending_orders'])
            scores.append((score, a))
        scores.sort(reverse=True)
        return [a for _, a in scores[:k]]

    def select_best_action(self, state, csp=None):
        top_actions = self.top_k_actions(state, k=5)
        for action in top_actions:
            if csp and csp.is_feasible(state['csp_plan']):
                return action
        return self.safe_action(state)

    def detect_disruption(self, state, csp=None):
        # Simple disruption: some pending orders remain while feasible CSP exists
        if not state['pending_orders']:
            return False
        # If top actions cannot reduce pending orders, consider disruption
        for action in self.top_k_actions(state, k=5):
            next_state = self.step(state, action)
            if len(next_state['pending_orders']) < len(state['pending_orders']):
                return False