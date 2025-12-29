class Csps:
    def __init__(self, orders, vehicles, capacities, travel_time):
        self.orders = orders
        self.vehicles = vehicles
        self.capacities = capacities
        self.travel_time = travel_time

    # Solve initial plan
    def solve_plan(self):
        plan = {v: [] for v in self.vehicles}
        cap = {v: 0 for v in self.vehicles}
        for order in self.orders:
            for v in self.vehicles:
                if cap[v] < self.capacities[v]:
                    plan[v].append((order['id'], 0))  # scheduled_time placeholder
                    cap[v] += 1
                    break
        return plan

    def is_feasible(self, plan):
        """
        Check if the plan is feasible:
        - Each vehicle does not exceed its capacity
        - All orders are assigned to exactly one vehicle
        """
        # 1. Check capacities
        for v, orders in plan.items():
            if len(orders) > self.capacities.get(v, float('inf')):
                return False
        
        # 2. Check all orders are assigned exactly once
        assigned_orders = [o for route in plan.values() for o,_ in route]
        if set(assigned_orders) != set(self.orders):
            return False
        if len(assigned_orders) != len(set(assigned_orders)):
            return False  # duplicate assignments
        
        return True

    # Partial replan for remaining orders
    def replan(self, remaining_orders, vehicle_progress):
        plan = {v: [] for v in self.vehicles}
        cap = {v: 0 for v in self.vehicles}
        for o in remaining_orders:
            for v in self.vehicles:
                if cap[v] < self.capacities[v]:
                    plan[v].append((o, 0))
                    cap[v] += 1
                    break
        return plan