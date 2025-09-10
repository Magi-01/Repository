from ortools.sat.python import cp_model

class DeliveryCSP:
    def __init__(self, orders, vehicles, travel_time, capacities, service_time, time_windows=None):
        """
        orders: list of order IDs
        vehicles: list of vehicle IDs
        travel_time: dict of (i,j) -> travel time
        capacities: dict of vehicle_id -> capacity
        service_time: dict of order_id -> service duration
        time_windows: dict of order_id -> (start, end), optional
        """
        self.orders = orders
        self.vehicles = vehicles
        self.travel_time = travel_time
        self.capacities = capacities
        self.service_time = service_time
        self.time_windows = time_windows if time_windows else {}

    # -------------------
    # Full CSP plan
    # -------------------
    def solve_plan(self):
        model = cp_model.CpModel()
        x = {}  # x[v,o] = 1 if vehicle v serves order o
        t = {}  # service start time for order o

        # Variables
        for v in self.vehicles:
            for o in self.orders:
                x[v,o] = model.NewBoolVar(f'x_{v}_{o}')
        for o in self.orders:
            t[o] = model.NewIntVar(0, 1000, f't_{o}')  # adjust horizon as needed

        # Each order assigned exactly once
        for o in self.orders:
            model.Add(sum(x[v,o] for v in self.vehicles) == 1)

        # Vehicle capacity
        for v in self.vehicles:
            model.Add(sum(x[v,o] for o in self.orders) <= self.capacities[v])

        # Time windows
        for o in self.orders:
            if o in self.time_windows:
                a,b = self.time_windows[o]
                model.Add(t[o] >= a)
                model.Add(t[o] <= b - self.service_time[o])

        # Temporal feasibility (sequence)
        for v in self.vehicles:
            assigned_orders = self.orders
            for i in range(len(assigned_orders)-1):
                o1 = assigned_orders[i]
                o2 = assigned_orders[i+1]
                # enforce if both assigned
                model.Add(t[o2] >= t[o1] + self.service_time[o1] + self.travel_time.get((o1,o2),0)).OnlyEnforceIf([x[v,o1], x[v,o2]])

        # Objective: weighted sum lateness + distance + vehicles used
        lateness = []
        for o in self.orders:
            deadline = self.time_windows.get(o,(0,1000))[1]
            lateness_var = model.NewIntVar(0,1000,f'lateness_{o}')
            model.Add(lateness_var >= t[o] + self.service_time[o] - deadline)
            lateness.append(lateness_var)

        total_lateness = sum(lateness)
        total_vehicles = sum([model.NewIntVar(0,1,f'used_{v}') for v in self.vehicles])
        total_distance = sum([self.travel_time.get((i,j),0) for i in self.orders for j in self.orders])

        wL, wD, wR = 10,1,5
        model.Minimize(wL*total_lateness + wD*total_distance + wR*total_vehicles)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)

        plan = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for v in self.vehicles:
                plan[v] = []
                for o in self.orders:
                    if solver.Value(x[v,o]):
                        plan[v].append((o, solver.Value(t[o])))
        return plan

    # -------------------
    # Partial replan
    # -------------------
    def replan(self, remaining_orders, current_time=0):
        self.orders = remaining_orders
        return self.solve_plan()

    # -------------------
    # Scenario-based replan
    # -------------------
    def replan_scenarios(self, remaining_orders, travel_time_samples, current_time=0):
        plans = []
        for tt_sample in travel_time_samples:
            self.travel_time = tt_sample
            plan = self.replan(remaining_orders, current_time)
            plans.append(plan)
        # pick plan minimizing total lateness
        best_plan = min(plans, key=lambda p: sum([t for v in p for o,t in p[v]]))
        return best_plan

    # -------------------
    # Feasibility check (fast, incremental)
    # -------------------
    def is_feasible(self, plan):
        """
        Check vehicle capacity and time windows quickly
        """
        for v, route in plan.items():
            load = 0
            prev_o = None
            prev_time = 0
            for o, t_start in route:
                load += 1  # could replace with order size
                if load > self.capacities[v]:
                    return False
                # time window
                if o in self.time_windows:
                    a,b = self.time_windows[o]
                    if not (a <= t_start <= b - self.service_time[o]):
                        return False
                # travel time feasibility
                if prev_o is not None:
                    tt = self.travel_time.get((prev_o,o),0)
                    if t_start < prev_time + self.service_time[prev_o] + tt:
                        return False
                prev_o = o
                prev_time = t_start
        return True
