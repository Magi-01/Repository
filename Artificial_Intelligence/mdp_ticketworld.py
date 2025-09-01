# mdp_ticketworld.py

import itertools
from collections import defaultdict
from calculations import (
    manhattan, is_valid_position, next_position,
    detect_head_on_collision, penalty_for_idling,
    all_horns_active,
)

MAX_SPEED = 5
SOFT_DEADLINE = 20

MOVE_DIRS = [(1,0), (-1,0), (0,1), (0,-1)]
SPEED_ACTIONS = ["accelerate", "brake", "idle", "horn"]

def all_actions():
    return SPEED_ACTIONS + MOVE_DIRS

class TicketMDP:
    def __init__(self, grid_size, obstacles, tickets, deliveries, num_trains):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.deliveries = dict(deliveries)
        self.num_trains = num_trains
        self.tickets = dict(tickets)

    def initial_state(self, trains):
        train_states = []
        for t in trains:
            pos = t["pos"]
            speed = t["speed"]
            collected = None
            deadline = None
            train_states.append( (pos, speed, collected, deadline) )
        
        tickets_left = frozenset(self.tickets.keys())
        idle_counters = tuple(0 for _ in range(self.num_trains))
        print(f"[DEBUG] Initial state constructed: {train_states} with tickets {tickets_left}")
        return (tuple(train_states), tickets_left, idle_counters)


    def is_terminal(self, state):
        trains_state, tickets_left, _ = state
        positions = [ts[0] for ts in trains_state]

        # Collision terminal condition (disabled for now)
        # if len(set(positions)) < len(positions):
        #     print("[DEBUG] Terminal state due to collision detected.")
        #     return True

        collected = [ts[2] for ts in trains_state]
        if len(tickets_left) == 0 and all(c is None for c in collected):
            print("[DEBUG] Terminal state due to all tickets delivered.")
            return True

        return False


    def actions_for_train(self, train_state):
        pos, speed, collected, deadline = train_state
        actions = []
        if speed < MAX_SPEED:
            actions.append("accelerate")
        if speed > 0:
            actions.append("brake")
        actions.append("idle")
        actions.append("horn")
        if speed > 0:
            for dx, dy in MOVE_DIRS:
                nx, ny = pos[0] + dx, pos[1] + dy
                if is_valid_position((nx, ny), self.grid_size, self.obstacles):
                    actions.append((dx, dy))
        print(f"[DEBUG] Possible actions for train at {pos} with speed {speed}: {actions}")
        return actions

        return actions

    def all_joint_actions(self, state):
        trains_state, tickets_left, _ = state
        train_actions = [self.actions_for_train(ts) for ts in trains_state]
        return itertools.product(*train_actions)

    def step_train(self, train_state, action):
        print(f"[DEBUG][Train step] Received state: {train_state}")
        pos, speed, collected, deadline = train_state
        new_speed = speed
        new_pos = pos
        new_collected = collected
        new_deadline = deadline

        if action == "accelerate":
            new_speed = min(speed + 1, MAX_SPEED)
        elif action == "brake":
            new_speed = max(speed - 1, 0)
        elif action == "idle":
            pass
        elif action == "horn":
            pass
        elif isinstance(action, tuple):
            if speed == 0:
                print(f"[DEBUG] Train at {pos} tried to move with speed 0, ignoring move action.")
                return train_state
            dx, dy = action
            for step in range(speed):
                tx, ty = new_pos[0] + dx, new_pos[1] + dy
                if is_valid_position((tx, ty), self.grid_size, self.obstacles):
                    new_pos = (tx, ty)
                else:
                    print(f"[DEBUG] Movement blocked at {(tx, ty)} for train starting at {pos}")
                    break
        else:
            print(f"[DEBUG] Unknown action {action} for train at {pos}")

        if new_deadline is not None:
            new_deadline = max(new_deadline - 1, -1)

        new_state = (new_pos, new_speed, new_collected, new_deadline)
        print(f"[DEBUG] Train step: from {train_state} with action {action} to {new_state}")
        return new_state

    def step(self, state, joint_action):
        trains_state, tickets_left, idle_counters = state
        new_trains_state = []
        new_idle_counters = list(idle_counters)  # copy to update

        reward = 0
        print(f"[DEBUG][step] Received state: {state}")
        print(f"[DEBUG] Joint action: {joint_action}")

        for i, (ts, action) in enumerate(zip(trains_state, joint_action)):
            pos, speed, collected, deadline = ts
            new_speed = speed
            new_pos = pos
            new_collected = collected
            new_deadline = deadline

            if action == "accelerate":
                new_speed = min(speed + 1, MAX_SPEED)
                new_idle_counters[i] = 0  # reset idle counter on move
            elif action == "brake":
                new_speed = max(speed - 1, 0)
                new_idle_counters[i] = 0
            elif action == "idle":
                new_idle_counters[i] += 1  # increase idle count
            elif action == "horn":
                new_idle_counters[i] = 0
            elif isinstance(action, tuple):
                if speed == 0:
                    print(f"[DEBUG] Train at {pos} tried to move with speed 0, ignoring move action.")
                else:
                    dx, dy = action
                    for _ in range(speed):
                        tx, ty = new_pos[0] + dx, new_pos[1] + dy
                        if is_valid_position((tx, ty), self.grid_size, self.obstacles):
                            new_pos = (tx, ty)
                        else:
                            print(f"[DEBUG] Movement blocked at {(tx, ty)} for train starting at {pos}")
                            break
                    new_idle_counters[i] = 0
            else:
                print(f"[DEBUG] Unknown action {action} for train at {pos}")

            if new_deadline is not None:
                new_deadline = max(new_deadline - 1, -1)

            new_trains_state.append((new_pos, new_speed, new_collected, new_deadline))

        # Check collisions
        positions = [ts[0] for ts in new_trains_state]
        if len(set(positions)) < len(positions):
            reward -= 1000
            print("[DEBUG] Collision detected after move.")
            next_state = (tuple(new_trains_state), tickets_left, tuple(new_idle_counters))
            return next_state, reward, True

        # Head-on collisions
        for i in range(self.num_trains):
            for j in range(i + 1, self.num_trains):
                old_pos1 = trains_state[i][0]
                old_pos2 = trains_state[j][0]
                dir1 = (new_trains_state[i][0][0] - old_pos1[0], new_trains_state[i][0][1] - old_pos1[1])
                dir2 = (new_trains_state[j][0][0] - old_pos2[0], new_trains_state[j][0][1] - old_pos2[1])
                if detect_head_on_collision(
                    {"pos": old_pos1, "dir": dir1, "speed": trains_state[i][1]},
                    {"pos": old_pos2, "dir": dir2, "speed": trains_state[j][1]},
                    self.grid_size, self.obstacles):
                    reward -= 1000
                    print(f"[DEBUG] Head-on collision detected between train {i} and {j}.")
                    next_state = (tuple(new_trains_state), tickets_left, tuple(new_idle_counters))
                    return next_state, reward, True

        new_tickets_left = set(tickets_left)
        updated_trains_state = []
        for ts in new_trains_state:
            pos, speed, collected, deadline = ts
            if collected is None and pos in new_tickets_left:
                collected = self.deliveries.get(pos, None)
                deadline = SOFT_DEADLINE
                new_tickets_left.remove(pos)
                reward += 50
                print(f"[DEBUG] Ticket collected at {pos}, delivery to {collected}. Reward +50.")
            if collected is not None and pos == collected:
                collected = None
                deadline = None
                reward += 100
                print(f"[DEBUG] Ticket delivered at {pos}. Reward +100.")
            if deadline is not None and deadline < 0:
                reward -= 50
                print(f"[DEBUG] Deadline missed for train at {pos}. Penalty -50.")
            updated_trains_state.append((pos, speed, collected, deadline))

        # Idle penalty increasing exponentially with idle count
        for i, idle_count in enumerate(new_idle_counters):
            if idle_count > 0:
                penalty = min(2 ** idle_count, 100)  # cap penalty at 100 to avoid explosion
                reward -= penalty
                print(f"[DEBUG] Penalty for idling (count={idle_count}) on train {i} at {updated_trains_state[i][0]}: {penalty}")

        # Horn penalty if all blow horns
        horned = all(action == "horn" for action in joint_action)
        if horned:
            visibility_loss = sum(ts[1] for ts in updated_trains_state)
            horn_penalty = visibility_loss * 10
            reward -= horn_penalty
            print(f"[DEBUG] All horns blown. Visibility loss penalty: {horn_penalty}")

        next_state = (tuple(updated_trains_state), frozenset(new_tickets_left), tuple(new_idle_counters))
        done = self.is_terminal(next_state)

        print(f"[DEBUG] Step result - New State: {next_state}, Reward: {reward}")
        return next_state, reward, done


    def value_iteration(self, trains, gamma=0.9, epsilon=1e-3, max_iter=100):
        from collections import deque

        V = defaultdict(float)
        policy = {}

        init_state = self.initial_state(trains)

        # Initialize idle counters for each train to zero
        idle_counters = tuple(0 for _ in trains)

        # Construct initial state including idle counters
        start_state = (init_state[0], frozenset(self.tickets.keys()), idle_counters)

        states_to_expand = deque([start_state])
        expanded_states = set()

        iteration = 0
        print("[DEBUG] Starting Value Iteration...")
        while iteration < max_iter and states_to_expand:
            iteration += 1
            new_states_to_expand = deque()
            delta = 0

            while states_to_expand:
                state = states_to_expand.popleft()
                if state in expanded_states:
                    continue
                expanded_states.add(state)

                if self.is_terminal(state):
                    V[state] = 0
                    continue

                max_action_value = float("-inf")
                best_action = None

                for joint_action in self.all_joint_actions(state):
                    next_state, reward, done = self.step(state, joint_action)
                    value = reward + gamma * V[next_state]
                    if value > max_action_value:
                        max_action_value = value
                        best_action = joint_action
                    if next_state not in expanded_states:
                        new_states_to_expand.append(next_state)

                delta = max(delta, abs(V[state] - max_action_value))
                V[state] = max_action_value
                policy[state] = best_action

            states_to_expand = new_states_to_expand
            print(f"[DEBUG] Iteration {iteration}, delta={delta}, new states: {len(states_to_expand)}")
            if delta < epsilon:
                print(f"[DEBUG] Value iteration converged in {iteration} iterations.")
                break

        self.V = V
        self.policy = policy
        print("[DEBUG] Value iteration done.")

    def policy_for_train(self, state, train_idx):
        joint_action = self.policy.get(state, None)
        if joint_action is None:
            print(f"[DEBUG] No policy found for state, train {train_idx} defaults to idle.")
            return "idle"
        return joint_action[train_idx]
