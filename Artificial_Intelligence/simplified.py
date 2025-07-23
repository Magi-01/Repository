import numpy as np
import random
from collections import deque, defaultdict
import pygame
import sys

# Constants (some will be overwritten by user input)
DEFAULT_GRID_SIZE = (6, 6)
CELL_SIZE = 80
WINDOW_MARGIN = 100
ACTIONS = ['up', 'down', 'left', 'right', 'stay']
DISCOUNT = 0.9
THRESHOLD = 1e-4
NOISE = 0.2
COLLISION_PENALTY = -20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
RED = (200, 30, 30)
GREEN = (30, 200, 30)
BLUE = (30, 30, 200)
YELLOW = (230, 230, 30)
ORANGE = (255, 140, 0)
PURPLE = (160, 32, 240)
CYAN = (0, 255, 255)

AGENT_COLORS = [RED, GREEN, BLUE, ORANGE, PURPLE, CYAN]

def neighbors(pos, grid_size):
    x, y = pos
    candidates = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
    return [(i,j) for i,j in candidates if 0 <= i < grid_size[0] and 0 <= j < grid_size[1]]

def bfs(start, goal, obstacles, grid_size):
    queue = deque([start])
    visited = set([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for n in neighbors(current, grid_size):
            if n not in visited and n not in obstacles:
                visited.add(n)
                queue.append(n)
    return False

def generate_random_obstacles(grid_size, obstacle_ratio=0.15, reserved=set()):
    total_cells = grid_size[0]*grid_size[1]
    num_obstacles = int(total_cells * obstacle_ratio)
    all_cells = {(x,y) for x in range(grid_size[0]) for y in range(grid_size[1])} - reserved
    obstacles = set(random.sample(all_cells, num_obstacles))
    return obstacles

def is_valid_map(agents_data, obstacles, grid_size):
    # Check for each agent group that paths exist: start->ticket and ticket->goal
    for item in agents_data:
        if not bfs(item['start'], item['ticket'], obstacles, grid_size):
            return False
        if not bfs(item['ticket'], item['goal'], obstacles, grid_size):
            return False
    return True

def generate_agents_and_items(num_agents, grid_size, obstacles):
    empty_cells = {(x,y) for x in range(grid_size[0]) for y in range(grid_size[1])} - obstacles
    used = set()
    agents_data = []

    if num_agents == 1:
        while True:
            start = random.choice(list(empty_cells - used))
            ticket = random.choice(list(empty_cells - used - {start}))
            goal = random.choice(list(empty_cells - used - {start, ticket}))
            if bfs(start, ticket, obstacles, grid_size) and bfs(ticket, goal, obstacles, grid_size):
                agents_data.append({'agents': [0], 'start': start, 'ticket': ticket, 'goal': goal, 'required_agents': 1})
                break
    else:
        agent_starts = random.sample(list(empty_cells), num_agents)
        used.update(agent_starts)

        needed = [2]*(num_agents//2) + [1]*(num_agents - (num_agents//2))
        needed = needed[:num_agents]

        agent_idx = 0
        for req in needed:
            attempts = 0
            while True:
                attempts +=1
                if attempts > 1000:
                    raise RuntimeError("Failed to generate valid item positions")
                start = random.choice(list(empty_cells - used))
                ticket = random.choice(list(empty_cells - used - {start}))
                goal = random.choice(list(empty_cells - used - {start, ticket}))
                if bfs(start, ticket, obstacles, grid_size) and bfs(ticket, goal, obstacles, grid_size):
                    group = list(range(agent_idx, min(agent_idx+req, num_agents)))
                    agent_idx += req
                    agents_data.append({'agents': group, 'start': start, 'ticket': ticket, 'goal': goal, 'required_agents': req})
                    used.update({start, ticket, goal})
                    break
            if agent_idx >= num_agents:
                break

    return agents_data

class WarehouseMDP:
    def __init__(self, grid_size, obstacles, ticket_pos, goal_pos, noise=NOISE):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.ticket_pos = ticket_pos
        self.goal_pos = goal_pos
        self.noise = noise
        self.states = self._generate_states()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.V = np.zeros(len(self.states))
        self.policy = np.zeros(len(self.states), dtype=int)

    def _generate_states(self):
        states = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x,y) in self.obstacles:
                    continue
                for has_ticket in [False, True]:
                    states.append(((x,y), has_ticket))
        return states

    def is_valid_pos(self, pos):
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and pos not in self.obstacles

    def get_next_state(self, state, action):
        (x,y), has_ticket = state
        if action == 'up':
            nx, ny = x-1, y
        elif action == 'down':
            nx, ny = x+1, y
        elif action == 'left':
            nx, ny = x, y-1
        elif action == 'right':
            nx, ny = x, y+1
        else:
            nx, ny = x, y

        if not self.is_valid_pos((nx, ny)):
            nx, ny = x, y

        new_has_ticket = has_ticket
        if not has_ticket and (nx, ny) == self.ticket_pos:
            new_has_ticket = True

        return ((nx, ny), new_has_ticket)

    def reward(self, state, action, next_state):
        # Delivering goal reward
        if state[1] == True and next_state[0] == self.goal_pos:
            return 10
        # Default step cost
        return -1

    def value_iteration(self, gamma=DISCOUNT, threshold=THRESHOLD):
        iteration = 0
        while True:
            delta = 0
            for i, state in enumerate(self.states):
                if state[1] == True and state[0] == self.goal_pos:
                    continue

                max_value = -float('inf')
                best_action = None

                for a_idx, action in enumerate(ACTIONS):
                    expected_value = 0.0
                    for actual_a in ACTIONS:
                        p = 1 - self.noise if actual_a == action else self.noise / (len(ACTIONS) - 1)
                        next_state = self.get_next_state(state, actual_a)
                        r = self.reward(state, actual_a, next_state)
                        next_state_idx = self.state_to_idx[next_state]
                        expected_value += p * (r + gamma * self.V[next_state_idx])

                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a_idx

                delta = max(delta, abs(max_value - self.V[i]))
                self.V[i] = max_value
                self.policy[i] = best_action

            iteration += 1
            if delta < threshold:
                print(f"Value iteration converged after {iteration} iterations.")
                break

    def get_action(self, state):
        return ACTIONS[self.policy[self.state_to_idx[state]]]

def detect_collisions(next_positions, current_positions):
    pos_count = defaultdict(list)
    for i, pos in enumerate(next_positions):
        pos_count[pos].append(i)

    collision_agents = set()

    # Same cell collisions
    for pos, agents in pos_count.items():
        if len(agents) > 1:
            collision_agents.update(agents)

    # Swap collisions
    for i, pos in enumerate(next_positions):
        for j, curr_pos in enumerate(current_positions):
            if i != j and pos == current_positions[j] and next_positions[j] == current_positions[i]:
                collision_agents.update({i,j})

    return collision_agents

def agents_in_same_group(agent1, agent2, agents_data):
    for item in agents_data:
        if agent1 in item['agents'] and agent2 in item['agents']:
            return True
    return False

def draw_grid(screen, grid_size):
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

def draw_obstacles(screen, obstacles):
    for (x,y) in obstacles:
        rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GRAY, rect)

def draw_agents(screen, states, has_ticket):
    for i, (pos, has_t) in enumerate(states):
        x, y = pos
        center = (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2)
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        pygame.draw.circle(screen, color, center, CELL_SIZE//3)
        # Draw ticket status ring
        if has_ticket:
            pygame.draw.circle(screen, YELLOW, center, CELL_SIZE//4, 3)

def draw_items(screen, agents_data):
    for idx, item in enumerate(agents_data):
        # Draw ticket
        ticket_pos = item['ticket']
        rect_ticket = pygame.Rect(ticket_pos[1]*CELL_SIZE + CELL_SIZE//4,
                                  ticket_pos[0]*CELL_SIZE + CELL_SIZE//4,
                                  CELL_SIZE//2, CELL_SIZE//2)
        pygame.draw.rect(screen, ORANGE, rect_ticket)

        # Draw goal
        goal_pos = item['goal']
        rect_goal = pygame.Rect(goal_pos[1]*CELL_SIZE + CELL_SIZE//4,
                                goal_pos[0]*CELL_SIZE + CELL_SIZE//4,
                                CELL_SIZE//2, CELL_SIZE//2)
        pygame.draw.rect(screen, GREEN, rect_goal)

def simulate_agents(mdp_agents, agents_data, deadlines, grid_size, max_steps=100):
    pygame.init()
    window_size = (grid_size[1]*CELL_SIZE, grid_size[0]*CELL_SIZE + WINDOW_MARGIN)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Warehouse Multi-Agent MDP Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    num_agents = len(mdp_agents)
    states = []
    for i in range(num_agents):
        for item in agents_data:
            if i in item['agents']:
                states.append((item['start'], False))
                break

    steps_taken = [0]*num_agents
    delivered = [False]*len(agents_data)
    has_ticket = [False]*num_agents
    coordination_announced = set()
    total_rewards = [0]*num_agents

    for step in range(max_steps):
        # Event loop (quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BLACK)
        draw_grid(screen, grid_size)
        draw_obstacles(screen, mdp_agents[0].obstacles)
        draw_items(screen, agents_data)
        draw_agents(screen, states, has_ticket)

        # Text info
        info_text = f"Step {step}"
        text_surface = font.render(info_text, True, WHITE)
        screen.blit(text_surface, (10, grid_size[0]*CELL_SIZE + 10))

        # Rewards and agent info
        for i in range(num_agents):
            status = f"Agent {i}: Pos={states[i][0]}, HasTicket={has_ticket[i]}, Reward={total_rewards[i]}"
            text_surface = font.render(status, True, AGENT_COLORS[i % len(AGENT_COLORS)])
            screen.blit(text_surface, (10, grid_size[0]*CELL_SIZE + 30 + 20*i))

        pygame.display.flip()
        clock.tick(5)  # ~5 FPS

        all_done = True

        intended_actions = [None]*num_agents
        for i in range(num_agents):
            if delivered_agent(i, agents_data, delivered):
                continue
            all_done = False
            mdp = mdp_agents[i]
            state = states[i]
            intended_actions[i] = mdp.get_action(state)

        actual_actions = []
        for i, action in enumerate(intended_actions):
            if action is None:
                actual_actions.append('stay')
            else:
                if random.random() < (1 - NOISE):
                    actual_actions.append(action)
                else:
                    others = [a for a in ACTIONS if a != action]
                    actual_actions.append(random.choice(others))

        tentative_next_positions = []
        for i, (state, action) in enumerate(zip(states, actual_actions)):
            pos, _ = state
            nx, ny = pos
            if action == 'up':
                nx -= 1
            elif action == 'down':
                nx += 1
            elif action == 'left':
                ny -= 1
            elif action == 'right':
                ny += 1

            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and (nx, ny) not in mdp_agents[i].obstacles:
                tentative_next_positions.append((nx, ny))
            else:
                tentative_next_positions.append(pos)

        # Detect collisions *before* applying moves
        collision_agents = detect_collisions(tentative_next_positions, [s[0] for s in states])

        # Filter collision agents by group membership: 
        penalty_agents = set()
        for a in collision_agents:
            for b in collision_agents:
                if a != b:
                    if not agents_in_same_group(a, b, agents_data):
                        penalty_agents.add(a)
                        penalty_agents.add(b)

        if penalty_agents:
            print(f"Collision risk detected for agents {sorted(penalty_agents)}! Applying penalty.")

        # Communication trigger for multi-agent items
        for idx, item in enumerate(agents_data):
            if item['required_agents'] > 1 and idx not in coordination_announced:
                print(f"Agents {item['agents']} coordinating to carry multi-agent item {idx}.")
                coordination_announced.add(idx)

        # Update states and has_ticket
        for i in range(num_agents):
            pos_old, has_t = states[i]
            new_pos = tentative_next_positions[i]
            item_group = None
            for item in agents_data:
                if i in item['agents']:
                    item_group = item
                    break
            if not has_ticket[i] and new_pos == item_group['ticket']:
                has_ticket[i] = True
            states[i] = (new_pos, has_ticket[i])
            steps_taken[i] += 1

            reward = -1
            if has_t and new_pos == item_group['goal']:
                reward += 10
            if i in penalty_agents:
                reward += COLLISION_PENALTY
            total_rewards[i] += reward

        # Check deliveries
        for idx, item in enumerate(agents_data):
            group_agents = item['agents']
            if delivered[idx]:
                continue
            if all(has_ticket[a] for a in group_agents) and all(states[a][0] == item['goal'] for a in group_agents):
                delivered[idx] = True
                max_steps_taken = max(steps_taken[a] for a in group_agents)
                if max_steps_taken > deadlines[idx]:
                    print(f"Item group {idx} delivered LATE at step {max_steps_taken}! Penalty applies.")
                else:
                    print(f"Item group {idx} delivered ON TIME at step {max_steps_taken}!")

        if all_done and all(delivered):
            print("All item groups delivered!")
            pygame.time.wait(3000)
            break

def delivered_agent(agent_idx, agents_data, delivered):
    for idx, item in enumerate(agents_data):
        if agent_idx in item['agents'] and delivered[idx]:
            return True
    return False

def main():
    print("Warehouse Multi-Agent Simulation")
    grid_rows = int(input("Enter grid rows (min 4): "))
    grid_cols = int(input("Enter grid columns (min 4): "))
    if grid_rows < 4 or grid_cols < 4:
        print("Grid size too small. Must be at least 4x4.")
        return
    grid_size = (grid_rows, grid_cols)

    num_agents = int(input("Enter number of agents (1-4): "))
    if num_agents < 1 or num_agents > 4:
        print("Number of agents must be between 1 and 4.")
        return

    # We must reserve agent start, ticket and goal positions from obstacles
    # So we try random obstacles until paths exist
    max_attempts = 1000
    obstacles = set()
    agents_data = []
    for attempt in range(max_attempts):
        obstacles_candidate = generate_random_obstacles(grid_size, obstacle_ratio=0.15, reserved=set())
        agents_data_candidate = generate_agents_and_items(num_agents, grid_size, obstacles_candidate)
        if is_valid_map(agents_data_candidate, obstacles_candidate, grid_size):
            obstacles = obstacles_candidate
            agents_data = agents_data_candidate
            break
    else:
        print("Failed to generate a valid warehouse map after many attempts.")
        return

    deadlines = [random.randint(20, 50) for _ in agents_data]

    print(f"Obstacles at: {obstacles}")
    for i, item in enumerate(agents_data):
        print(f"Item group {i}: Agents={item['agents']}, Start={item['start']}, Pickup={item['ticket']}, Drop-off={item['goal']}, Deadline={deadlines[i]} steps")

    mdp_agents = []
    for i in range(num_agents):
        for item in agents_data:
            if i in item['agents']:
                mdp = WarehouseMDP(grid_size, obstacles, item['ticket'], item['goal'], noise=NOISE)
                mdp.value_iteration()
                mdp_agents.append(mdp)
                break

    global CELL_SIZE
    # Adjust cell size if grid bigger than 10 to keep window manageable
    max_dim = max(grid_size)
    if max_dim > 10:
        CELL_SIZE = max(20, 600 // max_dim)
    else:
        CELL_SIZE = 80

    simulate_agents(mdp_agents, agents_data, deadlines, grid_size, max_steps=200)

if __name__ == "__main__":
    main()
