import pygame
import random
import numpy as np
from dataclasses import dataclass

# ----- Config -----
GRID_SIZE = 5
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 60
NUM_AGENTS = 2
DYNAMIC_POLICY_UPDATE_FREQ = 3
ORDER_SPAWN_FREQ = 5  # new order every 5 steps
P_SUCCESS = 0.8
GAMMA = 0.8
THRESHOLD = 0.01
ACTIONS = ['up','down','left','right','idle']

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)    # Pick-up
RED = (200, 0, 0)      # Drop-off
BLUE = (0, 0, 200)     # Agent
YELLOW = (255, 255, 0) # Ready order
AGENT_COLORS = [(0,0,200), (0,150,255)]

STEP_PENALTY = -1
PICKUP_REWARD = 5
DROPOFF_REWARD = 10

# ----- Classes -----
@dataclass
class Order:
    id: int
    pickup: tuple
    dropoff: tuple
    processing_time: int
    deadline: int
    ready: bool = False
    picked_up: bool = False
    delivered: bool = False

class GridEnvironment:
    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles else []
        self.pickups = []
        self.dropoffs = []
        self.orders = []
        self.time = 0

    def add_pickup(self, location):
        self.pickups.append(location)

    def add_dropoff(self, location):
        self.dropoffs.append(location)

    def spawn_order(self, order_id, processing_time, deadline):
        pickup = random.choice(self.pickups)
        dropoff = random.choice(self.dropoffs)
        order = Order(order_id, pickup, dropoff, processing_time, deadline)
        self.orders.append(order)

    def step(self):
        self.time += 1
        for order in self.orders:
            if not order.ready and self.time >= order.processing_time:
                order.ready = True

    def resolve_pickup_conflicts(self, agents):
        attempted = {}
        for agent in agents:
            if not agent.can_pickup():
                continue
            for order in self.orders:
                if order.ready and not order.picked_up and order.pickup == agent.pos:
                    if order.id not in attempted:
                        attempted[order.id] = []
                    attempted[order.id].append(agent)
        for order_id, agent_list in attempted.items():
            chosen_agent = min(agent_list, key=lambda a: a.id)
            if chosen_agent.can_pickup():
                chosen_agent.inventory.append(self.orders[order_id])
                self.orders[order_id].picked_up = True

class Agent:
    def __init__(self, agent_id, start_pos, color, capacity=1):
        self.id = agent_id
        self.pos = start_pos
        self.color = color
        self.capacity = capacity
        self.inventory = []

    def can_pickup(self):
        return len(self.inventory) < self.capacity

    def move_by_policy(self, policy, grid):
        if not policy:
            return
        action = policy.get(self.pos, 'idle')
        dx, dy = 0, 0
        if action=='up': dy=-1
        elif action=='down': dy=1
        elif action=='left': dx=-1
        elif action=='right': dx=1
        new_pos = (self.pos[0]+dx, self.pos[1]+dy)
        if 0<=new_pos[0]<grid.width and 0<=new_pos[1]<grid.height and new_pos not in grid.obstacles:
            self.pos = new_pos

    def dropoff_order(self, grid):
        delivered = []
        for order in self.inventory:
            if order.dropoff == self.pos:
                order.delivered = True
                delivered.append(order)
        for order in delivered:
            self.inventory.remove(order)
        return delivered

# ----- MDP Functions -----
def compute_rewards(grid, agent):
    rewards = {}
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) in grid.obstacles:
                continue
            reward = STEP_PENALTY

            if agent.inventory:  # carrying item → focus on delivery
                order = agent.inventory[0]
                if (x, y) == order.dropoff:
                    reward += DROPOFF_REWARD if grid.time <= order.deadline else -20
                else:
                    dist = abs(x - order.dropoff[0]) + abs(y - order.dropoff[1])
                    reward += 1/(dist+1)
            else:  # empty → focus on pick-ups
                for order in grid.orders:
                    if order.ready and not order.picked_up and agent.can_pickup():
                        if (x, y) == order.pickup:
                            reward += PICKUP_REWARD
                        else:
                            dist = abs(x - order.pickup[0]) + abs(y - order.pickup[1])
                            reward += 1/(dist+1)
            rewards[(x,y)] = reward
    return rewards

def value_iteration(grid, agent, rewards, max_iter=50):
    states = [(x,y) for x in range(grid.width) for y in range(grid.height) if (x,y) not in grid.obstacles]
    V = {s:0 for s in states}
    policy = {s:'idle' for s in states}
    for _ in range(max_iter):
        delta = 0
        for s in states:
            best_value = -np.inf
            best_action = 'idle'
            for a in ACTIONS:
                if a=='idle':
                    next_states=[(s,1.0)]
                else:
                    dx,dy=0,0
                    if a=='up': dy=-1
                    elif a=='down': dy=1
                    elif a=='left': dx=-1
                    elif a=='right': dx=1
                    intended=(s[0]+dx,s[1]+dy)
                    if intended not in states: intended=s
                    next_states=[(intended,P_SUCCESS),(s,1-P_SUCCESS)]
                expected=sum(prob*(rewards.get(ns,0)+GAMMA*V[ns]) for ns,prob in next_states)
                if expected>best_value:
                    best_value=expected
                    best_action=a
            delta=max(delta, abs(V[s]-best_value))
            V[s]=best_value
            policy[s]=best_action
        if delta<THRESHOLD:
            break
    return V, policy

# ----- Metrics -----
def init_metrics(agents):
    return {agent.id:{'reward':0,'deliveries':0,'missed':0,'steps':0} for agent in agents}

# ----- Visualization helpers -----
pygame.font.init()
font = pygame.font.SysFont('Arial', 20)

def draw_value_function(screen, V, grid, font):
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) in grid.obstacles:
                continue
            v = V.get((x,y),0)
            text = font.render(f"{v:.1f}", True, BLACK)
            text_rect = text.get_rect(center=(x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2))
            screen.blit(text, text_rect)

def draw_policy_arrows(screen, policy, grid):
    for x in range(grid.width):
        for y in range(grid.height):
            if (x,y) in grid.obstacles:
                continue
            action = policy.get((x,y),'idle')
            cx, cy = x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2
            if action == 'up':
                pygame.draw.line(screen, BLACK, (cx, cy), (cx, cy-15), 2)
                pygame.draw.polygon(screen, BLACK, [(cx-5,cy-10),(cx+5,cy-10),(cx,cy-15)])
            elif action == 'down':
                pygame.draw.line(screen, BLACK, (cx, cy), (cx, cy+15), 2)
                pygame.draw.polygon(screen, BLACK, [(cx-5,cy+10),(cx+5,cy+10),(cx,cy+15)])
            elif action == 'left':
                pygame.draw.line(screen, BLACK, (cx, cy), (cx-15, cy), 2)
                pygame.draw.polygon(screen, BLACK, [(cx-10,cy-5),(cx-10,cy+5),(cx-15,cy)])
            elif action == 'right':
                pygame.draw.line(screen, BLACK, (cx, cy), (cx+15, cy), 2)
                pygame.draw.polygon(screen, BLACK, [(cx+10,cy-5),(cx+10,cy+5),(cx+15,cy)])

# ----- Initialize Pygame -----
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Delivery Simulation Hybrid CSP-MDP")

# ----- Initialize Grid and Agents -----
grid = GridEnvironment(GRID_SIZE, GRID_SIZE, obstacles=[(2,2)])
grid.add_pickup((0,0))
grid.add_pickup((4,0))
grid.add_dropoff((0,4))
grid.add_dropoff((4,4))
agents = [Agent(i, start_pos=(0,i), color=AGENT_COLORS[i%len(AGENT_COLORS)], capacity=1) for i in range(NUM_AGENTS)]
for i in range(4):
    grid.spawn_order(i, processing_time=random.randint(1,3), deadline=10)

metrics = init_metrics(agents)
algorithm = "PI"  # Value Iteration
agent_policies = {}
next_order_id = len(grid.orders)
time_counter = 0

# ----- Initialize policies -----
for agent in agents:
    rewards = compute_rewards(grid, agent)
    V, policy = value_iteration(grid, agent, rewards)
    agent_policies[agent.id] = policy

# ----- Main loop -----
running = True
while running:
    clock.tick(FPS)
    grid.step()
    time_counter += 1

    # Dynamic order spawning
    if time_counter % ORDER_SPAWN_FREQ == 0:
        grid.spawn_order(next_order_id, processing_time=grid.time+2, deadline=grid.time+10)
        next_order_id += 1

    # Update policies every few steps
    if time_counter % DYNAMIC_POLICY_UPDATE_FREQ == 0:
        for agent in agents:
            rewards = compute_rewards(grid, agent)
            V, policy = value_iteration(grid, agent, rewards)
            agent_policies[agent.id] = policy

    # ---- Move agents ----
    for agent in agents:
        policy = agent_policies.get(agent.id, {})
        if policy:
            agent.move_by_policy(policy, grid)
            metrics[agent.id]['steps'] += 1

    # ---- Resolve pick-up conflicts ----
    grid.resolve_pickup_conflicts(agents)

    # ---- Drop-off actions and metrics ----
    for agent in agents:
        delivered = agent.dropoff_order(grid)
        for order in delivered:
            if grid.time <= order.deadline:
                metrics[agent.id]['reward'] += DROPOFF_REWARD
            else:
                metrics[agent.id]['reward'] -= 20
                metrics[agent.id]['missed'] += 1
            metrics[agent.id]['deliveries'] += 1
        for order in agent.inventory:
            metrics[agent.id]['reward'] += PICKUP_REWARD

    # ---- Event handling ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---- Draw everything ----
    screen.fill(WHITE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if (x,y) in grid.obstacles:
                pygame.draw.rect(screen, (200,200,200), rect)

    # Pick-ups
    for order in grid.orders:
        if order.ready and not order.picked_up:
            ox, oy = order.pickup
            pygame.draw.circle(screen, YELLOW, (ox*CELL_SIZE + CELL_SIZE//2, oy*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//6)

    # Drop-offs
    for dx, dy in grid.dropoffs:
        pygame.draw.rect(screen, RED, pygame.Rect(dx*CELL_SIZE+10, dy*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20))

    # Agents
    for agent in agents:
        ax, ay = agent.pos
        pygame.draw.circle(screen, agent.color, (ax*CELL_SIZE + CELL_SIZE//2, ay*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)
        for i, order in enumerate(agent.inventory):
            pygame.draw.circle(screen, agent.color, (ax*CELL_SIZE + CELL_SIZE//2, ay*CELL_SIZE + CELL_SIZE//2 - (i+1)*15), CELL_SIZE//8)

    # Draw MDP value function and policy for agent 0
    rewards = compute_rewards(grid, agents[0])
    V, policy = value_iteration(grid, agents[0], rewards)
    draw_value_function(screen, V, grid, font)
    draw_policy_arrows(screen, policy, grid)

    pygame.display.flip()

pygame.quit()

# ----- Print Metrics -----
print(f"\nAlgorithm: {algorithm}")
for agent_id, data in metrics.items():
    avg_steps = data['steps']/data['deliveries'] if data['deliveries']>0 else 0
    print(f"Agent {agent_id}: Reward={data['reward']}, Deliveries={data['deliveries']}, Missed={data['missed']}, Avg Steps/Delivery={avg_steps:.2f}")