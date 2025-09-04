import random
import pygame
from config import Config
from csp import neighbors, feasible_actions, ensure_free_neighbor, valid_obstacle_positions

# -------------------
# Initialize positions
# -------------------

def initialize_positions(grid_size, num_obstacles):
    # Terminal
    terminal_state = (random.randint(0,grid_size-1), random.randint(0,grid_size-1))

    # Obstacles
    obstacles = set()
    while len(obstacles) < num_obstacles:
        obs = valid_obstacle_positions(grid_size, terminal_state, obstacles)
        obstacles.add(obs)

    # Agent spawn
    while True:
        agent_pos = (random.randint(0,grid_size-1), random.randint(0,grid_size-1))
        if agent_pos != terminal_state and agent_pos not in obstacles:
            break

    # Cop spawn
    while True:
        cop_pos = (random.randint(0,grid_size-1), random.randint(0,grid_size-1))
        if cop_pos != terminal_state and cop_pos not in obstacles and cop_pos != agent_pos:
            break

    return agent_pos, terminal_state, cop_pos, obstacles

# -------------------
# Cop stochastic move
# -------------------

def cop_stochastic_move(cop_pos, obstacles, terminal_state, grid_size, distance=1):
    for _ in range(distance):
        n = neighbors(cop_pos, obstacles, terminal_state, terminal_state, grid_size)
        if n:
            cop_pos = random.choice(n)
    return cop_pos

# -------------------
# Naive Cop stochastic move
# -------------------

def move_cop(cop_pos, obstacles, grid_size=Config.GRID_SIZE):
    """
    Moves the cop randomly one step (up/down/left/right), avoiding obstacles.
    """
    directions = [(0,-1), (0,1), (-1,0), (1,0)]
    random.shuffle(directions)  # randomize the order
    
    for dx, dy in directions:
        nx, ny = cop_pos[0] + dx, cop_pos[1] + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
            return (nx, ny)
    # If blocked in all directions, stay in place
    return cop_pos

def draw(agent, cop, terminal_state, obstacles, status, grid_size, V=None, optimal_actions=None):
    """
    Draws the grid world.
    - agent, cop, terminal_state: (x,y)
    - obstacles: set of (x,y)
    - status: string
    - V: dict mapping positions to value (optional)
    - optimal_actions: dict mapping positions to action (optional)
    """
    pygame.init()
    cell = 40
    win = pygame.display.set_mode((cell*grid_size, cell*grid_size + 40))
    win.fill((255,255,255))

    # Draw terminal
    pygame.draw.rect(win, (0,255,0), (terminal_state[0]*cell, terminal_state[1]*cell, cell, cell))
    
    # Draw obstacles
    for obs in obstacles:
        pygame.draw.rect(win, (128,128,128), (obs[0]*cell, obs[1]*cell, cell, cell))
    
    # Draw cop
    pygame.draw.rect(win, (0,0,255), (cop[0]*cell, cop[1]*cell, cell, cell))
    
    # Draw agent
    pygame.draw.rect(win, (255,0,0), (agent[0]*cell, agent[1]*cell, cell, cell))
    
    # Draw optimal action vectors
    if optimal_actions:
        for pos, action in optimal_actions.items():
            cx, cy = pos
            start = (cx*cell + cell//2, cy*cell + cell//2)
            dx, dy = {"up": (0,-15), "down": (0,15), "left": (-15,0), "right": (15,0)}[action]
            end = (start[0]+dx, start[1]+dy)
            pygame.draw.line(win, (0,0,0), start, end, 2)
            # arrowhead
            pygame.draw.circle(win, (0,0,0), end, 3)

    # Draw status
    font = pygame.font.SysFont(None,30)
    if status:
        text = font.render(status, True, (0,0,0))
        win.blit(text, (10, cell*grid_size))

    # Draw optimal value
    if V:
        agent_val = V.get(agent, 0)
        text_val = font.render(f"Optimal value: {agent_val:.2f}", True, (0,0,0))
        win.blit(text_val, (cell*grid_size//2, cell*grid_size))

    pygame.display.update()


def optimal_action(state, obstacles=set(), cop_pos=None, terminal_state=None):
        """
        Returns the action that maximizes the value function from the given state.
        Only considers feasible actions.
        """
        grid_size = Config.GRID_SIZE  # or the value used for the simulation
        feasible = feasible_actions(state, obstacles, cop_pos, terminal_state, grid_size=grid_size)
        if not feasible:
            return None

        x, y = state
        # Compute Q-values for feasible actions
        q_vals = {}
        for a in feasible:
            dx, dy = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}[a]
            nx, ny = x + dx, y + dy
            if (nx, ny) in obstacles or (cop_pos and (nx, ny)==cop_pos):
                nx, ny = x, y  # stay in place if blocked

            # Lookup value from vectorized V — assume states is [(x,y)]
            idx = states.index((nx, ny))
            q_vals[a] = V[idx]

        # Pick action with highest value
        best_val = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == best_val]
        return random.choice(best_actions)