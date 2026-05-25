# Utils.py
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

def move_cop(cop_pos, obstacles, terminal_state, grid_size, COP_BEHAVIOR):
    """
    Moves the cop based on COP_BEHAVIOR:
    - "stochastic": random valid neighbor
    - "goal_patrol": deterministic orbit around goal (distance=1)
    - "still": cop does not move
    """

    if COP_BEHAVIOR == "stochastic":
        # Random movement
        directions = [(0,-1), (0,1), (-1,0), (1,0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cop_pos[0] + dx, cop_pos[1] + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
                return (nx, ny)
        return cop_pos  # blocked, stay in place

    elif COP_BEHAVIOR == "goal_patrol":
        # Orbit around terminal_state at distance 1
        gx, gy = terminal_state
        x, y = cop_pos
        neighbors_goal = [
            (gx+dx, gy+dy)
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]
            if 0 <= gx+dx < grid_size and 0 <= gy+dy < grid_size
            and (gx+dx, gy+dy) not in obstacles
        ]
        if not neighbors_goal:
            return cop_pos
        if (x, y) in neighbors_goal:
            idx = neighbors_goal.index((x, y))
            next_idx = (idx + 1) % len(neighbors_goal)
            return neighbors_goal[next_idx]
        else:
            neighbors_goal.sort(key=lambda pos: abs(pos[0]-x) + abs(pos[1]-y))
            return neighbors_goal[0]

    elif COP_BEHAVIOR == "still":
        # Cop does not move
        return cop_pos

    else:
        # Default to staying still if unknown behavior
        return cop_pos

def draw(agent, cop, terminal_state, obstacles, status, grid_size,
         win, background, clock, V=None, optimal_actions=None, mdp=None, fps=60):
    """
    Draws the grid world using the given pygame window.
    """
    cell = 40
    win.blit(background, (0, 0))  # clear screen

    # Draw grid lines
    for x in range(0, grid_size*cell, cell):
        pygame.draw.line(win, (200,200,200), (x,0), (x,grid_size*cell))
    for y in range(0, grid_size*cell, cell):
        pygame.draw.line(win, (200,200,200), (0,y), (grid_size*cell,y))

    # Draw terminal
    pygame.draw.rect(win, (0,255,0), (terminal_state[0]*cell, terminal_state[1]*cell, cell, cell))
    
    # Draw obstacles
    for obs in obstacles:
        pygame.draw.rect(win, (128,128,128), (obs[0]*cell, obs[1]*cell, cell, cell))
    
    # Draw cop
    pygame.draw.rect(win, (0,0,255), (cop[0]*cell, cop[1]*cell, cell, cell))
    
    # Draw agent
    pygame.draw.rect(win, (255,0,0), (agent[0]*cell, agent[1]*cell, cell, cell))

    # Optional: draw Q-values, optimal actions, V here...
    # (same as before)

    # Draw status
    font = pygame.font.SysFont(None, 30)
    text = font.render(status, True, (0,0,0))
    win.blit(text, (10, grid_size*cell))

    pygame.display.update()
    clock.tick(fps)  # limit FPS



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

def manhattan(agent_pos, adversary_position):
    return abs(
        agent_pos[0]-adversary_position[0])+abs(
        agent_pos[1]-adversary_position[1])