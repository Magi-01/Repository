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

def move_cop(cop_pos, obstacles, terminal_state, grid_size, is_random=True):
    """
    Moves the cop. If is_random=True: stochastic movement (random valid neighbor).
    If is_random=False: deterministic movement around goal (distance=1 orbit).
    """
    if is_random:
        # Random movement
        directions = [(0,-1), (0,1), (-1,0), (1,0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cop_pos[0] + dx, cop_pos[1] + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
                return (nx, ny)
        return cop_pos  # blocked, stay in place

    else:
        # Orbit around terminal_state at distance 1
        gx, gy = terminal_state
        x, y = cop_pos

        # Valid neighbors at distance 1 from goal
        neighbors_goal = [
            (gx+dx, gy+dy) 
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]
            if 0 <= gx+dx < grid_size and 0 <= gy+dy < grid_size
            and (gx+dx, gy+dy) not in obstacles
        ]

        # If cop is already at goal-neighbor, move clockwise or pick next
        if (x, y) in neighbors_goal:
            idx = neighbors_goal.index((x, y))
            next_idx = (idx + 1) % len(neighbors_goal)
            return neighbors_goal[next_idx]
        else:
            # Move cop to a neighbor of goal
            # Pick closest neighbor to current position
            neighbors_goal.sort(key=lambda pos: abs(pos[0]-x) + abs(pos[1]-y))
            return neighbors_goal[0]

def draw(agent, cop, terminal_state, obstacles, status, grid_size, V=None, optimal_actions=None, mdp=None):
    """
    Draws the grid world.
    - agent, cop, terminal_state: (x,y)
    - obstacles: set of (x,y)
    - status: string
    - V: dict mapping positions to value (optional)
    - optimal_actions: dict mapping positions to action (optional)
    - mdp: MDP instance (optional) to show Q-values
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

    font = pygame.font.SysFont(None, 20)

    # Draw Q-values (if mdp provided)
    if mdp:
        state_idx = mdp.state_to_idx.get((agent, cop))
        if state_idx is not None:
            for a_idx, action in enumerate(mdp.actions):
                dx, dy = mdp.action_helper.dir_map[action]
                nx, ny = agent[0] + dx, agent[1] + dy

                # Skip invalid
                if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                    continue
                if (nx, ny) in obstacles:
                    continue

                q_val = mdp.Q[state_idx, a_idx]
                norm_q = max(min(q_val / 50, 1), -1)  # simple clamp
                color = (255,0,0) if norm_q < 0 else (0,255,0)
                thickness = max(1, int(abs(norm_q)*5))

                start_pixel = (agent[0]*cell + cell//2, agent[1]*cell + cell//2)
                end_pixel = (nx*cell + cell//2, ny*cell + cell//2)
                pygame.draw.line(win, color, start_pixel, end_pixel, thickness)

                # Draw Q-value as short float
                q_val = mdp.Q[state_idx, a_idx]  # state_idx needs to be computed for agent pos
                q_text = font.render(f"{q_val:.4f}", True, (0,0,0))
                text_pos = (end_pixel[0]+2, end_pixel[1]+2)
                win.blit(q_text, text_pos)

    # Draw optimal action vectors
    if optimal_actions:
        for pos, action in optimal_actions.items():
            cx, cy = pos
            start = (cx*cell + cell//2, cy*cell + cell//2)
            dx, dy = {"up": (0,-15), "down": (0,15), "left": (-15,0), "right": (15,0)}[action]
            end = (start[0]+dx, start[1]+dy)
            pygame.draw.line(win, (0,0,0), start, end, 2)
            pygame.draw.circle(win, (0,0,0), end, 3)

    # Draw status
    status_font = pygame.font.SysFont(None,30)
    if status:
        text = status_font.render(status, True, (0,0,0))
        win.blit(text, (10, cell*grid_size))

    # Draw optimal value
    if V:
        agent_val = V.get(agent, 0)
        text_val = status_font.render(f"Optimal value: {agent_val:.2f}", True, (0,0,0))
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

def manhattan(agent_pos, adversary_position):
    return abs(
        agent_pos[0]-adversary_position[0])+abs(
        agent_pos[1]-adversary_position[1])