import random

def neighbors(pos, obstacles=set(), cop_pos=None, terminal_state=None, grid_size=10):
    """
    Returns all valid neighbor positions from `pos` considering obstacles,
    cop, and terminal state.
    """
    x, y = pos
    potential = [(x+dx, y+dy) for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]]
    valid = [
        (nx, ny) for nx, ny in potential
        if 0 <= nx < grid_size and 0 <= ny < grid_size
        and (nx, ny) not in obstacles
        and (nx, ny) != cop_pos
        and (nx, ny) != terminal_state
    ]
    return valid

def feasible_actions(pos, obstacles=set(), cop_pos=None, terminal_state=None, grid_size=10):
    """
    Returns all feasible action names ("up", "down", "left", "right")
    from current position, given obstacles, cop, and terminal state.
    """
    dirs = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}
    valid = []
    x, y = pos
    for a, (dx, dy) in dirs.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            # 🚨 Terminal is now allowed, only block obstacles and cops
            if (nx, ny) not in obstacles and (nx, ny) != cop_pos:
                valid.append(a)
    return valid

def ensure_free_neighbor(pos, obstacles, grid_size=10):
    """
    Returns True if there is at least one free neighbor for `pos`
    (agent or cop), False if trapped.
    """
    x, y = pos
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            if (nx, ny) not in obstacles:
                return True
    return False

def valid_obstacle_positions(grid_size, terminal_state, existing_obstacles=set()):
    """
    Generates a random obstacle position not overlapping terminal or existing obstacles.
    """
    while True:
        pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        if pos != terminal_state and pos not in existing_obstacles:
            return pos
