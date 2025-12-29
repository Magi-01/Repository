import random

# -------------------------
# Utilities (can be standalone or part of a class)
# -------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid_position(pos, grid_size, obstacles):
    x, y = pos
    width, height = grid_size
    if not (0 <= x < width and 0 <= y < height):
        return False
    if pos in obstacles:
        return False
    return True

def valid_moves(pos, grid_size, obstacles):
    x, y = pos
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [c for c in candidates if is_valid_position(c, grid_size, obstacles)]

def next_position(pos, direction, speed, grid_size, obstacles):
    x, y = pos
    dx, dy = direction
    nx, ny = x, y
    for _ in range(speed):
        tx, ty = nx + dx, ny + dy
        if is_valid_position((tx, ty), grid_size, obstacles):
            nx, ny = tx, ty
        else:
            break
    return (nx, ny)

def get_visible_cells(pos, radius=1, grid_size=None):
    x, y = pos
    visible = set()
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if grid_size is None or (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]):
            visible.add((nx, ny))
    return visible

def penalty_for_idling(idle_count, base=2, max_penalty=1000):
    if idle_count == 0:
        return 0
    return min(base ** (idle_count - 1), max_penalty)

def all_horns_active(trains):
    """Return True if all trains have their horn active."""
    return all(t.get("horn", False) for t in trains)

def visibility_after_horn_penalty(trains):
    """
    When all trains blow their horns simultaneously, 
    they lose visibility equal to the sum of their speeds.
    """
    total_speed = sum(t.get("speed", 0) for t in trains)
    return total_speed

def detect_head_on_collision(train1, train2, grid_size, obstacles):
    pos1 = train1['pos']
    pos2 = train2['pos']
    next1 = next_position(pos1, train1['dir'], train1['speed'], grid_size, obstacles)
    next2 = next_position(pos2, train2['dir'], train2['speed'], grid_size, obstacles)
    return next1 == pos2 and next2 == pos1

def resolve_head_on_direction():
    return random.choice(['left', 'right'])

def direction_left(direction):
    dx, dy = direction
    return (-dy, dx)

def direction_right(direction):
    dx, dy = direction
    return (dy, -dx)

# -------------------------
# Train-specific methods assuming class context
# -------------------------

def find_goal(self, train):
    tid = train["id"]
    if tid in self.collected_tickets:
        return self.collected_tickets[tid]

    pos = train["pos"]
    visible_tickets = list(self.tickets.keys())
    if not visible_tickets:
        return None

    closest = min(visible_tickets, key=lambda tpos: manhattan(pos, tpos))
    return closest

def get_next_direction(self, train):
    current_pos = train["pos"]
    goal = train.get("goal")

    if goal is None:
        goal = self.find_goal(train)
        train["goal"] = goal

    if goal is None:
        return (0, 0)

    valid_cells = valid_moves(current_pos, self.grid_size, self.obstacles)
    if not valid_cells:
        return (0, 0)

    best_move = min(valid_cells, key=lambda c: manhattan(c, goal))
    dx = best_move[0] - current_pos[0]
    dy = best_move[1] - current_pos[1]
    return (dx, dy)

def check_collisions(self):
    positions = {}
    for train in self.trains:
        pos = train["pos"]
        if pos in positions:
            return True
        positions[pos] = train["id"]
    return False
