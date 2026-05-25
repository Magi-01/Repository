import threading
import time
import random
import pygame
from datetime import datetime
import q_learning_agent
from db_utils import fetch_bin_location, mark_item_carrying, update_item_status

# --- Config ---
NUM_AGENTS = 2
ITEM_INTERVAL = 2
GRID_WIDTH, GRID_HEIGHT = 10, 10
CELL_SIZE = 50
FPS = 60

AGENT_COLORS = {1:(0,255,0), 2:(0,0,255)}
ITEM_COLORS = {'A':(255,0,0), 'B':(255,165,0), 'C':(255,255,0)}
BIN_IDS = [1,2,3]

ITEM_TYPES = ['A','B','C']
BIN_MAP = {'A':1,'B':2,'C':3}
MIN_DEADLINE, MAX_DEADLINE = 10, 30

# --- Shared state ---
agent_positions = {}
item_positions = {}  # item_id -> (x,y)
items = []           # list of dicts
new_items_event = threading.Event()  # signal new items

# --- Utility ---
def random_free_cell(exclude_positions):
    while True:
        x = random.randint(0, GRID_WIDTH-1)
        y = random.randint(0, GRID_HEIGHT-1)
        if (x,y) not in exclude_positions:
            return (x,y)

# --- Item Generator ---
def generator_loop():
    while True:
        item_type = random.choice(ITEM_TYPES)
        correct_bin = BIN_MAP[item_type]
        deadline = random.randint(MIN_DEADLINE, MAX_DEADLINE)
        drop_bin = correct_bin
        item_id = int(time.time() * 1000)

        item = {
            'item_id': item_id,
            'item_type': item_type,
            'correct_bin': correct_bin,
            'drop_bin': drop_bin,
            'status': 'pending',
            'deadline_seconds': deadline,
            'arrival_time': datetime.now()
        }

        # Assign a free position for the item
        bin_coords = [ (fetch_bin_location(b)['x'], fetch_bin_location(b)['y']) for b in BIN_IDS ]
        item_positions[item_id] = random_free_cell(bin_coords + list(agent_positions.values()))
        items.append(item)
        print(f"[NEW ITEM] {item_type} -> Bin {correct_bin}, Deadline {deadline}s")
        new_items_event.set()
        time.sleep(ITEM_INTERVAL)
        new_items_event.clear()

# --- Agent Thread ---
def agent_loop(agent_id):
    # Initialize agent state
    if agent_id not in q_learning_agent.AGENT_STATE:
        q_learning_agent.init_agent(agent_id)

    # Random start position
    agent_positions[agent_id] = (random.randint(0, GRID_WIDTH-1),
                                 random.randint(0, GRID_HEIGHT-1))

    while True:
        # Only consider available items
        available_items = [itm for itm in items if itm['status'] == 'pending']

        state = q_learning_agent.AGENT_STATE[agent_id]

        # --- Not carrying an item ---
        if state['carrying_item'] is None:
            # Choose target item if none
            if state.get('target_item_id') is None and available_items:
                # Simple heuristic: choose first available item
                target_item = available_items[0]
                state['target_item_id'] = target_item['item_id']
                q_learning_agent.AGENT_STATE[agent_id] = state

            # Move toward target item using Q-learning
            if state.get('target_item_id') is not None:
                item_id = state['target_item_id']
                item = next((itm for itm in available_items if itm['item_id'] == item_id), None)
                if item:
                    # Compute Q-learning action
                    agent_pos = agent_positions[agent_id]
                    item_pos = item_positions[item_id]
                    move_action = q_learning_agent.choose_action(agent_id, agent_pos, item_pos, carrying=False)
                    new_pos = q_learning_agent.simulate_move(agent_pos, move_action, GRID_WIDTH, GRID_HEIGHT)
                    agent_positions[agent_id] = new_pos

                    # Pick up item only if reached
                    if new_pos == item_pos:
                        mark_item_carrying(item_id, agent_id)
                        state['carrying_item'] = item
                        bin_loc = fetch_bin_location(item['drop_bin'])
                        state['target_bin'] = (bin_loc['x'], bin_loc['y'])
                        state['target_item_id'] = None
                        q_learning_agent.AGENT_STATE[agent_id] = state
                        print(f"[AGENT {agent_id}] Picked up {item['item_type']}")

        # --- Carrying an item ---
        else:
            item = state['carrying_item']
            target_x, target_y = state['target_bin']
            ax, ay = agent_positions[agent_id]

            # Use Q-learning to move toward target bin
            move_action = q_learning_agent.choose_action(agent_id, (ax, ay), (target_x, target_y), carrying=True)
            new_pos = q_learning_agent.simulate_move((ax, ay), move_action, GRID_WIDTH, GRID_HEIGHT)
            agent_positions[agent_id] = new_pos

            # Drop item if reached
            if new_pos == (target_x, target_y):
                q_learning_agent.process_item(agent_id, item)
                item_positions.pop(item['item_id'], None)
                items[:] = [itm for itm in items if itm['item_id'] != item['item_id']]
                state['carrying_item'] = None
                state['target_bin'] = None
                q_learning_agent.AGENT_STATE[agent_id] = state
                print(f"[AGENT {agent_id}] Dropped {item['item_type']} at bin {item['drop_bin']}")
                
        now = datetime.now()
        # Remove expired items
        items[:] = [itm for itm in items if (now - itm['arrival_time']).seconds < itm['deadline_seconds']]
        for itm in list(item_positions):
            if itm not in [i['item_id'] for i in items]:
                del item_positions[itm]

        time.sleep(0.1)

# --- Visualization ---
def visualization_loop():
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH*CELL_SIZE, GRID_HEIGHT*CELL_SIZE))
    pygame.display.set_caption("Warehouse Simulation")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((200,200,200))

        # Draw grid
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen,(255,255,255),rect)
                pygame.draw.rect(screen,(0,0,0),rect,1)

        # Draw bins
        for bin_id in BIN_IDS:
            loc = fetch_bin_location(bin_id)
            pygame.draw.rect(screen, (150,150,255),
                             (loc['x']*CELL_SIZE+10, loc['y']*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20))

        # Draw items
        for itm in items:
            x, y = item_positions[itm['item_id']]
            pygame.draw.circle(screen, ITEM_COLORS[itm['item_type']],
                               (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//3)

        # Draw agents
        for aid, pos in agent_positions.items():
            x, y = pos
            pygame.draw.rect(screen, AGENT_COLORS[aid],
                             (x*CELL_SIZE+5, y*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10))
            carrying_item = q_learning_agent.AGENT_STATE[aid]['carrying_item']
            if carrying_item:
                pygame.draw.circle(screen, ITEM_COLORS[carrying_item['item_type']],
                                   (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE-10), CELL_SIZE//4)

        pygame.display.flip()
        clock.tick(FPS)

# --- Main ---
if __name__ == "__main__":
    # Start item generator
    threading.Thread(target=generator_loop, daemon=True).start()

    # Start agent threads
    for aid in range(1, NUM_AGENTS+1):
        threading.Thread(target=agent_loop, args=(aid,), daemon=True).start()

    # Run visualization in main thread
    visualization_loop()
