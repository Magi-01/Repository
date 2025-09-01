import pygame
import random
from db_utils import fetch_pending_items
from q_learning_agent import init_agent, process_item

CELL_SIZE = 50
GRID_WIDTH = 5
GRID_HEIGHT = 5
FPS = 2
AGENT_COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}
ITEM_COLORS = {'A': (255, 0, 0), 'B': (255, 165, 0), 'C': (255, 255, 0)}

# Shared state for visualization
agent_positions = {}
item_positions = {}

# Initialize Pygame (only once)
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Warehouse Visualization")
clock = pygame.time.Clock()

def warehouse_loop(agent_id):
    """Callable loop for one agent."""
    # Initialize agent in Q-table
    init_agent(agent_id)
    # Random start position
    if agent_id not in agent_positions:
        agent_positions[agent_id] = (random.randint(0, GRID_WIDTH - 1),
                                     random.randint(0, GRID_HEIGHT - 1))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((200, 200, 200))

        # Draw grid
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        # Draw items
        items = fetch_pending_items()
        for itm in items:
            if itm['item_id'] not in item_positions:
                item_positions[itm['item_id']] = (random.randint(0, GRID_WIDTH - 1),
                                                 random.randint(0, GRID_HEIGHT - 1))
            x, y = item_positions[itm['item_id']]
            pygame.draw.circle(screen, ITEM_COLORS[itm['item_type']],
                               (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2),
                               CELL_SIZE // 3)

        # Agent movement toward first item
        if items:
            target = items[0]
            target_x, target_y = item_positions[target['item_id']]
            ax, ay = agent_positions[agent_id]

            if ax < target_x: ax += 1
            elif ax > target_x: ax -= 1
            if ay < target_y: ay += 1
            elif ay > target_y: ay -= 1

            agent_positions[agent_id] = (ax, ay)

            if (ax, ay) == (target_x, target_y):
                # Process item via Q-learning
                action, success, reward = process_item(agent_id, target)
                # Remove item after drop
                items.pop(0)

        # Draw agents
        for aid, pos in agent_positions.items():
            x, y = pos
            pygame.draw.rect(screen, AGENT_COLORS[aid],
                             (x * CELL_SIZE + 5, y * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10))

        pygame.display.flip()
        clock.tick(FPS)
