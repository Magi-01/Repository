import pygame
from db_utils import fetch_pending_items
import random

# --- Configuration ---
CELL_SIZE = 50
GRID_WIDTH = 5   # number of columns
GRID_HEIGHT = 5  # number of rows
FPS = 2          # frames per second

AGENT_IDS = [1, 2]
AGENT_COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}  # green and blue

ITEM_COLORS = {'A': (255, 0, 0), 'B': (255, 165, 0), 'C': (255, 255, 0)}  # red, orange, yellow

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Warehouse Grid")
clock = pygame.time.Clock()

# --- Agent Positions (random start) ---
agent_positions = {agent_id: (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
                   for agent_id in AGENT_IDS}

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((200, 200, 200))  # background gray

    # --- Draw Grid ---
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (255, 255, 255), rect)
            pygame.draw.rect(screen, (0,0,0), rect, 1)  # black border

    # --- Draw Items ---
    items = []
    item = fetch_pending_items()
    while item:
        items.append(item)
        item = fetch_pending_items()
    for itm in items:
        x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
        pygame.draw.circle(screen, ITEM_COLORS[itm['item_type']], 
                           (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)

    # --- Draw Agents ---
    for agent_id, pos in agent_positions.items():
        x, y = pos
        pygame.draw.rect(screen, AGENT_COLORS[agent_id],
                         (x*CELL_SIZE+5, y*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
