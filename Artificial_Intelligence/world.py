import pygame

class World:
    def __init__(self, grid_size, obstacles, tickets, deliveries):
        self.grid_size = grid_size
        self.obstacles = set(obstacles)
        self.tickets = tickets
        self.deliveries = deliveries

        self.CELL_SIZE = 60
        self.COLORS = {
            "background": (255, 255, 255),
            "grid_line": (200, 200, 200),
            "obstacle": (0, 0, 0),
            "ticket": (255, 255, 0),
            "delivery": (0, 255, 0),
            "horn_border": (0, 0, 255),
            "train_colors": [(200, 0, 0), (0, 0, 200), (0, 150, 0), (200, 200, 0)],
            "text": (255, 255, 255),
        }

    def draw(self, screen, trains, font, step, agent_statuses):
        screen.fill(self.COLORS["background"])

        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(screen, self.COLORS["grid_line"], rect, 1)
                if (x, y) in self.obstacles:
                    pygame.draw.rect(screen, self.COLORS["obstacle"], rect)

        for pos in self.tickets:
            rect = pygame.Rect(pos[0]*self.CELL_SIZE+10, pos[1]*self.CELL_SIZE+10, self.CELL_SIZE-20, self.CELL_SIZE-20)
            pygame.draw.rect(screen, self.COLORS["ticket"], rect)

        for pos in self.deliveries:
            rect = pygame.Rect(pos[0]*self.CELL_SIZE+15, pos[1]*self.CELL_SIZE+15, self.CELL_SIZE-30, self.CELL_SIZE-30)
            pygame.draw.rect(screen, self.COLORS["delivery"], rect)

        for train in trains:
            x, y = train["pos"]
            rect = pygame.Rect(x * self.CELL_SIZE + 5, y * self.CELL_SIZE + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            color = self.COLORS["train_colors"][train["id"] % len(self.COLORS["train_colors"])]
            pygame.draw.rect(screen, color, rect)

            if train["horn"]:
                pygame.draw.rect(screen, self.COLORS["horn_border"], rect, 3)

            speed_text = font.render(str(train["speed"]), True, self.COLORS["text"])
            screen.blit(speed_text, (x * self.CELL_SIZE + 10, y * self.CELL_SIZE + 10))

            id_text = font.render(f"T{train['id']}", True, self.COLORS["text"])
            screen.blit(id_text, (x * self.CELL_SIZE + 10, y * self.CELL_SIZE + 30))

        status_y = self.grid_size * self.CELL_SIZE + 5
        step_text = font.render(f"Step: {step}", True, (0, 0, 0))
        screen.blit(step_text, (10, status_y))

        for i, status in enumerate(agent_statuses):
            txt = font.render(f"Train {i}: {status}", True, (0, 0, 0))
            screen.blit(txt, (150, status_y + i * 20))
