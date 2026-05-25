import pygame
import sys
import random
from mdp_ticketworld import TicketMDP
from train_agent import TrainAgent
from calculations import manhattan, is_valid_position

GRID_SIZE = (8,8)
FPS = 5

def init_trains(grid_size, num_trains, obstacles):
    trains = []
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    taken_positions = set(obstacles)
    while len(trains) < num_trains:
        pos = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
        if pos in taken_positions:
            continue
        if any(manhattan(pos, t['pos']) <= 1 for t in trains):
            continue
        trains.append({"id": len(trains), "pos": pos, "speed": 0, "dir": (0,0), "horn": False})
        taken_positions.add(pos)
    print(f"[DEBUG] Initialized trains at positions: {[t['pos'] for t in trains]}")
    return trains

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE[0]*60, GRID_SIZE[1]*60 + 100))
    pygame.display.set_caption("Multi-Train MDP Simulation with Debugging")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    obstacles = [(3,3), (4,4)]
    tickets = {(2,3):(5,5), (4,1):(0,6)}
    deliveries = {(5,5), (0,6)}

    trains = init_trains(GRID_SIZE, 2, obstacles)
    mdp = TicketMDP(GRID_SIZE, obstacles, tickets, deliveries, num_trains=2)
    initial_state = mdp.initial_state(trains)

    print("[DEBUG] Running value iteration (may take a while)...")
    print(f"[DEBUG] Passing initial state to value iteration: {initial_state}")
    mdp.value_iteration(trains)

    agents = {t["id"]: TrainAgent(t["id"], mdp) for t in trains}
    state = initial_state
    done = False
    step = 0

    idle_counters = [0 for _ in trains]

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not done:
            joint_action = []
            for train in trains:
                action, status = agents[train["id"]].act(train, state)
                joint_action.append(action)

            next_state, reward, done = mdp.step(state, joint_action)
            print(f"[DEBUG] Step {step}: Joint action={joint_action}, Reward={reward}, Done={done}")
            state = next_state
            step += 1

            trains_state, tickets_left, idle_counters = state
            for i, ts in enumerate(trains_state):
                trains[i]["pos"] = ts[0]
                trains[i]["speed"] = ts[1]
                trains[i]["horn"] = joint_action[i] == "horn"
                trains[i]["idle_counter"] = idle_counters[i]

        screen.fill((255,255,255))
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                rect = pygame.Rect(x*60, y*60, 60, 60)
                pygame.draw.rect(screen, (200,200,200), rect, 1)
                if (x,y) in obstacles:
                    pygame.draw.rect(screen, (0,0,0), rect)
                if (x,y) in tickets:
                    pygame.draw.rect(screen, (255,255,0), rect.inflate(-20,-20))
                if (x,y) in deliveries:
                    pygame.draw.rect(screen, (0,255,0), rect.inflate(-30,-30))
        for train in trains:
            x,y = train["pos"]
            rect = pygame.Rect(x*60+5, y*60+5, 50, 50)
            pygame.draw.rect(screen, (255,0,0), rect)
            speed_text = font.render(str(train["speed"]), True, (255,255,255))
            screen.blit(speed_text, (x*60+10, y*60+10))
            if train["horn"]:
                pygame.draw.rect(screen, (0,0,255), rect, 3)

        step_text = font.render(f"Step: {step} {'DONE' if done else ''}", True, (0,0,0))
        screen.blit(step_text, (10, GRID_SIZE[1]*60 + 10))

        pygame.display.flip()

if __name__ == "__main__":
    main()
