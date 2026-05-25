import random
import pygame
from mdp import MDP
from csp import feasible_actions
from config import Config
from utils import draw, move_cop, optimal_action, initialize_positions

# Fixed parameters
GRID_SIZE = Config.GRID_SIZE
CELL_SIZE = 40  # visualization

# Initialize states (just agent positions)
states = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
actions = ["up", "down", "left", "right"]

# Initialize MDP
mdp = MDP(states=states, actions=actions, gamma=Config.GAMMA)

# Metrics
metrics = {"episodes": 0, "steps": [], "success": 0, "arrests": 0}


def simulate_value_iteration(episodes=Config.EPISODES, visualize=True):
    for ep in range(episodes):

        agent_pos, terminal_state, cop_pos, obstacles = initialize_positions(Config.GRID_SIZE, Config.NUM_OBSTACLES)

        # Run vectorized value iteration
        mdp.value_iteration()

        done = False
        status = ""
        step_count = 0

        while not done:
            if visualize:
                draw(agent_pos, cop_pos, terminal_state, obstacles, status, GRID_SIZE)
                pygame.time.delay(int(Config.STEP_DELAY*1000))

            # Get feasible actions
            feasible = feasible_actions(agent_pos, obstacles, cop_pos, terminal_state, GRID_SIZE)
            if not feasible:
                status = "Trapped!"
                done = True
                break

            # Pick optimal action from value function
            a = optimal_action(agent_pos, obstacles, cop_pos, terminal_state)
            dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[a]
            next_pos = (agent_pos[0]+dx, agent_pos[1]+dy)

            # Boundaries and obstacles
            if next_pos[0] < 0 or next_pos[0] >= GRID_SIZE or next_pos[1] < 0 or next_pos[1] >= GRID_SIZE:
                next_pos = agent_pos
            if next_pos in obstacles:
                next_pos = agent_pos

            agent_pos = next_pos
            step_count += 1

            cop_pos = move_cop(cop_pos, obstacles, GRID_SIZE)

            # Check terminal / cop
            if agent_pos == terminal_state:
                status = "Agent reached terminal!"
                metrics["success"] += 1
                done = True
            elif agent_pos == cop_pos:
                status = "Agent arrested!"
                metrics["arrests"] += 1
                done = True

        metrics["steps"].append(step_count)
        metrics["episodes"] += 1

        if visualize:
            draw(agent_pos, cop_pos, terminal_state, obstacles, status, GRID_SIZE)
            pygame.time.delay(1000)

    if visualize:
        pygame.quit()

if __name__ == "__main__":
    simulate_value_iteration()
    print("Simulation finished.")
    print("Episodes:", metrics["episodes"])
    print("Success:", metrics["success"])
    print("Arrests:", metrics["arrests"])
    print("Average steps:", sum(metrics["steps"]) / len(metrics["steps"]))
