# simulations_learning.py
import pygame
import random
from mdp import MDP
from csp import neighbors, feasible_actions, ensure_free_neighbor, valid_obstacle_positions
from config import Config
import numpy as np
from utils import draw, cop_stochastic_move, initialize_positions


# -------------------
# Main simulation
# -------------------
def run_simulation():
    grid_size = Config.GRID_SIZE
    mdp = MDP(
        states= [((ax, ay), (tx, ty))
          for ax in range(grid_size)
          for ay in range(grid_size)
          for tx in range(grid_size)
          for ty in range(grid_size)],
        actions=['up','down','left','right'],
        gamma=Config.GAMMA
    )

    wins = 0
    arrests = 0
    rewards_history = []
    steps_history = []

    for episode in range(Config.EPISODES):
        agent_pos, terminal_state, cop_pos, obstacles = initialize_positions(grid_size, Config.NUM_OBSTACLES)
        cop_timer = 0
        cop_timer_max = random.randint(*Config.COP_TIMING_RANGE)
        cop_distance = random.randint(*Config.COP_DISTANCE_RANGE)
        done = False
        status = ""
        total_reward = 0
        steps = 0

        pygame.init()
        win = pygame.display.set_mode((Config.GRID_SIZE*40, Config.GRID_SIZE*40))
        pygame.display.set_caption("Dynamic Q-learning Simulation")


        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Agent step
            agent_pos, reward, done = mdp.learn_q(
                agent_pos,
                terminal_state=terminal_state,
                cop_pos=cop_pos,
                obstacles=obstacles,
                alpha=Config.ALPHA,
                epsilon=Config.EPSILON,
                single_step=False,
                episode=episode,
                grid_size=grid_size
            )

            total_reward += reward
            steps += 1

            # Cop stochastic move
            cop_timer += 1
            if cop_timer >= cop_timer_max:
                cop_pos = cop_stochastic_move(cop_pos, obstacles, terminal_state, grid_size, cop_distance)
                cop_timer = 0
                cop_timer_max = random.randint(*Config.COP_TIMING_RANGE)
                cop_distance = random.randint(*Config.COP_DISTANCE_RANGE)

            # Check win/loss
            if agent_pos == terminal_state:
                status = "Agent WON!"
                wins += 1
                done = True
            elif agent_pos == cop_pos:
                status = "Agent ARRESTED!"
                arrests += 1
                done = True

            # Draw
            draw(agent_pos, cop_pos, terminal_state, obstacles, status, grid_size)
            pygame.time.delay(int(Config.STEP_DELAY*1000))

        # Record metrics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        print(f"Episode {episode+1}: {status} | Steps: {steps} | Total reward: {total_reward}")

    # Summary metrics
    print("\n=== Simulation Summary ===")
    print(f"Total episodes: {Config.EPISODES}")
    print(f"Wins: {wins} | Arrests: {arrests}")
    print(f"Average reward: {np.mean(rewards_history):.2f}")
    print(f"Average steps per episode: {np.mean(steps_history):.2f}")

if __name__ == "__main__":
    run_simulation()
