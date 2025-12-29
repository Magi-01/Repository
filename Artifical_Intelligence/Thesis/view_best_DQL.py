# view_dqn.py
import torch
import pygame
from simulations_learning import DroneEnv, DQNAgent, state_to_vector, VISUALIZE, CELL_SIZE, GRID_SIZE, RewardTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load trained agent
# -----------------------------
agent = DQNAgent()
agent.model.load_state_dict(torch.load("best_model02.pt", map_location=device))
agent.target.load_state_dict(agent.model.state_dict())
agent.model.eval()  # evaluation mode
agent.epsilon = 1

# -----------------------------
# Create environment
# -----------------------------
env = DroneEnv()

reward_tracker = RewardTracker()  # dummy tracker

if VISUALIZE:
    pygame.init()
    screen_size = GRID_SIZE * CELL_SIZE
    win = pygame.display.set_mode((screen_size, screen_size + 40))
    pygame.display.set_caption("Drone Simulation")
    clock = pygame.time.Clock()
    background = pygame.Surface(win.get_size())
    background.fill((255, 255, 255))

# -----------------------------
# Run one episode
# -----------------------------
state = env.reset()
done = False
total_reward = 0

while not done:
    
    state_vec = state_to_vector(state)
    action = agent.choose_action(state_vec)

    next_state, reward, done, delivered = env.step(action, reward_tracker)  # reward_tracker not needed for viewing
    total_reward = (reward + total_reward)
    state = next_state

    # Visualization
    if VISUALIZE:
        win.blit(background, (0,0))
        # draw function from drone_simulation.py
        from utils import draw
        draw(env.agent_pos, env.cop_pos, env.goal_pos, env.obstacles,
             status=f"Reward: {total_reward:.2f}",
             grid_size=GRID_SIZE,
             win=win,
             background=background,
             clock=clock,
             fps=20)

pygame.quit()
print(f"Episode finished. Total reward: {total_reward:.2f}")
