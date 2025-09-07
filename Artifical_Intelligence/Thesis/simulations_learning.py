# drone_simulation.py
import random
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from utils import initialize_positions, move_cop, draw, manhattan
from csp import feasible_actions
import math

# -----------------------------
# GLOBAL PARAMETERS (editable)
# -----------------------------
GRID_SIZE = 20           # Grid width/height
NUM_OBSTACLES = 5        # Number of fixed obstacles
MAX_EPISODES = 50000       # Number of episodes
MAX_STEPS = 200          # Max steps per episode
ALPHA = 0.1              # Learning rate for Q-learning
GAMMA = 0.99             # Discount factor
EPSILON = 1            # Exploration rate
DQN_LR = 1e-3            # Learning rate for DQN
DQN_BATCH_SIZE = 64
DQN_MEMORY_SIZE = 500000
CELL_SIZE = 50            # Pixels per grid cell
VISUALIZE = False         # Enable Pygame visualization
USE_DQN = False          # Set True for large grids
IS_RANDOM = False
SLOWDOWN = 200   # milliseconds delay per step (0 = no delay)
REWARD_DELIVERY = 150       # reward for successful delivery
PENALTY_CAUGHT = -15       # penalty for being caught by cop
STEP_PENALTY = -1        # small penalty for each move
DISTANCE_REWARD = 5.0      # bonus for moving closer to goal
USE_DISTANCE_SHAPING = True  # toggle for shaping
USE_VISIT_BONUS = True        # toggle for exploration bonus
VISIT_BONUS = 2


class RewardTracker:
    def __init__(self):
        self.visited = set()

    def reset(self):
        """Reset visited cells at start of episode"""
        self.visited.clear()

    def compute_reward(self, prev_pos, new_pos, goal_pos, caught, delivered):
        reward = 0.0

        # success
        if delivered:
            reward += REWARD_DELIVERY

        # failure
        if caught:
            reward += PENALTY_CAUGHT

        # per-step penalty
        reward += STEP_PENALTY

        # shaping: bonus if closer to goal
        if USE_DISTANCE_SHAPING:
            prev_dist = abs(prev_pos[0] - goal_pos[0]) + abs(prev_pos[1] - goal_pos[1])
            new_dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
            if new_dist < prev_dist:
                reward += DISTANCE_REWARD

        # exploration bonus
        if USE_VISIT_BONUS and new_pos not in self.visited:
            reward += VISIT_BONUS
            self.visited.add(new_pos)

        return reward
# -----------------------------
# 1. Environment
# -----------------------------
class DroneEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.reset()

    def reset(self):
        self.agent_pos, self.goal_pos, self.cop_pos, self.obstacles = initialize_positions(self.grid_size, self.num_obstacles)
        self.carrying_food = True
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return (self.agent_pos[0], self.agent_pos[1],
                self.cop_pos[0], self.cop_pos[1],
                self.goal_pos[0], self.goal_pos[1],
                int(self.carrying_food))

    def step(self, action_idx, reward_tracker):
        # Map action index to feasible action
        action_names = ["up","down","left","right"]
        reward = 0
        feasible = feasible_actions(self.agent_pos, self.obstacles, self.cop_pos, self.goal_pos, self.grid_size)
        if not feasible:
            action = None
        else:
            action = action_names[action_idx % len(action_names)]
            if action not in feasible:
                action = random.choice(feasible)
        # Apply movement
        dxdy = {"up":(0,-1),"down":(0,1),"left":(-1,0),"right":(1,0)}
        if action:
            dx, dy = dxdy[action]
            new_pos = (self.agent_pos[0]+dx, self.agent_pos[1]+dy)
            if new_pos not in self.obstacles and 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self.agent_pos = new_pos

        # Move cop
        self.cop_pos = move_cop(self.cop_pos, self.obstacles, self.goal_pos, self.grid_size, IS_RANDOM)

        # Collision
        done = False
        if self.agent_pos == self.cop_pos:
            done = True
        elif self.agent_pos == self.goal_pos and self.carrying_food:
            reward = 10
            done = True
            self.carrying_food = False
            # New delivery location
            self.goal_pos = (random.randint(0,self.grid_size-1), random.randint(0,self.grid_size-1))

        # Step penalty
        reward += reward_tracker.compute_reward(
            prev_pos=self.agent_pos,
            new_pos=new_pos,
            goal_pos=self.goal_pos,
            caught=new_pos==self.cop_pos,
            delivered=self.carrying_food == False
            )

        self.steps += 1

        return self.get_state(), reward, done

# -----------------------------
# 2. Q-learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self, n_actions=4):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions-1)
        return int(np.argmax(self.Q[state]))

    def learn(self, state, action, reward, next_state, done):
        q_next = 0 if done else max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma*q_next - self.Q[state][action])

# -----------------------------
# 3. DQN Agent
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_dim)
        )
    def forward(self,x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_dim=7, n_actions=4):
        self.model = DQN(input_dim,n_actions)
        self.target = DQN(input_dim,n_actions)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=DQN_LR)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.memory = deque(maxlen=DQN_MEMORY_SIZE)
        self.batch_size = DQN_BATCH_SIZE
        self.n_actions = n_actions
        self.steps = 0
        self.update_freq = 10

    def choose_action(self,state_vec):
        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions-1)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        q_vals = self.model(state_tensor)
        return int(torch.argmax(q_vals).item())

    def store(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory,self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_pred = self.model(states).gather(1,actions)
        q_next = self.target(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma*q_next*(1-dones)
        loss = nn.MSELoss()(q_pred,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

# -----------------------------
# 4. State normalization for DQN
# -----------------------------
def state_to_vector(state):
    return np.array([state[0]/GRID_SIZE,state[1]/GRID_SIZE,
                     state[2]/GRID_SIZE,state[3]/GRID_SIZE,
                     state[4]/GRID_SIZE,state[5]/GRID_SIZE,
                     state[6]])

# -----------------------------
# 5. Pygame Visualization
# -----------------------------
def visualize(env, agent_state=None, status=""):
    if not VISUALIZE:
        return
    pygame.init()
    screen_size = GRID_SIZE*CELL_SIZE
    win = pygame.display.set_mode((screen_size, screen_size+40))
    win.fill((255,255,255))

    # Draw grid lines
    for x in range(0,screen_size,CELL_SIZE):
        pygame.draw.line(win,(200,200,200),(x,0),(x,screen_size))
    for y in range(0,screen_size,CELL_SIZE):
        pygame.draw.line(win,(200,200,200),(0,y),(screen_size,y))

    # Draw obstacles
    for obs in env.obstacles:
        pygame.draw.rect(win,(128,128,128),(obs[0]*CELL_SIZE,obs[1]*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    # Draw goal
    pygame.draw.rect(win,(0,255,0),(env.goal_pos[0]*CELL_SIZE,env.goal_pos[1]*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    # Draw cop
    pygame.draw.rect(win,(0,0,255),(env.cop_pos[0]*CELL_SIZE,env.cop_pos[1]*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    # Draw agent
    pygame.draw.rect(win,(255,0,0),(env.agent_pos[0]*CELL_SIZE,env.agent_pos[1]*CELL_SIZE,CELL_SIZE,CELL_SIZE))

    font = pygame.font.SysFont(None,30)
    text = font.render(status,True,(0,0,0))
    win.blit(text,(10,screen_size))
    pygame.display.update()

# -----------------------------
# 6. Training Loop
# -----------------------------
def train():
    env = DroneEnv()
    reward_tracker = RewardTracker()
    if USE_DQN:
        agent = DQNAgent()
    else:
        agent = QLearningAgent()
    reward_tracker.reset()
    for ep in range(MAX_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if USE_DQN:
                state_vec = state_to_vector(state)
                action = agent.choose_action(state_vec)
            else:
                action = agent.choose_action(state)
            next_state, reward, done = env.step(action, reward_tracker)
            total_reward += reward
            if USE_DQN:
                agent.epsilon = max(0.05, agent.epsilon * 0.995)
                next_vec = state_to_vector(next_state)
                agent.store(state_vec,action,reward,next_vec,int(done))
                agent.learn()
            else:
                agent.learn(state,action,reward,next_state,done)
            state = next_state
            if VISUALIZE:
                visualize(env,status=f"Episode {ep} Reward {total_reward:.1f}")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            if VISUALIZE and SLOWDOWN > 0:
                pygame.time.delay(SLOWDOWN)
        print(f"Episode {ep} Reward {total_reward:.2f}")
    pygame.quit()

if __name__ == "__main__":
    train()
