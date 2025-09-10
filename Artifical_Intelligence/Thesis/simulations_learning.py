# drone_simulation_parallel.py
import random
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from utils import initialize_positions, move_cop, draw, manhattan
from csp import feasible_actions
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
import pickle
import multiprocessing as mp

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
GRID_SIZE = 20
NUM_OBSTACLES = 5
MAX_EPISODES = 1000
MAX_STEPS = 500
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1
DQN_LR = 1e-3
DQN_BATCH_SIZE = 64
DQN_MEMORY_SIZE = 500000
CELL_SIZE = 50
VISUALIZE = True
USE_DQN = True
COP_BEHAVIOR = "still"
SLOWDOWN = 0
FPS = 200
REWARD_DELIVERY = 450
PENALTY_CAUGHT = -20
STEP_PENALTY = -3
DISTANCE_REWARD = 5.0
USE_DISTANCE_SHAPING = True
USE_VISIT_BONUS = True
VISIT_BONUS = 5
REWARD_SCALE = 100
NUM_THREADS = 6  # Number of parallel environments per batch
IS_GOAL_FIXED = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Reward tracker
# -----------------------------
class RewardTracker:
    def __init__(self):
        self.visited = set()
    def reset(self):
        self.visited.clear()
    def compute_reward(self, prev_pos, new_pos, goal_pos, caught, delivered):
        reward = 0.0
        if delivered: reward += REWARD_DELIVERY
        if caught: reward += PENALTY_CAUGHT
        reward += STEP_PENALTY
        if USE_DISTANCE_SHAPING:
            prev_dist = abs(prev_pos[0]-goal_pos[0]) + abs(prev_pos[1]-goal_pos[1])
            new_dist = abs(new_pos[0]-goal_pos[0]) + abs(new_pos[1]-goal_pos[1])
            if new_dist < prev_dist: reward += DISTANCE_REWARD
        if USE_VISIT_BONUS and new_pos not in self.visited:
            reward += VISIT_BONUS
            self.visited.add(new_pos)
        reward /= REWARD_SCALE
        return reward

# -----------------------------
# Environment
# -----------------------------
class DroneEnv:
    def __init__(self, curriculum=False) :
        self.grid_size = GRID_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.grid_size = GRID_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.curriculum = curriculum
        self.reset()

    def reset(self, episode_idx=0):
        """Reset the environment. If curriculum=True, start closer to goal in early episodes."""
        self.agent_pos, self.goal_pos, self.cop_pos, self.obstacles = initialize_positions(self.grid_size, self.num_obstacles)
        
        # Curriculum: start close to goal initially
        if self.curriculum:
            max_offset = max(1, self.grid_size // 2 - episode_idx // 50)  # smaller offset early
            self.agent_pos = (
                max(0, min(self.grid_size-1, self.goal_pos[0] + random.randint(-max_offset, max_offset))),
                max(0, min(self.grid_size-1, self.goal_pos[1] + random.randint(-max_offset, max_offset)))
            )

        self.carrying_food = True
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        return (self.agent_pos[0], self.agent_pos[1],
                self.cop_pos[0], self.cop_pos[1],
                self.goal_pos[0], self.goal_pos[1],
                int(self.carrying_food))
    def step(self, action_idx, reward_tracker):
        action_names = ["up","down","left","right"]
        reward = 0

        prev_pos = self.agent_pos  # save before moving

        feasible = feasible_actions(self.agent_pos, self.obstacles, self.cop_pos, self.goal_pos, self.grid_size)
        action = None
        if feasible:
            action = action_names[action_idx % len(action_names)]
            if action not in feasible: action = random.choice(feasible)
        dxdy = {"up":(0,-1),"down":(0,1),"left":(-1,0),"right":(1,0)}
        new_pos = self.agent_pos
        if action:
            dx, dy = dxdy[action]
            new_pos_candidate = (self.agent_pos[0]+dx, self.agent_pos[1]+dy)
            if new_pos_candidate not in self.obstacles and 0 <= new_pos_candidate[0] < self.grid_size and 0 <= new_pos_candidate[1] < self.grid_size:
                new_pos = new_pos_candidate
                self.agent_pos = new_pos
        # Move cop
        if COP_BEHAVIOR == "still":
            pass
        elif COP_BEHAVIOR == "stochastic":
            self.cop_pos = move_cop(self.cop_pos, self.obstacles, self.goal_pos, self.grid_size, is_random=True)
        elif COP_BEHAVIOR == "goal_patrol":
            self.cop_pos = move_cop(self.cop_pos, self.obstacles, self.goal_pos, self.grid_size, is_random=False)
        # Collision / delivery
        done = False
        delivered = False
        caught = False
        if self.agent_pos == self.cop_pos:
            done = True
            caught = True
        elif self.agent_pos == self.goal_pos and self.carrying_food:
            reward += REWARD_DELIVERY / REWARD_SCALE
            done = True
            delivered = True
            self.carrying_food = False
            if not IS_GOAL_FIXED:
                self.goal_pos = (random.randint(0,self.grid_size-1), random.randint(0,self.grid_size-1))

        reward += reward_tracker.compute_reward(prev_pos=self.agent_pos, new_pos=new_pos, goal_pos=self.goal_pos, caught=caught, delivered=delivered)
        self.steps += 1
        if self.steps >= MAX_STEPS: done = True
        return self.get_state(), reward, done, delivered

# -----------------------------
# Q-learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self,n_actions=4):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.n_actions = n_actions
    def choose_action(self,state):
        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions-1)
        return int(np.argmax(self.Q[state]))
    def learn(self,state,action,reward,next_state,done):
        q_next = 0 if done else max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma*q_next - self.Q[state][action])

# -----------------------------
# DQN Modules
# -----------------------------
class DQN(nn.Module):
    def __init__(self,input_dim,output_dim):
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
    def __init__(self,input_dim=7,n_actions=4):
        self.model = DQN(input_dim,n_actions).to(device)
        self.target = DQN(input_dim,n_actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(),lr=DQN_LR)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.memory = deque(maxlen=DQN_MEMORY_SIZE)
        self.batch_size = DQN_BATCH_SIZE
        self.n_actions = n_actions
        self.steps = 0
        self.update_freq = 10
    def choose_action(self,state_vec):
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        if random.random() < self.epsilon:
            return random.randint(0,self.n_actions-1)
        q_vals = self.model(state_tensor)
        return int(torch.argmax(q_vals).item())
    def store(self,state,action,reward,next_state,done,delivered):
        self.memory.append((state,action,reward,next_state,done,delivered))
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory,self.batch_size)
        states, actions, rewards, next_states, dones, delivers = zip(*batch)
        # parallel-friendly tensor conversion
        states = torch.from_numpy(np.array(states,dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.array(next_states,dtype=np.float32)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        delivers = torch.FloatTensor(delivers).unsqueeze(1).to(device)
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
# State to vector
# -----------------------------
def state_to_vector(state):
    return np.array([state[0]/GRID_SIZE,state[1]/GRID_SIZE,
                     state[2]/GRID_SIZE,state[3]/GRID_SIZE,
                     state[4]/GRID_SIZE,state[5]/GRID_SIZE,
                     state[6]])

# -----------------------------
# Training per environment (thread)
# -----------------------------
def run_episode(seed, global_agent=None, episode_idx=0):
    # --- Environment setup ---
    env = DroneEnv(curriculum=True)
    random.seed(seed)
    np.random.seed(seed)
    reward_tracker = RewardTracker()
    reward_tracker.reset()

    total_reward = 0
    state = env.reset(episode_idx=episode_idx)
    done = False

    # --- Agent setup ---
    if USE_DQN:
        agent = DQNAgent()
        if global_agent:
            agent.model.load_state_dict(global_agent.model.state_dict())
            agent.target.load_state_dict(global_agent.model.state_dict())
    else:
        agent = QLearningAgent()
        if global_agent:
            agent.Q = global_agent.Q.copy()

    # --- Episode loop ---
    while not done:
        if USE_DQN:
            vec = state_to_vector(state)
            action = agent.choose_action(vec)
        else:
            action = agent.choose_action(state)

        next_state, reward, done, delivered = env.step(action, reward_tracker)
        total_reward += reward

        # --- Learning ---
        if USE_DQN:
            agent.epsilon = max(0.05, agent.epsilon * 0.999)
            next_vec = state_to_vector(next_state)
            agent.store(vec, action, reward, next_vec, int(done), int(delivered))
            agent.learn()
        else:
            agent.learn(state, action, reward, next_state, done)

        state = next_state

    return agent.memory, total_reward, agent, delivered

# -----------------------------
# Merge DQN agents (average params)
# -----------------------------
def merge_dqn_agents(agents):
    state_dicts = [a.model.state_dict() for a in agents]
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.mean(torch.stack([sd[key].float() for sd in state_dicts]),dim=0)
    merged_agent = DQNAgent()
    merged_agent.model.load_state_dict(avg_state_dict)
    merged_agent.target.load_state_dict(avg_state_dict)
    return merged_agent

# -----------------------------
# Merge Q-learning agents
# -----------------------------
def merge_q_agents(agents):
    merged_agent = QLearningAgent()
    all_keys = set(k for a in agents for k in a.Q)
    for k in all_keys:
        merged_agent.Q[k] = np.mean([a.Q.get(k,np.zeros(4)) for a in agents],axis=0)
    return merged_agent

# -----------------------------
# Parallel training loop
# -----------------------------
def train_parallel():
    """Parallel training using threads + batch GPU update."""
    if USE_DQN:
        global_agent = DQNAgent()  # GPU agent
    else:
        global_agent = QLearningAgent()

    for ep_start in range(0, MAX_EPISODES, NUM_THREADS):
        thread_results = []

        # Run episodes in threads
        def worker(seed, episode_idx):
            return run_episode(seed, global_agent=global_agent, episode_idx=episode_idx)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(worker, random.randint(0, 1_000_000), ep_start + t) for t in range(NUM_THREADS)]
            for future in as_completed(futures):
                thread_results.append(future.result())


        # --- Select best agent from batch ---
        best_reward = -float('inf')
        best_agent_model = None
        all_experiences = []
        goal_reached = False
        epsilon = 0

        for experiences, total_reward, agent_obj, delivered in thread_results:
            all_experiences.extend(experiences)
            if total_reward > best_reward:
                best_reward = total_reward
                best_agent_model = agent_obj
                goal_reached = delivered   # take from best agent’s episode
                epsilon = agent_obj.epsilon

        # --- Update global agent ---
        if USE_DQN:
            for state_vec, action, reward, next_vec, done, delivered in all_experiences:
                global_agent.store(state_vec, action, reward, next_vec, done, delivered)
            global_agent.learn()  # batch GPU update
            if best_agent_model:
                # save the best agent’s model, not the global_agent
                torch.save(best_agent_model.model.state_dict(), "best_model02.pt")
        else:
            if best_agent_model:
                global_agent.Q = best_agent_model.Q.copy()
                with open("best_q.pkl", "wb") as f:
                    pickle.dump(global_agent.Q, f)

        print(f"Episodes {ep_start}-{ep_start+NUM_THREADS-1} Best reward: {best_reward:.2f} | Goal Reached: {goal_reached} | Epsilon: {epsilon}")



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    trained_agent = train_parallel()
