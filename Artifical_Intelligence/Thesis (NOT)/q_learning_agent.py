import random
from db_utils import update_item_status
from datetime import datetime

# --- Q-learning parameters ---
ACTIONS = ['up', 'down', 'left', 'right', 'pick', 'drop']
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

# --- State & Q-tables ---
AGENT_STATE = {}       # agent_id -> {'carrying_item': None, 'target_bin': None, 'target_item_id': None}
Q_TABLES = {}          # agent_id -> {(agent_x, agent_y, target_x, target_y, carrying): {action: q_value}}

# --- Initialize agent ---
def init_agent(agent_id):
    AGENT_STATE[agent_id] = {
        'carrying_item': None,
        'target_bin': None,
        'target_item_id': None
    }
    Q_TABLES[agent_id] = {}

# --- Helper to get Q-value ---
def get_q(agent_id, state_tuple, action):
    return Q_TABLES[agent_id].get(state_tuple, {}).get(action, 0.0)

# --- Helper to set Q-value ---
def set_q(agent_id, state_tuple, action, value):
    if state_tuple not in Q_TABLES[agent_id]:
        Q_TABLES[agent_id][state_tuple] = {}
    Q_TABLES[agent_id][state_tuple][action] = value

# --- Choose action using epsilon-greedy ---
def choose_action(agent_id, agent_pos, target_pos, carrying):
    state_tuple = (agent_pos[0], agent_pos[1], target_pos[0], target_pos[1], carrying)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_vals = {a: get_q(agent_id, state_tuple, a) for a in ACTIONS}
    max_q = max(q_vals.values())
    best_actions = [a for a, q in q_vals.items() if q == max_q]
    return random.choice(best_actions)

# --- Simulate movement action ---
def simulate_move(agent_pos, action, GRID_WIDTH, GRID_HEIGHT):
    x, y = agent_pos
    if action == 'up' and y > 0:
        y -= 1
    elif action == 'down' and y < GRID_HEIGHT - 1:
        y += 1
    elif action == 'left' and x > 0:
        x -= 1
    elif action == 'right' and x < GRID_WIDTH - 1:
        x += 1
    # 'pick' and 'drop' do not move the agent
    return (x, y)

# --- Process item pickup/drop and update Q-table ---
def process_item(agent_id, item, action_type):
    """
    action_type: 'pick' or 'drop'
    """
    carrying = action_type == 'drop'
    agent_state = AGENT_STATE[agent_id]
    agent_pos = agent_state.get('pos', (0,0))
    target_pos = (agent_state['target_bin'][0], agent_state['target_bin'][1])
    state_tuple = (agent_pos[0], agent_pos[1], target_pos[0], target_pos[1], carrying)

    # Reward assignment
    reward = -0.1  # step penalty
    if action_type == 'pick':
        reward += 5      # reward for picking item
    elif action_type == 'drop':
        reward += 10     # reward for successful drop
        success = True
        update_item_status(item['item_id'], item['correct_bin'], True)
    else:
        success = False

    # Deadline penalty
    deadline_passed = (datetime.now() - item['arrival_time']).seconds > item['deadline_seconds']
    if deadline_passed:
        reward -= 50

    # Q-table update
    next_state_tuple = state_tuple  # or new state if you want
    old_q = get_q(agent_id, state_tuple, action_type)
    next_max_q = max(get_q(agent_id, next_state_tuple, a) for a in ACTIONS)
    new_q = old_q + ALPHA * (reward + GAMMA * next_max_q - old_q)
    set_q(agent_id, state_tuple, action_type, new_q)

    return action_type, success, reward
