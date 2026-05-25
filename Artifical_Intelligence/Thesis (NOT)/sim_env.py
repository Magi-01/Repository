from datetime import datetime

def simulate_action(item, action):
    now = datetime.now()
    arrival_time = item['arrival_time']
    if isinstance(arrival_time, str):
        arrival_time = datetime.strptime(arrival_time, '%Y-%m-%d %H:%M:%S')
    elapsed = (now - arrival_time).total_seconds()
    overdue = elapsed > item['deadline_seconds']

    if action == 'drop':
        reward = 10 if not overdue else -5
        success = not overdue
    elif action == 'pick':
        reward = -0.5
        success = False
    else:
        reward = -1
        success = False

    return reward, success
