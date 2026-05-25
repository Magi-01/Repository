# train_agent.py

from mdp_ticketworld import TicketMDP

class TrainAgent:
    def __init__(self, id, mdp):
        self.id = id
        self.mdp = mdp
        self.current_state = None

    def act(self, train, global_state):
        self.current_state = global_state
        action = self.mdp.policy_for_train(global_state, self.id)
        print(f"[DEBUG] Agent {self.id} at train pos {train['pos']} selects action: {action}")
        return action, f"Train {self.id} action: {action}"

    def receive_reward(self, reward):
        # Optional: track total rewards
        print(f"[DEBUG] Agent {self.id} received reward: {reward}")
