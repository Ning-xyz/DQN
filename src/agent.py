import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import DQN

class DQNAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, batch_size=32):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")
        
        self.policy_net = DQN(input_shape, n_actions).to(self.device).float()
        self.target_net = DQN(input_shape, n_actions).to(self.device).float()
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, epsilon):

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():

                state_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()

    def learn(self, buffer):
        if len(buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)


        curr_q = self.policy_net(states).gather(1, actions)
        

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
