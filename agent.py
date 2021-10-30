from dataclasses import dataclass
from typing import Any
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np


@dataclass
class Params:
    batch_size: int
    gamma: float
    tau: float
    learning_rate: float
    state_dim: int
    action_cnt: int
    target_update_step: int
    epsilon: float


@dataclass
class Experience:
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Agent interactions with a Unity environment.
    
    Params:
        params (Params): Agent parameter dataclass. 

    Attributes:
        policy_dqn (Torch.Module):
        target_dqn (Torch.Module):
        optimizer (Optimizer):
        step_cnt (int): 
    
    """

    def __init__(self, params):
        self.params = params

        # setup neural net
        self.policy_dqn = model.DeepQNetwork(self.params.state_dim, self.params.action_cnt, 128).to(device)
        self.target_dqn = model.DeepQNetwork(self.params.state_dim, self.params.action_cnt, 128).to(device)
        self.optimizer = optim.RMSprop(self.policy_dqn.parameters(), lr=self.params.learning_rate)

        self.step_cnt = 0
        self.buffer = []

    def step(self, state, action, reward, next_state, done):
        # add experience into replay buffer
        self.buffer.append(Experience(state, action, reward, next_state, done))

        # update step cnt
        self.step_cnt += 1

        # update target network after target_update_step_size and if replay buffer has enough data
        if self.step_cnt % self.params.target_update_step == 0 and len(self.buffer) > self.params.batch_size:
            self.learn()

    def act(self, state):
        # update policy network
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_dqn.eval()
        with torch.no_grad():
            action_values = self.policy_dqn(state)
        self.policy_dqn.train()

        # Epsilon greedy action selection
        if random.random() > self.params.epsilon:
            self.params.epsilon = max(0.01, 0.999 * self.params.epsilon) # decrease epsilon
            return np.argmax(action_values.cpu().data.numpy())
        else:
            self.params.epsilon = max(0.01, 0.999 * self.params.epsilon) # decrease epsilon
            return random.randrange(self.params.action_cnt)

    def learn(self):
        # sample batch from replay buffer
        experience_batch = random.sample(self.buffer, k=self.params.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experience_batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience_batch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience_batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience_batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience_batch]).astype(np.uint8)).float().to(device)

        # update target network
        # Get max predicted Q values (for next states) from target model
        targets_next = self.target_dqn(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        targets = rewards + (self.params.gamma * targets_next * (1 - dones))

        # Get expected Q values from local model
        expected = self.policy_dqn(states).gather(1, actions)

        # Compute mean squared error
        loss = F.mse_loss(expected, targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # copy policy weights to target_dqn
        for target_param, local_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
            target_param.data.copy_(self.params.tau*local_param.data + (1.0-self.params.tau) * target_param.data)