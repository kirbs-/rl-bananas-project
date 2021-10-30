import torch
import torch.nn as nn

class DeepQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_cnt, num_of_neurons):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_of_neurons (int): Number of neurons in each fully connected layer
        """
        super(DeepQNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, num_of_neurons),
            nn.ReLU(),
            nn.Linear(num_of_neurons, num_of_neurons),
            nn.ReLU(),
            nn.Linear(num_of_neurons, action_cnt)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.linear_relu_stack(state)