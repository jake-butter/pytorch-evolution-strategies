import torch.nn as nn
from torch.distributions.categorical import Categorical


class ESPolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=40, fc2_dims=40):
        super(ESPolicyNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        x = self.layers(state)
        dist = Categorical(x)
        return dist
