import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# From: https://github.com/sfujim/TD3/blob/master/TD3.py

# Original paper parameters noted in comments where changes were made.


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """Create the Actor neural network layers."""
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)     # (state_dim, 400)
        self.l2 = nn.Linear(400, 300)           # (256, 256)
        self.l3 = nn.Linear(300, action_dim)    # (256, max_action)

        self.max_action = max_action

    def forward(self, state):
        """Forward pass in Actor neural network."""

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x
