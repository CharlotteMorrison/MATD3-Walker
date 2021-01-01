import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# From: https://github.com/sfujim/TD3/blob/master/TD3.py

# Original paper parameters noted in comments where changes were made.


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Implements a pair of Critic neural networks."""
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)    # (state_dim, action_dim, 256)
        self.l2 = nn.Linear(400, 300)                       # (256, 256)
        self.l3 = nn.Linear(300, 1)                         # (256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)    # (state_dim, action_dim, 256)
        self.l5 = nn.Linear(400, 300)                       # (256, 256)
        self.l6 = nn.Linear(300, 1)                         # (256, 1)

    def forward(self, state, action):
        """Returns the Q values for both Critic networks"""
        sa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def get_q(self, state, action):
        """Returns the Q value for only Critic 1"""
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1
