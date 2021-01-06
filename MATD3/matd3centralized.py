import copy
import torch.nn.functional as F
import numpy as np
import torch
import MATD3.params as p
from MATD3.actor import Actor
from MATD3.critic import Critic


class MATD3Centralized(object):
    """
    Agent class that handles the training of the networks and provides outputs as actions.
    """

    def __init__(self):  # may need this param for later, when doing cooperative
        self.n_agents = p.num_agents
        self.action_dim = p.num_agents * p.action_dim

        self.actor = Actor(p.state_dim, self.action_dim, p.max_action).to(p.device)
        self.actor_target = Actor(p.state_dim, self.action_dim, p.max_action).to(p.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=p.lr)

        self.critic = Critic(p.state_dim, self.action_dim).to(p.device)
        self.critic_target = Critic(p.state_dim, self.action_dim).to(p.device)
        self.critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=p.lr)

        self.total_iterations = 0

    def select_actions(self, state):
        """
        Selects and appropriate action for each agent from the agent's policy
        :param state: 31 dimensions for state
        :return: nested list num_actors x 4
        """

        state = torch.FloatTensor(state).to(p.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # creates Gaussian noise to add to action
        noise = np.random.normal(0, p.max_action * p.exp_noise, size=self.action_dim)
        # clip the actions
        action = (action + noise).clip(-p.max_action, p.max_action)
        actions = action.reshape(self.n_agents, 4)  # rows, columns
        return actions

    def train(self, replay, batch_size=100):
        self.total_iterations += 1

        # get a sample from the replay buffer (priority buffer option)
        if p.priority:
            state, actions, reward, next_state, done, weights, indexes = replay.sample(batch_size,
                                                                                       beta=p.beta_sched.value(self.total_iterations))
        else:
            state, actions, reward, next_state, done = replay.sample(batch_size)
            indexes = [0]   # just to remove the annoying pycharm warning.

        # convert to tensors and send to gpu                                 # batch size 256 sample
        state = torch.from_numpy(state).float().to(p.device)                 # torch.Size([256, 31])
        next_state = torch.from_numpy(next_state).float().to(p.device)       # torch.Size([256, 31])
        actions = torch.from_numpy(actions).float().to(p.device)             # torch.Size([256, n_agents, 4])
        reward = torch.as_tensor(reward, dtype=torch.float32).to(p.device)   # torch.Size([256])
        done = torch.as_tensor(done, dtype=torch.float32).to(p.device)       # torch.Size([256])

        # split the actions into discrete agents
        action = torch.flatten(actions, start_dim=1)
        with torch.no_grad():
            # get action according to policy and add clipped noise
            noise = (torch.rand_like(action) * p.policy_noise).clamp(-p.noise_clip, p.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-p.max_action, p.max_action)

            # compute the target Q values, minimum of the two values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(1) + done.unsqueeze(1) * p.discount * target_q

        # get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # compute critic loss from current to target
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # write the critic loss to the report, graphs
        # update values from main- this is janky- rework later.
        if p.write_reports:
            p.reports.write_critic_report(p.episode + 1, p.step, 'combined agents', critic_loss)
        if p.write_graphs:
            p.graphs.critic_list.append([p.episode + 1, p.step, 'combined agents', critic_loss.item()])

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # TODO: if using priority replay, need to add weighting
        if p.priority:
            # need to figure out weighting, by reward maybe, must be positive
            # placeholder list of ones
            new_priorities = torch.full((256,), 1)
            replay.update_priorities(indexes, new_priorities)

        # delayed policy updates
        if self.total_iterations % p.policy_freq == 0:
            # compute actor loss
            actor_loss = -self.critic.get_q(state, self.actor(state)).mean()

            # update values from main- this is janky- rework later.
            if p.write_reports:
                p.reports.write_actor_report(p.episode + 1, p.step, 'centralized', actor_loss)
            if p.write_graphs:
                p.graphs.actor_list.append([p.episode + 1, p.step, 'centralized', actor_loss.item()])

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(p.tau * param.data + (1 - p.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(p.tau * param.data + (1 - p.tau) * target_param.data)

    def save(self):
        torch.save(self.critic.state_dict(), 'models/centralized-policy_{}_'.format(p.timestr) + '_critic.pth')
        torch.save(self.critic_optimizer.state_dict(), 'models/centralized-policy_{}_'.format(p.timestr) +
                   '_critic_optimizer.pth')
        torch.save(self.actor.state_dict(), 'models/centralized-policy_{}_'.format(p.timestr) + '_actor.pth')
        torch.save(self.actor_optimizer.state_dict(), 'models/centralized-policy_{}_'.format(p.timestr) +
                   '_actor_optimizer.pth')

    def load(self):
        self.critic.load_state_dict(torch.load('models/centralized-policy_{}_'.format(p.timestr) + '_critic.pth'))
        self.critic_optimizer.load_state_dict(torch.load('models/centralized-policy_{}_'.format(p.timestr) +
                                                        '_critic_optimizer.pth'))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load('models/centralized-policy_{}_'.format(p.timestr) + '_agent.pth'))
        self.actor_optimizer.load_state_dict(torch.load('models/centralized-policy_{}_'.format(p.timestr) +
                                                        '_actor_optimizer.pth'))
        self.actor_target = copy.deepcopy(self.actor)
