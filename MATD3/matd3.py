import copy
import torch.nn.functional as F
import numpy as np
import torch
import MATD3.params as p
from MATD3 import main
from MATD3.actor import Actor
from MATD3.critic import Critic


class MATD3(object):
    """
    Agent class that handles the training of the networks and provides outputs as actions.
    """

    def __init__(self, n_agents=p.num_agents):  # may need this param for later, when doing cooperative
        self.actor = []
        self.actor_target = []
        self.actor_optimizer = []

        self.critic = []
        self.critic_target = []
        self.critic_optimizer = []

        for agent in range(n_agents):
            self.actor.append(Actor(p.state_dim, p.action_dim, p.max_action).to(p.device))
            self.actor_target.append(Actor(p.state_dim, p.action_dim, p.max_action).to(p.device))
            self.actor_target[agent].load_state_dict(self.actor[agent].state_dict())
            self.actor_optimizer.append(torch.optim.Adam(self.actor[agent].parameters(), lr=p.lr))

            self.critic.append(Critic(p.state_dim, p.action_dim).to(p.device))
            self.critic_target.append(Critic(p.state_dim, p.action_dim).to(p.device))
            self.critic[agent].load_state_dict(self.critic[agent].state_dict())
            self.critic_optimizer.append(torch.optim.Adam(self.critic[agent].parameters(), lr=p.lr))

        self.total_iterations = 0

    def select_actions(self, state):
        """
        Selects and appropriate action for each agent from the agent's policy
        :param state: 31 dimensions for state
        :return: nested list num_actors x 4
        """
        actions = []
        state = torch.FloatTensor(state).to(p.device)
        for i in range(p.num_agents):
            action = self.actor[i](state).cpu().data.numpy().flatten()
            # creates Gaussian noise to add to action
            noise = np.random.normal(0, p.max_action * p.exp_noise, size=p.action_dim)
            # clip the actions
            action = (action + noise).clip(-p.max_action, p.max_action)
            actions.append(action)
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
        actions = torch.from_numpy(actions).float().to(p.device)             # torch.Size([256, 3, 4])
        reward = torch.as_tensor(reward, dtype=torch.float32).to(p.device)   # torch.Size([256])
        done = torch.as_tensor(done, dtype=torch.float32).to(p.device)       # torch.Size([256])

        # split the actions into discrete agents
        action_chunk = torch.chunk(actions, p.num_agents, 1)
        for i in range(p.num_agents):
            action = action_chunk[i].squeeze()          # remove the 1 dimension
            with torch.no_grad():
                # get action according to policy and add clipped noise
                noise = (torch.rand_like(action) * p.policy_noise).clamp(-p.noise_clip, p.noise_clip)
                next_action = (self.actor_target[i](next_state) + noise).clamp(-p.max_action, p.max_action)

                # compute the target Q values, minimum of the two values
                target_q1, target_q2 = self.critic_target[i](next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + done * p.discount + target_q

            # get current Q estimates
            current_q1, current_q2 = self.critic[i](state, action)

            # compute critic loss from current to target
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # write the critic loss to the report
            main.reports.write_critic_loss(main.episode_num + 1, main.step, p.agent_names[i], critic_loss)

            # optimize the critic
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()

            # TODO: if using priority replay, need to add weighting
            if p.priority:
                # need to figure out weighting, by reward maybe, must be positive
                # placeholder list of ones
                new_priorities = torch.full((256,), 1)
                replay.update_priorities(indexes, new_priorities)

            # delayed policy updates
            if self.total_iterations % p.policy_freq == 0:
                # compute actor loss
                actor_loss = -self.critic[i].get_q(state, self.actor[i](state)).mean()
                main.reports.write_actor_loss(main.episode_num + 1, main.step, p.agent_names[i], actor_loss)

                # optimize the actor
                self.actor_optimizer[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizer[i].step()

                # update the frozen target models
                for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
                                        target_param.data.copy_(p.tau * param.data + (1 - p.tau) * target_param.data)
                for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                                        target_param.data.copy_(p.tau * param.data + (1 - p.tau) * target_param.data)

    def save(self):
        for i in range(p.num_agents):
            torch.save(self.critic[i].state_dict(),
                       'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_critic.pth')
            torch.save(self.critic_optimizer[i].state_dict(),
                       'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_critic_optimizer.pth')

            torch.save(self.actor[i].state_dict(),
                       'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_actor.pth')
            torch.save(self.actor_optimizer[i].state_dict(),
                       'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_actor_optimizer.pth')

    def load(self):
        for i in range(p.num_agents):
            self.critic[i].load_state_dict(torch.load(
                'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_critic.pth'))
            self.critic_optimizer[i].load_state_dict(torch.load(
                'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_critic_optimizer.pth'))
            self.critic_target[i] = copy.deepcopy(self.critic[i])

            self.actor[i].load_state_dict(torch.load(
                'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_agent.pth'))
            self.actor_optimizer[i].load_state_dict(
                'models/policy_{}_'.format(p.timestr) + p.agent_names[i] + '_actor_optimizer.pth')
            self.actor_target[i] = copy.deepcopy(self.actor[i])
