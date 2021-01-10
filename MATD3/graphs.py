import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import MATD3.params as p


class Graphs:
    def __init__(self):
        # setup styling for graphs
        sns.set()
        sns.set_style('whitegrid')
        plt.style.use('seaborn')

        self.mode = p.mode

        # temp storage for values, to costly to append data to frame... don't do it
        self.step_list = []
        self.actor_list = []
        self.critic_list = []
        self.evaluate_list = []

        self.datafile = pd.DataFrame()
        self.episode_df = pd.DataFrame()
        self.actor_df = pd.DataFrame()
        self.critic_df = pd.DataFrame()
        # display graphs while running... don't use except for debugging
        self.show_graphs = False

    def update_step_list_graphs(self):
        # load the datafile
        plt.close("all")
        headers = ["episode", "step", "reward", "solved", "time_elapsed"]
        self.datafile = pd.DataFrame(self.step_list, columns=headers)

        # datafile group by episode
        self.episode_df = self.datafile.groupby(['episode']).mean()

        # actor/critic datafiles

        self.actor_df = pd.DataFrame(self.actor_list, columns=['episode', 'step', 'agent', 'actor_loss'])
        self.critic_df = pd.DataFrame(self.critic_list, columns=['episode', 'step', 'agent', 'critic_loss'])

        # run the graphs
        self.avg_reward_episode()

        # include a rolling average for 10, 100, and 1000 episodes
        self.rolling_average_reward()
        self.episode_length()

        # graph actor/critic after initial observation
        if p.step > p.obs_timesteps:
            self.avg_actor_loss()
            self.avg_critic_loss()

    def avg_reward_episode(self):
        # average reward for each episode
        plt.plot(self.episode_df['reward'], label='Average Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('plots/{}_episode_average_reward_{}.png'.format(self.mode, p.timestr))
        if self.show_graphs:
            plt.show()
        plt.close()

    def rolling_average_reward(self):
        # rolling average of the reward
        rolling_avg_10 = self.episode_df['reward'].rolling(10).mean()
        rolling_avg_100 = self.episode_df['reward'].rolling(100).mean()
        rolling_avg_1000 = self.episode_df['reward'].rolling(1000).mean()
        fig, ax = plt.subplots(4, figsize=(12, 12), sharey=True)

        ax[0].plot(rolling_avg_10, label='window=10')
        next(ax[1]._get_lines.prop_cycler)
        ax[1].plot(rolling_avg_100, label='window=100')
        next(ax[2]._get_lines.prop_cycler)
        next(ax[2]._get_lines.prop_cycler)
        ax[2].plot(rolling_avg_1000, label='window=1000')
        ax[3].plot(rolling_avg_10, label='window=10')
        ax[3].plot(rolling_avg_100, label='window=100')
        ax[3].plot(rolling_avg_1000, label='window=1000')

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()

        plt.savefig('plots/{}_reward_rolling_average_{}.png'.format(self.mode, p.timestr))
        if self.show_graphs:
            plt.show()
        plt.close()

    def episode_length(self):
        # counts the number of steps in each episode
        plt.plot(self.datafile.groupby(['episode']).count()['step'], label='Length of Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps In Episode')
        plt.legend()
        plt.savefig('plots/{}_episode_length_{}.png'.format(self.mode, p.timestr))
        if self.show_graphs:
            plt.show()
        plt.close()

    def avg_actor_loss(self):
        # agents = self.actor_df.agent.unique()
        plt.plot(self.actor_df.groupby(['episode']).mean()['actor_loss'], label="Actor 1 Loss")
        plt.xlabel('Episode')
        plt.ylabel('Actor Loss')
        plt.legend()
        plt.savefig('plots/{}_actor_loss_plot_{}.png'.format(self.mode, p.timestr))
        if p.show_graphs:
            plt.show()
        plt.close()

    def avg_critic_loss(self):
        plt.plot(self.critic_df.groupby(['episode']).mean()['critic_loss'], label="Critic 1 Loss")
        plt.xlabel('Episode')
        plt.ylabel('Critic Loss')
        plt.legend()
        plt.savefig('plots/{}_critic_loss_plot_{}.png'.format(self.mode, p.timestr))
        if self.show_graphs:
            plt.show()
        plt.close()

    def avg_evaluation_reward(self):
        evaluate_df = pd.DataFrame(self.evaluate_list, columns=['episode', 'step', 'reward'])
        plt.plot(evaluate_df['episode'], evaluate_df['reward'], label='Avg Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig('plots/{}_evaluation_{}.png'.format(self.mode, p.timestr))
        if self.show_graphs:
            plt.show()
        plt.close()
