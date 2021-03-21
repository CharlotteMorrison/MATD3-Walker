from pettingzoo.sisl import multiwalker_v5
import torch
from datetime import datetime
from MATD3.graphs import Graphs
from MATD3.reports import Reports

# ----------------------------------------------------------------------------------------------------------------------
# Project parameters
# ----------------------------------------------------------------------------------------------------------------------
seed = 42                           # seed value for numpy, env, torch
# torch.manual_seed(seed)             # set torch seed, currently unused
# numpy.random.seed(seed)             # set numpy seed, currently unused
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# write csv reports
write_reports = True
write_graphs = True
show_graphs = False
# global date/time for file naming
timestr = datetime.now().strftime("%d-%b-%y_%H-%M")
eval_frequency = 5e3                # current evaluation every 5000 timesteps, param from td3 paper
save_model = True                   # save the model and optimizer params
# mode selection for run: centralized, concurrent, parameter sharing
mode = 'concurrent'

if write_reports:
    reports = Reports()
if write_graphs:
    graphs = Graphs()

# global variables for tracking current episode, step
step = 0
episode = 0
# ----------------------------------------------------------------------------------------------------------------------
# SISL Environment setup
# ----------------------------------------------------------------------------------------------------------------------
num_agents = 3
multiwalker_env = multiwalker_v5.env(n_walkers=num_agents,
                                    position_noise=1e-3,
                                    angle_noise=1e-3,
                                    local_ratio=1.0,
                                    forward_reward=1.0,
                                    terminate_reward=-100.0,
                                    fall_reward=-10.0,
                                    terminate_on_fall=True,
                                    remove_on_fall=True,
                                    max_cycles=500)
multiwalker_env.seed(seed)              # set the environment seed
agent_names = ["walker_" + str(num) for num in range(num_agents)]
# assuming identical agents, just get the first one's info
state_dim = multiwalker_env.observation_spaces.get('walker_0').shape[0]
action_dim = multiwalker_env.action_spaces.get('walker_0').shape[0]
max_action = multiwalker_env.action_spaces.get('walker_0').high[0]

render = False                           # render the environment
# ----------------------------------------------------------------------------------------------------------------------
# TD3 algorithm parameters
# ----------------------------------------------------------------------------------------------------------------------
policy_noise = 0.2 * max_action         # noise added for critic update
noise_clip = 0.5 * max_action           # range to clip the noise for policy
policy_freq = 2                         # how frequently delayed policy updates occur
discount = 0.99                         # discount factor
tau = 0.005                             # target network update rate
lr = 3e-4                               # learning rate
obs_timesteps = 10000                   # number of random observations to pre-populate the replay
max_timsteps = 3000000                  # number of training steps to run
exp_noise = 0.1                         # Std of Gaussian exploration noise
batch_size = 256                        # size of the sample from experience replay
# ----------------------------------------------------------------------------------------------------------------------
# replay memory parameters
# ----------------------------------------------------------------------------------------------------------------------
priority = False    # true runs with priority replay
buffer_size = 1e6   # size of the replay memory
alpha = .6          # alpha param for priority replay buffer
beta = .4           # initial value of beta
beta_iters = None   # number of iterations over which beta will be annealed from initial value
eps = 1e-6          # epsilon to add to the TD errors when updating priorities
beta_sched = None   # do you want manually scheduled beta
