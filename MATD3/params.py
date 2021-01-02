from pettingzoo.sisl import multiwalker_v5

# ----------------------------------------------------------------------------------------------------------------------
# SISL Environment setup
# ----------------------------------------------------------------------------------------------------------------------
multiwalker_env = multiwalker_v5.env(n_walkers=3,
                                     position_noise=1e-3,
                                     angle_noise=1e-3,
                                     local_ratio=1.0,
                                     forward_reward=1.0,
                                     terminate_reward=-100.0,
                                     fall_reward=-10.0,
                                     terminate_on_fall=True,
                                     remove_on_fall=True,
                                     max_cycles=500)
# assuming identical agents, just get the first one's info
state_dim = multiwalker_env.observation_spaces.get('walker_0').shape[0]
action_dim = multiwalker_env.action_spaces.get('walker_0').shape[0]
max_action = multiwalker_env.action_spaces.get('walker_0').high[0]


render = True                           # render the environment
# ----------------------------------------------------------------------------------------------------------------------
# TD3 algorithm parameters
# ----------------------------------------------------------------------------------------------------------------------
policy_noise = 0.2 * max_action         # noise added for critic update
noise_clip = 0.5 * max_action           # range to clip the noise for policy
policy_freq = 2                         # how frequently delayed policy updates occur
observation = 10000                     # number of random observations to pre-populate the replay
exploration = 5e6                       # number of training steps to run
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
