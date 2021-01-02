import time
import MATD3.params as p
from MATD3.matd3 import MATD3
from MATD3.replay.priority_replay_buffer import PrioritizedReplayBuffer
from MATD3.replay.replay_buffer import ReplayBuffer
from MATD3.replay.schedules import LinearSchedule
from MATD3.reports import Reports

if __name__ == "__main__":

    env = p.multiwalker_env
    env.reset()
    if p.render:
        env.render()

    # initialize the reports
    reports = Reports()

    # initialize policy
    policy = MATD3(p.num_agents)

    # initialize replay
    if p.priority:
        replay_buffer = PrioritizedReplayBuffer(p.buffer_size, alpha=p.alpha)
        if p.beta_iters is None:
            p.beta_iters = p.max_timsteps    # if no value for annealing specified, anneal over all exploration
        p.beta_sched = LinearSchedule(p.beta_iters, initial_p=p.beta, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(p.buffer_size)

    # get the start time for the program run
    start_time = time.time()

    # get starting state
    state, reward, done, _ = env.last()

    # initialize actions list to use indexing for actions
    actions = [[0] * 4 for num in range(p.num_agents)]

    # initialize the best reward
    best_episode_reward = - 100

    # set episode variables
    episode_reward = []
    episode_timesteps = 0
    episode_num = 0
    episode_start_time = time.time()

    # iterate through training
    for step in range(p.max_timsteps):
        episode_timesteps += 1

        # select random action for first obs_timesteps, then according to policy
        # TODO add a max episode length if needed
        if step < p.obs_timesteps:
            _, _, done, _ = env.last()
            if done:
                if not p.render:
                    env.close()
                env.reset()
                time.sleep(2)
            else:
                for agent in p.agent_names:
                    actions[int(agent[-1:])] = env.action_spaces[agent].sample()
            if p.render:
                env.render()
        # if exploration complete, then choose actions via policy
        else:
            actions = policy.select_actions(state)
        # perform the action for each agent, use the final reward for the cooperative task
        # the reward will be the distance moved after all n walkers move
        for i in range(p.num_agents):
            env.step(actions[i])

        # get the values after the step
        next_state, reward, done, info = env.last()

        # store the step in the replay buffer
        replay_buffer.add(state, actions, reward, next_state, done)

        # set the old state to the new state
        state = next_state

        # store the episode reward
        episode_reward.append(reward)

        # start training if exploration is complete
        if step >= p.obs_timesteps:
            policy.train(replay_buffer, p.batch_size)

        elapsed_time = time.time() - start_time
        if p.write_reports:
            reports.write_step_report(episode_num + 1, step, reward, done, elapsed_time)

        if done:
            avg_reward = round(sum(episode_reward)/len(episode_reward), 4)
            sum_reward = round(sum(episode_reward), 4)
            episode_elapsed_time = time.time() - episode_start_time

            # print out some stuff....
            # add one to the step, episode numbers to deal with zero indexing
            print("\n********** Episode {} ***********".format(episode_num + 1))
            print("Episode Steps: {}".format(episode_timesteps))
            print("Average Reward: {}".format(episode_timesteps, avg_reward))
            print("Sum of Rewards: {}".format(episode_timesteps, sum_reward))
            print("Total Timesteps: {}".format(step + 1))
            print("Total elapsed Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            print("Episode elapsed Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(episode_elapsed_time))))

            # if the total reward is better than best, save new model
            if sum_reward > best_episode_reward:
                best_episode_reward = sum_reward
                policy.save()

            # reset the environment
            if p.render:
                env.close()
            env.reset()
            time.sleep(2)
            state, _, done, _ = env.last()

            # reset episode counters
            episode_reward = []
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = time.time()

    reports.write_final_values()    # reports written in a batch, make sure final batch is written
