import time
import MATD3.params as p
from MATD3.replay.priority_replay_buffer import PrioritizedReplayBuffer
from MATD3.replay.replay_buffer import ReplayBuffer
from MATD3.replay.schedules import LinearSchedule
from MATD3.matd3concurrent import MATD3Concurrent
from MATD3.matd3centralized import MATD3Centralized
from MATD3.matd3paramshare import MATD3ParamShare
from MATD3.utils import eval_policy

if __name__ == "__main__":

    env = p.multiwalker_env
    env.reset()
    if p.render:
        env.render()

    # initialize policy
    if p.mode == 'centralized':
        policy = MATD3Centralized()
    elif p.mode == 'concurrent':
        policy = MATD3Concurrent()
    elif p.mode == 'param-share':
        policy = MATD3ParamShare()
    else:
        policy = None
        print('not a valid policy')

    # initialize replay
    if p.priority:
        replay_buffer = PrioritizedReplayBuffer(p.buffer_size, alpha=p.alpha)
        if p.beta_iters is None:
            p.beta_iters = p.max_timsteps    # if no value for annealing specified, anneal over all exploration
        p.beta_sched = LinearSchedule(p.beta_iters, initial_p=p.beta, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(p.buffer_size)

    evaluations = [eval_policy(policy)]

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
    episode_start_time = time.time()

    # iterate through training
    for step in range(p.max_timsteps):
        episode_timesteps += 1
        p.step += 1                  # fix this iterator if you have energy

        # select random action for first obs_timesteps, then according to policy
        if p.step < p.obs_timesteps:
            _, _, done, _ = env.last()
            if done:    # if done, close render window, reset the environment, wait to allow processes to finish
                if not p.render:
                    env.close()
                env.reset()
                time.sleep(2)
            else:       # else get random action for exploration
                for agent in p.agent_names:
                    actions[int(agent[-1:])] = env.action_spaces[agent].sample()
            if p.render:
                env.render()
        else:           # if exploration complete, then choose actions via policy
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
        if p.step >= p.obs_timesteps:
            policy.train(replay_buffer, p.batch_size)

        # get elapsed_time, write step information to .csv file
        elapsed_time = time.time() - start_time
        if p.write_reports:
            p.reports.write_step_report(p.episode + 1, p.step, reward, done, elapsed_time)

        # add step to graph data
        if p.write_graphs:
            p.graphs.step_list.append([p.episode, p.step, reward, done, elapsed_time])

        if done:        # at the end of the episode, print console info for monitoring
            avg_reward = round(sum(episode_reward)/len(episode_reward), 4)
            sum_reward = round(sum(episode_reward), 4)
            episode_elapsed_time = time.time() - episode_start_time

            # add one to the step, episode numbers to deal with zero indexing
            print("\n********** Episode {} ***********".format(p.episode + 1))
            print("Episode Steps: {}".format(episode_timesteps))
            print("Average Reward: {}".format(avg_reward))
            print("Sum of Rewards: {}".format(sum_reward))
            print("Total Timesteps: {}".format(p.step + 1))
            print("Total elapsed Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            print("Episode elapsed Time: {}".format(time.strftime("%H:%M:%S", time.gmtime(episode_elapsed_time))))

            # update the graphs at the end of the episode
            if p.write_graphs:
                p.graphs.update_step_list_graphs()

            # if the total reward is better than best, save new model, REMOVED - using periodic evaluations
            '''
            if sum_reward > best_episode_reward:   
                best_episode_reward = sum_reward
                policy.save()
            '''
            # reset the environment
            if p.render:
                env.close()
            env.reset()
            time.sleep(2)
            state, _, done, _ = env.last()

            # reset episode counters
            episode_reward = []
            episode_timesteps = 0
            p.episode += 1
            episode_start_time = time.time()

        # run the policy evaluation
        if step % p.eval_frequency == 0:
            evaluation = eval_policy(policy)
            evaluations.append(evaluation)                                  # append the new evaluation
            if p.write_reports:                                             # write the evaluation to report
                p.reports.write_evaluate_step(p.episode, step, evaluation)
            if p.write_graphs:                                              # update the evaluation graph
                p.graphs.evaluate_list.append(evaluations)
            if p.save_model:                                                # save the policy
                policy.save()

    if p.write_reports:
        p.reports.write_final_values()        # reports written in a batch, make sure final batch is written
    if p.write_graphs:
        p.graphs.update_step_list_graphs()    # at the end do a final update of the graphs
