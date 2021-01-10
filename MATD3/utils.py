from pettingzoo.sisl import multiwalker_v5
import numpy as np
import MATD3.params as p


def eval_policy(policy, env_name='multiwalker_v5', seed=42, eval_episodes=10):
    """
    Runs the policy for eval_episode episodes
    :param policy: current policy
    :param env_name: petting zoo environment, default multiwalker_v5
    :param seed: seed used for reproducibility of measurement
    :param eval_episodes: number of episodes to run evaluation
    :return: the average reward of the evaluation
    """
    eval_env = multiwalker_v5.env()

    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        eval_env.reset()
        state, reward, done, _ = eval_env.last()
        while not done:
            actions = policy.select_actions(state)
            for i in range(p.num_agents):
                eval_env.step(actions[i])
            state, reward, done, _ = eval_env.last()
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes at timestep {p.step}: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
