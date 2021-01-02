import pettingzoo.tests.api_test as api_test
import pettingzoo.tests.bombardment_test as bombardment_test
import pettingzoo.tests.performance_benchmark as performance_benchmark
from pettingzoo.utils import random_demo
from pettingzoo.sisl import multiwalker_v5
import random


if __name__ == "__main__":
    env = multiwalker_v5.env()
    env.reset()
    env.render()
    total_reward = 0
    done = False
    for agent in env.agent_iter():
        obs, reward, done, _ = env.last()
        total_reward += reward
        if done:
            action = None
        elif 'legal_moves' in env.infos[agent]:
            action = random.choice(env.infos[agent]['legal_moves'])
        else:
            action = env.action_spaces[agent].sample()
        env.step(action)
    print("Total reward", total_reward, "done", done)

    env.close()
    '''
        # API Test
        env = multiwalker_v5.env()
        api_test.api_test(env, render=True, verbose_progress=True)
    
        # doesn't pass the bombardment.
        # bombardment_test.bombardment_test(env, cycles=1000)
    
        # tests performance
        performance_benchmark.performance_benchmark(env)
    
        random_demo(env)
    '''
