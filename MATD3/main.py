from pettingzoo.sisl import multiwalker_v5
import MATD3.params as p
from MATD3.replay.priority_replay_buffer import PrioritizedReplayBuffer
from MATD3.replay.replay_buffer import ReplayBuffer
from MATD3.replay.schedules import LinearSchedule

if __name__ == "__main__":

    env = p.multiwalker_env
    env.reset()
    if p.render:
        env.render()

    if p.priority:
        replay_buffer = PrioritizedReplayBuffer(p.buffer_size, alpha=p.alpha)
        if p.beta_iters is None:
            p.beta_iters = p.exploration    # if no value for annealing specified, anneal over all exploration
        p.beta_sched = LinearSchedule(p.beta_iters, initial_p=p.beta, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(p.buffer_size)

    print(p.noise_clip)
    print(p.policy_noise)
    print(p.max_action)