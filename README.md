# MATD3-Walker
Multi-Agent TD3 Cooperative Learning 

### Walker Environment
[PettingZoo](https://github.com/PettingZoo-Team/PettingZoo)

@article{terry2020pettingzoo,
  Title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
  Author = {Terry, Justin K and Black, Benjamin and Jayakumar, Mario and Hari, Ananth and Santos, Luis and Dieffendahl, Clemens and Williams, Niall L and Lokesh, Yashas and Sullivan, Ryan and Horsch, Caroline and Ravi, Praveen},
  journal={arXiv preprint arXiv:2009.14471},
  year={2020}
}

### SISL Environments

    n_walkers: number of bipedal walker agents in environment
    position_noise: noise applied to agent positional sensor observations
    angle_noise: noise applied to agent rotational sensor observations
    local_ratio: Proportion of reward allocated locally vs distributed among all agents
    forward_reward: reward applied for an agent standing, scaled by agent’s x coordinate
    fall_reward: reward applied when an agent falls down
    terminate_reward: reward applied to a walker for failing the environment
    terminate_on_fall: toggles whether agent is done if it falls down
    remove_on_fall: Remove walker when it falls (only does anything when terminate_on_fall is False)
    max_cycles: after max_cycles steps all agents will return done
    

@inproceedings{gupta2017cooperative,
  title={Cooperative multi-agent control using deep reinforcement learning},
  author={Gupta, Jayesh K and Egorov, Maxim and Kochenderfer, Mykel},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems},
  pages={66--83},
  year={2017},
  organization={Springer}
}

### Experience Replay

[AI Baselines](https://github.com/openai/baselines)

Simple experience reply and priority experience replay.

@misc{baselines,
  author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
  title = {OpenAI Baselines},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openai/baselines}},
}
