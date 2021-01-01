from torchviz import make_dot
import torch

from MATD3.actor import Actor
from MATD3.critic import Critic

if __name__ == "__main__":
    # give random inputs
    state = torch.randn(1, 31)
    action = torch.randn(1, 4)

    actor_model = Actor(31, 4, 1)
    make_dot(actor_model(state), params=dict(list(actor_model.named_parameters()))).render(filename="Actor-NN",
                                                                                           format="png")
    critic_model = Critic(31, 4)
    make_dot(actor_model(state), params=dict(list(actor_model.named_parameters()))).render(filename="Critic-NN",
                                                                                           format="png")
