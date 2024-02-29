import random

from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import ActorCriticPolicy
from api.policy import register


class AtariVisionPolicy(ActorCriticPolicy):

    def __init__(self, **kwargs):
        obs_dim = kwargs.get("obs_dim", {"obs": (3, 160, 210)})
        action_dim = kwargs.get("action_dim", 18)
        hidden_dim = kwargs.get("hidden_dim", 32)
        rnn_type = kwargs.get("rnn_type", "gru")
        num_rnn_layers = kwargs.get("num_rnn_layers", 0)
        chunk_len = kwargs.get("chunk_len", 10)
        cnn_layers = kwargs.get("cnn_layers", {"obs": [(3, 5, 2, 0, "zeros"), (3, 5, 2, 0, "zeros")]})
        popart = kwargs.get("popart", True)
        seed = kwargs.get("seed", random.randint(0, 10000))
        auxiliary_head = kwargs.get("auxiliary_head", False)

        super(AtariVisionPolicy, self).__init__(obs_dim=obs_dim,
                                                action_dim=action_dim,
                                                hidden_dim=hidden_dim,
                                                chunk_len=chunk_len,
                                                rnn_type=rnn_type,
                                                num_rnn_layers=num_rnn_layers,
                                                num_dense_layers=1,
                                                cnn_layers=cnn_layers,
                                                popart=popart,
                                                seed=seed,
                                                auxiliary_head=auxiliary_head)


register("atari-vision", AtariVisionPolicy)
register("atari_naive_rnn", AtariVisionPolicy)
