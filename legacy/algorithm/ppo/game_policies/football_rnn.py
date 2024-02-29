import random

from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import ActorCriticPolicy
from api.policy import register


class FootballSeparatePolicy(ActorCriticPolicy):

    def __init__(self, **kwargs):
        obs_dim = kwargs.get("obs_dim", 115)
        action_dim = kwargs.get("action_dim", 19)
        hidden_dim = kwargs.get("hidden_dim", 128)
        rnn_type = kwargs.get("rnn_type", "gru")
        num_rnn_layers = kwargs.get("num_rnn_layers", 1)
        chunk_len = kwargs.get("chunk_len", 10)
        popart = kwargs.get("popart", True)
        seed = kwargs.get("seed", random.randint(0, 10000))
        shared = kwargs.get("shared_backbone", False)
        auxiliary_head = kwargs.get("auxiliary_head", False)

        super(FootballSeparatePolicy, self).__init__(obs_dim=obs_dim,
                                                     action_dim=action_dim,
                                                     hidden_dim=hidden_dim,
                                                     chunk_len=chunk_len,
                                                     rnn_type=rnn_type,
                                                     num_rnn_layers=num_rnn_layers,
                                                     popart=popart,
                                                     shared_backbone=shared,
                                                     auxiliary_head=auxiliary_head,
                                                     seed=seed)


class FootballSMMPolicy(ActorCriticPolicy):

    def __init__(self, **kwargs):
        obs_dim = kwargs.get("obs_dim", {"obs": (4, 96, 72)})
        action_dim = kwargs.get("action_dim", 19)
        hidden_dim = kwargs.get("hidden_dim", 128)
        rnn_type = kwargs.get("rnn_type", "gru")
        num_rnn_layers = kwargs.get("num_rnn_layers", 1)
        chunk_len = kwargs.get("chunk_len", 10)
        popart = kwargs.get("popart", True)
        seed = kwargs.get("seed", random.randint(0, 10000))
        auxiliary_head = kwargs.get("auxiliary_head", False)

        super(FootballSMMPolicy, self).__init__(obs_dim=obs_dim,
                                                action_dim=action_dim,
                                                hidden_dim=hidden_dim,
                                                chunk_len=chunk_len,
                                                rnn_type=rnn_type,
                                                num_rnn_layers=num_rnn_layers,
                                                popart=popart,
                                                seed=seed,
                                                auxiliary_head=auxiliary_head)


register("football-simple115-separate", FootballSeparatePolicy)
register("football-smm-separate", FootballSMMPolicy)
