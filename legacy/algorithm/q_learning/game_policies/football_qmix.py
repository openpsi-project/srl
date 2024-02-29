import random

from legacy.algorithm.q_learning.qmix.qmix import QtotPolicy
from api.policy import register, RolloutResult, RolloutRequest
from api.trainer import SampleBatch
from api.env_utils import DiscreteAction

map_agent_registry = {
    # evn_name: (left, right, game_length)
    "11_vs_11_competition": (11, 11, 3000),
    "11_vs_11_easy_stochastic": (11, 11, 3000),
    "11_vs_11_hard_stochastic": (11, 11, 3000),
    "11_vs_11_kaggle": (11, 11, 3000),
    "11_vs_11_stochastic": (11, 11, 3000),
    "1_vs_1_easy": (1, 1, 500),
    "5_vs_5": (4, 4, 3000),
    "academy_3_vs_1_with_keeper": (3, 1, 400),
    "academy_corner": (11, 11, 400),
    "academy_counterattack_easy": (11, 11, 400),
    "academy_counterattack_hard": (11, 11, 400),
}


class FootballQMixPolicy(QtotPolicy):

    def __init__(self, **kwargs):
        env_name = kwargs["env_name"]
        obs_dim = 115
        state_dim = 115 - 11
        action_dim = 19
        num_agents = map_agent_registry[env_name][0]

        chunk_len = kwargs.get("chunk_len", 10)
        use_double_q = kwargs.get("use_double_q", True)
        epsilon_start = kwargs.get("epsilon_start", 1.0)
        epsilon_finish = kwargs.get("epsilon_finish", 0.05)
        epsilon_anneal_time = kwargs.get("epsilon_anneal_time", 5000)
        q_i_config = kwargs["q_i_config"]
        if "hidden_dim" not in q_i_config:
            q_i_config["hidden_dim"] = 128
        if "num_dense_layers" not in q_i_config:
            q_i_config["num_dense_layers"] = 2
        if "rnn_type" not in q_i_config:
            q_i_config["rnn_type"] = "gru"
        if "num_rnn_layers" not in q_i_config:
            q_i_config["num_rnn_layers"] = 1
        mixer_config = kwargs["mixer_config"]
        if "popart" not in mixer_config:
            mixer_config["popart"] = False
        if "hidden_dim" not in mixer_config:
            mixer_config["hidden_dim"] = 64
        if "num_hypernet_layers" not in mixer_config:
            mixer_config["num_hypernet_layers"] = 2
        if "hypernet_hidden_dim" not in mixer_config:
            mixer_config["hypernet_hidden_dim"] = 64
        state_use_all_local_obs = kwargs.get("state_use_all_local_obs", False)
        state_concate_all_local_obs = kwargs.get("state_concate_all_local_obs", False)
        seed = kwargs.get("seed", random.randint(0, 10000))

        super().__init__(num_agents=num_agents,
                         obs_dim=obs_dim,
                         action_dim=action_dim,
                         state_dim=state_dim,
                         chunk_len=chunk_len,
                         use_double_q=use_double_q,
                         epsilon_start=epsilon_start,
                         epsilon_finish=epsilon_finish,
                         epsilon_anneal_time=epsilon_anneal_time,
                         q_i_config=q_i_config,
                         mixer=kwargs.get("mixer_type", "qmix"),
                         mixer_config=mixer_config,
                         state_use_all_local_obs=state_use_all_local_obs,
                         state_concate_all_local_obs=state_concate_all_local_obs,
                         seed=seed)

    def analyze(self, sample: SampleBatch, **kwargs):
        return super().analyze(sample, target="qmix", **kwargs)

    def rollout(self, requests: RolloutRequest, **kwargs):
        action, policy_state = super().rollout(requests, **kwargs)
        return RolloutResult(action=DiscreteAction(action), policy_state=policy_state)


register("football-qmix", FootballQMixPolicy)
