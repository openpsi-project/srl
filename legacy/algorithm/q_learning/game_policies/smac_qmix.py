import random

from legacy.algorithm.q_learning.qmix.qmix import QtotPolicy
from api.policy import register, RolloutResult, RolloutRequest
from api.trainer import SampleBatch
from legacy.environment.smac.smac_env import SMACAction, get_smac_shapes


class SMACQMixPolicy(QtotPolicy):

    def __init__(self, **kwargs):
        map_name = kwargs["map_name"]
        obs_dim, state_dim, action_dim, num_agents = get_smac_shapes(map_name, use_state_agent=False)
        obs_dim = obs_dim[0]
        state_dim = state_dim[0]

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

        super(SMACQMixPolicy, self).__init__(num_agents=num_agents,
                                             obs_dim=obs_dim,
                                             action_dim=action_dim,
                                             state_dim=state_dim,
                                             chunk_len=chunk_len,
                                             use_double_q=use_double_q,
                                             epsilon_start=epsilon_start,
                                             epsilon_finish=epsilon_finish,
                                             epsilon_anneal_time=epsilon_anneal_time,
                                             q_i_config=q_i_config,
                                             mixer="qmix",
                                             mixer_config=mixer_config,
                                             state_use_all_local_obs=state_use_all_local_obs,
                                             state_concate_all_local_obs=state_concate_all_local_obs,
                                             seed=seed)

    def analyze(self, sample: SampleBatch, **kwargs):
        return super(SMACQMixPolicy, self).analyze(sample, target="qmix", **kwargs)

    def rollout(self, requests: RolloutRequest, **kwargs):
        action, policy_state = super(SMACQMixPolicy, self).rollout(requests, **kwargs)
        return RolloutResult(action=SMACAction(action), policy_state=policy_state)


register("smac-qmix", SMACQMixPolicy)
