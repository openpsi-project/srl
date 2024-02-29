import torch.nn
import numpy as np

from api.policy import Policy, RolloutResult
from api.policy import register


class RandomPolicy(Policy):
    """A un-trainable random policy for testing
    """

    def get_checkpoint(self):
        return {"state_dict": {}}

    def load_checkpoint(self, checkpoint):
        pass

    def train_mode(self):
        pass

    def eval_mode(self):
        pass

    @property
    def net(self):
        pass

    @property
    def version(self) -> int:
        return 0

    def inc_version(self):
        pass

    @property
    def neural_networks(self):
        return []

    def distributed(self):
        pass

    @property
    def default_policy_state(self):
        return None

    def __init__(self, action_space: int):
        super(RandomPolicy, self).__init__()
        self.action_space = action_space
        self.state_dim = 10
        self.__net = torch.nn.Module()

    def analyze(self, sample, **kwargs):
        pass

    def rollout(self, requests, **kwargs):
        num_requests = requests.length(dim=0)
        actions_scores = np.random.randint(low=10, high=100, size=(num_requests, self.action_space))
        actions_probs = actions_scores / actions_scores.sum()
        actions = actions_probs.argmax(axis=1)
        policy_states = np.random.random((num_requests, 1, self.state_dim))

        return RolloutResult(action=actions, log_probs=actions_probs, policy_state=policy_states)

    def parameters(self):
        return self.state_dim


register("random_policy", RandomPolicy)
