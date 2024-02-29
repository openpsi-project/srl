import torch
import typing

import numpy as np
import torch.nn as nn

from typing import List, Union, Any


class NetworkOutput(typing.NamedTuple):
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    entropy: Union[None, List[float]]
    actions: Union[None, List[Any]]
    hidden_state: List[float]
    reward_hidden: object


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst):
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.value_prefix)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    # hidden_state_lst = torch.cat(hidden_state_lst, 0)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (reward_hidden_c_lst,
                                                                        reward_hidden_h_lst)


class BaseNet(nn.Module):

    def __init__(self,
                 act_dim,
                 action_space,
                 inverse_value_transform,
                 inverse_reward_transform,
                 lstm_hidden_size,
                 mcts_use_sampled_actions=False,
                 mcts_num_sampled_actions=0):
        """Base Network
        Parameters
        ----------
        act_dim: int
            Number of actions when doing MCTS
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        mcts_use_sampled_actions: bool
            use sampled actions to do tree search. Recommended for complex action space.
            Reference: Learning and Planning in Complex Action Spaces. http://arxiv.org/abs/2104.06303.
        mcts_num_sampled_actions: int
            number of sampled actions for MCTS when mcts_use_sampled_actions=True
        """
        super(BaseNet, self).__init__()
        self.act_dim = act_dim
        self.action_space = action_space
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.lstm_hidden_size = lstm_hidden_size
        self.mcts_use_sampled_actions = mcts_use_sampled_actions
        self.mcts_num_sampled_actions = mcts_num_sampled_actions

        if not self.mcts_use_sampled_actions:
            self.mcts_num_sampled_actions = 0

            # if use full action space for mcts, action space must be discrete
            assert (
                self.action_space.__class__.__name__ == "Discrete" and self.act_dim == self.action_space.n
            ), (f"Expected discrete action space with act_dim={self.act_dim} dims. But get act_dim={self.act_dim}, action_space={self.action_space}"
                )

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self,
                          obs,
                          sampled_actions=None,
                          num_sampled_actions=None,
                          evaluate_action_log_prob=False) -> NetworkOutput:
        num = obs.size(0)

        state = self.representation(obs)
        action_sampler, value = self.prediction(state)
        actions, actor_logit, entropy = self.sample_actions(state.size(0),
                                                            action_sampler,
                                                            sampled_actions=sampled_actions,
                                                            num_sampled_actions=num_sampled_actions
                                                            or self.mcts_num_sampled_actions,
                                                            evaluate_action_log_prob=evaluate_action_log_prob)

        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            if type(actor_logit) is torch.Tensor:
                actor_logit = actor_logit.detach().cpu().numpy()
            # zero initialization for reward (value prefix) hidden states
            if self.lstm_hidden_size == 0:
                reward_hidden = None
            else:
                reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                                 torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy())
        else:
            # zero initialization for reward (value prefix) hidden states
            if self.lstm_hidden_size == 0:
                reward_hidden = None
            else:
                reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'),
                                 torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))

        return NetworkOutput(value, np.zeros((num,), dtype=np.float32), actor_logit, entropy, actions, state,
                             reward_hidden)

    def recurrent_inference(self,
                            hidden_state,
                            reward_hidden,
                            action,
                            sampled_actions=None,
                            num_sampled_actions=None,
                            evaluate_action_log_prob=False) -> NetworkOutput:
        state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        action_sampler, value = self.prediction(state)
        actions, actor_logit, entropy = self.sample_actions(state.size(0),
                                                            action_sampler,
                                                            sampled_actions=sampled_actions,
                                                            num_sampled_actions=num_sampled_actions
                                                            or self.mcts_num_sampled_actions,
                                                            evaluate_action_log_prob=evaluate_action_log_prob)

        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            value_prefix = self.inverse_reward_transform(value_prefix).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            if reward_hidden is not None:
                reward_hidden = (reward_hidden[0].detach().cpu().numpy(),
                                 reward_hidden[1].detach().cpu().numpy())
            if type(actor_logit) is torch.Tensor:
                actor_logit = actor_logit.detach().cpu().numpy()

        return NetworkOutput(value, value_prefix, actor_logit, entropy, actions, state, reward_hidden)

    def sample_actions(self,
                       batch_size,
                       action_sampler,
                       sampled_actions=None,
                       num_sampled_actions=0,
                       evaluate_action_log_prob=False):
        if self.mcts_use_sampled_actions:
            if sampled_actions is not None:
                actions = sampled_actions.cpu().numpy()
                policy, entropy = action_sampler.evaluate_actions(actions)
            elif num_sampled_actions > 0:
                actions, policy = action_sampler.sample_actions(num_sampled_actions)
                if isinstance(policy, list):
                    policy = [np.log(np.array(x) + 1e-9) for x in policy]
                else:
                    policy = np.log(np.array(policy) + 1e-9)
                entropy = None
                if evaluate_action_log_prob:
                    lengths = [len(x) for x in actions]
                    _, _, pad_actions = pad_sampled_actions(-1, actions, policy)
                    policy, _ = action_sampler.evaluate_actions(pad_actions)
                    policy = [p[:l] for l, p in zip(lengths, policy)]
            else:
                raise RuntimeError("Expected sampled_actions != None or num_sampled_actions > 0")
        else:
            assert (
                self.action_space.__class__.__name__ == "Discrete" and self.act_dim == self.action_space.n
            ), (f"Expected discrete action space with act_dim={self.act_dim} dims. But get act_dim={self.act_dim}, action_space={self.action_space}"
                )
            actions = np.arange(self.action_space.n).reshape(1, -1, 1).repeat(batch_size, axis=0)
            policy, entropy = action_sampler.evaluate_actions(actions)
        return actions, policy, entropy

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def renormalize(tensor, first_dim=1):
    # normalize the tensor (states)
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)
