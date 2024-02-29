"""
Modified from marlbenchmark/onpoliy for MuZero
"""

from .distributions import Bernoulli, Categorical, DiagGaussian
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6


class ActionSampler:

    def __init__(self, action_space, action_dist):
        self.action_space = action_space
        self.action_dist = action_dist

    def sample_actions(self, num):
        """Sample actions according to the policy and bincount the sampled actions.
        """
        if self.action_space.__class__.__name__ == "Discrete":
            batch_size = self.action_dist.probs.size(0)
            actions = self.action_dist.sample((num,)).transpose(0, 1).long().cpu()
            assert len(actions.shape) == 2 and actions.shape[0] == batch_size and actions.shape[1] == num
            # bincount the sampled actions and filter duplicated actions
            action_list = []
            policy_list = []
            for b in range(batch_size):
                cnt = torch.zeros_like(self.action_dist.probs[b]).cpu().numpy()
                acts = []
                pols = []
                for i in range(num):
                    cnt[actions[b, i]] += 1
                    if cnt[actions[b, i]] == 1:
                        acts.append(actions[b, i])
                for a in acts:
                    pols.append(cnt[a])
                action_list.append(np.array(acts)[:, np.newaxis])
                policy_list.append(np.array(pols) / num)
            return action_list, policy_list

        elif self.action_space.__class__.__name__ == "Box":
            # tanh & then scale to the right range
            batch_size = self.action_dist.mean.size(0)
            action_scale = (self.action_space.high - self.action_space.low).reshape(1, 1, -1) / 2
            action_bias = (self.action_space.high + self.action_space.low).reshape(1, 1, -1) / 2
            prescaled_actions = torch.tanh(self.action_dist.sample((num,)).transpose(0, 1).cpu()).numpy()
            actions = (prescaled_actions * action_scale + action_bias)
            policy = np.ones((
                batch_size,
                num,
            )) / num
            return actions, policy

        elif self.action_space.__class__.__name__ == "MultiDiscrete":
            batch_size = self.action_dist[0].probs.size(0)
            actions = torch.stack(
                [act_dist.sample((num,)).transpose(0, 1).long().cpu() for act_dist in self.action_dist],
                dim=-1).numpy()
            assert len(actions.shape) == 3 and actions.shape[0] == batch_size and actions.shape[
                1] == num and actions.shape[2] == np.prod(self.action_space.shape)
            # bincount the sampled actions and filter duplicated actions
            action_list = []
            policy_list = []

            # use hash to filter out duplicated actions since the action space is now MultiDiscrete
            K = 119  # MAX number of hash
            base = [
                1,
            ]
            for i in range(1, np.prod(self.action_space.shape)):
                base.append(base[-1] * 29 % K)
            base = np.array(base)
            hash_value = (actions * base.reshape(1, 1, -1)).astype(np.int64).sum(axis=-1) % K

            for b in range(batch_size):
                acts = []
                pols = []
                hash_ptr = -np.ones((K,), dtype=np.int32)
                nxt_ptr = -np.ones((num,), dtype=np.int32)
                act_id = -np.ones((num,), dtype=np.int32)
                num_unique_actions = 0
                for i in range(num):
                    w = hash_ptr[hash_value[b, i]]
                    exists = False
                    while w != -1:
                        if all(actions[b, i] == actions[b, w]):
                            exists = True
                            break
                        w = nxt_ptr[w]
                    if not exists:
                        acts.append(actions[b, i])
                        pols.append(0)
                        act_id[i] = num_unique_actions
                        num_unique_actions += 1

                        nxt_ptr[i] = hash_ptr[hash_value[b, i]]
                        hash_ptr[hash_value[b, i]] = i
                    else:
                        act_id[i] = act_id[w]
                    pols[act_id[i]] += 1
                    assert all(acts[act_id[i]] == actions[b, i])
                action_list.append(np.array(acts).reshape(num_unique_actions, *self.action_space.shape))
                policy_list.append(np.array(pols) / num)
                assert abs(policy_list[-1].sum() - 1) < 1e-3
            return action_list, policy_list
        else:
            raise NotImplementedError(
                f"ActionSampler doesn't support {self.actino_space.__class__.__name__} now.")

    def evaluate_actions(self, actions):
        """Evaluate action log probability for given actions
        """
        if self.action_space.__class__.__name__ == "Discrete":
            if type(actions) is not torch.Tensor:
                actions = torch.from_numpy(actions).to(self.action_dist.probs)
            batch_size = self.action_dist.probs.size(0)
            actions = actions.long().squeeze(-1)
            assert len(actions.shape) == 2 and actions.shape[0] == batch_size
            log_probs = self.action_dist.log_prob(actions.transpose(0, 1)).transpose(0, 1)
            return log_probs, self.action_dist.entropy()

        elif self.action_space.__class__.__name__ == "Box":
            # convert continuous action back to gaussian (rescale + tanh^-1)
            batch_size = self.action_dist.mean.size(0)
            action_scale = torch.from_numpy(
                (self.action_space.high - self.action_space.low).reshape(1, 1, -1) / 2).to(
                    self.action_dist.mean)
            action_bias = torch.from_numpy(
                (self.action_space.high + self.action_space.low).reshape(1, 1, -1) / 2).to(
                    self.action_dist.mean)
            if type(actions) is not torch.Tensor:
                actions = torch.from_numpy(actions).to(self.action_dist.mean)
            prescaled_actions = (actions - action_bias) / action_scale
            gaussian = torch.atanh(prescaled_actions)
            log_probs = self.action_dist.log_prob(gaussian.transpose(0, 1)).transpose(0, 1)
            log_probs -= torch.log(action_scale * (1 - prescaled_actions.pow(2)) + EPSILON)
            return log_probs.sum(dim=-1), self.action_dist.entropy().sum(dim=-1)

        elif self.action_space.__class__.__name__ == "MultiDiscrete":
            if type(actions) is not torch.Tensor:
                actions = torch.from_numpy(actions).to(self.action_dist[0].probs)
            batch_size = self.action_dist[0].probs.size(0)
            actions = actions.reshape(batch_size, -1, np.prod(self.action_space.shape)).long()
            log_probs = []
            entropy = []
            for i, act_dist in enumerate(self.action_dist):
                log_probs.append(act_dist.log_prob(actions[:, :, i].transpose(0, 1)).transpose(0, 1))
                entropy.append(act_dist.entropy())
            log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)
            entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
            return log_probs, entropy

        else:
            raise NotImplementedError(
                f"ActionSampler doesn't support {self.actino_space.__class__.__name__} now.")


class ACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal=True, gain=0.01, init_zero=True):
        super(ACTLayer, self).__init__()
        self.action_space = action_space

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain, init_zero)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.action_outs = []
            for d in action_space.nvec.reshape(-1):
                self.action_outs.append(Categorical(inputs_dim, d, use_orthogonal, gain, init_zero))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:
            raise NotImplementedError(f"ACTLayer doesn't support {self.action_space.__class__.__name__} now.")

    def forward(self, x, available_actions=None):
        if self.action_space.__class__.__name__ == "Discrete":
            action_dist = self.action_out(x, available_actions)
        elif self.action_space.__class__.__name__ == "Box":
            action_dist = self.action_out(x)
        elif self.action_space.__class__.__name__ == "MultiDiscrete":
            action_dist = []
            for action_out in self.action_outs:
                action_dist.append(action_out(x))
        else:
            raise NotImplementedError(f"ACTLayer doesn't support {self.actino_space.__class__.__name__} now.")
        return ActionSampler(self.action_space, action_dist)
