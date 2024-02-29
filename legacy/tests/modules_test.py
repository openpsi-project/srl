import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import unittest

from base.testing import *

from api.trainer import SampleBatch
from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import PPORolloutAnalyzedResult
from legacy.algorithm.q_learning.game_policies.atari_dqn_policy import DQNRolloutAnalyzedResult
import api.config
import api.trainer
import legacy.algorithm.modules as modules


# this is copied from tianshou repository
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma * (1 - end_flag) - v_s
    m = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        returns[i] = gae
    return returns


def _calc_masked_adv(adv, mask):
    advantages_copy = adv.copy()
    advantages_copy[mask == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)
    advantages = (adv - mean_advantages) / (std_advantages + 1e-5)

    adv_ = modules.masked_normalization(torch.from_numpy(adv), torch.from_numpy(mask), unbiased=False)
    return advantages, adv_


def verify_dist_mask_norm(_, rank, port, adv, mask, result):
    dist.init_process_group(backend="gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    adv_ = modules.masked_normalization(torch.from_numpy(adv),
                                        torch.from_numpy(mask) if mask is not None else None,
                                        unbiased=False)
    if mask is None:
        np.testing.assert_almost_equal(adv_, result, decimal=6)
    else:
        np.testing.assert_almost_equal(adv_ * mask, result * mask, decimal=6)


def verify_popart_update(_, rank, port, popart, x, total_mean, total_std, x2, total_mean_2, total_std_2):
    dist.init_process_group(backend="gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")

    popart.update(x, mask=None)
    y = torch.randn(20, 3)
    torch.testing.assert_allclose(popart.normalize(y), (y - total_mean) / total_std)
    torch.testing.assert_allclose(popart.denormalize(y), (y * total_std + total_mean))

    for _ in range(4):
        popart.update(x, mask=None)
    torch.testing.assert_allclose(popart.normalize(y), (y - total_mean) / total_std)
    torch.testing.assert_allclose(popart.denormalize(y), (y * total_std + total_mean))
    popart.update(x2, mask=None)
    torch.testing.assert_allclose(popart.normalize(y), (y - total_mean_2) / total_std_2)
    torch.testing.assert_allclose(popart.denormalize(y), (y * total_std_2 + total_mean_2))


class AlgorithmModulesTest(unittest.TestCase):

    def test_to_chunk(self):
        x = torch.randn(24, 8, 1)
        num_chunks = 8
        x_ = modules.to_chunk(x, num_chunks)
        _x = x.view(8, 3, 8, 1).transpose(0, 1).reshape(3, 64, 1)
        self.assertTrue((x_ == _x).all())

    def test_to_traj(self):
        x = torch.randn(24, 8, 1)
        num_chunks = 8
        self.assertTrue((x == modules.back_to_trajectory(modules.to_chunk(x, num_chunks), num_chunks)).all())

    def test_gae(self):
        value = np.random.randn(101, 8, 1)
        rew = np.random.randn(100, 8, 1)
        done = np.random.randint(0, 2, (101, 8, 1))
        truncated = np.zeros((101, 8, 1))
        on_reset = np.concatenate([np.zeros_like(done[:1]), done[:-1]], axis=0)

        rew = rew * (1 - on_reset[1:])  # Reward is zero on reset
        value = value * (1 - done)  # Value is zero when not done.

        adv = _gae_return(value[:-1], value[1:], rew, done[:-1], 0.99, 0.97)

        value = torch.from_numpy(value)
        rew = torch.from_numpy(rew)
        done = torch.from_numpy(done)
        truncated = torch.from_numpy(truncated)
        on_reset = torch.from_numpy(on_reset)
        adv_ = modules.gae_trace(
            reward=rew,
            value=value,
            truncated=truncated,
            done=done,
            on_reset=on_reset,
            gamma=0.99,
            lmbda=0.97,
        )
        self.assertTrue(((torch.from_numpy(adv) - adv_).abs() < 1e-5).all())

    def test_gae_truncated(self):
        # single-agent
        on_reset = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32)
        rew = np.array([1, 2, 0, 1, 3, 0, 1, 2, 3], dtype=np.float32)
        value = np.array([2, 0, 1, 2, 2, 0, 1, 1, 1])
        truncated = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
        done = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])

        adv = modules.gae_trace(reward=torch.from_numpy(rew)[:-1],
                                value=torch.from_numpy(value),
                                truncated=torch.from_numpy(truncated),
                                done=torch.from_numpy(done),
                                on_reset=torch.from_numpy(on_reset),
                                gamma=0.1,
                                lmbda=0.1).numpy()
        ep3_adv = [0.111, 1.1]
        ep2_adv = [-0.8 + 0.01, 1, 0]
        ep1_adv = [2.1 * 0.01 - 1, 2.1, 0]
        np.testing.assert_array_almost_equal(adv * (1 - on_reset[1:]),
                                             np.array(ep1_adv + ep2_adv + ep3_adv) * (1 - on_reset[1:]))

    def test_traj_gae(self):
        # yapf: disable
        rew =       [1, 2, 0]
        value =     [2, 0, 1]
        done =      [0, 0, 0]
        truncated = [0, 0, 1]
        # yapf: enable
        memory = [
            SampleBatch(obs=None,
                        reward=np.array([r]),
                        analyzed_result=PPORolloutAnalyzedResult(value=np.array([v]), log_probs=None),
                        done=np.array([d]),
                        truncated=np.array([t])) for r, v, d, t in zip(rew, value, done, truncated)
        ]
        processor = api.trainer.make_traj_postprocessor(
            api.config.TrajPostprocessor('gae', args=dict(gamma=0.1, lmbda=0.1)))
        memory = processor.process(memory)
        adv = [m.analyzed_result.adv.item() for m in memory[:-1]]
        for i, (a, b) in enumerate(zip(adv, [2.1 * 0.01 - 1, 2.1])):
            self.assertAlmostEqual(a, b)
        # yapf: disable
        rew =       [1, 3, 0]
        value =     [2, 2, 0]
        done =      [0, 0, 1]
        truncated = [0, 0, 0]
        # yapf: enable
        memory = [
            SampleBatch(obs=None,
                        reward=np.array([r]),
                        analyzed_result=PPORolloutAnalyzedResult(value=np.array([v]), log_probs=None),
                        done=np.array([d]),
                        truncated=np.array([t])) for r, v, d, t in zip(rew, value, done, truncated)
        ]
        processor = api.trainer.make_traj_postprocessor(
            api.config.TrajPostprocessor('gae', args=dict(gamma=0.1, lmbda=0.1)))
        memory = processor.process(memory)
        adv = [m.analyzed_result.adv.item() for m in memory[:-1]]
        for i, (a, b) in enumerate(zip(adv, [-0.8 + 0.01, 1])):
            self.assertAlmostEqual(a, b)

    def test_n_step_return(self):
        # yapf: disable
        rew =       [1, 2,  1, -3, 0, 3, -1,  2,   1, -2,  10,   1]
        value =     [2, 10, 1,  5, 0, 3,  2, -2, -10,  5, -1, -100]
        done =      [0, 0,  0,  0, 1, 0,  0,  0,   0,  0,  0,    0]
        truncated = [0, 0,  0,  0, 0, 0,  0,  0,   0,  1,  0,    0]
        # yapf: enable
        ret = modules.n_step_return(n=2,
                                    reward=torch.from_numpy(np.array(rew[:-1])),
                                    nex_value=torch.from_numpy(np.array(value[1:])),
                                    nex_done=torch.from_numpy(np.array(done[1:])),
                                    nex_truncated=torch.from_numpy(np.array(truncated[1:])),
                                    gamma=0.1)
        self.assertEqual(len(ret), len(value[:-1]) - 2 + 1)
        ret1 = [1.21, 2.15, 0.7, -3]
        ret2 = [2.88, -0.9, 2.15, 1.5]
        np.testing.assert_almost_equal(np.array(ret1), ret[:4])
        np.testing.assert_almost_equal(np.array(ret2), ret[5:9])

        ret = modules.n_step_return(n=3,
                                    reward=torch.from_numpy(np.array(rew[:-1])),
                                    nex_value=torch.from_numpy(np.array(value[1:])),
                                    nex_done=torch.from_numpy(np.array(done[1:])),
                                    nex_truncated=torch.from_numpy(np.array(truncated[1:])),
                                    gamma=0.1)
        self.assertEqual(len(ret), len(value[:-1]) - 3 + 1)
        ret1 = [1.215, 2.07, 0.7, -3]
        ret2 = [2.91, -0.79 + 5e-3, 2.15, 1.5]
        np.testing.assert_almost_equal(np.array(ret1), ret[:4])
        np.testing.assert_almost_equal(np.array(ret2), ret[5:])

    def test_traj_n_step_return(self):
        # yapf: disable
        rew =           [1, 2,  1, -3, 0]
        target_value =  [2, 10, 1,  5, 0]
        done =          [0, 0,  0,  0, 1]
        truncated =     [0, 0,  0,  0, 0]
        value = np.random.randn(5).tolist()
        # yapf: enable
        ret1 = {2: [1.21, 2.15, 0.7, -3], 3: [1.215, 2.07, 0.7, -3]}
        memory = [
            SampleBatch(obs=None,
                        reward=np.array([r]),
                        analyzed_result=DQNRolloutAnalyzedResult(value=np.array([v]),
                                                                 target_value=np.array([t_v])),
                        done=np.array([d]),
                        truncated=np.array([t]))
            for r, v, t_v, d, t in zip(rew, value, target_value, done, truncated)
        ]
        for n in range(2, 4):
            processor = api.trainer.make_traj_postprocessor(
                api.config.TrajPostprocessor('n-step-return', args=dict(n=n, gamma=0.1)))
            memory = processor.process(memory)
            ret_ = [m.analyzed_result.ret.item() for m in memory[:-1]]
            for a, b in zip(ret_, ret1[n]):
                self.assertAlmostEqual(a, b)

        # yapf: disable
        rew =       [3, -1,  2,   1, -2]
        target_value =     [3,  2, -2, -10,  5]
        done =      [0,  0,  0,   0,  0]
        truncated = [0,  0,  0,   0,  1]
        value = np.random.randn(5).tolist()
        # yapf: enable
        ret1 = {2: [2.88, -0.9, 2.15, 1.5], 3: [2.91, -0.79 + 5e-3, 2.15, 1.5]}
        memory = [
            SampleBatch(obs=None,
                        reward=np.array([r]),
                        analyzed_result=DQNRolloutAnalyzedResult(value=np.array([v]),
                                                                 target_value=np.array([t_v])),
                        done=np.array([d]),
                        truncated=np.array([t]))
            for r, v, t_v, d, t in zip(rew, value, target_value, done, truncated)
        ]
        for n in range(2, 4):
            processor = api.trainer.make_traj_postprocessor(
                api.config.TrajPostprocessor('n-step-return', args=dict(n=n, gamma=0.1)))
            memory = processor.process(memory)
            ret_ = [m.analyzed_result.ret.item() for m in memory[:-1]]
            for a, b in zip(ret_, ret1[n]):
                self.assertAlmostEqual(a, b)

    def test_scalar_transform(self):
        a = np.random.randn(100)
        b = modules.inverse_scalar_transform(modules.scalar_transform(torch.from_numpy(a))).numpy()
        np.testing.assert_almost_equal(a, b, decimal=4)

    def test_mask_norm(self):
        adv = np.random.randn(10, 8, 1)
        mask = np.random.randint(0, 2, (10, 8, 1))
        r1, r2 = _calc_masked_adv(adv, mask)
        np.testing.assert_almost_equal(r1 * mask, r2 * mask, decimal=6)

    def test_mask_norm_distributed(self):
        adv = np.random.randn(1, 16, 1)
        mask = np.random.randint(0, 2, (1, 16, 1))
        r1, _ = _calc_masked_adv(adv, mask)
        port = get_testing_port()
        p1 = torch.multiprocessing.spawn(verify_dist_mask_norm,
                                         args=(0, port, adv[:, :8], mask[:, :8], r1[:, :8]),
                                         join=False)
        p2 = torch.multiprocessing.spawn(verify_dist_mask_norm,
                                         args=(1, port, adv[:, 8:], mask[:, 8:], r1[:, 8:]),
                                         join=False)
        p1.join()
        p2.join()

    def test_no_mask_norm_distributed(self):
        adv = np.random.randn(1, 16, 1)
        mask = np.ones((1, 16, 1))
        r1, _ = _calc_masked_adv(adv, mask)
        port = get_testing_port()
        p1 = torch.multiprocessing.spawn(verify_dist_mask_norm,
                                         args=(0, port, adv[:, :8], None, r1[:, :8]),
                                         join=False)
        p2 = torch.multiprocessing.spawn(verify_dist_mask_norm,
                                         args=(1, port, adv[:, 8:], None, r1[:, 8:]),
                                         join=False)
        p1.join()
        p2.join()

    def test_popart(self):
        popart = modules.PopArtValueHead(32, 3, beta=0.999)
        inp = torch.randn(10, 32)
        self.assertTrue(popart(inp).shape == (10, 3))
        x = torch.randn(100, 3)
        x_mean, x_std = x.mean(dim=0), (x.square().mean(dim=0) - x.mean(dim=0).square()).sqrt()
        popart.update(x, mask=None)
        y = torch.randn(20, 3)
        torch.testing.assert_allclose(popart.normalize(y), (y - x_mean) / x_std)
        torch.testing.assert_allclose(popart.denormalize(y), (y * x_std + x_mean))

        for _ in range(4):
            popart.update(x, mask=None)
        torch.testing.assert_allclose(popart.normalize(y), (y - x_mean) / x_std)
        torch.testing.assert_allclose(popart.denormalize(y), (y * x_std + x_mean))
        x2 = torch.randn(20, 3)
        x2_mean, x2_std = x2.mean(dim=0), (x2.square().mean(dim=0) - x2.mean(dim=0).square()).sqrt()
        mean = x_mean + (x2_mean - x_mean) / (1 - 0.999**6) * 0.001
        mean_sq = x.square().mean(
            dim=0) + (x2.square().mean(dim=0) - x.square().mean(dim=0)) / (1 - 0.999**6) * 0.001
        popart.update(x2, mask=None)
        std = (mean_sq - mean**2).sqrt()
        torch.testing.assert_allclose(popart.normalize(y), (y - mean) / std)
        torch.testing.assert_allclose(popart.denormalize(y), (y * std + mean))

    def test_popart_distributed(self):
        popart1 = modules.PopArtValueHead(32, 3, beta=0.999)
        popart2 = modules.PopArtValueHead(32, 3, beta=0.999)
        popart2.load_state_dict(popart1.state_dict())
        x = torch.randn(100, 3)
        x_mean, x_std = x.mean(dim=0), (x.square().mean(dim=0) - x.mean(dim=0).square()).sqrt()
        x2 = torch.randn(20, 3)
        x2_mean, x2_std = x2.mean(dim=0), (x2.square().mean(dim=0) - x2.mean(dim=0).square()).sqrt()
        mean = x_mean + (x2_mean - x_mean) / (1 - 0.999**6) * 0.001
        mean_sq = x.square().mean(
            dim=0) + (x2.square().mean(dim=0) - x.square().mean(dim=0)) / (1 - 0.999**6) * 0.001
        std = (mean_sq - mean**2).sqrt()
        port = get_testing_port()
        p1 = torch.multiprocessing.spawn(verify_popart_update,
                                         args=(0, port, popart1, x[:50], x_mean, x_std, x2[10:], mean, std),
                                         join=False)
        p2 = torch.multiprocessing.spawn(verify_popart_update,
                                         args=(1, port, popart2, x[50:], x_mean, x_std, x2[:10], mean, std),
                                         join=False)
        p1.join()
        p2.join()

    def test_autoreset_rnn(self):
        gru = modules.AutoResetRNN(64, 64)
        x = torch.randn(7, 10, 64)
        h = torch.randn(1, 10, 64)
        mask = torch.randint(0, 2, (7, 10, 1))
        x_, h_ = gru(x, h)
        x2_, h2_ = gru(x, h, mask)
        self.assertTrue((not (h_ == h2_).all()))
        self.assertTrue((not (x_ == x2_).all()))

        mask = torch.zeros(1, 10, 1)
        x_, h_ = gru(x[:1], torch.zeros(1, 10, 64))
        x2_, h2_ = gru(x[:1], h, 1 - mask)
        self.assertTrue((h_ == h2_).all())
        self.assertTrue((x_ == x2_).all())

        gru = modules.AutoResetRNN(64, 64, rnn_type='lstm')
        x = torch.randn(7, 10, 64)
        h = torch.randn(1, 10, 128)
        mask = torch.randint(0, 2, (7, 10, 1))
        x_, h_ = gru(x, h)
        x2_, h2_ = gru(x, h, mask)
        self.assertTrue((not (h_ == h2_).all()))
        self.assertTrue((not (x_ == x2_).all()))

        mask = torch.zeros(1, 10, 1)
        x_, h_ = gru(x[:1], torch.zeros(1, 10, 128))
        x2_, h2_ = gru(x[:1], h, 1 - mask)
        self.assertTrue((h_ == h2_).all())
        self.assertTrue((x_ == x2_).all())

    def test_cnn(self):
        cnn = modules.cnn.Convolution(input_shape=(3, 51, 37),
                                      cnn_layers=[(3, 3, 1, 0, "zeros")] * 3,
                                      hidden_size=64,
                                      activation=nn.ReLU,
                                      use_orthogonal=True,
                                      use_maxpool=True)
        self.assertEqual(tuple(cnn(torch.randn(100, 10, 3, 51, 37)).size()), (100, 10, 64))

        cnn = modules.cnn.Convolution(input_shape=(3, 15, 13, 11),
                                      cnn_layers=[(3, 3, 1, 0, "zeros")] * 3,
                                      hidden_size=64,
                                      activation=nn.ReLU,
                                      use_orthogonal=True,
                                      use_maxpool=False)
        self.assertEqual(tuple(cnn(torch.randn(100, 10, 3, 15, 13, 11)).size()), (100, 10, 64))

        cnn = modules.cnn.Convolution(input_shape=(3, 59),
                                      cnn_layers=[(3, 3, 1, 0, "zeros")] * 3,
                                      hidden_size=64,
                                      activation=nn.ReLU,
                                      use_orthogonal=True,
                                      use_maxpool=True)
        self.assertEqual(tuple(cnn(torch.randn(100, 10, 3, 59)).size()), (100, 10, 64))


if __name__ == "__main__":
    unittest.main()
