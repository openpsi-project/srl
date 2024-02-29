from copy import deepcopy
from typing import Dict
import numpy as np
import torch
import types
import unittest

from base.namedarray import NamedArray, recursive_apply, recursive_aggregate
import base.namedarray

T, bs = 40, 10


class Observation(NamedArray):

    def __init__(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        mask: np.ndarray = None,
    ):
        super(Observation, self).__init__(x=x, y=y, mask=mask)


class PolicyState(NamedArray):

    def __init__(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
    ):
        super(PolicyState, self).__init__(x=x, y=y)


class SimpleSampleBatch(NamedArray):

    def __init__(self, obs: Observation = None, **kwargs):
        super(SimpleSampleBatch, self).__init__(obs=obs, **kwargs)


class SampleBatch(SimpleSampleBatch):

    def __init__(
        self,
        # test inheritance
        action: np.ndarray = None,
        log_probs: np.ndarray = None,
        policy_state: PolicyState = None,
        **kwargs,
    ):
        super(SampleBatch, self).__init__(action=action,
                                          log_probs=log_probs,
                                          policy_state=policy_state,
                                          **kwargs)


class WrongSampleBatch(NamedArray):

    def __init__(
        self,
        action: np.array = None,
        log_probs: np.ndarray = None,
        reward: np.ndarray = None,
    ):
        super(WrongSampleBatch, self).__init__(action=action, log_probs=log_probs, reward=reward)


class NamedArrayTest(unittest.TestCase):

    def setUp(self):
        self.obs = Observation(x=np.random.randn(T, bs, 20),
                               y=np.random.randn(T, bs, 30),
                               mask=np.random.randn(T, bs, 1))
        self.batch = SampleBatch(obs=self.obs,
                                 action=np.random.randn(T, bs, 1),
                                 log_probs=None,
                                 policy_state=None)
        # print(self.batch)

        self.batch_mean = recursive_apply(self.batch, np.mean)

    def test_sub_class(self):
        self.assertTrue(isinstance(self.batch, NamedArray))
        # self.assertTrue(issubclass(type(self.batch), Mapping))
        self.assertTrue(isinstance(self.obs, NamedArray))

    def test_get_item(self):
        self.assertIsInstance(self.batch.obs, NamedArray)
        self.assertIsInstance(self.batch['obs'], NamedArray)
        subbatch = self.batch[:-1, :5]
        recursive_apply(subbatch, lambda x: self.assertEqual(x.shape[:2], (T - 1, 5)))

    def test_recursive_apply(self):

        def fn(x):
            x[0] = 1

        tmp = self.batch[0]

        recursive_apply(self.batch, fn)
        recursive_apply(self.batch, lambda x: self.assertTrue(np.all(x[0] == 1)))
        self.batch[0] = tmp

    def test_iops(self):
        tmp = deepcopy(self.batch)
        self.batch[:] = 0
        self.batch += 5
        recursive_apply(self.batch, lambda x: self.assertTrue((x == 5).all()))

        self.batch /= 2
        recursive_apply(self.batch, lambda x: self.assertTrue((x == 2.5).all()))

        self.batch *= 3
        recursive_apply(self.batch, lambda x: self.assertTrue((x == 7.5).all()))

        self.batch -= 3
        recursive_apply(self.batch, lambda x: self.assertTrue((x == 4.5).all()))

        self.batch[..., 0] += np.array([0.5])
        recursive_apply(self.batch, lambda x: self.assertTrue((x[..., 0] == 5).all()))
        recursive_apply(self.batch, lambda x: self.assertTrue((x[..., 1:] == 4.5).all()))
        self.batch += tmp

    def test_ops(self):
        tmp1 = self.batch - self.batch
        tmp2 = tmp1 + 5
        self.assertTrue((tmp1 is not tmp2))
        recursive_apply(tmp2, lambda x: self.assertTrue((x == 5).all()))

        tmp = tmp2 / 2
        self.assertTrue((tmp is not tmp2))
        recursive_apply(tmp, lambda x: self.assertTrue((x == 2.5).all()))

        tmp = tmp * 3
        recursive_apply(tmp, lambda x: self.assertTrue((x == 7.5).all()))

        tmp = tmp - 3
        recursive_apply(tmp, lambda x: self.assertTrue((x == 4.5).all()))

        tmp[..., 0] = self.batch[..., 0] - np.array([4.5])
        recursive_apply(tmp[..., 1:], lambda x: self.assertTrue((x == 4.5).all()))
        self.assertTrue(np.all(np.abs(tmp.action[..., 0] + 4.5 - self.batch.action[..., 0]) < 1e-8))

    def test_set_item(self):
        tmp = deepcopy(self.batch)
        # simple slicing
        self.batch[0] = 1
        recursive_apply(self.batch, lambda x: self.assertTrue(np.all(x[0] == 1)))
        # ":" slicing
        self.batch = tmp + 1
        batch_mean_ = recursive_apply(self.batch, np.mean)
        self.assertTrue(np.all(np.abs(self.batch_mean.action + 1 - batch_mean_.action) < 1e-8))

        # "-seg:" slicing
        self.batch[:] = 0
        seg = T // 4
        micro_obs = Observation(x=np.ones((seg, bs, 20)),
                                y=np.ones((seg, bs, 30)),
                                mask=np.ones((seg, bs, 1)))
        micro_batch = SampleBatch(obs=micro_obs,
                                  action=np.ones((seg, bs, 1)),
                                  log_probs=None,
                                  policy_state=None)
        self.batch[-seg:] = micro_batch
        recursive_apply(self.batch, lambda x: self.assertTrue(np.all(x[-seg:] == 1)))
        recursive_apply(self.batch, lambda x: self.assertEqual(x.mean(), 0.25))
        self.batch[:] = 0
        recursive_apply(self.batch, lambda x: self.assertTrue(np.all(x == 0)))
        self.batch[:] = tmp

        # set item using key
        new_obs = Observation(x=np.random.randn(T, bs, 20),
                              y=np.random.randn(T, bs, 30),
                              mask=np.random.randn(T, bs, 1))
        self.batch['obs'] = new_obs
        self.assertEqual(recursive_apply(self.batch, np.mean).obs.x, recursive_apply(new_obs, np.mean).x)

        # set item using attributes
        new_obs = Observation(x=np.random.randn(T, bs, 20),
                              y=np.random.randn(T, bs, 30),
                              mask=np.random.randn(T, bs, 1))
        self.batch.obs = new_obs
        self.assertEqual(recursive_apply(self.batch, np.mean).obs.x, recursive_apply(new_obs, np.mean).x)

        # setting item with wrong data structure will raise a value error
        wrong_batch = WrongSampleBatch(action=np.random.randn(2, bs, 1),
                                       log_probs=np.random.randn(2, bs, 5),
                                       reward=np.random.randn(2, bs, 1))
        with self.assertRaises(ValueError):
            self.batch[:2] = wrong_batch
        with self.assertRaises(AttributeError):
            self.batch.aa = np.zeros(2)

        self.batch.log_probs = np.zeros_like(self.batch.log_probs)
        self.batch.register_metadata(a=1, b=2)
        with self.assertRaises(AttributeError):
            self.batch.a = 3

        self.batch.clear_metadata()
        self.batch[:] = tmp

    def test_stacking(self):
        batch2 = deepcopy(self.batch)
        batch2[:] = 1
        xs = [self.batch, batch2]
        stackx = recursive_aggregate(xs, lambda x: np.stack(x, axis=-1))
        # print(type(stackx))
        recursive_apply(stackx, lambda x: self.assertEqual(x.shape[-1], 2))

        recursive_apply(stackx, lambda x: self.assertEqual(len(x.shape), 4))
        batch_mean_ = recursive_apply(stackx, np.mean)
        self.assertTrue(np.all(np.abs(self.batch_mean.action / 2 + 0.5 - batch_mean_.action) < 1e-8))

    def test_concat(self):
        batch2 = deepcopy(self.batch)
        batch2[:] = 1
        xs = [self.batch, batch2]
        concatenatex = recursive_aggregate(xs, lambda x: np.concatenate(x, axis=1))
        recursive_apply(concatenatex, lambda x: self.assertEqual(x.shape[1], 2 * bs))
        recursive_apply(concatenatex, lambda x: self.assertEqual(len(x.shape), 3))
        batch_mean_ = recursive_apply(concatenatex, np.mean)
        self.assertTrue(np.all(np.abs(self.batch_mean.action / 2 + 0.5 - batch_mean_.action) < 1e-8))

    def test_unpacking(self):

        def fn(obs, action, log_probs, policy_state):
            # self.assertIsInstance(obs, Observation)
            self.assertTrue(np.all(action.mean() == self.batch_mean.action))
            self.assertIs(policy_state, None)

        fn(**self.batch)

    #
    # def test_name(self):
    #     self.assertEqual(type(self.batch).__name__, 'SampleBatch')
    #     self.assertEqual(type(self.batch.obs).__name__, 'Observation')

    def test_shape(self):
        shape = self.batch.shape
        self.assertIsInstance(shape, dict)
        self.assertIsInstance(shape['obs'], dict)
        self.assertEqual(shape['obs']['x'], (T, bs, 20))
        self.assertEqual(shape['obs']['y'], (T, bs, 30))
        self.assertEqual(shape['obs']['mask'], (T, bs, 1))
        self.assertEqual(shape['action'], (T, bs, 1))
        self.assertIs(shape['log_probs'], None)
        self.assertIs(shape['policy_state'], None)

    def test_to_dict(self):
        batch = self.batch
        dict_obs = self.batch.to_dict()
        self.assertIsInstance(dict_obs, Dict)
        self.assertIsInstance(dict_obs['obs'], Dict)
        self.assertTrue((dict_obs['obs']['x'] == batch.obs.x).all())
        self.assertTrue((dict_obs['obs']['y'] == batch.obs.y).all())
        self.assertTrue((dict_obs['obs']['mask'] == batch.obs.mask).all())
        self.assertTrue((dict_obs['action'] == batch.action).all())
        self.assertIs(dict_obs['log_probs'], None)
        self.assertIs(dict_obs['policy_state'], None)

    def test_serialization(self):

        def assert_namedarray_close(a1, a2):
            if isinstance(a1, NamedArray) and isinstance(a2, NamedArray):
                for k, v in a1.items():
                    assert k in a2.keys()
                    if isinstance(v, np.ndarray):
                        np.testing.assert_allclose(v, a2[k])
                    elif isinstance(v, torch.Tensor):
                        torch.testing.assert_allclose(v, a2[k])
                    else:
                        assert_namedarray_close(v, a2[k])
            else:
                if not a1 == a2:
                    raise TypeError(
                        f"Left and right should both be namedarray instance. Got {type(a1)} and {type(a2)}")

        self.batch.register_metadata(a=1, b='xxx')
        for m in base.namedarray.NamedArrayEncodingMethod:
            batch = self.batch
            b = base.namedarray.dumps(batch, method=m.name.lower())
            batch_ = base.namedarray.loads(b)
            assert_namedarray_close(batch, batch_)
            assert_namedarray_close(batch.metadata, batch_.metadata)

    def test_metadata(self):
        with self.assertRaises(KeyError):
            self.batch.register_metadata(obs=3)

        self.batch.register_metadata(a=1, b='xxx', c=np.zeros(3))
        self.assertIsInstance(self.batch.metadata, types.MappingProxyType)
        self.assertEqual(self.batch.metadata['a'], 1)
        self.assertEqual(self.batch.metadata['b'], 'xxx')
        self.assertEqual(len(self.batch[0].metadata), 0)

        self.batch.register_metadata(b='abc')
        self.assertEqual(self.batch.metadata['b'], 'abc')

        v = self.batch.pop_metadata('c')
        np.testing.assert_array_equal(v, np.zeros(3))
        with self.assertRaises(KeyError):
            self.batch.metadata['c']

        batch_ = recursive_apply(self.batch, np.mean)
        self.assertIs(len(batch_.metadata), 0)
        self.batch.clear_metadata()


if __name__ == '__main__':
    unittest.main()
