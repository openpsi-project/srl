import unittest
import numpy as np
import torch

from base.conditions import make as make_condition
import api.config


def make_test_condition(type_, args):
    return make_condition(api.config.Condition(type_=type_, args=args))


class TestMathematical(unittest.TestCase):

    def sample_data(self):
        return {"win_rate": 75}

    def sample_numpy_data(self):
        return {"win_rate": np.array([50, 100])}

    def sample_torch_data(self):
        return {"win_rate": torch.Tensor([50, 100])}

    def test_condition(self):
        s = api.config.Condition.Type.SimpleBound
        condition_1 = make_test_condition(s, dict(field="win_rate", lower_limit=50, upper_limit=100))
        condition_2 = make_test_condition(s, dict(field="win_rate", lower_limit=50, upper_limit=None))
        condition_3 = make_test_condition(s, dict(field="win_rate", lower_limit=None, upper_limit=100))
        condition_4 = make_test_condition(s, dict(field="win_rate", lower_limit=None, upper_limit=None))
        condition_5 = make_test_condition(s, dict(field="win_rate", lower_limit=100, upper_limit=None))
        condition_6 = make_test_condition(s, dict(field="win_rate", lower_limit=10, upper_limit=74))

        for c in [condition_1, condition_2, condition_3, condition_4]:
            self.assertTrue(c.is_met_with(self.sample_data()))
            self.assertTrue(c.is_met_with(self.sample_numpy_data()))
            self.assertTrue(c.is_met_with(self.sample_torch_data()))

        for c in [condition_5, condition_6]:
            self.assertFalse(c.is_met_with(self.sample_data()))
            self.assertFalse(c.is_met_with(self.sample_numpy_data()))
            self.assertFalse(c.is_met_with(self.sample_torch_data()))

        test_condition = make_test_condition(s, dict(field="loss_rate", lower_limit=50, upper_limit=100))
        self.assertRaises(ValueError, test_condition.is_met_with, self.sample_data())


class TestConvergedCondition(unittest.TestCase):

    def test_converged(self):
        warmup_step = 10
        duration = 20
        confidence = 0.9
        threshold = 0.1
        condition = make_test_condition(
            api.config.Condition.Type.Converged,
            dict(value_field="reward",
                 step_field="version",
                 warmup_step=warmup_step,
                 duration=duration,
                 confidence=confidence,
                 threshold=threshold))

        # Test warm up.
        version = 0
        for _ in range(warmup_step):
            self.assertFalse(condition.is_met_with(dict(reward=0, version=version)))
            version += 1

        # Test duration.
        for _ in range(duration):
            self.assertFalse(condition.is_met_with(dict(reward=0, version=version)))
            version += 1
        self.assertTrue(condition.is_met_with(dict(reward=0, version=version)))
        version += 1

        # Test confidence and threshold
        num_ignore = int(duration * (1 - confidence))
        for _ in range(num_ignore):
            self.assertFalse(condition.is_met_with(dict(reward=2 * threshold, version=version)))
            version += 1
        self.assertTrue(condition.is_met_with(dict(reward=0, version=version)))
        version += 1
        self.assertFalse(condition.is_met_with(dict(reward=2, version=version)))
        version += 1
        self.assertFalse(condition.is_met_with(dict(reward=2 * threshold, version=version)))


if __name__ == '__main__':
    unittest.main()
