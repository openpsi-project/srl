import unittest

import numpy as np

from api.curriculum import make as make_curriculum
from base import name_resolve
from base.names import curriculum_stage
import api.config
import base.name_resolve


def make_test_curriculum():
    test_config = api.config.Curriculum(type_=api.config.Curriculum.Type.Linear,
                                        name="test_curriculum",
                                        stages=["training1", "training2"],
                                        conditions=[
                                            api.config.Condition(type_=api.config.Condition.Type.SimpleBound,
                                                                 args=dict(field="win_rate",
                                                                           lower_limit=0.9,
                                                                           upper_limit=None)),
                                            api.config.Condition(type_=api.config.Condition.Type.SimpleBound,
                                                                 args=dict(field="episode_return",
                                                                           lower_limit=10,
                                                                           upper_limit=None))
                                        ])

    return make_curriculum(test_config,
                           api.config.WorkerInformation(experiment_name="test_exp", trial_name="test_run"))


class MyTestCase(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory")

    def test_linear_curr(self):
        curriculum = make_test_curriculum()
        curriculum.reset()
        self.assertEqual(curriculum.get_stage(), "training1")
        self.assertFalse(curriculum.submit({"win_rate": 0.5, "episode_return": 15}))
        self.assertEqual(curriculum.get_stage(), "training1")
        self.assertFalse(
            curriculum.submit({
                "win_rate": np.array([0.9, 0.95, 0.98]),
                "episode_return": np.array([15, 20, 25])
            }))
        self.assertEqual(base.name_resolve.get(curriculum_stage("test_exp", "test_run", "test_curriculum")),
                         "training2")
        self.assertNotEqual(
            base.name_resolve.get(curriculum_stage("test_exp", "test_run", "test_curriculum")), "training1")
        self.assertTrue(
            curriculum.submit({
                "win_rate": np.array([0.9, 0.95, 0.98]),
                "episode_return": np.array([15, 20, 25])
            }))


if __name__ == '__main__':
    unittest.main()
