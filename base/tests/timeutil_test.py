import time
import mock
import unittest

import base.timeutil


class MyTestCase(unittest.TestCase):

    def set_time(self, value):
        time.monotonic = mock.MagicMock(return_value=value)

    def test_timeutil_frequency(self):
        self.set_time(0.)
        freq_control = base.timeutil.FrequencyControl(frequency_seconds=1)
        self.set_time(0.8)
        self.assertFalse(freq_control.check())
        self.set_time(0.9)
        self.assertFalse(freq_control.check())

        self.set_time(1.2)
        self.assertTrue(freq_control.check())
        self.set_time(2.1)
        self.assertFalse(freq_control.check())

        self.set_time(2.3)
        self.assertTrue(freq_control.check())

    def test_timeutil_step(self):
        freq_control = base.timeutil.FrequencyControl(frequency_steps=10)
        self.assertFalse(freq_control.check(steps=0))
        self.assertTrue(freq_control.check(steps=12))
        for i in range(9):
            self.assertFalse(freq_control.check())
        self.assertTrue(freq_control.check())

    def test_timeutil_freq_and_step(self):
        self.set_time(0)
        freq_control = base.timeutil.FrequencyControl(frequency_seconds=1, frequency_steps=10)
        self.assertFalse(freq_control.check(steps=9999))
        self.set_time(2)
        self.assertTrue(freq_control.check())
        self.assertFalse(freq_control.check())
        self.set_time(4)
        self.assertFalse(freq_control.check())
        self.assertTrue(freq_control.check(8))

    def test_timeutil_initial(self):
        self.set_time(0)
        freq_control = base.timeutil.FrequencyControl(frequency_seconds=1,
                                                      frequency_steps=10,
                                                      initial_value=True)
        self.assertTrue(freq_control.check())
        self.assertFalse(freq_control.check(9))
        self.set_time(1.01)
        self.assertTrue(freq_control.check())

    def test_frequency_never(self):
        self.set_time(0)
        freq_control = base.timeutil.FrequencyControl(frequency_seconds=None,
                                                      frequency_steps=None,
                                                      initial_value=True)
        self.assertTrue(freq_control.check())
        self.assertFalse(freq_control.check(99999))
        self.set_time(99999)
        self.assertFalse(freq_control.check())

    def test_scheduler(self):
        with self.assertRaises(ValueError):
            scheduler = base.timeutil.LinearScheduler(init_value=0, total_iters=-1, end_value=1)
        scheduler = base.timeutil.LinearScheduler(init_value=0, total_iters=10, end_value=1)
        self.assertEqual(scheduler.get(step=1), 0.1)
        self.assertEqual(scheduler.get(step=9), 0.9)
        self.assertEqual(scheduler.final_value, 1)
        with self.assertRaises(ValueError):
            scheduler.get(1001)
        with self.assertRaises(ValueError):
            scheduler.get(-1)

        scheduler2 = base.timeutil.CosineDecayScheduler(init_value=1, total_iters=10, end_value=0.1)
        self.assertEqual(scheduler2.get(step=5), 0.55)

        scheduler3 = base.timeutil.ExponentialScheduler(init_value=0.1, total_iters=6, decay=0.9)
        self.assertEqual(scheduler3.get(step=3), 0.1 * 0.9**3)

        chained = base.timeutil.ChainedScheduler(
            [scheduler,
             base.timeutil.ConstantScheduler(init_value=1, total_iters=4), scheduler2, scheduler3])
        for step in range(30):
            self.assertEqual(
                chained.get(step),
                scheduler.get(step) if step <= 10 else
                1 if step <= 14 else scheduler2.get(step - 14) if step <= 24 else scheduler3.get(step - 24),
            )


if __name__ == '__main__':
    unittest.main()
