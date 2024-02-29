import numpy as np
import unittest

import base.numpy_utils


class NumpyUtilsTest(unittest.TestCase):

    def test_moving_avg(self):
        a = np.random.randn(100)
        for window_size in np.random.randint(1, 101, (10,)):
            b1 = base.numpy_utils.moving_average(a, window_size)
            b2 = np.array([a[i:i + window_size] for i in range(len(a) - window_size + 1)])
            b2 = np.mean(b2, 1)
            np.testing.assert_almost_equal(b1, b2)

        with self.assertRaises(ValueError):
            base.numpy_utils.moving_average(np.random.randn(3, 4), 1)
        with self.assertRaises(ValueError):
            base.numpy_utils.moving_average(np.random.randn(3), 10)

    def test_moving_max(self):
        a = np.random.randn(100)
        for window_size in np.random.randint(1, 101, (10,)):
            b1 = base.numpy_utils.moving_maximum(a, window_size)
            b2 = np.array([a[i:i + window_size] for i in range(len(a) - window_size + 1)])
            b2 = np.max(b2, 1)
            np.testing.assert_almost_equal(b1, b2)

        with self.assertRaises(ValueError):
            base.numpy_utils.moving_maximum(np.random.randn(3, 4), 1)
        with self.assertRaises(ValueError):
            base.numpy_utils.moving_maximum(np.random.randn(3), 10)

    def test_split_to_shapes(self):
        shapes = dict(a=(3, 4), b=(5, 6))
        x = np.random.randn(2, 4, 42)
        splited = base.numpy_utils.split_to_shapes(x, shapes, -1)
        x_ = np.concatenate([splited['a'].reshape(2, 4, 12), splited['b'].reshape(2, 4, 30)], -1)
        np.testing.assert_almost_equal(x, x_)

        x = np.random.randn(42, 4, 2)
        splited = base.numpy_utils.split_to_shapes(x, shapes, 0)
        x_ = np.concatenate([splited['a'].reshape(12, 4, 2), splited['b'].reshape(30, 4, 2)], 0)
        np.testing.assert_almost_equal(x, x_)


if __name__ == "__main__":
    unittest.main()
