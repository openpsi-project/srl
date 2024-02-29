import mock
import os
import torch
import torch.distributed as dist
import platform
import unittest

from base.gpu_utils import resolve_cuda_environment, get_gpu_device


class TestNvidiaEnvironment(unittest.TestCase):

    def setUp(self):
        platform.system = mock.MagicMock(return_value="Linux")
        os.listdir = mock.MagicMock(return_value=([f"nvidia{i}" for i in range(8)]))
        torch.cuda.is_available = mock.MagicMock(return_value=True)
        dist.is_nccl_available = mock.MagicMock(return_value=True)

    def reset_env(self):
        try:
            os.environ.pop("MARL_CUDA_DEVICES")
        except KeyError:
            pass
        try:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        except KeyError:
            pass

    def test_gpu_assignment(self):
        self.reset_env()
        resolve_cuda_environment()
        self.assertEqual(get_gpu_device(), ["cuda:0"])
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.reset_env()
        resolve_cuda_environment()
        self.assertEqual(get_gpu_device(), ["cuda:0"])

        self.reset_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        resolve_cuda_environment()
        self.assertListEqual(get_gpu_device(), ["cuda:1"])

        self.reset_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        resolve_cuda_environment()
        self.assertSetEqual(set(get_gpu_device()), {"cuda:1", "cuda:0"})


class TestCPUEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        os.listdir = mock.MagicMock(return_value=(["random_file1", "random_file2"]))
        torch.cuda.is_available = mock.MagicMock(return_value=False)
        dist.is_nccl_available = mock.MagicMock(return_value=False)
        try:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        except KeyError:
            pass

    def reset_env(self):
        try:
            os.environ.pop("MARL_CUDA_DEVICES")
        except KeyError:
            pass
        try:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        except KeyError:
            pass

    def test_gpu_assignment(self):
        self.reset_env()
        resolve_cuda_environment()
        self.assertEqual(get_gpu_device(), ["cpu"])

        self.reset_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        resolve_cuda_environment()
        self.assertEqual(get_gpu_device(), ["cpu"])

        self.reset_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        self.assertRaises(AssertionError, resolve_cuda_environment)


if __name__ == '__main__':
    unittest.main()
