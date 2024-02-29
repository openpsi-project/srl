"""Testing only utilities. This module shall not be imported to production code.
"""
import random
import os
import sys
import time
import mock
import torch

_IS_GITHUB_WORKFLOW = len(os.environ.get("CI", "").strip()) > 0
os.environ["MARL_CUDA_DEVICES"] = "cpu"
_DEFAULT_WAIT_NETWORK_SECONDS = 0.5 if _IS_GITHUB_WORKFLOW else 0.05

_next_port = 20000 + random.randint(0, 10000)  # Random port for now, should be ok most of the time.


def get_testing_port():
    """Returns a local port for testing."""
    global _next_port
    _next_port += 1
    return _next_port


def wait_network(length=_DEFAULT_WAIT_NETWORK_SECONDS):
    time.sleep(length)


def get_test_param(version=0):
    return {
        "steps": version,
        "state_dict": {
            "linear_weights": torch.randn(10, 10)
        },
    }


# TODO: write a child class of unittest.
sys.modules['gfootball'] = mock.Mock()
sys.modules['gfootball.env'] = mock.Mock()
