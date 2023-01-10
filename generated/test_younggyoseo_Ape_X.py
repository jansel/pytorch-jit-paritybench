import sys
_module = sys.modules[__name__]
del sys
actor = _module
arguments = _module
enjoy = _module
eval = _module
learner = _module
memory = _module
model = _module
replay = _module
utils = _module
wrapper = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import queue


import torch


import numpy as np


import time


import torch.multiprocessing as mp


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


import random


import torch.nn as nn


class Flatten(nn.Module):
    """
    Simple module for flattening parameters
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init(module):
    return init_(module, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


class DuelingDQN(nn.Module):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """

    def __init__(self, env):
        super(DuelingDQN, self).__init__()
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.flatten = Flatten()
        self.features = nn.Sequential(init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)), nn.ReLU(), init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(), init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU())
        self.advantage = nn.Sequential(init(nn.Linear(self._feature_size(), 512)), nn.ReLU(), init(nn.Linear(512, self.num_actions)))
        self.value = nn.Sequential(init(nn.Linear(self._feature_size(), 512)), nn.ReLU(), init(nn.Linear(512, 1)))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.forward(state)
            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.numpy()[0]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_younggyoseo_Ape_X(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

