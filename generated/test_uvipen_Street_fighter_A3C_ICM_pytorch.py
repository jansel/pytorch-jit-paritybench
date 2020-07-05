import sys
_module = sys.modules[__name__]
del sys
env = _module
model = _module
optimizer = _module
process = _module
test = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Categorical


class BaseConv(nn.Module):

    def __init__(self, num_inputs):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_inputs, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv = BaseConv(num_inputs)
        self.lstm = nn.LSTMCell(64 * 6 * 6, 1024)
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = self.conv(x)
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx


class IntrinsicCuriosityModule(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.conv = BaseConv(num_inputs)
        self.feature_size = 64 * 6 * 6
        self.inverse_net = nn.Sequential(nn.Linear(self.feature_size * 2, 1024), nn.LeakyReLU(), nn.Linear(1024, num_actions))
        self.forward_net = nn.Sequential(nn.Linear(self.feature_size + num_actions, 1024), nn.LeakyReLU(), nn.Linear(1024, self.feature_size))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(torch.cat((state_ft, action), 1)), next_state_ft


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseConv,
     lambda: ([], {'num_inputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_uvipen_Street_fighter_A3C_ICM_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

