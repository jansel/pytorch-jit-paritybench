import sys
_module = sys.modules[__name__]
del sys
conf = _module
a2c = _module
ddpg = _module
dqn = _module
ppo = _module
test_examples = _module
setup = _module
torchrl = _module
contrib = _module
controllers = _module
a2c_controller = _module
ddpg_controller = _module
dqn_controller = _module
ppo_controller = _module
models = _module
controller = _module
test_controller = _module
envs = _module
env_utils = _module
parallel_envs = _module
test_env_utils = _module
test_wrappers = _module
wrappers = _module
experiments = _module
base_experiment = _module
test_base_experiment = _module
utils = _module
schedule = _module
storage = _module
test_schedule = _module
test_storage = _module

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


import torch


from torch.utils.data import DataLoader


import numpy as np


from torch import nn


from torch.distributions import Normal


from torch.distributions import Categorical


import functools


from torch.utils.data import Dataset


from typing import List


from typing import Optional


from collections import namedtuple


import random


class QNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = 128
        self.net = nn.Sequential(nn.Linear(self._input_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, self._output_size))

    def forward(self, obs):
        values = self.net(obs)
        return values


class DDPGActorNet(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(DDPGActorNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self._weight_init = 0.003
        self.net = nn.Sequential(nn.Linear(self.state_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.action_size), nn.Tanh())
        self._init_weights()

    def forward(self, obs):
        return self.net(obs)

    def _init_weights(self):
        nn.init.uniform_(self.net[-2].weight, -self._weight_init, self._weight_init)
        nn.init.uniform_(self.net[-2].bias, -self._weight_init, self._weight_init)


class DDPGCriticNet(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(DDPGCriticNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self._weight_init = 0.003
        self.net = nn.Sequential(nn.Linear(self.state_size + self.action_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1))
        self._init_weights()

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=1))

    def _init_weights(self):
        nn.init.uniform_(self.net[-1].weight, -self._weight_init, self._weight_init)
        nn.init.uniform_(self.net[-1].bias, -self._weight_init, self._weight_init)


class A2CNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(A2CNet, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self.critic = nn.Sequential(nn.Linear(self._input_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, 1))
        self.actor = nn.Sequential(nn.Linear(self._input_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, self._output_size), nn.Softmax(dim=1))

    def forward(self, obs):
        value = self.critic(obs)
        policy = self.actor(obs)
        dist = Categorical(policy)
        return value, dist


class ActorCriticNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, std=0.0):
        super(ActorCriticNet, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self.critic = nn.Sequential(nn.Linear(self._input_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, 1))
        self.actor = nn.Sequential(nn.Linear(self._input_size, self._hidden_size), nn.ReLU(), nn.Linear(self._hidden_size, self._output_size))
        self.log_std = nn.Parameter(torch.ones(1, self._output_size) * std)
        self.apply(self.init_weights)

    def forward(self, obs):
        value = self.critic(obs)
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return value, dist

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            nn.init.constant_(module.bias, 0.1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (A2CNet,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ActorCriticNet,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DDPGActorNet,
     lambda: ([], {'state_size': 4, 'action_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DDPGCriticNet,
     lambda: ([], {'state_size': 4, 'action_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (QNet,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_activatedgeek_torchrl(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

