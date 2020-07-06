import sys
_module = sys.modules[__name__]
del sys
ddpg = _module
dqn = _module
env = _module
hyperparams = _module
models = _module
ppo = _module
sac = _module
td3 = _module
trpo = _module
utils = _module
vpg = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from collections import deque


import random


import torch


from torch import optim


import copy


from torch import nn


from torch.distributions import Distribution


from torch.distributions import Normal


from functools import partial


from torch import autograd


from torch.distributions.kl import kl_divergence


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class TanhNormal(Distribution):

    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)

    def sample(self):
        return torch.tanh(self.normal.sample())

    def rsample(self):
        return torch.tanh(self.normal.rsample())

    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-06)

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
        policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
        policy = TanhNormal(policy_mean, policy_log_std.exp())
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        return policy, value


class DQN(nn.Module):

    def __init__(self, hidden_size, num_actions=5):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, num_actions)]
        self.dqn = nn.Sequential(*layers)

    def forward(self, state):
        values = self.dqn(state)
        return values


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Actor,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([3, 3])], {}),
     True),
    (ActorCritic,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([3, 3])], {}),
     False),
    (Critic,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([3, 3])], {}),
     False),
    (DQN,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([3, 3])], {}),
     True),
    (SoftActor,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([3, 3])], {}),
     False),
]

class Test_Kaixhin_spinning_up_basic(_paritybench_base):
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

