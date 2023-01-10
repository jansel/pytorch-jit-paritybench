import sys
_module = sys.modules[__name__]
del sys
Breakout_DQN_class = _module
Breakout_PolicyGradient = _module
Play_DQN = _module
breakout_dqn_pytorch = _module
CartPole_A2C_episodic = _module
CartPole_C51 = _module
CartPole_DDQN = _module
CartPole_DQN_NIPS2013 = _module
CartPole_DQN_Nature2015 = _module
CartPole_PAAC = _module
CartPole_PAAC_multiproc = _module
CartPole_PolicyGradient = _module
Cartpole_A2C_nstep = _module
Cartpole_A2C_onestep = _module
cartpole_dqn = _module
cartpole_ppo = _module
play_Cartpole = _module
Pong_A2C_episodic = _module
Pong_PolicyGradient = _module
windygridworld = _module
pendulum_ddpg = _module
pendulum_ppo = _module

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


import random


import numpy as np


from collections import deque


from copy import deepcopy


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


import torch.multiprocessing as mp


from torch.distributions.categorical import Categorical


from torch.distributions.normal import Normal


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class DQN(nn.Module):

    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(), Flatten(), nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, action_size))

    def forward(self, x):
        return self.fc(x)


class ActorCriticNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_size, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.mu = nn.Linear(256, output_size)
        self.critic = nn.Linear(256, 1)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.critic.weight.data.mul_(0.1)
        self.critic.bias.data.mul_(0.0)

    def forward(self, state):
        x = self.feature(state)
        mu = self.mu(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        value = self.critic(x)
        return mu, std, logstd, value


class Actor(nn.Module):

    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Actor, self).__init__()
        self.network = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, action_size), nn.Tanh())

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):

    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Critic, self).__init__()
        self.before_action = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU())
        self.after_action = nn.Sequential(nn.Linear(400 + action_size, 300), nn.ReLU(), nn.Linear(300, 1))

    def forward(self, x, action):
        x = self.before_action(x)
        x = torch.cat([x, action], dim=1)
        x = self.after_action(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Actor,
     lambda: ([], {'obs_size': 4, 'action_size': 4, 'action_range': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ActorCriticNetwork,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Critic,
     lambda: ([], {'obs_size': 4, 'action_size': 4, 'action_range': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jcwleo_Reinforcement_Learning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

