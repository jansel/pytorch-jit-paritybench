import sys
_module = sys.modules[__name__]
del sys
master = _module
agents = _module
a2c = _module
common = _module
buffers = _module
networks = _module
utils = _module
ddpg = _module
dqn = _module
ppo = _module
sac = _module
td3 = _module
trpo = _module
vpg = _module
run_cartpole = _module
run_mujoco = _module
run_pendulum = _module
cartpole_test = _module
networks = _module
utils = _module
mujoco_test = _module
pendulum_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.optim as optim


import torch.nn.functional as F


import torch.nn as nn


from torch.distributions import Categorical


from torch.distributions import Normal


import time


from torch.utils.tensorboard import SummaryWriter


def identity(x):
    """Return input without any change."""
    return x


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=F.relu, output_activation=identity, use_output_layer=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x


class CategoricalPolicy(MLP):

    def forward(self, x):
        x = super(CategoricalPolicy, self).forward(x)
        pi = F.softmax(x, dim=-1)
        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        return action, pi, log_pi


class FlattenMLP(MLP):

    def forward(self, x, a):
        q = torch.cat([x, a], dim=-1)
        return super(FlattenMLP, self).forward(q)


class GaussianPolicy(MLP):

    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=torch.tanh):
        super(GaussianPolicy, self).__init__(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes, activation=activation)

    def forward(self, x, pi=None):
        mu = super(GaussianPolicy, self).forward(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if pi == None:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)
        return mu, std, pi, log_pi


LOG_STD_MAX = 2


LOG_STD_MIN = -20


class ReparamGaussianPolicy(MLP):

    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=F.relu, action_scale=1.0, log_type='log', q=1.5, device=None):
        super(ReparamGaussianPolicy, self).__init__(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes, activation=activation, use_output_layer=False)
        in_size = hidden_sizes[-1]
        self.action_scale = action_scale
        self.log_type = log_type
        self.q = 2.0 - q
        self.device = device
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)

    def clip_but_pass_gradient(self, x, l=-1.0, u=1.0):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x) * clip_up + (l - x) * clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        if self.log_type == 'log':
            log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0.0, u=1.0) + 1e-06), dim=-1)
        elif self.log_type == 'log-q':
            log_pi -= torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0.0, u=1.0) + 1e-06)
        return mu, pi, log_pi

    def tsallis_entropy_log_q(self, x, q):
        safe_x = torch.max(x, torch.Tensor([1e-06]))
        log_q_x = torch.log(safe_x) if q == 1.0 else (safe_x.pow(1 - q) - 1) / (1 - q)
        return log_q_x.sum(dim=-1)

    def forward(self, x):
        x = super(ReparamGaussianPolicy, self).forward(x)
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        pi = dist.rsample()
        if self.log_type == 'log':
            log_pi = dist.log_prob(pi).sum(dim=-1)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
        elif self.log_type == 'log-q':
            log_pi = dist.log_prob(pi)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
            exp_log_pi = torch.exp(log_pi)
            log_pi = self.tsallis_entropy_log_q(exp_log_pi, self.q)
        mu = mu * self.action_scale
        pi = pi * self.action_scale
        return mu, pi, log_pi


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CategoricalPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReparamGaussianPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_dongminlee94_deep_rl(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

