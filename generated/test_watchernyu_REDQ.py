import sys
_module = sys.modules[__name__]
del sys
grid_train_redq_sac = _module
grid_utils = _module
train_redq_sac = _module
plot_REDQ = _module
redq_plot_helper = _module
redq = _module
algos = _module
core = _module
redq_sac = _module
user_config = _module
utils = _module
bias_utils = _module
logx = _module
run_utils = _module
serialization_utils = _module
setup = _module

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


import numpy as np


import torch


import time


import torch.nn as nn


from torch.nn import functional as F


from torch.distributions import Distribution


from torch.distributions import Normal


from torch import Tensor


import torch.optim as optim


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Mlp(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation=F.relu):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_layers = nn.ModuleList()
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output


ACTION_BOUND_EPSILON = 1e-06


LOG_SIG_MAX = 2


LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes, hidden_activation=F.relu, action_limit=1.0):
        super().__init__(input_size=obs_dim, output_size=action_dim, hidden_sizes=hidden_sizes, hidden_activation=hidden_activation)
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(self, obs, deterministic=False, return_log_prob=True):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, take determinisitc (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)
        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None
        return action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Mlp,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TanhGaussianPolicy,
     lambda: ([], {'obs_dim': 4, 'action_dim': 4, 'hidden_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_watchernyu_REDQ(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

