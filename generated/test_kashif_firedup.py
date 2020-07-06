import sys
_module = sys.modules[__name__]
del sys
fireup = _module
algos = _module
core = _module
ddpg = _module
core = _module
dqn = _module
core = _module
ppo = _module
core = _module
sac = _module
core = _module
td3 = _module
core = _module
trpo = _module
vpg = _module
core = _module
vpg = _module
bench_ppo_cartpole = _module
bench_vpg_cartpole = _module
run = _module
user_config = _module
utils = _module
logx = _module
mpi_tools = _module
mpi_torch = _module
plot = _module
run_entrypoint = _module
run_utils = _module
serialization_utils = _module
test_policy = _module
version = _module
setup = _module
test_ppo = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


from torch.distributions.categorical import Categorical


from torch.distributions.normal import Normal


import scipy.signal


from torch.nn.utils import parameters_to_vector


from torch.distributions.kl import kl_divergence


from torch.nn.utils import vector_to_parameters


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=None, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation, output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.logits = MLP(layers=[in_features] + list(hidden_sizes) + [action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None
        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation, output_activation, action_dim):
        super(GaussianPolicy, self).__init__()
        self.mu = MLP(layers=[in_features] + list(hidden_sizes) + [action_dim], activation=activation, output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(64, 64), activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(in_features, hidden_sizes, activation, output_activation, action_dim=action_space.shape[0])
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(in_features, hidden_sizes, activation, output_activation, action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation, output_activation, action_space)
        self.value_function = MLP(layers=[in_features] + list(hidden_sizes) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_function(x)
        return pi, logp, logp_pi, v


class DQNetwork(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(400, 300), activation=torch.relu, output_activation=None):
        super(DQNetwork, self).__init__()
        action_dim = action_space.n
        self.q = MLP(layers=[in_features] + list(hidden_sizes) + [action_dim], activation=activation, output_activation=output_activation)

    def forward(self, x):
        return self.q(x)

    def policy(self, x):
        return torch.argmax(self.q(x), dim=1, keepdim=True)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kashif_firedup(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

