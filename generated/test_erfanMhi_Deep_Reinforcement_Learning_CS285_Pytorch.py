import sys
_module = sys.modules[__name__]
del sys
agents = _module
base_agent = _module
bc_agent = _module
infrastructure = _module
logger = _module
replay_buffer = _module
rl_trainer = _module
torch_utils = _module
utils = _module
MLP_policy = _module
policies = _module
base_policy = _module
loaded_gaussian_policy = _module
run_hw1_behavior_cloning = _module
setup = _module
pg_agent = _module
rl_trainer = _module
tf_utils = _module
torch_utils = _module
MLP_policy = _module
loaded_gaussian_policy = _module
run_hw2_policy_gradient = _module
ac_agent = _module
dqn_agent = _module
critics = _module
base_critic = _module
bootstrapped_continuous_critic = _module
dqn_critic = _module
atari_wrappers = _module
dqn_utils = _module
rl_trainer = _module
torch_utils = _module
MLP_policy = _module
argmax_policy = _module
run_hw3_actor_critic = _module
run_hw3_dqn = _module
mb_agent = _module
bootstrapped_continuous_critic = _module
dqn_critic = _module
envs = _module
ant = _module
cheetah = _module
obstacles = _module
obstacles_env = _module
reacher = _module
reacher_env = _module
dqn_utils = _module
rl_trainer = _module
torch_utils = _module
base_model = _module
ff_model = _module
MPC_policy = _module
argmax_policy = _module
filter_events = _module
run_hw4_mb = _module
density_model = _module
ex_utils = _module
exploration = _module
logz = _module
plot = _module
pointmass = _module
replay = _module
sparse_half_cheetah = _module
pytorch_utils = _module

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


import time


from collections import OrderedDict


import numpy as np


import torch


import torch.nn as nn


import functools


import inspect


import torch.nn.functional as F


from torch import optim


from torch.distributions import Categorical


from torch import nn


import random


from collections import namedtuple


import tensorflow as tf


class MLP(nn.Module):

    def __init__(self, input_size, output_size, n_layers, size, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.size = size
        self.n_layers = n_layers
        self.output_activation = output_activation
        layers_size = [self.input_size] + [self.size] * self.n_layers + [self.output_size]
        self.layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1]) for i in range(len(layers_size) - 1)])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                out = self.activation(layer(out))
            else:
                out = layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class LoadedGaussianPolicy(nn.Module):

    def __init__(self, layer_params, out_layer_params, activation_func, obsnorm_mean, obsnorm_stdev):
        super(LoadedGaussianPolicy, self).__init__()
        self.obsnorm_mean = obsnorm_mean
        self.obsnorm_stdev = obsnorm_stdev
        self.activation_func = activation_func
        self.layers = nn.ModuleList()
        for layer_name in sorted(layer_params.keys()):
            W, b = self._read_layer(layer_params[layer_name])
            height, width = W.shape
            layer = nn.Linear(height, width)
            layer.weight.data = torch.from_numpy(W.transpose())
            layer.bias.data = torch.from_numpy(b.squeeze(0))
            self.layers.append(layer)
        W, b = self._read_layer(out_layer_params)
        height, width = W.shape
        self.out_layer = nn.Linear(height, width)
        self.out_layer.weight.data = torch.from_numpy(W.transpose())
        self.out_layer.bias.data = torch.from_numpy(b.squeeze(0))

    def _read_layer(self, l):
        assert list(l.keys()) == ['AffineLayer']
        assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
        return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

    def _apply_nonlin(self, x):
        if self.activation_func == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation_func == 'tanh':
            return torch.tanh(x)
        else:
            raise NotImplementedError(self.nonlin_type)

    def _normalize_obs(self, obs):
        return (obs - self.obsnorm_mean) / (self.obsnorm_stdev + 1e-06)

    def forward(self, x):
        out = self._normalize_obs(x)
        for layer in self.layers:
            out = self._apply_nonlin(layer(out))
        out = self.out_layer(out)
        return out


class LanderModel(nn.Module):

    def __init__(self, input_size, output_size, output_activation=None):
        super(LanderModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(self.input_size, 64)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        self.layer2 = nn.Linear(64, 64)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        self.layer3 = nn.Linear(64, output_size)
        None
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        out = x
        out = F.relu(self.layer1(out))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = h_w, h_w
    if type(kernel_size) is not tuple:
        kernel_size = kernel_size, kernel_size
    if type(stride) is not tuple:
        stride = stride, stride
    if type(pad) is not tuple:
        pad = pad, pad
    h = (h_w[0] + 2 * pad[0] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
    w = (h_w[1] + 2 * pad[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return h, w


def get_same_padding_size(kernel_size=1, stride=1, dilation=1):
    """
    A utility function which calculated the padding size needed to 
    get the same padding functionality as same as tensorflow Conv2D implementation
    """
    neg_padding_size = (stride - dilation * kernel_size + dilation - 1) / 2
    if neg_padding_size > 0:
        return 0
    return int(np.ceil(np.abs(neg_padding_size)))


class AtariModel(nn.Module):

    def __init__(self, img_input, num_actions):
        super(AtariModel, self).__init__()
        padding_size = get_same_padding_size(kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=padding_size)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        out_size = conv_output_shape(img_input, 8, 4, padding_size)
        padding_size = get_same_padding_size(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=padding_size)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        out_size = conv_output_shape(out_size, 4, 2, padding_size)
        padding_size = get_same_padding_size(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=padding_size)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.conv_out_size = np.prod(conv_output_shape(out_size, 3, 1, padding_size)) * 64
        self.fc1 = nn.Linear(self.conv_out_size, 512)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(512, num_actions)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.conv_out_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LanderModel,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'n_layers': 1, 'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_erfanMhi_Deep_Reinforcement_Learning_CS285_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

