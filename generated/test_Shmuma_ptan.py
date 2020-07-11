import sys
_module = sys.modules[__name__]
del sys
ptan = _module
actions = _module
agent = _module
common = _module
runfile = _module
utils = _module
wrappers = _module
wrappers_simple = _module
experience = _module
ignite = _module
a2c = _module
a2c_atari = _module
dqn_expreplay = _module
dqn_expreplay_doom = _module
lib = _module
atari_wrappers = _module
common = _module
dqn_model = _module
dqn_tweaks_atari = _module
dqn_tweaks_doom = _module
dqn_atari = _module
common = _module
dqn_model = _module
qr_atari = _module
prio_buffer_bench = _module
simple_buffer_bench = _module
distr_test = _module
common = _module
dqn_model = _module
reinforce = _module
setup = _module
tests = _module
test_utils = _module
test_wrappers_simple = _module
test_actions = _module
test_experience = _module

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


import copy


import numpy as np


import torch


import torch.nn.functional as F


import time


import collections


import torch.nn as nn


import random


from torch.autograd import Variable


from collections import namedtuple


from collections import deque


import logging


import torch.optim as optim


import itertools


import torch.multiprocessing as mp


import math


import matplotlib as mpl


import matplotlib.pylab as plt


class WeightedMSELoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        if weights is None:
            return nn.MSELoss(self.size_average)(input, target)
        loss_rows = (input - target) ** 2
        if len(loss_rows.size()) != 1:
            loss_rows = torch.sum(loss_rows, dim=1)
        res = (weights * loss_rows).sum()
        if self.size_average:
            res /= len(weights)
        return res


class Model(nn.Module):

    def __init__(self, n_actions, input_len):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_policy = nn.Linear(100, n_actions)
        self.out_value = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        policy = F.softmax(self.out_policy(x))
        value = self.out_value(x)
        return policy, value


class ConvNet(nn.Module):

    def __init__(self, input_shape, output_size):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(32, 32, kernel_size=5, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(32, 64, kernel_size=4, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * 6 * 6, output_size), nn.ReLU())

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class PolicyNet(nn.Module):

    def __init__(self, input_size, actions_n):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(input_size, actions_n)

    def forward(self, x):
        return self.fc(x)


class ValueNet(nn.Module):

    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module):

    def __init__(self, n_actions, input_shape):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 2)
        n_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(n_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        None
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class NoisyLinear(nn.Linear):

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform(self.weight, -std, std)
        nn.init.uniform(self.bias, -std, std)

    def forward(self, input):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        bias = self.bias
        if bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer('epsilon_input', torch.zeros(1, in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t())
        noise_v = Variable(torch.mul(eps_in, eps_out))
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


QUANT_N = 51


class QRDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(QRDQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions * QUANT_N))

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        return self.fc(conv_out).view(batch_size, -1, QUANT_N)

    def qvals(self, x):
        return self.qvals_from_quant(self(x))

    @classmethod
    def qvals_from_quant(cls, quant):
        return quant.mean(dim=2)


class NoisyDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [dqn_model.NoisyLinear(conv_out_size, 512), dqn_model.NoisyLinear(512, n_actions)]
        self.fc = nn.Sequential(self.noisy_layers[0], nn.ReLU(), self.noisy_layers[1])

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).data.cpu().numpy()[0] for layer in self.noisy_layers]


class DuelingDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.fc_val = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


N_ATOMS = 51


Vmax = 10


Vmin = -10


DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class DistributionalDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions * N_ATOMS))
        self.register_buffer('supports', torch.arange(Vmin, Vmax, DELTA_Z))
        self.softmax = nn.Softmax()

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * Variable(self.supports, volatile=True)
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


class RainbowDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(dqn_model.NoisyLinear(conv_out_size, 512), nn.ReLU(), dqn_model.NoisyLinear(512, N_ATOMS))
        self.fc_adv = nn.Sequential(dqn_model.NoisyLinear(conv_out_size, 512), nn.ReLU(), dqn_model.NoisyLinear(512, n_actions * N_ATOMS))
        self.register_buffer('supports', torch.arange(Vmin, Vmax, DELTA_Z))
        self.softmax = nn.Softmax()

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * Variable(self.supports, volatile=True)
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvNet,
     lambda: ([], {'input_shape': [4, 4], 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Model,
     lambda: ([], {'n_actions': 4, 'input_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyFactorizedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PolicyNet,
     lambda: ([], {'input_size': 4, 'actions_n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ValueNet,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Shmuma_ptan(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

