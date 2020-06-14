import sys
_module = sys.modules[__name__]
del sys
lunarlander_continuous_v2 = _module
a2c = _module
bc_ddpg = _module
bc_sac = _module
ddpg = _module
ddpgfd = _module
per_ddpg = _module
ppo = _module
sac = _module
sacfd = _module
td3 = _module
lunarlander_v2 = _module
dqfd = _module
dqn = _module
pong_no_frameskip_v4 = _module
c51 = _module
iqn = _module
iqn_resnet = _module
reacher_v2 = _module
bc_ddpg = _module
ddpg = _module
td3 = _module
rl_algorithms = _module
agent = _module
bc = _module
ddpg_agent = _module
her = _module
sac_agent = _module
common = _module
abstract = _module
reward_fn = _module
buffer = _module
priortized_replay_buffer = _module
replay_buffer = _module
segment_tree = _module
env = _module
atari_wrappers = _module
multiprocessing_env = _module
normalizers = _module
utils = _module
grad_cam = _module
helper_functions = _module
networks = _module
backbones = _module
cnn = _module
resnet = _module
brain = _module
heads = _module
noise = _module
agent = _module
agent = _module
linear = _module
losses = _module
networks = _module
fd = _module
ddpg_agent = _module
dqn_agent = _module
per = _module
ddpg_agent = _module
agent = _module
registry = _module
agent = _module
agent = _module
config = _module
registry = _module
run_lunarlander_continuous_v2 = _module
run_lunarlander_v2 = _module
run_pong_no_frameskip_v4 = _module
run_reacher_v2 = _module
setup = _module
test_cnn_cfg = _module
test_config_registry = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn.functional as F


from typing import Tuple


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


from collections import OrderedDict


from typing import Callable


from torch.nn import functional as F


from collections import deque


import random


from typing import Deque


from typing import List


from torch.distributions import Normal


import time


from torch.nn.utils import clip_grad_norm_


import math


import inspect


def identity(x: torch.Tensor) ->torch.Tensor:
    """Return input without any change."""
    return x


class CNNLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, kernel_size: int,
        stride: int=1, padding: int=0, pre_activation_fn: Callable=identity,
        activation_fn: Callable=F.relu, post_activation_fn: Callable=identity):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(input_size, output_size, kernel_size=
            kernel_size, stride=stride, padding=padding)
        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        return x


class Registry:

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbones')


class BasicBlock(nn.Module):
    """Basic building block for ResNet."""

    def __init__(self, in_planes: int, planes: int, stride: int=1,
        expansion: int=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck building block."""

    def __init__(self, in_planes: int, planes: int, stride: int=1,
        expansion: int=1):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


HEADS = Registry('heads')


def init_layer_uniform(layer: nn.Linear, init_w: float=0.003) ->nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    References:
        https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
        https://github.com/Kaixhin/Rainbow/blob/master/model.py

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float=0.5
        ):
        """Initialize."""
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features,
            in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features,
            in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.
            in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features)
            )

    @staticmethod
    def scale_noise(size: int) ->torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(x, self.weight_mu + self.weight_sigma * self.
            weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_medipixel_rl_algorithms(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(CNNLayer(*[], **{'input_size': 4, 'output_size': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(NoisyLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

