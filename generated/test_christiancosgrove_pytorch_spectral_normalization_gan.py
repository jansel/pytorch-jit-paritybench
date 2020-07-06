import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
model_resnet = _module
spectral_normalization = _module
spectral_normalization_nondiff = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.optim.lr_scheduler import ExponentialLR


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


import numpy as np


from torch import nn


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch import Tensor


from torch.nn import Parameter


GEN_SIZE = 128


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(), nn.Upsample(scale_factor=2), self.conv1, nn.BatchNorm2d(out_channels), nn.ReLU(), self.conv2)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


channels = 3


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)
        self.model = nn.Sequential(ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), nn.BatchNorm2d(GEN_SIZE), nn.ReLU(), self.final, nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))


DISC_SIZE = 128


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations

    def _update_u_v(self):
        if not self._made_params():
            self._make_params()
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + '_u')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u))
            u = l2normalize(torch.mv(w.view(height, -1).data, v))
        setattr(self.module, self.name + '_u', u)
        w.data = w.data / torch.dot(u, torch.mv(w.view(height, -1).data, v))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = l2normalize(w.data.new(height).normal_(0, 1))
        self.module.register_buffer(self.name + '_u', u)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        self.model = nn.Sequential(SpectralNorm(self.conv1), nn.ReLU(), SpectralNorm(self.conv2), nn.AvgPool2d(2))
        self.bypass = nn.Sequential(nn.AvgPool2d(2), SpectralNorm(self.bypass_conv))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        if stride == 1:
            self.model = nn.Sequential(nn.ReLU(), SpectralNorm(self.conv1), nn.ReLU(), SpectralNorm(self.conv2))
        else:
            self.model = nn.Sequential(nn.ReLU(), SpectralNorm(self.conv1), nn.ReLU(), SpectralNorm(self.conv2), nn.AvgPool2d(2, stride=stride, padding=0))
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(SpectralNorm(self.bypass_conv), nn.AvgPool2d(2, stride=stride, padding=0))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE), nn.ReLU(), nn.AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, DISC_SIZE))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FirstResBlockDiscriminator,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Generator,
     lambda: ([], {'z_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlockDiscriminator,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_christiancosgrove_pytorch_spectral_normalization_gan(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

