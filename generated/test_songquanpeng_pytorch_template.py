import sys
_module = sys.modules[__name__]
del sys
resize_images = _module
split_images = _module
template = _module
config = _module
dataset = _module
fetcher = _module
loader = _module
main = _module
eval = _module
fid = _module
build = _module
discriminator = _module
generator = _module
layers = _module
mapping_network = _module
loss = _module
misc = _module
solver = _module
utils = _module
checkpoint = _module
file = _module
image = _module
logger = _module
model = _module

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


import random


import numpy as np


import torch


from torch.backends import cudnn


from torch.utils import data


from torch.utils.data.sampler import WeightedRandomSampler


from torchvision import transforms


from torchvision.datasets import ImageFolder


import torch.nn as nn


from scipy import linalg


from torchvision import models


from torch import nn


import math


import torch.nn.functional as F


import copy


from torchvision.utils import make_grid


import time


import functools


class InceptionV3(nn.Module):

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2), normalize=False, down_sample=False):
        super().__init__()
        self.activation = activation
        self.normalize = normalize
        self.down_sample = down_sample
        self.change_dim = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.change_dim:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.change_dim:
            x = self.conv1x1(x)
        if self.down_sample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.down_sample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = args.img_size
        num_domains = args.num_domains
        dim_in = 2 ** 14 // img_size
        dim_out = None
        max_conv_dim = 512
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, down_sample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        idx = torch.LongTensor(range(y.size(0)))
        out = out[idx, y]
        return out


class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, style_dim=64, activation=nn.LeakyReLU(0.2), up_sample=False):
        super().__init__()
        self.activation = activation
        self.up_sample = up_sample
        self.change_dim = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.change_dim:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.change_dim:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.activation(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size = args.img_size
        dim_in = 2 ** 14 // img_size
        max_conv_dim = 512
        dim_out = None
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2), nn.Conv2d(dim_in, 3, 1, 1, 0))
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, down_sample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim=args.style_dim, up_sample=True))
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim=args.style_dim))

    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        x = self.to_rgb(x)
        return x


class MappingNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        latent_dim, style_dim, num_domains = args.latent_dim, args.style_dim, args.num_domains
        layers = []
        layers += [nn.Linear(latent_dim, style_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(style_dim, style_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(style_dim, style_dim), nn.ReLU(), nn.Linear(style_dim, style_dim), nn.ReLU(), nn.Linear(style_dim, style_dim), nn.ReLU(), nn.Linear(style_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0)))
        s = out[idx, y]
        return s


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaIN,
     lambda: ([], {'style_dim': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (InceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (MappingNetwork,
     lambda: ([], {'args': _mock_config(latent_dim=4, style_dim=4, num_domains=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (ResBlk,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_songquanpeng_pytorch_template(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

