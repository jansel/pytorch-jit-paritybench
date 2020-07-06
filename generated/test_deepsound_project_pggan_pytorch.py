import sys
_module = sys.modules[__name__]
del sys
dataset = _module
generate = _module
network = _module
output_postprocess = _module
plugins = _module
train = _module
trainer = _module
utils = _module
wgan_gp_loss = _module

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


from torch.utils.data import Dataset


import numpy as np


import torch


import math


from functools import reduce


from torch.autograd import Variable


from functools import partial


from torch import nn


from torch.nn import functional as F


import time


from torch.optim import Adam


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import DataLoader


from torch.utils.data.sampler import RandomSampler


from collections import OrderedDict


import inspect


from torch.autograd import grad


class PGConv2d(nn.Module):

    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1, pixelnorm=True, wscale=True, act='lrelu'):
        super(PGConv2d, self).__init__()
        if wscale:
            init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Conv2d(ch_in, ch_out, ksize, stride, pad)
        init(self.conv.weight)
        if wscale:
            self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
            self.conv.weight.data /= self.c
        else:
            self.c = 1.0
        self.eps = 1e-08
        self.pixelnorm = pixelnorm
        if act is not None:
            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
        else:
            self.act = None
        self.conv

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.act is not None:
            h = self.act(h)
        if self.pixelnorm:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        return h


class GFirstBlock(nn.Module):

    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, 4, 1, 3, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            return self.toRGB(x)
        return x


class GBlock(nn.Module):

    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            x = self.toRGB(x)
        return x


class Generator(nn.Module):

    def __init__(self, dataset_shape, fmap_base=4096, fmap_decay=1.0, fmap_max=512, latent_size=512, normalize_latents=True, wscale=True, pixelnorm=True, leakyrelu=True):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        None
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = {'wscale': wscale, 'pixelnorm': pixelnorm, 'act': 'lrelu' if leakyrelu else 'relu'}
        self.block0 = GFirstBlock(latent_size, nf(1), num_channels, **layer_settings)
        self.blocks = nn.ModuleList([GBlock(nf(i - 1), nf(i), num_channels, **layer_settings) for i in range(2, R)])
        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-08
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)

    def forward(self, x):
        h = x.unsqueeze(2).unsqueeze(3)
        if self.normalize_latents:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        h = self.block0(h, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h)
            h = F.upsample(h, scale_factor=2)
            ult = self.blocks[self.depth - 1](h, True)
            if self.alpha < 1.0:
                if self.depth > 1:
                    preult_rgb = self.blocks[self.depth - 2].toRGB(h)
                else:
                    preult_rgb = self.block0.toRGB(h)
            else:
                preult_rgb = 0
            h = preult_rgb * (1 - self.alpha) + ult * self.alpha
        return h


class DBlock(nn.Module):

    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.c1 = PGConv2d(ch_in, ch_in, **layer_settings)
        self.c2 = PGConv2d(ch_in, ch_out, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


def Tstdeps(val):
    return torch.sqrt(((val - val.mean()) ** 2).mean() + 1e-08)


class MinibatchStddev(nn.Module):

    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        h = torch.cat((x, new_channel), dim=1)
        return h


class DLastBlock(nn.Module):

    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.stddev = MinibatchStddev()
        self.c1 = PGConv2d(ch_in + 1, ch_in, **layer_settings)
        self.c2 = PGConv2d(ch_in, ch_out, 4, 1, 0, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.stddev(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, dataset_shape, fmap_base=4096, fmap_decay=1.0, fmap_max=512, wscale=True, pixelnorm=False, leakyrelu=True):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        layer_settings = {'wscale': wscale, 'pixelnorm': pixelnorm, 'act': 'lrelu' if leakyrelu else 'relu'}
        self.blocks = nn.ModuleList([DBlock(nf(i), nf(i - 1), num_channels, **layer_settings) for i in range(R - 1, 1, -1)] + [DLastBlock(nf(1), nf(0), num_channels, **layer_settings)])
        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-08
        self.max_depth = len(self.blocks) - 1

    def forward(self, x):
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        if self.depth > 0:
            h = F.avg_pool2d(h, 2)
            if self.alpha < 1.0:
                xlowres = F.avg_pool2d(xhighres, 2)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = F.avg_pool2d(h, 2)
        h = self.linear(h.squeeze(-1).squeeze(-1))
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DBlock,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DLastBlock,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GBlock,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GFirstBlock,
     lambda: ([], {'ch_in': 4, 'ch_out': 4, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinibatchStddev,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PGConv2d,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_deepsound_project_pggan_pytorch(_paritybench_base):
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

