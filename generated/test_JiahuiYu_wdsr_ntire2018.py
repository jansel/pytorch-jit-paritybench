import sys
_module = sys.modules[__name__]
del sys
wdsr_a = _module
wdsr_b = _module

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


import math


import torch


import torch.nn as nn


from torch.nn.parameter import Parameter


class Block(nn.Module):

    def __init__(self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(wn(nn.Conv2d(n_feats, n_feats * expand, 1, padding=1 // 2)))
        body.append(act)
        body.append(wn(nn.Conv2d(n_feats * expand, int(n_feats * linear), 1, padding=1 // 2)))
        body.append(wn(nn.Conv2d(int(n_feats * linear), n_feats, kernel_size, padding=kernel_size // 2)))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class MODEL(nn.Module):

    def __init__(self, args):
        super(MODEL, self).__init__()
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])
        head = []
        head.append(wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3 // 2)))
        body = []
        for i in range(n_resblocks):
            body.append(Block(n_feats, kernel_size, act=act, res_scale=args.res_scale, wn=wn))
        tail = []
        out_feats = scale * scale * args.n_colors
        tail.append(wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(scale))
        skip = []
        skip.append(wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5 // 2)))
        skip.append(nn.PixelShuffle(scale))
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.rgb_mean * 255
        return x

