import sys
_module = sys.modules[__name__]
del sys
mobile_seg = _module
const = _module
dataset = _module
loss = _module
modules = _module
net = _module
wrapper = _module
params = _module
mylib = _module
albumentations = _module
augmentations = _module
transforms = _module
gcp = _module
util = _module
lgb = _module
callbacks = _module
model_extraction = _module
metrics = _module
null_imp = _module
numpy = _module
functional = _module
pandas = _module
cache = _module
corr = _module
pytorch_lightning = _module
base_module = _module
logging = _module
sklearn = _module
fe = _module
pair_count_encoder = _module
target_encoder = _module
split = _module
data = _module
dataset = _module
ensemble = _module
ema = _module
swa = _module
bert_emb = _module
functional = _module
nn = _module
functional = _module
init = _module
losses = _module
regression = _module
mish_init = _module
dense = _module
frelu = _module
gauss_rank_transform = _module
graph_norm = _module
mlp = _module
pair_norm = _module
se_layer = _module
optim = _module
SGD = _module
gc = _module
sched = _module
utils = _module
dev = _module
plt = _module
text = _module
run_convert_coreml = _module
run_eval = _module
run_train = _module

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


import pandas as pd


from sklearn.model_selection import KFold


from torch.utils.data import Dataset


from torch.nn.functional import interpolate


import torch


import torch.nn as nn


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from logging import Logger


from logging import getLogger


from typing import Optional


from typing import Tuple


from typing import Protocol


from typing import Sequence


from typing import Mapping


from typing import Generic


from typing import TypeVar


from typing import TypedDict


from typing import Dict


from typing import Any


from typing import Union


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import copy


from typing import Callable


import torch.nn.functional as F


from functools import partial


from torch.nn.init import constant_


import math


from torch import nn


from torch.nn.init import xavier_uniform_


import matplotlib.pyplot as plt


from torch.optim import Optimizer


from torch.optim.optimizer import required


from functools import cached_property


from logging import FileHandler


from time import time


from torch.optim import Adam


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import OneCycleLR


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), norm_layer(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UpSampleBlock, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.invres = InvertedResidual(out_channels * 2, out_channels, 1, 6)

    def forward(self, x0, x1):
        x = torch.cat([x0, self.dconv(x1)], dim=1)
        x = self.invres(x)
        return x


class MobileNetV2_unet(nn.Module):

    def __init__(self, **kwargs):
        super(MobileNetV2_unet, self).__init__()
        self.backbone = mobilenetv2_100(pretrained=True, **kwargs)
        self.up_sample_blocks = nn.ModuleList([UpSampleBlock(1280, 96), UpSampleBlock(96, 32), UpSampleBlock(32, 24), UpSampleBlock(24, 16)])
        self.conv_last = nn.Sequential(nn.Conv2d(16, 3, 1), nn.Conv2d(3, 1, 1), nn.Sigmoid())
        del self.backbone.bn2, self.backbone.act2, self.backbone.global_pool, self.backbone.classifier
        efficientnet_init_weights(self.up_sample_blocks)
        efficientnet_init_weights(self.conv_last)

    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        down_feats = []
        for b in self.backbone.blocks:
            x = b(x)
            if x.shape[1] in [16, 24, 32, 96]:
                down_feats.append(x)
        x = self.backbone.conv_head(x)
        for f, b in zip(reversed(down_feats), self.up_sample_blocks):
            x = b(f, x)
        x = self.conv_last(x)
        return x


class Wrapper(nn.Module):

    def __init__(self, unet: MobileNetV2_unet, scale: float=255.0):
        super().__init__()
        self.unet = unet
        self.scale = scale

    def forward(self, x):
        x = x / self.scale
        x = self.unet(x)
        x = x * self.scale
        x = torch.cat((x, x, x), dim=1)
        return x


class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


zeros_initializer = partial(constant_, val=0.0)


class Dense(nn.Linear):
    """Fully connected linear layer with activation function.

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """

    def __init__(self, in_features, out_features, bias=True, activation=None, weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize models weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y


class FReLU(nn.Module):
    """ FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x


class GaussRankTransform(nn.Module):

    def __init__(self, data: torch.Tensor, eps=1e-06):
        super(GaussRankTransform, self).__init__()
        tformed = self._erfinv(data, eps)
        data, sort_idx = data.sort()
        self.register_buffer('src', data)
        self.register_buffer('dst', tformed[sort_idx])

    @staticmethod
    def _erfinv(data: torch.Tensor, eps):
        rank = data.argsort().argsort().float()
        rank_scaled = (rank / rank.max() - 0.5) * 2
        rank_scaled = rank_scaled.clamp(-1 + eps, 1 - eps)
        tformed = rank_scaled.erfinv()
        return tformed

    def forward(self, x):
        return self._transform(x, self.dst, self.src)

    def invert(self, x):
        return self._transform(x, self.src, self.dst)

    def _transform(self, x, src, dst):
        pos = src.argsort()[x.argsort().argsort()]
        N = len(self.src)
        pos[pos >= N] = N - 1
        pos[pos - 1 <= 0] = 0
        x1 = dst[pos]
        x2 = dst[pos - 1]
        y1 = src[pos]
        y2 = src[pos - 1]
        relative = (x - x2) / (x1 - x2)
        return (1 - relative) * y2 + relative * y1


class GraphNorm(nn.Module):

    def __init__(self, norm_type='gn', hidden_dim=64, print_info=None):
        super(GraphNorm, self).__init__()
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, batch_list, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        device = tensor.device
        batch_size = len(batch_list)
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:], device=device).type_as(tensor)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean * self.mean_scale
        std = torch.zeros(batch_size, *tensor.shape[1:], device=device).type_as(tensor)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-06).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class PairNorm(nn.Module):

    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-06 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-06 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-06 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(Dense(channel, channel // reduction, bias=False), Mish(inplace=True), Dense(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dense,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FReLU,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogCoshLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Wrapper,
     lambda: ([], {'unet': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_akirasosa_mobile_semantic_segmentation(_paritybench_base):
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

