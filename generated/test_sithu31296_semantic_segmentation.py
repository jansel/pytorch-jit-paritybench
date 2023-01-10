import sys
_module = sys.modules[__name__]
del sys
calc_class_weights = _module
export_data = _module
onnx_infer = _module
openvino_infer = _module
preprocess_celebamaskhq = _module
tflite_infer = _module
semseg = _module
augmentations = _module
datasets = _module
ade20k = _module
atr = _module
camvid = _module
celebamaskhq = _module
cihp = _module
cityscapes = _module
cocostuff = _module
facesynthetics = _module
helen = _module
ibugmask = _module
lapa = _module
lip = _module
mapillary = _module
mhpv1 = _module
mhpv2 = _module
pascalcontext = _module
suim = _module
sunrgbd = _module
losses = _module
metrics = _module
models = _module
backbones = _module
convnext = _module
micronet = _module
mit = _module
mobilenetv2 = _module
mobilenetv3 = _module
poolformer = _module
pvt = _module
resnet = _module
resnetd = _module
rest = _module
uniformer = _module
base = _module
bisenetv1 = _module
bisenetv2 = _module
custom_cnn = _module
custom_vit = _module
ddrnet = _module
fchardnet = _module
heads = _module
condnet = _module
fapn = _module
fcn = _module
fpn = _module
lawin = _module
segformer = _module
sfnet = _module
upernet = _module
lawin = _module
layers = _module
common = _module
initialize = _module
modules = _module
ppm = _module
psa = _module
segformer = _module
sfnet = _module
optimizers = _module
schedulers = _module
utils = _module
utils = _module
visualize = _module
setup = _module
benchmark = _module
export = _module
infer = _module
train = _module
val = _module

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


import torch


from torchvision import io


import torchvision.transforms.functional as TF


import random


import math


from torch import Tensor


from typing import Tuple


from typing import List


from typing import Union


from typing import Optional


from torch.utils.data import Dataset


from torchvision import transforms as T


import numpy as np


from torchvision.transforms import functional as TF


from scipy import io as sio


from torch import nn


from torch.nn import functional as F


from torchvision.ops import DeformConv2d


import warnings


from torch.optim import AdamW


from torch.optim import SGD


from torch.optim.lr_scheduler import _LRScheduler


import time


import functools


from torch.backends import cudnn


from torch.autograd import profiler


from torch import distributed as dist


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from torchvision.utils import make_grid


from torch.utils.tensorboard import SummaryWriter


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DistributedSampler


from torch.utils.data import RandomSampler


class CrossEntropy(nn.Module):

    def __init__(self, ignore_label: int=255, weight: Tensor=None, aux_weights: list=[1, 0.4, 0.4]) ->None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) ->Tensor:
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) ->Tensor:
        if isinstance(preds, tuple):
            return sum([(w * self._forward(pred, labels)) for pred, w in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):

    def __init__(self, ignore_label: int=255, weight: Tensor=None, thresh: float=0.7, aux_weights: list=[1, 1]) ->None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) ->Tensor:
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) ->Tensor:
        if isinstance(preds, tuple):
            return sum([(w * self._forward(pred, labels)) for pred, w in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):

    def __init__(self, delta: float=0.5, aux_weights: list=[1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) ->Tensor:
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels * preds, dim=(2, 3))
        fn = torch.sum(labels * (1 - preds), dim=(2, 3))
        fp = torch.sum((1 - labels) * preds, dim=(2, 3))
        dice_score = (tp + 1e-06) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-06)
        dice_score = torch.sum(1 - dice_score, dim=-1)
        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) ->Tensor:
        if isinstance(preds, tuple):
            return sum([(w * self._forward(pred, targets)) for pred, w in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class LayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-06) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) ->Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8) ->None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) ->Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: float=None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) ->Tensor:
        if self.p == 0.0 or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(kp) * random_tensor


class MLP(nn.Module):

    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) ->Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, head, sr_ratio=1, dpr=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) ->Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Stem(nn.Sequential):

    def __init__(self, c1, c2):
        super().__init__(nn.Conv2d(c1, c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(True), nn.Conv2d(c2, c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(True))


class Downsample(nn.Sequential):

    def __init__(self, c1, c2, k, s):
        super().__init__(LayerNorm(c1), nn.Conv2d(c1, c2, k, s))


convnext_settings = {'T': [[3, 3, 9, 3], [96, 192, 384, 768], 0.0], 'S': [[3, 3, 27, 3], [96, 192, 384, 768], 0.0], 'B': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.0]}


class ConvNeXt(nn.Module):

    def __init__(self, model_name: str='T') ->None:
        super().__init__()
        assert model_name in convnext_settings.keys(), f'ConvNeXt model name should be in {list(convnext_settings.keys())}'
        depths, embed_dims, drop_path_rate = convnext_settings[model_name]
        self.channels = embed_dims
        self.downsample_layers = nn.ModuleList([Stem(3, embed_dims[0], 4, 4), *[Downsample(embed_dims[i], embed_dims[i + 1], 2, 2) for i in range(3)]])
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(embed_dims[i], dpr[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        for i in range(4):
            self.add_module(f'norm{i}', LayerNorm(embed_dims[i]))

    def forward(self, x: Tensor):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f'norm{i}')
            outs.append(norm_layer(x))
        return outs


class HSigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(True)

    def forward(self, x: Tensor) ->Tensor:
        return self.relu(x + 3) / 6


class HSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = HSigmoid()

    def forward(self, x: Tensor) ->Tensor:
        return x * self.sigmoid(x)


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)
        return out


def _make_divisible(v: float, divisor: int, min_value: Optional[int]=None) ->int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DYShiftMax(nn.Module):

    def __init__(self, c1, c2, init_a=[0.0, 0.0], init_b=[0.0, 0.0], act_relu=True, g=None, reduction=4, expansion=False):
        super().__init__()
        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b
        self.c2 = c2
        self.avg_pool = nn.Sequential(nn.Sequential(), nn.AdaptiveAvgPool2d(1))
        squeeze = _make_divisible(c1 // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(c1, squeeze), nn.ReLU(True), nn.Linear(squeeze, c2 * self.exp), HSigmoid())
        g = g[1]
        if g != 1 and expansion:
            g = c1 // g
        gc = c1 // g
        index = torch.Tensor(range(c1)).view(1, c1, 1, 1)
        index = index.view(1, g, gc, 1, 1)
        indexgs = torch.split(index, [1, g - 1], dim=1)
        indexgs = torch.cat([indexgs[1], indexgs[0]], dim=1)
        indexs = torch.split(indexgs, [1, gc - 1], dim=2)
        indexs = torch.cat([indexs[1], indexs[0]], dim=2)
        self.index = indexs.view(c1).long()

    def forward(self, x: Tensor) ->Tensor:
        B, C, H, W = x.shape
        x_out = x
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, -1, 1, 1)
        y = (y - 0.5) * 4.0
        x2 = x_out[:, self.index, :, :]
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.c2, dim=1)
            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_b[1]
            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]
            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2
            out = torch.max(z1, z2)
        elif self.exp == 2:
            a1, b1 = torch.split(y, self.c2, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1
        return out


class SwishLinear(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(c1, c2), nn.BatchNorm1d(c2), HSwish())

    def forward(self, x: Tensor) ->Tensor:
        return self.linear(x)


class SpatialSepConvSF(nn.Module):

    def __init__(self, c1, outs, k, s):
        super().__init__()
        o1, o2 = outs
        self.conv = nn.Sequential(nn.Conv2d(c1, o1, (k, 1), (s, 1), (k // 2, 0), bias=False), nn.BatchNorm2d(o1), nn.Conv2d(o1, o1 * o2, (1, k), (1, s), (0, k // 2), groups=o1, bias=False), nn.BatchNorm2d(o1 * o2), ChannelShuffle(o1))

    def forward(self, x: Tensor) ->Tensor:
        return self.conv(x)


class DepthSpatialSepConv(nn.Module):

    def __init__(self, c1, expand, k, s):
        super().__init__()
        exp1, exp2 = expand
        ch = c1 * exp1
        c2 = c1 * exp1 * exp2
        self.conv = nn.Sequential(nn.Conv2d(c1, ch, (k, 1), (s, 1), (k // 2, 0), groups=c1, bias=False), nn.BatchNorm2d(ch), nn.Conv2d(ch, c2, (1, k), (1, s), (0, k // 2), groups=ch, bias=False), nn.BatchNorm2d(c2))

    def forward(self, x: Tensor) ->Tensor:
        return self.conv(x)


class PWConv(nn.Module):

    def __init__(self, c1, c2, g=2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c2, 1, 1, 0, groups=g[0], bias=False), nn.BatchNorm2d(c2))

    def forward(self, x: Tensor) ->Tensor:
        return self.conv(x)


class MicroBlock(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, t1=(2, 2), gs1=4, groups_1x1=(1, 1), dy=(2, 0, 1), r=1, init_a=(1.0, 1.0), init_b=(0.0, 0.0)):
        super().__init__()
        self.identity = s == 1 and c1 == c2
        y1, y2, y3 = dy
        _, g1, g2 = groups_1x1
        reduction = 8 * r
        ch2 = c1 * t1[0] * t1[1]
        if gs1[0] == 0:
            self.layers = nn.Sequential(DepthSpatialSepConv(c1, t1, k, s), DYShiftMax(ch2, ch2, init_a, init_b, True if y2 == 2 else False, gs1, reduction) if y2 > 0 else nn.ReLU6(True), ChannelShuffle(gs1[1]), ChannelShuffle(ch2 // 2) if y2 != 0 else nn.Sequential(), PWConv(ch2, c2, (g1, g2)), DYShiftMax(c2, c2, [1.0, 0.0], [0.0, 0.0], False, (g1, g2), reduction // 2) if y3 > 0 else nn.Sequential(), ChannelShuffle(g2), ChannelShuffle(c2 // 2) if c2 % 2 == 0 and y3 != 0 else nn.Sequential())
        elif g2 == 0:
            self.layers = nn.Sequential(PWConv(c1, ch2, gs1), DYShiftMax(ch2, ch2, [1.0, 0.0], [0.0, 0.0], False, gs1, reduction) if y3 > 0 else nn.Sequential())
        else:
            self.layers = nn.Sequential(PWConv(c1, ch2, gs1), DYShiftMax(ch2, ch2, init_a, init_b, True if y1 == 2 else False, gs1, reduction) if y1 > 0 else nn.ReLU6(True), ChannelShuffle(gs1[1]), DepthSpatialSepConv(ch2, (1, 1), k, s), nn.Sequential(), DYShiftMax(ch2, ch2, init_a, init_b, True if y2 == 2 else False, gs1, reduction, True) if y2 > 0 else nn.ReLU6(True), ChannelShuffle(ch2 // 4) if y1 != 0 and y2 != 0 else nn.Sequential() if y1 == 0 and y2 == 0 else ChannelShuffle(ch2 // 2), PWConv(ch2, c2, (g1, g2)), DYShiftMax(c2, c2, [1.0, 0.0], [0.0, 0.0], False, (g1, g2), reduction=reduction // 2 if c2 < ch2 else reduction) if y3 > 0 else nn.Sequential(), ChannelShuffle(g2), ChannelShuffle(c2 // 2) if y3 != 0 else nn.Sequential())

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.layers(x)
        if self.identity:
            out += identity
        return out


micronet_settings = {'M1': [6, [3, 2], 960, [1.0, 1.0], [0.0, 0.0], [1, 2, 4, 7], [8, 16, 32, 576], [[2, 8, 3, 2, 2, 0, 6, 8, 2, 2, 2, 0, 1, 1], [2, 16, 3, 2, 2, 0, 8, 16, 4, 4, 2, 2, 1, 1], [2, 16, 5, 2, 2, 0, 16, 16, 4, 4, 2, 2, 1, 1], [1, 32, 5, 1, 6, 4, 4, 32, 4, 4, 2, 2, 1, 1], [2, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 1], [1, 96, 3, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2], [1, 576, 3, 1, 6, 12, 12, 0, 0, 0, 2, 2, 1, 2]]], 'M2': [8, [4, 2], 1024, [1.0, 1.0], [0.0, 0.0], [1, 3, 6, 9], [12, 24, 64, 768], [[2, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 0, 1, 1], [2, 16, 3, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1], [1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 2, 2, 1, 1], [2, 32, 5, 1, 6, 6, 6, 32, 4, 4, 2, 2, 1, 1], [1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 2, 2, 1, 2], [1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 2], [2, 96, 5, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2], [1, 128, 3, 1, 6, 12, 12, 128, 8, 8, 2, 2, 1, 2], [1, 768, 3, 1, 6, 16, 16, 0, 0, 0, 2, 2, 1, 2]]], 'M3': [12, [4, 3], 1024, [1.0, 0.5], [0.0, 0.5], [1, 3, 8, 12], [16, 24, 80, 864], [[2, 16, 3, 2, 2, 0, 12, 16, 4, 4, 0, 2, 0, 1], [2, 24, 3, 2, 2, 0, 16, 24, 4, 4, 0, 2, 0, 1], [1, 24, 3, 2, 2, 0, 24, 24, 4, 4, 0, 2, 0, 1], [2, 32, 5, 1, 6, 6, 6, 32, 4, 4, 0, 2, 0, 1], [1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 0, 2, 0, 2], [1, 64, 5, 1, 6, 8, 8, 48, 8, 8, 0, 2, 0, 2], [1, 80, 5, 1, 6, 8, 8, 80, 8, 8, 0, 2, 0, 2], [1, 80, 5, 1, 6, 10, 10, 80, 8, 8, 0, 2, 0, 2], [2, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], [1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], [1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], [1, 864, 3, 1, 6, 12, 12, 0, 0, 0, 0, 2, 0, 2]]]}


class MicroNet(nn.Module):

    def __init__(self, variant: str='M1') ->None:
        super().__init__()
        self.inplanes = 64
        assert variant in micronet_settings.keys(), f'MicroNet model name should be in {list(micronet_settings.keys())}'
        input_channel, stem_groups, _, init_a, init_b, out_indices, channels, cfgs = micronet_settings[variant]
        self.out_indices = out_indices
        self.channels = channels
        self.features = nn.ModuleList([Stem(3, input_channel, 2, stem_groups)])
        for s, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r in cfgs:
            self.features.append(MicroBlock(input_channel, c, ks, s, (c1, c2), (g1, g2), (c3, g3, g4), (y1, y2, y3), r, init_a, init_b))
            input_channel = c

    def forward(self, x: Tensor) ->Tensor:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class DWConv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) ->Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_ch=3, dim=96, type='pool') ->None:
        super().__init__()
        self.patch_size = patch_size
        self.type = type
        self.dim = dim
        if type == 'conv':
            self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size, groups=patch_size * patch_size)
        else:
            self.proj = nn.ModuleList([nn.MaxPool2d(patch_size, patch_size), nn.AvgPool2d(patch_size, patch_size)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) ->Tensor:
        _, _, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
        if self.type == 'conv':
            x = self.proj(x)
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.dim, Wh, Ww)
        return x


mit_settings = {'B0': [[32, 64, 160, 256], [2, 2, 2, 2]], 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]], 'B2': [[64, 128, 320, 512], [3, 4, 6, 3]], 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]], 'B4': [[64, 128, 320, 512], [3, 8, 27, 3]], 'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]}


class MiT(nn.Module):

    def __init__(self, model_name: str='B0'):
        super().__init__()
        assert model_name in mit_settings.keys(), f'MiT model name should be in {list(mit_settings.keys())}'
        embed_dims, depths = mit_settings[model_name]
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) ->Tensor:
        B = x.shape[0]
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x1, x2, x3, x4


class ConvModule(nn.Sequential):

    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(nn.Conv2d(c1, c2, k, s, p, d, g, bias=False), nn.BatchNorm2d(c2), nn.ReLU(True))


class InvertedResidual(nn.Module):

    def __init__(self, c1, c2, s, expand_ratio):
        super().__init__()
        ch = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2
        layers = []
        if expand_ratio != 1:
            layers.append(ConvModule(c1, ch, 1))
        layers.extend([ConvModule(ch, ch, 3, s, 1, g=ch), nn.Conv2d(ch, c2, 1, bias=False), nn.BatchNorm2d(c2)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) ->Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, variant: str=None):
        super().__init__()
        self.out_indices = [3, 6, 13, 17]
        self.channels = [24, 32, 96, 320]
        input_channel = 32
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        self.features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

    def forward(self, x: Tensor) ->Tensor:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class SqueezeExcitation(nn.Module):

    def __init__(self, ch, squeeze_factor=4):
        super().__init__()
        squeeze_ch = _make_divisible(ch // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(ch, squeeze_ch, 1)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Conv2d(squeeze_ch, ch, 1)

    def _scale(self, x: Tensor) ->Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return F.hardsigmoid(scale, True)

    def forward(self, x: Tensor) ->Tensor:
        scale = self._scale(x)
        return scale * x


class MobileNetV3(nn.Module):

    def __init__(self, variant: str=None):
        super().__init__()
        self.out_indices = [3, 6, 13, 17]
        self.channels = [24, 32, 96, 320]
        input_channel = 32
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        self.features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

    def forward(self, x: Tensor) ->Tensor:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class Pooling(nn.Module):

    def __init__(self, pool_size=3) ->None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size // 2, count_include_pad=False)

    def forward(self, x: Tensor) ->Tensor:
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):

    def __init__(self, dim, pool_size=3, dpr=0.0, layer_scale_init_value=1e-05):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.mlp = MLP(dim, int(dim * 4))
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor) ->Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


poolformer_settings = {'S24': [[4, 4, 12, 4], [64, 128, 320, 512], 0.1], 'S36': [[6, 6, 18, 6], [64, 128, 320, 512], 0.2], 'M36': [[6, 6, 18, 6], [96, 192, 384, 768], 0.3]}


class PoolFormer(nn.Module):

    def __init__(self, model_name: str='S24') ->None:
        super().__init__()
        assert model_name in poolformer_settings.keys(), f'PoolFormer model name should be in {list(poolformer_settings.keys())}'
        layers, embed_dims, drop_path_rate = poolformer_settings[model_name]
        self.channels = embed_dims
        self.patch_embed = PatchEmbed(7, 4, 2, 3, embed_dims[0])
        network = []
        for i in range(len(layers)):
            blocks = []
            for j in range(layers[i]):
                dpr = drop_path_rate * (j + sum(layers[:i])) / (sum(layers) - 1)
                blocks.append(PoolFormerBlock(embed_dims[i], 3, dpr))
            network.append(nn.Sequential(*blocks))
            if i >= len(layers) - 1:
                break
            network.append(PatchEmbed(3, 2, 1, embed_dims[i], embed_dims[i + 1]))
        self.network = nn.ModuleList(network)
        self.out_indices = [0, 2, 4, 6]
        for i, index in enumerate(self.out_indices):
            self.add_module(f'norm{index}', nn.GroupNorm(1, embed_dims[i]))

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        outs = []
        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f'norm{i}')(x)
                outs.append(out)
        return outs


pvtv2_settings = {'B1': [2, 2, 2, 2], 'B2': [3, 4, 6, 3], 'B3': [3, 4, 18, 3], 'B4': [3, 8, 27, 3], 'B5': [3, 6, 40, 3]}


class PVTv2(nn.Module):

    def __init__(self, model_name: str='B1') ->None:
        super().__init__()
        assert model_name in pvtv2_settings.keys(), f'PVTv2 model name should be in {list(pvtv2_settings.keys())}'
        depths = pvtv2_settings[model_name]
        embed_dims = [64, 128, 320, 512]
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, 8, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, 4, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, 4, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) ->Tensor:
        B = x.shape[0]
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x1, x2, x3, x4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


resnet_settings = {'18': [BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]], '34': [BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512]], '50': [Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048]], '101': [Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048]], '152': [Bottleneck, [3, 8, 36, 3], [256, 512, 1024, 2048]]}


class ResNet(nn.Module):

    def __init__(self, model_name: str='50') ->None:
        super().__init__()
        assert model_name in resnet_settings.keys(), f'ResNet model name should be in {list(resnet_settings.keys())}'
        block, depths, channels = resnet_settings[model_name]
        self.inplanes = 64
        self.channels = channels
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2)

    def _make_layer(self, block, planes, depth, s=1) ->nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = nn.Sequential(block(self.inplanes, planes, s, downsample), *[block(planes * block.expansion, planes) for _ in range(1, depth)])
        self.inplanes = planes * block.expansion
        return layers

    def forward(self, x: Tensor) ->Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


resnetd_settings = {'18': [BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]], '50': [Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048]], '101': [Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048]]}


class ResNetD(nn.Module):

    def __init__(self, model_name: str='50') ->None:
        super().__init__()
        assert model_name in resnetd_settings.keys(), f'ResNetD model name should be in {list(resnetd_settings.keys())}'
        block, depths, channels = resnetd_settings[model_name]
        self.inplanes = 128
        self.channels = channels
        self.stem = Stem(3, 64, self.inplanes)
        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2, d=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2, d=4)

    def _make_layer(self, block, planes, depth, s=1, d=1) ->nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = nn.Sequential(block(self.inplanes, planes, s, d, downsample=downsample), *[block(planes * block.expansion, planes, d=d) for _ in range(1, depth)])
        self.inplanes = planes * block.expansion
        return layers

    def forward(self, x: Tensor) ->Tensor:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class PA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return x * self.pa_conv(x).sigmoid()


rest_settings = {'S': [[64, 128, 256, 512], [2, 2, 6, 2], 0.1], 'B': [[96, 192, 384, 768], [2, 2, 6, 2], 0.2], 'L': [[96, 192, 384, 768], [2, 2, 18, 2], 0.3]}


class ResT(nn.Module):

    def __init__(self, model_name: str='S') ->None:
        super().__init__()
        assert model_name in rest_settings.keys(), f'ResT model name should be in {list(rest_settings.keys())}'
        embed_dims, depths, drop_path_rate = rest_settings[model_name]
        self.channels = embed_dims
        self.stem = Stem(3, embed_dims[0])
        self.patch_embed_2 = PatchEmbed(embed_dims[0], embed_dims[1])
        self.patch_embed_3 = PatchEmbed(embed_dims[1], embed_dims[2])
        self.patch_embed_4 = PatchEmbed(embed_dims[2], embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        cur += depths[0]
        self.stage2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        cur += depths[1]
        self.stage3 = nn.ModuleList([Block(embed_dims[2], 4, 2, dpr[cur + i]) for i in range(depths[2])])
        cur += depths[2]
        self.stage4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm = nn.LayerNorm(embed_dims[-1])

    def forward(self, x: Tensor) ->Tensor:
        B = x.shape[0]
        x, H, W = self.stem(x)
        for blk in self.stage1:
            x = blk(x, H, W)
        x1 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x, H, W = self.patch_embed_2(x1)
        for blk in self.stage2:
            x = blk(x, H, W)
        x2 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x, H, W = self.patch_embed_3(x2)
        for blk in self.stage3:
            x = blk(x, H, W)
        x3 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x, H, W = self.patch_embed_4(x3)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x4 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x1, x2, x3, x4


class CMLP(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim=None) ->None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) ->Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CBlock(nn.Module):

    def __init__(self, dim, dpr=0.0):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMLP(dim, int(dim * 4))

    def forward(self, x: Tensor) ->Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):

    def __init__(self, dim, num_heads, dpr=0.0) ->None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor) ->Tensor:
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


uniformer_settings = {'S': [3, 4, 8, 3], 'B': [5, 8, 20, 7]}


class UniFormer(nn.Module):

    def __init__(self, model_name: str='S') ->None:
        super().__init__()
        assert model_name in uniformer_settings.keys(), f'UniFormer model name should be in {list(uniformer_settings.keys())}'
        depth = uniformer_settings[model_name]
        head_dim = 64
        drop_path_rate = 0.0
        embed_dims = [64, 128, 320, 512]
        for i in range(4):
            self.add_module(f'patch_embed{i + 1}', PatchEmbed(4 if i == 0 else 2, 3 if i == 0 else embed_dims[i - 1], embed_dims[i]))
            self.add_module(f'norm{i + 1}', nn.LayerNorm(embed_dims[i]))
        self.pos_drop = nn.Dropout(0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [(dim // head_dim) for dim in embed_dims]
        self.blocks1 = nn.ModuleList([CBlock(embed_dims[0], dpr[i]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([CBlock(embed_dims[1], dpr[i + depth[0]]) for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([SABlock(embed_dims[2], num_heads[2], dpr[i + depth[0] + depth[1]]) for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([SABlock(embed_dims[3], num_heads[3], dpr[i + depth[0] + depth[1] + depth[2]]) for i in range(depth[3])])

    def forward(self, x: torch.Tensor):
        outs = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x_out = self.norm1(x.permute(0, 2, 3, 1))
        outs.append(x_out.permute(0, 3, 1, 2))
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x_out = self.norm2(x.permute(0, 2, 3, 1))
        outs.append(x_out.permute(0, 3, 1, 2))
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        outs.append(x_out.permute(0, 3, 1, 2))
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        outs.append(x_out.permute(0, 3, 1, 2))
        return outs


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class BaseModel(nn.Module):

    def __init__(self, backbone: str='MiT-B0', num_classes: int=19) ->None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)

    def _init_weights(self, m: nn.Module) ->None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str=None) ->None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)


class SpatialPath(nn.Module):

    def __init__(self, c1, c2) ->None:
        super().__init__()
        ch = 64
        self.conv_7x7 = ConvModule(c1, ch, 7, 2, 3)
        self.conv_3x3_1 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_3x3_2 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_1x1 = ConvModule(ch, c2, 1, 1, 0)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class AttentionRefinmentModule(nn.Module):

    def __init__(self, c1, c2) ->None:
        super().__init__()
        self.conv_3x3 = ConvModule(c1, c2, 3, 1, 1)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.Sigmoid())

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se


class ContextPath(nn.Module):

    def __init__(self, backbone: nn.Module) ->None:
        super().__init__()
        self.backbone = backbone
        c3, c4 = self.backbone.channels[-2:]
        self.arm16 = AttentionRefinmentModule(c3, 128)
        self.arm32 = AttentionRefinmentModule(c4, 128)
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvModule(c4, 128, 1, 1, 0))
        self.up16 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.refine16 = ConvModule(128, 128, 3, 1, 1)
        self.refine32 = ConvModule(128, 128, 3, 1, 1)

    def forward(self, x):
        _, _, down16, down32 = self.backbone(x)
        arm_down16 = self.arm16(down16)
        arm_down32 = self.arm32(down32)
        global_down32 = self.global_context(down32)
        global_down32 = F.interpolate(global_down32, size=down32.size()[2:], mode='bilinear', align_corners=True)
        arm_down32 = arm_down32 + global_down32
        arm_down32 = self.up32(arm_down32)
        arm_down32 = self.refine32(arm_down32)
        arm_down16 = arm_down16 + arm_down32
        arm_down16 = self.up16(arm_down16)
        arm_down16 = self.refine16(arm_down16)
        return arm_down16, arm_down32


class FeatureFusionModule(nn.Module):

    def __init__(self, c1, c2, reduction=1) ->None:
        super().__init__()
        self.conv_1x1 = ConvModule(c1, c2, 1, 1, 0)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2 // reduction, 1, bias=False), nn.ReLU(True), nn.Conv2d(c2 // reduction, c2, 1, bias=False), nn.Sigmoid())

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        return fm + fm * fm_se


class Head(nn.Module):

    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) ->None:
        super().__init__()
        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)


class BiSeNetv1(nn.Module):

    def __init__(self, backbone: str='ResNet-18', num_classes: int=19) ->None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.context_path = ContextPath(eval(backbone)(variant))
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionModule(256, 256)
        self.output_head = Head(256, num_classes, upscale_factor=8, is_aux=False)
        self.context16_head = Head(128, num_classes, upscale_factor=8, is_aux=True)
        self.context32_head = Head(128, num_classes, upscale_factor=16, is_aux=True)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) ->None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str=None) ->None:
        if pretrained:
            self.context_path.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        context16, context32 = self.context_path(x)
        fm_fuse = self.ffm(spatial_out, context16)
        output = self.output_head(fm_fuse)
        if self.training:
            context_out16 = self.context16_head(context16)
            context_out32 = self.context32_head(context32)
            return output, context_out16, context_out32
        return output


class DetailBranch(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.S1 = nn.Sequential(ConvModule(3, 64, 3, 2, 1), ConvModule(64, 64, 3, 1, 1))
        self.S2 = nn.Sequential(ConvModule(64, 64, 3, 2, 1), ConvModule(64, 64, 3, 1, 1), ConvModule(64, 64, 3, 1, 1))
        self.S3 = nn.Sequential(ConvModule(64, 128, 3, 2, 1), ConvModule(128, 128, 3, 1, 1), ConvModule(128, 128, 3, 1, 1))

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.conv_3x3 = ConvModule(3, 16, 3, 2, 1)
        self.left = nn.Sequential(ConvModule(16, 8, 1, 1, 0), ConvModule(8, 16, 3, 2, 1))
        self.right = nn.MaxPool2d(3, 2, 1, ceil_mode=False)
        self.fuse = ConvModule(32, 16, 3, 1, 1)

    def forward(self, x):
        x = self.conv_3x3(x)
        x_left = self.left(x)
        x_right = self.right(x)
        y = torch.cat([x_left, x_right], dim=1)
        return self.fuse(y)


class ContextEmbeddingBlock(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.inner = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.BatchNorm2d(128), ConvModule(128, 128, 1, 1, 0))
        self.conv = ConvModule(128, 128, 3, 1, 1)

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.conv(out)


class GatherExpansionLayerv1(nn.Module):

    def __init__(self, in_ch, out_ch, e=6) ->None:
        super().__init__()
        self.inner = nn.Sequential(ConvModule(in_ch, in_ch, 3, 1, 1), ConvModule(in_ch, in_ch * e, 3, 1, 1, g=in_ch), nn.Conv2d(in_ch * e, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.relu(out)


class GatherExpansionLayerv2(nn.Module):

    def __init__(self, in_ch, out_ch, e=6) ->None:
        super().__init__()
        self.inner = nn.Sequential(ConvModule(in_ch, in_ch, 3, 1, 1), nn.Conv2d(in_ch, in_ch * e, 3, 2, 1, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch * e), ConvModule(in_ch * e, in_ch * e, 3, 1, 1, g=in_ch * e), nn.Conv2d(in_ch * e, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.outer = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.inner(x)
        x2 = self.outer(x)
        out = x1 + x2
        return self.relu(out)


class SemanticBranch(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(GatherExpansionLayerv2(16, 32), GatherExpansionLayerv1(32, 32))
        self.S4 = nn.Sequential(GatherExpansionLayerv2(32, 64), GatherExpansionLayerv1(64, 64))
        self.S5_1 = nn.Sequential(GatherExpansionLayerv2(64, 128), GatherExpansionLayerv1(128, 128), GatherExpansionLayerv1(128, 128), GatherExpansionLayerv1(128, 128))
        self.S5_2 = ContextEmbeddingBlock()

    def forward(self, x):
        x2 = self.S1S2(x)
        x3 = self.S3(x2)
        x4 = self.S4(x3)
        x5_1 = self.S5_1(x4)
        x5_2 = self.S5_2(x5_1)
        return x2, x3, x4, x5_1, x5_2


class AggregationLayer(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.left1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 1, 1, 0, bias=False))
        self.left2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.AvgPool2d(3, 2, 1, ceil_mode=False))
        self.right1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.Upsample(scale_factor=4), nn.Sigmoid())
        self.right2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 1, 1, 0, bias=False), nn.Sigmoid())
        self.up = nn.Upsample(scale_factor=4)
        self.conv = ConvModule(128, 128, 3, 1, 1)

    def forward(self, x_d, x_s):
        x1 = self.left1(x_d)
        x2 = self.left2(x_d)
        x3 = self.right1(x_s)
        x4 = self.right2(x_s)
        left = x1 * x3
        right = x2 * x4
        right = self.up(right)
        out = left + right
        return self.conv(out)


class SegHead(nn.Module):

    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) ->Tensor:
        x = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))
        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class BiSeNetv2(nn.Module):

    def __init__(self, backbone: str=None, num_classes: int=19) ->None:
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_layer = AggregationLayer()
        self.output_head = SegHead(128, 1024, num_classes, upscale_factor=8, is_aux=False)
        self.aux2_head = SegHead(16, 128, num_classes, upscale_factor=4)
        self.aux3_head = SegHead(32, 128, num_classes, upscale_factor=8)
        self.aux4_head = SegHead(64, 128, num_classes, upscale_factor=16)
        self.aux5_head = SegHead(128, 128, num_classes, upscale_factor=32)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) ->None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str=None) ->None:
        pass

    def forward(self, x):
        x_d = self.detail_branch(x)
        aux2, aux3, aux4, aux5, x_s = self.semantic_branch(x)
        output = self.aggregation_layer(x_d, x_s)
        output = self.output_head(output)
        if self.training:
            aux2 = self.aux2_head(aux2)
            aux3 = self.aux3_head(aux3)
            aux4 = self.aux4_head(aux4)
            aux5 = self.aux5_head(aux5)
            return output, aux2, aux3, aux4, aux5
        return output


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """

    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvModule(c1, c2, 1)) for scale in scales])
        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) ->Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))
        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """

    def __init__(self, in_channels, channel=128, num_classes: int=19, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channel, scales)
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        for in_ch in in_channels[:-1]:
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))
        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) ->Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]
        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))
        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output


class CustomCNN(BaseModel):

    def __init__(self, backbone: str='ResNet-50', num_classes: int=19):
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) ->Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class CustomVIT(BaseModel):

    def __init__(self, backbone: str='ResT-S', num_classes: int=19) ->None:
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) ->Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class ConvBN(nn.Sequential):

    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(nn.Conv2d(c1, c2, k, s, p, bias=False), nn.BatchNorm2d(c2))


class Conv2BN(nn.Sequential):

    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(nn.Conv2d(c1, ch, k, s, p, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True), nn.Conv2d(ch, c2, k, s, p, bias=False), nn.BatchNorm2d(c2))


class Scale(nn.Sequential):

    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(nn.AvgPool2d(k, s, p), nn.BatchNorm2d(c1), nn.ReLU(True), nn.Conv2d(c1, c2, 1, bias=False))


class ScaleLast(nn.Sequential):

    def __init__(self, c1, c2, k):
        super().__init__(nn.AdaptiveAvgPool2d(k), nn.BatchNorm2d(c1), nn.ReLU(True), nn.Conv2d(c1, c2, 1, bias=False))


class DAPPM(nn.Module):

    def __init__(self, c1, ch, c2):
        super().__init__()
        self.scale1 = Scale(c1, ch, 5, 2, 2)
        self.scale2 = Scale(c1, ch, 9, 4, 4)
        self.scale3 = Scale(c1, ch, 17, 8, 8)
        self.scale4 = ScaleLast(c1, ch, 1)
        self.scale0 = ConvModule(c1, ch, 1)
        self.process1 = ConvModule(ch, ch, 3, 1, 1)
        self.process2 = ConvModule(ch, ch, 3, 1, 1)
        self.process3 = ConvModule(ch, ch, 3, 1, 1)
        self.process4 = ConvModule(ch, ch, 3, 1, 1)
        self.compression = ConvModule(ch * 5, c2, 1)
        self.shortcut = ConvModule(c1, c2, 1)

    def forward(self, x: Tensor) ->Tensor:
        outs = [self.scale0(x)]
        outs.append(self.process1(F.interpolate(self.scale1(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1]))
        outs.append(self.process2(F.interpolate(self.scale2(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1]))
        outs.append(self.process3(F.interpolate(self.scale3(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1]))
        outs.append(self.process4(F.interpolate(self.scale4(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1]))
        out = self.compression(torch.cat(outs, dim=1)) + self.shortcut(x)
        return out


class DDRNet(nn.Module):

    def __init__(self, backbone: str=None, num_classes: int=19) ->None:
        super().__init__()
        planes, spp_planes, head_planes = [32, 64, 128, 256, 512], 128, 64
        self.conv1 = Stem(3, planes[0])
        self.layer1 = self._make_layer(BasicBlock, planes[0], planes[0], 2)
        self.layer2 = self._make_layer(BasicBlock, planes[0], planes[1], 2, 2)
        self.layer3 = self._make_layer(BasicBlock, planes[1], planes[2], 2, 2)
        self.layer4 = self._make_layer(BasicBlock, planes[2], planes[3], 2, 2)
        self.layer5 = self._make_layer(Bottleneck, planes[3], planes[3], 1)
        self.layer3_ = self._make_layer(BasicBlock, planes[1], planes[1], 2)
        self.layer4_ = self._make_layer(BasicBlock, planes[1], planes[1], 2)
        self.layer5_ = self._make_layer(Bottleneck, planes[1], planes[1], 1)
        self.compression3 = ConvBN(planes[2], planes[1], 1)
        self.compression4 = ConvBN(planes[3], planes[1], 1)
        self.down3 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down4 = Conv2BN(planes[1], planes[2], planes[3], 3, 2, 1)
        self.spp = DAPPM(planes[-1], spp_planes, planes[2])
        self.seghead_extra = SegHead(planes[1], head_planes, num_classes, 8)
        self.final_layer = SegHead(planes[2], head_planes, num_classes, 8)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) ->None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str=None) ->None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'], strict=False)

    def _make_layer(self, block, inplanes, planes, depths, s=1) ->nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, depths):
            if i == depths - 1:
                layers.append(block(inplanes, planes, no_relu=True))
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) ->Tensor:
        H, W = x.shape[-2] // 8, x.shape[-1] // 8
        layers = []
        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(F.relu(x))
        layers.append(x)
        x = self.layer3(F.relu(x))
        layers.append(x)
        x_ = self.layer3_(F.relu(layers[1]))
        x = x + self.down3(F.relu(x_))
        x_ = x_ + F.interpolate(self.compression3(F.relu(layers[2])), size=(H, W), mode='bilinear', align_corners=False)
        if self.training:
            x_aux = self.seghead_extra(x_)
        x = self.layer4(F.relu(x))
        layers.append(x)
        x_ = self.layer4_(F.relu(x_))
        x = x + self.down4(F.relu(x_))
        x_ = x_ + F.interpolate(self.compression4(F.relu(layers[3])), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.layer5_(F.relu(x_))
        x = F.interpolate(self.spp(self.layer5(F.relu(x))), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.final_layer(x + x_)
        return (x_, x_aux) if self.training else x_


def get_link(layer, base_ch, growth_rate):
    if layer == 0:
        return base_ch, 0, []
    link = []
    out_channels = growth_rate
    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            link.append(layer - dv)
            if i > 0:
                out_channels *= 1.7
    out_channels = int((out_channels + 1) / 2) * 2
    in_channels = 0
    for i in link:
        ch, _, _ = get_link(i, base_ch, growth_rate)
        in_channels += ch
    return out_channels, in_channels, link


class HarDBlock(nn.Module):

    def __init__(self, c1, growth_rate, n_layers):
        super().__init__()
        self.links = []
        layers = []
        self.out_channels = 0
        for i in range(n_layers):
            out_ch, in_ch, link = get_link(i + 1, c1, growth_rate)
            self.links.append(link)
            layers.append(ConvModule(in_ch, out_ch))
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) ->Tensor:
        layers = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers[i])
            if len(tin) > 1:
                x = torch.cat(tin, dim=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers.append(out)
        t = len(layers)
        outs = []
        for i in range(t):
            if i == t - 1 or i % 2 == 1:
                outs.append(layers[i])
        out = torch.cat(outs, dim=1)
        return out


class FCHarDNet(nn.Module):

    def __init__(self, backbone: str=None, num_classes: int=19) ->None:
        super().__init__()
        first_ch, ch_list, gr, n_layers = [16, 24, 32, 48], [64, 96, 160, 224, 320], [10, 16, 18, 24, 32], [4, 4, 8, 8, 8]
        self.base = nn.ModuleList([])
        self.base.append(ConvModule(3, first_ch[0], 3, 2))
        self.base.append(ConvModule(first_ch[0], first_ch[1], 3))
        self.base.append(ConvModule(first_ch[1], first_ch[2], 3, 2))
        self.base.append(ConvModule(first_ch[2], first_ch[3], 3))
        self.shortcut_layers = []
        skip_connection_channel_counts = []
        ch = first_ch[-1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], n_layers[i])
            ch = blk.out_channels
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < len(n_layers) - 1:
                self.shortcut_layers.append(len(self.base) - 1)
            self.base.append(ConvModule(ch, ch_list[i], k=1))
            ch = ch_list[i]
            if i < len(n_layers) - 1:
                self.base.append(nn.AvgPool2d(2, 2))
        prev_block_channels = ch
        self.n_blocks = len(n_layers) - 1
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])
        for i in range(self.n_blocks - 1, -1, -1):
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            blk = HarDBlock(cur_channels_count // 2, gr[i], n_layers[i])
            prev_block_channels = blk.out_channels
            self.conv1x1_up.append(ConvModule(cur_channels_count, cur_channels_count // 2, 1))
            self.denseBlocksUp.append(blk)
        self.finalConv = nn.Conv2d(prev_block_channels, num_classes, 1, 1, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) ->None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str=None) ->None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) ->Tensor:
        H, W = x.shape[-2:]
        skip_connections = []
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, skip], dim=1)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out


class CondHead(nn.Module):

    def __init__(self, in_channel: int=2048, channel: int=512, num_classes: int=19):
        super().__init__()
        self.num_classes = num_classes
        self.weight_num = channel * num_classes
        self.bias_num = num_classes
        self.conv = ConvModule(in_channel, channel, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.guidance_project = nn.Conv2d(channel, num_classes, 1)
        self.filter_project = nn.Conv2d(channel * num_classes, self.weight_num + self.bias_num, 1, groups=num_classes)

    def forward(self, features) ->Tensor:
        x = self.dropout(self.conv(features[-1]))
        B, C, H, W = x.shape
        guidance_mask = self.guidance_project(x)
        cond_logit = guidance_mask
        key = x
        value = x
        guidance_mask = guidance_mask.softmax(dim=1).view(*guidance_mask.shape[:2], -1)
        key = key.view(B, C, -1).permute(0, 2, 1)
        cond_filters = torch.matmul(guidance_mask, key)
        cond_filters /= H * W
        cond_filters = cond_filters.view(B, -1, 1, 1)
        cond_filters = self.filter_project(cond_filters)
        cond_filters = cond_filters.view(B, -1)
        weight, bias = torch.split(cond_filters, [self.weight_num, self.bias_num], dim=1)
        weight = weight.reshape(B * self.num_classes, -1, 1, 1)
        bias = bias.reshape(B * self.num_classes)
        value = value.view(-1, H, W).unsqueeze(0)
        seg_logit = F.conv2d(value, weight, bias, 1, 0, groups=B).view(B, self.num_classes, H, W)
        if self.training:
            return cond_logit, seg_logit
        return seg_logit


class DCNv2(nn.Module):

    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) ->Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))
        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


class FaPNHead(nn.Module):

    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])
        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) ->Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])
        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        out = self.conv_seg(self.dropout(out))
        return out


class FCNHead(nn.Module):

    def __init__(self, c1, c2, num_classes: int=19):
        super().__init__()
        self.conv = ConvModule(c1, c2, 1)
        self.cls = nn.Conv2d(c2, num_classes, 1)

    def forward(self, features) ->Tensor:
        x = self.conv(features[-1])
        x = self.cls(x)
        return x


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """

    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])
        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) ->Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out


class LawinAttn(nn.Module):

    def __init__(self, in_ch=512, head=4, patch_size=8, reduction=2) ->None:
        super().__init__()
        self.head = head
        self.position_mixing = nn.ModuleList([nn.Linear(patch_size * patch_size, patch_size * patch_size) for _ in range(self.head)])
        self.inter_channels = max(in_ch // reduction, 1)
        self.g = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_ch, self.inter_channels, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(self.inter_channels, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch))

    def forward(self, query: Tensor, context: Tensor) ->Tensor:
        B, C, H, W = context.shape
        context = context.reshape(B, C, -1)
        context_mlp = []
        for i, pm in enumerate(self.position_mixing):
            context_crt = context[:, C // self.head * i:C // self.head * (i + 1), :]
            context_mlp.append(pm(context_crt))
        context_mlp = torch.cat(context_mlp, dim=1)
        context = context + context_mlp
        context = context.reshape(B, C, H, W)
        g_x = self.g(context).view(B, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(query).view(B, self.inter_channels, -1)
        theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(context).view(B, self.inter_channels, -1)
        phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        y = y.permute(0, 2, 1).contiguous().reshape(B, self.inter_channels, *query.shape[-2:])
        output = query + self.conv_out(y)
        return output


class LawinHead(nn.Module):

    def __init__(self, in_channels: list, embed_dim=512, num_classes=19) ->None:
        super().__init__()
        for i, dim in enumerate(in_channels):
            self.add_module(f'linear_c{i + 1}', MLP(dim, 48 if i == 0 else embed_dim))
        self.lawin_8 = LawinAttn(embed_dim, 64)
        self.lawin_4 = LawinAttn(embed_dim, 16)
        self.lawin_2 = LawinAttn(embed_dim, 4)
        self.ds_8 = PatchEmbed(8, embed_dim, embed_dim)
        self.ds_4 = PatchEmbed(4, embed_dim, embed_dim)
        self.ds_2 = PatchEmbed(2, embed_dim, embed_dim)
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvModule(embed_dim, embed_dim))
        self.linear_fuse = ConvModule(embed_dim * 3, embed_dim)
        self.short_path = ConvModule(embed_dim, embed_dim)
        self.cat = ConvModule(embed_dim * 5, embed_dim)
        self.low_level_fuse = ConvModule(embed_dim + 48, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def get_lawin_att_feats(self, x: Tensor, patch_size: int):
        _, _, H, W = x.shape
        query = F.unfold(x, patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=H // patch_size, nw=W // patch_size)
        outs = []
        for r in [8, 4, 2]:
            context = F.unfold(x, patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            context = rearrange(context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size * r, pw=patch_size * r, nh=H // patch_size, nw=W // patch_size)
            context = getattr(self, f'ds_{r}')(context)
            output = getattr(self, f'lawin_{r}')(query, context)
            output = rearrange(output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size, nh=H // patch_size, nw=W // patch_size)
            outs.append(output)
        return outs

    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [self.linear_c2(features[1]).permute(0, 2, 1).reshape(B, -1, *features[1].shape[-2:])]
        for i, feature in enumerate(features[2:]):
            cf = eval(f'self.linear_c{i + 3}')(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape
        feat_short = self.short_path(feat)
        feat_pool = F.interpolate(self.image_pool(feat), size=(H, W), mode='bilinear', align_corners=False)
        feat_lawin = self.get_lawin_att_feats(feat, 8)
        output = self.cat(torch.cat([feat_short, feat_pool, *feat_lawin], dim=1))
        c1 = self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])
        output = F.interpolate(output, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))
        seg = self.linear_pred(self.dropout(fused))
        return seg


class SegFormerHead(nn.Module):

    def __init__(self, dims: list, embed_dim: int=256, num_classes: int=19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f'linear_c{i + 1}', MLP(dim, embed_dim))
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) ->Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]
        for i, feature in enumerate(features[1:]):
            cf = eval(f'self.linear_c{i + 2}')(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg


class AlignedModule(nn.Module):

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.down_h = nn.Conv2d(c1, c2, 1, bias=False)
        self.down_l = nn.Conv2d(c1, c2, 1, bias=False)
        self.flow_make = nn.Conv2d(c2 * 2, 2, k, 1, 1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor) ->Tensor:
        high_feature_origin = high_feature
        H, W = low_feature.shape[-2:]
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(high_feature, size=(H, W), mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim=1))
        high_feature = self.flow_warp(high_feature_origin, flow, (H, W))
        return high_feature

    def flow_warp(self, x: Tensor, flow: Tensor, size: tuple) ->Tensor:
        norm = torch.tensor([[[[*size]]]]).type_as(x)
        H = torch.linspace(-1.0, 1.0, size[0]).view(-1, 1).repeat(1, size[1])
        W = torch.linspace(-1.0, 1.0, size[1]).repeat(size[0], 1)
        grid = torch.cat((W.unsqueeze(2), H.unsqueeze(2)), dim=2)
        grid = grid.repeat(x.shape[0], 1, 1, 1).type_as(x)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=False)
        return output


class SFHead(nn.Module):

    def __init__(self, in_channels, channel=256, num_classes=19, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channel, scales)
        self.fpn_in = nn.ModuleList([])
        self.fpn_out = nn.ModuleList([])
        self.fpn_out_align = nn.ModuleList([])
        for in_ch in in_channels[:-1]:
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))
            self.fpn_out_align.append(AlignedModule(channel, channel // 2))
        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)

    def forward(self, features: list) ->Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]
        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + self.fpn_out_align[i](feature, f)
            fpn_features.append(self.fpn_out[i](f))
        fpn_features.reverse()
        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=True)
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output


class Lawin(BaseModel):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, backbone: str='MiT-B0', num_classes: int=19) ->None:
        super().__init__(backbone, num_classes)
        self.decode_head = LawinHead(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) ->Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class PSAP(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Conv2d(ch, c2, 1, bias=False)
        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) ->Tensor:
        input_x = self.conv_v_right(x)
        context_mask = self.conv_q_right(x)
        B, C, _, _ = input_x.shape
        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)
        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) ->Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1)
        context = avg_x @ theta_x
        context = context.softmax(dim=2).view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self.spatial_pool(x) + self.channel_pool(x)


class PSAS(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Sequential(nn.Conv2d(ch, ch // 4, 1), nn.LayerNorm([ch // 4, 1, 1]), nn.ReLU(), nn.Conv2d(ch // 4, c2, 1))
        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) ->Tensor:
        input_x = self.conv_v_right(x)
        context_mask = self.conv_q_right(x)
        B, C, _, _ = input_x.shape
        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)
        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) ->Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1).softmax(dim=2)
        context = avg_x @ theta_x
        context = context.view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self.channel_pool(self.spatial_pool(x))


class SegFormer(BaseModel):

    def __init__(self, backbone: str='MiT-B0', num_classes: int=19) ->None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) ->Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class SFNet(BaseModel):

    def __init__(self, backbone: str='ResNetD-18', num_classes: int=19):
        assert 'ResNet' in backbone
        super().__init__(backbone, num_classes)
        self.head = SFHead(self.backbone.channels, 128 if '18' in backbone else 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) ->Tensor:
        outs = self.backbone(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AggregationLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 256, 256]), torch.rand([4, 128, 64, 64])], {}),
     True),
    (AlignedModule,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionRefinmentModule,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiSeNetv1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (CBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CMLP,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2BN,
     lambda: ([], {'c1': 4, 'ch': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (ConvBN,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvModule,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DAPPM,
     lambda: ([], {'c1': 4, 'ch': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DDRNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DepthSpatialSepConv,
     lambda: ([], {'c1': 4, 'expand': [4, 4], 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DetailBranch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Downsample,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FSM,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureFusionModule,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (GatherExpansionLayerv1,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatherExpansionLayerv2,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'c1': 4, 'c2': 4, 's': 4, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'dim': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MobileNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PA,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PPM,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSAP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Scale,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleLast,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SegHead,
     lambda: ([], {'c1': 4, 'ch': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SemanticBranch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SpatialPath,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialSepConvSF,
     lambda: ([], {'c1': 4, 'outs': [4, 4], 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExcitation,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stem,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StemBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SwishLinear,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_sithu31296_semantic_segmentation(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

