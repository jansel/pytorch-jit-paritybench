import sys
_module = sys.modules[__name__]
del sys
detect = _module
models = _module
export = _module
models = _module
test = _module
train = _module
tune = _module
utils = _module
activations = _module
autoanchor = _module
datasets = _module
general = _module
google_utils = _module
layers = _module
loss = _module
metrics = _module
parse_config = _module
plots = _module
torch_utils = _module

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


import torch


import torch.backends.cudnn as cudnn


from numpy import random


import numpy as np


import logging


import math


import random


from warnings import warn


import torch.distributed as dist


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data


from torch.cuda import amp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from scipy.cluster.vq import kmeans


from itertools import repeat


from torch.utils.data import Dataset


from copy import deepcopy


from torchvision.utils import save_image


import re


import matplotlib


from torch import nn


from copy import copy


import matplotlib.pyplot as plt


from scipy.signal import butter


from scipy.signal import filtfilt


import torchvision


ONNX_EXPORT = False


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec
            self.anchor_wh = self.anchor_wh

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            w = torch.sigmoid(p[:, -n:]) * (2 / n)
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh
        else:
            io = p.sigmoid()
            io[..., :2] = io[..., :2] * 2.0 - 0.5 + self.grid
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            return io.view(bs, -1, self.no), p


class JDELayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(JDELayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec
            self.anchor_wh = self.anchor_wh

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            w = torch.sigmoid(p[:, -n:]) * (2 / n)
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) * 2.0 - 0.5 + self.grid
            io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            io[..., 4:] = F.softmax(io[..., 4:])
            return io.view(bs, -1, self.no), p


class AlternateChannel(nn.Module):

    def __init__(self, layers):
        super(AlternateChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return torch.cat([a.expand_as(x), x], dim=1)


class AlternateChannel2D(nn.Module):

    def __init__(self, layers):
        super(AlternateChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return torch.cat([a.expand_as(x), x], dim=1)


class ControlChannel(nn.Module):

    def __init__(self, layers):
        super(ControlChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) * x


class ControlChannel2D(nn.Module):

    def __init__(self, layers):
        super(ControlChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.expand_as(x) * x


class DeformConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


class FeatureConcat(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class FeatureConcat2(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat2, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach()], 1)


class FeatureConcat3(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat3, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach(), outputs[self.layers[2]].detach()], 1)


class FeatureConcat_l(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat_l, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i][:, :outputs[i].shape[1] // 2, :, :] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]][:, :outputs[self.layers[0]].shape[1] // 2, :, :]


class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.avg_pool(x)


class Implicit2DA(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class Implicit2DC(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class Implicit2DM(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, atom, channel, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self):
        return self.implicit


class ImplicitA(nn.Module):

    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class ImplicitC(nn.Module):

    def __init__(self, channel):
        super(ImplicitC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):

    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self):
        return self.implicit


class Mish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MixConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if method == 'equal_ch':
            i = torch.linspace(0, groups - 1e-06, out_ch).floor()
            ch = [(i == g).sum() for g in range(groups)]
        else:
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)
        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch, out_channels=ch[g], kernel_size=k[g], stride=stride, padding=k[g] // 2, dilation=dilation, bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


class Reorg(nn.Module):

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class ScaleChannel(nn.Module):

    def __init__(self, layers):
        super(ScaleChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a


class ScaleSpatial(nn.Module):

    def __init__(self, layers):
        super(ScaleSpatial, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x * a


class SelectChannel(nn.Module):

    def __init__(self, layers):
        super(SelectChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.sigmoid().expand_as(x) * x


class SelectChannel2D(nn.Module):

    def __init__(self, layers):
        super(SelectChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.sigmoid().expand_as(x) * x


class ShiftChannel(nn.Module):

    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x


class ShiftChannel2D(nn.Module):

    def __init__(self, layers):
        super(ShiftChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.expand_as(x) + x


class Silence(nn.Module):

    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        return x


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class WeightedFeatureFusion(nn.Module):

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]
            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
        return x


def create_modules(module_defs, img_size, cfg):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    _ = module_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=k, stride=stride, padding=k // 2 if mdef['pad'] else 0, groups=mdef['groups'] if 'groups' in mdef else 1, bias=not bn))
            else:
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001))
            else:
                routs.append(i)
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'emb':
                modules.add_module('activation', F.normalize())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())
        elif mdef['type'] == 'deformableconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('DeformConv2d', DeformConv2d(output_filters[-1], filters, kernel_size=k, padding=k // 2 if mdef['pad'] else 0, stride=stride, bias=not bn, modulation=True))
            else:
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001))
            else:
                routs.append(i)
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())
        elif mdef['type'] == 'dropout':
            p = mdef['probability']
            modules = nn.Dropout(p)
        elif mdef['type'] == 'avgpool':
            modules = GAP()
        elif mdef['type'] == 'silence':
            filters = output_filters[-1]
            modules = Silence()
        elif mdef['type'] == 'scale_channels':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ScaleChannel(layers=layers)
        elif mdef['type'] == 'shift_channels':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ShiftChannel(layers=layers)
        elif mdef['type'] == 'shift_channels_2d':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ShiftChannel2D(layers=layers)
        elif mdef['type'] == 'control_channels':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ControlChannel(layers=layers)
        elif mdef['type'] == 'control_channels_2d':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ControlChannel2D(layers=layers)
        elif mdef['type'] == 'alternate_channels':
            layers = mdef['from']
            filters = output_filters[-1] * 2
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = AlternateChannel(layers=layers)
        elif mdef['type'] == 'alternate_channels_2d':
            layers = mdef['from']
            filters = output_filters[-1] * 2
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = AlternateChannel2D(layers=layers)
        elif mdef['type'] == 'select_channels':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = SelectChannel(layers=layers)
        elif mdef['type'] == 'select_channels_2d':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = SelectChannel2D(layers=layers)
        elif mdef['type'] == 'sam':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = ScaleSpatial(layers=layers)
        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001)
            if i == 0 and filters == 3:
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])
        elif mdef['type'] == 'maxpool':
            k = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
        elif mdef['type'] == 'local_avgpool':
            k = mdef['size']
            stride = mdef['stride']
            avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('AvgPool2d', avgpool)
            else:
                modules = avgpool
        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2 / 32
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])
        elif mdef['type'] == 'route':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = FeatureConcat(layers=layers)
        elif mdef['type'] == 'route2':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = FeatureConcat2(layers=layers)
        elif mdef['type'] == 'route3':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = FeatureConcat3(layers=layers)
        elif mdef['type'] == 'route_lhalf':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers]) // 2
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = FeatureConcat_l(layers=layers)
        elif mdef['type'] == 'shortcut':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
        elif mdef['type'] == 'reorg3d':
            pass
        elif mdef['type'] == 'reorg':
            filters = 4 * output_filters[-1]
            modules.add_module('Reorg', Reorg())
        elif mdef['type'] == 'dwt':
            filters = 4 * output_filters[-1]
            modules.add_module('DWT', DWT())
        elif mdef['type'] == 'implicit_add':
            filters = mdef['filters']
            modules = ImplicitA(channel=filters)
        elif mdef['type'] == 'implicit_mul':
            filters = mdef['filters']
            modules = ImplicitM(channel=filters)
        elif mdef['type'] == 'implicit_cat':
            filters = mdef['filters']
            modules = ImplicitC(channel=filters)
        elif mdef['type'] == 'implicit_add_2d':
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DA(atom=filters, channel=channels)
        elif mdef['type'] == 'implicit_mul_2d':
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DM(atom=filters, channel=channels)
        elif mdef['type'] == 'implicit_cat_2d':
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DC(atom=filters, channel=channels)
        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):
                stride = [32, 16, 8]
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']], nc=mdef['classes'], img_size=img_size, yolo_index=yolo_index, layers=layers, stride=stride[yolo_index])
            try:
                j = layers[yolo_index] if 'from' in mdef else -2
                bias_ = module_list[j][0].bias
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                None
        elif mdef['type'] == 'jde':
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):
                stride = [32, 16, 8]
            layers = mdef['from'] if 'from' in mdef else []
            modules = JDELayer(anchors=mdef['anchors'][mdef['mask']], nc=mdef['classes'], img_size=img_size, yolo_index=yolo_index, layers=layers, stride=stride[yolo_index])
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                None
        else:
            None
        module_list.append(modules)
        output_filters.append(filters)
    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ in ['YOLOLayer', 'JDELayer']]


def parse_model_cfg(path):
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split('=')
            key = key.rstrip()
            if key == 'anchors':
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif key in ['from', 'layers', 'mask'] or key == 'size' and ',' in val:
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if int(val) - float(val) == 0 else float(val)
                else:
                    mdefs[-1][key] = val
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups', 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random', 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind', 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'atoms', 'na', 'nc']
    f = []
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]
    assert not any(u), 'Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631' % (u, path)
    return mdefs


class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        self.version = np.array([0, 2, 5], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)
        self.info(verbose) if not ONNX_EXPORT else None

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:
            img_size = x.shape[-2:]
            s = [0.83, 0.67]
            y = []
            for i, xi in enumerate((x, torch_utils.scale_img(x.flip(3), s[0], same_shape=False), torch_utils.scale_img(x, s[1], same_shape=False))):
                y.append(self.forward_once(xi)[0])
            y[1][..., :4] /= s[0]
            y[1][..., 0] = img_size[1] - y[1][..., 0]
            y[2][..., :4] /= s[1]
            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            None
            str = ''
        if augment:
            nb = x.shape[0]
            s = [0.83, 0.67]
            x = torch.cat((x, torch_utils.scale_img(x.flip(3), s[0]), torch_utils.scale_img(x, s[1])), 0)
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ShiftChannel', 'ShiftChannel2D', 'ControlChannel', 'ControlChannel2D', 'AlternateChannel', 'AlternateChannel2D', 'SelectChannel', 'SelectChannel2D', 'ScaleSpatial']:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str = ' >> ' + ' + '.join([('layer %g %s' % x) for x in zip(l, sh)])
                x = module(x, out)
            elif name in ['ImplicitA', 'ImplicitM', 'ImplicitC', 'Implicit2DA', 'Implicit2DM', 'Implicit2DC']:
                x = module()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            elif name == 'JDELayer':
                yolo_out.append(module(x, out))
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])
            if verbose:
                None
                str = ''
        if self.training:
            return yolo_out
        elif ONNX_EXPORT:
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            if augment:
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]
                x[1][..., 0] = img_size[1] - x[1][..., 0]
                x[2][..., :4] /= s[1]
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        None
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


class Hardswish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class MishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientMish(nn.Module):

    def forward(self, x):
        return MishImplementation.apply(x)


class FReLU(nn.Module):

    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class HardSwish(nn.Module):

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeformConv2d,
     lambda: ([], {'inc': 4, 'outc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FReLU,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureConcat,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     True),
    (FeatureConcat2,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GAP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hardswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Implicit2DA,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {}),
     True),
    (Implicit2DC,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {}),
     True),
    (Implicit2DM,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {}),
     True),
    (ImplicitA,
     lambda: ([], {'channel': 4}),
     lambda: ([], {}),
     True),
    (ImplicitC,
     lambda: ([], {'channel': 4}),
     lambda: ([], {}),
     True),
    (ImplicitM,
     lambda: ([], {'channel': 4}),
     lambda: ([], {}),
     True),
    (MemoryEfficientMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reorg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Silence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedFeatureFusion,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     False),
]

class Test_WongKinYiu_yolor(_paritybench_base):
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

