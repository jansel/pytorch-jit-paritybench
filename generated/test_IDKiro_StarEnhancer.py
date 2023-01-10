import sys
_module = sys.modules[__name__]
del sys
generate = _module
dataset = _module
loader = _module
fetch_embedding = _module
model = _module
basis = _module
enhancer = _module
loss = _module
mapping = _module
stylish = _module
test = _module
train_enhancer = _module
train_stylish = _module
utils = _module
common = _module
data_parallel = _module

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


import math


import torch


import numpy as np


from torch.utils.data import Dataset


import torch.nn as nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import time


from collections import OrderedDict


from torch.nn.parallel import DataParallel


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel.parallel_apply import parallel_apply


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlk(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlk, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.actv1(x + self.bias1a)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out + self.bias1b)
        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)
        out = out * self.scale
        out += identity
        return out


class DualAdainResBlk(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DualAdainResBlk, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sa, sb):
        identity = x
        out = self.actv1(x + self.bias1a)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out + self.bias1b)
        alpha = sb[0] / sa[0]
        beta = sb[1] - sa[1] * alpha
        out = out * alpha + beta
        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)
        out = out * self.scale
        out += identity
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CurveEncoder(nn.Module):

    def __init__(self, dims):
        super(CurveEncoder, self).__init__()
        self.layers = [2, 2, 2, 2]
        self.planes = [64, 128, 256, 512]
        self.dims = dims
        self.num_layers = sum(self.layers)
        self.inplanes = self.planes[0]
        self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*self._make_layer(ResBlk, self.planes[0], self.layers[0]))
        self.layer2 = nn.Sequential(*self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2))
        self.layer3 = nn.Sequential(*self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2))
        self.layer4 = self._make_layer(DualAdainResBlk, self.planes[3], self.layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(self.planes[3], self.dims)
        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, ResBlk) or isinstance(m, DualAdainResBlk):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** -0.5)
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(self.inplanes, planes, stride)
        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def forward(self, x, sa, sb):
        x = self.conv1(x)
        x = self.actv(x + self.bias1)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.layers[3]):
            x = self.layer4[i](x, sa[i], sb[i])
        x = self.gap(x).flatten(1)
        x = self.fc(x + self.bias2)
        return x


class Enhancer(nn.Module):

    def __init__(self):
        super(Enhancer, self).__init__()
        self.cd = 256
        self.cl = [86, 52, 52, 18, 18, 52, 86, 52, 18, 18, 52, 52, 86, 18, 18]
        self.encoder = CurveEncoder(sum(self.cl))

    def interp(self, param, length):
        return F.interpolate(param.unsqueeze(1).unsqueeze(2), (1, length), mode='bicubic', align_corners=True).squeeze(2).squeeze(1)

    def curve(self, x, func, depth):
        x_ind = torch.clamp(x, 0, 1) * (depth - 1)
        x_ind = x_ind.round_().long().flatten(1).detach()
        out = torch.gather(func, 1, x_ind)
        return out.reshape(x.size())

    def forward(self, x, sa, sb):
        _, _, H, W = x.size()
        fl = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)), sa, sb).split(self.cl, dim=1)
        residual = torch.cat([(self.curve(x[:, [0], ...], self.interp(fl[i * 5 + 0], self.cd), self.cd) + self.curve(x[:, [1], ...], self.interp(fl[i * 5 + 1], self.cd), self.cd) + self.curve(x[:, [2], ...], self.interp(fl[i * 5 + 2], self.cd), self.cd) + self.interp(fl[i * 5 + 3], H).unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, W) + self.interp(fl[i * 5 + 4], W).unsqueeze(1).unsqueeze(2).expand(-1, -1, H, -1)) for i in range(3)], dim=1)
        return x + residual


class SlimEnhancer(nn.Module):

    def __init__(self):
        super(SlimEnhancer, self).__init__()
        self.cd = 256
        self.cl = 64
        self.encoder = CurveEncoder(self.cl * 9)

    def interp(self, param, length):
        return F.interpolate(param.unsqueeze(1).unsqueeze(2), (1, length), mode='bicubic', align_corners=True).squeeze(2).squeeze(1)

    def curve(self, x, func, depth):
        x_ind = x * (depth - 1)
        x_ind = x_ind.long().flatten(2).detach()
        out = torch.gather(func, 2, x_ind)
        return out.reshape(x.size())

    def forward(self, x, sa, sb):
        B, _, H, W = x.size()
        params = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)), sa, sb).view(B, 9, self.cl, 1)
        curves = F.interpolate(params, (self.cd, 1), mode='bicubic', align_corners=True).squeeze(3)
        residual = self.curve(x.repeat(1, 3, 1, 1), curves, self.cd).view(B, 3, 3, H, W).sum(dim=2)
        return x + residual


class TotalLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def rgb2lab(self, rgb_image):
        rgb_to_xyz = torch.FloatTensor([[0.412453, 0.212671, 0.019334], [0.35758, 0.71516, 0.119193], [0.180423, 0.072169, 0.950227]]).t()
        fxfyfz_to_lab = torch.FloatTensor([[0.0, 500.0, 0.0], [116.0, -500.0, 200.0], [0.0, 0.0, -200.0]]).t()
        img = rgb_image / 12.92 * rgb_image.le(0.04045).float() + ((torch.clamp(rgb_image, min=0.0001) + 0.055) / 1.055) ** 2.4 * rgb_image.gt(0.04045).float()
        img = img.permute(1, 0, 2, 3).contiguous().view(3, -1)
        img = torch.matmul(rgb_to_xyz, img)
        img = torch.mul(img, torch.FloatTensor([1 / 0.950456, 1.0, 1 / 1.088754]).view(3, 1))
        epsilon = 6 / 29
        img = (img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float() + torch.clamp(img, min=0.0001) ** (1.0 / 3.0) * img.gt(epsilon ** 3).float()
        img = torch.matmul(fxfyfz_to_lab, img) + torch.FloatTensor([-16.0, 0.0, 0.0]).view(3, 1)
        img = img.view(3, rgb_image.size(0), rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2, 3)
        img[:, 0, :, :] = img[:, 0, :, :] / 100
        img[:, 1, :, :] = (img[:, 1, :, :] / 110 + 1) / 2
        img[:, 2, :, :] = (img[:, 2, :, :] / 110 + 1) / 2
        img[(img != img).detach()] = 0
        img = img.contiguous()
        return img

    def forward(self, out_image, gt_image):
        out_lab = self.rgb2lab(out_image)
        gt_lab = self.rgb2lab(gt_image)
        loss = F.l1_loss(out_lab, gt_lab)
        return loss


class Mapping(nn.Module):

    def __init__(self, in_dim):
        super(Mapping, self).__init__()
        self.layers = 2
        self.planes = 512
        self.mlp_in = nn.Sequential(nn.Linear(in_dim, 512), nn.PReLU(), nn.Linear(512, 512), nn.PReLU(), nn.Linear(512, 512), nn.PReLU(), nn.Linear(512, 512), nn.PReLU())
        self.mlp_out = nn.ModuleList()
        for _ in range(self.layers):
            self.mlp_out.append(nn.Sequential(nn.Linear(512, 512), nn.PReLU(), nn.Linear(512, self.planes * 2), nn.Sigmoid()))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp_in(x)
        s_list = []
        for i in range(self.layers):
            out = self.mlp_out[i](x).view(x.size(0), -1, 1, 1)
            s_list.append(list(torch.chunk(out, chunks=2, dim=1)))
        return s_list


class StyleEncoder(nn.Module):

    def __init__(self, dim):
        super(StyleEncoder, self).__init__()
        self.layers = [4, 4, 4, 4]
        self.planes = [64, 128, 256, 512]
        self.num_layers = sum(self.layers)
        self.inplanes = self.planes[0]
        self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlk, self.planes[0], self.layers[0])
        self.layer2 = self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlk, self.planes[3], self.layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(self.planes[3], dim)
        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, ResBlk):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** -0.5)
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(self.inplanes, planes, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x + self.bias1)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        avg_x = self.gap(x)
        max_x = self.gmp(x)
        x = (max_x + avg_x).flatten(1)
        x = self.fc(x + self.bias2)
        x = F.normalize(x, p=2, dim=1)
        return x


class Proxy(nn.Module):

    def __init__(self, dim, cN):
        super(Proxy, self).__init__()
        self.fc = Parameter(torch.Tensor(dim, cN))
        torch.nn.init.xavier_normal_(self.fc)

    def forward(self, input):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        return simInd


class Stylish(nn.Module):

    def __init__(self, dim, cN):
        super(Stylish, self).__init__()
        self.encoder = StyleEncoder(dim)
        self.proxy = Proxy(dim, cN)

    def forward(self, x):
        x = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)))
        x = self.proxy(x)
        return x


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                None
                None
                None
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if len(self.device_ids) == 1:
            inputs, kwargs = super().scatter(inputs, kwargs, self.device_ids)
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DualAdainResBlk,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Enhancer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Mapping,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Proxy,
     lambda: ([], {'dim': 4, 'cN': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlk,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SlimEnhancer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (StyleEncoder,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Stylish,
     lambda: ([], {'dim': 4, 'cN': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
]

class Test_IDKiro_StarEnhancer(_paritybench_base):
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

