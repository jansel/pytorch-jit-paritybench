import sys
_module = sys.modules[__name__]
del sys
basic = _module
blocks = _module
blocks_repvgg = _module
convert = _module
convnet_utils = _module
dbb_transforms = _module
repvgg = _module
resnet = _module
resnext = _module
test = _module
train = _module
utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch.nn.init as init


import math


import copy


import time


import torch.backends.cudnn as cudnn


import torchvision.datasets as datasets


import random


import warnings


import torch.nn.parallel


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


from torch.optim.lr_scheduler import CosineAnnealingLR


import torchvision.transforms as transforms


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * (gamma / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * gamma / std


class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPadLayer(nn.Module):

    def __init__(self, pad_pixels, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class PriorFilter(nn.Module):

    def __init__(self, channels, stride, padding, width=4):
        super(PriorFilter, self).__init__()
        self.stride = stride
        self.width = width
        self.group = int(channels / width)
        self.padding = padding
        self.prior_tensor = torch.Tensor(self.group, 1, 3, 3)
        half_g = int(self.group / 2)
        for i in range(self.group):
            for h in range(3):
                for w in range(3):
                    if i < half_g:
                        self.prior_tensor[i, 0, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        self.prior_tensor[i, 0, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_g) / 3)
        self.register_buffer('prior', self.prior_tensor)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.width, self.group, h, w)
        return F.conv2d(input=x, weight=self.prior, bias=None, stride=self.stride, padding=self.padding, groups=self.group).view(b, c, int(h / self.stride), int(w / self.stride))


class PriorConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, width=4):
        super(PriorConv, self).__init__()
        self.stride = stride
        self.width = width
        self.group = int(in_channels / width)
        self.padding = padding
        self.prior_tensor = torch.Tensor(self.group, 3, 3)
        half_g = int(self.group / 2)
        for i in range(self.group):
            for h in range(3):
                for w in range(3):
                    if i < half_g:
                        self.prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        self.prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_g) / 3)
        self.register_buffer('prior', self.prior_tensor)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_channels, width, self.group, kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        c_out, wi, g, kh, kw = self.weight.size()
        weight = torch.einsum('gmn,cwgmn->cwgmn', self.prior, self.weight).view(c_out, int(wi * g), kh, kw)
        return F.conv2d(input=x, weight=weight, bias=None, stride=self.stride, padding=self.padding)


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


class DBB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False):
        super(DBB, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3, kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))
        if single_init:
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)
        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0
        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second
        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels, kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride, padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))
        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, 'dbb_origin'):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, 'dbb_1x1'):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, 'dbb_avg'):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, 'dbb_1x1_kxk'):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, 'dbb_origin'):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class OREPA_1x1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None, single_init=False):
        super(OREPA_1x1, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert groups == 1
        assert kernel_size == 1
        assert padding == kernel_size // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if deploy:
            self.or1x1_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.branch_counter = 0
            self.weight_or1x1_origin = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
            init.kaiming_uniform_(self.weight_or1x1_origin, a=math.sqrt(1.0))
            self.branch_counter += 1
            if out_channels > in_channels:
                self.weight_or1x1_l2i_conv1 = nn.Parameter(torch.eye(in_channels).unsqueeze(2).unsqueeze(3))
                self.weight_or1x1_l2i_conv2 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
                init.kaiming_uniform_(self.weight_or1x1_l2i_conv2, a=math.sqrt(1.0))
            else:
                self.weight_or1x1_l2i_conv1 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
                init.kaiming_uniform_(self.weight_or1x1_l2i_conv1, a=math.sqrt(1.0))
                self.weight_or1x1_l2i_conv2 = nn.Parameter(torch.eye(out_channels).unsqueeze(2).unsqueeze(3))
            self.branch_counter += 1
            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            self.bn = nn.BatchNorm2d(self.out_channels)
            init.constant_(self.vector[0, :], 1.0)
            init.constant_(self.vector[1, :], 0.5)
            if single_init:
                self.single_init()

    def weight_gen(self):
        weight_or1x1_origin = torch.einsum('oihw,o->oihw', self.weight_or1x1_origin, self.vector[0, :])
        weight_or1x1_l2i = torch.einsum('tihw,othw->oihw', self.weight_or1x1_l2i_conv1, self.weight_or1x1_l2i_conv2)
        weight_or1x1_l2i = torch.einsum('oihw,o->oihw', weight_or1x1_l2i, self.vector[1, :])
        return weight_or1x1_origin + weight_or1x1_l2i

    def forward(self, inputs):
        if hasattr(self, 'or1x1_reparam'):
            return self.nonlinear(self.or1x1_reparam(inputs))
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or1x1_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or1x1_reparam.weight.data = kernel
        self.or1x1_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_or1x1_origin')
        self.__delattr__('weight_or1x1_l2i_conv1')
        self.__delattr__('weight_or1x1_l2i_conv2')
        self.__delattr__('vector')
        self.__delattr__('bn')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)


class OREPA(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False, weight_only=False, init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.weight_only = weight_only
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.branch_counter = 0
            self.weight_orepa_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1
            self.weight_orepa_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_orepa_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer('weight_orepa_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1
            self.weight_orepa_1x1 = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels
            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1
            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1
            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)
            self.fre_init()
            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))
            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)
            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))
            if single_init:
                self.single_init()

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw', self.weight_orepa_origin, self.vector[0, :])
        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum('oihw,o->oihw', torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2), self.weight_orepa_avg_avg), self.vector[1, :])
        weight_orepa_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2), self.weight_orepa_prior), self.vector[2, :])
        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 + self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2
        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2).reshape(o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw', weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])
        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1, self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1, self.vector[4, :])
        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw, self.weight_orepa_gconv_pw, self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv, self.vector[5, :])
        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv
        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)
        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i / groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))
        weight = self.weight_gen()
        if self.weight_only is True:
            return weight
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')
        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)


class OREPA_LargeConvBase(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deploy=False, nonlinear=None):
        super(OREPA_LargeConvBase, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 3
        self.stride = stride
        self.padding = padding
        self.layers = int((kernel_size - 1) / 2)
        self.groups = groups
        self.dilation = dilation
        internal_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.or_large_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            for i in range(self.layers):
                if i == 0:
                    self.__setattr__('weight' + str(i), nn.Parameter(torch.Tensor(internal_channels, int(in_channels / self.groups), 3, 3)))
                elif i == self.layers - 1:
                    self.__setattr__('weight' + str(i), nn.Parameter(torch.Tensor(out_channels, int(internal_channels / self.groups), 3, 3)))
                else:
                    self.__setattr__('weight' + str(i), nn.Parameter(torch.Tensor(internal_channels, int(internal_channels / self.groups), 3, 3)))
                init.kaiming_uniform_(getattr(self, 'weight' + str(i)), a=math.sqrt(5))
            self.bn = nn.BatchNorm2d(out_channels)

    def weight_gen(self):
        weight = getattr(self, 'weight' + str(0)).transpose(0, 1)
        for i in range(self.layers - 1):
            weight2 = getattr(self, 'weight' + str(i + 1))
            weight = F.conv2d(weight, weight2, groups=self.groups, padding=2)
        return weight.transpose(0, 1)
        """
        weight = getattr(self, 'weight'+str(0)).transpose(0, 1)
        for i in range(self.layers - 1):
            weight = self.unfold(weight)
            weight2 = getattr(self, 'weight'+str(i+1))

            weight = torch.einsum('akl,bk->abl', weight, weight2.view(weight2.size(0), -1))
            k = i * 2 + 5
            weight = weight.view(weight.size(0), weight.size(1), k, k)
        
        return weight.transpose(0, 1)
        """

    def forward(self, inputs):
        if hasattr(self, 'or_large_reparam'):
            return self.nonlinear(self.or_large_reparam(inputs))
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or_large_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or_large_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or_large_reparam.weight.data = kernel
        self.or_large_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for i in range(self.layers):
            self.__delattr__('weight' + str(i))
        self.__delattr__('bn')


class OREPA_LargeConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deploy=False, nonlinear=None):
        super(OREPA_LargeConv, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 3
        self.stride = stride
        self.padding = padding
        self.layers = int((kernel_size - 1) / 2)
        self.groups = groups
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        internal_channels = out_channels
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.or_large_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            for i in range(self.layers):
                if i == 0:
                    self.__setattr__('weight' + str(i), OREPA(in_channels, internal_channels, kernel_size=3, stride=1, padding=1, groups=groups, weight_only=True))
                elif i == self.layers - 1:
                    self.__setattr__('weight' + str(i), OREPA(internal_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, weight_only=True))
                else:
                    self.__setattr__('weight' + str(i), OREPA(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, weight_only=True))
            self.bn = nn.BatchNorm2d(out_channels)

    def weight_gen(self):
        weight = getattr(self, 'weight' + str(0)).weight_gen().transpose(0, 1)
        for i in range(self.layers - 1):
            weight2 = getattr(self, 'weight' + str(i + 1)).weight_gen()
            weight = F.conv2d(weight, weight2, groups=self.groups, padding=2)
        return weight.transpose(0, 1)
        """
        weight = getattr(self, 'weight'+str(0))(inputs=None).transpose(0, 1)
        for i in range(self.layers - 1):
            weight = self.unfold(weight)
            weight2 = getattr(self, 'weight'+str(i+1))(inputs=None)

            weight = torch.einsum('akl,bk->abl', weight, weight2.view(weight2.size(0), -1))
            k = i * 2 + 5
            weight = weight.view(weight.size(0), weight.size(1), k, k)
        
        return weight.transpose(0, 1)
        """

    def forward(self, inputs):
        if hasattr(self, 'or_large_reparam'):
            return self.nonlinear(self.or_large_reparam(inputs))
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or_large_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or_large_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or_large_reparam.weight.data = kernel
        self.or_large_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for i in range(self.layers):
            self.__delattr__('weight' + str(i))
        self.__delattr__('bn')


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class OREPA_3x3_RepVGG(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepVGG, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.branch_counter = 0
        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1
        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
        else:
            raise NotImplementedError
        self.branch_counter += 1
        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)
        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1
        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels * expand_ratio, 1, 1))
        init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1
        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1
        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.fre_init()
        init.constant_(self.vector[0, :], 0.25)
        init.constant_(self.vector[1, :], 0.25)
        init.constant_(self.vector[2, :], 0.0)
        init.constant_(self.vector[3, :], 0.5)
        init.constant_(self.vector[4, :], 0.5)

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):
        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])
        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])
        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2
        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)
        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])
        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])
        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv
        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))


class RepVGGBlock_OREPA(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, nonlinear=nn.ReLU()):
        super(RepVGGBlock_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = OREPA_3x3_RepVGG(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups, dilation=1)
            None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3
        return self.nonlinearity(self.se(out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepVGG):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels, kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride, padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


DEPLOY_FLAG = False


CONV_BN_IMPL = 'base'


def choose_blk(kernel_size):
    if CONV_BN_IMPL == 'OREPA':
        if kernel_size == 1:
            blk_type = OREPA_1x1
        elif kernel_size >= 7:
            blk_type = OREPA_LargeConv
        else:
            blk_type = OREPA
    elif CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif CONV_BN_IMPL == 'DBB':
        blk_type = DBB
    elif CONV_BN_IMPL == 'RepVGG':
        blk_type = RepVGGBlock
    elif CONV_BN_IMPL == 'OREPA_VGG':
        blk_type = RepVGGBlock_OREPA
    else:
        raise NotImplementedError
    return blk_type


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(conv_bn_relu(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, assign_type=ConvBN)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv_bn(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, cardinality=32, width_per_group=4):
        super(Bottleneck, self).__init__()
        D = cardinality * int(planes * width_per_group / 64)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_planes, self.expansion * planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_planes, D, kernel_size=1)
        self.conv2 = conv_bn_relu(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = conv_bn(D, self.expansion * planes, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1):
        super(ResNet, self).__init__()
        self.in_planes = int(64 * width_multiplier)
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * block.expansion * width_multiplier), num_classes)
        output_channels = int(512 * block.expansion * width_multiplier)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            if block is Bottleneck:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride))
            else:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1, cardinality=32, width_per_group=4):
        super(ResNeXt, self).__init__()
        self.in_planes = int(64 * width_multiplier)
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1, cardinality=cardinality, width_per_group=width_per_group)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * block.expansion * width_multiplier), num_classes)
        output_channels = int(512 * block.expansion * width_multiplier)

    def _make_stage(self, block, planes, num_blocks, stride, cardinality, width_per_group):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, cardinality=cardinality, width_per_group=width_per_group))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNAndPadLayer,
     lambda: ([], {'pad_pixels': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityBasedConv1x1,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OREPA_1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEBlock,
     lambda: ([], {'input_channels': 4, 'internal_neurons': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_JUGGHM_OREPA_CVPR2022(_paritybench_base):
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

