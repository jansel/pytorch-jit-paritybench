import sys
_module = sys.modules[__name__]
del sys
DeepRFT_MIMO = _module
data_RGB = _module
dataset_RGB = _module
doconv_pytorch = _module
evaluate_RealBlur = _module
get_parameter_number = _module
layers = _module
losses = _module
setup = _module
warmup_scheduler = _module
run = _module
scheduler = _module
test = _module
test_speed = _module
train = _module
train_wo_warmup = _module
utils = _module
dataset_utils = _module
dir_utils = _module
image_utils = _module
model_utils = _module

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


from torch.utils.data import Dataset


import torch


import torchvision.transforms.functional as TF


import random


import math


from torch.nn import functional as F


from torch._jit_internal import Optional


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch import nn


from torch.nn import init


import torch.nn as nn


import torch.nn.functional as F


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.sgd import SGD


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import DataLoader


import time


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


from collections import OrderedDict


class simam_module(torch.nn.Module):

    def __init__(self, e_lambda=0.0001):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels', 'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d, self).__init__()
        kernel_size = kernel_size, kernel_size
        stride = stride, stride
        padding = padding, padding
        dilation = dilation, dilation
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)
            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:
                self.D_diag = Parameter(D_diag, requires_grad=False)
        if simam:
            self.simam_block = simam_module()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode), weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = self.out_channels, self.in_channels // self.groups, M, N
        if M * N > 1:
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        if self.simam:
            DoW_h1, DoW_h2 = torch.chunk(DoW, 2, dim=2)
            DoW = torch.cat([self.simam_block(DoW_h1), DoW_h2], dim=2)
        return self._conv_forward(input, DoW)


class BasicConv_do(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False, relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DOConv2d_eval(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels', 'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d_eval, self).__init__()
        kernel_size = kernel_size, kernel_size
        stride = stride, stride
        padding = padding, padding
        dilation = dilation, dilation
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.register_parameter('bias', None)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode), weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.W)


class BasicConv_do_eval(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False, relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do_eval, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(DOConv2d_eval(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock_do_fft_bench(nn.Module):

    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True), BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False))
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class ResBlock_do_fft_bench_eval(nn.Module):

    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_eval, self).__init__()
        self.main = nn.Sequential(BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True), BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False))
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class DeepRFT_Small(nn.Module):

    def __init__(self, num_res=4, inference=False):
        super(DeepRFT_Small, self).__init__()
        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval
        base_channel = 32
        self.Encoder = nn.ModuleList([EBlock(base_channel, num_res, ResBlock=ResBlock), EBlock(base_channel * 2, num_res, ResBlock=ResBlock), EBlock(base_channel * 4, num_res, ResBlock=ResBlock)])
        self.feat_extract = nn.ModuleList([BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])
        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, num_res, ResBlock=ResBlock), DBlock(base_channel * 2, num_res, ResBlock=ResBlock), DBlock(base_channel, num_res, ResBlock=ResBlock)])
        self.Convs = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)])
        self.ConvsOut = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1), BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)])
        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv), AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)])
        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False, channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock_fft_bench(nn.Module):

    def __init__(self, n_feat, norm='backward'):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True), BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=True), BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False))
        self.dim = n_feat
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class DeepRFT_flops(nn.Module):

    def __init__(self, num_res=8, inference=True):
        super(DeepRFT_flops, self).__init__()
        self.inference = inference
        ResBlock = ResBlock_fft_bench
        base_channel = 32
        self.Encoder = nn.ModuleList([EBlock(base_channel, num_res, ResBlock=ResBlock), EBlock(base_channel * 2, num_res, ResBlock=ResBlock), EBlock(base_channel * 4, num_res, ResBlock=ResBlock)])
        self.feat_extract = nn.ModuleList([BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])
        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, num_res, ResBlock=ResBlock), DBlock(base_channel * 2, num_res, ResBlock=ResBlock), DBlock(base_channel, num_res, ResBlock=ResBlock)])
        self.Convs = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)])
        self.ConvsOut = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1), BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)])
        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv), AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)])
        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x


class DeepRFT(nn.Module):

    def __init__(self, num_res=8, inference=False):
        super(DeepRFT, self).__init__()
        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval
        base_channel = 32
        self.Encoder = nn.ModuleList([EBlock(base_channel, num_res, ResBlock=ResBlock), EBlock(base_channel * 2, num_res, ResBlock=ResBlock), EBlock(base_channel * 4, num_res, ResBlock=ResBlock)])
        self.feat_extract = nn.ModuleList([BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])
        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, num_res, ResBlock=ResBlock), DBlock(base_channel * 2, num_res, ResBlock=ResBlock), DBlock(base_channel, num_res, ResBlock=ResBlock)])
        self.Convs = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)])
        self.ConvsOut = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1), BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)])
        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv), AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)])
        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x


class DeepRFTPLUS(nn.Module):

    def __init__(self, num_res=20, inference=False):
        super(DeepRFTPLUS, self).__init__()
        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval
        base_channel = 32
        self.Encoder = nn.ModuleList([EBlock(base_channel, num_res, ResBlock=ResBlock), EBlock(base_channel * 2, num_res, ResBlock=ResBlock), EBlock(base_channel * 4, num_res, ResBlock=ResBlock)])
        self.feat_extract = nn.ModuleList([BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])
        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, num_res, ResBlock=ResBlock), DBlock(base_channel * 2, num_res, ResBlock=ResBlock), DBlock(base_channel, num_res, ResBlock=ResBlock)])
        self.Convs = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)])
        self.ConvsOut = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1), BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)])
        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv), AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)])
        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x


class ResBlock(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False), BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False))

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False))

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_eval(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do_eval, self).__init__()
        self.main = nn.Sequential(BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False))

    def forward(self, x):
        return self.main(x) + x


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=0.001):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class fftLoss(nn.Module):

    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv_do,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicConv_do_eval,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DOConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DOConv2d_eval,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 3, 3]), torch.rand([4, 3, 3, 3])], {}),
     True),
    (ResBlock,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock_do,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock_do_eval,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock_do_fft_bench,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock_do_fft_bench_eval,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock_fft_bench,
     lambda: ([], {'n_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (fftLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (simam_module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_INVOKERer_DeepRFT(_paritybench_base):
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

