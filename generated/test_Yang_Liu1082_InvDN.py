import sys
_module = sys.modules[__name__]
del sys
Crop_DIV2K = _module
Crop_SIDD = _module
Download_full_SIDD = _module
codes = _module
LQGTRN_dataset = _module
LQGTSN_dataset = _module
data = _module
data_sampler = _module
util = _module
InvDN_model = _module
models = _module
base_model = _module
lr_scheduler = _module
Inv_arch = _module
MSKResnet = _module
ResAttentionBlock = _module
Subnet_constructor = _module
modules = _module
denoised_LR = _module
discriminator_vgg_arch = _module
loss = _module
module_util = _module
networks = _module
options = _module
test_Real = _module
train = _module
utils = _module
util = _module
extract_data = _module
calculate_PSNR_SSIM = _module

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


import torch.utils.data as data


import logging


import torch.utils.data


import math


from torch.utils.data.sampler import Sampler


import torch.distributed as dist


from collections import OrderedDict


import torch.nn as nn


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn.functional as F


from torch import nn


import time


import torchvision


import torch.nn.init as init


import scipy


import scipy.io as sio


import torch.multiprocessing as mp


from torchvision.utils import make_grid


class InvBlockExp(nn.Module):

    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.0):
        super(InvBlockExp, self).__init__()
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)
        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)
        return jac / x.shape[0]


class HaarDownsampling(nn.Module):

    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.0)
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.0)
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvNet(nn.Module):

    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvNet, self).__init__()
        operations = []
        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        if cal_jacobian:
            return out, jacobian
        else:
            return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, init='xavier'):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, init=None, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):

    def __init__(self, in_channels, out_channels, init='xavier', ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.Sigmoid())

    def forward(self, x):
        out = self.body(x)
        return out


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = BasicBlock(channel, channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel, 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


class ResidualUnit(nn.Module):

    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block1 = ResidualBlock(channels)
        self.block2 = ResidualBlock(channels)
        self.block3 = ResidualBlock(channels)
        self.block4 = ResidualBlock(channels)
        self.conv = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.ca = CALayer(channels)

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        mid = torch.cat((x, res), dim=1)
        out = self.conv(mid)
        out = self.ca(out)
        return out


class ResidualModule(nn.Module):

    def __init__(self, channels):
        super(ResidualModule, self).__init__()
        self.block1 = ResidualUnit(channels)
        self.block2 = ResidualUnit(channels)
        self.block3 = ResidualUnit(channels)
        self.block4 = ResidualUnit(channels)

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        return x + res


class MSKResnet(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(MSKResnet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=channels_in, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False), nn.ReLU(inplace=True))
        self.block2 = ResidualModule(features)
        self.block3 = nn.Conv2d(in_channels=features, out_channels=channels_out, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        out = self.block3(block2)
        return out


class Merge_Run(nn.Module):

    def __init__(self, in_channels, out_channels, init='xavier', ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run, self).__init__()
        self.body1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body3 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class Merge_Run_dual(nn.Module):

    def __init__(self, in_channels, out_channels, init='xavier', ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run_dual, self).__init__()
        self.body1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body3 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class EResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, init='xavier', group=1):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class _UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, init='xavier', group=1):
        super(_UpsampleBlock, self).__init__()
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()
        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class RABlock(nn.Module):

    def __init__(self, in_channels, out_channels, group=1):
        super(RABlock, self).__init__()
        self.r1 = Merge_Run_dual(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels, out_channels)
        self.r3 = EResidualBlock(in_channels, out_channels)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        out = self.ca(r3)
        return out


class ResBlock(nn.Module):

    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature + channel_in, channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


class DenseBlock(nn.Module):

    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class Discriminator_VGG_128(nn.Module):

    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer + 1])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class ReconstructionLoss(nn.Module):

    def __init__(self, losstype='l2', eps=0.001):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-06
            return torch.mean(torch.sum(-torch.log(1 - diff.abs() + eps), (1, 2, 3)))
        else:
            None
            return 0


class Gradient_Loss(nn.Module):

    def __init__(self, losstype='l2'):
        super(Gradient_Loss, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = torch.stack((a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        self.conv1 = conv1
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = torch.stack((b, b, b))
        conv2.weight = nn.Parameter(b, requires_grad=False)
        self.conv2 = conv2
        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        l_h = self.Loss_criterion(x1, y1)
        l_v = self.Loss_criterion(x2, y2)
        return l_h + l_v


class SSIM_Loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class GANLoss(nn.Module):

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockSig,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Gradient_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (HaarDownsampling,
     lambda: ([], {'channel_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Merge_Run,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Merge_Run_dual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReconstructionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock_noBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SSIM_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleBlock,
     lambda: ([], {'n_channels': 4, 'scale': 1.0, 'multi_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (VGGFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_UpsampleBlock,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Yang_Liu1082_InvDN(_paritybench_base):
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

