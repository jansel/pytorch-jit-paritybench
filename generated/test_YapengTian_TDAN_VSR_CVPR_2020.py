import sys
_module = sys.modules[__name__]
del sys
SR_datasets = _module
_ext = _module
deform_conv = _module
build = _module
dataset = _module
eval = _module
functions = _module
deform_conv = _module
loss = _module
model = _module
modules = _module
deform_conv = _module
pytorch_ssim = _module
solver = _module
train = _module

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


import torchvision


import torchvision.transforms as T


import scipy.misc


import scipy.ndimage


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import matplotlib.pyplot as plt


import random


import scipy


from torch.autograd import Variable


import time


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import math


from torch.nn.modules.module import Module


from math import exp


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


class MSE_and_SSIM_loss(nn.Module):

    def __init__(self, alpha=0.9):
        super(MSE_and_SSIM_loss, self).__init__()
        self.MSE = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.alpha = alpha

    def forward(self, img1, img2):
        loss = self.alpha * self.MSE(img1, img2) + (1 - self.alpha) * (1 - self.SSIM(img1, img2))
        return loss


class TDAN_L(nn.Module):

    def __init__(self, nets):
        super(TDAN_L, self).__init__()
        self.name = 'TDAN_L'
        self.align_net, self.rec_net = nets
        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):
        lrs = self.align_net(x)
        y = self.rec_net(lrs)
        return y, lrs


class TDAN_F(nn.Module):

    def __init__(self, nets):
        super(TDAN_F, self).__init__()
        self.name = 'TDAN_F'
        self.align_net, self.rec_net = nets
        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):
        lrs, feat = self.align_net(x)
        y = self.rec_net(lrs, feat)
        return y, lrs


class ConvOffset2dFunction(Function):

    def __init__(self, stride, padding, dilation, deformable_groups=1):
        super(ConvOffset2dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)
        output = input.new(*self._output_size(input, weight))
        self.bufs_ = [input.new(), input.new()]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(input, torch.FloatTensor):
                raise NotImplementedError
            deform_conv.deform_conv_forward_cuda(input, weight, offset, output, self.bufs_[0], self.bufs_[1], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(grad_output, torch.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                deform_conv.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, self.bufs_[0], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups)
            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, self.bufs_[0], self.bufs_[1], weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], self.deformable_groups, 1)
        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


def conv_offset2d(input, offset, weight, stride=1, padding=0, dilation=1, deform_groups=1):
    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    f = ConvOffset2dFunction(_pair(stride), _pair(padding), _pair(dilation), deform_groups)
    return f(input, offset, weight)


class ConvOffset2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, num_deformable_groups=1):
        super(ConvOffset2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return conv_offset2d(input, offset, self.weight, self.stride, self.padding, self.dilation, self.num_deformable_groups)


class Res_Block(nn.Module):

    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        modules = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feat))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*modules)


def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channelss, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class TDAN_VSR(nn.Module):

    def __init__(self):
        super(TDAN_VSR, self).__init__()
        self.name = 'TDAN'
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)
        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True), nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        self.recon_layer = self.make_layer(Res_Block, 10)
        upscaling = [Upsampler(default_conv, 4, 64, act=False), nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, (center), :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, (i), :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, (center), :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        lrs = self.align(out, x_center)
        y = lrs.view(batch_size, -1, w, h)
        fea = self.fea_ex(y)
        out = self.recon_layer(fea)
        out = self.up(out)
        return out, lrs


class TDAN(nn.Module):

    def __init__(self, nets):
        super(TDAN, self).__init__()
        self.name = 'TDAN'
        self.align_net, self.rec_net = nets

    def forward(self, x):
        lrs = self.align_net(x)
        y = self.rec_net(lrs)
        return y, lrs


class align_net_w_feat(nn.Module):

    def __init__(self):
        super(align_net_w_feat, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        feats = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, (center), :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                feats.append(ref.unsqueeze(1))
                continue
            supp = x[:, (i), :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
            feats.append(fea.unsqueeze(1))
        y = torch.cat(y, dim=1)
        feats = torch.cat(feats, dim=1)
        return y, feats

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, (center), :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        lrs, feats = self.align(out, x_center)
        return lrs, feats


class align_net(nn.Module):

    def __init__(self):
        super(align_net, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, (center), :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, (i), :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, (center), :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        lrs = self.align(out, x_center)
        return lrs


class Res_Block_s(nn.Module):

    def __init__(self, scale=1.0):
        super(Res_Block_s, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.scale = scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res.mul(self.scale)


class SR_Rec(nn.Module):

    def __init__(self, nb_block=10, scale=1.0):
        super(SR_Rec, self).__init__()
        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True), nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        upscaling = [Upsampler(default_conv, 4, 64, act=False), nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        y = y.view(batch_size, -1, w, h)
        fea = self.fea_ex(y)
        out = self.recon_layer(fea)
        out = self.up(out)
        return out


class VSR_Rec(nn.Module):

    def __init__(self, nb_block=10, scale=1.0):
        super(VSR_Rec, self).__init__()
        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True), nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        self.fuse = nn.Conv2d(6 * 64, 64, 3, padding=1, bias=True)
        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        upscaling = [Upsampler(default_conv, 4, 64, act=False), nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y, feats):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        y = y.view(batch_size, -1, w, h)
        feat = self.fea_ex(y)
        feat = torch.cat((feats, feat.unsqueeze(1)), 1).view(batch_size, -1, w, h)
        feat = self.fuse(feat)
        out = self.recon_layer(feat)
        out = self.up(out)
        return out


class Conv_ReLU_Block(nn.Module):

    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv_ReLU_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (L1_Charbonnier_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSE_and_SSIM_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Res_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Res_Block_s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_YapengTian_TDAN_VSR_CVPR_2020(_paritybench_base):
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

