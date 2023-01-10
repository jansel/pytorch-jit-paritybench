import sys
_module = sys.modules[__name__]
del sys
count_flops = _module
creat_PCA = _module
inference = _module
SRGAN_model = _module
SR_model = _module
models = _module
base_model = _module
blind_model = _module
lr_scheduler = _module
RRDBNet_arch = _module
SRResNet_arch = _module
modules = _module
dcls_arch = _module
discriminator_vgg_arch = _module
loss = _module
module_util = _module
networks = _module
options = _module
test = _module
train = _module
GT_dataset = _module
LQGT_dataset = _module
LQ_dataset = _module
data = _module
data_sampler = _module
util = _module
color2gray = _module
create_lmdb = _module
extract_subimgs_single = _module
generate_mod_LR_bic = _module
generate_mod_blur_LR_bic = _module
kernel_visual = _module
utils = _module
dcls_utils = _module
deg_utils = _module
file_utils = _module
img_utils = _module
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


import torch


import numpy as np


import logging


import time


from collections import OrderedDict


import torch.nn as nn


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import torchvision.utils as tvutils


import torch.nn.functional as F


import torch.nn.init as init


import math


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


import functools


import torchvision


import random


import copy


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.utils.data as data


import torch.utils.data


from torch.utils.data.sampler import Sampler


from scipy.io import loadmat


from torch.autograd import Variable


from torchvision.utils import make_grid


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


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlock_noBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out.mul(self.res_scale)


class MSRResNet(nn.Module):
    """ modified SRResNet"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class CALayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DPCAB(nn.Module):

    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()
        self.body1 = nn.Sequential(nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2))
        self.body2 = nn.Sequential(nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2))
        self.CA_body1 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf1 + nf2, nf1, ksize1, 1, ksize1 // 2), CALayer(nf1, reduction))
        self.CA_body2 = CALayer(nf2, reduction)

    def forward(self, x):
        f1 = self.body1(x[0])
        f2 = self.body2(x[1])
        ca_f1 = self.CA_body1(torch.cat([f1, f2], dim=1))
        ca_f2 = self.CA_body2(f2)
        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        return x


class DPCAG(nn.Module):

    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()
        self.body = nn.Sequential(*[DPCAB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


def convert_psf2otf(ker, size):
    psf = torch.zeros(size)
    centre = ker.shape[2] // 2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, centre - 1:, centre - 1:]
    psf[:, :, :centre, -(centre - 1):] = ker[:, :, centre - 1:, :centre - 1]
    psf[:, :, -(centre - 1):, :centre] = ker[:, :, :centre - 1, centre - 1:]
    psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :centre - 1, :centre - 1]
    otf = torch.rfft(psf, 3, onesided=False)
    return otf


def deconv(inv_ker_f, fft_input_blur):
    deblur_f = torch.zeros_like(inv_ker_f)
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.irfft(deblur_f, 3, onesided=False)
    return deblur


def inv_fft_kernel_est(ker_f, ker_p):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + ker_p[:, :, :, :, 0] * ker_p[:, :, :, :, 0] + ker_p[:, :, :, :, 1] * ker_p[:, :, :, :, 1]
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return inv_ker_f


def get_uperleft_denominator(img, kernel, grad_kernel):
    ker_f = convert_psf2otf(kernel, img.size())
    ker_p = convert_psf2otf(grad_kernel, img.size())
    denominator = inv_fft_kernel_est(ker_f, ker_p)
    numerator = torch.rfft(img, 3, onesided=False)
    deblur = deconv(denominator, numerator)
    return deblur


class CLS(nn.Module):

    def __init__(self, nf, reduction=4):
        super().__init__()
        self.reduce_feature = nn.Conv2d(nf, nf // reduction, 1, 1, 0)
        self.grad_filter = nn.Sequential(nn.Conv2d(nf // reduction, nf // reduction, 3), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf // reduction, nf // reduction, 3), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf // reduction, nf // reduction, 3), nn.AdaptiveAvgPool2d((3, 3)), nn.Conv2d(nf // reduction, nf // reduction, 1))
        self.expand_feature = nn.Conv2d(nf // reduction, nf, 1, 1, 0)

    def forward(self, x, kernel):
        cls_feats = self.reduce_feature(x)
        kernel_P = torch.exp(self.grad_filter(cls_feats))
        kernel_P = kernel_P - kernel_P.mean(dim=(2, 3), keepdim=True)
        clear_features = torch.zeros(cls_feats.size())
        ks = kernel.shape[-1]
        dim = ks, ks, ks, ks
        feature_pad = F.pad(cls_feats, dim, 'replicate')
        for i in range(feature_pad.shape[1]):
            feature_ch = feature_pad[:, i:i + 1, :, :]
            clear_feature_ch = get_uperleft_denominator(feature_ch, kernel, kernel_P[:, i:i + 1, :, :])
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]
        x = self.expand_feature(clear_features)
        return x


class Estimator(nn.Module):

    def __init__(self, in_nc=1, nf=64, para_len=10, num_blocks=3, kernel_size=4, filter_structures=[]):
        super(Estimator, self).__init__()
        self.filter_structures = filter_structures
        self.ksize = kernel_size
        self.G_chan = 16
        self.in_nc = in_nc
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.head = nn.Sequential(nn.Conv2d(in_nc, nf, 7, 1, 3))
        self.body = nn.Sequential(make_layer(basic_block, num_blocks))
        self.tail = nn.Sequential(nn.Conv2d(nf, nf, 3), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(nf, nf, 3), nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(nf, para_len, 1), nn.Flatten())
        self.dec = nn.ModuleList()
        for i, f_size in enumerate(self.filter_structures):
            if i == 0:
                in_chan = in_nc
            elif i == len(self.filter_structures) - 1:
                in_chan = in_nc
            else:
                in_chan = self.G_chan
            self.dec.append(nn.Linear(para_len, self.G_chan * in_chan * f_size ** 2))
        self.apply(initialize_weights)

    def calc_curr_k(self, kernels, batch):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.ones([1, batch * self.in_nc]).unsqueeze(-1).unsqueeze(-1)
        for ind, w in enumerate(kernels):
            curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w, groups=batch)
        curr_k = curr_k.reshape(batch, self.in_nc, self.ksize, self.ksize).flip([2, 3])
        return curr_k

    def forward(self, LR):
        batch, channel = LR.shape[0:2]
        f1 = self.head(LR)
        f = self.body(f1) + f1
        latent_kernel = self.tail(f)
        kernels = [self.dec[0](latent_kernel).reshape(batch * self.G_chan, channel, self.filter_structures[0], self.filter_structures[0])]
        for i in range(1, len(self.filter_structures) - 1):
            kernels.append(self.dec[i](latent_kernel).reshape(batch * self.G_chan, self.G_chan, self.filter_structures[i], self.filter_structures[i]))
        kernels.append(self.dec[-1](latent_kernel).reshape(batch * channel, self.G_chan, self.filter_structures[-1], self.filter_structures[-1]))
        K = self.calc_curr_k(kernels, batch).mean(dim=1, keepdim=True)
        K = K / torch.sum(K, dim=(2, 3), keepdim=True)
        return K


class Restorer(nn.Module):

    def __init__(self, in_nc=1, nf=64, nb=8, ng=1, scale=4, input_para=10, reduction=4, min=0.0, max=1.0):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        out_nc = in_nc
        nf2 = nf // reduction
        self.conv_first = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_block = make_layer(basic_block, 3)
        self.head1 = nn.Conv2d(nf, nf2, 3, 1, 1)
        self.head2 = CLS(nf, reduction=reduction)
        body = [DPCAG(nf, nf2, 3, 3, nb) for _ in range(ng)]
        self.body = nn.Sequential(*body)
        self.fusion = nn.Conv2d(nf + nf2, nf, 3, 1, 1)
        if scale == 4:
            self.upscale = nn.Sequential(nn.Conv2d(nf, nf * scale, 3, 1, 1, bias=True), nn.PixelShuffle(scale // 2), nn.Conv2d(nf, nf * scale, 3, 1, 1, bias=True), nn.PixelShuffle(scale // 2), nn.Conv2d(nf, out_nc, 3, 1, 1))
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)
        else:
            self.upscale = nn.Sequential(nn.Conv2d(nf, nf * scale ** 2, 3, 1, 1, bias=True), nn.PixelShuffle(scale), nn.Conv2d(nf, out_nc, 3, 1, 1))

    def forward(self, input, kernel):
        f = self.conv_first(input)
        feature = self.feature_block(f)
        f1 = self.head1(feature)
        f2 = self.head2(feature, kernel)
        inputs = [f2, f1]
        f2, f1 = self.body(inputs)
        f = self.fusion(torch.cat([f1, f2], dim=1)) + f
        out = self.upscale(f)
        return torch.clamp(out, min=self.min, max=self.max)


class DCLS(nn.Module):

    def __init__(self, nf=64, nb=16, ng=5, in_nc=3, reduction=4, upscale=4, input_para=128, kernel_size=21, pca_matrix_path=None):
        super(DCLS, self).__init__()
        self.ksize = kernel_size
        self.scale = upscale
        if kernel_size == 21:
            filter_structures = [11, 7, 5, 1]
        elif kernel_size == 11:
            filter_structures = [7, 3, 3, 1]
        elif kernel_size == 31:
            filter_structures = [11, 9, 7, 5, 3]
        else:
            None
        self.Restorer = Restorer(nf=nf, in_nc=in_nc, nb=nb, ng=ng, scale=self.scale, input_para=input_para, reduction=reduction)
        self.Estimator = Estimator(kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, nf=nf, filter_structures=filter_structures)

    def forward(self, lr):
        kernel = self.Estimator(lr)
        sr = self.Restorer(lr, kernel.detach())
        return sr, kernel


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


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


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


def convert_otf2psf(otf, size):
    ker = torch.zeros(size)
    psf = torch.irfft(otf, 3, onesided=False)
    ksize = size[-1]
    centre = ksize // 2 + 1
    ker[:, :, centre - 1:, centre - 1:] = psf[:, :, :centre, :centre]
    ker[:, :, centre - 1:, :centre - 1] = psf[:, :, :centre, -(centre - 1):]
    ker[:, :, :centre - 1, centre - 1:] = psf[:, :, -(centre - 1):, :centre]
    ker[:, :, :centre - 1, :centre - 1] = psf[:, :, -(centre - 1):, -(centre - 1):]
    return ker


def normkernel_to_downkernel(rescaled_blur_hr, rescaled_hr, ksize, eps=1e-10):
    blur_img = torch.rfft(rescaled_blur_hr, 3, onesided=False)
    img = torch.rfft(rescaled_hr, 3, onesided=False)
    denominator = img[:, :, :, :, 0] * img[:, :, :, :, 0] + img[:, :, :, :, 1] * img[:, :, :, :, 1] + eps
    inv_denominator = torch.zeros_like(img)
    inv_denominator[:, :, :, :, 0] = img[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -img[:, :, :, :, 1] / denominator
    kernel = torch.zeros_like(blur_img)
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 0] - inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 1] + inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 0]
    ker = convert_otf2psf(kernel, ksize)
    return ker


def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    pc = k.shape[-1] // 2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k


class CorrectionLoss(nn.Module):

    def __init__(self, scale=4.0, eps=1e-06):
        super(CorrectionLoss, self).__init__()
        self.scale = scale
        self.eps = eps
        self.cri_pix = nn.L1Loss()

    def forward(self, k_pred, lr_blured, lr):
        ks = []
        mask = torch.ones_like(k_pred)
        for c in range(lr_blured.shape[1]):
            k_correct = normkernel_to_downkernel(lr_blured[:, c:c + 1, ...], lr[:, c:c + 1, ...], k_pred.size(), self.eps)
            ks.append(k_correct.clone())
            mask *= k_correct
        ks = torch.cat(ks, dim=1)
        k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask > 0)
        k_correct = zeroize_negligible_val(k_correct, n=40)
        return self.cri_pix(k_pred, k_correct), k_correct


class ResidualBlock_BN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlock_BN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        return identity + out.mul(self.res_scale)


class PCAEncoder(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)).view((B, -1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CALayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSRResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDBNet,
     lambda: ([], {'in_nc': 4, 'out_nc': 4, 'nf': 4, 'nb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock_BN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualBlock_noBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (VGGFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_megvii_research_DCLS_SR(_paritybench_base):
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

