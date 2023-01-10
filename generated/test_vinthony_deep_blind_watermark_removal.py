import sys
_module = sys.modules[__name__]
del sys
main = _module
options = _module
scripts = _module
BIH = _module
COCO = _module
datasets = _module
BasicMachine = _module
S2AM = _module
VX = _module
machines = _module
models = _module
backbone_unet = _module
blocks = _module
discriminator = _module
rasc = _module
sa_resunet = _module
unet = _module
vgg = _module
vmu = _module
utils = _module
evaluation = _module
imutils = _module
logger = _module
losses = _module
misc = _module
model_init = _module
osutils = _module
parallel = _module
transforms = _module
test = _module

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


import time


import numpy as np


import random


import math


import matplotlib.pyplot as plt


from collections import namedtuple


import torch.utils.data as data


import torchvision.transforms as transforms


import torch.nn as nn


import torch.backends.cudnn as cudnn


import torch.optim


from math import log10


from torch.autograd import Variable


import torchvision


import torch.nn.functional as F


import functools


import numbers


from torch import nn


from torch import cuda


from torch import Tensor


from torch.nn import Parameter


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.nn import init


from torchvision import models


from random import randint


import scipy.misc


import scipy.io


from torch.autograd import Function


import torch.cuda.comm as comm


from torch.nn.parallel import DistributedDataParallel


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


class MeanShift(nn.Conv2d):

    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if i + 1 in indices:
                out.append(X)
        return out


class VGGLossA(nn.Module):

    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLossA, self).__init__()
        if vgg is None:
            self.vgg = Vgg19()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True)
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def l1_relative(reconstructed, real, mask):
    batch = real.size(0)
    area = torch.sum(mask.view(batch, -1), dim=1)
    reconstructed = reconstructed * mask
    real = real * mask
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def resize_to_match(fm, to):
    return F.interpolate(fm, to.size()[-2:], mode='bilinear', align_corners=False)


class VGGLossX(nn.Module):

    def __init__(self, normalize=True, mask=False, relative=False):
        super(VGGLossX, self).__init__()
        self.vgg = VGG16FeatureExtractor()
        self.criterion = nn.L1Loss() if not relative else l1_relative
        self.use_mask = mask
        self.relative = relative
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True)
        else:
            self.normalize = None

    def forward(self, x, y, Xmask=None):
        if not self.use_mask:
            mask = torch.ones_like(x)[:, 0:1, :, :]
        else:
            mask = Xmask
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(3):
            if self.relative:
                loss += self.criterion(x_vgg[i], y_vgg[i].detach(), resize_to_match(mask, x_vgg[i]))
            else:
                loss += self.criterion(resize_to_match(mask, x_vgg[i]) * x_vgg[i], resize_to_match(mask, y_vgg[i]) * y_vgg[i].detach())
        return loss


def VGGLoss(losstype):
    if losstype == 'vgg':
        return VGGLossA()
    elif losstype == 'vggx':
        return VGGLossX(mask=False)
    elif losstype == 'mvggx':
        return VGGLossX(mask=True)
    elif losstype == 'rvggx':
        return VGGLossX(mask=True, relative=True)
    else:
        raise Exception('error in %s' % losstype)


class WeightedBCE(nn.Module):

    def __init__(self):
        super(WeightedBCE, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1.0 - gt) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)
        return loss


def is_dic(x):
    return type(x) == type([])


class Losses(nn.Module):

    def __init__(self, argx, device):
        super(Losses, self).__init__()
        self.args = argx
        if self.args.loss_type == 'l1bl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), nn.BCELoss(), nn.MSELoss()
        elif self.args.loss_type == 'l1wbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), WeightedBCE(), nn.MSELoss()
        elif self.args.loss_type == 'l2wbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), WeightedBCE(), nn.MSELoss()
        elif self.args.loss_type == 'l2xbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.MSELoss()
        else:
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCELoss(), nn.MSELoss()
        if self.args.style_loss > 0:
            self.vggloss = VGGLoss(self.args.sltype)
        if self.args.ssim_loss > 0:
            self.ssimloss = pytorch_ssim.SSIM()
        self.outputLoss = self.outputLoss
        self.attLoss = self.attLoss
        self.wrloss = self.wrloss

    def forward(self, imgx, target, attx, mask, wmx, wm):
        pixel_loss, att_loss, wm_loss, vgg_loss, ssim_loss = 0, 0, 0, 0, 0
        if is_dic(imgx):
            if self.args.masked:
                pixel_loss = self.outputLoss(imgx[0], target) + sum([self.outputLoss(im, resize_to_match(mask, im) * resize_to_match(target, im)) for im in imgx[1:]])
            else:
                pixel_loss = sum([self.outputLoss(im, resize_to_match(target, im)) for im in imgx])
            if self.args.style_loss > 0:
                vgg_loss = sum([self.vggloss(im, resize_to_match(target, im), resize_to_match(mask, im)) for im in imgx])
            if self.args.ssim_loss > 0:
                ssim_loss = sum([(1 - self.ssimloss(im, resize_to_match(target, im))) for im in imgx])
        else:
            if self.args.masked:
                pixel_loss = self.outputLoss(imgx, mask * target)
            else:
                pixel_loss = self.outputLoss(imgx, target)
            if self.args.style_loss > 0:
                vgg_loss = self.vggloss(imgx, target, mask)
            if self.args.ssim_loss > 0:
                ssim_loss = 1 - self.ssimloss(imgx, target)
        if is_dic(attx):
            att_loss = sum([self.attLoss(at, resize_to_match(mask, at)) for at in attx])
        else:
            att_loss = self.attLoss(attx, mask)
        if is_dic(wmx):
            wm_loss = sum([self.wrloss(w, resize_to_match(wm, w)) for w in wmx])
        elif self.args.masked:
            wm_loss = self.wrloss(wmx, mask * wm)
        else:
            wm_loss = self.wrloss(wmx, wm)
        return pixel_loss, att_loss, wm_loss, vgg_loss, ssim_loss


class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""

    def __init__(self, channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel, channel * 2, 3, padding=1, bias=False)
        self.rbn1 = nn.BatchNorm2d(channel * 2)
        self.rconv2 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self, feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature))))))


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class ChannelPool(nn.Module):

    def __init__(self, types):
        super(ChannelPool, self).__init__()
        if types == 'avg':
            self.poolingx = nn.AdaptiveAvgPool1d(1)
        elif types == 'max':
            self.poolingx = nn.AdaptiveMaxPool1d(1)
        else:
            raise 'inner error'

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = self.poolingx(input)
        _, _, c = pooled.size()
        return pooled.view(n, c, w, h)


class SEBlock(nn.Module):
    """docstring for SEBlock"""

    def __init__(self, channel, reducation=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reducation), nn.ReLU(inplace=True), nn.Linear(channel // reducation, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y = self.fc(y1).view(b, c, 1, 1)
        return x * y


class GlobalAttentionModule(nn.Module):
    """docstring for GlobalAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel * 2, channel // reducation), nn.ReLU(inplace=True), nn.Linear(channel // reducation, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.cat([y1, y2], 1)).view(b, c, 1, 1)
        return x * y


class SpatialAttentionModule(nn.Module):
    """docstring for SpatialAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = ChannelPool('avg')
        self.max_pool = ChannelPool('max')
        self.fc = nn.Sequential(nn.Conv2d(2, reducation, 7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv2d(reducation, 1, 7, stride=1, padding=3), nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.fc(torch.cat([y1, y2], 1))
        yr = 1 - y
        return y, yr


class GlobalAttentionModuleJustSigmoid(nn.Module):
    """docstring for GlobalAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(GlobalAttentionModuleJustSigmoid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel * 2, channel // reducation), nn.ReLU(inplace=True), nn.Linear(channel // reducation, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.cat([y1, y2], 1)).view(b, c, 1, 1)
        return y


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPoolX(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPoolX()
        self.spatial = BasicBlock(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class SNCoXvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNCoXvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class SNDiscriminator(nn.Module):

    def __init__(self, channel=6):
        super(SNDiscriminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(SNCoXvWithActivation(channel, 2 * cnum, 4, 2, padding=get_pad(256, 5, 2)), SNCoXvWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 5, 2)), SNCoXvWithActivation(4 * cnum, 8 * cnum, 4, 2, padding=get_pad(64, 5, 2)), SNCoXvWithActivation(8 * cnum, 8 * cnum, 4, 2, padding=get_pad(32, 5, 2)), SNCoXvWithActivation(8 * cnum, 8 * cnum, 4, 2, padding=get_pad(16, 5, 2)))

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        x = self.discriminator_net(img_input)
        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*discriminator_block(in_channels * 2, 64, normalization=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(512, 1, 4, padding=1, bias=False))

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class CAWapper(nn.Module):
    """docstring for SENet"""

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(CAWapper, self).__init__()
        self.attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, use_cuda=True)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        mask = torch.round(F.avg_pool2d(mask, 2, stride=mw // w))
        result = self.attention(feature, mask)
        return result


class NLWapper(nn.Module):
    """docstring for SENet"""

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(NLWapper, self).__init__()
        self.attention = NONLocalBlock2D(channel)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        result = self.attention(feature)
        return result


class SENet(nn.Module):
    """docstring for SENet"""

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(SENet, self).__init__()
        self.attention = SEBlock(channel, 16)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        mask = torch.round(F.avg_pool2d(mask, 2, stride=mw // w))
        result = self.attention(feature)
        return result


class CBAMConnect(nn.Module):

    def __init__(self, channel):
        super(CBAMConnect, self).__init__()
        self.attention = CBAM(channel)

    def forward(self, feature, mask):
        results = self.attention(feature)
        return results


class RASC(nn.Module):

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(RASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel, 16)
        self.mixed_attention = GlobalAttentionModule(channel, 16)
        self.spliced_attention = GlobalAttentionModule(channel, 16)
        self.gaussianMask = GaussianSmoothing(1, 5, 1)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        if w != mw:
            mask = torch.round(F.avg_pool2d(mask, 2, stride=mw // w))
        reverse_mask = -1 * (mask - 1)
        mask = self.gaussianMask(F.pad(mask, (2, 2, 2, 2), mode='reflect'))
        reverse_mask = self.gaussianMask(F.pad(reverse_mask, (2, 2, 2, 2), mode='reflect'))
        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature)
        spliced = (self.connection(spliced_feature) + selected_feature) * mask
        return background + spliced


class UNO(nn.Module):

    def __init__(self, channel):
        super(UNO, self).__init__()

    def forward(self, feature, _m):
        return feature


class URASC(nn.Module):

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(URASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel, 16)
        self.mixed_attention = GlobalAttentionModule(channel, 16)
        self.spliced_attention = GlobalAttentionModule(channel, 16)
        self.mask_attention = SpatialAttentionModule(channel, 16)

    def forward(self, feature, m=None):
        _, _, w, _ = feature.size()
        mask, reverse_mask = self.mask_attention(feature)
        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature)
        spliced = (self.connection(spliced_feature) + selected_feature) * mask
        return background + spliced


class MaskedURASC(nn.Module):

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(MaskedURASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel, 16)
        self.mixed_attention = GlobalAttentionModule(channel, 16)
        self.spliced_attention = GlobalAttentionModule(channel, 16)
        self.mask_attention = SpatialAttentionModule(channel, 16)

    def forward(self, feature):
        _, _, w, _ = feature.size()
        mask, reverse_mask = self.mask_attention(feature)
        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature)
        spliced = (self.connection(spliced_feature) + selected_feature) * mask
        return background + spliced, mask


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))


class UpCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True, batch_norm=True, transpose=True, concat=True, use_att=False):
        super(UpCoXvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.use_att = use_att
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        if self.use_att:
            self.s2am = RASC(2 * out_channels)
        else:
            self.s2am = None
        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)

    def forward(self, from_up, from_down, mask=None):
        from_up = self.up_conv(from_up)
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        elif from_down is not None:
            x1 = from_up + from_down
        else:
            x1 = from_up
        if self.use_att:
            x1 = self.s2am(x1, mask)
        x1 = F.relu(self.conv1(x1))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        return x2


class DownCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, batch_norm=True):
        super(DownCoXvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


class UnetDecoderD(nn.Module):

    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True, transpose=True, concat=True, is_final=True):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        outs = in_channels
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool)
        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetDecoderDatt(nn.Module):

    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True, transpose=True, concat=True, is_final=True, norm=nn.BatchNorm2d, act=F.relu):
        super(UnetDecoderDatt, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.im_atts = []
        self.vm_atts = []
        self.mask_atts = []
        outs = in_channels
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, norm=nn.BatchNorm2d, act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(outs))
            self.vm_atts.append(SEBlock(outs))
            self.mask_atts.append(SEBlock(outs))
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, norm=nn.BatchNorm2d, act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(out_channels))
            self.vm_atts.append(SEBlock(out_channels))
            self.mask_atts.append(SEBlock(out_channels))
        self.up_convs = nn.ModuleList(self.up_convs)
        self.im_atts = nn.ModuleList(self.im_atts)
        self.vm_atts = nn.ModuleList(self.vm_atts)
        self.mask_atts = nn.ModuleList(self.mask_atts)
        reset_params(self)

    def forward(self, input, encoder_outs=None):
        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=self.im_atts[i])
        x_im = x
        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=self.mask_atts[i])
        x_mask = x
        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool, se=self.vm_atts[i])
        x_vm = x
        return x_im, x_mask, x_vm


class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters * 2 ** i
            pooling = True if i < depth - 1 else False
            down_conv = DownCoXvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs


class ResDown(nn.Module):

    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDown, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling)

    def forward(self, x):
        return self.model(x)


class ResUp(nn.Module):

    def __init__(self, in_size, out_size, use_att=False):
        super(ResUp, self).__init__()
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att)

    def forward(self, x, skip_input, mask=None):
        return self.model(x, skip_input, mask)


class ResDownNew(nn.Module):

    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDownNew, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling, norm=nn.InstanceNorm2d, act=F.leaky_relu)

    def forward(self, x):
        return self.model(x)


class ResUpNew(nn.Module):

    def __init__(self, in_size, out_size, use_att=False):
        super(ResUpNew, self).__init__()
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att, norm=nn.InstanceNorm2d)

    def forward(self, x, skip_input, mask=None):
        return self.model(x, skip_input, mask)


class VMSingle(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, down=ResDown, up=ResUp, ngf=32, res=True, use_att=False):
        super(VMSingle, self).__init__()
        self.down1 = down(in_channels, ngf)
        self.down2 = down(ngf, ngf * 2)
        self.down3 = down(ngf * 2, ngf * 4)
        self.down4 = down(ngf * 4, ngf * 8)
        self.down5 = down(ngf * 8, ngf * 16, pooling=False)
        self.up1 = up(ngf * 16, ngf * 8)
        self.up2 = up(ngf * 8, ngf * 4, use_att=use_att)
        self.up3 = up(ngf * 4, ngf * 2, use_att=use_att)
        self.up4 = up(ngf * 2, ngf * 1, use_att=use_att)
        self.im = nn.Conv2d(ngf, 3, 1)
        self.res = res

    def forward(self, input):
        img, mask = input[:, 0:3, :, :], input[:, 3:4, :, :]
        x, d1 = self.down1(input)
        x, d2 = self.down2(x)
        x, d3 = self.down3(x)
        x, d4 = self.down4(x)
        x, _ = self.down5(x)
        x = self.up1(x, d4)
        x = self.up2(x, d3, mask)
        x = self.up3(x, d2, mask)
        x = self.up4(x, d1, mask)
        im = self.im(x)
        return im


class VMSingleS2AM(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, down=ResDown, up=ResUp, ngf=32):
        super(VMSingleS2AM, self).__init__()
        self.down1 = down(in_channels, ngf)
        self.down2 = down(ngf, ngf * 2)
        self.down3 = down(ngf * 2, ngf * 4)
        self.down4 = down(ngf * 4, ngf * 8)
        self.down5 = down(ngf * 8, ngf * 16, pooling=False)
        self.up1 = up(ngf * 16, ngf * 8)
        self.up2 = up(ngf * 8, ngf * 4)
        self.s2am2 = RASC(ngf * 4)
        self.up3 = up(ngf * 4, ngf * 2)
        self.s2am3 = RASC(ngf * 2)
        self.up4 = up(ngf * 2, ngf * 1)
        self.s2am4 = RASC(ngf)
        self.im = nn.Conv2d(ngf, 3, 1)

    def forward(self, input):
        img, mask = input[:, 0:3, :, :], input[:, 3:4, :, :]
        x, d1 = self.down1(input)
        x, d2 = self.down2(x)
        x, d3 = self.down3(x)
        x, d4 = self.down4(x)
        x, _ = self.down5(x)
        x = self.up1(x, d4)
        x = self.up2(x, d3)
        x = self.s2am2(x, mask)
        x = self.up3(x, d2)
        x = self.s2am3(x, mask)
        x = self.up4(x, d1)
        x = self.s2am4(x, mask)
        im = self.im(x)
        return im


class MinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnetV2, self).__init__()
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)
        if self.withoutskip:
            x_out = self.up(x_up)
        elif self.is_attention:
            x_out = self.attention(torch.cat([x, self.up(x_up)], 1), mask), mask
        else:
            x_out = torch.cat([x, self.up(x_up)], 1), mask
        return x_out


class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""

    def __init__(self, down=None, up=None, submodule=None, attention=None, withoutskip=False, **kwags):
        super(MinimalUnet, self).__init__()
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None
        self.is_sub = not submodule == None

    def forward(self, x, mask=None):
        if self.is_sub:
            x_up, _ = self.sub(self.down(x), mask)
        else:
            x_up = self.down(x)
        if self.is_attention:
            x = self.attention(x, mask)
        if self.withoutskip:
            x_out = self.up(x_up)
        else:
            x_out = torch.cat([x, self.up(x_up)], 1), mask
        return x_out


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_attention_layer=False, attention_model=RASC, basicblock=MinimalUnet, outermostattention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = basicblock(down, up, submodule, withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = basicblock(down, up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if is_attention_layer:
                if MinimalUnetV2.__qualname__ in basicblock.__qualname__:
                    attention_model = attention_model(input_nc * 2)
                else:
                    attention_model = attention_model(input_nc)
            else:
                attention_model = None
            if use_dropout:
                model = basicblock(down, up.append(nn.Dropout(0.5)), submodule, attention_model, outermostattention=outermostattention)
            else:
                model = basicblock(down, up, submodule, attention_model, outermostattention=outermostattention)
        self.model = model

    def forward(self, x, mask=None):
        return self.model(x, mask)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, is_attention_layer=False, attention_model=RASC, use_inner_attention=False, basicblock=MinimalUnet):
        super(UnetGenerator, self).__init__()
        self.need_mask = not input_nc == output_nc
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, basicblock=basicblock)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_attention_layer=use_inner_attention, attention_model=attention_model, basicblock=basicblock)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_attention_layer=is_attention_layer, attention_model=attention_model, basicblock=basicblock)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_attention_layer=is_attention_layer, attention_model=attention_model, basicblock=basicblock)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_attention_layer=is_attention_layer, attention_model=attention_model, basicblock=basicblock, outermostattention=True)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, basicblock=basicblock, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        if self.need_mask:
            return self.model(input, input[:, 3:4, :, :])
        else:
            return self.model(input[:, 0:3, :, :], input[:, 3:4, :, :])


class UnetVMS2AMv4(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1, out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True, transpose=True, concat=True, transfer_data=True, long_skip=False, s2am='unet', use_coarser=True, no_stage2=False):
        super(UnetVMS2AMv4, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder, self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = blocks, blocks, blocks, blocks, blocks
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0], start_filters=start_filters, residual=residual, batch_norm=batch_norm, norm=nn.InstanceNorm2d, act=F.leaky_relu)
        self.image_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1), out_channels=out_channels_image, depth=depth - shared_depth, blocks=blocks[1], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.mask_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1), out_channels=out_channels_mask, depth=depth - shared_depth, blocks=blocks[2], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.vm_decoder = None
        if use_vm_decoder:
            self.vm_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1), out_channels=out_channels_image, depth=depth - shared_depth, blocks=blocks[3], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.shared_decoder = None
        self.use_coarser = use_coarser
        self.long_skip = long_skip
        self.no_stage2 = no_stage2
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderDatt(in_channels=start_filters * 2 ** (depth - 1), out_channels=start_filters * 2 ** (depth - shared_depth - 1), depth=shared_depth, blocks=blocks[4], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, is_final=False, norm=nn.InstanceNorm2d)
        if s2am == 'unet':
            self.s2am = UnetGenerator(4, 3, is_attention_layer=True, attention_model=RASC, basicblock=MinimalUnetV2)
        elif s2am == 'vm':
            self.s2am = VMSingle(4)
        elif s2am == 'vms2am':
            self.s2am = VMSingleS2AM(4, down=ResDownNew, up=ResUpNew)

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_image = torch.optim.Adam(self.image_decoder.parameters(), lr=0.001)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=0.001)
        self.optimizer_s2am = torch.optim.Adam(self.s2am.parameters(), lr=0.001)
        if self.vm_decoder is not None:
            self.optimizer_vm = torch.optim.Adam(self.vm_decoder.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        self.optimizer_s2am.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        self.optimizer_s2am.step()
        if self.vm_decoder is not None:
            self.optimizer_vm.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def step_optimizer_image(self):
        self.optimizer_image.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if not self.transfer_data:
            before_pool = None
        reconstructed_image = torch.tanh(self.image_decoder(image_code, before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(image_code, before_pool))
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask

    def shared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            shared_before_pool = before_pool[-self.shared - 1:]
            unshared_before_pool = before_pool[:-self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        im, mask, vm = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_image = torch.tanh(self.image_decoder(im, unshared_before_pool))
        if self.long_skip:
            reconstructed_image = reconstructed_image + synthesized
        reconstructed_mask = torch.sigmoid(self.mask_decoder(mask, unshared_before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(vm, unshared_before_pool))
            if self.long_skip:
                reconstructed_vm = reconstructed_vm + synthesized
        coarser = reconstructed_image * reconstructed_mask + (1 - reconstructed_mask) * synthesized
        if self.use_coarser:
            refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + coarser
        elif self.no_stage2:
            refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1)))
        else:
            refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + synthesized
        if self.vm_decoder is not None:
            return [refine, reconstructed_image], reconstructed_mask, reconstructed_vm
        else:
            return [refine, reconstructed_image], reconstructed_mask


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


class UnetVM(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1, out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True, transpose=True, concat=True, transfer_data=True, long_skip=False):
        super(UnetVM, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder, self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = blocks, blocks, blocks, blocks, blocks
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0], start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.image_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1), out_channels=out_channels_image, depth=depth - shared_depth, blocks=blocks[1], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat)
        self.mask_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1), out_channels=out_channels_mask, depth=depth, blocks=blocks[2], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat)
        self.vm_decoder = None
        if use_vm_decoder:
            self.vm_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1), out_channels=out_channels_image, depth=depth - shared_depth, blocks=blocks[3], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat)
        self.shared_decoder = None
        self.long_skip = long_skip
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1), out_channels=start_filters * 2 ** (depth - shared_depth - 1), depth=shared_depth, blocks=blocks[4], residual=residual, batch_norm=batch_norm, transpose=transpose, concat=concat, is_final=False)

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_image = torch.optim.Adam(self.image_decoder.parameters(), lr=0.001)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=0.001)
        if self.vm_decoder is not None:
            self.optimizer_vm = torch.optim.Adam(self.vm_decoder.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        if self.vm_decoder is not None:
            self.optimizer_vm.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def step_optimizer_image(self):
        self.optimizer_image.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if not self.transfer_data:
            before_pool = None
        reconstructed_image = torch.tanh(self.image_decoder(image_code, before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(image_code, before_pool))
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask

    def shared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            shared_before_pool = before_pool[-self.shared - 1:]
            unshared_before_pool = before_pool[:-self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        x = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_image = torch.tanh(self.image_decoder(x, unshared_before_pool))
        if self.long_skip:
            reconstructed_image = reconstructed_image + synthesized
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(x, unshared_before_pool))
            if self.long_skip:
                reconstructed_vm = reconstructed_vm + synthesized
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask


class gen_gan(nn.Module):

    def __init__(self, gantype):
        super(gen_gan, self).__init__()
        if gantype == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gantype == 'naive':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise Exception('error gan type')

    def forward(self, dis_fake):
        return self.criterion(dis_fake, torch.ones_like(dis_fake))


class dis_gan(nn.Module):

    def __init__(self, gantype):
        super(dis_gan, self).__init__()
        if gantype == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gantype == 'naive':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise Exception('error gan type')

    def forward(self, dis_fake, dis_real):
        loss_fake = self.criterion(dis_fake, torch.zeros_like(dis_fake))
        loss_real = self.criterion(dis_real, torch.ones_like(dis_real))
        return loss_fake, loss_real


class DistributedDataParallelModel(DistributedDataParallel):
    """Implements data parallelism at the module level for the DistributedDataParallel module.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = encoding.nn.DistributedDataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.device(device):
                if not isinstance(input, (list, tuple)):
                    input = input,
                if not isinstance(target, (list, tuple)):
                    target = target,
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input, target, kwargs, device)) for i, (module, input, target, kwargs, device) in enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage.
    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicLearningBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CBAMConnect,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPoolX,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownCoXvD,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianSmoothing,
     lambda: ([], {'channels': 4, 'kernel_size': 4, 'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAttentionModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAttentionModuleJustSigmoid,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedURASC,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RASC,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     True),
    (ResDown,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SEBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SENet,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SNCoXvWithActivation,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialAttentionModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNO,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (URASC,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetEncoderD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UnetVM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG16FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGGLossA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGGLossX,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (VMSingle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (WeightedBCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_vinthony_deep_blind_watermark_removal(_paritybench_base):
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

