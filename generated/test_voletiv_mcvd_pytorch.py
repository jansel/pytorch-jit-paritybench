import sys
_module = sys.modules[__name__]
del sys
datasets = _module
bair = _module
bair_convert = _module
celeba = _module
cityscapes = _module
cityscapes_convert = _module
ffhq = _module
ffhq_tfrecords = _module
h5 = _module
imagenet = _module
kinetics600_convert = _module
kth = _module
kth_convert = _module
moving_mnist = _module
stochastic_moving_mnist = _module
ucf101 = _module
ucf101_convert = _module
utils = _module
vision = _module
fid_PR = _module
fid_score_OLD = _module
inception = _module
nearest_neighbor = _module
pr = _module
simple_sample = _module
load_model_from_ckpt = _module
losses = _module
dsm = _module
main = _module
models = _module
base_model = _module
better = _module
layers = _module
layers3d = _module
layerspp = _module
ncsnpp_more = _module
normalization = _module
op = _module
fused_act = _module
upfirdn2d = _module
up_or_down_sampling = _module
utils = _module
dist_model = _module
ema = _module
eval_models = _module
fvd = _module
convert_tf_pretrained = _module
fvd = _module
pytorch_i3d = _module
networks_basic = _module
pndm = _module
pretrained_networks = _module
unet = _module
quick_sample = _module
runners = _module
ncsn_runner = _module

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


import torch


import torchvision.transforms as transforms


from torchvision.datasets import CIFAR10


from torchvision.datasets import LSUN


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torch.utils.data import Dataset


from torchvision import transforms


from torch.utils.data.distributed import DistributedSampler


import math


import random


import torch.nn as nn


import torch.utils.data as data


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


from torchvision import datasets


from torch.utils.model_zoo import tqdm


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torch.nn.functional as F


import torchvision


import sklearn.metrics


from torchvision.datasets import CelebA


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


from torchvision.utils import save_image


from functools import partial


from torchvision.utils import make_grid


import torch.optim as optim


from torch.distributions.gamma import Gamma


import copy


import logging


import time


from scipy.stats import hmean


import string


import functools


from torch import nn


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.cpp_extension import load


from torch.autograd import Variable


import itertools


from scipy.ndimage import zoom


from typing import Tuple


from scipy.linalg import sqrtm


import torch.nn.init as init


from collections import namedtuple


from torchvision import models as tv


import matplotlib


import matplotlib.pyplot as plt


import scipy.stats as st


import torchvision.transforms as Transforms


from math import ceil


from math import log10


from torchvision.transforms.functional import resized_crop


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`

    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class Dense(nn.Module):
    """Linear layer with `default_init`."""

    def __init__(self):
        super().__init__()


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(in_planes, out_planes, stride=stride, bias=bias, dilation=dilation, padding=padding, kernel_size=3)
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class CRPBlock(nn.Module):

    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):

    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class RCUBlock(nn.Module):

    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class CondRCUBlock(nn.Module):

    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class MSFBlock(nn.Module):

    def __init__(self, in_planes, features):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):

    def __init__(self, in_planes, features, num_classes, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):

    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
        super().__init__()
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)
        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))
        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)
        if not start:
            self.msf = MSFBlock(in_planes, features)
        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)
        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]
        h = self.crp(h)
        h = self.output_convs(h)
        return h


class CondRefineBlock(nn.Module):

    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
        super().__init__()
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)
        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act))
        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)
        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)
        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)
        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]
        h = self.crp(h, y)
        h = self.output_convs(h, y)
        return h


class ConvMeanPool(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return output


class MeanPoolConv(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        output = inputs
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return self.conv(output)


class UpsampleConv(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalInstanceNorm2dPlus(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / torch.sqrt(v + 1e-05)
        h = self.instance_norm(x)
        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim, num_classes, resample=1, act=nn.ELU(), normalization=ConditionalInstanceNorm2dPlus, adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)
        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=0):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation, padding=padding)
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(), normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output


class Nin(nn.Module):
    """ Shared weights """

    def __init__(self, channel_in: int, channel_out: int, init_scale=1.0):
        super().__init__()
        self.channel_out = channel_out
        self.weights = nn.Parameter(torch.zeros(channel_out, channel_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights, math.sqrt(1e-10 if init_scale == 0.0 else init_scale))
        self.bias = nn.Parameter(torch.zeros(channel_out), requires_grad=True)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        bs, _, width, _ = x.shape
        res = torch.bmm(self.weights.repeat(bs, 1, 1), x.flatten(2)) + self.bias.unsqueeze(0).unsqueeze(-1)
        return res.view(bs, self.channel_out, width, width)


def Normalize(num_channels):
    return nn.GroupNorm(eps=1e-06, num_groups=32, num_channels=num_channels)


class AttnBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.Q = Nin(channels, channels)
        self.K = Nin(channels, channels)
        self.V = Nin(channels, channels)
        self.OUT = Nin(channels, channels, init_scale=0.0)
        self.normalize = Normalize(channels)
        self.c = channels

    def forward(self, x):
        h = self.normalize(x)
        q, k, v = self.Q(h), self.K(h), self.V(h)
        w = torch.einsum('abcd,abef->acdef', q, k) * (1 / math.sqrt(self.c))
        batch_size, width, *_ = w.shape
        w = F.softmax(w.view(batch_size, width, width, width * width), dim=-1)
        w = w.view(batch_size, *([width] * 4))
        h = torch.einsum('abcde,afde->afbc', w, v)
        return x + self.OUT(h)


def default_init(module, scale):
    if scale == 0:
        scale = 1e-10
    torch.nn.init.xavier_uniform_(module.weight, math.sqrt(scale))
    torch.nn.init.zeros_(module.bias)


def init_weights(module, scale=1, module_is_list=False):
    if module_is_list:
        for module_ in module.modules():
            if isinstance(module_, nn.Conv2d) or isinstance(module_, nn.Linear):
                default_init(module_, scale)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        default_init(module, scale)


class Upsample(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):

    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        elif with_conv:
            self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, down=True, resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        elif not self.with_conv:
            x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
        else:
            x = self.Conv2d_0(x)
        return x


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0, n_heads=1, n_head_channels=-1):
        super().__init__()
        num_groups = min(channels // 4, 32)
        while channels % num_groups != 0:
            num_groups -= 1
        self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-06)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        if n_head_channels == -1:
            self.n_heads = n_heads
        elif channels < n_head_channels:
            self.n_heads = 1
        else:
            assert channels % n_head_channels == 0
            self.n_heads = channels // n_head_channels

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        C = C // self.n_heads
        w = torch.einsum('bchw,bcij->bhwij', q.reshape(B * self.n_heads, C, H, W), k.reshape(B * self.n_heads, C, H, W)) * int(C) ** -0.5
        w = torch.reshape(w, (B * self.n_heads, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B * self.n_heads, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v.reshape(B * self.n_heads, C, H, W))
        h = h.reshape(B, C * self.n_heads, H, W)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class NIN1d(nn.Module):

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 2, 1)


class AttnBlockpp1d(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM. in 1D"""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0, n_heads=1, n_head_channels=-1):
        super().__init__()
        num_groups = min(channels // 4, 32)
        while channels % num_groups != 0:
            num_groups -= 1
        self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-06)
        self.NIN_0 = NIN1d(channels, channels)
        self.NIN_1 = NIN1d(channels, channels)
        self.NIN_2 = NIN1d(channels, channels)
        self.NIN_3 = NIN1d(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        if n_head_channels == -1:
            self.n_heads = n_heads
        elif channels < n_head_channels:
            self.n_heads = 1
        else:
            assert channels % n_head_channels == 0
            self.n_heads = channels // n_head_channels

    def forward(self, x):
        B, C, T = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        C = C // self.n_heads
        w = torch.einsum('bct,bci->bti', q.reshape(B * self.n_heads, C, T), k.reshape(B * self.n_heads, C, T)) * int(C) ** -0.5
        w = torch.reshape(w, (B * self.n_heads, T, T))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B * self.n_heads, T, T))
        h = torch.einsum('bti,bci->bct', w, v.reshape(B * self.n_heads, C, T))
        h = h.reshape(B, C * self.n_heads, T)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class NIN3d(nn.Module):

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 4, 1, 2, 3)


class AttnBlockpp3d_old(nn.Module):
    """Channel-wise 3d self-attention block."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0, n_heads=1, n_head_channels=-1, n_frames=1):
        super().__init__()
        self.N = n_frames
        self.channels = self.Cin = channels // n_frames
        num_groups = min(self.channels // 4, 32)
        while self.channels % num_groups != 0:
            num_groups -= 1
        self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=self.channels, eps=1e-06)
        self.NIN_0 = NIN3d(self.channels, self.channels)
        self.NIN_1 = NIN3d(self.channels, self.channels)
        self.NIN_2 = NIN3d(self.channels, self.channels)
        self.NIN_3 = NIN3d(self.channels, self.channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        if n_head_channels == -1:
            self.n_heads = n_heads
        elif self.channels < n_head_channels:
            self.n_heads = 1
        else:
            assert self.channels % n_head_channels == 0
            self.n_heads = self.channels // n_head_channels

    def forward(self, x):
        B, CN, H, W = x.shape
        C = self.Cin
        N = self.N
        x = x.reshape(B, C, N, H, W)
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        C = C // self.n_heads
        w = torch.einsum('bcnhw,bcnij->bnhwij', q.reshape(B * self.n_heads, C, N, H, W), k.reshape(B * self.n_heads, C, N, H, W)) * int(C) ** -0.5
        w = torch.reshape(w, (B * self.n_heads, N, H, W, N * H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B * self.n_heads, N, H, W, N, H, W))
        h = torch.einsum('bnhwijk,bcijk->bcnhw', w, v.reshape(B * self.n_heads, C, N, H, W))
        h = h.reshape(B, C * self.n_heads, N, H, W)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            x = x + h
        else:
            x = (x + h) / np.sqrt(2.0)
        return x.reshape(B, C * N, H, W)


class AttnBlockpp3d(nn.Module):
    """Channel-wise 3d self-attention block."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0, n_heads=1, n_head_channels=-1, n_frames=1, act=None):
        super().__init__()
        self.N = n_frames
        self.channels = self.Cin = channels // n_frames
        self.space_att = AttnBlockpp(channels=self.channels, skip_rescale=skip_rescale, init_scale=init_scale, n_heads=n_heads, n_head_channels=n_head_channels)
        self.time_att = AttnBlockpp1d(channels=self.channels, skip_rescale=skip_rescale, init_scale=init_scale, n_heads=n_heads, n_head_channels=n_head_channels)
        self.act = act

    def forward(self, x):
        B, CN, H, W = x.shape
        C = self.Cin
        N = self.N
        x = x.reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
        x = self.space_att(x)
        x = x.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4)
        if self.act is not None:
            x = self.act(x)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, N)
        x = self.time_att(x)
        x = x.reshape(B, H, W, C, N).permute(0, 3, 4, 1, 2).reshape(B, C * N, H, W)
        return x


class MyConv3d(nn.Module):
    """3d convolution."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, bias=True, init_scale=1.0, padding=0, dilation=1, n_frames=1):
        super().__init__()
        self.N = n_frames
        self.Cin = in_planes // n_frames
        self.Cout = out_planes // n_frames
        self.conv = nn.Conv3d(self.Cin, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.conv.weight.data = default_init(init_scale)(self.conv.weight.data.shape)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        B, CN, H, W = x.shape
        x = x.reshape(B, self.Cin, self.N, H, W)
        x = self.conv(x)
        x = x.reshape(B, self.Cout * self.N, H, W)
        return x


class PseudoConv3d(nn.Module):
    """Pseudo3d convolution."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, bias=True, init_scale=1.0, padding=0, dilation=1, n_frames=1, act=None):
        super().__init__()
        self.N = n_frames
        self.Cin = in_planes // n_frames
        self.Cout = out_planes // n_frames
        self.space_conv = nn.Conv2d(self.Cin, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.space_conv.weight.data = default_init(init_scale)(self.space_conv.weight.data.shape)
        nn.init.zeros_(self.space_conv.bias)
        self.time_conv = nn.Conv1d(self.Cout, self.Cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.time_conv.weight.data = default_init(init_scale)(self.time_conv.weight.data.shape)
        nn.init.zeros_(self.time_conv.bias)
        self.act = act

    def forward(self, x):
        B, CN, H, W = x.shape
        C = self.Cin
        N = self.N
        x = x.reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
        x = self.space_conv(x)
        C = self.Cout
        x = x.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4)
        if self.act is not None:
            x = self.act(x)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, N)
        x = self.time_conv(x)
        x = x.reshape(B, H, W, C, N).permute(0, 3, 4, 1, 2).reshape(B, C * N, H, W)
        return x


class SPADE(nn.Module):

    def __init__(self, norm_nc, label_nc):
        super().__init__()
        param_free_norm_type = 'group'
        ks = 3
        if param_free_norm_type == 'group':
            num_groups = min(norm_nc // 4, 32)
            while norm_nc % num_groups != 0:
                num_groups -= 1
            self.param_free_norm = nn.GroupNorm(num_groups=num_groups, num_channels=norm_nc, affine=False, eps=1e-06)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1, skip_rescale=False, init_scale=0.0, is3d=False, n_frames=1, pseudo3d=False, act3d=False):
        super().__init__()
        if pseudo3d or is3d:
            conv3x3_3d = layers3d.ddpm_conv3x3_3d
            conv1x1_3d = layers3d.ddpm_conv1x1_3d
            conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
            conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d
        if pseudo3d:
            conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
        elif is3d:
            conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
            conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
        else:
            conv3x3_ = conv3x3
            conv1x1_ = conv1x1
        out_ch = out_ch if out_ch else in_ch
        num_groups = min(in_ch // 4, 32)
        while in_ch % num_groups != 0:
            num_groups -= 1
        self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=in_ch, eps=1e-06)
        self.Conv_0 = conv3x3_(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        num_groups = min(out_ch // 4, 32)
        while in_ch % num_groups != 0:
            num_groups -= 1
        self.GroupNorm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch, eps=1e-06)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3_(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockDDPMppSPADE(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, spade_dim=128, dropout=0.1, skip_rescale=False, init_scale=0.0, is3d=False, n_frames=1, num_frames_cond=0, cond_ch=0, pseudo3d=False, act3d=False):
        super().__init__()
        if pseudo3d or is3d:
            conv3x3_3d = layers3d.ddpm_conv3x3_3d
            conv1x1_3d = layers3d.ddpm_conv1x1_3d
            conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
            conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d
        if pseudo3d:
            conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_cond = functools.partial(conv1x1_pseudo3d, n_frames=cond_ch // num_frames_cond, act=act if act3d else None)
        elif is3d:
            conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
            conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
            conv1x1_cond = functools.partial(conv1x1_3d, n_frames=cond_ch // num_frames_cond)
        else:
            conv3x3_ = conv3x3
            conv1x1_ = conv1x1
            conv1x1_cond = conv1x1
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = MySPADE(norm_nc=in_ch // n_frames if is3d else in_ch, label_nc=cond_ch, param_free_norm_type='group', act=act, conv=conv3x3_, spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=conv1x1_cond)
        self.Conv_0 = conv3x3_(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = MySPADE(norm_nc=out_ch // n_frames if is3d else out_ch, label_nc=cond_ch, param_free_norm_type='group', act=act, conv=conv3x3_, spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=conv1x1_cond)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3_(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None, cond=None):
        h = self.act(self.GroupNorm_0(x, cond))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, cond))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


def get_norm(norm, ch, affine=True):
    """Get activation functions from the opt file."""
    if norm == 'none':
        return nn.Identity()
    elif norm == 'batch':
        return nn.BatchNorm1d(ch, affine=affine)
    elif norm == 'evo':
        return EvoNorm2D(ch=ch, affine=affine, eps=1e-05, groups=min(ch // 4, 32))
    elif norm == 'group':
        num_groups = min(ch // 4, 32)
        while ch % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-05, affine=affine)
    elif norm == 'layer':
        return nn.LayerNorm(normalized_shape=ch, eps=1e-05, elementwise_affine=affine)
    elif norm == 'instance':
        return nn.InstanceNorm2d(num_features=ch, eps=1e-05, affine=affine)
    else:
        raise NotImplementedError('norm choice does not exist')


class get_act_norm(nn.Module):

    def __init__(self, act, act_emb, norm, ch, emb_dim=None, spectral=False, is3d=False, n_frames=1, num_frames_cond=0, cond_ch=0, spade_dim=128, cond_conv=None, cond_conv1=None):
        super(get_act_norm, self).__init__()
        self.norm = norm
        self.act = act
        self.act_emb = act_emb
        self.is3d = is3d
        self.n_frames = n_frames
        self.cond_ch = cond_ch
        if emb_dim is not None:
            if self.is3d:
                out_dim = 2 * (ch // self.n_frames)
            else:
                out_dim = 2 * ch
            if spectral:
                self.Dense_0 = torch.nn.utils.spectral_norm(nn.Linear(emb_dim, out_dim))
            else:
                self.Dense_0 = nn.Linear(emb_dim, out_dim)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
            affine = False
        else:
            affine = True
        if norm == 'spade':
            self.Norm_0 = MySPADE(norm_nc=ch // n_frames if is3d else ch, label_nc=cond_ch, param_free_norm_type='group', act=act, conv=cond_conv, spade_dim=spade_dim, is3d=is3d, num_frames=n_frames, num_frames_cond=num_frames_cond, conv1x1=cond_conv1)
        else:
            self.Norm_0 = get_norm(norm, ch // n_frames if is3d else ch, affine)

    def forward(self, x, emb=None, cond=None):
        if emb is not None:
            emb_out = self.Dense_0(self.act_emb(emb))[:, :, None, None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            if self.is3d:
                B, CN, H, W = x.shape
                N = self.n_frames
                scale = scale.reshape(B, -1, 1, 1, 1)
                shift = shift.reshape(B, -1, 1, 1, 1)
                x = x.reshape(B, -1, N, H, W)
            if self.norm == 'spade':
                emb_norm = self.Norm_0(x, cond)
                emb_norm = emb_norm.reshape(B, -1, N, H, W) if self.is3d else emb_norm
            else:
                emb_norm = self.Norm_0(x)
            x = emb_norm * (1 + scale) + shift
            if self.is3d:
                x = x.reshape(B, -1, H, W)
        else:
            if self.is3d:
                B, CN, H, W = x.shape
                N = self.n_frames
                x = x.reshape(B, -1, N, H, W)
            if self.norm == 'spade':
                x = self.Norm_0(x, cond)
            else:
                x = self.Norm_0(x)
                x = x.reshape(B, CN, H, W) if self.is3d else x
        x = self.act(x)
        return x


class ResnetBlockBigGANppGN(nn.Module):

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, init_scale=0.0, is3d=False, n_frames=1, pseudo3d=False, act3d=False):
        super().__init__()
        if pseudo3d or is3d:
            conv3x3_3d = layers3d.ddpm_conv3x3_3d
            conv1x1_3d = layers3d.ddpm_conv1x1_3d
            conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
            conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d
        if pseudo3d:
            conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
        elif is3d:
            conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
            conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
        else:
            conv3x3_ = conv3x3
            conv1x1_ = conv1x1
        out_ch = out_ch if out_ch else in_ch
        self.actnorm0 = get_act_norm(act, act, 'group', in_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.Conv_0 = conv3x3_(in_ch, out_ch)
        self.actnorm1 = get_act_norm(act, act, 'group', out_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1_(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.actnorm0(x, temb)
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        h = self.actnorm1(h, temb)
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANppSPADE(nn.Module):

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False, spade_dim=128, dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), n_frames=1, num_frames_cond=0, cond_ch=0, skip_rescale=True, init_scale=0.0, is3d=False, pseudo3d=False, act3d=False):
        super().__init__()
        if pseudo3d or is3d:
            conv3x3_3d = layers3d.ddpm_conv3x3_3d
            conv1x1_3d = layers3d.ddpm_conv1x1_3d
            conv3x3_pseudo3d = layers3d.ddpm_conv3x3_pseudo3d
            conv1x1_pseudo3d = layers3d.ddpm_conv1x1_pseudo3d
        if pseudo3d:
            conv3x3_ = functools.partial(conv3x3_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_ = functools.partial(conv1x1_pseudo3d, n_frames=n_frames, act=act if act3d else None)
            conv1x1_cond = functools.partial(conv1x1_pseudo3d, n_frames=cond_ch // num_frames_cond, act=act if act3d else None)
        elif is3d:
            conv3x3_ = functools.partial(conv3x3_3d, n_frames=n_frames)
            conv1x1_ = functools.partial(conv1x1_3d, n_frames=n_frames)
            conv1x1_cond = functools.partial(conv1x1_3d, n_frames=cond_ch // num_frames_cond)
        else:
            conv3x3_ = conv3x3
            conv1x1_ = conv1x1
            conv1x1_cond = conv1x1
        out_ch = out_ch if out_ch else in_ch
        self.actnorm0 = get_act_norm(act, act, 'spade', in_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames, num_frames_cond=num_frames_cond, cond_ch=cond_ch, spade_dim=spade_dim, cond_conv=conv3x3_, cond_conv1=conv1x1_cond)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.Conv_0 = conv3x3_(in_ch, out_ch)
        self.actnorm1 = get_act_norm(act, act, 'spade', out_ch, emb_dim=temb_dim, spectral=False, is3d=is3d, n_frames=n_frames, num_frames_cond=num_frames_cond, cond_ch=cond_ch, spade_dim=spade_dim, cond_conv=conv3x3_, cond_conv1=conv1x1_cond)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1_(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, cond=None):
        h = self.actnorm0(x, temb, cond)
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        h = self.actnorm1(h, temb, cond)
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppGN


default_initializer = layers.default_init


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
    sigmas = np.exp(np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))
    return sigmas


class UNetMore_DDPM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.version = getattr(config.model, 'version', 'DDPM').upper()
        assert self.version == 'DDPM' or self.version == 'DDIM' or self.version == 'FPNDM', f'models/unet : version is not DDPM or DDIM! Given: {self.version}'
        self.config = config
        if getattr(config.model, 'spade', False):
            self.unet = SPADE_NCSNpp(config)
        else:
            self.unet = NCSNpp(config)
        self.schedule = getattr(config.model, 'sigma_dist', 'linear')
        if self.schedule == 'linear':
            self.register_buffer('betas', get_sigmas(config))
            self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0])]))
        elif self.schedule == 'cosine':
            self.register_buffer('alphas', get_sigmas(config))
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0])]))
            self.register_buffer('betas', 1 - self.alphas / self.alphas_prev)
        self.gamma = getattr(config.model, 'gamma', False)
        if self.gamma:
            self.theta_0 = 0.001
            self.register_buffer('k', self.betas / (self.alphas * self.theta_0 ** 2))
            self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0))
            self.register_buffer('theta_t', torch.sqrt(self.alphas) * self.theta_0)
        self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

    def forward(self, x, y, cond=None, cond_mask=None):
        if self.noise_in_cond and cond is not None:
            alphas = self.alphas
            labels = y
            used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
            if self.gamma:
                used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                z = torch.distributions.gamma.Gamma(used_k, 1 / used_theta).sample()
                z = (z - used_k * used_theta) / (1 - used_alphas).sqrt()
            else:
                z = torch.randn_like(cond)
            cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z
        return self.unet(x, y, cond, cond_mask=cond_mask)


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalVarianceNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-05)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class VarianceNorm2d(nn.Module):

    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-05)
        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalNoneNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * x
        return out


class NoneNorm2d(nn.Module):

    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm2dPlus(nn.Module):

    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / torch.sqrt(v + 1e-05)
        h = self.instance_norm(x)
        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    else:
        upfirdn2d_op = load('upfirdn2d', sources=[os.path.join(module_path, 'upfirdn2d.cpp'), os.path.join(module_path, 'upfirdn2d_kernel.cu')])


        class UpFirDn2dBackward(Function):

            @staticmethod
            def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
                up_x, up_y = up
                down_x, down_y = down
                g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
                grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
                grad_input = upfirdn2d_op.upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
                grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
                ctx.save_for_backward(kernel)
                pad_x0, pad_x1, pad_y0, pad_y1 = pad
                ctx.up_x = up_x
                ctx.up_y = up_y
                ctx.down_x = down_x
                ctx.down_y = down_y
                ctx.pad_x0 = pad_x0
                ctx.pad_x1 = pad_x1
                ctx.pad_y0 = pad_y0
                ctx.pad_y1 = pad_y1
                ctx.in_size = in_size
                ctx.out_size = out_size
                return grad_input

            @staticmethod
            def backward(ctx, gradgrad_input):
                kernel, = ctx.saved_tensors
                gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
                gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
                gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
                return gradgrad_out, None, None, None, None, None, None, None, None


        class UpFirDn2d(Function):

            @staticmethod
            def forward(ctx, input, kernel, up, down, pad):
                up_x, up_y = up
                down_x, down_y = down
                pad_x0, pad_x1, pad_y0, pad_y1 = pad
                kernel_h, kernel_w = kernel.shape
                batch, channel, in_h, in_w = input.shape
                ctx.in_size = input.shape
                input = input.reshape(-1, in_h, in_w, 1)
                ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
                out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
                out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
                ctx.out_size = out_h, out_w
                ctx.up = up_x, up_y
                ctx.down = down_x, down_y
                ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
                g_pad_x0 = kernel_w - pad_x0 - 1
                g_pad_y0 = kernel_h - pad_y0 - 1
                g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
                g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
                ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
                out = upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
                out = out.view(-1, channel, out_h, out_w)
                return out

            @staticmethod
            def backward(ctx, grad_output):
                kernel, grad_kernel = ctx.saved_tensors
                grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
                return grad_input, None, None, None, None
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _shape(x, dim):
    return x.shape[dim]


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """
    assert isinstance(factor, int) and factor >= 1
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * factor ** 2)
    p = k.shape[0] - factor - (convW - 1)
    stride = factor, factor
    stride = [1, 1, factor, factor]
    output_shape = (_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW
    output_padding = output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH, output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))
    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
    return upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(self, in_ch, out_ch, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1), use_bias=True, kernel_init=None):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)
        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)
        return x


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, device='cpu'):
        super(PerceptualLoss, self).__init__()
        None
        self.device = device
        self.spatial = spatial
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, colorspace=colorspace, spatial=self.spatial, device=device)
        None
        None

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]
        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1
        return self.model.forward(target, pred)


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels, kernel_size=self._kernel_shape, stride=self._stride, padding=0, bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=1e-05, momentum=0.001)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """
    VALID_ENDPOINTS = 'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions'

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        logits = logits.mean(dim=2)
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):
    in_H = in_tens.shape[2]
    scale_factor = 1.0 * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, device='cpu', colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.device = device
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = util.dssim(1.0 * util.tensor2im(in0.data), 1.0 * util.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)), util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        return ret_var


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


class Swish(nn.Module):
    """
    Swish out-performs Relu for deep NN (more than 40 layers). Although, the performance of relu and swish model
    degrades with increasing batch size, swish performs better than relu.
    https://jmlb.github.io/ml/2017/12/31/swish_activation_function/ (December 31th 2017)
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, channel_in, channel_out, dropout, tembdim, conditional=False):
        super().__init__()
        self.dropout = dropout
        self.nonlinearity = Swish()
        self.normalize0 = Normalize(channel_in)
        self.conv0 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv0)
        self.conditional = conditional
        if conditional:
            self.dense = nn.Linear(tembdim, channel_out)
            init_weights(self.dense)
        self.normalize1 = Normalize(channel_out)
        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        init_weights(self.conv1, scale=0)
        if channel_in != channel_out:
            self.nin = Nin(channel_in, channel_out)
        else:
            self.nin = nn.Identity()
        self.channel_in = channel_in

    def forward(self, x, temb=None):
        h = self.nonlinearity(self.normalize0(x))
        h = self.conv0(h)
        if temb is not None and self.conditional:
            h += self.dense(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.nonlinearity(self.normalize1(h))
        return self.nin(x) + self.conv1(self.dropout(h))


def get_timestep_embedding(timesteps, embedding_dim: int=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = nn.ZeroPad2d((0, 1, 0, 0))(emb)
    assert [*emb.shape] == [timesteps.shape[0], embedding_dim], f'{emb.shape}, {str([timesteps.shape[0], embedding_dim])}'
    return emb


def partialclass(cls, *args, **kwds):


    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewCls


class UNet(nn.Module):

    def __init__(self, config):
        super(UNet, self).__init__()
        self.locals = [config]
        self.config = config
        self.n_channels = n_channels = config.data.channels
        self.ch = ch = config.model.ngf
        self.mode = mode = getattr(config, 'mode', 'deep')
        assert mode in ['deep', 'deeper', 'deepest']
        self.dropout = nn.Dropout2d(p=getattr(config.model, 'dropout', 0.0))
        self.time_conditional = time_conditional = getattr(config.model, 'time_conditional', False)
        self.version = getattr(config.model, 'version', 'SMLD').upper()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.num_frames = num_frames = getattr(config.data, 'num_frames', 1)
        self.num_frames_cond = num_frames_cond = getattr(config.data, 'num_frames_cond', 0) + getattr(config.data, 'num_frames_future', 0)
        ResnetBlock_ = partialclass(ResnetBlock, dropout=self.dropout, tembdim=ch * 4, conditional=time_conditional)
        if mode == 'deepest':
            ch_mult = [(ch * n) for n in (1, 2, 2, 2, 4, 4)]
        elif mode == 'deeper':
            ch_mult = [(ch * n) for n in (1, 2, 2, 4, 4)]
        else:
            ch_mult = [(ch * n) for n in (1, 2, 2, 2)]
        self.downblocks = nn.ModuleList()
        self.downblocks.append(nn.Conv2d(n_channels * (num_frames + num_frames_cond), ch, kernel_size=3, padding=1, stride=1))
        prev_ch = ch_mult[0]
        ch_size = [ch]
        for i, ich in enumerate(ch_mult):
            for firstarg in [prev_ch, ich]:
                self.downblocks.append(ResnetBlock_(firstarg, ich))
                ch_size += [ich]
                if i == 1:
                    self.downblocks.append(AttnBlock(ich))
            if i != len(ch_mult) - 1:
                self.downblocks.append(nn.Conv2d(ich, ich, kernel_size=3, stride=2, padding=1))
                ch_size += [ich]
            prev_ch = ich
        init_weights(self.downblocks, module_is_list=True)
        self.middleblocks = nn.ModuleList()
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))
        self.middleblocks.append(AttnBlock(ch_mult[-1]))
        self.middleblocks.append(ResnetBlock_(ch_mult[-1], ch_mult[-1]))
        self.upblocks = nn.ModuleList()
        prev_ich = ch_mult[-1]
        for i, ich in reversed(list(enumerate(ch_mult))):
            for _ in range(3):
                self.upblocks.append(ResnetBlock_(prev_ich + ch_size.pop(), ich))
                if i == 1:
                    self.upblocks.append(AttnBlock(ich))
                prev_ich = ich
            if i != 0:
                self.upblocks.append(Upsample(ich))
        self.normalize = Normalize(ch)
        self.nonlinearity = Swish()
        self.out = nn.Conv2d(ch, n_channels * (num_frames + num_frames_cond) if getattr(config.model, 'output_all_frames', False) else n_channels * num_frames, kernel_size=3, stride=1, padding=1)
        init_weights(self.out, scale=0)
        self.temb_dense = nn.Sequential(nn.Linear(ch, ch * 4), self.nonlinearity, nn.Linear(ch * 4, ch * 4), self.nonlinearity)
        init_weights(self.temb_dense, module_is_list=True)

    def forward(self, x, y=None, cond=None):
        if y is not None and self.time_conditional:
            temb = get_timestep_embedding(y, self.ch)
            temb = self.temb_dense(temb)
        else:
            temb = None
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        if not self.logit_transform and not self.rescaled:
            x = 2 * x - 1.0
        hs = []
        for module in self.downblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)
            if isinstance(module, AttnBlock):
                hs.pop()
            hs += [x]
        for module in self.middleblocks:
            if isinstance(module, ResnetBlock):
                x = module(x, temb)
            else:
                x = module(x)
        for module in self.upblocks:
            if isinstance(module, ResnetBlock):
                x = module(torch.cat((x, hs.pop()), dim=1), temb)
            else:
                x = module(x)
        x = self.nonlinearity(self.normalize(x))
        output = self.out(x)
        if getattr(self.config.model, 'output_all_frames', False) and cond is not None:
            _, output = torch.split(output, [self.num_frames_cond * self.config.data.channels, self.num_frames * self.config.data.channels], dim=1)
        return output


class UNet_SMLD(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.version = getattr(config.model, 'version', 'SMLD').upper()
        assert self.version == 'SMLD', f'models/unet : version is not SMLD! Given: {self.version}'
        self.config = config
        self.unet = UNet(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

    def forward(self, x, y, cond=None, labels=None):
        if self.noise_in_cond and cond is not None:
            sigmas = self.sigmas
            labels = y
            used_sigmas = sigmas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
            z = torch.randn_like(cond)
            cond = cond + used_sigmas * z
        return self.unet(x, y, cond)


class UNet_DDPM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.version = getattr(config.model, 'version', 'DDPM').upper()
        assert self.version == 'DDPM' or self.version == 'DDIM' or self.version == 'FPNDM', f'models/unet : version is not DDPM or DDIM! Given: {self.version}'
        self.config = config
        self.unet = UNet(config)
        self.schedule = getattr(config.model, 'sigma_dist', 'linear')
        if self.schedule == 'linear':
            self.register_buffer('betas', get_sigmas(config))
            self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0])]))
        elif self.schedule == 'cosine':
            self.register_buffer('alphas', get_sigmas(config))
            self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0])]))
            self.register_buffer('betas', (1 - self.alphas / self.alphas_prev).clip_(0, 0.999))
        self.gamma = getattr(config.model, 'gamma', False)
        if self.gamma:
            self.theta_0 = 0.001
            self.register_buffer('k', self.betas / (self.alphas * self.theta_0 ** 2))
            self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0))
            self.register_buffer('theta_t', torch.sqrt(self.alphas) * self.theta_0)
        self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

    def forward(self, x, y, cond=None, labels=None, cond_mask=None):
        if self.noise_in_cond and cond is not None:
            alphas = self.alphas
            labels = y
            used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
            if self.gamma:
                used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
                z = Gamma(used_k, 1 / used_theta).sample()
                z = (z - used_k * used_theta) / (1 - used_alphas).sqrt()
            else:
                z = torch.randn_like(cond)
            cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z
        return self.unet(x, y, cond)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvMeanPool,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InstanceNorm2dPlus,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanPoolConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Nin,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoneNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPADE,
     lambda: ([], {'norm_nc': 4, 'label_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Unit3D,
     lambda: ([], {'in_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VarianceNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_voletiv_mcvd_pytorch(_paritybench_base):
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

