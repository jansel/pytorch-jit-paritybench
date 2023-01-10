import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset_rand_seg_onlymap = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
single_dataset = _module
unaligned_dataset = _module
models = _module
base_model = _module
colorhistogram_model = _module
modules = _module
architecture = _module
block = _module
iccv_model = _module
sft_arch = _module
stdunet_woIN = _module
networks = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
util = _module
get_data = _module
html = _module
image_pool = _module
util = _module
visualizer = _module

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


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.transforms.functional as TF


import numpy


import time


from math import exp


import random


import torch.utils.data


import torch


import numpy as np


from collections import OrderedDict


from torch.autograd import Variable


import itertools


import torch.nn as nn


import torchvision


import torch.nn.functional as F


import copy


import matplotlib.pyplot as plt


import math


import functools


from torch.nn import init


from torch.optim import lr_scheduler


import inspect


import re


import collections


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, tensor=torch.FloatTensor):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = Variable(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
            std = Variable(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
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


class ConcatBlock(nn.Module):

    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class StdUnetDec_deep1(nn.Module):

    def __init__(self, input_nc, output_nc, use_tanh, norm_layer=nn.BatchNorm2d):
        super(StdUnetDec_deep1, self).__init__()
        self.up = self.build_up()
        self.block = self.build_block(input_nc, output_nc, use_tanh, norm_layer=norm_layer)

    def build_up(self):
        block_full = [nn.Upsample(scale_factor=2, mode='bilinear')]
        return nn.Sequential(*block_full)

    def build_block(self, input_nc, output_nc, use_tanh, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.Conv2d(input_nc, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*block_full)

    def forward(self, input_prev, input_skip, enc1, enc2, target_size1, target_size2):
        input_prev = self.up(input_prev)
        input_prev = F.upsample(input_prev, size=(target_size1, target_size2), mode='bilinear')
        enc1 = F.upsample(enc1, size=(target_size1, target_size2), mode='bilinear')
        enc2 = F.upsample(enc2, size=(target_size1, target_size2), mode='bilinear')
        input_skip = F.upsample(input_skip, size=(target_size1, target_size2), mode='bilinear')
        out = torch.cat([input_prev, input_skip, enc1, enc2], 1)
        out = self.block(out)
        return out


class StdUnetDec_deep2(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(StdUnetDec_deep2, self).__init__()
        self.up = self.build_up()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_up(self):
        block_full = [nn.Upsample(scale_factor=2, mode='bilinear')]
        return nn.Sequential(*block_full)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        return nn.Sequential(*block_full)

    def forward(self, input_prev, input_skip, enc1, enc2, target_size1, target_size2):
        input_prev = self.up(input_prev)
        input_prev = F.upsample(input_prev, size=(target_size1, target_size2), mode='bilinear')
        enc1 = F.upsample(enc1, size=(target_size1, target_size2), mode='bilinear')
        enc2 = F.upsample(enc2, size=(target_size1, target_size2), mode='bilinear')
        input2 = F.upsample(input_skip, size=(target_size1, target_size2), mode='bilinear')
        out = torch.cat([input_prev, input2, enc1, enc2], 1)
        out = self.block(out)
        return out


class StdUnetEnc_deep1(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep1, self).__init__()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class StdUnetEnc_deep2(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep2, self).__init__()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class StdUnetEnc_deep(nn.Module):

    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep, self).__init__()
        self.netEnc1 = StdUnetEnc_deep1(input_channel, ngf, norm_layer)
        self.netEnc2 = StdUnetEnc_deep2(ngf * 1, ngf * 2, norm_layer)
        self.netEnc3 = StdUnetEnc_deep2(ngf * 2, ngf * 4, norm_layer)
        self.netEnc4 = StdUnetEnc_deep2(ngf * 4, ngf * 8, norm_layer)
        self.netEnc5 = StdUnetEnc_deep2(ngf * 8, ngf * 8, norm_layer)

    def forward(self, input):
        output1 = self.netEnc1.forward(input)
        output2 = self.netEnc2.forward(output1)
        output3 = self.netEnc3.forward(output2)
        output4 = self.netEnc4.forward(output3)
        output5 = self.netEnc5.forward(output4)
        return output1, output2, output3, output4, output5


class StdUnet_woIN(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(StdUnet_woIN, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        enc_nc = 64
        self.block_init = self.build_init_block(input_nc, enc_nc, padding_type, use_bias)
        self.HistUnetEnc = StdUnetEnc_deep(input_nc, 64, norm_layer)
        self.HistUnetDec1 = StdUnetDec_deep2(512 + 512 + enc_nc + enc_nc, 512, norm_layer)
        self.HistUnetDec2 = StdUnetDec_deep2(512 + 512 + enc_nc + enc_nc, 256, norm_layer)
        self.HistUnetDec3 = StdUnetDec_deep2(256 + 256 + enc_nc + enc_nc, 128, norm_layer)
        self.HistUnetDec4 = StdUnetDec_deep2(128 + 128 + enc_nc + enc_nc, 64, norm_layer)
        self.HistUnetDec5 = StdUnetDec_deep1(64 + 64 + enc_nc + enc_nc, output_nc, norm_layer)
        self.InterOut1 = self.build_inter_out2(512, output_nc, 'zero', use_bias)
        self.InterOut2 = self.build_inter_out2(256, output_nc, 'zero', use_bias)
        self.InterOut3 = self.build_inter_out2(128, output_nc, 'zero', use_bias)
        self.InterOut4 = self.build_inter_out2(64, output_nc, 'zero', use_bias)
        self.block_last = self.build_last_block(64, output_nc, padding_type, use_bias)

    def build_init_block(self, input_nc, dim_img, padding_type, use_bias):
        block_init = []
        p = 0
        if padding_type == 'reflect':
            block_init += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_init += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_init += [nn.Conv2d(input_nc, dim_img, kernel_size=3, padding=p, stride=1), nn.InstanceNorm2d(dim_img), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, stride=1)]
        return nn.Sequential(*block_init)

    def build_last_block(self, dim_img, output_nc, padding_type, use_bias):
        block_last = []
        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_last += [nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, output_nc, kernel_size=3, padding=p, bias=use_bias)]
        return nn.Sequential(*block_last)

    def build_inter_out2(self, dim_img, dim_out, padding_type, use_bias):
        block_last = []
        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_last += [nn.Conv2d(dim_img, dim_img, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(dim_img), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(dim_img), nn.ReLU(True), nn.Conv2d(dim_img, dim_out, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*block_last)

    def forward(self, input_img, hist1_enc, hist2_enc):
        mid_img1, mid_img2, mid_img3, mid_img4, mid_img5 = self.HistUnetEnc(input_img)
        out_img2 = self.HistUnetDec2(mid_img5, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3))
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3))
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3))
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3))
        out_img5 = F.upsample(out_img5, size=(input_img.size(2), input_img.size(3)), mode='bilinear')
        out_img1 = out_img5
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)
        out_img = out_img5
        return out_img1, out_img2, out_img3, out_img4, out_img


class SFTLayer(nn.Module):

    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):

    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return x[0] + fea, x[1]


class SFT_Net(nn.Module):

    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)
        self.HR_branch = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True), nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1))
        self.CondNet = nn.Sequential(nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

    def forward(self, x):
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out


class ACD_VGG_BN_96(nn.Module):

    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512, affine=True), nn.LeakyReLU(0.1, True), nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512, affine=True), nn.LeakyReLU(0.1, True))
        self.gan = nn.Sequential(nn.Linear(512 * 6 * 6, 100), nn.LeakyReLU(0.1, True), nn.Linear(100, 1))
        self.cls = nn.Sequential(nn.Linear(512 * 6 * 6, 100), nn.LeakyReLU(0.1, True), nn.Linear(100, 8))

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]


class SFTLayer_torch(nn.Module):

    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT_torch(nn.Module):

    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return x[0] + fea, x[1]


class SFT_Net_torch(nn.Module):

    def __init__(self):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT_torch())
        sft_branch.append(SFTLayer_torch())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)
        self.HR_branch = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1))
        self.CondNet = nn.Sequential(nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

    def forward(self, x):
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out


class ConditionNetwork2(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        dim2 = 128
        model = [nn.Conv2d(self.input_nc, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias), nn.LeakyReLU(0.1, True), nn.Conv2d(dim2, self.output_nc, kernel_size=1, padding=0, bias=use_bias)]
        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(nn.Linear(output_nc, output_nc))
        self.model3 = nn.Sequential(nn.Linear(output_nc, 324))

    def forward(self, input):
        a1 = self.model(input)
        a2 = a1.view(a1.size(0), -1)
        a3 = self.model2(a2)
        a3 = a3.unsqueeze(0).unsqueeze(0).permute(2, 3, 0, 1)
        return a3


class ENCResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ENCResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim * 3), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim * 3, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x, enc1, enc2):
        enc1 = F.upsample(enc1, size=(x.size(2), x.size(3)), mode='bilinear')
        enc2 = F.upsample(enc2, size=(x.size(2), x.size(3)), mode='bilinear')
        x_cat = torch.cat((x, enc1, enc2), 1)
        out = x + self.conv_block(x_cat)
        return out


class UnetDec_deep1(nn.Module):

    def __init__(self, input_nc, output_nc, use_tanh, norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep1, self).__init__()
        self.block = self.build_block(input_nc, output_nc, use_tanh, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, use_tanh, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(input_nc, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.InstanceNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(64), nn.ReLU(True), nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):
        input1 = F.upsample(input1, size=(target_size1, target_size2), mode='bilinear')
        enc1 = F.upsample(enc1, size=(target_size1, target_size2), mode='bilinear')
        enc2 = F.upsample(enc2, size=(target_size1, target_size2), mode='bilinear')
        input2 = F.upsample(input2, size=(target_size1, target_size2), mode='bilinear')
        out = self.block(torch.cat([input1, input2, enc1, enc2], 1))
        return out


class UnetDec_deep2(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep2, self).__init__()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.InstanceNorm2d(output_nc), nn.ReLU(True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc), nn.ReLU(True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc)]
        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):
        input1 = F.upsample(input1, size=(target_size1, target_size2), mode='bilinear')
        enc1 = F.upsample(enc1, size=(target_size1, target_size2), mode='bilinear')
        enc2 = F.upsample(enc2, size=(target_size1, target_size2), mode='bilinear')
        input2 = F.upsample(input2, size=(target_size1, target_size2), mode='bilinear')
        out = self.block(torch.cat([input1, input2, enc1, enc2], 1))
        return out


class UnetEnc_deep1(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep1, self).__init__()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(output_nc)]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class UnetEnc_deep2(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep2, self).__init__()
        self.block = self.build_block(input_nc, output_nc, norm_layer=norm_layer)

    def build_block(self, input_nc, output_nc, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        block_full = [nn.LeakyReLU(0.2, True), nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(output_nc), nn.LeakyReLU(0.2, True), nn.Conv2d(output_nc, output_nc, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(output_nc)]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class UnetEnc_deep(nn.Module):

    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep, self).__init__()
        self.netEnc1 = UnetEnc_deep1(input_channel, ngf, norm_layer)
        self.netEnc2 = UnetEnc_deep2(ngf * 1, ngf * 2, norm_layer)
        self.netEnc3 = UnetEnc_deep2(ngf * 2, ngf * 4, norm_layer)
        self.netEnc4 = UnetEnc_deep2(ngf * 4, ngf * 8, norm_layer)
        self.netEnc5 = UnetEnc_deep2(ngf * 8, ngf * 8, norm_layer)

    def forward(self, input):
        output1 = self.netEnc1.forward(input)
        output2 = self.netEnc2.forward(output1)
        output3 = self.netEnc3.forward(output2)
        output4 = self.netEnc4.forward(output3)
        output5 = self.netEnc5.forward(output4)
        return output1, output2, output3, output4, output5


class HISTUnet3_Res(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(HISTUnet3_Res, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        enc_nc = 64
        self.block_init = self.build_init_block(input_nc, enc_nc, padding_type, use_bias)
        self.HistUnetEnc = UnetEnc_deep(input_nc, 64, norm_layer)
        self.HistUnetDec1 = UnetDec_deep2(512 + 512 + enc_nc + enc_nc, 512, norm_layer)
        self.HistUnetDec2 = UnetDec_deep2(512 + 512 + enc_nc + enc_nc, 256, norm_layer)
        self.HistUnetDec3 = UnetDec_deep2(256 + 256 + enc_nc + enc_nc, 128, norm_layer)
        self.HistUnetDec4 = UnetDec_deep2(128 + 128 + enc_nc + enc_nc, 64, norm_layer)
        self.HistUnetDec5 = UnetDec_deep1(64 + 64 + enc_nc + enc_nc, 64, norm_layer)
        self.ENC_Block1 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)
        self.ENC_Block2 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)
        self.ENC_Block3 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)
        self.InterOut1 = self.build_inter_out2(512, output_nc, 'zero', use_bias)
        self.InterOut2 = self.build_inter_out2(256, output_nc, 'zero', use_bias)
        self.InterOut3 = self.build_inter_out2(128, output_nc, 'zero', use_bias)
        self.InterOut4 = self.build_inter_out2(64, output_nc, 'zero', use_bias)
        self.block_last = self.build_last_block(64, output_nc, padding_type, use_bias)

    def build_init_block(self, input_nc, dim_img, padding_type, use_bias):
        block_init = []
        p = 0
        if padding_type == 'reflect':
            block_init += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_init += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_init += [nn.Conv2d(input_nc, dim_img, kernel_size=3, padding=p, stride=1), nn.InstanceNorm2d(dim_img), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, stride=1)]
        return nn.Sequential(*block_init)

    def build_last_block(self, dim_img, output_nc, padding_type, use_bias):
        block_last = []
        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_last += [nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True), nn.Conv2d(dim_img, output_nc, kernel_size=3, padding=p, bias=use_bias)]
        return nn.Sequential(*block_last)

    def build_inter_out(self, dim_img, dim_out, padding_type, use_bias):
        block_last = []
        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_last += [nn.Conv2d(dim_img, dim_out, kernel_size=3, padding=p, bias=use_bias)]
        return nn.Sequential(*block_last)

    def build_inter_out2(self, dim_img, dim_out, padding_type, use_bias):
        block_last = []
        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        block_last += [nn.ReLU(True), nn.Conv2d(dim_img, dim_img, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(dim_img), nn.ReLU(True), nn.Conv2d(dim_img, dim_out, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*block_last)

    def forward(self, input_img, hist1_enc, hist2_enc):
        mid_img1, mid_img2, mid_img3, mid_img4, mid_img5 = self.HistUnetEnc(input_img)
        out_img1 = self.HistUnetDec1(mid_img5, mid_img5, hist1_enc, hist2_enc, mid_img5.size(2), mid_img5.size(3))
        out_img2 = self.HistUnetDec2(out_img1, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3))
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3))
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3))
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3))
        out_img5 = F.upsample(out_img5, size=(input_img.size(2), input_img.size(3)), mode='bilinear')
        out_img6 = self.ENC_Block1.forward(out_img5 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img7 = self.ENC_Block2.forward(out_img6 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img = self.ENC_Block3.forward(out_img7 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img1 = self.InterOut1(out_img1)
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)
        out_img = self.block_last(out_img + self.block_init(input_img))
        return out_img1, out_img2, out_img3, out_img4, out_img


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConcatBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConditionNetwork2,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (ResNetBlock,
     lambda: ([], {'in_nc': 4, 'mid_nc': 4, 'out_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShortcutBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StdUnetEnc_deep,
     lambda: ([], {'input_channel': 4, 'ngf': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (StdUnetEnc_deep1,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StdUnetEnc_deep2,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetEnc_deep1,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetEnc_deep2,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGGFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_codeslake_Color_Transfer_Histogram_Analogy(_paritybench_base):
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

