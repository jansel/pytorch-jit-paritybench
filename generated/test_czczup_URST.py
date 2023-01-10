import sys
_module = sys.modules[__name__]
del sys
function = _module
model = _module
sampler = _module
test = _module
torch_to_pytorch = _module
train = _module
trainv2 = _module
model_original = _module
model_test = _module
rename = _module
test = _module
train = _module
utils = _module
Loader = _module
WCT = _module
modelsNIPS = _module
test = _module
util = _module
TestPhotoReal = _module
TestVideo = _module
Train = _module
TrainSPN = _module
Criterion = _module
Loader = _module
LoaderPhotoReal = _module
Matrix = _module
MatrixTest = _module
SPN = _module
libs = _module
models = _module
pytorch_spn = _module
_ext = _module
gaterecurrent2dnoind = _module
build = _module
functions = _module
gaterecurrent2dnoind = _module
left_right_demo = _module
modules = _module
gaterecurrent2dnoind = _module
smooth_filter = _module
utils = _module
test = _module
trainv2 = _module
WCT = _module
crop_center = _module
resize_img = _module
data_loader = _module
test = _module
util_wct = _module
utils = _module
data_loader = _module
main = _module
model = _module
model_cd = _module
model_kd2sd = _module
model_original = _module
convert_caffemodel_to_npy = _module
data_loader = _module
model = _module
normalise_caffe = _module
normalise_pth = _module
utils = _module
utils = _module
convert_original_mobilenet_to_mine = _module
copy_pth1_to_pth2 = _module
plot_loss = _module
prune = _module
utils = _module
model = _module
option = _module
test = _module
train = _module
utils = _module
thumb_instance_norm = _module
tools = _module

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


from torch.utils import data


from torchvision import transforms


import time


import math


from functools import reduce


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import torch.utils.data as data


from collections import namedtuple


from torchvision import models


import random


from torch.optim import Adam


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision.utils import save_image


import torchvision.transforms as transforms


import torchvision.utils as vutils


import torch.optim as optim


from torchvision.models import vgg16


from collections import OrderedDict


from torch.autograd import Function


from numpy.lib.stride_tricks import as_strided


from numpy import linalg


import collections


import matplotlib


import matplotlib.pyplot as plt


import torch.utils.data as Data


import torchvision


from torch.distributions.one_hot_categorical import OneHotCategorical


import torchvision.datasets as datasets


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class GramMatrix(nn.Module):

    def forward(self, y):
        b, ch, h, w = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C), X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.C) + ')'


class ThumbInstanceNorm(nn.Module):

    def __init__(self, out_channels=None, affine=True):
        super(ThumbInstanceNorm, self).__init__()
        self.thumb_mean = None
        self.thumb_std = None
        self.collection = True
        if affine == True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))

    def calc_mean_std(self, feat, eps=1e-05):
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, thumb=None):
        if self.training:
            thumb_mean, thumb_std = self.calc_mean_std(thumb)
            x = (x - thumb_mean) / thumb_std * self.weight + self.bias
            thumb = (thumb - thumb_mean) / thumb_std * self.weight + self.bias
            return x, thumb
        else:
            if self.collection:
                thumb_mean, thumb_std = self.calc_mean_std(x)
                self.thumb_mean = thumb_mean
                self.thumb_std = thumb_std
            x = (x - self.thumb_mean) / self.thumb_std * self.weight + self.bias
            return x


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpBottleneck(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion, kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class Net(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=ThumbInstanceNorm, n_blocks=6, gpu_ids=[]):
        super(Net, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()
        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4
        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1), norm_layer(64), nn.ReLU(inplace=True), block(64, 32, 2, 1, norm_layer), block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)
        model = []
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]
        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]
        model += [upblock(ngf * expansion, 32, 2, norm_layer), upblock(32 * expansion, 16, 2, norm_layer), norm_layer(16 * expansion), nn.ReLU(inplace=True), ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]
        self.model = nn.Sequential(*model)

    def setTarget(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class NetV2(nn.Module):

    def __init__(self, encoder, decoder):
        super(NetV2, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def thumb_adaptive_instance_normalization(self, content_thumb_feat, content_patch_feat, style_thumb_feat):
        size = content_thumb_feat.size()
        style_mean, style_std = calc_mean_std(style_thumb_feat)
        content_thumb_mean, content_thumb_std = calc_mean_std(content_thumb_feat)
        content_thumb_feat = (content_thumb_feat - content_thumb_mean.expand(size)) / content_thumb_std.expand(size)
        content_thumb_feat = content_thumb_feat * style_std.expand(size) + style_mean.expand(size)
        content_patch_feat = (content_patch_feat - content_thumb_mean.expand(size)) / content_thumb_std.expand(size)
        content_patch_feat = content_patch_feat * style_std.expand(size) + style_mean.expand(size)
        return content_thumb_feat, content_patch_feat

    def forward(self, content_patches, content_thumbs, style_thumbs, position, alpha=1.0):
        assert 0 <= alpha <= 1
        with torch.no_grad():
            style_thumb_feats = self.encode_with_intermediate(style_thumbs)
            content_thumb_feat = self.encode(content_thumbs)
            content_patch_feat = self.encode(content_patches)
            t_thumb, t_patch = self.thumb_adaptive_instance_normalization(content_thumb_feat, content_patch_feat, style_thumb_feats[-1])
            t_thumb = alpha * t_thumb + (1 - alpha) * content_thumb_feat
            t_patch = alpha * t_patch + (1 - alpha) * content_patch_feat
        g_t_thumb = self.decoder(t_thumb)
        g_t_patch = self.decoder(t_patch)
        with torch.no_grad():
            g_t_thumb_up = F.interpolate(g_t_thumb, scale_factor=2, mode='bilinear', align_corners=False)
            g_t_thumb_crop = g_t_thumb_up[..., position[0]:position[1], position[2]:position[3]]
            g_t_thumb_crop_feats = self.encode_with_intermediate(g_t_thumb_crop)
        g_t_thumb_feats = self.encode_with_intermediate(g_t_thumb)
        g_t_patch_feats = self.encode_with_intermediate(g_t_patch)
        loss_c = self.calc_content_loss(g_t_thumb_feats[-1], t_thumb)
        loss_s = self.calc_style_loss(g_t_thumb_feats[0], style_thumb_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_thumb_feats[i], style_thumb_feats[i])
        loss_sp = self.calc_content_loss(g_t_patch_feats[-1], g_t_thumb_crop_feats[-1])
        return loss_c, loss_s, loss_sp


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


class VGG16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        self.norm = ThumbInstanceNorm(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True)
        self.block2 = ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False)

    def forward(self, x):
        x_ = self.block1(x)
        x_ = self.block2(x_)
        x = x + x_
        return x


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvBlock(3, 32, kernel_size=9, stride=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = ConvBlock(128, 64, kernel_size=3, upsample=True)
        self.deconv2 = ConvBlock(64, 32, kernel_size=3, upsample=True)
        self.deconv3 = ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class encoder1(nn.Module):

    def __init__(self, vgg1):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(torch.Tensor(vgg1['modules'][0]['weight']))
        self.conv1.bias = torch.nn.Parameter(torch.Tensor(vgg1['modules'][0]['bias']))
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(torch.Tensor(vgg1['modules'][2]['weight']))
        self.conv2.bias = torch.nn.Parameter(torch.Tensor(vgg1['modules'][2]['bias']))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class decoder1(nn.Module):

    def __init__(self, d1):
        super(decoder1, self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(torch.Tensor(d1['modules'][1]['weight']))
        self.conv3.bias = torch.nn.Parameter(torch.Tensor(d1['modules'][1]['bias']))

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


class encoder2(nn.Module):

    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(torch.Tensor(vgg['modules'][0]['weight']))
        self.conv1.bias = torch.nn.Parameter(torch.Tensor(vgg['modules'][0]['bias']))
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(torch.Tensor(vgg['modules'][2]['weight']))
        self.conv2.bias = torch.nn.Parameter(torch.Tensor(vgg['modules'][2]['bias']))
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(torch.Tensor(vgg['modules'][5]['weight']))
        self.conv3.bias = torch.nn.Parameter(torch.Tensor(vgg['modules'][5]['bias']))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(torch.Tensor(vgg['modules'][9]['weight']))
        self.conv4.bias = torch.nn.Parameter(torch.Tensor(vgg['modules'][9]['bias']))
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out


class decoder2(nn.Module):

    def __init__(self, d):
        super(decoder2, self).__init__()
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(torch.Tensor(d['modules'][1]['weight']))
        self.conv5.bias = torch.nn.Parameter(torch.Tensor(d['modules'][1]['bias']))
        self.relu5 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(torch.Tensor(d['modules'][5]['weight']))
        self.conv6.bias = torch.nn.Parameter(torch.Tensor(d['modules'][5]['bias']))
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(torch.Tensor(d['modules'][8]['weight']))
        self.conv7.bias = torch.nn.Parameter(torch.Tensor(d['modules'][8]['bias']))

    def forward(self, x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out


class encoder3(nn.Module):

    def __init__(self):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out


class decoder3(nn.Module):

    def __init__(self):
        super(decoder3, self).__init__()
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        output = {}
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out


class encoder4(nn.Module):

    def __init__(self):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x, sF=None, matrix11=None, matrix21=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        output['r11'] = out
        out = self.reflecPad7(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        output['r21'] = out
        out = self.reflecPad7(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        output['r31'] = out
        if matrix31 is not None:
            feature3, transmatrix3 = matrix31(out, sF['r31'])
            out = self.reflecPad7(feature3)
        else:
            out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        output['r41'] = out
        return output


class decoder4(nn.Module):

    def __init__(self):
        super(decoder4, self).__init__()
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class encoder5(nn.Module):

    def __init__(self):
        super(encoder5, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self, x, sF=None, contentV256=None, styleV256=None, matrix11=None, matrix21=None, matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])
        out = self.conv3(out)
        output['r12'] = self.relu3(out)
        output['p1'] = self.maxPool(output['r12'])
        out = self.reflecPad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])
        out = self.conv5(out)
        output['r22'] = self.relu5(out)
        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if styleV256 is not None:
            feature = matrix31(output['r31'], sF['r31'], contentV256, styleV256)
            out = self.reflecPad7(feature)
        else:
            out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)
        out = self.reflecPad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)
        out = self.reflecPad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)
        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)
        out = self.reflecPad11(output['r41'])
        out = self.conv11(out)
        output['r42'] = self.relu11(out)
        out = self.reflecPad12(output['r42'])
        out = self.conv12(out)
        output['r43'] = self.relu12(out)
        out = self.reflecPad13(output['r43'])
        out = self.conv13(out)
        output['r44'] = self.relu13(out)
        output['p4'] = self.maxPool4(output['r44'])
        out = self.reflecPad14(output['p4'])
        out = self.conv14(out)
        output['r51'] = self.relu14(out)
        return output


class decoder5(nn.Module):

    def __init__(self):
        super(decoder5, self).__init__()
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)
        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


def load_param_from_t7(model, in_layer_index, out_layer):
    out_layer.weight = torch.nn.Parameter(torch.Tensor(model['modules'][in_layer_index]['weight']))
    out_layer.bias = torch.nn.Parameter(torch.Tensor(model['modules'][in_layer_index]['bias']))


load_param = load_param_from_t7


class Decoder1(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder1, self).__init__()
        self.fixed = fixed
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 1, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv11(self.pad(input)))
        return y

    def forward_branch(self, input):
        out11 = self.relu(self.conv11(self.pad(input)))
        return out11,


class Decoder2(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder2, self).__init__()
        self.fixed = fixed
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 1, self.conv21)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 8, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv21(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out21 = self.relu(self.conv21(self.pad(input)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out21, out11


class Decoder3(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder3, self).__init__()
        self.fixed = fixed
        self.conv31 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 1, self.conv31)
                load_param(t7_model, 5, self.conv22)
                load_param(t7_model, 8, self.conv21)
                load_param(t7_model, 12, self.conv12)
                load_param(t7_model, 15, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv31(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out31 = self.relu(self.conv31(self.pad(input)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out31, out21, out11


class Decoder4(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder4, self).__init__()
        self.fixed = fixed
        self.conv41 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv31 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 1, self.conv41)
                load_param(t7_model, 5, self.conv34)
                load_param(t7_model, 8, self.conv33)
                load_param(t7_model, 11, self.conv32)
                load_param(t7_model, 14, self.conv31)
                load_param(t7_model, 18, self.conv22)
                load_param(t7_model, 21, self.conv21)
                load_param(t7_model, 25, self.conv12)
                load_param(t7_model, 28, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv41(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_norule(self, input):
        y = self.relu(self.conv41(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.conv11(self.pad(y))
        return y

    def forward_branch(self, input):
        out41 = self.relu(self.conv41(self.pad(input)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out41, out31, out21, out11


class Decoder5(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Decoder5, self).__init__()
        self.fixed = fixed
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv44 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv41 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv31 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv21 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 1, self.conv51)
                load_param(t7_model, 5, self.conv44)
                load_param(t7_model, 8, self.conv43)
                load_param(t7_model, 11, self.conv42)
                load_param(t7_model, 14, self.conv41)
                load_param(t7_model, 18, self.conv34)
                load_param(t7_model, 21, self.conv33)
                load_param(t7_model, 24, self.conv32)
                load_param(t7_model, 27, self.conv31)
                load_param(t7_model, 31, self.conv22)
                load_param(t7_model, 34, self.conv21)
                load_param(t7_model, 38, self.conv12)
                load_param(t7_model, 41, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.relu(self.conv51(self.pad(input)))
        y = self.unpool(y)
        y = self.relu(self.conv44(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out51 = self.relu(self.conv51(self.pad(input)))
        out51 = self.unpool(out51)
        out44 = self.relu(self.conv44(self.pad(out51)))
        out43 = self.relu(self.conv43(self.pad(out44)))
        out42 = self.relu(self.conv42(self.pad(out43)))
        out41 = self.relu(self.conv41(self.pad(out42)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out51, out41, out31, out21, out11


EigenValueThre = 1e-100


class Encoder1(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder1, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        return out11,


class Encoder2(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder2, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 9, self.conv21)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        return out11, out21


class Encoder3(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder3, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 9, self.conv21)
                load_param(t7_model, 12, self.conv22)
                load_param(t7_model, 16, self.conv31)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        return out11, out21, out31


class Encoder4(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder4, self).__init__()
        self.fixed = fixed
        self.vgg = nn.Sequential(nn.Conv2d(3, 3, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU())
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 0, self.vgg[0])
                load_param(t7_model, 2, self.vgg[2])
                load_param(t7_model, 5, self.vgg[5])
                load_param(t7_model, 9, self.vgg[9])
                load_param(t7_model, 12, self.vgg[12])
                load_param(t7_model, 16, self.vgg[16])
                load_param(t7_model, 19, self.vgg[19])
                load_param(t7_model, 22, self.vgg[22])
                load_param(t7_model, 25, self.vgg[25])
                load_param(t7_model, 29, self.vgg[29])
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)


class Encoder5(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(Encoder5, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array([[[[0]], [[0]], [[255]]], [[[0]], [[255]], [[0]]], [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array([-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv34 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv44 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = torchfile.load(model)
                load_param(t7_model, 0, self.conv0)
                load_param(t7_model, 2, self.conv11)
                load_param(t7_model, 5, self.conv12)
                load_param(t7_model, 9, self.conv21)
                load_param(t7_model, 12, self.conv22)
                load_param(t7_model, 16, self.conv31)
                load_param(t7_model, 19, self.conv32)
                load_param(t7_model, 22, self.conv33)
                load_param(t7_model, 25, self.conv34)
                load_param(t7_model, 29, self.conv41)
                load_param(t7_model, 32, self.conv42)
                load_param(t7_model, 35, self.conv43)
                load_param(t7_model, 38, self.conv44)
                load_param(t7_model, 42, self.conv51)
            else:
                self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        y = self.conv0(input)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv34(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv41(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv44(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv51(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        return out11, out21, out31, out41, out51


class SmallDecoder1_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder1_16x, self).__init__()
        self.fixed = fixed
        self.conv11 = nn.Conv2d(24, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, input):
        out11 = self.conv11(self.pad(input))
        return out11


class SmallDecoder1_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder1_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv11 = nn.Conv2d(24, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_aux(self, x, relu=False):
        out11 = self.relu(self.conv11(self.pad(x)))
        return out11, out11


class SmallDecoder2_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder2_16x, self).__init__()
        self.fixed = fixed
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
        out21 = self.relu(self.conv21(self.pad(x)))
        out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.conv11(self.pad(out12))
        return out11


class SmallDecoder2_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder2_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.aux21 = nn.Conv2d(16, 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_aux(self, x, relu=False):
        out21 = self.relu(self.conv21(self.pad(x)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        if relu:
            out21_aux = self.relu(self.aux21(out21))
        else:
            out21_aux = self.aux21(out21)
        return out21_aux, out11


class SmallDecoder3_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder3_16x, self).__init__()
        self.fixed = fixed
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
        out31 = self.relu(self.conv31(self.pad(x)))
        out31 = self.unpool_pwct(out31, pool2_idx, output_size=pool2_size)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.conv11(self.pad(out12))
        return out11


class SmallDecoder3_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder3_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.aux31 = nn.Conv2d(32, 128, 1, 1, 0)
        self.aux21 = nn.Conv2d(16, 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_aux(self, x, relu=False):
        out31 = self.relu(self.conv31(self.pad(x)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        if relu:
            out31_aux = self.relu(self.aux31(out31))
            out21_aux = self.relu(self.aux21(out21))
        else:
            out31_aux = self.aux31(out31)
            out21_aux = self.aux21(out21)
        return out31_aux, out21_aux, out11


class SmallDecoder4_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder4_16x, self).__init__()
        self.fixed = fixed
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
        out41 = self.relu(self.conv41(self.pad(x)))
        out41 = self.unpool_pwct(out41, pool3_idx, output_size=pool3_size)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool_pwct(out31, pool2_idx, output_size=pool2_size)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.conv11(self.pad(out12))
        return out11


class SmallDecoder4_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder4_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.aux41 = nn.Conv2d(64, 256, 1, 1, 0)
        self.aux31 = nn.Conv2d(32, 128, 1, 1, 0)
        self.aux21 = nn.Conv2d(16, 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_aux(self, x, relu=False):
        out41 = self.relu(self.conv41(self.pad(x)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        if relu:
            out41_aux = self.relu(self.aux41(out41))
            out31_aux = self.relu(self.aux31(out31))
            out21_aux = self.relu(self.aux21(out21))
        else:
            out41_aux = self.aux41(out41)
            out31_aux = self.aux31(out31)
            out21_aux = self.aux21(out21)
        return out41_aux, out31_aux, out21_aux, out11


class SmallDecoder5_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder5_16x, self).__init__()
        self.fixed = fixed
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv51(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv44(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out51 = self.relu(self.conv51(self.pad(input)))
        out51 = self.unpool(out51)
        out44 = self.relu(self.conv44(self.pad(out51)))
        out43 = self.relu(self.conv43(self.pad(out44)))
        out42 = self.relu(self.conv42(self.pad(out43)))
        out41 = self.relu(self.conv41(self.pad(out42)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        return out11


class SmallDecoder5_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder5_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.aux51 = nn.Conv2d(128, 512, 1, 1, 0)
        self.aux41 = nn.Conv2d(64, 256, 1, 1, 0)
        self.aux31 = nn.Conv2d(32, 128, 1, 1, 0)
        self.aux21 = nn.Conv2d(16, 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv51(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv44(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_aux(self, x, relu=False):
        out51 = self.relu(self.conv51(self.pad(x)))
        out51 = self.unpool(out51)
        out44 = self.relu(self.conv44(self.pad(out51)))
        out43 = self.relu(self.conv43(self.pad(out44)))
        out42 = self.relu(self.conv42(self.pad(out43)))
        out41 = self.relu(self.conv41(self.pad(out42)))
        out41 = self.unpool(out41)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool(out31)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool(out21)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.relu(self.conv11(self.pad(out12)))
        if relu:
            out51_aux = self.relu(self.aux51(out51))
            out41_aux = self.relu(self.aux41(out41))
            out31_aux = self.relu(self.aux31(out31))
            out21_aux = self.relu(self.aux21(out21))
        else:
            out51_aux = self.aux51(out51)
            out41_aux = self.aux41(out41)
            out31_aux = self.aux31(out31)
            out21_aux = self.aux21(out21)
        return out51_aux, out41_aux, out31_aux, out21_aux, out11


class SmallEncoder1_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder1_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 24, 3, 1, 0, dilation=1)
        self.conv11_aux = nn.Conv2d(24, 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        return out11,

    def forward_aux(self, input, relu=True):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        if relu:
            out11_aux = self.relu(self.conv11_aux(out11))
        else:
            out11_aux = self.conv11_aux(out11)
        return out11_aux,

    def forward_aux2(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out11_aux = self.relu(self.conv11_aux(out11))
        return out11_aux, out11


class SmallEncoder2_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder2_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv11_aux = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv21_aux = nn.Conv2d(32, 128, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        return out11, out21

    def forward_aux(self, input, relu=True):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        if relu:
            out11_aux = self.relu(self.conv11_aux(out11))
            out21_aux = self.relu(self.conv21_aux(out21))
        else:
            out11_aux = self.conv11_aux(out11)
            out21_aux = self.conv21_aux(out21)
        return out11_aux, out21_aux

    def forward_aux2(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out11_aux = self.relu(self.conv11_aux(out11))
        out21_aux = self.relu(self.conv21_aux(out21))
        return out11_aux, out21_aux, out21

    def forward_pwct(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        pool12, out12_ix = self.pool2(out12)
        out21 = self.relu(self.conv21(self.pad(pool12)))
        return out21, out12_ix, out12.size()


class SmallEncoder3_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder3_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv11_aux = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv21_aux = nn.Conv2d(32, 128, 1, 1, 0)
        self.conv31_aux = nn.Conv2d(64, 256, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        return out11, out21, out31

    def forward_aux(self, input, relu=True):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        if relu:
            out11_aux = self.relu(self.conv11_aux(out11))
            out21_aux = self.relu(self.conv21_aux(out21))
            out31_aux = self.relu(self.conv31_aux(out31))
        else:
            out11_aux = self.conv11_aux(out11)
            out21_aux = self.conv21_aux(out21)
            out31_aux = self.conv31_aux(out31)
        return out11_aux, out21_aux, out31_aux

    def forward_aux2(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out11_aux = self.relu(self.conv11_aux(out11))
        out21_aux = self.relu(self.conv21_aux(out21))
        out31_aux = self.relu(self.conv31_aux(out31))
        return out11_aux, out21_aux, out31_aux, out31

    def forward_pwct(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        pool12, out12_ix = self.pool2(out12)
        out21 = self.relu(self.conv21(self.pad(pool12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        pool22, out22_ix = self.pool2(out22)
        out31 = self.relu(self.conv31(self.pad(pool22)))
        return out31, out12_ix, out12.size(), out22_ix, out22.size()


class SmallEncoder4_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder4_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv11_aux = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv21_aux = nn.Conv2d(32, 128, 1, 1, 0)
        self.conv31_aux = nn.Conv2d(64, 256, 1, 1, 0)
        self.conv41_aux = nn.Conv2d(128, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv34(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv41(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        return out11, out21, out31, out41

    def forward_pwct(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        pool12, out12_ix = self.pool2(out12)
        out21 = self.relu(self.conv21(self.pad(pool12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        pool22, out22_ix = self.pool2(out22)
        out31 = self.relu(self.conv31(self.pad(pool22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        pool34, out34_ix = self.pool2(out34)
        out41 = self.relu(self.conv41(self.pad(pool34)))
        return out41, out12_ix, out12.size(), out22_ix, out22.size(), out34_ix, out34.size()

    def forward_aux(self, input, relu=True):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        if relu:
            out11_aux = self.relu(self.conv11_aux(out11))
            out21_aux = self.relu(self.conv21_aux(out21))
            out31_aux = self.relu(self.conv31_aux(out31))
            out41_aux = self.relu(self.conv41_aux(out41))
        else:
            out11_aux = self.conv11_aux(out11)
            out21_aux = self.conv21_aux(out21)
            out31_aux = self.conv31_aux(out31)
            out41_aux = self.conv41_aux(out41)
        return out11_aux, out21_aux, out31_aux, out41_aux

    def forward_aux2(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out11_aux = self.relu(self.conv11_aux(out11))
        out21_aux = self.relu(self.conv21_aux(out21))
        out31_aux = self.relu(self.conv31_aux(out31))
        out41_aux = self.relu(self.conv41_aux(out41))
        return out11_aux, out21_aux, out31_aux, out41_aux, out41


class SmallEncoder5_16x_aux(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder5_16x_aux, self).__init__()
        self.fixed = fixed
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv11_aux = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv21_aux = nn.Conv2d(32, 128, 1, 1, 0)
        self.conv31_aux = nn.Conv2d(64, 256, 1, 1, 0)
        self.conv41_aux = nn.Conv2d(128, 512, 1, 1, 0)
        self.conv51_aux = nn.Conv2d(128, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv34(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv41(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv44(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv51(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        return out11, out21, out31, out41, out51

    def forward_aux(self, input, relu=True):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        if relu:
            out11_aux = self.relu(self.conv11_aux(out11))
            out21_aux = self.relu(self.conv21_aux(out21))
            out31_aux = self.relu(self.conv31_aux(out31))
            out41_aux = self.relu(self.conv41_aux(out41))
            out51_aux = self.relu(self.conv51_aux(out51))
        else:
            out11_aux = self.conv11_aux(out11)
            out21_aux = self.conv21_aux(out21)
            out31_aux = self.conv31_aux(out31)
            out41_aux = self.conv41_aux(out41)
            out51_aux = self.conv51_aux(out51)
        return out11_aux, out21_aux, out31_aux, out41_aux, out51_aux

    def forward_aux2(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        out11_aux = self.relu(self.conv11_aux(out11))
        out21_aux = self.relu(self.conv21_aux(out21))
        out31_aux = self.relu(self.conv31_aux(out31))
        out41_aux = self.relu(self.conv41_aux(out41))
        out51_aux = self.relu(self.conv51_aux(out51))
        return out11_aux, out21_aux, out31_aux, out41_aux, out51_aux, out51

    def forward_aux3(self, input, relu=False):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        if relu:
            out51_aux = self.relu(self.conv51_aux(out51))
        else:
            out51_aux = self.conv51_aux(out51)
        return out11, out21, out31, out41, out51, out51_aux


class ThumbWhitenColorTransform(ThumbInstanceNorm):

    def __init__(self):
        super(ThumbWhitenColorTransform, self).__init__(affine=False)
        self.thumb_mean = None
        self.style_mean = None
        self.trans_matrix = None

    def forward(self, cF, sF, wct_mode):
        if self.collection:
            cFSize = cF.size()
            c_mean = torch.mean(cF, 1)
            c_mean = c_mean.unsqueeze(1)
            self.thumb_mean = c_mean
        cF = cF - self.thumb_mean
        if self.collection:
            if wct_mode == 'cpu':
                contentCov = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
            else:
                contentCov = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
            c_u, c_e, c_v = torch.svd(contentCov, some=False)
            k_c = cFSize[0]
            for i in range(cFSize[0]):
                if c_e[i] < 1e-05:
                    k_c = i
                    break
            c_d = c_e[0:k_c].pow(-0.5)
            step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
            step2 = torch.mm(step1, c_v[:, 0:k_c].t())
            thumb_cov = step2
            sFSize = sF.size()
            s_mean = torch.mean(sF, 1)
            self.style_mean = s_mean.unsqueeze(1)
            sF = sF - s_mean.unsqueeze(1).expand_as(sF)
            styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
            s_u, s_e, s_v = torch.svd(styleConv, some=False)
            k_s = sFSize[0]
            for i in range(sFSize[0]):
                if s_e[i] < 1e-05:
                    k_s = i
                    break
            s_d = s_e[0:k_s].pow(0.5)
            style_cov = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), s_v[:, 0:k_s].t())
            self.trans_matrix = torch.mm(style_cov, thumb_cov)
        targetFeature = torch.mm(self.trans_matrix, cF)
        targetFeature = targetFeature + self.style_mean.expand_as(targetFeature)
        return targetFeature


class WCT(nn.Module):

    def __init__(self, args):
        super(WCT, self).__init__()
        self.args = args
        if args.mode == None or args.mode == 'original':
            self.e1 = Encoder1(args.e1)
            self.d1 = Decoder1(args.d1)
            self.e2 = Encoder2(args.e2)
            self.d2 = Decoder2(args.d2)
            self.e3 = Encoder3(args.e3)
            self.d3 = Decoder3(args.d3)
            self.e4 = Encoder4(args.e4)
            self.d4 = Decoder4(args.d4)
            self.e5 = Encoder5(args.e5)
            self.d5 = Decoder5(args.d5)
        elif args.mode == '16x':
            self.e5 = SmallEncoder5_16x_aux(args.e5)
            self.d5 = SmallDecoder5_16x(args.d5)
            self.e4 = SmallEncoder4_16x_aux(args.e4)
            self.d4 = SmallDecoder4_16x(args.d4)
            self.e3 = SmallEncoder3_16x_aux(args.e3)
            self.d3 = SmallDecoder3_16x(args.d3)
            self.e2 = SmallEncoder2_16x_aux(args.e2)
            self.d2 = SmallDecoder2_16x(args.d2)
            self.e1 = SmallEncoder1_16x_aux(args.e1)
            self.d1 = SmallDecoder1_16x(args.d1)
        elif args.mode == '16x_kd2sd':
            self.e5 = SmallEncoder5_16x_aux(args.e5)
            self.d5 = SmallDecoder5_16x_aux(args.d5)
            self.e4 = SmallEncoder4_16x_aux(args.e4)
            self.d4 = SmallDecoder4_16x_aux(args.d4)
            self.e3 = SmallEncoder3_16x_aux(args.e3)
            self.d3 = SmallDecoder3_16x_aux(args.d3)
            self.e2 = SmallEncoder2_16x_aux(args.e2)
            self.d2 = SmallDecoder2_16x_aux(args.d2)
            self.e1 = SmallEncoder1_16x_aux(args.e1)
            self.d1 = SmallDecoder1_16x_aux(args.d1)
        else:
            None
            exit(1)
        self.wct1 = ThumbWhitenColorTransform()
        self.wct2 = ThumbWhitenColorTransform()
        self.wct3 = ThumbWhitenColorTransform()
        self.wct4 = ThumbWhitenColorTransform()
        self.wct5 = ThumbWhitenColorTransform()

    def whiten_and_color_torch(self, cF, sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1).unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1)
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < EigenValueThre:
                k_c = i
                break
        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < EigenValueThre:
                k_s = i
                break
        c_d = c_e[0:k_c].pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, c_v[:, 0:k_c].t())
        whiten_cF = torch.mm(step2, cF)
        s_d = s_e[0:k_s].pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), s_v[:, 0:k_s].t()), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def whiten_and_color_np(self, cF, sF):
        cF = cF.data.cpu().numpy()
        cFSize = cF.shape
        c_mean = np.repeat(np.mean(cF, 1), cFSize[1], axis=0).reshape(cFSize)
        cF = cF - c_mean
        contentConv = np.divide(np.matmul(cF, np.transpose(cF)), cFSize[1] - 1) + np.eye(cFSize[0])
        c_u, c_e, c_v = linalg.svd(contentConv)
        c_v = np.transpose(c_v)
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < EigenValueThre:
                k_c = i
                break
        sF = sF.data.cpu().numpy()
        sFSize = sF.shape
        s_mean = np.mean(sF, 1)
        sF = sF - np.repeat(s_mean, sFSize[1], axis=0).reshape(sFSize)
        styleConv = np.divide(np.matmul(sF, np.transpose(sF)), sFSize[1] - 1)
        s_u, s_e, s_v = linalg.svd(styleConv)
        s_v = np.transpose(s_v)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < EigenValueThre:
                k_s = i
                break
        c_d = pow(c_e[0:k_c], -0.5)
        step1 = np.matmul(c_v[:, 0:k_c], np.diag(c_d))
        step2 = np.matmul(step1, np.transpose(c_v[:, 0:k_c]))
        whiten_cF = np.matmul(step2, cF)
        s_d = pow(s_e[0:k_s], 0.5)
        targetFeature = np.matmul(np.matmul(np.matmul(s_v[:, 0:k_s], np.diag(s_d)), np.transpose(s_v[:, 0:k_s])), whiten_cF)
        targetFeature = targetFeature + np.repeat(s_mean, cFSize[1], axis=0).reshape(cFSize)
        return torch.from_numpy(targetFeature)

    def whiten_and_color(self, cF, sF):
        if self.args.numpy:
            return self.whiten_and_color_np(cF, sF)
        else:
            return self.whiten_and_color_torch(cF, sF)

    def transform(self, cF, sF, alpha):
        cF = cF.double()
        sF = sF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        torch.cuda.empty_cache()
        return csF

    def transform_v2(self, cF, sF, alpha=1.0, index=0, wct_mode='cpu'):
        cF = cF.double()
        sF = sF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        if index == 1:
            targetFeature = self.wct1(cFView, sFView, wct_mode)
        elif index == 2:
            targetFeature = self.wct2(cFView, sFView, wct_mode)
        elif index == 3:
            targetFeature = self.wct3(cFView, sFView, wct_mode)
        elif index == 4:
            targetFeature = self.wct4(cFView, sFView, wct_mode)
        elif index == 5:
            targetFeature = self.wct5(cFView, sFView, wct_mode)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        return csF


class styleLoss(nn.Module):

    def forward(self, input, target):
        ib, ic, ih, iw = input.size()
        iF = input.view(ib, ic, -1)
        iMean = torch.mean(iF, dim=2)
        iCov = GramMatrix()(input)
        tb, tc, th, tw = target.size()
        tF = target.view(tb, tc, -1)
        tMean = torch.mean(tF, dim=2)
        tCov = GramMatrix()(target)
        loss = nn.MSELoss(size_average=False)(iMean, tMean) + nn.MSELoss(size_average=False)(iCov, tCov)
        return loss / tb


class LossCriterion(nn.Module):

    def __init__(self, style_layers, content_layers, style_weight, content_weight, sp_weight=None):
        super(LossCriterion, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.sp_weight = sp_weight
        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self, tF, sF, cF):
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, cf_i)
        totalContentLoss = totalContentLoss * self.content_weight
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i, sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight
        loss = totalStyleLoss + totalContentLoss
        return loss, totalStyleLoss, totalContentLoss

    def forwardv2(self, tF, sF, ttF, tpF, ttpF):
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            tf_i = tF[layer]
            tf_i = tf_i.detach()
            ttf_i = ttF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, ttf_i)
        totalContentLoss = totalContentLoss * self.content_weight
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            ttf_i = ttF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(ttf_i, sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight
        totalSPLoss = 0
        for i, layer in enumerate(self.content_layers):
            ttpf_i = ttpF[layer]
            ttpf_i = ttpf_i.detach()
            tpf_i = tpF[layer]
            loss_i = self.contentLosses[i]
            totalSPLoss += loss_i(tpf_i, ttpf_i)
        totalSPLoss = totalSPLoss * self.sp_weight
        loss = totalStyleLoss + totalContentLoss + totalSPLoss
        return loss, totalStyleLoss, totalContentLoss, totalSPLoss


class CNN(nn.Module):

    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        if layer == 'r31':
            self.convs = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, matrixSize, 3, 1, 1))
        elif layer == 'r41':
            self.convs = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, matrixSize, 3, 1, 1))
        self.fc = nn.Linear(32 * 32, 32 * 32)

    def forward(self, x, masks, style=False):
        color_code_number = 9
        xb, xc, xh, xw = x.size()
        x = x.view(xc, -1)
        feature_sub_mean = x.clone()
        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(), (xw, xh), interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if torch.sum(mask) >= 10:
                mask = mask.view(-1)
                """
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = mask.cpu().numpy()
                mask = cv2.dilate(mask.astype(np.float32), kernel)
                mask = torch.from_numpy(mask)
                mask = mask.squeeze()
                """
                fgmask = (mask > 0).nonzero().squeeze(1)
                fgmask = fgmask
                selectFeature = torch.index_select(x, 1, fgmask)
                f_mean = torch.mean(selectFeature, 1)
                f_mean = f_mean.unsqueeze(1).expand_as(selectFeature)
                selectFeature = selectFeature - f_mean
                feature_sub_mean.index_copy_(1, fgmask, selectFeature)
        feature = self.convs(feature_sub_mean.view(xb, xc, xh, xw))
        b, c, h, w = feature.size()
        transMatrices = {}
        feature = feature.view(c, -1)
        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if torch.sum(mask) >= 10:
                mask = mask.view(-1)
                fgmask = Variable((mask == 1).nonzero().squeeze(1))
                fgmask = fgmask
                selectFeature = torch.index_select(feature, 1, fgmask)
                tc, tN = selectFeature.size()
                covMatrix = torch.mm(selectFeature, selectFeature.transpose(0, 1)).div(tN)
                transmatrix = self.fc(covMatrix.view(-1))
                transMatrices[i] = transmatrix
        return transMatrices, feature_sub_mean


class MulLayer(nn.Module):

    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(layer)
        self.cnet = CNN(layer)
        self.matrixSize = matrixSize
        if layer == 'r41':
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        elif layer == 'r31':
            self.compress = nn.Conv2d(256, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 256, 1, 1, 0)

    def forward(self, cF, sF, cmasks, smasks):
        sb, sc, sh, sw = sF.size()
        sMatrices, sF_sub_mean = self.snet(sF, smasks, style=True)
        cMatrices, cF_sub_mean = self.cnet(cF, cmasks, style=False)
        compress_content = self.compress(cF_sub_mean.view(cF.size()))
        cb, cc, ch, cw = compress_content.size()
        compress_content = compress_content.view(cc, -1)
        transfeature = compress_content.clone()
        color_code_number = 9
        finalSMean = Variable(torch.zeros(cF.size()))
        finalSMean = finalSMean.view(sc, -1)
        for i in range(color_code_number):
            cmask = cmasks[i].clone().squeeze(0)
            smask = smasks[i].clone().squeeze(0)
            cmask = cv2.resize(cmask.numpy(), (cw, ch), interpolation=cv2.INTER_NEAREST)
            cmask = torch.FloatTensor(cmask)
            cmask = cmask.long()
            smask = cv2.resize(smask.numpy(), (sw, sh), interpolation=cv2.INTER_NEAREST)
            smask = torch.FloatTensor(smask)
            smask = smask.long()
            if torch.sum(cmask) >= 10 and torch.sum(smask) >= 10 and i in sMatrices and i in cMatrices:
                cmask = cmask.view(-1)
                fgcmask = Variable((cmask == 1).nonzero().squeeze(1))
                fgcmask = fgcmask
                smask = smask.view(-1)
                fgsmask = Variable((smask == 1).nonzero().squeeze(1))
                fgsmask = fgsmask
                sFF = sF.view(sc, -1)
                sFF_select = torch.index_select(sFF, 1, fgsmask)
                sMean = torch.mean(sFF_select, dim=1, keepdim=True)
                sMean = sMean.view(1, sc, 1, 1)
                sMean = sMean.expand_as(cF)
                sMatrix = sMatrices[i]
                cMatrix = cMatrices[i]
                sMatrix = sMatrix.view(self.matrixSize, self.matrixSize)
                cMatrix = cMatrix.view(self.matrixSize, self.matrixSize)
                transmatrix = torch.mm(sMatrix, cMatrix)
                compress_content_select = torch.index_select(compress_content, 1, fgcmask)
                transfeatureFG = torch.mm(transmatrix, compress_content_select)
                transfeature.index_copy_(1, fgcmask, transfeatureFG)
                sMean = sMean.contiguous()
                sMean_select = torch.index_select(sMean.view(sc, -1), 1, fgcmask)
                finalSMean.index_copy_(1, fgcmask, sMean_select)
        out = self.unzip(transfeature.view(cb, cc, ch, cw))
        return out + finalSMean.view(out.size())


class GateRecurrent2dnoindFunction(Function):

    def __init__(self, horizontal_, reverse_):
        self.horizontal = horizontal_
        self.reverse = reverse_

    def forward(self, X, G1, G2, G3):
        num, channels, height, width = X.size()
        output = torch.zeros(num, channels, height, width)
        if not X.is_cuda:
            None
            return 0
        else:
            output = output
            gaterecurrent2d.gaterecurrent2dnoind_forward_cuda(self.horizontal, self.reverse, X, G1, G2, G3, output)
            self.X = X
            self.G1 = G1
            self.G2 = G2
            self.G3 = G3
            self.output = output
            self.hiddensize = X.size()
            return output

    def backward(self, grad_output):
        num, channels, height, width = self.hiddensize
        grad_X = torch.zeros(num, channels, height, width)
        grad_G1 = torch.zeros(num, channels, height, width)
        grad_G2 = torch.zeros(num, channels, height, width)
        grad_G3 = torch.zeros(num, channels, height, width)
        gaterecurrent2d.gaterecurrent2dnoind_backward_cuda(self.horizontal, self.reverse, self.output, grad_output, self.X, self.G1, self.G2, self.G3, grad_X, grad_G1, grad_G2, grad_G3)
        del self.hiddensize
        del self.G1
        del self.G2
        del self.G3
        del self.output
        del self.X
        return grad_X, grad_G1, grad_G2, grad_G3


class GateRecurrent2dnoind(nn.Module):
    """docstring for ."""

    def __init__(self, horizontal_, reverse_):
        super(GateRecurrent2dnoind, self).__init__()
        self.horizontal = horizontal_
        self.reverse = reverse_

    def forward(self, X, G1, G2, G3):
        return GateRecurrent2dnoindFunction(self.horizontal, self.reverse)(X, G1, G2, G3)


class spn_block(nn.Module):

    def __init__(self, horizontal, reverse):
        super(spn_block, self).__init__()
        self.propagator = GateRecurrent2dnoind(horizontal, reverse)

    def forward(self, x, G1, G2, G3):
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        sum_abs.data[sum_abs.data == 0] = 1e-06
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)
        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm
        return self.propagator(x, G1, G2, G3)


class VGG(nn.Module):

    def __init__(self, nf):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(nf, nf * 2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 3, padding=1)

    def forward(self, x):
        output = {}
        output['conv1'] = self.conv1(x)
        x = F.relu(output['conv1'])
        x = self.pool1(x)
        output['conv2'] = self.conv2(x)
        x = F.relu(output['conv2'])
        x = self.pool2(x)
        output['conv3'] = self.conv3(x)
        x = F.relu(output['conv3'])
        output['pool3'] = self.pool3(x)
        output['conv4'] = self.conv4(output['pool3'])
        return output


class Decoder(nn.Module):

    def __init__(self, nf=32, spn=1):
        super(Decoder, self).__init__()
        self.layer0 = nn.Conv2d(nf * 8, nf * 4, 1, 1, 0)
        self.layer1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer2 = nn.Sequential(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), nn.ELU(inplace=True))
        self.layer3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer4 = nn.Sequential(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1), nn.ELU(inplace=True))
        self.layer5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer6 = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1), nn.ELU(inplace=True))
        if spn == 1:
            self.layer7 = nn.Conv2d(nf, nf * 12, 3, 1, 1)
        else:
            self.layer7 = nn.Conv2d(nf, nf * 24, 3, 1, 1)
        self.spn = spn

    def forward(self, encode_feature):
        output = {}
        output['0'] = self.layer0(encode_feature['conv4'])
        output['1'] = self.layer1(output['0'])
        output['2'] = self.layer2(output['1'])
        output['2res'] = output['2'] + encode_feature['conv3']
        output['3'] = self.layer3(output['2res'])
        output['4'] = self.layer4(output['3'])
        output['4res'] = output['4'] + encode_feature['conv2']
        output['5'] = self.layer5(output['4res'])
        output['6'] = self.layer6(output['5'])
        output['6res'] = output['6'] + encode_feature['conv1']
        output['7'] = self.layer7(output['6res'])
        return output['7']


class SPN(nn.Module):

    def __init__(self, nf=32, spn=1):
        super(SPN, self).__init__()
        self.mask_conv = nn.Conv2d(3, nf, 3, 1, 1)
        self.encoder = VGG(nf)
        self.decoder = Decoder(nf, spn)
        self.left_right = spn_block(True, False)
        self.right_left = spn_block(True, True)
        self.top_down = spn_block(False, False)
        self.down_top = spn_block(False, True)
        self.post = nn.Conv2d(nf, 3, 3, 1, 1)
        self.nf = nf

    def forward(self, x, rgb):
        X = self.mask_conv(x)
        features = self.encoder(rgb)
        guide = self.decoder(features)
        G = torch.split(guide, self.nf, 1)
        out1 = self.left_right(X, G[0], G[1], G[2])
        out2 = self.right_left(X, G[3], G[4], G[5])
        out3 = self.top_down(X, G[6], G[7], G[8])
        out4 = self.down_top(X, G[9], G[10], G[11])
        out = torch.max(out1, out2)
        out = torch.max(out, out3)
        out = torch.max(out, out4)
        return self.post(out)


class TrainSE_With_WCTDecoder(nn.Module):

    def __init__(self, args):
        super(TrainSE_With_WCTDecoder, self).__init__()
        self.BE = eval('model_original.Encoder%d' % args.stage)(args.BE, fixed=True)
        self.BD = eval('model_original.Decoder%d' % args.stage)(args.BD, fixed=True)
        self.SE = eval('model_cd.SmallEncoder%d_%dx_aux' % (args.stage, args.speedup))(args.SE, fixed=False)
        self.args = args

    def forward(self, c, iter):
        cF_BE = self.BE.forward_branch(c)
        cF_SE = self.SE.forward_aux(c, self.args.updim_relu)
        rec = self.BD(cF_SE[-1])
        sd_BE = 0
        if iter % self.args.save_interval == 0:
            rec_BE = self.BD(cF_BE[-1])
        feat_loss = 0
        for i in range(len(cF_BE)):
            feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
        rec_pixl_loss = nn.MSELoss()(rec, c.data)
        recF_BE = self.BE.forward_branch(rec)
        rec_perc_loss = 0
        for i in range(len(recF_BE)):
            rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
        return feat_loss, rec_pixl_loss, rec_perc_loss, rec, c


class TrainSD_With_WCTSE(nn.Module):

    def __init__(self, args):
        super(TrainSD_With_WCTSE, self).__init__()
        self.BE = eval('model_original.Encoder%d' % args.stage)(args.BE, fixed=True)
        self.SE = eval('model_cd.SmallEncoder%d_%dx_aux' % (args.stage, args.speedup))(args.SE, fixed=True)
        self.SD = eval('model_cd.SmallDecoder%d_%dx' % (args.stage, args.speedup))(args.SD, fixed=False)
        self.args = args

    def forward(self, c, iter):
        rec = self.SD(self.SE(c))
        rec_pixl_loss = nn.MSELoss()(rec, c.data)
        recF_BE = self.BE.forward_branch(rec)
        cF_BE = self.BE.forward_branch(c)
        rec_perc_loss = 0
        for i in range(len(recF_BE)):
            rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
        return rec_pixl_loss, rec_perc_loss, rec


class TrainSD_With_WCTSE_KD2SD(nn.Module):

    def __init__(self, args):
        super(TrainSD_With_WCTSE_KD2SD, self).__init__()
        self.BE = eval('model_original.Encoder%d' % args.stage)(args.BE, fixed=True)
        self.BD = eval('model_original.Decoder%d' % args.stage)(None, fixed=True)
        self.SE = eval('model_cd.SmallEncoder%d_%dx_aux' % (args.stage, args.speedup))(None, fixed=True)
        self.SD = eval('model_cd.SmallDecoder%d_%dx_aux' % (args.stage, args.speedup))(args.SD, fixed=False)
        self.args = args

    def forward(self, c, iter):
        feats_BE = self.BE.forward_branch(c)
        *_, feat_SE_aux, feat_SE = self.SE.forward_aux2(c)
        feats_BD = self.BD.forward_branch(feat_SE_aux)
        feats_SD = self.SD.forward_aux(feat_SE, relu=self.args.updim_relu)
        rec = feats_SD[-1]
        rec_pixl_loss = nn.MSELoss()(rec, c.data)
        rec_feats_BE = self.BE.forward_branch(rec)
        rec_perc_loss = 0
        for i in range(len(rec_feats_BE)):
            rec_perc_loss += nn.MSELoss()(rec_feats_BE[i], feats_BE[i].data)
        kd_feat_loss = 0
        for i in range(len(feats_BD)):
            kd_feat_loss += nn.MSELoss()(feats_SD[i], feats_BD[i].data)
        return rec_pixl_loss, rec_perc_loss, kd_feat_loss, rec


class SmallEncoder4_2(nn.Module):

    def __init__(self, model=None):
        super(SmallEncoder4_2, self).__init__()
        self.vgg = nn.Sequential(nn.Conv2d(3, 3, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 16, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(16, 16, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(32, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(32, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 512, (3, 3)), nn.ReLU())
        if model:
            self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage)['model'])

    def forward(self, x):
        return self.vgg(x)


class SmallEncoder4_16x_plus(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallEncoder4_16x_plus, self).__init__()
        self.fixed = fixed
        self.vgg = nn.Sequential(nn.Conv2d(3, 3, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 16, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(16, 16, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(32, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(32, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU())
        if model:
            assert os.path.splitext(model)[1] in {'.t7', '.pth'}
            if model.endswith('.t7'):
                t7_model = load_lua(model)
                load_param(t7_model, 0, self.vgg[0])
                load_param(t7_model, 2, self.vgg[2])
                load_param(t7_model, 5, self.vgg[5])
                load_param(t7_model, 9, self.vgg[9])
                load_param(t7_model, 12, self.vgg[12])
                load_param(t7_model, 16, self.vgg[16])
                load_param(t7_model, 19, self.vgg[19])
                load_param(t7_model, 22, self.vgg[22])
                load_param(t7_model, 25, self.vgg[25])
                load_param(t7_model, 29, self.vgg[29])
            else:
                net = torch.load(model)
                odict_keys = list(net.keys())
                cnt = 0
                i = 0
                for m in self.vgg.children():
                    if isinstance(m, nn.Conv2d):
                        None
                        m.weight.data.copy_(net[odict_keys[cnt]])
                        cnt += 1
                        m.bias.data.copy_(net[odict_keys[cnt]])
                        cnt += 1
                    i += 1

    def forward(self, x):
        return self.vgg(x)


class MultConst(nn.Module):

    def forward(self, input):
        return 255 * input


class Basicblock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Basicblock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), ConvLayer(inplanes, planes, kernel_size=3, stride=stride), norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=1), norm_layer(planes)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        if self.downsample is not None:
            residual = self.residual_layer(input)
        else:
            residual = input
        return residual + self.conv_block(input)


class UpBasicblock(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBasicblock, self).__init__()
        self.residual_layer = UpsampleConvLayer(inplanes, planes, kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride), norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.residual_layer(input) + self.conv_block(input)


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        return [relu1_2, relu2_2, relu3_3, relu4_3]


class ThumbAdaptiveInstanceNorm(ThumbInstanceNorm):

    def __init__(self):
        super(ThumbAdaptiveInstanceNorm, self).__init__(affine=False)

    def forward(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        if self.collection == True:
            thumb_mean, thumb_std = self.calc_mean_std(content_feat)
            self.thumb_mean = thumb_mean
            self.thumb_std = thumb_std
        normalized_feat = (content_feat - self.thumb_mean.expand(size)) / self.thumb_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Basicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     True),
    (Decoder2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (Decoder3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {}),
     True),
    (Decoder4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (Decoder5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (Encoder1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Encoder2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Encoder3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Encoder4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Encoder5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (GramMatrix,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inspiration,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LossCriterion,
     lambda: ([], {'style_layers': [4, 4], 'content_layers': [4, 4], 'style_weight': 4, 'content_weight': 4}),
     lambda: ([torch.rand([5, 4, 4, 4, 4]), torch.rand([5, 4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     False),
    (MultConst,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SmallDecoder1_16x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 24, 4, 4])], {}),
     True),
    (SmallDecoder1_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 24, 4, 4])], {}),
     True),
    (SmallDecoder2_16x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32, 4, 4])], {}),
     False),
    (SmallDecoder2_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32, 4, 4])], {}),
     True),
    (SmallDecoder3_16x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     False),
    (SmallDecoder3_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     True),
    (SmallDecoder4_16x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     False),
    (SmallDecoder4_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (SmallDecoder5_16x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (SmallDecoder5_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (SmallEncoder1_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder2_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder3_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder4_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder4_16x_plus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder4_2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SmallEncoder5_16x_aux,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ThumbAdaptiveInstanceNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UpBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UpBottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (decoder3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {}),
     True),
    (decoder4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (decoder5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (encoder3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (encoder4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (encoder5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (styleLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_czczup_URST(_paritybench_base):
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

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

