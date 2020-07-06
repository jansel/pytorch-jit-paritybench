import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
model = _module
model = _module
model_util = _module
pspnet = _module
seg_model = _module
train_dise_gta2city = _module
train_dise_synthia2city = _module
util = _module
CityDemoLoader = _module
CityLoader = _module
CityTestLoader = _module
GTA5Loader = _module
SYNTHIALoader = _module
loader = _module
augmentations = _module
loss = _module
metrics = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


import torchvision


import torchvision.utils as vutils


import torchvision.models as models


import torch.utils.data as torch_data


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


from math import ceil


import math


import torch.utils.model_zoo as model_zoo


import random


import collections


from torch.utils import data


from torchvision import models


from collections import OrderedDict


affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'


class Classifier_Module(nn.Module):

    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer5(x)
        x2 = self.layer4(x)
        x2 = self.layer6(x2)
        return x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate}, {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabMulti(pretrained=True, num_classes=21):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        saved_state_dict = model_zoo.load_url(RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    return model


pspnet_specs = {'pascalvoc': {'n_classes': 21, 'input_size': (473, 473), 'block_config': [3, 4, 23, 3]}, 'cityscapes': {'n_classes': 19, 'input_size': (713, 713), 'block_config': [3, 4, 23, 3]}, 'ade20k': {'n_classes': 150, 'input_size': (473, 473), 'block_config': [3, 4, 6, 3]}}


class SharedEncoder(nn.Module):

    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']
        Seg_Model = DeeplabMulti(num_classes=self.n_classes)
        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4
        self.final1 = Seg_Model.layer5
        self.final2 = Seg_Model.layer6

    def forward(self, x):
        inp_shape = x.shape[2:]
        low = self.layer0(x)
        x = self.layer1(low)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.final1(x)
        rec = self.layer4(x)
        x2 = self.final2(rec)
        return low, x1, x2, rec

    def get_1x_lr_params_NOscale(self):
        b = []
        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate}, {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class Classifier(nn.Module):

    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x


class PrivateEncoder(nn.Module):

    def __init__(self, input_channels, code_size):
        super(PrivateEncoder, self).__init__()
        self.input_channels = input_channels
        self.code_size = code_size
        self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.model = []
        self.model += [self.cnn]
        self.model += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        bs = x.size(0)
        output = self.model(x).view(bs, -1)
        return output


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0, dilation=1, norm='none', activation='relu', pad_type='zero', bias=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class PrivateDecoder(nn.Module):

    def __init__(self, shared_code_channel, private_code_size):
        super(PrivateDecoder, self).__init__()
        num_att = 256
        self.shared_code_channel = shared_code_channel
        self.private_code_size = private_code_size
        self.main = []
        self.upsample = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True), Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True), Conv2dBlock(128, 64, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True), Conv2dBlock(64, 32, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())
        self.main += [Conv2dBlock(shared_code_channel + num_att + 1, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
        self.main += [ResBlocks(3, 256, 'ln', 'relu', pad_type='zero')]
        self.main += [self.upsample]
        self.main = nn.Sequential(*self.main)
        self.mlp_att = nn.Sequential(nn.Linear(private_code_size, private_code_size), nn.ReLU(), nn.Linear(private_code_size, private_code_size), nn.ReLU(), nn.Linear(private_code_size, private_code_size), nn.ReLU(), nn.Linear(private_code_size, num_att))

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params[:, :m.num_features]
                std = torch.exp(adain_params[:, m.num_features:2 * m.num_features])
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def forward(self, shared_code, private_code, d):
        d = Variable(torch.FloatTensor(shared_code.shape[0], 1).fill_(d))
        d = d.unsqueeze(1)
        d_img = d.view(d.size(0), d.size(1), 1, 1).expand(d.size(0), d.size(1), shared_code.size(2), shared_code.size(3))
        att_params = self.mlp_att(private_code)
        att_img = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(att_params.size(0), att_params.size(1), shared_code.size(2), shared_code.size(3))
        code = torch.cat([shared_code, att_img, d_img], 1)
        output = self.main(code)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(Conv2dBlock(3, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False), Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False), Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False), Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False), nn.Conv2d(512, 1, 1, padding=0))
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        return x


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.feature = nn.Sequential(Conv2dBlock(n_classes, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False), Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False), Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False), Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False), nn.Conv2d(512, 1, 4, padding=2))

    def forward(self, x):
        x = self.feature(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module('c{}'.format(i + 1), Conv2dBlock(in_channels, out_channels, 3, stride=1, padding=padding, dilation=dilation, norm='bn', activation='relu', pad_type='reflect', bias=False))

    def forward(self, x):
        h = []
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class PyramidPooling(nn.Module):

    def __init__(self, fc_dim=2048, pool_scales=(1, 2, 3, 6)):
        super(PyramidPooling, self).__init__()
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        return ppm_out


class GaussianNoiseLayer(nn.Module):

    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()))
        return x + noise


class pspnet(nn.Module):
    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(self, n_classes=21, block_config=[3, 4, 23, 3], input_size=(473, 473), version=None):
        super(pspnet, self).__init__()
        """        
        self.block_config = pspnet_specs[version]['block_config'] if version is not None else block_config
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        self.input_size = pspnet_specs[version]['input_size'] if version is not None else input_size
        
        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=64,
                                                 padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)
        
        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)
        """
        self.n_classes = pspnet_specs[version]['n_classes'] if version is not None else n_classes
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

    def forward(self, x):
        inp_shape = x.shape[2:]
        """
        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)
        x = self.cbr_final(x)
        x = self.dropout(x)
        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGLoss_for_trans(nn.Module):

    def __init__(self, gpu_id=0):
        super(VGGLoss_for_trans, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, trans_img, struct_img, texture_img, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        while trans_img.size()[3] > 1024:
            trans_img, struct_img, texture_img = self.downsample(trans_img), self.downsample(struct_img), self.downsample(texture_img)
        trans_vgg, struct_vgg, texture_vgg = self.vgg(trans_img), self.vgg(struct_img), self.vgg(texture_img)
        loss = 0
        for i in range(len(trans_vgg)):
            if i < 3:
                x_feat_mean = trans_vgg[i].view(trans_vgg[i].size(0), trans_vgg[i].size(1), -1).mean(2)
                y_feat_mean = texture_vgg[i].view(texture_vgg[i].size(0), texture_vgg[i].size(1), -1).mean(2)
                loss += self.criterion(x_feat_mean, y_feat_mean.detach())
            else:
                loss += weights[i] * self.criterion(trans_vgg[i], struct_vgg[i].detach())
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPPModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'pyramids': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier_Module,
     lambda: ([], {'inplanes': 4, 'dilation_series': [4, 4], 'padding_series': [4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (GaussianNoiseLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PrivateDecoder,
     lambda: ([], {'shared_code_channel': 4, 'private_code_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (PrivateEncoder,
     lambda: ([], {'input_channels': 4, 'code_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PyramidPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlocks,
     lambda: ([], {'num_blocks': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_a514514772_DISE_Domain_Invariant_Structure_Extraction(_paritybench_base):
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

