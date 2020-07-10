import sys
_module = sys.modules[__name__]
del sys
data = _module
base_dataset = _module
concat_dataset = _module
df_dataset = _module
image_and_df_dataset = _module
image_and_voxel_dataset = _module
images_dataset = _module
sampler = _module
voxel_dataset = _module
models = _module
base_model = _module
basics = _module
full_model = _module
networks = _module
networks_3d = _module
shape_gan_model = _module
test_model = _module
texture_model = _module
texture_real_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
render_module = _module
build = _module
calc_prob = _module
functions = _module
calc_prob = _module
setup = _module
render_sketch = _module
build = _module
vtn = _module
affine_grid3d = _module
grid_sample3d = _module
AffineGridGen3D = _module
GridSampler3D = _module
modules = _module
test = _module
test_shape = _module
train = _module
util = _module
html = _module
image_pool = _module
util = _module
util_print = _module
util_render = _module
util_voxel = _module
visualizer = _module

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
xrange = range
wraps = functools.wraps


import torch.utils.data


import torch.utils.data as data


import torchvision.transforms as transforms


from abc import ABC


from abc import abstractmethod


import numpy as np


import torch


import random


from torch.nn.functional import pad as pad_tensor


from torch.utils.data.sampler import Sampler


import math


from scipy.io import loadmat


from collections import OrderedDict


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


from torch import nn


import itertools


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.autograd import Variable


from torch.nn.functional import grid_sample


from scipy import ndimage


import time


class D_NLayersMulti(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module('model_0', nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / 2 ** i))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module('model_%d' % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, 'model_%d' % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class G_NLayers(nn.Module):

    def __init__(self, output_nc=3, nz=100, ngf=64, n_layers=3, norm_layer=None, nl_layer=None):
        super(G_NLayers, self).__init__()
        kw, s, padw = 4, 2, 1
        sequence = [nn.ConvTranspose2d(nz, ngf * 4, kernel_size=kw, stride=1, padding=0, bias=True)]
        if norm_layer is not None:
            sequence += [norm_layer(ngf * 4)]
        sequence += [nl_layer()]
        nf_mult = 4
        nf_mult_prev = 4
        for n in range(n_layers, 0, -1):
            nf_mult_prev = nf_mult
            nf_mult = min(n, 4)
            sequence += [nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=kw, stride=s, padding=padw, bias=True)]
            if norm_layer is not None:
                sequence += [norm_layer(ngf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=s, padding=padw, bias=True)]
        sequence += [nn.Tanh()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class D_NLayers(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=None, nl_layer=None):
        super(D_NLayers, self).__init__()
        kw, padw, use_bias = 4, 1, True
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias), nl_layer()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=0, bias=use_bias)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            predictions (tensor list) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        return sum(all_losses)


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [Upsample(scale_factor=2, mode='bilinear'), nn.ReflectionPad2d(1), nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv


class UnetBlock(nn.Module):

    def __init__(self, input_nc, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        if outermost:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class G_Unet_add_input(nn.Module):

    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x
        return self.model(x_with_z)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)


def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):

    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):

    def __init__(self, input_nc=3, output_nc=1, nef=64, n_blocks=4, norm_layer=None, nl_layer=None, vae=False, all_z=False):
        super(E_ResNet, self).__init__()
        self.vae = vae
        max_ndf = 4
        conv_layers = [nn.Conv2d(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)
            output_ndf = nef * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vae:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vae:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


class UnetBlock_with_z(nn.Module):

    def __init__(self, input_nc, outer_nc, inner_nc, nz=0, submodule=None, outermost=False, innermost=False, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()
        if outermost:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x
        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class G_Unet_add_all(nn.Module):

    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class E_NLayers(nn.Module):

    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3, norm_layer=None, nl_layer=None, vae=False):
        super(E_NLayers, self).__init__()
        self.vae = vae
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vae:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vae:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


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

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
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
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class Decoder(nn.Module):

    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)


class Decoder_all(nn.Module):

    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        for i in range(n_upsample):
            block = [Upsample(scale_factor=2), Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class G_Resnet(nn.Module):

    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64, norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None):
        content = self.enc_content(image)
        images_recon = self.decode(content, style)
        return images_recon


class StyleEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)
            self.fc_var = nn.Linear(dim, style_dim)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class E_adaIN(nn.Module):

    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4, norm=None, nl_layer=None, vae=False):
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
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

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


def deconvBlock(input_nc, output_nc, bias, norm_layer=None, nl='relu'):
    layers = [nn.ConvTranspose3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    if nl == 'relu':
        layers += [nn.ReLU(True)]
    elif nl == 'lrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    else:
        raise NotImplementedError('NL layer {} is not implemented' % nl)
    return nn.Sequential(*layers)


def get_norm_layer(layer_type='inst'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'batch3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif layer_type == 'inst':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif layer_type == 'inst3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


def toRGB(input_nc, output_nc, bias, zero_mean=False, sig=True):
    layers = [nn.ConvTranspose3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    if sig:
        layers += [nn.Sigmoid()]
    return nn.Sequential(*layers)


class _netG0(nn.Module):

    def __init__(self, bias, res, nz=200, ngf=64, max_nf=8, nc=1, norm='batch'):
        super(_netG0, self).__init__()
        norm_layer = get_norm_layer(layer_type=norm)
        self.res = res
        self.block_0 = nn.Sequential(*[nn.ConvTranspose3d(nz, ngf * max_nf, 4, 1, 0, bias=bias), norm_layer(ngf * 8), nn.ReLU(True)])
        self.n_blocks = 1
        input_dim = ngf * max_nf
        n_layers = int(math.log(res, 2)) - 3
        for n in range(n_layers):
            input_nc = int(max(ngf, input_dim))
            output_nc = int(max(ngf, input_dim // 2))
            setattr(self, 'block_{:d}'.format(self.n_blocks), deconvBlock(input_nc, output_nc, bias, norm_layer=norm_layer, nl='relu'))
            input_dim /= 2
            self.n_blocks += 1
        setattr(self, 'toRGB_{:d}'.format(res), toRGB(output_nc, nc, bias, sig=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, return_stat=False):
        output = input
        for n in range(self.n_blocks):
            block = getattr(self, 'block_{:d}'.format(n))
            output = block(output)
        toRGB = getattr(self, 'toRGB_{:d}'.format(self.res))
        output = toRGB(output)
        output = output / 2
        if return_stat:
            stat = [output.max().item(), output.min().item(), output.std().item(), output.mean().item()]
            return self.sigmoid(output), stat
        else:
            return self.sigmoid(output)


def convBlock(input_nc, output_nc, bias, norm_layer=None):
    layers = [nn.Conv3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


def fromRGB(input_nc, output_nc, bias):
    layers = []
    layers += [nn.Conv3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class _netD0(nn.Module):

    def __init__(self, bias=False, res=128, final_res=128, nc=1, ndf=64, max_nf=8, norm='none'):
        super(_netD0, self).__init__()
        self.res = res
        self.n_blocks = 0
        norm_layer = get_norm_layer(layer_type=norm)
        n_layers = int(math.log(res, 2)) - 3
        n_final_layers = int(math.log(final_res, 2)) - 3
        self.offset = n_final_layers - n_layers
        setattr(self, 'fromRGB_{:d}'.format(res), fromRGB(1, ndf * min(2 ** max(0, self.offset - 1), max_nf), bias))
        for n in range(n_final_layers - n_layers, n_final_layers):
            input_nc = ndf * min(2 ** max(0, n - 1), max_nf)
            output_nc = ndf * min(2 ** n, max_nf)
            block_name = 'block_{}'.format(n)
            setattr(self, block_name, convBlock(input_nc, output_nc, bias, norm_layer))
            self.n_blocks += 1
        block_name = 'block_{:d}'.format(n_final_layers)
        setattr(self, block_name, nn.Conv3d(ndf * max_nf, 1, 4, 1, 0, bias=bias))
        self.n_blocks += 1

    def forward(self, input):
        fromRGB = getattr(self, 'fromRGB_{:d}'.format(self.res))
        output = fromRGB(input)
        for n in range(self.n_blocks):
            block = getattr(self, 'block_{:d}'.format(n + self.offset))
            output = block(output)
        return output.view(-1, 1).squeeze(1)


class GetRotationMatrix(nn.Module):

    def __init__(self, az_min=-np.pi / 2, az_max=np.pi / 2, ele_min=0, ele_max=2 * np.pi / 9):
        super().__init__()
        self.az_max = az_max
        self.az_min = az_min
        self.ele_max = ele_max
        self.ele_min = ele_min

    def forward(self, angles_in):
        is_cuda = angles_in.is_cuda
        assert angles_in.shape[1] == 2
        bn = angles_in.shape[0]
        az_in = angles_in[:, (0)]
        ele_in = angles_in[:, (1)]
        az_in = torch.clamp(az_in, self.az_min, self.az_max)
        ele_in = torch.clamp(ele_in, self.ele_min, self.ele_max)
        az_sin = torch.sin(az_in)
        az_cos = torch.cos(az_in)
        ele_sin = torch.sin(ele_in)
        ele_cos = torch.cos(ele_in)
        R_az = self.create_Raz(az_cos, az_sin)
        R_ele = self.create_Rele(ele_cos, ele_sin)
        R_rot = torch.bmm(R_az, R_ele)
        R_0 = angles_in.data.new(bn, 3, 3).zero_()
        R_0[:, (0), (1)] = 1
        R_0[:, (1), (0)] = -1
        R_0[:, (2), (2)] = 1
        R_0 = R_0.requires_grad_(True)
        if is_cuda:
            R_0 = R_0
        R = torch.bmm(R_rot, R_0)
        zeros = angles_in.data.new_zeros([bn, 3, 1]).zero_().requires_grad_(True)
        return torch.cat((R, zeros), dim=2)

    def create_Rele(self, ele_cos, ele_sin):
        bn = ele_cos.shape[0]
        one = Variable(ele_cos.data.new(bn, 1, 1).fill_(1))
        zero = Variable(ele_cos.data.new(bn, 1, 1).zero_())
        ele_cos = ele_cos.view(bn, 1, 1)
        ele_sin = ele_sin.view(bn, 1, 1)
        c1 = torch.cat((one, zero, zero), dim=1)
        c2 = torch.cat((zero, ele_cos, ele_sin), dim=1)
        c3 = torch.cat((zero, -ele_sin, ele_cos), dim=1)
        return torch.cat((c1, c2, c3), dim=2)

    def create_Raz(self, cos, sin):
        bn = cos.shape[0]
        one = Variable(cos.data.new(bn, 1, 1).fill_(1))
        zero = Variable(cos.data.new(bn, 1, 1).zero_())
        cos = cos.view(bn, 1, 1)
        sin = sin.view(bn, 1, 1)
        c1 = torch.cat((cos, sin, zero), dim=1)
        c2 = torch.cat((-sin, cos, zero), dim=1)
        c3 = torch.cat((zero, zero, one), dim=1)
        return torch.cat((c1, c2, c3), dim=2)


class FineSizeCroppingLayer(nn.Module):
    """
    crop the input list of images to specified size.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, random_number):
        """
        random number is from 0 to 1
        """
        N, C, H, W = x.shape
        output_size = self.output_size
        assert H >= output_size and W >= output_size
        h_beg = round((H - output_size) * random_number)
        w_beg = round((W - output_size) * random_number)
        return x[:, :, h_beg:h_beg + output_size, w_beg:w_beg + output_size]


class CroppingLayer(nn.Module):
    """
    crop and pad output to have consistant size
    """

    def __init__(self, output_size, sil_th=0.8, padding_pct=0.03, no_largest=False):
        super().__init__()
        self.output_size = output_size
        self.threshold = nn.Threshold(sil_th, 0)
        size = [output_size, output_size]
        h = torch.arange(0, size[0]).float() / (size[0] - 1.0) * 2.0 - 1.0
        w = torch.arange(0, size[1]).float() / (size[1] - 1.0) * 2.0 - 1.0
        grid = torch.zeros(size[0], size[1], 2)
        grid[:, :, (0)] = w.unsqueeze(0).repeat(size[0], 1)
        grid[:, :, (1)] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
        grid = grid.unsqueeze(0)
        self.register_buffer('grid', grid)
        self.kernel = np.ones((5, 5), np.uint8)
        self.padding_pct = padding_pct
        self.th = sil_th
        self.no_largest = no_largest

    def forward(self, exp_sil, exp_depth):
        sil, depth, _, _ = self.crop_depth_sil(exp_sil, exp_depth)
        return sil, depth

    def bbox_from_sil(self, exp_sil, padding_pct=0.03):
        n, c, h, w = exp_sil.shape
        assert c == 1
        mask_th = exp_sil.data.cpu().numpy()
        mask_th_binary = np.where(mask_th < self.th, 0.0, 1.0)
        bbox = np.zeros([n, 4]).astype(int)
        mask_largest_batch = torch.FloatTensor(n, c, h, w).zero_()
        for x in range(n):
            if self.no_largest:
                nz = np.nonzero(mask_th_binary[(x), (0), :, :])
                bbox[x, 0] = np.min(nz[0])
                bbox[x, 1] = np.min(nz[1])
                bbox[x, 2] = np.max(nz[0])
                bbox[x, 3] = np.max(nz[1])
                mask_largest_batch[(x), (0), :, :] = torch.from_numpy(mask_th_binary[(x), (0), :, :].astype(np.float32)).float()
            else:
                mask_th_binary_pad = np.pad(mask_th_binary[(x), (0), :, :], ((1, 1),), 'constant', constant_values=0).astype(np.uint8)
                labeled, nr_objects = ndimage.measurements.label(mask_th_binary_pad)
                counts = np.bincount(labeled.flatten())
                largest = np.argmax(counts[1:]) + 1
                mask_largest = np.where(labeled == largest, 1, 0)[1:-1, 1:-1]
                mask_largest_batch[(x), (0), :, :] = torch.from_numpy(mask_largest.astype(np.float32)).float()
                nz = np.nonzero(mask_largest)
                bbox[x, 0] = np.min(nz[0])
                bbox[x, 1] = np.min(nz[1])
                bbox[x, 2] = np.max(nz[0])
                bbox[x, 3] = np.max(nz[1])
        return bbox, mask_largest_batch

    def crop_depth_sil(self, exp_sil_full, exp_depth_full, is_debug=False):
        bbox, mask_largest_batch = self.bbox_from_sil(exp_sil_full)
        mask_largest_batch = exp_sil_full.new_tensor(mask_largest_batch)
        bbox = bbox.astype(np.int32)
        exp_sil = exp_sil_full * mask_largest_batch
        exp_depth = exp_depth_full * mask_largest_batch
        exp_depth = exp_depth + 3 * (1 - mask_largest_batch)
        n, c, h, w = exp_sil.shape
        new_sil = []
        new_depth = []
        shape_stat = []
        for x in range(n):
            h = bbox[x, 2] + 1 - bbox[x, 0]
            w = bbox[x, 3] + 1 - bbox[x, 1]
            h = int(h)
            w = int(w)
            cropped_sil = exp_sil[(x), (0), bbox[x, 0]:bbox[x, 0] + h, bbox[x, 1]:bbox[x, 1] + w]
            cropped_sil = cropped_sil.contiguous().view(1, 1, h, w)
            cropped_depth = exp_depth[(x), (0), bbox[x, 0]:bbox[x, 0] + h, bbox[x, 1]:bbox[x, 1] + w]
            cropped_depth = cropped_depth.contiguous().view(1, 1, h, w)
            if h > w:
                dim = h
                m_sil = nn.ConstantPad2d(((h - w) // 2, (h - w) // 2, 0, 0), 0)
                m_depth = nn.ConstantPad2d(((h - w) // 2, (h - w) // 2, 0, 0), 3)
            else:
                dim = w
                m_sil = nn.ConstantPad2d((0, 0, (w - h) // 2, (w - h) // 2), 0)
                m_depth = nn.ConstantPad2d((0, 0, (w - h) // 2, (w - h) // 2), 3)
            pad = int(np.floor(dim * self.padding_pct))
            space_pad_depth = nn.ConstantPad2d((pad, pad, pad, pad), 3)
            space_pad_sil = nn.ConstantPad2d((pad, pad, pad, pad), 0)
            sq_depth = space_pad_depth(m_depth(cropped_depth))
            sq_sil = space_pad_sil(m_sil(cropped_sil))
            shape_stat.append(sq_depth.shape)
            new_sil.append(grid_sample(sq_sil, self.grid))
            new_depth.append(grid_sample(sq_depth, self.grid))
        new_sil = torch.cat(new_sil, dim=0)
        new_depth = torch.cat(new_depth, dim=0)
        if is_debug:
            return sq_sil, cropped_depth, bbox, sq_depth.shape
        else:
            return new_sil, new_depth, bbox, shape_stat


class CalcStopProb(Function):

    @staticmethod
    def forward(ctx, prob_in):
        assert prob_in.dim() == 5
        assert prob_in.type() == 'torch.cuda.FloatTensor'
        stop_prob = prob_in.new_zeros(prob_in.shape)
        calc_prob_lib.calc_prob_forward(prob_in, stop_prob)
        ctx.save_for_backward(prob_in, stop_prob)
        return stop_prob

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_in):
        prob_in, stop_prob = ctx.saved_tensors
        grad_out = grad_in.new_zeros(grad_in.shape)
        stop_prob_weighted = stop_prob * grad_in
        calc_prob_lib.calc_prob_backward(prob_in, stop_prob_weighted, grad_out)
        if torch.isnan(grad_out).any():
            None
        elif torch.isinf(grad_out).any():
            None
        return grad_out


class AffineGridGen3DFunction(Function):
    """
    Generate a 3D affine grid of size (batch*sz1*sz2*sz3*3)
    The affine grid is defined by a 3x4 matrix theta.
    The grid is initialized as a grid in [-1,1] in all dimensions,
    then transformed by matrix multiplication by theta.

    When theta is set to eye(3,4), the grid should match the original grid in a box.
    """

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        assert len(size) == 5, 'Grid size should be specified by size of tensor to interpolate (5D)'
        assert theta.dim() == 3 and theta.size()[1:] == torch.Size([3, 4]), '3D affine transformation defined by a 3D matrix of batch*3*4'
        assert theta.size(0) == size[0], 'batch size mismatch'
        N, C, sz1, sz2, sz3 = size
        ctx.size = size
        ctx.is_cuda = theta.is_cuda
        theta = theta.contiguous()
        base_grid = theta.new(N, sz1, sz2, sz3, 4)
        linear_points = torch.linspace(-1, 1, sz1) if sz1 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, (0)] = linear_points.view(1, -1, 1, 1).expand_as(base_grid[:, :, :, :, (0)])
        linear_points = torch.linspace(-1, 1, sz2) if sz2 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, (1)] = linear_points.view(1, 1, -1, 1).expand_as(base_grid[:, :, :, :, (1)])
        linear_points = torch.linspace(-1, 1, sz3) if sz3 > 1 else torch.Tensor([-1])
        base_grid[:, :, :, :, (2)] = linear_points.view(1, 1, 1, -1).expand_as(base_grid[:, :, :, :, (2)])
        base_grid[:, :, :, :, (3)] = 1
        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, sz1 * sz2 * sz3, 4), theta.transpose(1, 2))
        grid = grid.view(N, sz1, sz2, sz3, 3)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        N, C, sz1, sz2, sz3 = ctx.size
        assert grad_output.size() == torch.Size([N, sz1, sz2, sz3, 3])
        grad_output = grad_output.contiguous()
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(base_grid.view(N, sz1 * sz2 * sz3, 4).transpose(1, 2), grad_output.view(N, sz1 * sz2 * sz3, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None


def affine_grid3d(theta, size):
    return AffineGridGen3DFunction.apply(theta, size)


class GridSampler3D(nn.Module):

    def forward(self, theta, size):
        return grid_sample3d(theta, size)


def grid_sample3d(input, grid):
    """
    Perform trilinear interpolation on 3D matrices
    input: batch * channel * x * y * z
    grid: batch * gridx * gridy * gridz * 3
    output: batch * channel * gridx * gridy * gridz
    The interpolation is performed on each channel independently
    """
    return GridSampler3D.apply(input, grid)


class VoxelRenderLayer(nn.Module):

    def __init__(self, voxel_shape, camera_distance=2.0, fl=0.05, w=0.0612, res=128, nsamples_factor=1.5):
        super().__init__()
        self.camera_distance = camera_distance
        self.fl = fl
        self.w = w
        self.voxel_shape = voxel_shape
        self.nsamples_factor = nsamples_factor
        self.res = res
        self.register_buffer('grid', self.grid_gen())
        self.grid_sampler3d = grid_sample3d
        self.calc_stop_prob = CalcStopProb().apply
        self.affine_grid3d = affine_grid3d

    def forward(self, voxel_in, rotation_matrix=None):
        if rotation_matrix is None:
            voxel_rot = voxel_in
        else:
            voxel_rot_grid = self.affine_grid3d(rotation_matrix, voxel_in.shape)
            voxel_rot = self.grid_sampler3d(voxel_in, voxel_rot_grid)
        voxel_align = self.grid_sampler3d(voxel_rot, self.grid)
        voxel_align = voxel_align.permute(0, 1, 3, 4, 2)
        voxel_align = torch.clamp(voxel_align, 0.0001, 1 - 0.0001)
        voxel_align = voxel_align.contiguous()
        stop_prob = self.calc_stop_prob(voxel_align)
        exp_depth = torch.matmul(stop_prob, self.depth_weight)
        back_groud_prob = torch.prod(1.0 - voxel_align, dim=4)
        back_groud_prob = torch.clamp(back_groud_prob, 0.0001, 1 - 0.0001)
        back_groud_prob = back_groud_prob * (self.camera_distance + 1.0)
        exp_depth = exp_depth + back_groud_prob
        exp_sil = torch.sum(stop_prob, dim=4)
        return torch.transpose(exp_sil, 2, 3), torch.transpose(exp_depth, 2, 3)

    def grid_gen(self, numtype=np.float32):
        n, c, sx, sy, sz = self.voxel_shape
        nsamples = int(sz * self.nsamples_factor)
        res = self.res
        w = self.w
        dist = self.camera_distance
        self.register_buffer('depth_weight', torch.linspace(dist - 1, dist + 1, nsamples))
        fl = self.fl
        grid = np.zeros([n, nsamples, res, res, 3], dtype=numtype)
        h_linspace = np.linspace(w / 2, -w / 2, res)
        w_linspace = np.linspace(w / 2, -w / 2, res)
        H, W = np.meshgrid(h_linspace, w_linspace)
        cam = np.array([[[-dist, 0, 0]]])
        grid_vec = np.zeros([res, res, 3], dtype=numtype)
        grid_vec[:, :, (1)] = W
        grid_vec[:, :, (2)] = H
        grid_vec[:, :, (0)] = -(dist - fl)
        grid_vec = grid_vec - cam
        self.grid_vec = grid_vec
        grid_vec_a = grid_vec * ((dist - 1) / fl)
        grid_vec_b = grid_vec * ((dist + 1) / fl)
        for idn in range(n):
            for ids in range(nsamples):
                grid[(idn), (ids), :, :, :] = grid_vec_b - (1 - ids / nsamples) * (grid_vec_b - grid_vec_a)
        grid = grid + cam
        return torch.from_numpy(grid.astype(numtype))


class AffineGridGen3D(nn.Module):

    def forward(self, theta, size):
        return affine_grid3d(theta, size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2dBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (D_NLayersMulti,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Decoder,
     lambda: ([], {'n_upsample': 4, 'n_res': 4, 'dim': 18, 'output_dim': 4}),
     lambda: ([torch.rand([4, 18, 4, 4])], {}),
     False),
    (Decoder_all,
     lambda: ([], {'n_upsample': 4, 'n_res': 4, 'dim': 18, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (E_adaIN,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlocks,
     lambda: ([], {'num_blocks': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'scale_factor': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_junyanz_VON(_paritybench_base):
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

