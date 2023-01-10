import sys
_module = sys.modules[__name__]
del sys
data = _module
base_dataset = _module
iharmony4_dataset = _module
preprocess_iharmony4 = _module
test_dataset = _module
models = _module
base_model = _module
networks = _module
normalize = _module
rainnet_model = _module
options = _module
base_options = _module
train_options = _module
test = _module
train = _module
util = _module
config = _module
image_pool = _module
spectral_norm = _module
util = _module

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


import torch.utils.data


import random


import numpy as np


import torch.utils.data as data


import torchvision.transforms as transforms


from abc import ABC


from abc import abstractmethod


import torch


import torchvision.transforms.functional as tf


from torch.utils.data import Dataset


from collections import OrderedDict


import torch.nn as nn


from torch.nn import init


import torch.nn.functional as F


import functools


from torch.optim import lr_scheduler


from torch.nn.utils import spectral_norm


from torch import nn


from torch import cuda


from torch.autograd import Variable


import time


from torch.utils.tensorboard import SummaryWriter


from torch.nn.functional import normalize


class Identity(nn.Module):

    def forward(self, x):
        return x


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
            self.relu = nn.ReLU()
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

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class RAIN(nn.Module):

    def __init__(self, dims_in, eps=1e-05):
        """Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        """
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        mean_back, std_back = self.get_foreground_mean_std(x * (1 - mask), 1 - mask)
        normalized = (x - mean_back) / std_back
        normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) + self.background_beta[None, :, None, None]) * (1 - mask)
        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask)
        normalized = (x - mean_fore) / std_fore * std_back + mean_back
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) + self.foreground_beta[None, :, None, None]) * mask
        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])
        num = torch.sum(mask, dim=[2, 3])
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var + self.eps)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetBlockCodec(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=RAIN, use_dropout=False, use_attention=False, enc=True, dec=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetBlockCodec) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
            enc (bool) -- if use give norm_layer in encoder part.
            dec (bool) -- if use give norm_layer in decoder part.
        """
        super(UnetBlockCodec, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        self.norm_namebuffer = ['RAIN', 'RAIN_Method_Learnable', 'RAIN_Method_BN']
        if outermost:
            self.down = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.submodule = submodule
            self.up = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1), nn.Tanh())
        elif innermost:
            self.up = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias), nn.ReLU(True), nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias))
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
        else:
            self.down = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias))
            self.downnorm = norm_layer(inner_nc) if enc else get_norm_layer('instance')(inner_nc)
            self.submodule = submodule
            self.up = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias))
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
            if use_dropout:
                self.dropout = nn.Dropout(0.5)
        if use_attention:
            attention_conv = nn.Conv2d(outer_nc + input_nc, outer_nc + input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

    def forward(self, x, mask):
        if self.outermost:
            x = self.down(x)
            x = self.submodule(x, mask)
            ret = self.up(x)
            return ret
        elif self.innermost:
            ret = self.up(x)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret
        else:
            ret = self.down(x)
            if self.downnorm._get_name() in self.norm_namebuffer:
                ret = self.downnorm(ret, mask)
            else:
                ret = self.downnorm(ret)
            ret = self.submodule(ret, mask)
            ret = self.up(ret)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            if self.use_dropout:
                ret = self.dropout(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


def get_act_conv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.Conv2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)


def get_act_dconv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.ConvTranspose2d(dims_in, dims_out, kernel_size=kernel, stride=2, padding=1, bias=False))
    return nn.Sequential(*conv)


class RainNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=RAIN, norm_type_indicator=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], use_dropout=False, use_attention=True):
        super(RainNet, self).__init__()
        self.input_nc = input_nc
        self.norm_namebuffer = ['RAIN']
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        norm_type_list = [get_norm_layer('instance'), norm_layer]
        self.model_layer0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.model_layer1 = get_act_conv(nn.LeakyReLU(0.2, True), ngf, ngf * 2, 4, 2, 1, False)
        self.model_layer1norm = norm_type_list[norm_type_indicator[0]](ngf * 2)
        self.model_layer2 = get_act_conv(nn.LeakyReLU(0.2, True), ngf * 2, ngf * 4, 4, 2, 1, False)
        self.model_layer2norm = norm_type_list[norm_type_indicator[1]](ngf * 4)
        self.model_layer3 = get_act_conv(nn.LeakyReLU(0.2, True), ngf * 4, ngf * 8, 4, 2, 1, False)
        self.model_layer3norm = norm_type_list[norm_type_indicator[2]](ngf * 8)
        unet_block = UnetBlockCodec(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, enc=norm_type_indicator[6], dec=norm_type_indicator[7])
        unet_block = UnetBlockCodec(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, enc=norm_type_indicator[5], dec=norm_type_indicator[8])
        unet_block = UnetBlockCodec(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, enc=norm_type_indicator[4], dec=norm_type_indicator[9])
        self.unet_block = UnetBlockCodec(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, enc=norm_type_indicator[3], dec=norm_type_indicator[10])
        self.model_layer11 = get_act_dconv(nn.ReLU(True), ngf * 16, ngf * 4, 4, 2, 1, False)
        self.model_layer11norm = norm_type_list[norm_type_indicator[11]](ngf * 4)
        if use_attention:
            self.model_layer11att = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=1, stride=1), nn.Sigmoid())
        self.model_layer12 = get_act_dconv(nn.ReLU(True), ngf * 8, ngf * 2, 4, 2, 1, False)
        self.model_layer12norm = norm_type_list[norm_type_indicator[12]](ngf * 2)
        if use_attention:
            self.model_layer12att = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, stride=1), nn.Sigmoid())
        self.model_layer13 = get_act_dconv(nn.ReLU(True), ngf * 4, ngf, 4, 2, 1, False)
        self.model_layer13norm = norm_type_list[norm_type_indicator[13]](ngf)
        if use_attention:
            self.model_layer13att = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 2, kernel_size=1, stride=1), nn.Sigmoid())
        self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x, mask):
        x0 = self.model_layer0(x)
        x1 = self.model_layer1(x0)
        if self.model_layer1norm._get_name() in self.norm_namebuffer:
            x1 = self.model_layer1norm(x1, mask)
        else:
            x1 = self.model_layer1norm(x1)
        x2 = self.model_layer2(x1)
        if self.model_layer2norm._get_name() in self.norm_namebuffer:
            x2 = self.model_layer2norm(x2, mask)
        else:
            x2 = self.model_layer2norm(x2)
        x3 = self.model_layer3(x2)
        if self.model_layer3norm._get_name() in self.norm_namebuffer:
            x3 = self.model_layer3norm(x3, mask)
        else:
            x3 = self.model_layer3norm(x3)
        ox3 = self.unet_block(x3, mask)
        ox2 = self.model_layer11(ox3)
        if self.model_layer11norm._get_name() in self.norm_namebuffer:
            ox2 = self.model_layer11norm(ox2, mask)
        else:
            ox2 = self.model_layer11norm(ox2)
        ox2 = torch.cat([x2, ox2], 1)
        if self.use_attention:
            ox2 = self.model_layer11att(ox2) * ox2
        ox1 = self.model_layer12(ox2)
        if self.model_layer12norm._get_name() in self.norm_namebuffer:
            ox1 = self.model_layer12norm(ox1, mask)
        else:
            ox1 = self.model_layer12norm(ox1)
        ox1 = torch.cat([x1, ox1], 1)
        if self.use_attention:
            ox1 = self.model_layer12att(ox1) * ox1
        ox0 = self.model_layer13(ox1)
        if self.model_layer13norm._get_name() in self.norm_namebuffer:
            ox0 = self.model_layer13norm(ox0, mask)
        else:
            ox0 = self.model_layer13norm(ox0)
        ox0 = torch.cat([x0, ox0], 1)
        if self.use_attention:
            ox0 = self.model_layer13att(ox0) * ox0
        out = self.model_out(ox0)
        return out

    def processImage(self, x, mask, background=None):
        if background is not None:
            x = x * mask + background * (1 - mask)
        if self.input_nc == 4:
            x = torch.cat([x, mask], dim=1)
        pred = self.forward(x, mask)
        return pred * mask + x[:, :3, :, :] * (1 - mask)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PartialConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        self.return_mask = True
        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
        self.last_size = None, None, None, None
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater
                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3])
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3])
                else:
                    mask = mask_in
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-08)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OrgDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d, global_stages=0):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(OrgDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 3
        padw = 0
        self.conv1 = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        if global_stages < 1:
            self.conv1f = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        else:
            self.conv1f = self.conv1
        self.relu1 = nn.LeakyReLU(0.2, True)
        nf_mult = 1
        nf_mult_prev = 1
        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm2 = norm_layer(ndf * nf_mult)
        if global_stages < 2:
            self.conv2f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm2f = norm_layer(ndf * nf_mult)
        else:
            self.conv2f = self.conv2
            self.norm2f = self.norm2
        self.relu2 = nn.LeakyReLU(0.2, True)
        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm3 = norm_layer(ndf * nf_mult)
        if global_stages < 3:
            self.conv3f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm3f = norm_layer(ndf * nf_mult)
        else:
            self.conv3f = self.conv3
            self.norm3f = self.norm3
        self.relu3 = nn.LeakyReLU(0.2, True)
        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.norm4 = norm_layer(ndf * nf_mult)
        self.conv4 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv4f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm4f = norm_layer(ndf * nf_mult)
        self.relu4 = nn.LeakyReLU(0.2, True)
        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv5f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm5 = norm_layer(ndf * nf_mult)
        self.norm5f = norm_layer(ndf * nf_mult)
        self.relu5 = nn.LeakyReLU(0.2, True)
        n = 5
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv6 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv6f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm6 = norm_layer(ndf * nf_mult)
        self.norm6f = norm_layer(ndf * nf_mult)
        self.relu6 = nn.LeakyReLU(0.2, True)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv7 = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        self.conv7f = spectral_norm(PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))

    def forward(self, input, mask=None):
        x = input
        x, _ = self.conv1(x)
        x = self.relu1(x)
        x, _ = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x, _ = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x, _ = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x, _ = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x, _ = self.conv6(x)
        x = self.norm6(x)
        x = self.relu6(x)
        x, _ = self.conv7(x)
        """Standard forward."""
        xf, xb = input, input
        mf, mb = mask, 1 - mask
        xf, mf = self.conv1f(xf, mf)
        xf = self.relu1(xf)
        xf, mf = self.conv2f(xf, mf)
        xf = self.norm2f(xf)
        xf = self.relu2(xf)
        xf, mf = self.conv3f(xf, mf)
        xf = self.norm3f(xf)
        xf = self.relu3(xf)
        xf, mf = self.conv4f(xf, mf)
        xf = self.norm4f(xf)
        xf = self.relu4(xf)
        xf, mf = self.conv5f(xf, mf)
        xf = self.norm5f(xf)
        xf = self.relu5(xf)
        xf, mf = self.conv6f(xf, mf)
        xf = self.norm6f(xf)
        xf = self.relu6(xf)
        xf, mf = self.conv7f(xf, mf)
        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.norm5f(xb)
        xb = self.relu5(xb)
        xb, mb = self.conv6f(xb, mb)
        xb = self.norm6f(xb)
        xb = self.relu6(xb)
        xb, mb = self.conv7f(xb, mb)
        return x, xf, xb


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        num_outputs = ndf * min(2 ** n_layers, 8)
        self.D = OrgDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.convl1 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul1 = nn.LeakyReLU(0.2)
        self.convl2 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul2 = nn.LeakyReLU(0.2)
        self.convl3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)
        self.convg3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)

    def forward(self, input, mask=None, gp=False, feat_loss=False):
        x, xf, xb = self.D(input, mask)
        feat_l, feat_g = torch.cat([xf, xb]), x
        x = self.convg3(x)
        sim = xf * xb
        sim = self.convl1(sim)
        sim = self.relul1(sim)
        sim = self.convl2(sim)
        sim = self.relul2(sim)
        sim = self.convl3(sim)
        sim_sum = sim
        if not gp:
            if feat_loss:
                return x, sim_sum, feat_g, feat_l
            return x, sim_sum
        return (x + sim_sum) * 0.5


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PartialConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RAIN,
     lambda: ([], {'dims_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_junleen_RainNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

