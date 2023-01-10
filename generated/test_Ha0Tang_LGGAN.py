import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
image_folder = _module
single_dataset = _module
unaligned_dataset = _module
models = _module
base_model = _module
lggan_model = _module
networks = _module
test_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
get_data = _module
html = _module
image_pool = _module
util = _module
visualizer = _module
generator = _module
pix2pix_model = _module
pix2pix_trainer = _module
data = _module
ade20k_dataset = _module
base_dataset = _module
cityscapes_dataset = _module
coco_dataset = _module
custom_dataset = _module
facades_dataset = _module
image_folder = _module
pix2pix_dataset = _module
models = _module
networks = _module
architecture = _module
base_network = _module
discriminator = _module
encoder = _module
generator = _module
loss = _module
normalization = _module
pix2pix_model = _module
base_options = _module
test_ade = _module
test_city = _module
train_ade = _module
train_city = _module
trainers = _module
coco = _module
iter_counter = _module
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


import torchvision.transforms as transforms


import torch


import torch.utils.data as data


from collections import OrderedDict


import itertools


import numpy as np


from matplotlib import cm


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


import torch.nn.functional as F


import numpy


import collections


from numpy import inf


from torchvision import models


import torchvision


import torch.nn.utils.spectral_norm as spectral_norm


import re


class VGG19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
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

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(torch.nn.Module):

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


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        elif target_is_real:
            return -input.mean()
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(pw), norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)), activation, nn.ReflectionPad2d(pw), norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)))

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        None

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer


class NLayerDiscriminator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), nn.LeakyReLU(0.2, False)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


def custom_replace(tensor, label):
    res = tensor.clone()
    res[tensor == label] = 1
    res[tensor != label] = 0
    return res


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class resnet_block(nn.Module):

    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))
        return input + x


class ResnetGenerator_local(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_local, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(input_nc, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks1.weight_init(0, 0.02)
        self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks2.weight_init(0, 0.02)
        self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks3.weight_init(0, 0.02)
        self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks4.weight_init(0, 0.02)
        self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks5.weight_init(0, 0.02)
        self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks6.weight_init(0, 0.02)
        self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks7.weight_init(0, 0.02)
        self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks8.weight_init(0, 0.02)
        self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks9.weight_init(0, 0.02)
        self.deconv3_local = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_local = nn.InstanceNorm2d(128)
        self.deconv4_local = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_local = nn.InstanceNorm2d(64)
        self.deconv5_1 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_2 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_3 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_4 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_5 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_6 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_7 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv3_global = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_global = nn.InstanceNorm2d(128)
        self.deconv4_global = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_global = nn.InstanceNorm2d(64)
        self.deconv5_global = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv3_attention = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_attention = nn.InstanceNorm2d(128)
        self.deconv4_attention = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_attention = nn.InstanceNorm2d(64)
        self.deconv5_attention = nn.Conv2d(64, 2, 1, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, mask):
        mask_1 = custom_replace(mask[:, 0:1, :, :], 10)
        mask_2 = custom_replace(mask[:, 0:1, :, :], 20)
        mask_3 = custom_replace(mask[:, 0:1, :, :], 30)
        mask_4 = custom_replace(mask[:, 0:1, :, :], 40)
        mask_5 = custom_replace(mask[:, 0:1, :, :], 50)
        mask_6 = custom_replace(mask[:, 0:1, :, :], 60)
        mask_1_64 = mask_1.repeat(1, 64, 1, 1)
        mask_2_64 = mask_2.repeat(1, 64, 1, 1)
        mask_3_64 = mask_3.repeat(1, 64, 1, 1)
        mask_4_64 = mask_4.repeat(1, 64, 1, 1)
        mask_5_64 = mask_5.repeat(1, 64, 1, 1)
        mask_6_64 = mask_6.repeat(1, 64, 1, 1)
        mask_1_3 = mask_1.repeat(1, 3, 1, 1)
        mask_2_3 = mask_2.repeat(1, 3, 1, 1)
        mask_3_3 = mask_3.repeat(1, 3, 1, 1)
        mask_4_3 = mask_4.repeat(1, 3, 1, 1)
        mask_5_3 = mask_5.repeat(1, 3, 1, 1)
        mask_6_3 = mask_6.repeat(1, 3, 1, 1)
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks1(x)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x)
        middle_x = self.resnet_blocks9(x)
        x_local = F.relu(self.deconv3_norm_local(self.deconv3_local(middle_x)))
        x_feature_local = F.relu(self.deconv4_norm_local(self.deconv4_local(x_local)))
        label_1 = x_feature_local * mask_1_64
        label_2 = x_feature_local * mask_2_64
        label_3 = x_feature_local * mask_3_64
        label_4 = x_feature_local * mask_4_64
        label_5 = x_feature_local * mask_5_64
        label_6 = x_feature_local * mask_6_64
        label_1 = F.pad(label_1, (3, 3, 3, 3), 'reflect')
        label_2 = F.pad(label_2, (3, 3, 3, 3), 'reflect')
        label_3 = F.pad(label_3, (3, 3, 3, 3), 'reflect')
        label_4 = F.pad(label_4, (3, 3, 3, 3), 'reflect')
        label_5 = F.pad(label_5, (3, 3, 3, 3), 'reflect')
        label_6 = F.pad(label_6, (3, 3, 3, 3), 'reflect')
        result_1 = torch.tanh(self.deconv5_1(label_1))
        result_2 = torch.tanh(self.deconv5_2(label_2))
        result_3 = torch.tanh(self.deconv5_3(label_3))
        result_4 = torch.tanh(self.deconv5_4(label_4))
        result_5 = torch.tanh(self.deconv5_5(label_5))
        result_6 = torch.tanh(self.deconv5_6(label_6))
        result_local = result_1 + result_2 + result_3 + result_4 + result_5 + result_6
        x_global = F.relu(self.deconv3_norm_global(self.deconv3_global(middle_x)))
        x_global = F.relu(self.deconv4_norm_global(self.deconv4_global(x_global)))
        x_global = F.pad(x_global, (3, 3, 3, 3), 'reflect')
        result_global = torch.tanh(self.deconv5_global(x_global))
        x_attention = F.relu(self.deconv3_norm_attention(self.deconv3_attention(middle_x)))
        x_attention = F.relu(self.deconv4_norm_attention(self.deconv4_attention(x_attention)))
        result_attention = self.deconv5_attention(x_attention)
        softmax_ = torch.nn.Softmax(dim=1)
        result_attention = softmax_(result_attention)
        attention_local = result_attention[:, 0:1, :, :]
        attention_global = result_attention[:, 1:2, :, :]
        attention_local = attention_local.repeat(1, 3, 1, 1)
        attention_global = attention_global.repeat(1, 3, 1, 1)
        final_result = attention_local * result_local + attention_global * result_global
        return mask_1_3, mask_2_3, mask_3_3, mask_4_3, mask_5_3, mask_6_3, result_1, result_2, result_3, result_4, result_5, result_6, result_local, attention_local, result_global, attention_global, final_result


class ResnetGenerator_global(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_global, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(input_nc, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks1.weight_init(0, 0.02)
        self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks2.weight_init(0, 0.02)
        self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks3.weight_init(0, 0.02)
        self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks4.weight_init(0, 0.02)
        self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks5.weight_init(0, 0.02)
        self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks6.weight_init(0, 0.02)
        self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks7.weight_init(0, 0.02)
        self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks8.weight_init(0, 0.02)
        self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks9.weight_init(0, 0.02)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm = nn.InstanceNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64 + 1, 3, 2, 1, 1)
        self.deconv4_norm = nn.InstanceNorm2d(64)
        self.deconv5 = nn.Conv2d(64, output_nc, 7, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks1(x)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x)
        x = self.resnet_blocks9(x)
        x = F.relu(self.deconv3_norm(self.deconv3(x)))
        x = F.relu(self.deconv4_norm(self.deconv4(x)))
        attention = x[:, :1, :, :]
        x_feature = x[:, 1:, :, :]
        x_feature = F.pad(x_feature, (3, 3, 3, 3), 'reflect')
        result_global = torch.tanh(self.deconv5(x_feature))
        return result_global, attention


class ResnetGenerator_fusion(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_fusion, self).__init__()

    def forward(self, gloabl_result, local_result, attention_global, attention_local):
        attention = torch.cat((attention_global, attention_local), 1)
        softmax_ = torch.nn.Softmax(dim=1)
        attention_2 = softmax_(attention)
        attention_local = attention_2[:, 0:1, :, :]
        attention_global = attention_2[:, 1:2, :, :]
        result = attention_local * local_result + attention_global * gloabl_result
        local_attention = attention_local.repeat(1, 3, 1, 1)
        global_attention = attention_global.repeat(1, 3, 1, 1)
        return result, local_attention, global_attention


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Pix2PixModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.ByteTensor if self.use_gpu() else torch.ByteTensor
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss(reduction='none')
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)
            return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return netG, netD, netE

    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label']
            data['instance'] = data['instance']
            data['image'] = data['image']
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local, KLD_loss = self.generate_fake(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_local, pred_real_local = self.discriminate(input_semantics, result_local, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake_generated, True, for_discriminator=False) + self.criterionGAN(pred_fake_global, True, for_discriminator=False) + self.criterionGAN(pred_fake_local, True, for_discriminator=False)
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake_generated)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake_generated[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake_generated[i][j], pred_real_generated[i][j].detach()) + self.criterionFeat(pred_fake_global[i][j], pred_real_global[i][j].detach()) + self.criterionFeat(pred_fake_local[i][j], pred_real_local[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg + self.criterionVGG(result_global, real_image) * self.opt.lambda_vgg + self.criterionVGG(result_local, real_image) * self.opt.lambda_vgg
        if not self.opt.no_l1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * self.opt.lambda_l1 + self.criterionL1(result_global, real_image) * self.opt.lambda_l1 + self.criterionL1(result_local, real_image) * self.opt.lambda_l1
        if not self.opt.no_class_loss:
            G_losses['class'] = torch.sum(self.criterionCE(feature_score, target) * index) / torch.sum(index) * self.opt.lambda_class
        if not self.opt.no_l1_local_loss:
            real_image = real_image
            real_0 = real_image * label_3_0
            real_1 = real_image * label_3_1
            real_2 = real_image * label_3_2
            real_3 = real_image * label_3_3
            real_4 = real_image * label_3_4
            real_5 = real_image * label_3_5
            real_6 = real_image * label_3_6
            real_7 = real_image * label_3_7
            real_8 = real_image * label_3_8
            real_9 = real_image * label_3_9
            real_10 = real_image * label_3_10
            real_11 = real_image * label_3_11
            real_12 = real_image * label_3_12
            real_13 = real_image * label_3_13
            real_14 = real_image * label_3_14
            real_15 = real_image * label_3_15
            real_16 = real_image * label_3_16
            real_17 = real_image * label_3_17
            real_18 = real_image * label_3_18
            real_19 = real_image * label_3_19
            real_20 = real_image * label_3_20
            real_21 = real_image * label_3_21
            real_22 = real_image * label_3_22
            real_23 = real_image * label_3_23
            real_24 = real_image * label_3_24
            real_25 = real_image * label_3_25
            real_26 = real_image * label_3_26
            real_27 = real_image * label_3_27
            real_28 = real_image * label_3_28
            real_29 = real_image * label_3_29
            real_30 = real_image * label_3_30
            real_31 = real_image * label_3_31
            real_32 = real_image * label_3_32
            real_33 = real_image * label_3_33
            real_34 = real_image * label_3_34
            G_losses['L1_Local'] = self.opt.lambda_l1 * (self.criterionL1(result_0, real_0) + self.criterionL1(result_1, real_1) + self.criterionL1(result_2, real_2) + self.criterionL1(result_3, real_3) + self.criterionL1(result_4, real_4) + self.criterionL1(result_5, real_5) + self.criterionL1(result_6, real_6) + self.criterionL1(result_7, real_7) + self.criterionL1(result_8, real_8) + self.criterionL1(result_9, real_9) + self.criterionL1(result_10, real_10) + self.criterionL1(result_11, real_11) + self.criterionL1(result_12, real_12) + self.criterionL1(result_13, real_13) + self.criterionL1(result_14, real_14) + self.criterionL1(result_15, real_15) + self.criterionL1(result_16, real_16) + self.criterionL1(result_17, real_17) + self.criterionL1(result_18, real_18) + self.criterionL1(result_19, real_19) + self.criterionL1(result_20, real_20) + self.criterionL1(result_21, real_21) + self.criterionL1(result_22, real_22) + self.criterionL1(result_23, real_23) + self.criterionL1(result_24, real_24) + self.criterionL1(result_25, real_25) + self.criterionL1(result_26, real_26) + self.criterionL1(result_27, real_27) + self.criterionL1(result_28, real_28) + self.criterionL1(result_29, real_29) + self.criterionL1(result_30, real_30) + self.criterionL1(result_31, real_31) + self.criterionL1(result_32, real_32) + self.criterionL1(result_33, real_33) + self.criterionL1(result_34, real_34))
        return G_losses, fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, valid_index, attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_lcoal, pred_real_local = self.discriminate(input_semantics, result_local, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake_generated, False, for_discriminator=True) + self.criterionGAN(pred_fake_global, False, for_discriminator=True) + self.criterionGAN(pred_fake_lcoal, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real_generated, True, for_discriminator=True) + self.criterionGAN(pred_real_global, True, for_discriminator=True) + self.criterionGAN(pred_real_local, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local = self.netG(input_semantics, z=z)
        assert not compute_kld_loss or self.opt.use_vae, 'You cannot compute KLD loss if opt.use_vae == False'
        return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, index, attention_global, attention_local, KLD_loss

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'instance':
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


class SPADEResnetBlock(nn.Module):

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class MultiscaleDiscriminator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator', 'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


def if_all_zero(tensor):
    if torch.sum(tensor) == 0:
        return 0
    else:
        return 1


class LGGANGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        final_nc = nf
        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(35 + 1, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv5_norm = nn.InstanceNorm2d(1024)
        self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks1.weight_init(0, 0.02)
        self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks2.weight_init(0, 0.02)
        self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks3.weight_init(0, 0.02)
        self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks4.weight_init(0, 0.02)
        self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks5.weight_init(0, 0.02)
        self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks6.weight_init(0, 0.02)
        self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks7.weight_init(0, 0.02)
        self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks8.weight_init(0, 0.02)
        self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks9.weight_init(0, 0.02)
        self.deconv3_local = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_local = nn.InstanceNorm2d(128)
        self.deconv4_local = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_local = nn.InstanceNorm2d(64)
        self.deconv9 = nn.Conv2d(3 * 35, 3, 3, 1, 1)
        self.deconv5_0 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_1 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_2 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_3 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_4 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_5 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_6 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_7 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_8 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_9 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_10 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_11 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_12 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_13 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_14 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_15 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_16 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_17 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_18 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_19 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_20 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_21 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_22 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_23 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_24 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_25 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_26 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_27 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_28 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_29 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_30 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_31 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_32 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_33 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_34 = nn.Conv2d(64, 3, 7, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(64, 35)
        self.deconv3_attention = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_attention = nn.InstanceNorm2d(128)
        self.deconv4_attention = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_attention = nn.InstanceNorm2d(64)
        self.deconv5_attention = nn.Conv2d(64, 2, 1, 1, 0)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        sw = opt.crop_size // 2 ** num_up_layers
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, z=None):
        seg = input
        if self.opt.use_vae:
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            x = F.pad(seg, (3, 3, 3, 3), 'reflect')
            x = F.relu(self.conv1_norm(self.conv1(x)))
            x = F.relu(self.conv2_norm(self.conv2(x)))
            x_encode = F.relu(self.conv3_norm(self.conv3(x)))
            x = F.relu(self.conv4_norm(self.conv4(x_encode)))
            x = F.relu(self.conv5_norm(self.conv5(x)))
            x = F.interpolate(x, size=(self.sh, self.sw))
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 0.2))
        result_global = F.tanh(x)
        label = input[:, 0:self.opt.label_nc, :, :]
        for i in range(self.opt.label_nc):
            globals()['label_' + str(i)] = label[:, i:i + 1, :, :]
            globals()['label_3_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 3, 1, 1)
            globals()['label_64_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 64, 1, 1)
        x = self.resnet_blocks1(x_encode)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x)
        middle_x = self.resnet_blocks9(x)
        x_local = F.relu(self.deconv3_norm_local(self.deconv3_local(middle_x)))
        x_feature_local = F.relu(self.deconv4_norm_local(self.deconv4_local(x_local)))
        feature_0 = x_feature_local * label_64_0
        feature_1 = x_feature_local * label_64_1
        feature_2 = x_feature_local * label_64_2
        feature_3 = x_feature_local * label_64_3
        feature_4 = x_feature_local * label_64_4
        feature_5 = x_feature_local * label_64_5
        feature_6 = x_feature_local * label_64_6
        feature_7 = x_feature_local * label_64_7
        feature_8 = x_feature_local * label_64_8
        feature_9 = x_feature_local * label_64_9
        feature_10 = x_feature_local * label_64_10
        feature_11 = x_feature_local * label_64_11
        feature_12 = x_feature_local * label_64_12
        feature_13 = x_feature_local * label_64_13
        feature_14 = x_feature_local * label_64_14
        feature_15 = x_feature_local * label_64_15
        feature_16 = x_feature_local * label_64_16
        feature_17 = x_feature_local * label_64_17
        feature_18 = x_feature_local * label_64_18
        feature_19 = x_feature_local * label_64_19
        feature_20 = x_feature_local * label_64_20
        feature_21 = x_feature_local * label_64_21
        feature_22 = x_feature_local * label_64_22
        feature_23 = x_feature_local * label_64_23
        feature_24 = x_feature_local * label_64_24
        feature_25 = x_feature_local * label_64_25
        feature_26 = x_feature_local * label_64_26
        feature_27 = x_feature_local * label_64_27
        feature_28 = x_feature_local * label_64_28
        feature_29 = x_feature_local * label_64_29
        feature_30 = x_feature_local * label_64_30
        feature_31 = x_feature_local * label_64_31
        feature_32 = x_feature_local * label_64_32
        feature_33 = x_feature_local * label_64_33
        feature_34 = x_feature_local * label_64_34
        feature_combine = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34), 0)
        feature_combine = self.avgpool(feature_combine)
        feature_combine_fc = torch.flatten(feature_combine, 1)
        feature_score = self.fc2(feature_combine_fc)
        target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
        valid_index = torch.tensor([if_all_zero(label_0), if_all_zero(label_1), if_all_zero(label_2), if_all_zero(label_3), if_all_zero(label_4), if_all_zero(label_5), if_all_zero(label_6), if_all_zero(label_7), if_all_zero(label_8), if_all_zero(label_9), if_all_zero(label_10), if_all_zero(label_11), if_all_zero(label_12), if_all_zero(label_13), if_all_zero(label_14), if_all_zero(label_15), if_all_zero(label_16), if_all_zero(label_17), if_all_zero(label_18), if_all_zero(label_19), if_all_zero(label_20), if_all_zero(label_21), if_all_zero(label_22), if_all_zero(label_23), if_all_zero(label_24), if_all_zero(label_25), if_all_zero(label_26), if_all_zero(label_27), if_all_zero(label_28), if_all_zero(label_29), if_all_zero(label_30), if_all_zero(label_31), if_all_zero(label_32), if_all_zero(label_33), if_all_zero(label_34)])
        feature_0 = F.pad(feature_0, (3, 3, 3, 3), 'reflect')
        feature_1 = F.pad(feature_1, (3, 3, 3, 3), 'reflect')
        feature_2 = F.pad(feature_2, (3, 3, 3, 3), 'reflect')
        feature_3 = F.pad(feature_3, (3, 3, 3, 3), 'reflect')
        feature_4 = F.pad(feature_4, (3, 3, 3, 3), 'reflect')
        feature_5 = F.pad(feature_5, (3, 3, 3, 3), 'reflect')
        feature_6 = F.pad(feature_6, (3, 3, 3, 3), 'reflect')
        feature_7 = F.pad(feature_7, (3, 3, 3, 3), 'reflect')
        feature_8 = F.pad(feature_8, (3, 3, 3, 3), 'reflect')
        feature_9 = F.pad(feature_9, (3, 3, 3, 3), 'reflect')
        feature_10 = F.pad(feature_10, (3, 3, 3, 3), 'reflect')
        feature_11 = F.pad(feature_11, (3, 3, 3, 3), 'reflect')
        feature_12 = F.pad(feature_12, (3, 3, 3, 3), 'reflect')
        feature_13 = F.pad(feature_13, (3, 3, 3, 3), 'reflect')
        feature_14 = F.pad(feature_14, (3, 3, 3, 3), 'reflect')
        feature_15 = F.pad(feature_15, (3, 3, 3, 3), 'reflect')
        feature_16 = F.pad(feature_16, (3, 3, 3, 3), 'reflect')
        feature_17 = F.pad(feature_17, (3, 3, 3, 3), 'reflect')
        feature_18 = F.pad(feature_18, (3, 3, 3, 3), 'reflect')
        feature_19 = F.pad(feature_19, (3, 3, 3, 3), 'reflect')
        feature_20 = F.pad(feature_20, (3, 3, 3, 3), 'reflect')
        feature_21 = F.pad(feature_21, (3, 3, 3, 3), 'reflect')
        feature_22 = F.pad(feature_22, (3, 3, 3, 3), 'reflect')
        feature_23 = F.pad(feature_23, (3, 3, 3, 3), 'reflect')
        feature_24 = F.pad(feature_24, (3, 3, 3, 3), 'reflect')
        feature_25 = F.pad(feature_25, (3, 3, 3, 3), 'reflect')
        feature_26 = F.pad(feature_26, (3, 3, 3, 3), 'reflect')
        feature_27 = F.pad(feature_27, (3, 3, 3, 3), 'reflect')
        feature_28 = F.pad(feature_28, (3, 3, 3, 3), 'reflect')
        feature_29 = F.pad(feature_29, (3, 3, 3, 3), 'reflect')
        feature_30 = F.pad(feature_30, (3, 3, 3, 3), 'reflect')
        feature_31 = F.pad(feature_31, (3, 3, 3, 3), 'reflect')
        feature_32 = F.pad(feature_32, (3, 3, 3, 3), 'reflect')
        feature_33 = F.pad(feature_33, (3, 3, 3, 3), 'reflect')
        feature_34 = F.pad(feature_34, (3, 3, 3, 3), 'reflect')
        result_0 = torch.tanh(self.deconv5_0(feature_0))
        result_1 = torch.tanh(self.deconv5_1(feature_1))
        result_2 = torch.tanh(self.deconv5_2(feature_2))
        result_3 = torch.tanh(self.deconv5_3(feature_3))
        result_4 = torch.tanh(self.deconv5_4(feature_4))
        result_5 = torch.tanh(self.deconv5_5(feature_5))
        result_6 = torch.tanh(self.deconv5_6(feature_6))
        result_7 = torch.tanh(self.deconv5_7(feature_7))
        result_8 = torch.tanh(self.deconv5_8(feature_8))
        result_9 = torch.tanh(self.deconv5_9(feature_9))
        result_10 = torch.tanh(self.deconv5_10(feature_10))
        result_11 = torch.tanh(self.deconv5_11(feature_11))
        result_12 = torch.tanh(self.deconv5_12(feature_12))
        result_13 = torch.tanh(self.deconv5_13(feature_13))
        result_14 = torch.tanh(self.deconv5_14(feature_14))
        result_15 = torch.tanh(self.deconv5_15(feature_15))
        result_16 = torch.tanh(self.deconv5_16(feature_16))
        result_17 = torch.tanh(self.deconv5_17(feature_17))
        result_18 = torch.tanh(self.deconv5_18(feature_18))
        result_19 = torch.tanh(self.deconv5_19(feature_19))
        result_20 = torch.tanh(self.deconv5_20(feature_20))
        result_21 = torch.tanh(self.deconv5_21(feature_21))
        result_22 = torch.tanh(self.deconv5_22(feature_22))
        result_23 = torch.tanh(self.deconv5_23(feature_23))
        result_24 = torch.tanh(self.deconv5_24(feature_24))
        result_25 = torch.tanh(self.deconv5_25(feature_25))
        result_26 = torch.tanh(self.deconv5_26(feature_26))
        result_27 = torch.tanh(self.deconv5_27(feature_27))
        result_28 = torch.tanh(self.deconv5_28(feature_28))
        result_29 = torch.tanh(self.deconv5_29(feature_29))
        result_30 = torch.tanh(self.deconv5_30(feature_30))
        result_31 = torch.tanh(self.deconv5_31(feature_31))
        result_32 = torch.tanh(self.deconv5_32(feature_32))
        result_33 = torch.tanh(self.deconv5_33(feature_33))
        result_34 = torch.tanh(self.deconv5_34(feature_34))
        combine_local = torch.cat((result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34), 1)
        result_local = torch.tanh(self.deconv9(combine_local))
        x_attention = F.relu(self.deconv3_norm_attention(self.deconv3_attention(x_encode)))
        x_attention = F.relu(self.deconv4_norm_attention(self.deconv4_attention(x_attention)))
        result_attention = self.deconv5_attention(x_attention)
        softmax_ = torch.nn.Softmax(dim=1)
        result_attention = softmax_(result_attention)
        attention_local = result_attention[:, 0:1, :, :]
        attention_global = result_attention[:, 1:2, :, :]
        attention_local = attention_local.repeat(1, 3, 1, 1)
        attention_global = attention_global.repeat(1, 3, 1, 1)
        final = attention_local * result_local + attention_global * result_global
        return final, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, label_3_21, label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_26, label_3_28, label_3_29, label_3_30, label_3_31, label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9, result_10, result_11, result_12, result_13, result_14, result_15, result_16, result_17, result_18, result_19, result_20, result_21, result_22, result_23, result_24, result_25, result_26, result_27, result_28, result_29, result_30, result_31, result_32, result_33, result_34, feature_score, target, valid_index.float(), attention_global, attention_local


class SPADEGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        final_nc = nf
        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        sw = opt.crop_size // 2 ** num_up_layers
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, z=None):
        seg = input
        if self.opt.use_vae:
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 0.2))
        x = F.tanh(x)
        return x


class Pix2PixHDGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3, help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7, help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)
        model = []
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2), norm_layer(nn.Conv2d(input_nc, opt.ngf, kernel_size=opt.resnet_initial_kernel_size, padding=0)), activation]
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2, kernel_size=3, stride=2, padding=1)), activation]
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)]
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int(opt.ngf * mult / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out, kernel_size=3, stride=2, padding=1, output_padding=1)), activation]
            mult = mult // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (KLDLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {'opt': _mock_config(ndf=4, label_nc=4, output_nc=4, contain_dontcare_label=4, no_instance=4, norm_D=4, n_layers_D=1, no_ganFeat_loss=MSELoss())}),
     lambda: ([torch.rand([4, 9, 64, 64])], {}),
     False),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResnetGenerator_fusion,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGGLoss,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_Ha0Tang_LGGAN(_paritybench_base):
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

