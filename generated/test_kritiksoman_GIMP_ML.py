import sys
_module = sys.modules[__name__]
del sys
data = _module
aligned_dataset = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
image_folder = _module
models = _module
base_model = _module
models = _module
networks = _module
pix2pixHD_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
util = _module
image_pool = _module
util = _module
adversarial_trainer = _module
aug = _module
dataset = _module
metric_counter = _module
fpn_densenet = _module
fpn_inception = _module
fpn_inception_simple = _module
fpn_mobilenet = _module
losses = _module
mobilenet_v2 = _module
models = _module
networks = _module
senet = _module
unet_seresnext = _module
predict = _module
predictorClass = _module
schedulers = _module
test_aug = _module
test_dataset = _module
test_metrics = _module
testing = _module
train = _module
image_pool = _module
metrics = _module
colorize = _module
deblur = _module
deeplabv3 = _module
evaluate = _module
face_dataset = _module
logger = _module
loss = _module
makeup = _module
model = _module
modules = _module
bn = _module
deeplab = _module
dense = _module
functions = _module
misc = _module
residual = _module
optimizer = _module
prepropess_data = _module
resnet = _module
test = _module
train = _module
transform = _module
facegen = _module
faceparse = _module
invert = _module
monodepth = _module
evaluate_depth = _module
evaluate_pose = _module
export_gt_depth = _module
layers = _module
depth_decoder = _module
pose_cnn = _module
pose_decoder = _module
resnet_encoder = _module
test_simple = _module
trainer = _module
utils = _module
build_dataset_directory = _module
colorize = _module
model = _module
resize_all_imgs = _module
train = _module
dataset = _module
demo = _module
eval = _module
main_srresnet = _module
srresnet = _module
super_resolution = _module

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


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import torch.utils.data


import torch.nn as nn


import functools


from torch.autograd import Variable


import torch.nn.functional as F


from torchvision import models


import copy


from copy import deepcopy


from functools import partial


from typing import Callable


from typing import Iterable


from typing import Optional


from typing import Tuple


from torch.utils.data import Dataset


from torchvision.models import resnet50


from torchvision.models import densenet121


from torchvision.models import densenet201


import torch.autograd as autograd


import torchvision.models as models


import math


from torch.nn import init


from collections import OrderedDict


from torch.utils import model_zoo


from torch import nn


import torch.nn.parallel


import torch.optim


from torch.nn import Sequential


import torchvision


from torch.nn import functional as F


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


from torchvision import transforms


import logging


import torch.optim as optim


from collections import deque


from math import exp


from scipy.ndimage import zoom


from torchvision import datasets


import torch.distributed as dist


import time


import torch.nn.functional as functional


import torch.cuda.comm as comm


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import load


import torch.utils.model_zoo as modelzoo


import torch.utils.model_zoo as model_zoo


import matplotlib as mpl


import matplotlib.cm as cm


from functools import reduce


from torch.utils import data


import scipy.io as sio


import matplotlib.pyplot as plt


import torch.backends.cudnn as cudnn


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        None
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
            if network_label == 'G':
                raise 'Generator must exist!'
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        None
                except:
                    None
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        not_initialized = Set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    None
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass


class GANLoss(nn.Module):

    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


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


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class LabelEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(LabelEncoder, self).__init__()
        self.model = []
        self.model_last = [nn.ReLU()]
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type)]
        dim *= 2
        for i in range(n_downsample - 3):
            self.model_last += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_last += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.model_last = nn.Sequential(*self.model_last)
        self.output_dim = dim

    def forward(self, x):
        fea = self.model(x)
        return fea, self.model_last(fea)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SFTLayer(nn.Module):

    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv2 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv2 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv2(F.leaky_relu(self.SFT_scale_conv1(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv2(F.leaky_relu(self.SFT_shift_conv1(x[1]), 0.1, inplace=True))
        return x[0] * scale + shift


class StyleEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model_middle = []
        self.model_last = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model_middle += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_last += [nn.AdaptiveAvgPool2d(1)]
        self.model_last += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.model_middle = nn.Sequential(*self.model_middle)
        self.model_last = nn.Sequential(*self.model_last)
        self.output_dim = dim
        self.sft1 = SFTLayer()
        self.sft2 = SFTLayer()

    def forward(self, x):
        fea = self.model(x[0])
        fea = self.sft1((fea, x[1]))
        fea = self.model_middle(fea)
        fea = self.sft2((fea, x[2]))
        return self.model_last(fea)


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='adain', padding_type=padding_type)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.enc_style = StyleEncoder(5, 3, 16, self.get_num_adain_params(self.model), norm='none', activ='relu', pad_type='reflect')
        self.enc_label = LabelEncoder(5, 19, 16, 64, norm='none', activ='relu', pad_type='reflect')

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
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

    def forward(self, input, input_ref, image_ref):
        fea1, fea2 = self.enc_label(input_ref)
        adain_params = self.enc_style((image_ref, fea1, fea2))
        self.assign_adain_params(adain_params, self.model)
        return self.model(input)


class BlendGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(BlendGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='in', padding_type=padding_type)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input1, input2):
        m = self.model(torch.cat([input1, input2], 1))
        return input1 * m + input2 * (1 - m), m


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class VAE(nn.Module):

    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.e5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.e6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf * 32)
        self.e7 = nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(ndf * 64)
        self.fc1 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)
        self.fc2 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)
        self.d1 = nn.Linear(latent_variable_size, ngf * 64 * 4 * 4)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf * 64, ngf * 32, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf * 32, 0.001)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf * 32, ngf * 16, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf * 16, 0.001)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf * 16, ngf * 8, 3, 1)
        self.bn10 = nn.BatchNorm2d(ngf * 8, 0.001)
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        self.bn11 = nn.BatchNorm2d(ngf * 4, 0.001)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1)
        self.bn12 = nn.BatchNorm2d(ngf * 2, 0.001)
        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(ngf * 2, ngf, 3, 1)
        self.bn13 = nn.BatchNorm2d(ngf, 0.001)
        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd7 = nn.ReplicationPad2d(1)
        self.d8 = nn.Conv2d(ngf, nc, 3, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.leakyrelu(self.bn6(self.e6(h5)))
        h7 = self.leakyrelu(self.bn7(self.e7(h6)))
        h7 = h7.view(-1, self.ndf * 64 * 4 * 4)
        return self.fc1(h7), self.fc2(h7)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf * 64, 4, 4)
        h2 = self.leakyrelu(self.bn8(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn9(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn10(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn11(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.leakyrelu(self.bn12(self.d6(self.pd5(self.up5(h5)))))
        h7 = self.leakyrelu(self.bn13(self.d7(self.pd6(self.up6(h6)))))
        return self.d8(self.pd7(self.up7(h7)))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar.mul(0.5).exp_()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, x, mu, logvar


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
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

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
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
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


class ConvBlock_SFT(nn.Module):

    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock_SFT, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return x[0] + fea, x[1]


class ConvBlock_SFT_last(nn.Module):

    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock_SFT_last, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return x[0] + fea


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


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = True, use_gan_feat_loss, use_vgg_loss, True, use_gan_feat_loss, use_vgg_loss, True, True, True, True

        def loss_filter(g_gan, g_gan_feat, g_vgg, gb_gan, gb_gan_feat, gb_vgg, d_real, d_fake, d_blend):
            return [l for l, f in zip((g_gan, g_gan_feat, g_vgg, gb_gan, gb_gan_feat, gb_vgg, d_real, d_fake, d_blend), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        netG_input_nc = input_nc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netB_input_nc = opt.output_nc * 2
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netB = networks.define_B(netB_input_nc, opt.output_nc, 32, 3, 3, opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            None
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            None
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netB, 'B', opt.which_epoch, pretrained_path)
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'GB_GAN', 'GB_GAN_Feat', 'GB_VGG', 'D_real', 'D_fake', 'D_blend')
            if opt.niter_fix_global > 0:
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    finetune_list = Set()
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                None
                None
            else:
                params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netG.parameters()) + list(self.netB.parameters())
            self.optimizer_GB = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, inter_label_map_1, label_map, inter_label_map_2, real_image, label_map_ref, real_image_ref, infer=False):
        if self.opt.label_nc == 0:
            if torch.cuda.is_available():
                input_label = label_map.data
                inter_label_1 = inter_label_map_1.data
                inter_label_2 = inter_label_map_2.data
                input_label_ref = label_map_ref.data
            else:
                input_label = label_map.data
                inter_label_1 = inter_label_map_1.data
                inter_label_2 = inter_label_map_2.data
                input_label_ref = label_map_ref.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            if torch.cuda.is_available():
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
                inter_label_1 = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                inter_label_1 = inter_label_1.scatter_(1, inter_label_map_1.data.long(), 1.0)
                inter_label_2 = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                inter_label_2 = inter_label_2.scatter_(1, inter_label_map_2.data.long(), 1.0)
                input_label_ref = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long(), 1.0)
            else:
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
                inter_label_1 = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                inter_label_1 = inter_label_1.scatter_(1, inter_label_map_1.data.long(), 1.0)
                inter_label_2 = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                inter_label_2 = inter_label_2.scatter_(1, inter_label_map_2.data.long(), 1.0)
                input_label_ref = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
                inter_label_1 = inter_label_1.half()
                inter_label_2 = inter_label_2.half()
                input_label_ref = input_label_ref.half()
        input_label = Variable(input_label, volatile=infer)
        inter_label_1 = Variable(inter_label_1, volatile=infer)
        inter_label_2 = Variable(inter_label_2, volatile=infer)
        input_label_ref = Variable(input_label_ref, volatile=infer)
        if torch.cuda.is_available():
            real_image = Variable(real_image.data)
            real_image_ref = Variable(real_image_ref.data)
        else:
            real_image = Variable(real_image.data)
            real_image_ref = Variable(real_image_ref.data)
        return inter_label_1, input_label, inter_label_2, real_image, input_label_ref, real_image_ref

    def encode_input_test(self, label_map, label_map_ref, real_image_ref, infer=False):
        if self.opt.label_nc == 0:
            if torch.cuda.is_available():
                input_label = label_map.data
                input_label_ref = label_map_ref.data
            else:
                input_label = label_map.data
                input_label_ref = label_map_ref.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            if torch.cuda.is_available():
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
                input_label_ref = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long(), 1.0)
                real_image_ref = Variable(real_image_ref.data)
            else:
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
                input_label_ref = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long(), 1.0)
                real_image_ref = Variable(real_image_ref.data)
            if self.opt.data_type == 16:
                input_label = input_label.half()
                input_label_ref = input_label_ref.half()
        input_label = Variable(input_label, volatile=infer)
        input_label_ref = Variable(input_label_ref, volatile=infer)
        return input_label, input_label_ref, real_image_ref

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, inter_label_1, label, inter_label_2, image, label_ref, image_ref, infer=False):
        inter_label_1, input_label, inter_label_2, real_image, input_label_ref, real_image_ref = self.encode_input(inter_label_1, label, inter_label_2, image, label_ref, image_ref)
        fake_inter_1 = self.netG.forward(inter_label_1, input_label, real_image)
        fake_image = self.netG.forward(input_label, input_label, real_image)
        fake_inter_2 = self.netG.forward(inter_label_2, input_label, real_image)
        blend_image, alpha = self.netB.forward(fake_inter_1, fake_inter_2)
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_blend_pool = self.discriminate(input_label, blend_image, use_pool=True)
        loss_D_blend = self.criterionGAN(pred_blend_pool, False)
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        pred_blend = self.netD.forward(torch.cat((input_label, blend_image), dim=1))
        loss_GB_GAN = self.criterionGAN(pred_blend, True)
        loss_G_GAN_Feat = 0
        loss_GB_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    loss_GB_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_blend[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        loss_GB_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            loss_GB_VGG += self.criterionVGG(blend_image, real_image) * self.opt.lambda_feat
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_GB_GAN, loss_GB_GAN_Feat, loss_GB_VGG, loss_D_real, loss_D_fake, loss_D_blend), None if not infer else fake_inter_1, fake_image, fake_inter_2, blend_image, alpha, real_image, inter_label_1, input_label, inter_label_2]

    def inference(self, label, label_ref, image_ref):
        image_ref = Variable(image_ref)
        input_label, input_label_ref, real_image_ref = self.encode_input_test(Variable(label), Variable(label_ref), image_ref, infer=True)
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        else:
            fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netB, 'B', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            None
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):

    def forward(self, inp):
        label = inp
        return self.inference(label)


class FPNSegHead(nn.Module):

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """
        super().__init__()
        net = MobileNetV2(n_class=1000)
        if pretrained:
            state_dict = torch.load('mobilenetv2.pth.tar')
            net.load_state_dict(state_dict)
        self.features = net.features
        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return lateral0, map1, map2, map3, map4


class FPNDense(nn.Module):

    def __init__(self, output_ch=3, num_filters=128, num_filters_fpn=256, pretrained=True):
        super().__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, pretrained=pretrained)
        self.head1 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNSegHead(num_filters_fpn, num_filters, num_filters)
        self.smooth = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1), nn.BatchNorm2d(num_filters // 2), nn.ReLU())
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode='nearest')
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        final = self.final(smoothed)
        nn.Tanh(final)


class FPNHead(nn.Module):

    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super(FPNInception, self).__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.smooth = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1), norm_layer(num_filters // 2), nn.ReLU())
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode='nearest')
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        final = self.final(smoothed)
        res = torch.tanh(final) + x
        return torch.clamp(res, min=-1, max=1)


class FPNInceptionSimple(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.smooth = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1), norm_layer(num_filters // 2), nn.ReLU())
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode='nearest')
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        final = self.final(smoothed)
        res = torch.tanh(final) + x
        return torch.clamp(res, min=-1, max=1)


class FPNMobileNet(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.smooth = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1), norm_layer(num_filters // 2), nn.ReLU())
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode='nearest')
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode='nearest')
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode='nearest')
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode='nearest')
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode='nearest')
        final = self.final(smoothed)
        res = torch.tanh(final) + x
        return torch.clamp(res, min=-1, max=1)


class DiscLoss(nn.Module):

    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()
        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):

    def name(self):
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()
        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(50)
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        self.pred_fake = net.forward(fakeB)
        self.pred_real = net.forward(realB)
        errG = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 0) + self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)
        self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) + self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLossLS(nn.Module):

    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        self.pred_fake = net.forward(fakeB)
        self.pred_real = net.forward(realB)
        errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) + torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)
        self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) + torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class DiscLossLS(DiscLoss):

    def name(self):
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        return DiscLoss.get_loss(self, net, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):

    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class DeblurModel(nn.Module):

    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs, targets
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, use_parallel=True, learn_residual=True, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
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
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


class DicsriminatorTail(nn.Module):

    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8
        self.scale_two = nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(planes * 2, affine=False)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.bn2 = nn.InstanceNorm2d(planes * 4, affine=False)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(planes * 4, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.InstanceNorm2d(planes, affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups)
        self.bn2 = nn.InstanceNorm2d(planes, affine=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(planes * 4, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1)
        self.bn1 = nn.InstanceNorm2d(width, affine=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.bn2 = nn.InstanceNorm2d(width, affine=False)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(planes * 4, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1)), ('bn1', nn.InstanceNorm2d(64, affine=False)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1)), ('bn2', nn.InstanceNorm2d(64, affine=False)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1)), ('bn3', nn.InstanceNorm2d(inplanes, affine=False)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3)), ('bn1', nn.InstanceNorm2d(inplanes, affine=False)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding), nn.InstanceNorm2d(planes * block.expansion, affine=False))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvRelu(nn.Module):

    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV, self).__init__()
        self.in_channels = in_channels
        if is_deconv:
            self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(out_channels, affine=False), nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), ConvRelu(in_channels, middle_channels), ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class DecoderCenter(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels
        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(out_channels, affine=False), nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    return model


class UNetSEResNext(nn.Module):

    def __init__(self, num_classes=3, num_filters=32, pretrained=True, is_deconv=True):
        super().__init__()
        self.num_classes = num_classes
        pretrain = 'imagenet' if pretrained is True else None
        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained=pretrain)
        bottom_channel_nr = 2048
        self.conv1 = self.encoder.layer0
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, False)
        self.dec5 = DecoderBlockV(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 2, is_deconv)
        self.dec4 = DecoderBlockV(bottom_channel_nr // 2 + num_filters * 2, num_filters * 8, num_filters * 2, is_deconv)
        self.dec3 = DecoderBlockV(bottom_channel_nr // 4 + num_filters * 2, num_filters * 4, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2, num_filters * 2, is_deconv)
        self.dec1 = DecoderBlockV(num_filters * 2, num_filters, num_filters * 2, is_deconv)
        self.dec0 = ConvRelu(num_filters * 10, num_filters * 2)
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        f = torch.cat((dec1, F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False), F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False), F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False), F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False)), 1)
        dec0 = self.dec0(f)
        return self.final(dec0)


class OhemCELoss(nn.Module):

    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):

    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.0 - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_chan))

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ContextPath(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        return feat8, feat16_up, feat32_up

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):

    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):

    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


ACT_ELU = 'elu'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


ACT_NONE = 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABN(autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * count / (count - 1))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz * weight.sign() if ctx.affine else None
        dbias = edz if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01, equal_batches=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size = x.new_tensor([x.shape[0]], dtype=torch.long)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.world_size > 1:
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)
                ctx.factor = x.shape[0] / float(batch_size.item())
                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)
                var_all = (var + (mean - mean_all) ** 2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)
                mean = mean_all
                var = var_all
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0], x.shape[1], -1).shape[-1]
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * (float(count) / (count - 1)))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            edz_local = edz.clone()
            eydz_local = eydz.clone()
            if ctx.world_size > 1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)
                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz_local * weight.sign() if ctx.affine else None
        dbias = edz_local if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class DeeplabV3(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=256, dilations=(12, 24, 36), norm_act=ABN, pooling_size=None):
        super(DeeplabV3, self).__init__()
        self.pooling_size = pooling_size
        self.map_convs = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, 1, bias=False), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])])
        self.map_bn = norm_act(hidden_channels * 4)
        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)
        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)
        self.reset_parameters(self.map_bn.activation, self.map_bn.slope)

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ABN):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))
        out += pool
        out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = min(try_index(self.pooling_size, 0), x.shape[2]), min(try_index(self.pooling_size, 1), x.shape[3])
            padding = (pooling_size[1] - 1) // 2, (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1, (pooling_size[0] - 1) // 2, (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode='replicate')
        return pool


class DenseModule(nn.Module):

    def __init__(self, in_channels, growth, layers, bottleneck_factor=4, norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([('bn', norm_act(in_channels)), ('conv', nn.Conv2d(in_channels, self.growth * bottleneck_factor, 1, bias=False))])))
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(self.growth * bottleneck_factor)), ('conv', nn.Conv2d(self.growth * bottleneck_factor, self.growth, 3, padding=dilation, bias=False, dilation=dilation))])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):

    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, input):
        return self.module(input)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)), ('bn2', norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups, dilation=dilation)), ('bn3', norm_act(channels[1])), ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-07):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, (2), :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')


class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs['dispconv', s] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs['upconv', i, 0](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs['upconv', i, 1](x)
            if i in self.scales:
                self.outputs['disp', i] = self.sigmoid(self.convs['dispconv', i](x))
        return self.outputs


class PoseCNN(nn.Module):

    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)
        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
        out = self.pose_conv(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        axisangle = out[(...), :3]
        translation = out[(...), 3:]
        return axisangle, translation


class PoseDecoder(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.convs = OrderedDict()
        self.convs['squeeze'] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs['pose', 0] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs['pose', 1] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs['pose', 2] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs['squeeze'](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        for i in range(3):
            out = self.convs['pose', i](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[(...), :3]
        translation = out[(...), 3:]
        return axisangle, translation


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'
    blocks = {(18): [2, 2, 2, 2], (50): [3, 4, 6, 3]}[num_layers]
    block_type = {(18): models.resnet.BasicBlock, (50): models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {(18): models.resnet18, (34): models.resnet34, (50): models.resnet50, (101): models.resnet101, (152): models.resnet152}
        if num_layers not in resnets:
            raise ValueError('{} is not a valid number of resnet layers'.format(num_layers))
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


class shave_block(nn.Module):

    def __init__(self, s):
        super(shave_block, self).__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, self.s:-self.s, self.s:-self.s]


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


class _Residual_Block(nn.Module):

    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _NetG(nn.Module):

    def __init__(self):
        super(_NetG, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block, 16)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        self.upscale4x = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True))
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out


class _NetD(nn.Module):

    def __init__(self):
        super(_NetD, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ABN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionRefinementModule,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BackprojectDepth,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiSeNet,
     lambda: ([], {'n_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (BiSeNetOutput,
     lambda: ([], {'in_chan': 4, 'mid_chan': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvRelu,
     lambda: ([], {'in_': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlockV,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderCenter,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeeplabV3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseModule,
     lambda: ([], {'in_channels': 4, 'growth': 4, 'layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DicsriminatorTail,
     lambda: ([], {'nf_mult': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (FPNHead,
     lambda: ([], {'num_in': 4, 'num_mid': 4, 'num_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FPNSegHead,
     lambda: ([], {'num_in': 4, 'num_mid': 4, 'num_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureFusionModule,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityResidualBlock,
     lambda: ([], {'in_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PoseCNN,
     lambda: ([], {'num_input_frames': 4}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (Project3D,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResnetGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SFTLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleGPU,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UNetSEResNext,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VAE,
     lambda: ([], {'nc': 4, 'ngf': 4, 'ndf': 4, 'latent_variable_size': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
    (VGGLoss,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_NetG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_Residual_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (shave_block,
     lambda: ([], {'s': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kritiksoman_GIMP_ML(_paritybench_base):
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

