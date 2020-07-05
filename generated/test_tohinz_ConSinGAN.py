import sys
_module = sys.modules[__name__]
del sys
config = _module
functions = _module
imresize = _module
models = _module
training_generation = _module
training_harmonization_editing = _module
evaluate_model = _module
main_train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


import math


import random


import copy


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


from torchvision.utils import make_grid


import torch.utils.data


def get_activation(opt):
    activations = {'lrelu': nn.LeakyReLU(opt.lrelu_alpha, inplace=True), 'elu': nn.ELU(alpha=1.0, inplace=True), 'prelu': nn.PReLU(num_parameters=1, init=0.25), 'selu': nn.SELU(inplace=True)}
    return activations[opt.activation]


class ConvBlock(nn.Sequential):

    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt)
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
            self.body.add_module('block%d' % i, block)
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


def upsample(x, size):
    x_up = torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


class GrowingGenerator(nn.Module):

    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()
        self.opt = opt
        N = int(opt.nfc)
        self._pad = nn.ZeroPad2d(1)
        self._pad_block = nn.ZeroPad2d(opt.num_layer - 1) if opt.train_mode == 'generation' or opt.train_mode == 'animation' else nn.ZeroPad2d(opt.num_layer)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)
        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
            _first_stage.add_module('block%d' % i, block)
        self.body.append(_first_stage)
        self.tail = nn.Sequential(nn.Conv2d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size), nn.Tanh())

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))
        if self.opt.train_mode == 'generation' or self.opt.train_mode == 'animation':
            x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)
        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == 'generation' or self.opt.train_mode == 'animation':
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer * 2, real_shapes[idx][3] + self.opt.num_layer * 2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1 + noise[idx] * noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1
        out = self.tail(self._pad(x_prev_out))
        return out

