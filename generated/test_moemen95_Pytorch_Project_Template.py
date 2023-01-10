import sys
_module = sys.modules[__name__]
del sys
master = _module
agents = _module
base = _module
condensenet = _module
dcgan = _module
dqn = _module
erfnet = _module
example = _module
mnist = _module
datasets = _module
celebA = _module
cifar10 = _module
example = _module
mnist = _module
voc2012 = _module
graphs = _module
losses = _module
bce = _module
cross_entropy = _module
example = _module
huber_loss = _module
models = _module
condensenet = _module
custom_layers = _module
denseblock = _module
erf_blocks = _module
learnedgroupconv = _module
dcgan_discriminator = _module
dcgan_generator = _module
dqn = _module
erfnet = _module
erfnet_imagenet = _module
example = _module
mnist = _module
weights_initializer = _module
main = _module
mnist = _module
utils = _module
config = _module
dirs = _module
env_utils = _module
generate_class_weights = _module
metrics = _module
misc = _module
replay_memory = _module
train_utils = _module
voc_utils = _module

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


from torch import nn


from torch.backends import cudnn


from torch.autograd import Variable


import random


import torchvision.utils as vutils


import math


from torch.optim import lr_scheduler


import torch.optim as optim


import torch.nn.functional as F


import torchvision.transforms as v_transforms


import torchvision.utils as v_utils


import torchvision.datasets as v_datasets


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.utils.data import Dataset


import logging


from torchvision import datasets


from torchvision import transforms


import scipy.io as sio


from torch.utils import data


import torchvision.transforms as standard_transforms


import torch.nn as nn


import torch.nn.init as init


import torchvision.transforms as transforms


from sklearn.utils.class_weight import compute_class_weight


import time


class BinaryCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config == None:
            self.loss = nn.CrossEntropyLoss()
        else:
            class_weights = np.load(config.class_weights)
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index, weight=torch.from_numpy(class_weights.astype(np.float32)), size_average=True, reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss


class HuberLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss


class LearnedGroupConv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, condense_factor=None, dropout_rate=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condense_factor = condense_factor
        self.groups = groups
        self.dropout_rate = dropout_rate
        assert self.in_channels % self.groups == 0, 'group value is not divisible by input channels'
        assert self.in_channels % self.condense_factor == 0, 'condensation factor is not divisible by input channels'
        assert self.out_channels % self.groups == 0, 'group value is not divisible by output channels'
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False)
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        self.check_if_drop()
        weight = self.conv.weight * self.mask
        out_conv = F.conv2d(input=out, weight=weight, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=1)
        return out_conv
    """
    Paper: Sec 3.1: Condensation procedure: number of epochs for each condensing stage: M/2(C-1)
    Paper: Sec 3.1: Condensation factor: allow each group to select R/C of inputs.
    - During training a fraction of (C−1)/C connections are removed after each of the C-1 condensing stages
    - we remove columns in Fg (by zeroing them out) if their L1-norm is small compared to the L1-norm of other columns.
    """

    def check_if_drop(self):
        current_progress = LearnedGroupConv.global_progress
        delta = 0
        for i in range(self.condense_factor - 1):
            if current_progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        if not self.reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
            None
        if delta > 0:
            self.drop(delta)
        return

    def drop(self, delta):
        weight = self.conv.weight * self.mask
        None
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        None
        weight = weight.view(d_out, self.groups, self.in_channels)
        None
        weight = weight.transpose(0, 1).contiguous()
        None
        weight = weight.view(self.out_channels, self.in_channels)
        None
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    def reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)


class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, config):
        super().__init__()
        self.config = config
        self.conv_bottleneck = self.config.conv_bottleneck
        self.group1x1 = self.config.group1x1
        self.group3x3 = self.config.group3x3
        self.condense_factor = self.config.condense_factor
        self.dropout_rate = self.config.dropout_rate
        self.conv_1 = LearnedGroupConv(in_channels=in_channels, out_channels=self.conv_bottleneck * growth_rate, kernel_size=1, groups=self.group1x1, condense_factor=self.condense_factor, dropout_rate=self.dropout_rate)
        self.batch_norm = nn.BatchNorm2d(self.conv_bottleneck * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=self.conv_bottleneck * growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, stride=1, groups=self.group3x3, bias=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv_2(out)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Sequential):

    def __init__(self, num_layers, in_channels, growth_rate, config):
        super().__init__()
        for layer_id in range(num_layers):
            layer = DenseLayer(in_channels=in_channels + layer_id * growth_rate, growth_rate=growth_rate, config=config)
            self.add_module('dense_layer_%d' % (layer_id + 1), layer)


def init_model_weights(m):
    for m in m.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


class CondenseNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = self.config.stages
        self.growth_rate = self.config.growth_rate
        assert len(self.stages) == len(self.growth_rate)
        self.init_stride = self.config.init_stride
        self.pool_size = self.config.pool_size
        self.num_classes = self.config.num_classes
        self.progress = 0.0
        self.num_filters = 2 * self.growth_rate[0]
        """
        Initializing layers
        """
        self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool = nn.AvgPool2d(self.pool_size)
        self.relu = nn.ReLU(inplace=True)
        self.init_conv = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.num_filters, kernel_size=3, stride=self.init_stride, padding=1, bias=False)
        self.denseblock_one = DenseBlock(num_layers=self.stages[0], in_channels=self.num_filters, growth_rate=self.growth_rate[0], config=self.config)
        self.num_filters += self.stages[0] * self.growth_rate[0]
        self.denseblock_two = DenseBlock(num_layers=self.stages[1], in_channels=self.num_filters, growth_rate=self.growth_rate[1], config=self.config)
        self.num_filters += self.stages[1] * self.growth_rate[1]
        self.denseblock_three = DenseBlock(num_layers=self.stages[2], in_channels=self.num_filters, growth_rate=self.growth_rate[2], config=self.config)
        self.num_filters += self.stages[2] * self.growth_rate[2]
        self.batch_norm = nn.BatchNorm2d(self.num_filters)
        self.classifier = nn.Linear(self.num_filters, self.num_classes)
        self.apply(init_model_weights)

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        x = self.init_conv(x)
        x = self.denseblock_one(x)
        x = self.transition_pool(x)
        x = self.denseblock_two(x)
        x = self.transition_pool(x)
        x = self.denseblock_three(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class non_bottleneck_1d(nn.Module):

    def __init__(self, n_channel, drop_rate, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.conv3x1_2 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn1 = nn.BatchNorm2d(n_channel, eps=0.001)
        self.bn2 = nn.BatchNorm2d(n_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return self.relu(output + input)


class DownsamplerBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return self.relu(output)


class UpsamplerBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.relu(output)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.LeakyReLU(self.config.relu_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.num_filt_d, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.config.num_filt_d, out_channels=self.config.num_filt_d * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.config.num_filt_d * 2)
        self.conv3 = nn.Conv2d(in_channels=self.config.num_filt_d * 2, out_channels=self.config.num_filt_d * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.config.num_filt_d * 4)
        self.conv4 = nn.Conv2d(in_channels=self.config.num_filt_d * 4, out_channels=self.config.num_filt_d * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.config.num_filt_d * 8)
        self.conv5 = nn.Conv2d(in_channels=self.config.num_filt_d * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.out = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.out(out)
        return out.view(-1, 1).squeeze(1)


class Generator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.config.g_input_size, out_channels=self.config.num_filt_g * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.config.num_filt_g * 8)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 8, out_channels=self.config.num_filt_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.config.num_filt_g * 4)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 4, out_channels=self.config.num_filt_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.config.num_filt_g * 2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g * 2, out_channels=self.config.num_filt_g, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(self.config.num_filt_g)
        self.deconv5 = nn.ConvTranspose2d(in_channels=self.config.num_filt_g, out_channels=self.config.input_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.out = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        out = self.deconv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.deconv4(out)
        out = self.batch_norm4(out)
        out = self.relu(out)
        out = self.deconv5(out)
        out = self.out(out)
        return out


class DQN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.conv_filters[0], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[0])
        self.conv2 = nn.Conv2d(in_channels=self.config.conv_filters[0], out_channels=self.config.conv_filters[1], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(self.config.conv_filters[1])
        self.conv3 = nn.Conv2d(in_channels=self.config.conv_filters[1], out_channels=self.config.conv_filters[2], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(self.config.conv_filters[2])
        self.linear = nn.Linear(448, self.config.num_classes)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        out = self.linear(x.view(x.size(0), -1))
        return out


class ERF(nn.Module):

    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.num_classes = self.config.num_classes
        if encoder == None:
            self.encoder_flag = True
            self.encoder_layers = nn.ModuleList()
            self.initial_block = DownsamplerBlock(self.config.input_channels, 16)
            self.encoder_layers.append(DownsamplerBlock(in_channel=16, out_channel=64))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(DownsamplerBlock(in_channel=64, out_channel=128))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))
        else:
            self.encoder_flag = False
            self.encoder = encoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(UpsamplerBlock(in_channel=128, out_channel=64))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))
        self.decoder_layers.append(UpsamplerBlock(in_channel=64, out_channel=16))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))
        self.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=self.num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        if self.encoder_flag:
            output = self.initial_block(x)
            for layer in self.encoder_layers:
                output = layer(output)
        else:
            output = self.encoder(x)
        for layer in self.decoder_layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, input):
        output = input.view(input.size(0), 128)
        output = self.linear(output)
        return output


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        return output


class Features(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.extralayer1 = nn.MaxPool2d(2, stride=2)
        self.extralayer2 = nn.AvgPool2d(14, 1, 0)

    def forward(self, input):
        output = self.encoder(input)
        output = self.extralayer1(output)
        output = self.extralayer2(output)
        return output


class ERFNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.features = Features()
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        return output


class Example(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        out = x.view(x.size(0), -1)
        return out


class Mnist(nn.Module):

    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BinaryCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 128])], {}),
     True),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLoss2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseBlock,
     lambda: ([], {'num_layers': 1, 'in_channels': 4, 'growth_rate': 4, 'config': _mock_config(conv_bottleneck=4, group1x1=4, group3x3=4, condense_factor=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseLayer,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4, 'config': _mock_config(conv_bottleneck=4, group1x1=4, group3x3=4, condense_factor=4, dropout_rate=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Discriminator,
     lambda: ([], {'config': _mock_config(relu_slope=4, input_channels=4, num_filt_d=4)}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (ERF,
     lambda: ([], {'config': _mock_config(num_classes=4, input_channels=4)}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Example,
     lambda: ([], {'config': _mock_config(input_channels=4, num_filters=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Features,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (Generator,
     lambda: ([], {'config': _mock_config(g_input_size=4, num_filt_g=4, input_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HuberLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsamplerBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (non_bottleneck_1d,
     lambda: ([], {'n_channel': 4, 'drop_rate': 0.5, 'dilated': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_moemen95_Pytorch_Project_Template(_paritybench_base):
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

