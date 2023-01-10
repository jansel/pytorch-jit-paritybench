import sys
_module = sys.modules[__name__]
del sys
data_function = _module
hparam = _module
loss_function = _module
main = _module
metrics = _module
densenet3d = _module
densevoxelnet3d = _module
fcn3d = _module
highresnet = _module
residual_unet3d = _module
unet3d = _module
unetr = _module
vnet3d = _module
deeplab = _module
fcn = _module
highresnet = _module
miniseg = _module
pspnet = _module
segnet = _module
unet = _module
unetpp = _module
convolution = _module
dilation = _module
metric = _module
residual = _module

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


from torch import nn


import torch.nn.functional as F


import random


from torch.autograd import Variable


import time


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms


import torch.distributed as dist


import math


from torchvision import utils


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import CosineAnnealingLR


import copy


import torch.nn as nn


from collections import OrderedDict


import torch.optim as optim


from torchvision import models


from torchvision.models.vgg import VGG


from torch.nn import functional as F


import torchvision


class Binary_Loss(nn.Module):

    def __init__(self):
        super(Binary_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        loss = self.criterion(model_output, targets)
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \\sum{x^p} + \\sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-05
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.2):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
    to keep the spatial dims o=i, this formula is applied
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        norm = nn.BatchNorm3d(num_input_features)
        relu = nn.ReLU(inplace=True)
        conv3d = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, padding=0, stride=1)
        self.conv = nn.Sequential(norm, relu, conv3d)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        k = self.conv(x)
        y = self.max_pool(k)
        return y, k


class SkipDenseNet3D(nn.Module):
    """Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Based on the implementation of https://github.com/tbuikr/3D-SkipDenseSeg
    Paper here : https://arxiv.org/pdf/1709.03199.pdf
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        classes (int) - number of classification classes
    """

    def __init__(self, in_channels=1, classes=1, growth_rate=16, block_config=(4, 4, 4, 4), num_init_features=32, drop_rate=0.1, bn_size=4):
        super(SkipDenseNet3D, self).__init__()
        self.num_classes = classes
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ('norm0', nn.BatchNorm3d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ('norm1', nn.BatchNorm3d(num_init_features)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        self.features_bn = nn.Sequential(OrderedDict([('norm2', nn.BatchNorm3d(num_init_features)), ('relu2', nn.ReLU(inplace=True))]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0, bias=False)
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            up_block = nn.ConvTranspose3d(num_features, classes, kernel_size=2 ** (i + 1) + 2, stride=2 ** (i + 1), padding=1, groups=classes, bias=False)
            self.upsampling_blocks.append(up_block)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                num_features = num_features // 2
        self.bn_class = nn.BatchNorm3d(classes * 4 + num_init_features)
        self.conv_class = nn.Conv3d(classes * 4 + num_init_features, classes, kernel_size=1, padding=0)
        self.relu_last = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features_bn)
        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)
        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)
        out = self.dense_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)
        out = self.dense_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)
        out = torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)
        out = self.conv_class(self.relu_last(self.bn_class(out)))
        return out


class _Upsampling(nn.Sequential):
    """
    For transpose conv
    o = output, p = padding, k = kernel_size, s = stride, d = dilation
    o = (i -1)*s - 2*p + k + output_padding = (i-1)*2 +2 = 2*i
    """

    def __init__(self, input_features, out_features):
        super(_Upsampling, self).__init__()
        self.tr_conv1_features = 128
        self.tr_conv2_features = out_features
        self.add_module('norm', nn.BatchNorm3d(input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(input_features, input_features, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('transp_conv_1', nn.ConvTranspose3d(input_features, self.tr_conv1_features, kernel_size=2, padding=0, output_padding=0, stride=2))
        self.add_module('transp_conv_2', nn.ConvTranspose3d(self.tr_conv1_features, self.tr_conv2_features, kernel_size=2, padding=0, output_padding=0, stride=2))


class DenseVoxelNet(nn.Module):
    """
    Implementation based on https://arxiv.org/abs/1708.00573
    Trainable params: 1,783,408 (roughly 1.8 mentioned in the paper)
    """

    def __init__(self, in_channels=1, classes=1):
        super(DenseVoxelNet, self).__init__()
        num_input_features = 16
        self.dense_1_out_features = 160
        self.dense_2_out_features = 304
        self.up_out_features = 64
        self.classes = classes
        self.in_channels = in_channels
        self.conv_init = nn.Conv3d(in_channels, num_input_features, kernel_size=1, stride=2, padding=0, bias=False)
        self.dense_1 = _DenseBlock(num_layers=12, num_input_features=num_input_features, bn_size=1, growth_rate=12)
        self.trans = _Transition(self.dense_1_out_features, self.dense_1_out_features)
        self.dense_2 = _DenseBlock(num_layers=12, num_input_features=self.dense_1_out_features, bn_size=1, growth_rate=12)
        self.up_block = _Upsampling(self.dense_2_out_features, self.up_out_features)
        self.conv_final = nn.Conv3d(self.up_out_features, classes, kernel_size=1, padding=0, bias=False)
        self.transpose = nn.ConvTranspose3d(self.dense_1_out_features, self.up_out_features, kernel_size=2, padding=0, output_padding=0, stride=2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.dense_1(x)
        x, t = self.trans(x)
        x = self.dense_2(x)
        x = self.up_block(x)
        y1 = self.conv_final(x)
        t = self.transpose(t)
        y2 = self.conv_final(t)
        return y2


class FCN_Net(nn.Module):

    def __init__(self, in_channels=1, n_class=1):
        super().__init__()
        self.conv1_1 = nn.Conv3d(in_channels, 8, 3, padding=60)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=15)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.conv5_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.fc6 = nn.Conv3d(64, 512, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout3d()
        self.fc7 = nn.Conv3d(512, 512, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout3d()
        self.score_fr = nn.Conv3d(512, n_class, 1)
        self.score_pool3 = nn.Conv3d(32, n_class, 1)
        self.score_pool4 = nn.Conv3d(64, n_class, 1)
        self.upscore2 = nn.ConvTranspose3d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose3d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose3d(n_class, n_class, 4, stride=2, bias=False)
        self._initialize_weights()

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :, :] = filt
        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.zero_()
                m.weight.data.normal_(0.0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose3d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h
        h = self.score_pool4(pool4 * 0.01)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3], 5:5 + upscore2.size()[4]]
        score_pool4c = h
        h = upscore2 + score_pool4c
        h = self.upscore_pool4(h)
        upscore_pool4 = h
        h = self.score_pool3(pool3 * 0.0001)
        h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3], 9:9 + upscore_pool4.size()[4]]
        score_pool3c = h
        h = upscore_pool4 + score_pool3c
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3], 31:31 + x.size()[4]].contiguous()
        return h


PADDING_MODES = {'reflect': 'Reflection', 'replicate': 'Replication', 'constant': 'Zero'}


class Pad3d(nn.Module):

    def __init__(self, pad, mode):
        assert mode in PADDING_MODES.keys()
        super().__init__()
        self.pad = 6 * [pad]
        self.mode = mode

    def forward(self, x):
        return F.pad(x, self.pad, self.mode)


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, dimensions, batch_norm=True, instance_norm=False, norm_affine=True, padding_mode='constant', preactivation=True, kernel_size=3, activation=True):
        assert padding_mode in PADDING_MODES.keys()
        assert not (batch_norm and instance_norm)
        super().__init__()
        if dimensions == 2:
            class_name = '{}Pad2d'.format(PADDING_MODES[padding_mode])
            padding_class = getattr(nn, class_name)
            padding_instance = padding_class(dilation)
        elif dimensions == 3:
            padding_instance = Pad3d(dilation, padding_mode)
        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
        if batch_norm:
            norm_class = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d
        if instance_norm:
            norm_class = nn.InstanceNorm2d if dimensions == 2 else nn.InstanceNorm3d
        layers = nn.ModuleList()
        if preactivation:
            if batch_norm or instance_norm:
                layers.append(norm_class(in_channels, affine=norm_affine))
            if activation:
                layers.append(nn.ReLU())
        if kernel_size > 1:
            layers.append(padding_instance)
        use_bias = not (instance_norm or batch_norm)
        conv_layer = conv_class(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=use_bias)
        layers.append(conv_layer)
        if not preactivation:
            if batch_norm or instance_norm:
                layers.append(norm_class(out_channels, affine=norm_affine))
            if activation:
                layers.append(nn.ReLU())
        self.convolutional_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolutional_block(x)


BATCH_DIM = 0


CHANNELS_DIM = 1


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, dilation, dimensions, batch_norm=True, instance_norm=False, residual=True, residual_type='pad', padding_mode='constant'):
        assert residual_type in ('pad', 'project')
        super().__init__()
        self.residual = residual
        self.change_dimension = in_channels != out_channels
        self.residual_type = residual_type
        self.dimensions = dimensions
        if self.change_dimension:
            if residual_type == 'project':
                conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d
                self.change_dim_layer = conv_class(in_channels, out_channels, kernel_size=1, dilation=dilation, bias=False)
        conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv_block = ConvolutionalBlock(in_channels, out_channels, dilation, dimensions, batch_norm=batch_norm, instance_norm=instance_norm, padding_mode=padding_mode)
            conv_blocks.append(conv_block)
            in_channels = out_channels
        self.residual_block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """
        From the original ResNet paper, page 4:
        "When the dimensions increase, we consider two options:
        (A) The shortcut still performs identity mapping,
        with extra zero entries padded for increasing dimensions.
        This option introduces no extra parameter
        (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1x1 convolutions).
        For both options, when the shortcuts go across feature maps of
        two sizes, they are performed with a stride of 2."
        """
        out = self.residual_block(x)
        if self.residual:
            if self.change_dimension:
                if self.residual_type == 'project':
                    x = self.change_dim_layer(x)
                elif self.residual_type == 'pad':
                    batch_size = x.shape[BATCH_DIM]
                    x_channels = x.shape[CHANNELS_DIM]
                    out_channels = out.shape[CHANNELS_DIM]
                    spatial_dims = x.shape[2:]
                    diff_channels = out_channels - x_channels
                    zeros_half = x.new_zeros(batch_size, diff_channels // 2, *spatial_dims)
                    x = torch.cat((zeros_half, x, zeros_half), dim=CHANNELS_DIM)
            out = x + out
        return out


class DilationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, dimensions, layers_per_block=2, num_residual_blocks=3, batch_norm=True, instance_norm=False, residual=True, padding_mode='constant'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            residual_block = ResidualBlock(in_channels, out_channels, layers_per_block, dilation, dimensions, batch_norm=batch_norm, instance_norm=instance_norm, residual=residual, padding_mode=padding_mode)
            residual_blocks.append(residual_block)
            in_channels = out_channels
        self.dilation_block = nn.Sequential(*residual_blocks)

    def forward(self, x):
        return self.dilation_block(x)


class HighResNet(nn.Module):

    def __init__(self, in_channels, out_channels, dimensions=None, initial_out_channels_power=4, layers_per_residual_block=2, residual_blocks_per_dilation=3, dilations=3, batch_norm=True, instance_norm=False, residual=True, padding_mode='constant', add_dropout_layer=False):
        assert dimensions in (2, 3)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation
        self.dilations = dilations
        blocks = nn.ModuleList()
        initial_out_channels = 2 ** initial_out_channels_power
        first_conv_block = ConvolutionalBlock(in_channels=self.in_channels, out_channels=initial_out_channels, dilation=1, dimensions=dimensions, batch_norm=batch_norm, instance_norm=instance_norm, preactivation=False, padding_mode=padding_mode)
        blocks.append(first_conv_block)
        in_channels = out_channels = initial_out_channels
        dilation_block = None
        for dilation_idx in range(dilations):
            if dilation_idx >= 1:
                in_channels = dilation_block.out_channels
            dilation = 2 ** dilation_idx
            dilation_block = DilationBlock(in_channels, out_channels, dilation, dimensions, layers_per_block=layers_per_residual_block, num_residual_blocks=residual_blocks_per_dilation, batch_norm=batch_norm, instance_norm=instance_norm, residual=residual, padding_mode=padding_mode)
            blocks.append(dilation_block)
            out_channels *= 2
        out_channels = out_channels // 2
        if add_dropout_layer:
            in_channels = out_channels
            out_channels = 80
            dropout_conv_block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, dilation=1, dimensions=dimensions, batch_norm=batch_norm, instance_norm=instance_norm, preactivation=False, kernel_size=1)
            blocks.append(dropout_conv_block)
            blocks.append(nn.Dropout3d())
        classifier = ConvolutionalBlock(in_channels=out_channels, out_channels=self.out_channels, dilation=1, dimensions=dimensions, batch_norm=batch_norm, instance_norm=instance_norm, preactivation=False, kernel_size=1, activation=False, padding_mode=padding_mode)
        blocks.append(classifier)
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

    @property
    def num_parameters(self):
        return sum(torch.prod(torch.tensor(p.shape)) for p in self.parameters())

    @property
    def receptive_field(self):
        """
        B: number of convolutional layers per residual block
        N: number of residual blocks per dilation factor
        D: number of different dilation factors
        """
        B = self.layers_per_residual_block
        D = self.dilations
        N = self.residual_blocks_per_dilation
        d = torch.arange(D)
        input_output_diff = 3 - 1 + torch.sum(B * N * 2 ** (d + 1))
        receptive_field = input_output_diff + 1
        return receptive_field

    def get_receptive_field_world(self, spacing=1):
        return self.receptive_field * spacing


class HighRes3DNet(HighResNet):

    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 3
        super().__init__(*args, **kwargs)


class UNet(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 8)
        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 4)
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 2)
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter)
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def forward(self, x):
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        return seg_layer


class UNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name='enc1')
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name='enc2')
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name='enc3')
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name='enc4')
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = UNet3D._block(features * 8, features * 16, name='bottleneck')
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 8 * 2, features * 8, name='dec4')
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 4 * 2, features * 4, name='dec3')
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 2 * 2, features * 2, name='dec2')
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name='dec1')
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([(name + 'conv1', nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=True)), (name + 'norm1', nn.BatchNorm3d(num_features=features)), (name + 'relu1', nn.ReLU(inplace=True)), (name + 'conv2', nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)), (name + 'norm2', nn.BatchNorm3d(num_features=features)), (name + 'relu2', nn.ReLU(inplace=True))]))


class SingleDeconv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(SingleConv3DBlock(in_planes, out_planes, kernel_size), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(SingleDeconv3DBlock(in_planes, out_planes), SingleConv3DBlock(out_planes, out_planes, kernel_size), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):

    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):

    def __init__(self, in_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):

    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int(cube_size[0] * cube_size[1] * cube_size[2] / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.mlp_dim = int(cube_size[0] * cube_size[1] * cube_size[2] / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x, weights


class Transformer(nn.Module):

    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)
        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)
        return extract_layers


class UNETR(nn.Module):

    def __init__(self, img_shape=(128, 128, 128), input_dim=4, output_dim=3, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.transformer = Transformer(input_dim, embed_dim, img_shape, patch_size, num_heads, self.num_layers, dropout, self.ext_layers)
        self.decoder0 = nn.Sequential(Conv3DBlock(input_dim, 32, 3), Conv3DBlock(32, 64, 3))
        self.decoder3 = nn.Sequential(Deconv3DBlock(embed_dim, 512), Deconv3DBlock(512, 256), Deconv3DBlock(256, 128))
        self.decoder6 = nn.Sequential(Deconv3DBlock(embed_dim, 512), Deconv3DBlock(512, 256))
        self.decoder9 = Deconv3DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, 512)
        self.decoder9_upsampler = nn.Sequential(Conv3DBlock(1024, 512), Conv3DBlock(512, 512), Conv3DBlock(512, 512), SingleDeconv3DBlock(512, 256))
        self.decoder6_upsampler = nn.Sequential(Conv3DBlock(512, 256), Conv3DBlock(256, 256), SingleDeconv3DBlock(256, 128))
        self.decoder3_upsampler = nn.Sequential(Conv3DBlock(256, 128), Conv3DBlock(128, 128), SingleDeconv3DBlock(128, 64))
        self.decoder0_header = nn.Sequential(Conv3DBlock(128, 64), Conv3DBlock(64, 64), SingleConv3DBlock(64, output_dim, 1))

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class InputTransition(nn.Module):

    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(self.num_features)
        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


def passthrough(x, **kwargs):
    return x


class DownTransition(nn.Module):

    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)
        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=1):
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, in_class, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_class, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        return x, x_3


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates=(12, 24, 36), hidden_channels=256, norm_act=nn.BatchNorm2d, pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size
        self.map_convs = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, 1, bias=False), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0], padding=dilation_rates[0]), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1], padding=dilation_rates[1]), nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2], padding=dilation_rates[2])])
        self.map_bn = norm_act(hidden_channels * 4)
        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)
        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)
        out = self.red_conv(out)
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.leak_relu(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))
        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = min(try_index(self.pooling_size, 0), x.shape[2]), min(try_index(self.pooling_size, 1), x.shape[3])
            padding = (pooling_size[1] - 1) // 2, (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1, (pooling_size[0] - 1) // 2, (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode='replicate')
        return pool


def get_resnet101(in_class, dilation=[1, 1, 1, 1], bn_momentum=0.0003, is_fpn=False):
    model = ResNet(in_class, Bottleneck, [3, 4, 23, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    return model


class DeepLabV3(nn.Module):

    def __init__(self, in_class, class_num, bn_momentum=0.01):
        super(DeepLabV3, self).__init__()
        self.Resnet101 = get_resnet101(in_class, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.ASPP = ASPP(2048, 256, [6, 12, 18], norm_act=nn.BatchNorm2d)
        self.classify = nn.Conv2d(256, class_num, 1, bias=True)

    def forward(self, input):
        x = self.Resnet101(input)
        aspp = self.ASPP(x)
        predict = self.classify(aspp)
        output = F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):

    def __init__(self, in_class=1, n_class=1):
        super(FCN32s, self).__init__()
        self.conv1_1 = nn.Conv2d(in_class, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        h = self.score_fr(h)
        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        return h


class HighRes2DNet(HighResNet):

    def __init__(self, *args, **kwargs):
        kwargs['dimensions'] = 2
        super().__init__(*args, **kwargs)


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.act(self.bn(self.conv(input)))
        return output


class DilatedParallelConvBlockD2(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output


class DilatedParallelConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(DilatedParallelConvBlock, self).__init__()
        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        p = self.pool(output)
        d1 = d1 + p
        d2 = d1 + d2
        d3 = d2 + d3
        d4 = d3 + d4
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))
        return output


class DownsamplerBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=2):
        super(DownsamplerBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 5, stride=stride, padding=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1(self.conv0(input))
        output = self.act(self.bn(output))
        return output


def split(x):
    c = int(x.size()[1])
    c1 = round(c // 2)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class MiniSeg(nn.Module):

    def __init__(self, in_input=3, classes=2, P1=2, P2=3, P3=8, P4=6, aux=False):
        super(MiniSeg, self).__init__()
        self.D1 = int(P1 / 2)
        self.D2 = int(P2 / 2)
        self.D3 = int(P3 / 2)
        self.D4 = int(P4 / 2)
        self.aux = aux
        self.long1 = DownsamplerBlock(in_input, 8, stride=2)
        self.down1 = ConvBlock(in_input, 8, stride=2)
        self.level1 = nn.ModuleList()
        self.level1_long = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(8, 8))
        for i in range(0, self.D1):
            self.level1_long.append(DownsamplerBlock(8, 8, stride=1))
        self.cat1 = nn.Sequential(nn.Conv2d(16, 16, 1, stride=1, padding=0, groups=1, bias=False), nn.BatchNorm2d(16))
        self.long2 = DownsamplerBlock(8, 24, stride=2)
        self.down2 = DilatedParallelConvBlock(8, 24, stride=2)
        self.level2 = nn.ModuleList()
        self.level2_long = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(DilatedParallelConvBlock(24, 24))
        for i in range(0, self.D2):
            self.level2_long.append(DownsamplerBlock(24, 24, stride=1))
        self.cat2 = nn.Sequential(nn.Conv2d(48, 48, 1, stride=1, padding=0, groups=1, bias=False), nn.BatchNorm2d(48))
        self.long3 = DownsamplerBlock(24, 32, stride=2)
        self.down3 = DilatedParallelConvBlock(24, 32, stride=2)
        self.level3 = nn.ModuleList()
        self.level3_long = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(DilatedParallelConvBlock(32, 32))
        for i in range(0, self.D3):
            self.level3_long.append(DownsamplerBlock(32, 32, stride=1))
        self.cat3 = nn.Sequential(nn.Conv2d(64, 64, 1, stride=1, padding=0, groups=1, bias=False), nn.BatchNorm2d(64))
        self.long4 = DownsamplerBlock(32, 64, stride=2)
        self.down4 = DilatedParallelConvBlock(32, 64, stride=2)
        self.level4 = nn.ModuleList()
        self.level4_long = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(DilatedParallelConvBlock(64, 64))
        for i in range(0, self.D4):
            self.level4_long.append(DownsamplerBlock(64, 64, stride=1))
        self.up4_conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(64)
        self.up4_act = nn.PReLU(64)
        self.up3_conv4 = DilatedParallelConvBlockD2(64, 32)
        self.up3_conv3 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(32)
        self.up3_act = nn.PReLU(32)
        self.up2_conv3 = DilatedParallelConvBlockD2(32, 24)
        self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(24)
        self.up2_act = nn.PReLU(24)
        self.up1_conv2 = DilatedParallelConvBlockD2(24, 8)
        self.up1_conv1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(8)
        self.up1_act = nn.PReLU(8)
        if self.aux:
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(32, classes, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(8, classes, 1, stride=1, padding=0))

    def forward(self, input):
        long1 = self.long1(input)
        output1 = self.down1(input)
        output1_add = output1 + long1
        for i, layer in enumerate(self.level1):
            if i < self.D1:
                output1 = layer(output1_add) + output1
                long1 = self.level1_long[i](output1_add) + long1
                output1_add = output1 + long1
            else:
                output1 = layer(output1_add) + output1
                output1_add = output1 + long1
        output1_cat = self.cat1(torch.cat([long1, output1], 1))
        output1_l, output1_r = split(output1_cat)
        long2 = self.long2(output1_l + long1)
        output2 = self.down2(output1_r + output1)
        output2_add = output2 + long2
        for i, layer in enumerate(self.level2):
            if i < self.D2:
                output2 = layer(output2_add) + output2
                long2 = self.level2_long[i](output2_add) + long2
                output2_add = output2 + long2
            else:
                output2 = layer(output2_add) + output2
                output2_add = output2 + long2
        output2_cat = self.cat2(torch.cat([long2, output2], 1))
        output2_l, output2_r = split(output2_cat)
        long3 = self.long3(output2_l + long2)
        output3 = self.down3(output2_r + output2)
        output3_add = output3 + long3
        for i, layer in enumerate(self.level3):
            if i < self.D3:
                output3 = layer(output3_add) + output3
                long3 = self.level3_long[i](output3_add) + long3
                output3_add = output3 + long3
            else:
                output3 = layer(output3_add) + output3
                output3_add = output3 + long3
        output3_cat = self.cat3(torch.cat([long3, output3], 1))
        output3_l, output3_r = split(output3_cat)
        long4 = self.long4(output3_l + long3)
        output4 = self.down4(output3_r + output3)
        output4_add = output4 + long4
        for i, layer in enumerate(self.level4):
            if i < self.D4:
                output4 = layer(output4_add) + output4
                long4 = self.level4_long[i](output4_add) + long4
                output4_add = output4 + long4
            else:
                output4 = layer(output4_add) + output4
                output4_add = output4 + long4
        up4_conv4 = self.up4_bn4(self.up4_conv4(output4))
        up4 = self.up4_act(up4_conv4)
        up4 = F.interpolate(up4, output3.size()[2:], mode='bilinear', align_corners=False)
        up3_conv4 = self.up3_conv4(up4)
        up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        up3 = self.up3_act(up3_conv4 + up3_conv3)
        up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        up2_conv3 = self.up2_conv3(up3)
        up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        up2 = self.up2_act(up2_conv3 + up2_conv2)
        up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        up1_conv2 = self.up1_conv2(up2)
        up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        up1 = self.up1_act(up1_conv2 + up1_conv1)
        if self.aux:
            pred4 = F.interpolate(self.pred4(up4), input.size()[2:], mode='bilinear', align_corners=False)
            pred3 = F.interpolate(self.pred3(up3), input.size()[2:], mode='bilinear', align_corners=False)
            pred2 = F.interpolate(self.pred2(up2), input.size()[2:], mode='bilinear', align_corners=False)
        pred1 = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)
        if self.aux:
            return pred1, pred2, pred3, pred4
        else:
            return pred1


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
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


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.PReLU())

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):

    def __init__(self, in_class=1, n_classes=1, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34', pretrained=True):
        super().__init__()
        self.feats = ResNet(in_class, BasicBlock, [3, 4, 6, 3])
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=1), nn.LogSoftmax())

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)
        p = self.drop_2(p)
        return self.final(p)


class SegNet(nn.Module):

    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()
        batchNorm_momentum = 0.1
        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return x11d


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels=512, out_channels=256, kernel_size=3, is_deconv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        """
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2,**up_kwargs)
        """
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


class ResNet34UnetPlus(nn.Module):

    def __init__(self, num_channels=1, num_class=1, is_deconv=False, decoder_kernel_size=3):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder0_1 = DecoderBlock(in_channels=64 + 64, out_channels=64, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1_1 = DecoderBlock(in_channels=128 + 64, out_channels=64, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder0_2 = DecoderBlock(in_channels=64 + 64 + 64, out_channels=64, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2_1 = DecoderBlock(in_channels=128 + 256, out_channels=128, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1_2 = DecoderBlock(in_channels=64 + 64 + 128, out_channels=128, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder0_3 = DecoderBlock(in_channels=64 + 64 + 64 + 128, out_channels=128, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3_1 = DecoderBlock(in_channels=512 + 256, out_channels=256, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2_2 = DecoderBlock(in_channels=128 + 128 + 256, out_channels=256, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1_3 = DecoderBlock(in_channels=64 + 64 + 128 + 256, out_channels=256, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder0_4 = DecoderBlock(in_channels=64 + 64 + 64 + 128 + 256, out_channels=256, kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.logit1 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128, num_class, kernel_size=1)
        self.logit4 = nn.Conv2d(256, num_class, kernel_size=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_ = self.firstrelu(x)
        x = self.firstmaxpool(x_)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        x0_0 = x_
        x1_0 = e1
        None
        x0_1 = self.decoder0_1([x0_0, upsize(x1_0)])
        x2_0 = e2
        x1_1 = self.decoder1_1([x1_0, upsize(x2_0)])
        None
        x0_2 = self.decoder0_2([x0_0, x0_1, upsize(x1_1)])
        x3_0 = e3
        x2_1 = self.decoder2_1([x2_0, upsize(x3_0)])
        x1_2 = self.decoder1_2([x1_0, x1_1, upsize(x2_1)])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, upsize(x1_2)])
        x4_0 = e4
        x3_1 = self.decoder3_1([x3_0, upsize(x4_0)])
        x2_2 = self.decoder2_2([x2_0, x2_1, upsize(x3_1)])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, upsize(x2_2)])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, upsize(x1_3)])
        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)
        None
        logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3 + self.mix[4] * logit4
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)
        return logit


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinaryDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Binary_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Deconv3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (DenseVoxelNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (DilatedParallelConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilatedParallelConvBlockD2,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownTransition,
     lambda: ([], {'inChans': 4, 'nConvs': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DownsamplerBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Embeddings,
     lambda: ([], {'input_dim': 4, 'embed_dim': 4, 'cube_size': [4, 4, 4], 'patch_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FCN32s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (FCN_Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (HighRes2DNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HighRes3DNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (InConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputTransition,
     lambda: ([], {'in_channels': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LUConv,
     lambda: ([], {'nchan': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (MiniSeg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (OutConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OutputTransition,
     lambda: ([], {'in_channels': 4, 'classes': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (PSPModule,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSPNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (PSPUpsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet34UnetPlus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (SegNet,
     lambda: ([], {'input_nbr': 4, 'label_nbr': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'num_heads': 4, 'embed_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SingleConv3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleDeconv3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerBlock,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4, 'dropout': 0.5, 'cube_size': [4, 4, 4], 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (UNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (Unet,
     lambda: ([], {'in_channels': 4, 'classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (VNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (_Upsampling,
     lambda: ([], {'input_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_MontaEllis_Pytorch_Medical_Segmentation(_paritybench_base):
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

