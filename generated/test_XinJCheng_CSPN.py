import sys
_module = sys.modules[__name__]
del sys
demo = _module
data_transform = _module
eval = _module
eval_kitti_dataset_loader = _module
eval_nyu_dataset_loader = _module
kitti_dataset_loader = _module
loss = _module
lr_scheduler = _module
models = _module
cspn = _module
torch_resnet_cspn_nyu = _module
update_model = _module
nyu_dataset_loader = _module
train = _module
utils = _module

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


import pandas as pd


import numpy as np


import matplotlib.pyplot as plt


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import math


import random


import numbers


import types


import scipy.ndimage as ndimage


import torch.nn as nn


import torch.optim as optim


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import warnings


from torch.optim.optimizer import Optimizer


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import string


import torchvision


import torchvision.transforms as transforms


class Wighted_L1_Loss(torch.nn.Module):

    def __init__(self):
        super(Wighted_L1_Loss, self).__init__()

    def forward(self, pred, label):
        label_mask = label > 0.0001
        _pred = pred[label_mask]
        _label = label[label_mask]
        n_valid_element = _label.size(0)
        diff_mat = torch.abs(_pred - _label)
        loss = torch.sum(diff_mat) / n_valid_element
        return loss


class Affinity_Propagate(nn.Module):

    def __init__(self, prop_time, prop_kernel, norm_type='8sum'):
        """

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'
        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']
        self.in_feature = 1
        self.out_feature = 1

    def forward(self, guidance, blur_depth, sparse_depth=None):
        self.sum_conv = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        weight = torch.ones(1, 8, 1, 1, 1)
        self.sum_conv.weight = nn.Parameter(weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False
        gate_wb, gate_sum = self.affinity_normalization(guidance)
        raw_depth_input = blur_depth
        result_depth = blur_depth
        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
        for i in range(self.prop_time):
            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum
            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)
            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input
        return result_depth

    def affinity_normalization(self, guidance):
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)
        gate1_wb_cmb = guidance.narrow(1, 0, self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)
        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb, gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)
        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)
        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]
        return gate_wb, gate_sum

    def pad_blur_depth(self, blur_depth):
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)
        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4, blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth

    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1, 0, 1)
        gate1_x1_g2 = guidance.narrow(1, 1, 1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb

    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4)


class Unpool(nn.Module):

    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, (0), (0)] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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


class UpProj_Block(nn.Module):

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        oheight = 0
        owidth = 0
        if self.oheight == 0 and self.owidth == 0:
            oheight = scale * x.size(2)
            owidth = scale * x.size(3)
            x = self._up_pool(x)
        else:
            oheight = self.oheight
            owidth = self.owidth
            x = self._up_pool(x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Simple_Gudi_UpConv_Block(nn.Module):

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        return out


class Simple_Gudi_UpConv_Block_Last_Layer(nn.Module):

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out


class Gudi_UpProj_Block(nn.Module):

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth

    def _up_pooling(self, x, scale):
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x[:, :, 0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:, :, (h), (w)] = 1
        x = torch.mul(mask, x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(nn.Module):

    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        x = self._up_pool(x)
        if self.oheight != 0 and self.owidth != 0:
            x = x.narrow(2, 0, self.oheight)
            x = x.narrow(3, 0, self.owidth)
        return x

    def forward(self, x, side_input):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, up_proj_block, cspn_config=None):
        self.inplanes = 64
        cspn_config_default = {'step': 24, 'kernel': 3, 'norm_type': '8sum'}
        if not cspn_config is None:
            cspn_config_default.update(cspn_config)
        None
        super(ResNet, self).__init__()
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mid_channel = 256 * block.expansion
        self.conv2 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.up_proj_layer1 = self._make_up_conv_layer(up_proj_block, self.mid_channel, int(self.mid_channel / 2))
        self.up_proj_layer2 = self._make_up_conv_layer(up_proj_block, int(self.mid_channel / 2), int(self.mid_channel / 4))
        self.up_proj_layer3 = self._make_up_conv_layer(up_proj_block, int(self.mid_channel / 4), int(self.mid_channel / 8))
        self.up_proj_layer4 = self._make_up_conv_layer(up_proj_block, int(self.mid_channel / 8), int(self.mid_channel / 16))
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.post_process_layer = self._make_post_process_layer(cspn_config_default)
        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 15, 19)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 29, 38)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 57, 76)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 114, 152)
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 228, 304)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 8, 228, 304)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)

    def _make_post_process_layer(self, cspn_config=None):
        return post_process.Affinity_Propagate(cspn_config['step'], cspn_config['kernel'], norm_type=cspn_config['norm_type'])

    def forward(self, x):
        [batch_size, channel, height, width] = x.size()
        sparse_depth = x.narrow(1, 3, 1).clone()
        x = self.conv1_1(x)
        skip4 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x
        x = self.layer2(x)
        skip2 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(self.conv2(x))
        x = self.gud_up_proj_layer1(x)
        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)
        guidance = self.gud_up_proj_layer6(x)
        x = self.gud_up_proj_layer5(x)
        x = self.post_process_layer(guidance, x, sparse_depth)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Gudi_UpProj_Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Gudi_UpProj_Block_Cat,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 8, 8])], {}),
     False),
    (Simple_Gudi_UpConv_Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Simple_Gudi_UpConv_Block_Last_Layer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Unpool,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpProj_Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Wighted_L1_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_XinJCheng_CSPN(_paritybench_base):
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

