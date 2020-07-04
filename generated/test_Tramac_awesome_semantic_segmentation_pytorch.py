import sys
_module = sys.modules[__name__]
del sys
core = _module
data = _module
dataloader = _module
ade = _module
cityscapes = _module
lip_parsing = _module
mscoco = _module
pascal_aug = _module
pascal_voc = _module
sbu_shadow = _module
segbase = _module
utils = _module
downloader = _module
ade20k = _module
models = _module
base_models = _module
densenet = _module
eespnet = _module
hrnet = _module
mobilenetv2 = _module
resnet = _module
resnetv1b = _module
resnext = _module
vgg = _module
xception = _module
bisenet = _module
ccnet = _module
cgnet = _module
danet = _module
deeplabv3 = _module
deeplabv3_plus = _module
denseaspp = _module
dfanet = _module
dunet = _module
encnet = _module
enet = _module
espnet = _module
fcn = _module
fcnv2 = _module
hrnet = _module
icnet = _module
lednet = _module
model_store = _module
model_zoo = _module
ocnet = _module
psanet = _module
psanet_old = _module
pspnet = _module
segbase = _module
nn = _module
basic = _module
ca_block = _module
jpu = _module
psa_block = _module
setup = _module
sync_bn = _module
functions = _module
lib = _module
gpu = _module
syncbn = _module
syncbn = _module
distributed = _module
download = _module
filesystem = _module
logger = _module
loss = _module
lr_scheduler = _module
parallel = _module
score = _module
visualize = _module
demo = _module
eval = _module
train = _module
test_model = _module
test_module = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import re


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import math


from torch.autograd.function import once_differentiable


import warnings


from torch.nn.modules.batchnorm import _BatchNorm


import torch.cuda.comm as comm


from torch.autograd import Function


import torch.utils.data as data


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


from torch.autograd import Variable


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel._functions import Broadcast


import torch.backends.cudnn as cudnn


from torchvision import transforms


import time


import numpy as np


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, 1, 1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, 3, 1, dilation, dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate, dilation, norm_layer)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_layer=
        nn.BatchNorm2d):
        super(_Transition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, 1, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, 7, 2, 3, bias=False)), ('norm0', norm_layer(
            num_init_features)), ('relu0', nn.ReLU(True)), ('pool0', nn.
            MaxPool2d(3, 2, 1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size,
                growth_rate, drop_rate, norm_layer=norm_layer)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2,
                    norm_layer=norm_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.num_features = num_features
        self.features.add_module('norm5', norm_layer(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class EESP(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7,
        down_method='esp', norm_layer=nn.BatchNorm2d):
        super(EESP, self).__init__()
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = _ConvBNPReLU(in_channels, n, 1, stride=1, groups=k,
            norm_layer=norm_layer)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            dilation = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(nn.Conv2d(n, n, 3, stride, dilation,
                dilation=dilation, groups=n, bias=False))
        self.conv_1x1_exp = _ConvBN(out_channels, out_channels, 1, 1,
            groups=k, norm_layer=norm_layer)
        self.br_after_cat = _BNPReLU(out_channels, norm_layer)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == x.size():
            expanded = expanded + x
        return self.module_act(expanded)


class DownSampler(nn.Module):

    def __init__(self, in_channels, out_channels, k=4, r_lim=9, reinf=True,
        inp_reinf=3, norm_layer=None):
        super(DownSampler, self).__init__()
        channels_diff = out_channels - in_channels
        self.eesp = EESP(in_channels, channels_diff, stride=2, k=k, r_lim=
            r_lim, down_method='avg', norm_layer=norm_layer)
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(_ConvBNPReLU(inp_reinf,
                inp_reinf, 3, 1, 1), _ConvBN(inp_reinf, out_channels, 1, 1))
        self.act = nn.PReLU(out_channels)

    def forward(self, x, x2=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = torch.cat([avg_out, eesp_out], 1)
        if x2 is not None:
            w1 = avg_out.size(2)
            while True:
                x2 = F.avg_pool2d(x2, kernel_size=3, padding=1, stride=2)
                w2 = x2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(x2)
        return self.act(output)


class EESPNet(nn.Module):

    def __init__(self, num_classes=1000, scale=1, reinf=True, norm_layer=nn
        .BatchNorm2d):
        super(EESPNet, self).__init__()
        inp_reinf = 3 if reinf else None
        reps = [0, 3, 7, 3]
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base, levels, base_s = 32, 5, 0
        out_channels = [base] * levels
        for i in range(levels):
            if i == 0:
                base_s = int(base * scale)
                base_s = math.ceil(base_s / K[0]) * K[0]
                out_channels[i] = base if base_s > base else base_s
            else:
                out_channels[i] = base_s * pow(2, i)
        if scale <= 1.5:
            out_channels.append(1024)
        elif scale in [1.5, 2]:
            out_channels.append(1280)
        else:
            raise ValueError('Unknown scale value.')
        self.level1 = _ConvBNPReLU(3, out_channels[0], 3, 2, 1, norm_layer=
            norm_layer)
        self.level2_0 = DownSampler(out_channels[0], out_channels[1], k=K[0
            ], r_lim=r_lim[0], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level3_0 = DownSampler(out_channels[1], out_channels[2], k=K[1
            ], r_lim=r_lim[1], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(out_channels[2], out_channels[2], k=K[2
                ], r_lim=r_lim[2], norm_layer=norm_layer))
        self.level4_0 = DownSampler(out_channels[2], out_channels[3], k=K[2
            ], r_lim=r_lim[2], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(out_channels[3], out_channels[3], k=K[3
                ], r_lim=r_lim[3], norm_layer=norm_layer))
        self.level5_0 = DownSampler(out_channels[3], out_channels[4], k=K[3
            ], r_lim=r_lim[3], reinf=reinf, inp_reinf=inp_reinf, norm_layer
            =norm_layer)
        self.level5 = nn.ModuleList()
        for i in range(reps[2]):
            self.level5.append(EESP(out_channels[4], out_channels[4], k=K[4
                ], r_lim=r_lim[4], norm_layer=norm_layer))
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[4], 3,
            1, 1, groups=out_channels[4], norm_layer=norm_layer))
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[5], 1,
            1, 0, groups=K[4], norm_layer=norm_layer))
        self.fc = nn.Linear(out_channels[5], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, seg=True):
        out_l1 = self.level1(x)
        out_l2 = self.level2_0(out_l1, x)
        out_l3_0 = self.level3_0(out_l2, x)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, x)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        if not seg:
            out_l5_0 = self.level5_0(out_l4)
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)
            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=0.2, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)
            return self.fc(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias
            =False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True, norm_layer=nn.
        BatchNorm2d):
        super(HighResolutionModule, self).__init__()
        assert num_branches == len(num_blocks)
        assert num_branches == len(num_channels)
        assert num_branches == len(num_inchannels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels, norm_layer=norm_layer)
        self.fuse_layers = self._make_fuse_layers(norm_layer)
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                1, stride, bias=False), norm_layer(num_channels[
                branch_index] * block.expansion))
        layers = list()
        layers.append(block(self.num_inchannels[branch_index], num_channels
            [branch_index], stride, downsample, norm_layer=norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index], norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels,
        norm_layer=nn.BatchNorm2d):
        branches = list()
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels, norm_layer=norm_layer))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, norm_layer=nn.BatchNorm2d):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = list()
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, bias=False
                        ), norm_layer(num_inchannels[i]), nn.Upsample(
                        scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = list()
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), norm_layer(
                                num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), norm_layer(
                                num_outchannels_conv3x3), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = list()
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolutionNet(nn.Module):

    def __init__(self, blocks, num_channels, num_modules, num_branches,
        num_blocks, fuse_method, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.num_branches = num_branches
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            norm_layer(64), nn.ReLU(True), nn.Conv2d(64, 64, 3, 2, 1, bias=
            False), norm_layer(64), nn.ReLU(True))
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_layer=
            norm_layer)
        num_channel, block = num_channels[0], blocks[0]
        channels = [(channel * block.expansion) for channel in num_channel]
        self.transition1 = self._make_transition_layer([256], channels,
            norm_layer)
        self.stage2, pre_stage_channels = self._make_stage(num_modules[0],
            num_branches[0], num_blocks[0], channels, block, fuse_method[0],
            channels, norm_layer=norm_layer)
        num_channel, block = num_channels[1], blocks[1]
        channels = [(channel * block.expansion) for channel in num_channel]
        self.transition1 = self._make_transition_layer(pre_stage_channels,
            channels, norm_layer)
        self.stage3, pre_stage_channels = self._make_stage(num_modules[1],
            num_branches[1], num_blocks[1], channels, block, fuse_method[1],
            channels, norm_layer=norm_layer)
        num_channel, block = num_channels[2], blocks[2]
        channels = [(channel * block.expansion) for channel in num_channel]
        self.transition1 = self._make_transition_layer(pre_stage_channels,
            channels, norm_layer)
        self.stage4, pre_stage_channels = self._make_stage(num_modules[2],
            num_branches[2], num_blocks[2], channels, block, fuse_method[2],
            channels, norm_layer=norm_layer)
        self.incre_modules, self.downsamp_modules, self.final_layer = (self
            ._make_head(pre_stage_channels, norm_layer))
        self.classifier = nn.Linear(2048, 1000)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1,
        norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.
                expansion, 1, stride, bias=False), norm_layer(planes *
                block.expansion))
        layers = list()
        layers.append(block(inplanes, planes, stride, downsample=downsample,
            norm_layer=norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer, norm_layer=nn.BatchNorm2d):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = list()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, padding=1, bias=False), norm_layer(
                        num_channels_cur_layer[i]), nn.ReLU(True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = list()
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i
                        ] if j == i - num_branches_pre else in_channels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(in_channels,
                        out_channels, 3, 2, 1, bias=False), norm_layer(
                        out_channels), nn.ReLU(True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks,
        num_channels, block, fuse_method, num_inchannels,
        multi_scale_output=True, norm_layer=nn.BatchNorm2d):
        modules = list()
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block,
                num_blocks, num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output, norm_layer=norm_layer))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, pre_stage_channels, norm_layer=nn.BatchNorm2d):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = list()
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels,
                head_channels[i], 1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, 3, 2, 1), norm_layer(out_channels), nn.ReLU(True)
                )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(head_channels[3] * head_block
            .expansion, 2048, 1), norm_layer(2048), nn.ReLU(True))
        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x_list = list()
        for i in range(self.num_branches[0]):
            if self.transition1[i] is not None:
                tmp = self.transition1[i](x)
                None
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]
                ) + self.downsamp_modules[i](y)
        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(MobileNet, self).__init__()
        conv_dw_setting = [[64, 1, 1], [128, 2, 2], [256, 2, 2], [512, 6, 2
            ], [1024, 2, 2]]
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, norm_layer=
            norm_layer)]
        for c, n, s in conv_dw_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_DepthwiseConv(input_channels, out_channels,
                    stride, norm_layer))
                input_channels = out_channels
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(int(1024 * multiplier), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(MobileNetV2, self).__init__()
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 
            3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]
            ]
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True,
            norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels,
                    out_channels, stride, t, norm_layer))
                input_channels = out_channels
        features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6
            =True, norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout2d(0.2), nn.Linear(
            last_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.
        BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
            dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetV1b(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        deep_stem=False, zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False
                ), norm_layer(64), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1,
                bias=False), norm_layer(64), nn.ReLU(True), nn.Conv2d(64, 
                128, 3, 1, 1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, 1, stride, bias=False), norm_layer(planes *
                block.expansion))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                previous_dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation,
            groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNext(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, dilated=False, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(ResNext, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, 1, stride, bias=False), norm_layer(planes *
                block.expansion))
        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, dilation=2, norm_layer=
                norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=dilation, norm_layer=
                norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, 0, dilation, groups=in_channels, bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1
            )
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=
        1, norm_layer=None, start_with_relu=True, grow_first=True, is_last=
        False):
        super(Block, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride,
                bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        filters = in_channels
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1,
                dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation,
                norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1,
                dilation, norm_layer=norm_layer))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3,
                stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        elif is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1,
                dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Xception65(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.
        BatchNorm2d):
        super(Xception65, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            norm_layer=norm_layer, start_with_relu=True, grow_first=True,
            is_last=True)
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=
                middle_block_dilation, norm_layer=norm_layer,
                start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride,
            dilation=exit_block_dilations[0], norm_layer=norm_layer,
            start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.midflow(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Xception71(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.
        BatchNorm2d):
        super(Xception71, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False)
        self.block2 = nn.Sequential(Block(128, 256, reps=2, stride=2,
            norm_layer=norm_layer, start_with_relu=False, grow_first=True),
            Block(256, 728, reps=2, stride=2, norm_layer=norm_layer,
            start_with_relu=False, grow_first=True))
        self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride,
            norm_layer=norm_layer, start_with_relu=True, grow_first=True,
            is_last=True)
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=
                middle_block_dilation, norm_layer=norm_layer,
                start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride,
            dilation=exit_block_dilations[0], norm_layer=norm_layer,
            start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.midflow(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BlockA(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
        norm_layer=None, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride,
                bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        inter_channels = out_channels // 4
        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1,
            dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3,
                stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1,
                norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Enc(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, norm_layer=None):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=
            norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=
                norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):

    def __init__(self, in_channels, norm_layer=None):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(nn.Conv2d(1000, in_channels, 1, bias=
            False), norm_layer(in_channels), nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):

    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            norm_layer(8), nn.ReLU(True))
        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)
        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BiSeNet(nn.Module):

    def __init__(self, nclass, backbone='resnet18', aux=False, jpu=False,
        pretrained_base=True, **kwargs):
        super(BiSeNet, self).__init__()
        self.aux = aux
        self.spatial_path = SpatialPath(3, 128, **kwargs)
        self.context_path = ContextPath(backbone, pretrained_base, **kwargs)
        self.ffm = FeatureFusion(256, 256, 4, **kwargs)
        self.head = _BiSeHead(256, 64, nclass, **kwargs)
        if aux:
            self.auxlayer1 = _BiSeHead(128, 256, nclass, **kwargs)
            self.auxlayer2 = _BiSeHead(128, 256, nclass, **kwargs)
        self.__setattr__('exclusive', ['spatial_path', 'context_path',
            'ffm', 'head', 'auxlayer1', 'auxlayer2'] if aux else [
            'spatial_path', 'context_path', 'ffm', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)
        fusion_out = self.ffm(spatial_out, context_out[-1])
        outputs = []
        x = self.head(fusion_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout1 = self.auxlayer1(context_out[0])
            auxout1 = F.interpolate(auxout1, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout1)
            auxout2 = self.auxlayer2(context_out[1])
            auxout2 = F.interpolate(auxout2, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout2)
        return tuple(outputs)


class _BiSeHead(nn.Module):

    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(_ConvBNReLU(in_channels, inter_channels,
            3, 1, 1, norm_layer=norm_layer), nn.Dropout(0.1), nn.Conv2d(
            inter_channels, nclass, 1))

    def forward(self, x):
        x = self.block(x)
        return x


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(SpatialPath, self).__init__()
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3,
            norm_layer=norm_layer)
        self.conv3x3_1 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer)
        self.conv3x3_2 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer)
        self.conv1x1 = _ConvBNReLU(inter_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class _GlobalAvgPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(AttentionRefinmentModule, self).__init__()
        self.conv3x3 = _ConvBNReLU(in_channels, out_channels, 3, 1, 1,
            norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels, 1, 1, 0, norm_layer=
            norm_layer), nn.Sigmoid())

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


model_urls = {'vgg11':
    'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13':
    'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16':
    'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19':
    'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn':
    'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn':
    'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn':
    'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn':
    'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ContextPath(nn.Module):

    def __init__(self, backbone='resnet18', pretrained_base=True,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(ContextPath, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        inter_channels = 128
        self.global_context = _GlobalAvgPooling(512, inter_channels, norm_layer
            )
        self.arms = nn.ModuleList([AttentionRefinmentModule(512,
            inter_channels, norm_layer, **kwargs), AttentionRefinmentModule
            (256, inter_channels, norm_layer, **kwargs)])
        self.refines = nn.ModuleList([_ConvBNReLU(inter_channels,
            inter_channels, 3, 1, 1, norm_layer=norm_layer), _ConvBNReLU(
            inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        context_blocks = []
        context_blocks.append(x)
        x = self.layer2(x)
        context_blocks.append(x)
        c3 = self.layer3(x)
        context_blocks.append(c3)
        c4 = self.layer4(c3)
        context_blocks.append(c4)
        context_blocks.reverse()
        global_context = self.global_context(c4)
        last_feature = global_context
        context_outputs = []
        for i, (feature, arm, refine) in enumerate(zip(context_blocks[:2],
            self.arms, self.refines)):
            feature = arm(feature)
            feature += last_feature
            last_feature = F.interpolate(feature, size=context_blocks[i + 1
                ].size()[2:], mode='bilinear', align_corners=True)
            last_feature = refine(last_feature)
            context_outputs.append(last_feature)
        return context_outputs


class FeatureFusion(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=
        nn.BatchNorm2d, **kwargs):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0,
            norm_layer=norm_layer), _ConvBNReLU(out_channels // reduction,
            out_channels, 1, 1, 0, norm_layer=norm_layer), nn.Sigmoid())

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


class _CCHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer, **kwargs)
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            True))
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels +
            inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels), nn.Dropout2d(0.1))

    def forward(self, x, recurrence=1):
        out = self.conva(x)
        for i in range(recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)
        return out


class CGNet(nn.Module):
    """CGNet

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Tianyi Wu, et al. "CGNet: A Light-weight Context Guided Network for Semantic Segmentation."
        arXiv preprint arXiv:1811.08201 (2018).
    """

    def __init__(self, nclass, backbone='', aux=False, jpu=False,
        pretrained_base=True, M=3, N=21, **kwargs):
        super(CGNet, self).__init__()
        self.stage1_0 = _ConvBNPReLU(3, 32, 3, 2, 1, **kwargs)
        self.stage1_1 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)
        self.stage1_2 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)
        self.sample1 = _InputInjection(1)
        self.sample2 = _InputInjection(2)
        self.bn_prelu1 = _BNPReLU(32 + 3, **kwargs)
        self.stage2_0 = ContextGuidedBlock(32 + 3, 64, dilation=2,
            reduction=8, down=True, residual=False, **kwargs)
        self.stage2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.stage2.append(ContextGuidedBlock(64, 64, dilation=2,
                reduction=8, **kwargs))
        self.bn_prelu2 = _BNPReLU(128 + 3, **kwargs)
        self.stage3_0 = ContextGuidedBlock(128 + 3, 128, dilation=4,
            reduction=16, down=True, residual=False, **kwargs)
        self.stage3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.stage3.append(ContextGuidedBlock(128, 128, dilation=4,
                reduction=16, **kwargs))
        self.bn_prelu3 = _BNPReLU(256, **kwargs)
        self.head = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256,
            nclass, 1))
        self.__setattr__('exclusive', ['stage1_0', 'stage1_1', 'stage1_2',
            'sample1', 'sample2', 'bn_prelu1', 'stage2_0', 'stage2',
            'bn_prelu2', 'stage3_0', 'stage3', 'bn_prelu3', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        out0 = self.stage1_0(x)
        out0 = self.stage1_1(out0)
        out0 = self.stage1_2(out0)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)
        out0_cat = self.bn_prelu1(torch.cat([out0, inp1], dim=1))
        out1_0 = self.stage2_0(out0_cat)
        for i, layer in enumerate(self.stage2):
            if i == 0:
                out1 = layer(out1_0)
            else:
                out1 = layer(out1)
        out1_cat = self.bn_prelu2(torch.cat([out1, out1_0, inp2], dim=1))
        out2_0 = self.stage3_0(out1_cat)
        for i, layer in enumerate(self.stage3):
            if i == 0:
                out2 = layer(out2_0)
            else:
                out2 = layer(out2)
        out2_cat = self.bn_prelu3(torch.cat([out2_0, out2], dim=1))
        outputs = []
        out = self.head(out2_cat)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)
        return tuple(outputs)


class _ChannelWiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, **kwargs):
        super(_ChannelWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, dilation,
            dilation, groups=in_channels, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class _FGlo(nn.Module):

    def __init__(self, in_channels, reduction=16, **kwargs):
        super(_FGlo, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels //
            reduction), nn.ReLU(True), nn.Linear(in_channels // reduction,
            in_channels), nn.Sigmoid())

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.gap(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out


class _InputInjection(nn.Module):

    def __init__(self, ratio):
        super(_InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, 2, 1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class _ConcatInjection(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConcatInjection, self).__init__()
        self.bn = norm_layer(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ContextGuidedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=2, reduction=16,
        down=False, residual=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ContextGuidedBlock, self).__init__()
        inter_channels = out_channels // 2 if not down else out_channels
        if down:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 3, 2, 1,
                norm_layer=norm_layer, **kwargs)
            self.reduce = nn.Conv2d(inter_channels * 2, out_channels, 1,
                bias=False)
        else:
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 1, 1, 0,
                norm_layer=norm_layer, **kwargs)
        self.f_loc = _ChannelWiseConv(inter_channels, inter_channels, **kwargs)
        self.f_sur = _ChannelWiseConv(inter_channels, inter_channels,
            dilation, **kwargs)
        self.bn = norm_layer(inter_channels * 2)
        self.prelu = nn.PReLU(inter_channels * 2)
        self.f_glo = _FGlo(out_channels, reduction, **kwargs)
        self.down = down
        self.residual = residual

    def forward(self, x):
        out = self.conv(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi_feat = torch.cat([loc, sur], dim=1)
        joi_feat = self.prelu(self.bn(joi_feat))
        if self.down:
            joi_feat = self.reduce(joi_feat)
        out = self.f_glo(joi_feat)
        if self.residual:
            out = out + x
        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(
            0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(
            batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0,
            2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0
            ].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height,
            width)
        out = self.beta * feat_e + x
        return out


class _DAHead(nn.Module):

    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.
        BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.conv_c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels, **{} if norm_kwargs is None else norm_kwargs),
            nn.ReLU(True))
        self.conv_c2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels, **{} if norm_kwargs is None else norm_kwargs),
            nn.ReLU(True))
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels,
            nclass, 1))
        if aux:
            self.conv_p3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(
                inter_channels, nclass, 1))
            self.conv_c3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(
                inter_channels, nclass, 1))

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)
        feat_fusion = feat_p + feat_c
        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)
        return tuple(outputs)


class _DeepLabHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer,
            norm_kwargs=norm_kwargs, **kwargs)
        self.block = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=
            False), norm_layer(256, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True), nn.Dropout(0.1), nn.Conv2d(256,
            nclass, 1))

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


class _ASPPConv(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer,
        norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs,
        **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels, **{} if norm_kwargs is None else norm_kwargs), nn
            .ReLU(True))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs,
        **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels, **{} if norm_kwargs is
            None else norm_kwargs), nn.ReLU(True))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer,
            norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer,
            norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer,
            norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=
            norm_layer, norm_kwargs=norm_kwargs)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels,
            out_channels, 1, bias=False), norm_layer(out_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True), nn.
            Dropout(0.5))

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_xception(pretrained=False, root='~/.torch/models', **kwargs):
    model = Xception65(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('xception', root=root))
            )
    return model


class DeepLabV3Plus(nn.Module):
    """DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, nclass, backbone='xception', aux=True,
        pretrained_base=True, dilated=True, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.aux = aux
        self.nclass = nclass
        output_stride = 8 if dilated else 32
        self.pretrained = get_xception(pretrained=pretrained_base,
            output_stride=output_stride, **kwargs)
        self.head = _DeepLabHead(nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, nclass, **kwargs)

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv2(x)
        x = self.pretrained.bn2(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.block1(x)
        x = self.pretrained.relu(x)
        low_level_feat = x
        x = self.pretrained.block2(x)
        x = self.pretrained.block3(x)
        x = self.pretrained.midflow(x)
        mid_level_feat = x
        x = self.pretrained.block20(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv3(x)
        x = self.pretrained.bn3(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv4(x)
        x = self.pretrained.bn4(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv5(x)
        x = self.pretrained.bn5(x)
        x = self.pretrained.relu(x)
        return low_level_feat, mid_level_feat, x

    def forward(self, x):
        size = x.size()[2:]
        c1, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _DeepLabHead(nn.Module):

    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, **kwargs)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1,
            norm_layer=norm_layer)
        self.block = nn.Sequential(_ConvBNReLU(304, 256, 3, padding=1,
            norm_layer=norm_layer), nn.Dropout(0.5), _ConvBNReLU(256, 256, 
            3, padding=1, norm_layer=norm_layer), nn.Dropout(0.1), nn.
            Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))


class DilatedDenseNet(DenseNet):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        dilate_scale=8, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config,
            num_init_features, bn_size, drop_rate, num_classes, norm_layer)
        assert dilate_scale == 8 or dilate_scale == 16, 'dilate_scale can only set as 8 or 16'
        from functools import partial
        if dilate_scale == 8:
            self.features.denseblock3.apply(partial(self._conv_dilate,
                dilate=2))
            self.features.denseblock4.apply(partial(self._conv_dilate,
                dilate=4))
            del self.features.transition2.pool
            del self.features.transition3.pool
        elif dilate_scale == 16:
            self.features.denseblock4.apply(partial(self._conv_dilate,
                dilate=2))
            del self.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = dilate, dilate
                m.dilation = dilate, dilate


densenet_spec = {(121): (64, 32, [6, 12, 24, 16]), (161): (96, 48, [6, 12, 
    36, 24]), (169): (64, 32, [6, 12, 32, 32]), (201): (64, 32, [6, 12, 48,
    32])}


def get_dilated_densenet(num_layers, dilate_scale, pretrained=False, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DilatedDenseNet(growth_rate, block_config, num_init_features,
        dilate_scale=dilate_scale)
    if pretrained:
        pattern = re.compile(
            '^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$'
            )
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def dilated_densenet121(dilate_scale, **kwargs):
    return get_dilated_densenet(121, dilate_scale, **kwargs)


def dilated_densenet161(dilate_scale, **kwargs):
    return get_dilated_densenet(161, dilate_scale, **kwargs)


def dilated_densenet169(dilate_scale, **kwargs):
    return get_dilated_densenet(169, dilate_scale, **kwargs)


def dilated_densenet201(dilate_scale, **kwargs):
    return get_dilated_densenet(201, dilate_scale, **kwargs)


class DenseASPP(nn.Module):

    def __init__(self, nclass, backbone='densenet121', aux=False, jpu=False,
        pretrained_base=True, dilate_scale=8, **kwargs):
        super(DenseASPP, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.dilate_scale = dilate_scale
        if backbone == 'densenet121':
            self.pretrained = dilated_densenet121(dilate_scale, pretrained=
                pretrained_base, **kwargs)
        elif backbone == 'densenet161':
            self.pretrained = dilated_densenet161(dilate_scale, pretrained=
                pretrained_base, **kwargs)
        elif backbone == 'densenet169':
            self.pretrained = dilated_densenet169(dilate_scale, pretrained=
                pretrained_base, **kwargs)
        elif backbone == 'densenet201':
            self.pretrained = dilated_densenet201(dilate_scale, pretrained=
                pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        in_channels = self.pretrained.num_features
        self.head = _DenseASPPHead(in_channels, nclass)
        if aux:
            self.auxlayer = _FCNHead(in_channels, nclass, **kwargs)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head']
            )

    def forward(self, x):
        size = x.size()[2:]
        features = self.pretrained.features(x)
        if self.dilate_scale > 8:
            features = F.interpolate(features, scale_factor=2, mode=
                'bilinear', align_corners=True)
        outputs = []
        x = self.head(features)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(features)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _DenseASPPHead(nn.Module):

    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d,
        norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64,
            norm_layer, norm_kwargs)
        self.block = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(in_channels +
            5 * 64, nclass, 1))

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):

    def __init__(self, in_channels, inter_channels, out_channels,
        atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None
        ):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3,
            dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **{} if norm_kwargs is
            None else norm_kwargs)),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.
                training)
        return features


class _DenseASPPBlock(nn.Module):

    def __init__(self, in_channels, inter_channels1, inter_channels2,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1,
            inter_channels2, 3, 0.1, norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1,
            inter_channels1, inter_channels2, 6, 0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2,
            inter_channels1, inter_channels2, 12, 0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3,
            inter_channels1, inter_channels2, 18, 0.1, norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4,
            inter_channels1, inter_channels2, 24, 0.1, norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)
        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        return x


def get_xception_a(pretrained=False, root='~/.torch/models', **kwargs):
    model = XceptionA(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('xception_a', root=
            root)))
    return model


class DFANet(nn.Module):

    def __init__(self, nclass, backbone='', aux=False, jpu=False,
        pretrained_base=False, **kwargs):
        super(DFANet, self).__init__()
        self.pretrained = get_xception_a(pretrained_base, **kwargs)
        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)
        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)
        self.enc2_1_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_2_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.enc2_3_reduce = _ConvBNReLU(48, 32, 1, **kwargs)
        self.conv_fusion = _ConvBNReLU(32, 32, 1, **kwargs)
        self.fca_1_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_2_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.fca_3_reduce = _ConvBNReLU(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, nclass, 1)
        self.__setattr__('exclusive', ['enc2_2', 'enc3_2', 'enc4_2',
            'fca_2', 'enc2_3', 'enc3_3', 'enc3_4', 'fca_3', 'enc2_1_reduce',
            'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion', 'fca_1_reduce',
            'fca_2_reduce', 'fca_3_reduce', 'conv_out'])

    def forward(self, x):
        stage1_conv1 = self.pretrained.conv1(x)
        stage1_enc2 = self.pretrained.enc2(stage1_conv1)
        stage1_enc3 = self.pretrained.enc3(stage1_enc2)
        stage1_enc4 = self.pretrained.enc4(stage1_enc3)
        stage1_fca = self.pretrained.fca(stage1_enc4)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode=
            'bilinear', align_corners=True)
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode=
            'bilinear', align_corners=True)
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)
        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2),
            scale_factor=2, mode='bilinear', align_corners=True)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2),
            scale_factor=4, mode='bilinear', align_corners=True)
        fusion = (stage1_enc2_decoder + stage2_enc2_docoder +
            stage3_enc2_decoder)
        fusion = self.conv_fusion(fusion)
        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca),
            scale_factor=4, mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca),
            scale_factor=8, mode='bilinear', align_corners=True)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca),
            scale_factor=16, mode='bilinear', align_corners=True)
        fusion = (fusion + stage1_fca_decoder + stage2_fca_decoder +
            stage3_fca_decoder)
        outputs = list()
        out = self.conv_out(fusion)
        out = F.interpolate(out, scale_factor=4, mode='bilinear',
            align_corners=True)
        outputs.append(out)
        return tuple(outputs)


class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(nn.Conv2d(512, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(1024, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        size = c4.size()[2:]
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear',
            align_corners=True))
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear',
            align_corners=True))
        fused_feature = torch.cat([c4, c3, c2], dim=1)
        return fused_feature


class _DUHead(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer, **kwargs)
        self.block = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1,
            bias=False), norm_layer(256), nn.ReLU(True), nn.Conv2d(256, 256,
            3, padding=1, bias=False), norm_layer(256), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor *
            scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (
            self.scale_factor * self.scale_factor))
        x = x.permute(0, 3, 1, 2)
        return x


class _EncHead(nn.Module):

    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1,
            bias=False), norm_layer(512, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True))
        if lateral:
            self.connect = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512,
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True)), nn.Sequential(nn.Conv2d
                (1024, 512, 1, bias=False), norm_layer(512, **{} if 
                norm_kwargs is None else norm_kwargs), nn.ReLU(True))])
            self.fusion = nn.Sequential(nn.Conv2d(3 * 512, 512, 3, padding=
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True))
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(512,
            nclass, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class EncModule(nn.Module):

    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1,
            bias=False), norm_layer(in_channels, **{} if norm_kwargs is
            None else norm_kwargs), nn.ReLU(True), Encoding(D=in_channels,
            K=ncodes), nn.BatchNorm1d(ncodes), nn.ReLU(True), Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.
            Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Module):

    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D = X.size(0), self.D
        if X.dim() == 3:
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class Mean(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class ENet(nn.Module):
    """Efficient Neural Network"""

    def __init__(self, nclass, backbone='', aux=False, jpu=False,
        pretrained_base=None, **kwargs):
        super(ENet, self).__init__()
        self.initial = InitialBlock(13, **kwargs)
        self.bottleneck1_0 = Bottleneck(16, 16, 64, downsampling=True, **kwargs
            )
        self.bottleneck1_1 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_2 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_3 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_4 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck2_0 = Bottleneck(64, 32, 128, downsampling=True, **
            kwargs)
        self.bottleneck2_1 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_2 = Bottleneck(128, 32, 128, dilation=2, **kwargs)
        self.bottleneck2_3 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs
            )
        self.bottleneck2_4 = Bottleneck(128, 32, 128, dilation=4, **kwargs)
        self.bottleneck2_5 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_6 = Bottleneck(128, 32, 128, dilation=8, **kwargs)
        self.bottleneck2_7 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs
            )
        self.bottleneck2_8 = Bottleneck(128, 32, 128, dilation=16, **kwargs)
        self.bottleneck3_1 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_2 = Bottleneck(128, 32, 128, dilation=2, **kwargs)
        self.bottleneck3_3 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs
            )
        self.bottleneck3_4 = Bottleneck(128, 32, 128, dilation=4, **kwargs)
        self.bottleneck3_5 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_6 = Bottleneck(128, 32, 128, dilation=8, **kwargs)
        self.bottleneck3_7 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs
            )
        self.bottleneck3_8 = Bottleneck(128, 32, 128, dilation=16, **kwargs)
        self.bottleneck4_0 = UpsamplingBottleneck(128, 16, 64, **kwargs)
        self.bottleneck4_1 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_2 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck5_0 = UpsamplingBottleneck(64, 4, 16, **kwargs)
        self.bottleneck5_1 = Bottleneck(16, 4, 16, **kwargs)
        self.fullconv = nn.ConvTranspose2d(16, nclass, 2, 2, bias=False)
        self.__setattr__('exclusive', ['bottleneck1_0', 'bottleneck1_1',
            'bottleneck1_2', 'bottleneck1_3', 'bottleneck1_4',
            'bottleneck2_0', 'bottleneck2_1', 'bottleneck2_2',
            'bottleneck2_3', 'bottleneck2_4', 'bottleneck2_5',
            'bottleneck2_6', 'bottleneck2_7', 'bottleneck2_8',
            'bottleneck3_1', 'bottleneck3_2', 'bottleneck3_3',
            'bottleneck3_4', 'bottleneck3_5', 'bottleneck3_6',
            'bottleneck3_7', 'bottleneck3_8', 'bottleneck4_0',
            'bottleneck4_1', 'bottleneck4_2', 'bottleneck5_0',
            'bottleneck5_1', 'fullconv'])

    def forward(self, x):
        x = self.initial(x)
        x, max_indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)
        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x)
        x = self.fullconv(x)
        return tuple([x])


class InitialBlock(nn.Module):
    """ENet initial block"""

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = norm_layer(out_channels + 3)
        self.act = nn.PReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Bottlenecks include regular, asymmetric, downsampling, dilated"""

    def __init__(self, in_channels, inter_channels, out_channels, dilation=
        1, asymmetric=False, downsampling=False, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        if downsampling:
            self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
            self.conv_down = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, 1, bias=False), norm_layer(out_channels))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1,
            bias=False), norm_layer(inter_channels), nn.PReLU())
        if downsampling:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, 2, stride=2, bias=False), norm_layer(
                inter_channels), nn.PReLU())
        elif asymmetric:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, (5, 1), padding=(2, 0), bias=False), nn.
                Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 
                2), bias=False), norm_layer(inter_channels), nn.PReLU())
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, 3, dilation=dilation, padding=dilation,
                bias=False), norm_layer(inter_channels), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 
            1, bias=False), norm_layer(out_channels), nn.Dropout2d(0.1))
        self.act = nn.PReLU()

    def forward(self, x):
        identity = x
        if self.downsamping:
            identity, max_indices = self.maxpool(identity)
            identity = self.conv_down(identity)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + identity)
        if self.downsamping:
            return out, max_indices
        else:
            return out


class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, in_channels, inter_channels, out_channels,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(UpsamplingBottleneck, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels))
        self.upsampling = nn.MaxUnpool2d(2)
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1,
            bias=False), norm_layer(inter_channels), nn.PReLU(), nn.
            ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=
            False), norm_layer(inter_channels), nn.PReLU(), nn.Conv2d(
            inter_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.Dropout2d(0.1))
        self.act = nn.PReLU()

    def forward(self, x, max_indices):
        out_up = self.conv(x)
        out_up = self.upsampling(out_up, max_indices)
        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out


def eespnet(pretrained=False, **kwargs):
    model = EESPNet(**kwargs)
    if pretrained:
        raise ValueError("Don't support pretrained")
    return model


class ESPNetV2(nn.Module):
    """ESPNetV2

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Sachin Mehta, et al. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network."
        arXiv preprint arXiv:1811.11431 (2018).
    """

    def __init__(self, nclass, backbone='', aux=False, jpu=False,
        pretrained_base=False, **kwargs):
        super(ESPNetV2, self).__init__()
        self.pretrained = eespnet(pretrained=pretrained_base, **kwargs)
        self.proj_L4_C = _ConvBNPReLU(256, 128, 1, **kwargs)
        self.pspMod = nn.Sequential(EESP(256, 128, stride=1, k=4, r_lim=7,
            **kwargs), _PSPModule(128, 128, **kwargs))
        self.project_l3 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(128,
            nclass, 1, bias=False))
        self.act_l3 = _BNPReLU(nclass, **kwargs)
        self.project_l2 = _ConvBNPReLU(64 + nclass, nclass, 1, **kwargs)
        self.project_l1 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(32 +
            nclass, nclass, 1, bias=False))
        self.aux = aux
        self.__setattr__('exclusive', ['proj_L4_C', 'pspMod', 'project_l3',
            'act_l3', 'project_l2', 'project_l1'])

    def forward(self, x):
        size = x.size()[2:]
        out_l1, out_l2, out_l3, out_l4 = self.pretrained(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode=
            'bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode=
            'bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear',
            align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        outputs = list()
        merge1_l1 = F.interpolate(merge_l1, scale_factor=2, mode='bilinear',
            align_corners=True)
        outputs.append(merge1_l1)
        if self.aux:
            auxout = F.interpolate(proj_merge_l3_bef_act, size, mode=
                'bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _PSPModule(nn.Module):

    def __init__(self, in_channels, out_channels=1024, sizes=(1, 2, 4, 8),
        **kwargs):
        super(_PSPModule, self).__init__()
        self.stages = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 3,
            1, 1, groups=in_channels, bias=False) for _ in sizes])
        self.project = _ConvBNPReLU(in_channels * (len(sizes) + 1),
            out_channels, 1, 1, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for stage in self.stages:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(x), size, mode='bilinear',
                align_corners=True)
            feats.append(upsampled)
        return self.project(torch.cat(feats, dim=1))


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


_global_config['D'] = 4


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


class FCN32s(nn.Module):
    """There are some difference from original fcn"""

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN32s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.head = _FCNHead(512, nclass, norm_layer)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head']
            )

    def forward(self, x):
        size = x.size()[2:]
        pool5 = self.pretrained(x)
        outputs = []
        out = self.head(pool5)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class FCN16s(nn.Module):

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN16s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool4 = nn.Sequential(*self.pretrained[:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)
        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if
            aux else ['head', 'score_pool4'])

    def forward(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)
        outputs = []
        score_fr = self.head(pool5)
        score_pool4 = self.score_pool4(pool4)
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode=
            'bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4
        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear',
            align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class FCN8s(nn.Module):

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool3 = nn.Conv2d(256, nclass, 1)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)
        self.__setattr__('exclusive', ['head', 'score_pool3', 'score_pool4',
            'auxlayer'] if aux else ['head', 'score_pool3', 'score_pool4'])

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        outputs = []
        score_fr = self.head(pool5)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode=
            'bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4
        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:],
            mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3
        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear',
            align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **
        kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            inplace=True), nn.Dropout(0.1), nn.Conv2d(inter_channels,
            channels, 1))

    def forward(self, x):
        return self.block(x)


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d,
        norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True), nn.
            Dropout(0.1), nn.Conv2d(inter_channels, channels, 1))

    def forward(self, x):
        return self.block(x)


class HRNet(nn.Module):
    """HRNet

        Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Ke Sun. "High-Resolution Representations for Labeling Pixels and Regions."
        arXiv preprint arXiv:1904.04514 (2019).
    """

    def __init__(self, nclass, backbone='', aux=False, pretrained_base=
        False, **kwargs):
        super(HRNet, self).__init__()

    def forward(self, x):
        pass


class PyramidPoolingModule(nn.Module):

    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear',
                align_corners=True)
            feat = feat + x
        return feat


class _ICHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer,
            **kwargs)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass,
            norm_layer, **kwargs)
        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)
        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear',
            align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear',
            align_corners=True)
        outputs.append(up_x8)
        outputs.reverse()
        return outputs


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, bias=
        False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(nn.Conv2d(low_channels, out_channels,
            3, padding=2, dilation=2, bias=False), norm_layer(out_channels))
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels,
            out_channels, 1, bias=False), norm_layer(out_channels))
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode=
            'bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)
        return x, x_low_cls


class LEDNet(nn.Module):
    """LEDNet

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Yu Wang, et al. "LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation."
        arXiv preprint arXiv:1905.02423 (2019).
    """

    def __init__(self, nclass, backbone='', aux=False, jpu=False,
        pretrained_base=True, **kwargs):
        super(LEDNet, self).__init__()
        self.encoder = nn.Sequential(Downsampling(3, 32), SSnbt(32, **
            kwargs), SSnbt(32, **kwargs), SSnbt(32, **kwargs), Downsampling
            (32, 64), SSnbt(64, **kwargs), SSnbt(64, **kwargs),
            Downsampling(64, 128), SSnbt(128, **kwargs), SSnbt(128, 2, **
            kwargs), SSnbt(128, 5, **kwargs), SSnbt(128, 9, **kwargs),
            SSnbt(128, 2, **kwargs), SSnbt(128, 5, **kwargs), SSnbt(128, 9,
            **kwargs), SSnbt(128, 17, **kwargs))
        self.decoder = APNModule(128, nclass)
        self.__setattr__('exclusive', ['encoder', 'decoder'])

    def forward(self, x):
        size = x.size()[2:]
        x = self.encoder(x)
        x = self.decoder(x)
        outputs = list()
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        return tuple(outputs)


class Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2,
            bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2,
            bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)
        x2 = self.conv2(x)
        x2 = self.pool(x2)
        return torch.cat([x1, x2], dim=1)


class SSnbt(nn.Module):

    def __init__(self, in_channels, dilation=1, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(SSnbt, self).__init__()
        inter_channels = in_channels // 2
        self.branch1 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, (3, 1), padding=(1, 0), bias=False), nn.ReLU(
            True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, 1), bias=False), norm_layer(inter_channels), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, (1, 3), padding=(0, 1), bias=False), nn.ReLU(
            True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(1, 0), bias=False), norm_layer(inter_channels), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False), nn.
            ReLU(True), nn.Conv2d(inter_channels, inter_channels, (3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.relu = nn.ReLU(True)

    @staticmethod
    def channel_shuffle(x, groups):
        n, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(n, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(n, -1, h, w)
        return x

    def forward(self, x):
        x1, x2 = x.split(x.size(1) // 2, 1)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.relu(out + x)
        out = self.channel_shuffle(out, groups=2)
        return out


class APNModule(nn.Module):

    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs
        ):
        super(APNModule, self).__init__()
        self.conv1 = _ConvBNReLU(in_channels, in_channels, 3, 2, 1,
            norm_layer=norm_layer)
        self.conv2 = _ConvBNReLU(in_channels, in_channels, 5, 2, 2,
            norm_layer=norm_layer)
        self.conv3 = _ConvBNReLU(in_channels, in_channels, 7, 2, 3,
            norm_layer=norm_layer)
        self.level1 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level2 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level3 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level4 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer
            )
        self.level5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), _ConvBNReLU(
            in_channels, nclass, 1))

    def forward(self, x):
        w, h = x.size()[2:]
        branch3 = self.conv1(x)
        branch2 = self.conv2(branch3)
        branch1 = self.conv3(branch2)
        out = self.level1(branch1)
        out = F.interpolate(out, ((w + 3) // 4, (h + 3) // 4), mode=
            'bilinear', align_corners=True)
        out = self.level2(branch2) + out
        out = F.interpolate(out, ((w + 1) // 2, (h + 1) // 2), mode=
            'bilinear', align_corners=True)
        out = self.level3(branch3) + out
        out = F.interpolate(out, (w, h), mode='bilinear', align_corners=True)
        out = self.level4(x) * out
        out = self.level5(x) + out
        return out


class _OCHead(nn.Module):

    def __init__(self, nclass, oc_arch, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_OCHead, self).__init__()
        if oc_arch == 'base':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                BaseOCModule(512, 512, 256, 256, scales=[1], norm_layer=
                norm_layer, **kwargs))
        elif oc_arch == 'pyramid':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                PyramidOCModule(512, 512, 256, 512, scales=[1, 2, 3, 6],
                norm_layer=norm_layer, **kwargs))
        elif oc_arch == 'asp':
            self.context = ASPOCModule(2048, 512, 256, 512, norm_layer=
                norm_layer, **kwargs)
        else:
            raise ValueError('Unknown OC architecture!')
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.context(x)
        return self.out(x)


class BaseAttentionBlock(nn.Module):
    """The basic implementation for self-attention block/non-local block."""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BaseAttentionBlock, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(scale)
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1
            ).permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1
            ).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.bmm(query, key) * self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(w, h), mode='bilinear',
                align_corners=True)
        return context


class BaseOCModule(nn.Module):
    """Base-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, concat=True,
        **kwargs):
        super(BaseOCModule, self).__init__()
        self.stages = nn.ModuleList([BaseAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        in_channels = in_channels * 2 if concat else in_channels
        self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1
            ), norm_layer(out_channels), nn.ReLU(True), nn.Dropout2d(0.05))
        self.concat = concat

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        if self.concat:
            context = torch.cat([context, x], 1)
        out = self.project(context)
        return out


class PyramidAttentionBlock(nn.Module):
    """The basic implementation for pyramid self-attention block/non-local block"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidAttentionBlock, self).__init__()
        self.scale = scale
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        local_x = list()
        local_y = list()
        step_w, step_h = w // self.scale, h // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = step_w * i, step_h * j
                end_x, end_y = min(start_x + step_w, w), min(start_y +
                    step_h, h)
                if i == self.scale - 1:
                    end_x = w
                if j == self.scale - 1:
                    end_y = h
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = list()
        local_block_cnt = self.scale ** 2 * 2
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:
                local_y[i + 1]]
            w_local, h_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.
                value_channels, -1).permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.
                key_channels, -1).permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.
                key_channels, -1)
            sim_map = torch.bmm(query_local, key_local
                ) * self.key_channels ** -0.5
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.bmm(sim_map, value_local).permute(0, 2, 1
                ).contiguous()
            context_local = context_local.view(batch_size, self.
                value_channels, w_local, h_local)
            local_list.append(context_local)
        context_list = list()
        for i in range(0, self.scale):
            row_tmp = list()
            for j in range(self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidOCModule(nn.Module):
    """Pyramid-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidOCModule, self).__init__()
        self.stages = nn.ModuleList([PyramidAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels * len
            (scales), 1), norm_layer(in_channels * len(scales)), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(in_channels * len(scales) * 
            2, out_channels, 1), norm_layer(out_channels), nn.ReLU(True),
            nn.Dropout2d(0.05))

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = [self.up_dr(x)]
        for i in range(len(priors)):
            context += [priors[i]]
        context = torch.cat(context, 1)
        out = self.project(context)
        return out


class ASPOCModule(nn.Module):
    """ASP-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, atrous_rates=(12, 24, 36), norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(ASPOCModule, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=1), norm_layer(out_channels), nn.ReLU(True),
            BaseOCModule(out_channels, out_channels, key_channels,
            value_channels, [2], norm_layer, False, **kwargs))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate1, dilation=rate1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate2, dilation=rate2, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate3, dilation=rate3, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5,
            out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU
            (True), nn.Dropout2d(0.1))

    def forward(self, x):
        feat1 = self.context(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.project(out)
        return out


class _PSAHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        self.psa = _PointwiseSpatialAttention(2048, 3600, norm_layer)
        self.conv_post = _ConvBNReLU(1024, 2048, 1, norm_layer=norm_layer)
        self.project = nn.Sequential(_ConvBNReLU(4096, 512, 3, padding=1,
            norm_layer=norm_layer), nn.Dropout2d(0.1, False), nn.Conv2d(512,
            nclass, 1))

    def forward(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)
        return out


class _PointwiseSpatialAttention(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = 512
        self.collect_attention = _AttentionGeneration(in_channels,
            reduced_channels, out_channels, norm_layer)
        self.distribute_attention = _AttentionGeneration(in_channels,
            reduced_channels, out_channels, norm_layer)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):

    def __init__(self, in_channels, reduced_channels, out_channels,
        norm_layer, **kwargs):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1,
            norm_layer=norm_layer)
        self.attention = nn.Sequential(_ConvBNReLU(reduced_channels,
            reduced_channels, 1, norm_layer=norm_layer), nn.Conv2d(
            reduced_channels, out_channels, 1, bias=False))
        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.size()
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))
        fm = fm.view(n, self.reduced_channels, h, w)
        return fm


class _PSAHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        self.collect = _CollectModule(2048, 512, 60, 60, norm_layer, **kwargs)
        self.distribute = _DistributeModule(2048, 512, 60, 60, norm_layer,
            **kwargs)
        self.conv_post = nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False),
            norm_layer(2048), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(4096, 512, 3, padding=1,
            bias=False), norm_layer(512), nn.ReLU(True), nn.Conv2d(512,
            nclass, 1))

    def forward(self, x):
        global_feature_collect = self.collect(x)
        global_feature_distribute = self.distribute(x)
        global_feature = torch.cat([global_feature_collect,
            global_feature_distribute], dim=1)
        out = self.conv_post(global_feature)
        out = F.interpolate(out, scale_factor=2, mode='bilinear',
            align_corners=True)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)
        return out


class _CollectModule(nn.Module):

    def __init__(self, in_channels, reduced_channels, feat_w, feat_h,
        norm_layer, **kwargs):
        super(_CollectModule, self).__init__()
        self.conv_reduce = nn.Sequential(nn.Conv2d(in_channels,
            reduced_channels, 1, bias=False), norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(nn.Conv2d(reduced_channels,
            reduced_channels, 1, bias=False), norm_layer(reduced_channels),
            nn.ReLU(True), nn.Conv2d(reduced_channels, (feat_w - 1) *
            feat_h, 1, bias=False))
        self.collect_attention = CollectAttention()
        self.reduced_channels = reduced_channels
        self.feat_w = feat_w
        self.feat_h = feat_h

    def forward(self, x):
        x = self.conv_reduce(x)
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear',
            align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)
        ca = self.collect_attention(x_adaption)
        global_feature_collect_list = list()
        for i in range(x_shrink.shape[0]):
            x_shrink_i = x_shrink[i].view(self.reduced_channels, -1)
            ca_i = ca[i].view(ca.shape[1], -1)
            global_feature_collect_list.append(torch.mm(x_shrink_i, ca_i).
                view(1, self.reduced_channels, self.feat_h // 2, self.
                feat_w // 2))
        global_feature_collect = torch.cat(global_feature_collect_list)
        return global_feature_collect


class _DistributeModule(nn.Module):

    def __init__(self, in_channels, reduced_channels, feat_w, feat_h,
        norm_layer, **kwargs):
        super(_DistributeModule, self).__init__()
        self.conv_reduce = nn.Sequential(nn.Conv2d(in_channels,
            reduced_channels, 1, bias=False), norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(nn.Conv2d(reduced_channels,
            reduced_channels, 1, bias=False), norm_layer(reduced_channels),
            nn.ReLU(True), nn.Conv2d(reduced_channels, (feat_w - 1) *
            feat_h, 1, bias=False))
        self.distribute_attention = DistributeAttention()
        self.reduced_channels = reduced_channels
        self.feat_w = feat_w
        self.feat_h = feat_h

    def forward(self, x):
        x = self.conv_reduce(x)
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear',
            align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)
        da = self.distribute_attention(x_adaption)
        global_feature_distribute_list = list()
        for i in range(x_shrink.shape[0]):
            x_shrink_i = x_shrink[i].view(self.reduced_channels, -1)
            da_i = da[i].view(da.shape[1], -1)
            global_feature_distribute_list.append(torch.mm(x_shrink_i, da_i
                ).view(1, self.reduced_channels, self.feat_h // 2, self.
                feat_w // 2))
        global_feature_distribute = torch.cat(global_feature_distribute_list)
        return global_feature_distribute


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False
        ), norm_layer(out_channels, **{} if norm_kwargs is None else
        norm_kwargs), nn.ReLU(True))


class _PyramidPooling(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode=
            'bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode=
            'bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode=
            'bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode=
            'bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):

    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs
            =norm_kwargs)
        self.block = nn.Sequential(nn.Conv2d(4096, 512, 3, padding=1, bias=
            False), norm_layer(512, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True), nn.Dropout(0.1), nn.Conv2d(512,
            nclass, 1))

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


def resnet101_v1s(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, **kwargs)
    if pretrained:
        from ..model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet101', root=
            root)), strict=False)
    return model


def resnet152_v1s(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, **kwargs)
    if pretrained:
        from ..model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet152', root=
            root)), strict=False)
    return model


def resnet50_v1s(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    if pretrained:
        from ..model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet50', root=
            root)), strict=False)
    return model


class SegBaseModel(nn.Module):
    """Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False,
        pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base,
                dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base,
                dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base,
                dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu6=False, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ConvBNPReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _BNPReLU(nn.Module):

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BNPReLU, self).__init__()
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), **kwargs):
        super(_PSPModule, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpool.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1, **
                kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for avgpool, conv in enumerate(zip(self.avgpools, self.convs)):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode=
                'bilinear', align_corners=True))
        return torch.cat(feats, dim=1)


class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(_ConvBNReLU(in_channels, in_channels, 3,
            stride, 1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6
                =True, norm_layer=norm_layer))
        layers.extend([_ConvBNReLU(inter_channels, inter_channels, 3,
            stride, 1, groups=inter_channels, relu6=True, norm_layer=
            norm_layer), nn.Conv2d(inter_channels, out_channels, 1, bias=
            False), norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _CAMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, g):
        out = _C.ca_map_forward(weight, g)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw, dg = _C.ca_map_backward(dout, weight, g)
        return dw, dg


ca_map = _CAMap.apply


class _CAWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        weight = _C.ca_forward(t, f)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt, df = _C.ca_backward(dw, t, f)
        return dt, df


ca_weight = _CAWeight.apply


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x
        return out


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
        dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride,
            padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):

    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels[-2], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels[-3], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.dilation1 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=1, dilation=1, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=2, dilation=2, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=4, dilation=4, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=8, dilation=8, bias=False), norm_layer(width), nn.ReLU(
            True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3
            (inputs[-3])]
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear',
            align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear',
            align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.
            dilation3(feat), self.dilation4(feat)], dim=1)
        return inputs[0], inputs[1], inputs[2], feat


class _PSACollect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hc):
        out = _C.psa_forward(hc, 1)
        ctx.save_for_backward(hc)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        hc = ctx.saved_tensors
        dhc = _C.psa_backward(dout, hc[0], 1)
        return dhc


psa_collect = _PSACollect.apply


class CollectAttention(nn.Module):
    """Collect Attention Generation Module"""

    def __init__(self):
        super(CollectAttention, self).__init__()

    def forward(self, x):
        out = psa_collect(x)
        return out


class _PSADistribute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hc):
        out = _C.psa_forward(hc, 2)
        ctx.save_for_backward(hc)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        hc = ctx.saved_tensors
        dhc = _C.psa_backward(dout, hc[0], 2)
        return dhc


psa_distribute = _PSADistribute.apply


class DistributeAttention(nn.Module):
    """Distribute Attention Generation Module"""

    def __init__(self):
        super(DistributeAttention, self).__init__()

    def forward(self, x):
        out = psa_distribute(x)
        return out


def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == 'leaky_relu':
        if x.is_cuda:
            lib.gpu.leaky_relu_backward(x, dx, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'


def _act_forward(ctx, x):
    if ctx.activation.lower() == 'leaky_relu':
        if x.is_cuda:
            lib.gpu.leaky_relu_forward(x, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'


class inp_syncbatchnorm_(Function):

    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var, extra,
        sync=True, training=True, momentum=0.1, eps=1e-05, activation=
        'none', slope=0.01):
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.gpu.expectation_forward(x)
            else:
                raise NotImplemented
            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsuqeeze(0))
                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)
                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.
                        get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()
            _var = _exs - _ex ** 2
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * _ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * _var)
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2
            ctx.mark_dirty(x)
        if x.is_cuda:
            lib.gpu.batchnorm_inp_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented
        _act_forward(ctx, x)
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = lib.gpu.batchnorm_inp_backward(dz,
                z, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented
        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))
                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)
                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex
                        .get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()
            if z.is_cuda:
                lib.gpu.expectation_inp_backward(dx, z, _dex, _dexs, _ex,
                    _exs, gamma, beta, ctx.eps)
            else:
                raise NotImplemented
        return (dx, dgamma, dbeta, None, None, None, None, None, None, None,
            None, None)

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra['is_master']
        if ctx.is_master:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queues = extra['worker_queues']
            ctx.worker_ids = extra['worker_ids']
        else:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queue = extra['worker_queue']


inp_syncbatchnorm = inp_syncbatchnorm_.apply


class _SyncBatchNorm(Function):

    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var, extra,
        sync=True, training=True, momentum=0.1, eps=1e-05, activation=
        'none', slope=0.01):
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        assert activation == 'none'
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        if ctx.training:
            _ex, _exs = _C.expectation_forward(x)
            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))
                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)
                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.
                        get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()
            _var = _exs - _ex ** 2
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * _ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * _var)
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2
        y = _C.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()
        dx, _dex, _dexs, dgamma, dbeta = _C.batchnorm_backward(dz, x, _ex,
            _exs, gamma, beta, ctx.eps)
        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))
                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)
                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex
                        .get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()
            dx_ = _C.expectation_backward(x, _dex, _dexs)
            dx = dx + dx_
        return (dx, dgamma, dbeta, None, None, None, None, None, None, None,
            None, None)

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra['is_master']
        if ctx.is_master:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queues = extra['worker_queues']
            ctx.worker_ids = extra['worker_ids']
        else:
            ctx.master_queue = extra['master_queue']
            ctx.worker_queue = extra['worker_queue']


syncbatchnorm = _SyncBatchNorm.apply


class SyncBatchNorm(_BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Parameters:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, sync=True,
        activation='none', slope=0.01, inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum
            =momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        self.slope = slope
        self.devices = list(range(torch.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            extra = {'is_master': True, 'master_queue': self.master_queue,
                'worker_queues': self.worker_queues, 'worker_ids': self.
                worker_ids}
        else:
            extra = {'is_master': False, 'master_queue': self.master_queue,
                'worker_queue': self.worker_queues[self.worker_ids.index(x.
                get_device())]}
        if self.inplace:
            return inp_syncbatchnorm(x, self.weight, self.bias, self.
                running_mean, self.running_var, extra, self.sync, self.
                training, self.momentum, self.eps, self.activation, self.slope
                ).view(input_shape)
        else:
            return syncbatchnorm(x, self.weight, self.bias, self.
                running_mean, self.running_var, extra, self.sync, self.
                training, self.momentum, self.eps, self.activation, self.slope
                ).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(self.sync,
                self.activation, self.slope, self.inplace)


class SyncBatchNorm(_BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Parameters:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, sync=True,
        activation='none', slope=0.01):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum
            =momentum, affine=True)
        self.activation = activation
        self.slope = slope
        self.devices = list(range(torch.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            extra = {'is_master': True, 'master_queue': self.master_queue,
                'worker_queues': self.worker_queues, 'worker_ids': self.
                worker_ids}
        else:
            extra = {'is_master': False, 'master_queue': self.master_queue,
                'worker_queue': self.worker_queues[self.worker_ids.index(x.
                get_device())]}
        return syncbatchnorm(x, self.weight, self.bias, self.running_mean,
            self.running_var, extra, self.sync, self.training, self.
            momentum, self.eps, self.activation, self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}'.format(self.sync, self.
                activation, self.slope)


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=
            ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target
            )
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds
                [i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).
                forward(*inputs))


class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
        aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.
                se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), bins=nclass,
                min=0, max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode=
            'bilinear', align_corners=True).squeeze(1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.
            aux_weight)


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000,
        use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            None
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.
                long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()
        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.criterion(pred, target)


class DataParallelModel(DataParallel):
    """Data parallelism

    Hide the difference of single/multiple GPUs to the user.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    sync : bool
        enable synchronization (default: False).
    Inputs:
        - **inputs**: list of input
    Outputs:
        - **outputs**: list of output
    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        return modules


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutputs):
        return Broadcast.apply(ctx.target_gpus, gradOutputs)


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    """Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional), attr:'targets' (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        targets (tensor): targets to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:'targets' :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(*(list(input) + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0]
            )
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Tramac_awesome_semantic_segmentation_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(APNModule(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ASPOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(AttentionRefinmentModule(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(BaseAttentionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(BaseOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(BasicBlockV1b(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(Bottleneck(*[], **{'in_channels': 4, 'inter_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(CGNet(*[], **{'nclass': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_009(self):
        self._check(CascadeFeatureFusion(*[], **{'low_channels': 4, 'high_channels': 4, 'out_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(DUpsampling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(DataParallelCriterion(*[], **{'module': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(DataParallelModel(*[], **{'module': _mock_layer()}), [], {'input': torch.rand([4, 4])})

    @_fails_compile()
    def test_013(self):
        self._check(DenseNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_014(self):
        self._check(DilatedDenseNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_015(self):
        self._check(Downsampling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_016(self):
        self._check(EESP(*[], **{'in_channels': 4, 'out_channels': 64}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(EESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_018(self):
        self._check(ENet(*[], **{'nclass': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_019(self):
        self._check(EncModule(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_020(self):
        self._check(Encoding(*[], **{'D': 4, 'K': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(FeatureFused(*[], **{}), [torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 4, 4, 4])], {})

    def test_022(self):
        self._check(FeatureFusion(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    def test_023(self):
        self._check(HRNet(*[], **{'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(InitialBlock(*[], **{'out_channels': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_025(self):
        self._check(InvertedResidual(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_026(self):
        self._check(LEDNet(*[], **{'nclass': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_027(self):
        self._check(Mean(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    def test_028(self):
        self._check(MobileNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_029(self):
        self._check(MobileNetV2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_030(self):
        self._check(PyramidAttentionBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_031(self):
        self._check(PyramidOCModule(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_032(self):
        self._check(PyramidPoolingModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_033(self):
        self._check(SSnbt(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_034(self):
        self._check(SeparableConv2d(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_035(self):
        self._check(SpatialPath(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_036(self):
        self._check(Xception65(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_037(self):
        self._check(Xception71(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_038(self):
        self._check(XceptionA(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_039(self):
        self._check(_BNPReLU(*[], **{'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_040(self):
        self._check(_BiSeHead(*[], **{'in_channels': 4, 'inter_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_041(self):
        self._check(_ChannelAttentionModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_042(self):
        self._check(_ChannelWiseConv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_043(self):
        self._check(_ConvBN(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_044(self):
        self._check(_ConvBNPReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_045(self):
        self._check(_ConvBNReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_046(self):
        self._check(_DAHead(*[], **{'in_channels': 64, 'nclass': 4}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_047(self):
        self._check(_DenseASPPBlock(*[], **{'in_channels': 4, 'inter_channels1': 4, 'inter_channels2': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_048(self):
        self._check(_DenseASPPConv(*[], **{'in_channels': 4, 'inter_channels': 4, 'out_channels': 4, 'atrous_rate': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_049(self):
        self._check(_DenseASPPHead(*[], **{'in_channels': 4, 'nclass': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_050(self):
        self._check(_DenseBlock(*[], **{'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_051(self):
        self._check(_DenseLayer(*[], **{'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    def test_052(self):
        self._check(_DepthwiseConv(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_053(self):
        self._check(_FCNHead(*[], **{'in_channels': 4, 'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_054(self):
        self._check(_GlobalAvgPooling(*[], **{'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer}), [torch.rand([4, 4, 4, 4])], {})

    def test_055(self):
        self._check(_InputInjection(*[], **{'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_056(self):
        self._check(_PSPHead(*[], **{'nclass': 4}), [torch.rand([4, 2048, 4, 4])], {})

    def test_057(self):
        self._check(_PositionAttentionModule(*[], **{'in_channels': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_058(self):
        self._check(_Transition(*[], **{'num_input_features': 4, 'num_output_features': 4}), [torch.rand([4, 4, 4, 4])], {})

