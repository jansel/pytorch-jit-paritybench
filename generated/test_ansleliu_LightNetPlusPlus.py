import sys
_module = sys.modules[__name__]
del sys
checkpoint = _module
datasets = _module
augmentations = _module
cityscapes = _module
list = _module
make_list = _module
select_hard = _module
utils = _module
bdd2cityscapes = _module
get_mean_std = _module
deploy = _module
evaluation = _module
eval = _module
examples = _module
video_demo = _module
video_demo = _module
weight_release = _module
models = _module
mixnetseg = _module
mobilenetv2plus = _module
shufflenetv2plus = _module
modules = _module
aspp = _module
attentions = _module
deformable = _module
functions = _module
deform_conv = _module
deform_pool = _module
deform_conv = _module
deform_pool = _module
setup = _module
dense = _module
dropout = _module
efficient = _module
inplace_abn = _module
iabn = _module
misc = _module
mobile = _module
residual = _module
shuffle = _module
usm = _module
netviz = _module
feat_viz = _module
adabound = _module
losses = _module
lr_scheduler = _module
metrics = _module
parallel = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import torch.nn as nn


import numpy as np


import torch


import time


from functools import partial


from torch.nn import functional as F


from collections import OrderedDict


from torch import nn


import math


from torch.nn import init


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import torch.nn.functional as functional


import scipy.ndimage as nd


import functools


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import torch.nn.init as init


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class SEBlock(nn.Module):

    def __init__(self, in_planes, reduced_dim, act_type='relu'):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([('linear1', nn.Conv2d(
            in_planes, reduced_dim, kernel_size=1, stride=1, padding=0,
            bias=True)), ('act', Swish(inplace=True) if act_type == 'swish'
             else nn.LeakyReLU(inplace=True, negative_slope=0.01)), (
            'linear2', nn.Conv2d(reduced_dim, in_planes, kernel_size=1,
            stride=1, padding=0, bias=True))]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x,
            output_size=(1, 1))))
        return torch.mul(x, x_se)


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups
        =1, dilate=1, act_type='relu'):
        super(ConvBlock, self).__init__()
        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        padding = (kernel_size - 1) // 2 * dilate
        self.conv_block = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilate,
            groups=groups, bias=False)), ('norm', nn.BatchNorm2d(
            num_features=out_planes, eps=0.001, momentum=0.01)), ('act', 
            Swish(inplace=True) if act_type == 'swish' else nn.LeakyReLU(
            inplace=True, negative_slope=0.01))]))

    def forward(self, x):
        return self.conv_block(x)


class GPConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_sizes):
        super(GPConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_in_dim = in_planes // self.num_groups
        sub_out_dim = out_planes // self.num_groups
        self.group_point_wise = nn.ModuleList()
        for _ in kernel_sizes:
            self.group_point_wise.append(nn.Conv2d(sub_in_dim, sub_out_dim,
                kernel_size=1, stride=1, padding=0, groups=1, dilation=1,
                bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.group_point_wise[0](x)
        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.group_point_wise[stream](chunks[stream]) for stream in
            range(self.num_groups)]
        return torch.cat(mix, dim=1)


class MDConv(nn.Module):

    def __init__(self, in_planes, kernel_sizes, stride=1, dilate=1):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        assert in_planes % self.num_groups == 0
        sub_hidden_dim = in_planes // self.num_groups
        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        self.mixed_depth_wise = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2 * dilate
            self.mixed_depth_wise.append(nn.Conv2d(sub_hidden_dim,
                sub_hidden_dim, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=sub_hidden_dim, dilation=dilate,
                bias=False))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depth_wise[0](x)
        chunks = torch.chunk(x, chunks=self.num_groups, dim=1)
        mix = [self.mixed_depth_wise[stream](chunks[stream]) for stream in
            range(self.num_groups)]
        return torch.cat(mix, dim=1)


class MixDepthBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expand_ratio,
        exp_kernel_sizes, kernel_sizes, poi_kernel_sizes, stride, dilate,
        reduction_ratio=4, dropout_rate=0.2, act_type='swish'):
        super(MixDepthBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.groups = len(kernel_sizes)
        self.use_se = reduction_ratio is not None and reduction_ratio > 1
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        if expand_ratio != 1:
            self.expansion = nn.Sequential(OrderedDict([('conv', GPConv(
                in_planes, hidden_dim, kernel_sizes=exp_kernel_sizes)), (
                'norm', nn.BatchNorm2d(hidden_dim, eps=0.001, momentum=0.01
                )), ('act', Swish(inplace=True) if act_type == 'swish' else
                nn.LeakyReLU(inplace=True, negative_slope=0.01))]))
        self.depth_wise = nn.Sequential(OrderedDict([('conv', MDConv(
            hidden_dim, kernel_sizes=kernel_sizes, stride=stride, dilate=
            dilate)), ('norm', nn.BatchNorm2d(hidden_dim, eps=0.001,
            momentum=0.01)), ('act', Swish(inplace=True) if act_type ==
            'swish' else nn.LeakyReLU(inplace=True, negative_slope=0.01))]))
        if self.use_se:
            reduced_dim = max(1, int(in_planes / reduction_ratio))
            self.se_block = SEBlock(hidden_dim, reduced_dim, act_type=act_type)
        self.point_wise = nn.Sequential(OrderedDict([('conv', GPConv(
            hidden_dim, out_planes, kernel_sizes=poi_kernel_sizes)), (
            'norm', nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01))]))

    def forward(self, x):
        res = x
        if self.expand_ratio != 1:
            x = self.expansion(x)
        x = self.depth_wise(x)
        if self.use_se:
            x = self.se_block(x)
        x = self.point_wise(x)
        if self.use_residual:
            if self.training and self.dropout_rate is not None:
                x = F.dropout2d(input=x, p=self.dropout_rate, training=self
                    .training, inplace=True)
            x = x + res
        return x


class DSASPPBlock(nn.Module):

    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(6, 12, 18)):
        super(DSASPPBlock, self).__init__()
        self.up_ratio = up_ratio
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((3, 3))), ('conv1_0', ConvBlock(in_chs,
            out_chs, kernel_size=1, stride=1, dilate=1, act_type='relu'))]))
        self.conv1x1 = ConvBlock(in_chs, out_chs, kernel_size=1, stride=1,
            dilate=1, act_type='relu')
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv', MDConv(
            in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=
            aspp_dilate[0])), ('norm', nn.BatchNorm2d(in_chs, eps=0.001,
            momentum=0.01)), ('act', nn.LeakyReLU(inplace=True,
            negative_slope=0.01))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv', MDConv(
            in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=
            aspp_dilate[1])), ('norm', nn.BatchNorm2d(in_chs, eps=0.001,
            momentum=0.01)), ('act', nn.LeakyReLU(inplace=True,
            negative_slope=0.01))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv', MDConv(
            in_planes=in_chs, kernel_sizes=[3, 5, 7, 9], stride=1, dilate=
            aspp_dilate[2])), ('norm', nn.BatchNorm2d(in_chs, eps=0.001,
            momentum=0.01)), ('act', nn.LeakyReLU(inplace=True,
            negative_slope=0.01))]))
        self.aspp_catdown = ConvBlock(3 * in_chs + 2 * out_chs, out_chs,
            kernel_size=1, stride=1, dilate=1, act_type='relu')

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()
        x = self.aspp_catdown(torch.cat((self.aspp_bra1(x), F.interpolate(
            input=self.gave_pool(x), size=(feat_h, feat_w), mode='bilinear',
            align_corners=True), self.aspp_bra2(x), self.conv1x1(x), self.
            aspp_bra3(x)), dim=1))
        return F.interpolate(input=x, size=(int(feat_h * self.up_ratio),
            int(feat_w * self.up_ratio)), mode='bilinear', align_corners=True)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, expand_ratio=1, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        self.p1_td = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p2_td = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p3_td = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p4_td = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p2_bu = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p3_bu = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p4_bu = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.p5_bu = MixDepthBlock(feature_size, feature_size, expand_ratio
            =expand_ratio, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=2,
            dropout_rate=0.0, act_type='relu')
        self.w1 = nn.Parameter(torch.Tensor(2, 4).fill_(0.5))
        self.w2 = nn.Parameter(torch.Tensor(3, 4).fill_(0.5))

    def forward(self, inputs):
        p1_x, p2_x, p3_x, p4_x, p5_x = inputs
        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * p5_td)
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * p4_td)
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * p3_td)
        p1_td = self.p1_td(w1[0, 3] * p1_x + w1[1, 3] * F.interpolate(p2_td,
            scale_factor=2, mode='bilinear', align_corners=True))
        p1_bu = p1_td
        p2_bu = self.p2_bu(w2[0, 0] * p2_x + w2[1, 0] * p2_td + w2[2, 0] *
            F.interpolate(p1_bu, scale_factor=0.5, mode='bilinear',
            align_corners=True))
        p3_bu = self.p3_bu(w2[0, 1] * p3_x + w2[1, 1] * p3_td + w2[2, 1] *
            p2_bu)
        p4_bu = self.p4_bu(w2[0, 2] * p4_x + w2[1, 2] * p4_td + w2[2, 2] *
            p3_bu)
        p5_bu = self.p5_bu(w2[0, 3] * p5_x + w2[1, 3] * p5_td + w2[2, 3] *
            p4_bu)
        return p1_bu, p2_bu, p3_bu, p4_bu, p5_bu


class BiFPNDecoder(nn.Module):

    def __init__(self, bone_feat_sizes, feature_size=64, expand_ratio=1,
        fpn_repeats=3):
        super(BiFPNDecoder, self).__init__()
        self.p1 = ConvBlock(bone_feat_sizes[0], feature_size, kernel_size=1,
            stride=1, act_type='relu')
        self.p2 = ConvBlock(bone_feat_sizes[1], feature_size, kernel_size=1,
            stride=1, act_type='relu')
        self.p3 = ConvBlock(bone_feat_sizes[2], feature_size, kernel_size=1,
            stride=1, act_type='relu')
        self.p4 = ConvBlock(bone_feat_sizes[3], feature_size, kernel_size=1,
            stride=1, act_type='relu')
        self.p5 = ConvBlock(bone_feat_sizes[4], feature_size, kernel_size=1,
            stride=1, act_type='relu')
        bifpns_seq = []
        for bifpn_id in range(fpn_repeats):
            bifpns_seq.append(('bi_fpn%d' % (bifpn_id + 1), BiFPNBlock(
                feature_size=feature_size, expand_ratio=expand_ratio)))
        self.bifpns = nn.Sequential(OrderedDict(bifpns_seq))

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        return self.bifpns([self.p1(feat1), self.p2(feat2), self.p3(feat3),
            self.p4(feat4), self.p5(feat5)])


class MixNetSeg(nn.Module):

    def __init__(self, arch='s', decoder_feat=64, fpn_repeats=3, num_classes=19
        ):
        super(MixNetSeg, self).__init__()
        self.num_classes = num_classes
        params = {'s': (16, [[1, 16, 1, [3], [1], [1], 1, 1, 'relu', None],
            [6, 24, 1, [3], [1, 1], [1, 1], 2, 1, 'relu', None], [3, 24, 1,
            [3], [1, 1], [1, 1], 1, 1, 'relu', None], [6, 40, 1, [3, 5, 7],
            [1], [1], 2, 1, 'relu', 2], [6, 40, 3, [3, 5], [1, 1], [1, 1], 
            1, 1, 'relu', 2], [6, 80, 1, [3, 5, 7], [1], [1, 1], 1, 2,
            'relu', 4], [6, 80, 2, [3, 5], [1], [1, 1], 1, 2, 'relu', 4], [
            6, 120, 1, [3, 5, 7], [1, 1], [1, 1], 1, 3, 'relu', 2], [3, 120,
            2, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'relu', 2], [6, 200, 1,
            [3, 5, 7, 9, 11], [1], [1], 1, 4, 'relu', 2], [6, 200, 2, [3, 5,
            7, 9], [1], [1, 1], 1, 4, 'relu', 2]], 1.0, 1.0, 0.2), 'm': (24,
            [[1, 24, 1, [3], [1], [1], 1, 1, 'relu', None], [6, 32, 1, [3, 
            5, 7], [1, 1], [1, 1], 2, 1, 'relu', None], [3, 32, 1, [3], [1,
            1], [1, 1], 1, 1, 'relu', None], [6, 40, 1, [3, 5, 7, 9], [1],
            [1], 2, 1, 'relu', 2], [6, 40, 3, [3, 5], [1, 1], [1, 1], 1, 1,
            'relu', 2], [6, 80, 1, [3, 5, 7], [1], [1], 1, 2, 'relu', 4], [
            6, 80, 3, [3, 5, 7, 9], [1, 1], [1, 1], 1, 2, 'relu', 4], [6, 
            120, 1, [3], [1], [1], 1, 3, 'relu', 2], [3, 120, 3, [3, 5, 7, 
            9], [1, 1], [1, 1], 1, 3, 'relu', 2], [6, 200, 1, [3, 5, 7, 9],
            [1], [1], 1, 4, 'relu', 2], [6, 200, 3, [3, 5, 7, 9], [1], [1, 
            1], 1, 4, 'relu', 2]], 1.0, 1.0, 0.25), 'l': (24, [[1, 24, 1, [
            3], [1], [1], 1, 1, 'relu', None], [6, 32, 1, [3, 5, 7], [1, 1],
            [1, 1], 2, 1, 'relu', None], [3, 32, 1, [3], [1, 1], [1, 1], 1,
            1, 'relu', None], [6, 40, 1, [3, 5, 7, 9], [1], [1], 2, 1,
            'relu', 2], [6, 40, 3, [3, 5], [1, 1], [1, 1], 1, 1, 'relu', 2],
            [6, 80, 1, [3, 5, 7], [1], [1], 1, 2, 'relu', 4], [6, 80, 3, [3,
            5, 7, 9], [1, 1], [1, 1], 1, 2, 'relu', 4], [6, 120, 1, [3], [1
            ], [1], 1, 3, 'relu', 2], [3, 120, 3, [3, 5, 7, 9], [1, 1], [1,
            1], 1, 3, 'relu', 2], [6, 200, 1, [3, 5, 7, 9], [1], [1], 1, 4,
            'relu', 2], [6, 200, 3, [3, 5, 7, 9], [1], [1, 1], 1, 4, 'relu',
            2]], 1.3, 1.0, 0.25)}
        (stem_planes, settings, width_multi, depth_multi, self.dropout_rate
            ) = params[arch]
        out_channels = self._round_filters(stem_planes, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2,
            groups=1, dilate=1, act_type='relu')
        in_channels = out_channels
        mod_id = 0
        for t, c, n, k, ek, pk, s, d, a, se in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d if stride == 1 else 1
                blocks.append(('block%d' % (block_id + 1), MixDepthBlock(
                    in_channels, out_channels, expand_ratio=t,
                    exp_kernel_sizes=ek, kernel_sizes=k, poi_kernel_sizes=
                    pk, stride=stride, dilate=dilate, reduction_ratio=se,
                    dropout_rate=0.0, act_type=a)))
                in_channels = out_channels
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(
                OrderedDict(blocks)))
            mod_id += 1
        org_last_planes = settings[0][1] + settings[2][1] + settings[4][1
            ] + settings[6][1] + settings[8][1] + settings[10][1]
        last_feat = 256
        self.feat_fuse = MixDepthBlock(org_last_planes, last_feat,
            expand_ratio=3, exp_kernel_sizes=[1], kernel_sizes=[3, 5, 7, 9],
            poi_kernel_sizes=[1], stride=1, dilate=1, reduction_ratio=1,
            dropout_rate=0.0, act_type='relu')
        self.bifpn_decoder = BiFPNDecoder(bone_feat_sizes=[settings[2][1],
            settings[4][1], settings[6][1], settings[8][1], last_feat],
            feature_size=decoder_feat, expand_ratio=2, fpn_repeats=fpn_repeats)
        self.aux_head = nn.Conv2d(last_feat, num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.cls_head = nn.Conv2d(decoder_feat, num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in',
                    nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_divisible(value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        if width_multi == 1.0:
            return filters
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    @staticmethod
    def usm(x, kernel_size=(7, 7), amount=1.0, threshold=0):
        res = x.clone()
        blurred = gaussian_blur2d(x, kernel_size=kernel_size, sigma=(1.0, 1.0))
        sharpened = res * (amount + 1.0) - amount * blurred
        if threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < threshold,
                sharpened, res)
        return F.relu(sharpened, inplace=True)

    def forward(self, x):
        _, _, in_h, in_w = x.size()
        assert in_h % 32 == 0 and in_w % 32 == 0, '> in_size must product of 32!!!'
        feat1 = self.mod2(self.mod1(x))
        feat1_1 = F.max_pool2d(input=feat1, kernel_size=3, stride=2, padding=1)
        feat2 = self.mod4(self.mod3(feat1))
        feat3 = self.mod6(self.mod5(feat2))
        feat4 = self.mod8(self.mod7(feat3))
        feat5 = self.mod10(self.mod9(feat4))
        feat6 = self.mod12(self.mod11(feat5))
        feat = self.feat_fuse(torch.cat([feat4, F.max_pool2d(input=feat1_1,
            kernel_size=3, stride=2, padding=1), feat3, feat6, F.max_pool2d
            (input=feat2, kernel_size=3, stride=2, padding=1), feat5], dim=1))
        feat = feat + F.interpolate(F.adaptive_avg_pool2d(feat, output_size
            =(3, 3)), size=(feat.size(2), feat.size(3)), mode='bilinear',
            align_corners=True)
        aux_score = self.aux_head(feat)
        feat_de2, feat_de3, feat_de4, feat_de5, feat_de = self.bifpn_decoder(
            feat2, feat3, feat4, feat5, feat)
        feat_final = feat_de2 + F.interpolate(feat_de3 + feat_de4 +
            feat_de5 + feat_de, scale_factor=2, mode='bilinear',
            align_corners=True)
        main_score = self.cls_head(feat_final)
        main_score = F.interpolate(input=main_score, size=(in_h, in_w),
            mode='bilinear', align_corners=True)
        aux_score = F.interpolate(input=aux_score, size=(in_h, in_w), mode=
            'bilinear', align_corners=True)
        return aux_score, main_score


class ASPPBlock(nn.Module):

    def __init__(self, in_chs, out_chs, up_ratio=2, aspp_dilate=(4, 8, 12)):
        super(ASPPBlock, self).__init__()
        self.up_ratio = up_ratio
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((3, 3))), ('conv1_0', nn.Conv2d(in_chs,
            out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=
            False, dilation=1)), ('bn1_1', nn.BatchNorm2d(num_features=
            out_chs))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False,
            groups=1, dilation=1)), ('bn1_1', nn.BatchNorm2d(num_features=
            out_chs))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv2_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_dilate[0
            ], bias=False, groups=1, dilation=aspp_dilate[0])), ('bn2_1',
            nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv2_2', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_dilate[1
            ], bias=False, groups=1, dilation=aspp_dilate[1])), ('bn2_2',
            nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv2_3', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_dilate[2
            ], bias=False, groups=1, dilation=aspp_dilate[2])), ('bn2_3',
            nn.BatchNorm2d(num_features=out_chs))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('conv_down', nn.
            Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1,
            bias=False, groups=1, dilation=1)), ('bn_down', nn.BatchNorm2d(
            num_features=out_chs)), ('act', nn.LeakyReLU(inplace=True,
            negative_slope=0.1)), ('dropout', nn.Dropout2d(p=0.25, inplace=
            True))]))

    def forward(self, x):
        _, _, feat_h, feat_w = x.size()
        x = torch.cat((self.aspp_bra1(x), F.interpolate(input=self.
            gave_pool(x), size=(feat_h, feat_w), mode='bilinear',
            align_corners=True), self.aspp_bra2(x), self.conv1x1(x), self.
            aspp_bra3(x)), dim=1)
        return F.interpolate(input=self.aspp_catdown(x), size=(int(feat_h *
            self.up_ratio), int(feat_w * self.up_ratio)), mode='bilinear',
            align_corners=True)


class SEBlock(nn.Module):

    def __init__(self, channel, reduct_ratio=16):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([('avgpool', nn.
            AdaptiveAvgPool2d(1)), ('linear1', nn.Conv2d(channel, channel //
            reduct_ratio, kernel_size=1, stride=1, padding=0)), ('relu', nn
            .ReLU(inplace=True)), ('linear2', nn.Conv2d(channel //
            reduct_ratio, channel, kernel_size=1, stride=1, padding=0))]))

    def forward(self, x):
        inputs = x
        chn_se = self.channel_se(x).sigmoid().exp()
        return torch.mul(inputs, chn_se)


class SCSEBlock(nn.Module):

    def __init__(self, channel, reduct_ratio=16, is_res=True):
        super(SCSEBlock, self).__init__()
        self.is_res = is_res
        self.channel_se = nn.Sequential(OrderedDict([('avgpool', nn.
            AdaptiveAvgPool2d(1)), ('linear1', nn.Conv2d(channel, channel //
            reduct_ratio, kernel_size=1, stride=1, padding=0)), ('relu', nn
            .ReLU(inplace=True)), ('linear2', nn.Conv2d(channel //
            reduct_ratio, channel, kernel_size=1, stride=1, padding=0))]))
        self.spatial_se = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            channel, 1, kernel_size=1, stride=1, padding=0, bias=False))]))

    def forward(self, x):
        inputs = x
        chn_se = self.channel_se(x).sigmoid().exp()
        spa_se = self.spatial_se(x).sigmoid().exp()
        if self.is_res:
            torch.mul(torch.mul(inputs, chn_se), spa_se) + inputs
        return torch.mul(torch.mul(inputs, chn_se), spa_se)


class ModifiedSCSEBlock(nn.Module):

    def __init__(self, in_chns, reduct_ratio=16, is_res=True):
        super(ModifiedSCSEBlock, self).__init__()
        self.is_res = is_res
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_se = nn.Sequential(nn.Conv2d(in_chns, in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0), nn.ReLU(
            inplace=True), nn.Conv2d(in_chns // reduct_ratio, in_chns,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(in_chns))
        self.spatial_se = nn.Sequential(nn.Conv2d(in_chns, 1, kernel_size=1,
            stride=1, padding=0, bias=False), nn.BatchNorm2d(1))

    def forward(self, x):
        res = x
        ch_att = self.channel_se(self.ch_avg_pool(x) + self.ch_max_pool(x))
        ch_att = torch.mul(x, ch_att.sigmoid().exp())
        sp_att = torch.mul(x, self.spatial_se(x).sigmoid().exp())
        if self.is_res:
            ch_att + res + sp_att
        return ch_att + sp_att


class SCSABlock(nn.Module):

    def __init__(self, in_chns, reduct_ratio=16, is_res=True):
        super(SCSABlock, self).__init__()
        self.is_res = is_res
        if is_res:
            self.gamma = nn.Parameter(torch.ones(1))
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0), nn.ReLU(
            inplace=True), nn.Conv2d(in_chns // reduct_ratio, in_chns,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(in_chns))
        self.sp_conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=
            1, padding=3, bias=False), nn.BatchNorm2d(1))

    def forward(self, x):
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))
        ch_att = torch.mul(x, (avg_p + max_p).sigmoid().exp())
        ch_avg = torch.mean(ch_att, dim=1, keepdim=True)
        ch_max = torch.max(ch_att, dim=1, keepdim=True)[0]
        sp_att = torch.mul(ch_att, self.sp_conv(torch.cat([ch_avg, ch_max],
            dim=1)).sigmoid().exp())
        if self.is_res:
            return sp_att + self.gamma * res
        return sp_att


class PBCSABlock(nn.Module):
    """
    Parallel Bottleneck Channel-Spatial Attention Block
    """

    def __init__(self, in_chns, reduct_ratio=16, dilation=4, use_res=True):
        super(PBCSABlock, self).__init__()
        self.use_res = use_res
        self.ch_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ch_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0), nn.ReLU(
            inplace=True), nn.Conv2d(in_chns // reduct_ratio, in_chns,
            kernel_size=1, stride=1, padding=0))
        self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
            kernel_size=3, stride=1, padding=dilation, dilation=dilation,
            bias=False), nn.Conv2d(in_chns // reduct_ratio, in_chns //
            reduct_ratio, kernel_size=3, stride=1, padding=dilation,
            dilation=dilation, bias=False), nn.Conv2d(in_chns //
            reduct_ratio, 1, kernel_size=1, stride=1, padding=0, bias=False
            ), nn.BatchNorm2d(1))

    def forward(self, x):
        ch_att = self.se_block(self.ch_avg_pool(x) + self.ch_max_pool(x))
        ch_att = torch.mul(x, ch_att.sigmoid().exp())
        sp_att = torch.mul(x, self.sp_conv(x).sigmoid().exp())
        if self.use_res:
            return sp_att + x + ch_att
        return sp_att + ch_att


class PABlock(nn.Module):
    """Position Attention Block"""

    def __init__(self, in_chns, reduct_ratio=8):
        super(PABlock, self).__init__()
        self.in_chns = in_chns
        self.query = nn.Conv2d(in_channels=in_chns, out_channels=in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=in_chns, out_channels=in_chns //
            reduct_ratio, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=in_chns, out_channels=in_chns,
            kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, feat_h, feat_w = x.size()
        proj_query = self.query(x).view(batch_size, -1, feat_h * feat_w
            ).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, feat_h * feat_w)
        proj_value = self.value(x).view(batch_size, -1, feat_h * feat_w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention).view(batch_size, channels,
            feat_h, feat_w)
        return self.gamma * out + x


class CABlock(nn.Module):
    """Channel Attention Block"""

    def __init__(self):
        super(CABlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, feat_h, feat_w = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        proj_value = x.view(batch_size, channels, -1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, dim=-1, keepdim=True)[0].expand_as(
            energy) - energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, proj_value).view(batch_size, channels,
            feat_h, feat_w)
        return self.gamma * out + x


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1,
        groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                'Expected 4D tensor as input, got {}D tensor instead.'.
                format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input,
            weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset,
                output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.
                size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.
                padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups,
                ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input,
                    offset, grad_output, grad_input, grad_offset, weight,
                    ctx.bufs_[0], weight.size(3), weight.size(2), ctx.
                    stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0
                    ], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.
                    deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input,
                    offset, grad_output, grad_weight, ctx.bufs_[0], ctx.
                    bufs_[1], weight.size(3), weight.size(2), ctx.stride[1],
                    ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.
                    dilation[1], ctx.dilation[0], ctx.groups, ctx.
                    deformable_groups, 1, cur_im2col_step)
        return (grad_input, grad_offset, grad_weight, None, None, None,
            None, None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.
                format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        assert not bias
        super(DeformConv, self).__init__()
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(
            in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(
            out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv(input, offset, self.weight, self.stride, self.
            padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if (weight.requires_grad or mask.requires_grad or offset.
            requires_grad or input.requires_grad):
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(
            ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight,
            bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.
            shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding,
            ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.
            deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight,
            bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input,
            grad_weight, grad_bias, grad_offset, grad_mask, grad_output,
            weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.
            padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups,
            ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
            None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h -
            1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 
            1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self
            .bias, self.stride, self.padding, self.dilation, self.groups,
            self.deformable_groups)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size,
        out_channels, no_trans, group_size=1, part_size=None,
        sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(data, rois,
            offset, output, output_count, ctx.no_trans, ctx.spatial_scale,
            ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size,
            ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        deform_pool_cuda.deform_psroi_pooling_cuda_backward(grad_output,
            data, rois, offset, output_count, grad_input, grad_offset, ctx.
            no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size,
            ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
            None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale,
            self.out_size, self.out_channels, self.no_trans, self.
            group_size, self.part_size, self.sample_per_part, self.trans_std)


class DropBlock2D(nn.Module):
    """Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1.0 - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M, torch.ones((input.shape[1], 1, self.block_size,
            self.block_size)).to(device=input.device, dtype=input.dtype),
            padding=self.block_size // 2, groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() / mask.sum()


class DSConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilate=1
        ):
        super(DSConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate
        padding = (kernel_size - 1) // 2 * dilate
        self.depth_wise = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_planes, in_planes, kernel_size, stride, padding, dilate,
            groups=in_planes, bias=False)), ('norm', nn.BatchNorm2d(
            num_features=out_planes, eps=0.001, momentum=0.01))]))
        self.point_wise = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False)), ('norm', nn.BatchNorm2d(
            num_features=out_planes, eps=0.001, momentum=0.01)), ('act', nn
            .LeakyReLU(negative_slope=0.01, inplace=True))]))

    def forward(self, x):
        return self.point_wise(self.depth_wise(x))


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups
        =1, dilate=1):
        super(ConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate
        padding = (kernel_size - 1) // 2 * dilate
        self.conv_block = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilate, groups=groups, bias=False)),
            ('norm', nn.BatchNorm2d(num_features=out_planes, eps=0.001,
            momentum=0.01)), ('act', nn.LeakyReLU(negative_slope=0.01,
            inplace=True))]))

    def forward(self, x):
        return self.conv_block(x)


class SEBlock(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([('linear1', nn.Conv2d(
            in_planes, reduced_dim, kernel_size=1, stride=1, padding=0,
            bias=True)), ('act', nn.LeakyReLU(negative_slope=0.01, inplace=
            True)), ('linear2', nn.Conv2d(reduced_dim, in_planes,
            kernel_size=1, stride=1, padding=0, bias=True))]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x,
            output_size=(1, 1))))
        return torch.mul(x, x_se)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        self.p1_td = DSConvBlock(feature_size, feature_size)
        self.p2_td = DSConvBlock(feature_size, feature_size)
        self.p3_td = DSConvBlock(feature_size, feature_size)
        self.p4_td = DSConvBlock(feature_size, feature_size)
        self.p2_out = DSConvBlock(feature_size, feature_size)
        self.p3_out = DSConvBlock(feature_size, feature_size)
        self.p4_out = DSConvBlock(feature_size, feature_size)
        self.p5_out = DSConvBlock(feature_size, feature_size)
        self.w1 = nn.Parameter(torch.Tensor(2, 4).fill_(0.5))
        self.w2 = nn.Parameter(torch.Tensor(3, 4).fill_(0.5))

    def forward(self, inputs):
        p1_x, p2_x, p3_x, p4_x, p5_x = inputs
        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * F.interpolate(p5_td,
            scale_factor=2, mode='bilinear', align_corners=True))
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * F.interpolate(p4_td,
            scale_factor=2, mode='bilinear', align_corners=True))
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * F.interpolate(p3_td,
            scale_factor=2, mode='bilinear', align_corners=True))
        p1_td = self.p1_td(w1[0, 3] * p1_x + w1[1, 3] * F.interpolate(p2_td,
            scale_factor=2, mode='bilinear', align_corners=True))
        p1_out = p1_td
        p2_out = self.p2_out(w2[0, 0] * p2_x + w2[1, 0] * p2_td + w2[2, 0] *
            F.interpolate(p1_out, scale_factor=0.5, mode='bilinear',
            align_corners=True))
        p3_out = self.p3_out(w2[0, 1] * p3_x + w2[1, 1] * p3_td + w2[2, 1] *
            F.interpolate(p2_out, scale_factor=0.5, mode='bilinear',
            align_corners=True))
        p4_out = self.p4_out(w2[0, 2] * p4_x + w2[1, 2] * p4_td + w2[2, 2] *
            F.interpolate(p3_out, scale_factor=0.5, mode='bilinear',
            align_corners=True))
        p5_out = self.p5_out(w2[0, 3] * p5_x + w2[1, 3] * p5_td + w2[2, 3] *
            F.interpolate(p4_out, scale_factor=0.5, mode='bilinear',
            align_corners=True))
        return p1_out, p2_out, p3_out, p4_out, p5_out


class BiFPNDecoder(nn.Module):

    def __init__(self, bone_feat_sizes, feature_size=64, fpn_repeats=2):
        super(BiFPNDecoder, self).__init__()
        self.p1 = ConvBlock(bone_feat_sizes[0], feature_size, kernel_size=1,
            stride=1)
        self.p2 = ConvBlock(bone_feat_sizes[1], feature_size, kernel_size=1,
            stride=1)
        self.p3 = ConvBlock(bone_feat_sizes[2], feature_size, kernel_size=1,
            stride=1)
        self.p4 = ConvBlock(bone_feat_sizes[3], feature_size, kernel_size=1,
            stride=1)
        self.p5 = ConvBlock(bone_feat_sizes[4], feature_size, kernel_size=1,
            stride=1)
        bifpns_seq = []
        for bifpn_id in range(fpn_repeats):
            bifpns_seq.append(('bi_fpn%d' % (bifpn_id + 1), BiFPNBlock(
                feature_size)))
        self.bifpns = nn.Sequential(OrderedDict(bifpns_seq))

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        p1 = self.p1(feat1)
        p2 = self.p2(feat2)
        p3 = self.p3(feat3)
        p4 = self.p4(feat4)
        p5 = self.p5(feat5)
        return self.bifpns([p1, p2, p3, p4, p5])


class MBConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expand_ratio, kernel_size,
        stride, dilate, reduction_ratio=4, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.use_se = reduction_ratio is not None and reduction_ratio > 1
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))
        if expand_ratio != 1:
            self.expansion = ConvBlock(in_planes, hidden_dim, 1)
        self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
            stride=stride, groups=hidden_dim, dilate=dilate)
        if self.use_se:
            self.se_block = SEBlock(hidden_dim, reduced_dim)
        self.point_wise = nn.Sequential(OrderedDict([('conv', nn.Conv2d(
            hidden_dim, out_planes, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False)), ('norm', nn.BatchNorm2d(
            out_planes, eps=0.001, momentum=0.01))]))

    def forward(self, x):
        res = x
        if self.expand_ratio != 1:
            x = self.expansion(x)
        x = self.depth_wise(x)
        if self.use_se:
            x = self.se_block(x)
        x = self.point_wise(x)
        if self.use_residual:
            if self.training and self.dropout_rate is not None:
                x = F.dropout2d(input=x, p=self.dropout_rate, training=self
                    .training, inplace=True)
            x = x + res
        return x


ACT_ELU = 'elu'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
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
        x = functional.batch_norm(x, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope,
                inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class CoordInfo(nn.Module):

    def __init__(self, with_r=True):
        super(CoordInfo, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Add Cartesian Coordination Info to Current Tensor
        :param x: shape(N, C, H, W)
        :return:  shape(N, C+2 or C+3, H, W)
        """
        batch_size, _, height, width = x.size()
        i_coords = torch.arange(height).repeat(1, width, 1).transpose(1, 2)
        j_coords = torch.arange(width).repeat(1, height, 1)
        i_coords = i_coords.float() / (height - 1)
        j_coords = j_coords.float() / (width - 1)
        i_coords = i_coords * 2 - 1
        j_coords = j_coords * 2 - 1
        i_coords = i_coords.repeat(batch_size, 1, 1, 1)
        j_coords = j_coords.repeat(batch_size, 1, 1, 1)
        ret = torch.cat([x, i_coords.type_as(x), j_coords.type_as(x)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(i_coords.type_as(x) - 0.5, 2) + torch
                .pow(j_coords.type_as(x) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class LightHeadBlock(nn.Module):

    def __init__(self, in_chs, mid_chs=64, out_chs=256, kernel_size=15):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv_l = nn.Sequential(OrderedDict([('conv_lu', nn.Conv2d(
            in_chs, mid_chs, kernel_size=(kernel_size, 1), padding=(pad, 0)
            )), ('conv_ld', nn.Conv2d(mid_chs, out_chs, kernel_size=(1,
            kernel_size), padding=(0, pad)))]))
        self.conv_r = nn.Sequential(OrderedDict([('conv_ru', nn.Conv2d(
            in_chs, mid_chs, kernel_size=(1, kernel_size), padding=(0, pad)
            )), ('conv_rd', nn.Conv2d(mid_chs, out_chs, kernel_size=(
            kernel_size, 1), padding=(pad, 0)))]))

    def forward(self, x):
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=
            inp * expand_ratio, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=
            inp * expand_ratio), nn.LeakyReLU(inplace=True, negative_slope=
            0.01), nn.Conv2d(in_channels=inp * expand_ratio, out_channels=
            inp * expand_ratio, kernel_size=3, stride=stride, padding=
            dilate, dilation=dilate, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio), nn.LeakyReLU(
            inplace=True, negative_slope=0.01), nn.Conv2d(in_channels=inp *
            expand_ratio, out_channels=oup, kernel_size=1, stride=1,
            padding=0, dilation=1, groups=1, bias=False), nn.BatchNorm2d(
            num_features=oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=
        1, norm_act=ABN, use_se=False, dropout=None):
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
            raise ValueError('channels must contain either two or three values'
                )
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3,
                stride=stride, padding=dilation, bias=False, dilation=
                dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.
                Conv2d(channels[0], channels[1], 3, stride=1, padding=
                dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                stride=stride, padding=0, bias=False)), ('bn2', norm_act(
                channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1],
                3, stride=1, padding=dilation, bias=False, groups=groups,
                dilation=dilation)), ('bn3', norm_act(channels[1])), (
                'conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1,
                padding=0, bias=False))]
            if use_se:
                layers.append(('se_block', SEBlock(channels[2],
                    reduct_ratio=16)))
            if dropout is not None:
                layers = layers[0:5] + [('dropout', dropout())] + layers[5:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride
                =stride, padding=0, bias=False)

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


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleRes(nn.Module):

    def __init__(self, in_chns, out_chns, stride, dilate, branch_model):
        super(ShuffleRes, self).__init__()
        self.branch_model = branch_model
        assert stride in [1, 2]
        self.stride = stride
        mid_chns = out_chns // 2
        if self.branch_model == 1:
            self.branch2 = nn.Sequential(nn.Conv2d(mid_chns, mid_chns, 1, 1,
                0, bias=False), nn.BatchNorm2d(mid_chns), nn.LeakyReLU(
                inplace=True, negative_slope=0.01), nn.Conv2d(mid_chns,
                mid_chns, kernel_size=3, stride=stride, padding=dilate,
                dilation=dilate, groups=mid_chns, bias=False), nn.
                BatchNorm2d(mid_chns), nn.Conv2d(mid_chns, mid_chns, 1, 1, 
                0, bias=False), nn.BatchNorm2d(mid_chns), nn.LeakyReLU(
                inplace=True, negative_slope=0.01))
        else:
            self.branch1 = nn.Sequential(nn.Conv2d(in_chns, in_chns,
                kernel_size=3, stride=stride, padding=dilate, dilation=
                dilate, groups=in_chns, bias=False), nn.BatchNorm2d(in_chns
                ), nn.Conv2d(in_chns, mid_chns, 1, 1, 0, bias=False), nn.
                BatchNorm2d(mid_chns), nn.LeakyReLU(inplace=True,
                negative_slope=0.01))
            self.branch2 = nn.Sequential(nn.Conv2d(in_chns, mid_chns, 1, 1,
                0, bias=False), nn.BatchNorm2d(mid_chns), nn.LeakyReLU(
                inplace=True, negative_slope=0.01), nn.Conv2d(mid_chns,
                mid_chns, kernel_size=3, stride=stride, padding=dilate,
                dilation=dilate, groups=mid_chns, bias=False), nn.
                BatchNorm2d(mid_chns), nn.Conv2d(mid_chns, mid_chns, 1, 1, 
                0, bias=False), nn.BatchNorm2d(mid_chns), nn.LeakyReLU(
                inplace=True, negative_slope=0.01))

    def forward(self, x):
        if 1 == self.branch_model:
            x1 = x[:, :x.shape[1] // 2, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        elif 2 == self.branch_model:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class UnsharpMaskV2(nn.Module):

    def __init__(self, channel, kernel_size=7, padding=3, amount=1.0,
        threshold=0, norm_act=ABN):
        super(UnsharpMaskV2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.amount = amount
        self.threshold = threshold
        self.norm_act = norm_act(channel)

    def forward(self, x):
        x = self.norm_act(x)
        res = x.clone()
        blurred = F.avg_pool2d(input=x, kernel_size=self.kernel_size,
            stride=1, padding=self.padding, ceil_mode=False,
            count_include_pad=True)
        sharpened = res * (self.amount + 1.0) - blurred * self.amount
        if self.threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < self.
                threshold, sharpened, res)
        return sharpened


class GaussianBlur(nn.Module):

    def __init__(self, channels, kernel_size=11, padding=5, sigma=1.6):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.sigma = sigma
        weights = self.calculate_weights()
        self.register_buffer('gaussian_filter', weights)

    def calculate_weights(self):
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size,
            self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (self.kernel_size - 1) / 2.0
        variance = self.sigma ** 2
        gaussian_kernel = 1.0 / (2.0 * math.pi * variance) * torch.exp(-
            torch.sum((xy_grid.float() - mean) ** 2, dim=-1) / (2.0 * variance)
            )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self
            .kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        return gaussian_kernel

    def forward(self, x):
        return F.conv2d(input=x, weight=self.gaussian_filter, stride=1,
            padding=self.padding, groups=x.size(1), bias=None)


class UnsharpMask(nn.Module):

    def __init__(self, channels, kernel_size=11, padding=5, sigma=1.0,
        amount=1.0, threshold=0, norm_act=ABN):
        super(UnsharpMask, self).__init__()
        self.amount = amount
        self.threshold = threshold
        self.norm_act = norm_act(channels)
        self.gauss_blur = GaussianBlur(channels=channels, kernel_size=
            kernel_size, padding=padding, sigma=sigma)

    def forward(self, x):
        x = self.norm_act(x)
        res = x.clone()
        blurred = self.gauss_blur(x)
        sharpened = res * (self.amount + 1.0) - blurred * self.amount
        if self.threshold > 0:
            sharpened = torch.where(torch.abs(res - blurred) < self.
                threshold, sharpened, res)
        return sharpened


class HookBasedFeatureExtractor(nn.Module):

    def __init__(self, model, layer_name, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        None

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        None

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)):
                self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        layers = self.model._modules['module']._modules
        target_layer = layers[self.layer_name]
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.model(x)
        h_inp.remove()
        h_out.remove()
        if self.upscale:
            self.rescale_output_array(x.size())
        return self.inputs, self.outputs


class BootstrappedCrossEntropy2D(nn.Module):

    def __init__(self, top_k=128, ignore_index=-100):
        """
        Bootstrapped CrossEntropy2D: The pixel-bootstrapped cross entropy loss

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(BootstrappedCrossEntropy2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544,
            1.03602707, 1.03413147, 1.68195437, 5.58540548, 3.56563995, 
            0.12704978, 1.0, 0.46783719, 1.34551528, 5.29974114, 0.28342531,
            0.9396095, 0.81551811, 0.42679146, 3.6399074, 2.78376194])
        self.top_k = top_k
        self.ignore_index = ignore_index

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :param top_k: <int> Top-K worst predictions
        :return: <torch.Tensor> loss
        """
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()
                batch_loss = F.cross_entropy(input=predict, target=targets,
                    weight=None, ignore_index=self.ignore_index, reduction=
                    'none')
                loss = 0.0
                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h * feat_w)
                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k
                loss_fuse += loss / float(batch_size)
        else:
            batch_size, channels, feat_h, feat_w = predictions.size()
            batch_loss = F.cross_entropy(input=predictions, target=targets,
                weight=None, ignore_index=self.ignore_index, reduction='none')
            loss = 0.0
            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)
                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k
            loss_fuse += loss / float(batch_size)
        return loss_fuse


class CriterionDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, top_k=512 * 512, ignore_index=255):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = BootstrappedCrossEntropy2D(top_k=top_k,
            ignore_index=ignore_index)

    def _update_topk(self, top_k):
        self.criterion._update_topk(top_k)

    def forward(self, predictions, targets):
        loss1 = self.criterion(predictions[0], targets)
        loss2 = self.criterion(predictions[1], targets)
        return loss1 + loss2 * 0.4


class OHEMBootstrappedCrossEntropy2D(nn.Module):

    def __init__(self, factor=8.0, thresh=0.7, min_kept=100000, top_k=128,
        ignore_index=-100):
        """
        Bootstrapped CrossEntropy2D: The pixel-bootstrapped cross entropy loss

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(OHEMBootstrappedCrossEntropy2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544,
            1.03602707, 1.03413147, 1.68195437, 5.58540548, 3.56563995, 
            0.12704978, 1.0, 0.46783719, 1.34551528, 5.29974114, 0.28342531,
            0.9396095, 0.81551811, 0.42679146, 3.6399074, 2.78376194])
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.factor = factor
        self.thresh = thresh
        self.min_kept = int(min_kept)

    def find_threshold(self, np_predict, np_target):
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor
            ), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)
        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)
        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        threshold = 1.0
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), int(min_kept)) - 1
                new_array = np.partition(pred, int(k_th))
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape
        threshold = self.find_threshold(np_predict, np_target)
        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long(
            )
        return new_target

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :param top_k: <int> Top-K worst predictions
        :return: <torch.Tensor> loss
        """
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()
                input_prob = F.softmax(predict, dim=1)
                targets = self.generate_new_target(input_prob, targets)
                batch_loss = F.cross_entropy(input=predict, target=targets,
                    weight=self.weight, ignore_index=self.ignore_index,
                    reduction='none')
                loss = 0.0
                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h * feat_w)
                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k
                loss_fuse += loss / float(batch_size)
        else:
            batch_size, channels, feat_h, feat_w = predictions.size()
            input_prob = F.softmax(predictions, dim=1)
            targets = self.generate_new_target(input_prob, targets)
            batch_loss = F.cross_entropy(input=predictions, target=targets,
                weight=self.weight, ignore_index=self.ignore_index,
                reduction='none')
            loss = 0.0
            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)
                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k
            loss_fuse += loss / float(batch_size)
        return loss_fuse


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """

    def __init__(self, top_k=128, ignore_label=250, alpha=0.25, gamma=2):
        """
        Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        :param ignore_label:  <int> ignore label
        :param alpha:         <torch.Tensor> the scalar factor
        :param gamma:         <float> gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        """
        super(FocalLoss2D, self).__init__()
        self.weight = torch.FloatTensor([0.05570516, 0.32337477, 0.08998544,
            1.03602707, 1.03413147, 1.68195437, 5.58540548, 3.56563995, 
            0.12704978, 1.0, 0.46783719, 1.34551528, 5.29974114, 0.28342531,
            0.9396095, 0.81551811, 0.42679146, 3.6399074, 2.78376194])
        self.alpha = alpha
        self.gamma = gamma
        self.top_k = top_k
        self.ignore_label = ignore_label
        self.one_hot = torch.eye(self.num_classes)

    def _update_topk(self, top_k):
        self.top_k = top_k

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        assert not targets.requires_grad
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                batch_size, channels, feat_h, feat_w = predict.size()
                batch_loss = F.cross_entropy(input=predict, target=targets,
                    weight=self.weight, ignore_index=self.ignore_index,
                    reduction='none')
                loss = 0.0
                for idx in range(batch_size):
                    single_loss = batch_loss[idx].view(feat_h * feat_w)
                    topk_loss, _ = single_loss.topk(self.top_k)
                    loss += topk_loss.sum() / self.top_k
                loss_fuse += loss / float(batch_size)
        else:
            batch_size, channels, feat_h, feat_w = predictions.size()
            batch_loss = F.cross_entropy(input=predictions, target=targets,
                weight=self.weight, ignore_index=self.ignore_index,
                reduction='none')
            loss = 0.0
            for idx in range(batch_size):
                single_loss = batch_loss[idx].view(feat_h * feat_w)
                topk_loss, _ = single_loss.topk(self.top_k)
                loss += topk_loss.sum() / self.top_k
            loss_fuse += loss / float(batch_size)
        log_pt = -loss_fuse
        return -(1.0 - torch.exp(log_pt)) ** self.gamma * self.alpha * log_pt


class SemanticEncodingLoss(nn.Module):

    def __init__(self, num_classes=19, ignore_label=250, weight=None, alpha
        =0.25):
        """
        Semantic Encoding Loss
        :param num_classes: <int> Number of classes
        :param ignore_label: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param alpha: <float> A manual rescaling weight given to Semantic Encoding Loss
        """
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_label = ignore_label

    def __unique_encode(self, msk_targets):
        """

        :param cls_targets: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :return:
        """
        batch_size, _, _ = msk_targets.size()
        target_mask = (msk_targets >= 0) * (msk_targets != self.ignore_label)
        cls_targets = [msk_targets[idx].masked_select(target_mask[idx]) for
            idx in np.arange(batch_size)]
        unique_cls = [torch.unique(label) for label in cls_targets]
        encode = torch.zeros(batch_size, self.num_classes, dtype=torch.
            float32, requires_grad=False)
        for idx in np.arange(batch_size):
            index = unique_cls[idx].long()
            encode[idx].index_fill_(dim=0, index=index, value=1.0)
        return encode

    def forward(self, predictions, targets):
        """
        
        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return:
        """
        enc_targets = self.__unique_encode(targets)
        se_loss = F.binary_cross_entropy_with_logits(predictions,
            enc_targets, weight=self.weight, reduction='elementwise_mean')
        return self.alpha * se_loss


class DiceLoss2D(nn.Module):

    def __init__(self, weight=None, ignore_index=-100):
        """
        Dice Loss for Semantic Segmentation

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(DiceLoss2D, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1e-06
        loss_fuse = 0.0
        if isinstance(predictions, tuple):
            for predict in predictions:
                predict = F.softmax(predict, dim=1)
                encoded_target = predict.detach() * 0
                mask = None
                if self.ignore_index is not None:
                    mask = targets == self.ignore_index
                    targets = targets.clone()
                    targets[mask] = 0
                    encoded_target.scatter_(dim=1, index=targets.unsqueeze(
                        dim=1), value=1.0)
                    mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
                    encoded_target[mask] = 0
                else:
                    encoded_target.scatter_(dim=1, index=targets.unsqueeze(
                        dim=1), value=1.0)
                if self.weight is None:
                    self.weight = 1.0
                intersection = predictions * encoded_target
                denominator = predictions + encoded_target
                if self.ignore_index is not None:
                    denominator[mask] = 0
                numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1
                    ) + smooth
                denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1
                    ) + smooth
                loss_per_channel = self.weight * (1.0 - numerator / denominator
                    )
                loss_fuse = loss_per_channel.sum() / predictions.size(1)
        else:
            predict = F.softmax(predictions, dim=1)
            encoded_target = predict.detach() * 0
            mask = None
            if self.ignore_index is not None:
                mask = targets == self.ignore_index
                targets = targets.clone()
                targets[mask] = 0
                encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=
                    1), value=1.0)
                mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=
                    1), value=1.0)
            if self.weight is None:
                self.weight = 1.0
            intersection = predictions * encoded_target
            denominator = predictions + encoded_target
            if self.ignore_index is not None:
                denominator[mask] = 0
            numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1
                ) + smooth
            denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
            loss_per_channel = self.weight * (1.0 - numerator / denominator)
            loss_fuse = loss_per_channel.sum() / predictions.size(1)
        return loss_fuse


class SoftJaccardLoss2D(nn.Module):

    def __init__(self, weight=None, ignore_index=-100):
        """
        Soft-Jaccard Loss for Semantic Segmentation

        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(SoftJaccardLoss2D, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1.0
        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
        if self.weight is None:
            self.weight = 1.0
        intersection = predictions * encoded_target
        denominator = predictions + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1)
        denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1)
        loss_per_channel = self.weight * (1.0 - (numerator + smooth) / (
            denominator - numerator + smooth))
        return loss_per_channel.sum() / predictions.size(1)


class TverskyLoss2D(nn.Module):

    def __init__(self, alpha=0.4, beta=0.6, weight=None, ignore_index=-100):
        """
        Tversky Loss for Semantic Segmentation

        :param alpha: <int> Parameter to control precision and recall
        :param beta: <int> Parameter to control precision and recall
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(TverskyLoss2D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        smooth = 1.0
        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
        if self.weight is None:
            self.weight = 1.0
        intersection = predictions * encoded_target
        numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
        ones = torch.ones_like(predictions)
        item1 = predictions * (ones - encoded_target)
        item2 = (ones - predictions) * encoded_target
        denominator = numerator + self.alpha * item1.sum(dim=0).sum(dim=1).sum(
            dim=1) + self.beta * item2.sum(dim=0).sum(dim=1).sum(dim=1)
        if self.ignore_index is not None:
            denominator[mask] = 0
        loss_per_channel = self.weight * (1.0 - numerator / (denominator -
            numerator))
        return loss_per_channel.sum() / predictions.size(1)


class AsymmetricSimilarityLoss2D(nn.Module):

    def __init__(self, beta=0.6, weight=None, ignore_index=-100):
        """
        Tversky Loss for Semantic Segmentation

        :param beta: <int> Parameter to control precision and recall
        :param weight: <torch.Tensor, optional> A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C, where C = number of classes.
        :param ignore_index: <int, optional> Specifies a target value that is ignored and does not
                                             contribute to the input gradient.
        """
        super(AsymmetricSimilarityLoss2D, self).__init__()
        self.beta = beta
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """

        :param predictions: <torch.FloatTensor> Network Predictions of size [N, C, H, W], where C = number of classes
        :param targets: <torch.LongTensor> Ground Truth label of size [N, H, W]
        :return: <torch.Tensor> loss
        """
        eps = 1e-08
        beta = self.beta ** 2
        predictions = F.softmax(predictions, dim=1)
        encoded_target = predictions.detach() * 0
        mask = None
        if self.ignore_index is not None:
            mask = targets == self.ignore_index
            targets = targets.clone()
            targets[mask] = 0
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
            mask = mask.unsqueeze(dim=1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(dim=1, index=targets.unsqueeze(dim=1),
                value=1.0)
        if self.weight is None:
            self.weight = 1.0
        intersection = predictions * encoded_target
        numerator = (1.0 + beta) * intersection.sum(dim=0).sum(dim=1).sum(dim=1
            )
        ones = torch.ones_like(predictions)
        item1 = predictions * (ones - encoded_target)
        item2 = (ones - predictions) * encoded_target
        denominator = numerator + beta * item1.sum(dim=0).sum(dim=1).sum(dim=1
            ) + item2.sum(dim=0).sum(dim=1).sum(dim=1) + eps
        if self.ignore_index is not None:
            denominator[mask] = 0
        loss_per_channel = self.weight * (1.0 - numerator / (denominator -
            numerator))
        return loss_per_channel.sum() / predictions.size(1)


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)
