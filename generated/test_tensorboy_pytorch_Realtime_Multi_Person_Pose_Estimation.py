import sys
_module = sys.modules[__name__]
del sys
picture_demo = _module
web_demo = _module
evaluate = _module
coco_eval = _module
evaluation = _module
experiments = _module
lib = _module
config = _module
default = _module
datasets = _module
_init_paths = _module
coco = _module
heatmap = _module
paf = _module
preprocessing = _module
test_dataloader = _module
transforms = _module
utils = _module
network = _module
atrous_model = _module
atrous_model_share_stages = _module
atrouspose = _module
im_transform = _module
openpose = _module
post = _module
rtpose_hourglass = _module
rtpose_mobilenetV2 = _module
rtpose_shufflenetV2 = _module
rtpose_vgg = _module
pafprocess = _module
setup = _module
common = _module
paf_to_pose = _module
train_SH = _module
train_ShuffleNetV2 = _module
train_VGG19 = _module

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


import re


import math


import scipy


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from collections import OrderedDict


from scipy.ndimage.morphology import generate_binary_structure


from scipy.ndimage.filters import gaussian_filter


from scipy.ndimage.filters import maximum_filter


from torch import load


from torch.nn import functional as F


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data as data


import torch.utils.model_zoo as model_zoo


from torch.nn import init


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, bn, **kwargs):
        super(BasicConv2d, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if self.bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, have_bn, have_bias):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1, bn=
            have_bn, bias=have_bias)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1, bn=
            have_bn, bias=have_bias)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2, bn
            =have_bn, bias=have_bias)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1,
            bn=have_bn, bias=have_bias)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1,
            bn=have_bn, bias=have_bias)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1,
            bn=have_bn, bias=have_bias)
        self.branch_pool = BasicConv2d(in_channels, pool_features,
            kernel_size=1, bn=have_bn, bias=have_bias)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class dilation_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=
        'same_padding', dilation=1):
        super(dilation_layer, self).__init__()
        if padding == 'same_padding':
            padding = (kernel_size - 1) / 2 * dilation
        self.Dconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding,
            dilation=dilation)
        self.Drelu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Dconv(x)
        x = self.Drelu(x)
        return x


class stage_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(stage_block, self).__init__()
        self.Dconv_1 = dilation_layer(in_channels, out_channels=64)
        self.Dconv_2 = dilation_layer(in_channels=64, out_channels=64)
        self.Dconv_3 = dilation_layer(in_channels=64, out_channels=64,
            dilation=2)
        self.Dconv_4 = dilation_layer(in_channels=64, out_channels=32,
            dilation=4)
        self.Dconv_5 = dilation_layer(in_channels=32, out_channels=32,
            dilation=8)
        self.Mconv_6 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=1, padding=0)
        self.Mrelu_6 = nn.ReLU(inplace=True)
        self.Mconv_7 = nn.Conv2d(in_channels=128, out_channels=out_channels,
            kernel_size=1, padding=0)

    def forward(self, x):
        x_1 = self.Dconv_1(x)
        x_2 = self.Dconv_2(x_1)
        x_3 = self.Dconv_3(x_2)
        x_4 = self.Dconv_4(x_3)
        x_5 = self.Dconv_5(x_4)
        x_cat = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x_out = self.Mconv_6(x_cat)
        x_out = self.Mrelu_6(x_out)
        x_pred = self.Mconv_7(x_out)
        return x_pred


class feature_extractor(nn.Module):

    def __init__(self, have_bn, have_bias):
        super(feature_extractor, self).__init__()
        None
        self.conv1_3x3_s2 = BasicConv2d(3, 32, kernel_size=3, stride=2,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv2_3x3_s1 = BasicConv2d(32, 32, kernel_size=3, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv3_3x3_s1 = BasicConv2d(32, 64, kernel_size=3, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv4_3x3_reduce = BasicConv2d(64, 80, kernel_size=1, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv4_3x3 = BasicConv2d(80, 192, kernel_size=3, bn=have_bn,
            bias=have_bias)
        self.inception_a1 = InceptionA(192, pool_features=32, have_bn=
            have_bn, have_bias=have_bias)
        self.inception_a2 = InceptionA(256, pool_features=64, have_bn=
            have_bn, have_bias=have_bias)

    def forward(self, x):
        x = self.conv1_3x3_s2(x)
        x = self.conv2_3x3_s1(x)
        x = self.conv3_3x3_s1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.conv4_3x3_reduce(x)
        x = self.conv4_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        return x


def build_names():
    names = []
    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


class Atrous_model(nn.Module):

    def __init__(self, stages=5, have_bn=True, have_bias=False):
        super(Atrous_model, self).__init__()
        self.stages = stages
        self.feature_extractor = feature_extractor(have_bn=have_bn,
            have_bias=have_bias)
        self.stage_0 = nn.Sequential(nn.Conv2d(in_channels=288,
            out_channels=256, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.Conv2d(in_channels=256, out_channels=128, kernel_size
            =3, padding=1), nn.ReLU(inplace=True))
        for i in range(stages):
            setattr(self, 'PAF_stage{}'.format(i + 2), stage_block(
                in_channels=128, out_channels=38) if i == 0 else
                stage_block(in_channels=185, out_channels=38))
            setattr(self, 'heatmap_stage{}'.format(i + 2), stage_block(
                in_channels=128, out_channels=19) if i == 0 else
                stage_block(in_channels=185, out_channels=19))
        self.init_weight()

    def forward(self, x):
        saved_for_loss = []
        x_in = self.feature_extractor(x)
        x_in_0 = self.stage_0(x_in)
        x_in = x_in_0
        for i in range(self.stages):
            x_PAF_pred = getattr(self, 'PAF_stage{}'.format(i + 2))(x_in)
            x_heatmap_pred = getattr(self, 'heatmap_stage{}'.format(i + 2))(
                x_in)
            saved_for_loss.append(x_PAF_pred)
            saved_for_loss.append(x_heatmap_pred)
            if i != self.stages - 1:
                x_in = torch.cat([x_PAF_pred, x_heatmap_pred, x_in_0], 1)
        return [x_PAF_pred, x_heatmap_pred], saved_for_loss

    def init_weight(self):
        for m in self.modules():
            if m in self.feature_extractor.modules():
                continue
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant(m.bias, 0.0)

    @staticmethod
    def build_loss(saved_for_loss, heat_temp, heat_weight, vec_temp,
        vec_weight, batch_size, gpus):
        names = build_names()
        saved_for_log = OrderedDict()
        criterion = nn.MSELoss(size_average=False)
        total_loss = 0
        div = 2 * batch_size
        for j in range(5):
            pred1 = saved_for_loss[2 * j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_weight * heat_temp
            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            loss1 = criterion(pred1, gt1) / div
            loss2 = criterion(pred2, gt2) / div
            total_loss += loss1
            total_loss += loss2
            saved_for_log[names[2 * j]] = loss1.data[0]
            saved_for_log[names[2 * j + 1]] = loss2.data[0]
        return total_loss, saved_for_log


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, bn, **kwargs):
        super(BasicConv2d, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if self.bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, have_bn, have_bias):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1, bn=
            have_bn, bias=have_bias)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1, bn=
            have_bn, bias=have_bias)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2, bn
            =have_bn, bias=have_bias)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1,
            bn=have_bn, bias=have_bias)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1,
            bn=have_bn, bias=have_bias)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1,
            bn=have_bn, bias=have_bias)
        self.branch_pool = BasicConv2d(in_channels, pool_features,
            kernel_size=1, bn=have_bn, bias=have_bias)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class dilation_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=
        'same_padding', dilation=1):
        super(dilation_layer, self).__init__()
        if padding == 'same_padding':
            padding = int((kernel_size - 1) / 2 * dilation)
        self.Dconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding,
            dilation=dilation)
        self.Drelu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Dconv(x)
        x = self.Drelu(x)
        return x


class stage_block(nn.Module):
    """This class makes sure the paf and heatmap branch out in every stage"""

    def __init__(self, in_channels, out_channels):
        super(stage_block, self).__init__()
        self.Dconv_1 = dilation_layer(in_channels, out_channels=64)
        self.Dconv_2 = dilation_layer(in_channels=64, out_channels=64)
        self.Dconv_3 = dilation_layer(in_channels=64, out_channels=64,
            dilation=2)
        self.Dconv_4 = dilation_layer(in_channels=64, out_channels=32,
            dilation=4)
        self.Dconv_5 = dilation_layer(in_channels=32, out_channels=32,
            dilation=8)
        self.Mconv_6 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=1, padding=0)
        self.Mrelu_6 = nn.ReLU(inplace=True)
        self.paf = nn.Conv2d(in_channels=128, out_channels=14, kernel_size=
            1, padding=0)
        self.heatmap = nn.Conv2d(in_channels=128, out_channels=9,
            kernel_size=1, padding=0)

    def forward(self, x):
        x_1 = self.Dconv_1(x)
        x_2 = self.Dconv_2(x_1)
        x_3 = self.Dconv_3(x_2)
        x_4 = self.Dconv_4(x_3)
        x_5 = self.Dconv_5(x_4)
        x_cat = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x_out = self.Mconv_6(x_cat)
        x_out = self.Mrelu_6(x_out)
        paf = self.paf(x_out)
        heatmap = self.heatmap(x_out)
        return paf, heatmap


class feature_extractor(nn.Module):

    def __init__(self, have_bn, have_bias):
        super(feature_extractor, self).__init__()
        None
        """
        annotations in inception_v3
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        """
        self.conv1_3x3_s2 = BasicConv2d(3, 32, kernel_size=3, stride=2,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv2_3x3_s1 = BasicConv2d(32, 32, kernel_size=3, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv3_3x3_s1 = BasicConv2d(32, 64, kernel_size=3, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv4_3x3_reduce = BasicConv2d(64, 80, kernel_size=1, stride=1,
            padding=1, bn=have_bn, bias=have_bias)
        self.conv4_3x3 = BasicConv2d(80, 192, kernel_size=3, bn=have_bn,
            bias=have_bias)
        self.inception_a1 = InceptionA(192, pool_features=32, have_bn=
            have_bn, have_bias=have_bias)
        self.inception_a2 = InceptionA(256, pool_features=64, have_bn=
            have_bn, have_bias=have_bias)

    def forward(self, x):
        x = self.conv1_3x3_s2(x)
        x = self.conv2_3x3_s1(x)
        x = self.conv3_3x3_s1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.conv4_3x3_reduce(x)
        x = self.conv4_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        return x


class Ying_model(nn.Module):

    def __init__(self, stages=5, have_bn=True, have_bias=False):
        super(Ying_model, self).__init__()
        self.stages = stages
        self.feature_extractor = feature_extractor(have_bn=have_bn,
            have_bias=have_bias)
        self.stage_0 = nn.Sequential(nn.Conv2d(in_channels=288,
            out_channels=256, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.Conv2d(in_channels=256, out_channels=128, kernel_size
            =3, padding=1), nn.ReLU(inplace=True))
        for i in range(stages):
            setattr(self, 'stage{}'.format(i + 2), stage_block(in_channels=
                128, out_channels=14) if i == 0 else stage_block(
                in_channels=151, out_channels=14))
        self._initialize_weights_norm()

    def forward(self, x):
        saved_for_loss = []
        paf_ret, heat_ret = [], []
        x_in = self.feature_extractor(x)
        x_in_0 = self.stage_0(x_in)
        x_in = x_in_0
        for i in range(self.stages):
            x_PAF_pred, x_heatmap_pred = getattr(self, 'stage{}'.format(i + 2)
                )(x_in)
            paf_ret.append(x_PAF_pred)
            heat_ret.append(x_heatmap_pred)
            if i != self.stages - 1:
                x_in = torch.cat([x_PAF_pred, x_heatmap_pred, x_in_0], 1)
        saved_for_loss.append(paf_ret)
        saved_for_loss.append(heat_ret)
        return [(paf_ret[-2], heat_ret[-2]), (paf_ret[-1], heat_ret[-1])
            ], saved_for_loss

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant(m.bias, 0.001)

    @staticmethod
    def build_loss(saved_for_loss, heat_temp, heat_weight, vec_temp,
        vec_weight, batch_size, gpus):
        names = build_names()
        saved_for_log = OrderedDict()
        criterion = nn.MSELoss(size_average=True)
        total_loss = 0
        for j in range(5):
            pred1 = saved_for_loss[0][j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            pred2 = saved_for_loss[1][j] * heat_weight
            gt2 = heat_weight * heat_temp
            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            loss1 = criterion(pred1, gt1)
            loss2 = criterion(pred2, gt2)
            total_loss += loss1
            total_loss += loss2
            saved_for_log[names[2 * j]] = loss1.data[0]
            saved_for_log[names[2 * j + 1]] = loss2.data[0]
        return total_loss, saved_for_log


class ASPP_ASP(nn.Module):

    def __init__(self, in_, out_=16):
        super(ASPP_ASP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1,
            padding=1, dilation=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)
        self.conv_3x3_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1,
            padding=4, dilation=4)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(128)
        self.conv_3x3_2 = nn.Conv2d(in_, 128, kernel_size=3, stride=1,
            padding=8, dilation=8)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(128)
        self.conv_3x3_3 = nn.Conv2d(in_, 128, kernel_size=3, stride=1,
            padding=16, dilation=16)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(128)
        self.bn_out = nn.BatchNorm2d(512)

    def forward(self, feature_map):
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        add1 = out_1x1
        add2 = add1 + out_3x3_1
        add3 = add2 + out_3x3_2
        add4 = add3 + out_3x3_3
        out = F.relu(self.bn_out(torch.cat([add1, add2, add3, add4], 1)))
        return out


class Upsample(nn.Module):

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.
            mode, align_corners=True)


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True,
    dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class AtrousPose(nn.Module):

    def __init__(self, paf_out_channels=38, heat_out_channels=19):
        super(AtrousPose, self).__init__()
        """
        mobile net
        """
        resnet = models.resnet50(pretrained=True)
        self.layer3 = resnet.layer3
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])
        self.smooth_ups2 = self._lateral(1024, 2)
        self.smooth_ups3 = self._lateral(512, 1)
        self.aspp1 = ASPP_ASP(512, out_=16)
        self.h1 = nn.Sequential(conv(512, 512, kernel_size=3, padding=1),
            conv(512, 512, kernel_size=3, padding=1), conv(512, 512,
            kernel_size=3, padding=1), conv(512, 512, kernel_size=1,
            padding=0, bn=False), conv(512, heat_out_channels, kernel_size=
            1, padding=0, bn=False, relu=False))
        self.p1 = nn.Sequential(conv(512, 512, kernel_size=3, padding=1),
            conv(512, 512, kernel_size=3, padding=1), conv(512, 512,
            kernel_size=3, padding=1), conv(512, 512, kernel_size=1,
            padding=0, bn=False), conv(512, paf_out_channels, kernel_size=1,
            padding=0, bn=False, relu=False))

    def _lateral(self, input_size, factor):
        layers = []
        layers.append(nn.Conv2d(input_size, 256, kernel_size=1, stride=1,
            bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Upsample(scale_factor=factor, mode='bilinear'))
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_map = self.resnet(x)
        _16x = self.layer3(feature_map)
        _16x = self.smooth_ups2(_16x)
        feature_map = self.smooth_ups3(feature_map)
        cat_feat = F.relu(torch.cat([feature_map, _16x], 1))
        out = self.aspp1(cat_feat)
        heatmap = self.h1(out)
        paf = self.p1(out)
        return paf, heatmap


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(ConvBlock, self).__init__()
        self.Mconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.MPrelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.Mconv(x)
        x = self.MPrelu(x)
        return x


class StageBlock(nn.Module):
    """ L1/L2 StageBlock Template """

    def __init__(self, in_channels, inner_channels, innerout_channels,
        out_channels):
        super(StageBlock, self).__init__()
        self.Mconv1_0 = ConvBlock(in_channels, inner_channels)
        self.Mconv1_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv1_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv2_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv3_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv4_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv5_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv6 = ConvBlock(inner_channels * 3, innerout_channels,
            kernel_size=1, stride=1, padding=0)
        self.Mconv7 = nn.Conv2d(in_channels=innerout_channels, out_channels
            =out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1_1 = self.Mconv1_0(x)
        out2_1 = self.Mconv1_1(out1_1)
        out3_1 = self.Mconv1_2(out2_1)
        x_cat_1 = torch.cat([out1_1, out2_1, out3_1], 1)
        out1_2 = self.Mconv2_0(x_cat_1)
        out2_2 = self.Mconv2_1(out1_2)
        out3_2 = self.Mconv2_2(out2_2)
        x_cat_2 = torch.cat([out1_2, out2_2, out3_2], 1)
        out1_3 = self.Mconv3_0(x_cat_2)
        out2_3 = self.Mconv3_1(out1_3)
        out3_3 = self.Mconv3_2(out2_3)
        x_cat_3 = torch.cat([out1_3, out2_3, out3_3], 1)
        out1_4 = self.Mconv4_0(x_cat_3)
        out2_4 = self.Mconv4_1(out1_4)
        out3_4 = self.Mconv4_2(out2_4)
        x_cat_4 = torch.cat([out1_4, out2_4, out3_4], 1)
        out1_5 = self.Mconv5_0(x_cat_4)
        out2_5 = self.Mconv5_1(out1_5)
        out3_5 = self.Mconv5_2(out2_5)
        x_cat_5 = torch.cat([out1_5, out2_5, out3_5], 1)
        out_6 = self.Mconv6(x_cat_5)
        stage_output = self.Mconv7(out_6)
        return stage_output


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                    padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                    kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class OpenPose_Model(nn.Module):

    def __init__(self, l2_stages=4, l1_stages=2, paf_out_channels=14,
        heat_out_channels=9):
        """
        :param feature_extractor:
        :param l2_stages:
        :param l1_stages:
        :param paf_out_channels:
        :param heat_out_channels:
        :param stage_input_mode: either 'from_first_stage' (original)
        or 'from_previous_stage' (i.e. take x_out from previous stage as
        input to next stage).
        """
        super(OpenPose_Model, self).__init__()
        self.stages = [0, 1]
        self.feature_extractor = make_vgg19_block()
        L2_IN_CHS = [128]
        L2_INNER_CHS = [96]
        L2_INNEROUT_CHS = [256]
        L2_OUT_CHS = [paf_out_channels]
        for _ in range(l2_stages - 1):
            L2_IN_CHS.append(128 + paf_out_channels)
            L2_INNER_CHS.append(128)
            L2_INNEROUT_CHS.append(512)
            L2_OUT_CHS.append(paf_out_channels)
        self.l2_stages = nn.ModuleList([StageBlock(in_channels=L2_IN_CHS[i],
            inner_channels=L2_INNER_CHS[i], innerout_channels=
            L2_INNEROUT_CHS[i], out_channels=L2_OUT_CHS[i]) for i in range(
            len(L2_IN_CHS))])
        L1_IN_CHS = [128 + paf_out_channels]
        L1_INNER_CHS = [96]
        L1_INNEROUT_CHS = [256]
        L1_OUT_CHS = [heat_out_channels]
        for _ in range(l1_stages - 1):
            L1_IN_CHS.append(128 + paf_out_channels + heat_out_channels)
            L1_INNER_CHS.append(128)
            L1_INNEROUT_CHS.append(512)
            L1_OUT_CHS.append(heat_out_channels)
        self.l1_stages = nn.ModuleList([StageBlock(in_channels=L1_IN_CHS[i],
            inner_channels=L1_INNER_CHS[i], innerout_channels=
            L1_INNEROUT_CHS[i], out_channels=L1_OUT_CHS[i]) for i in range(
            len(L1_IN_CHS))])
        self._initialize_weights_norm()

    def forward(self, x):
        saved_for_loss = []
        features = self.feature_extractor(x)
        paf_ret, heat_ret = [], []
        x_in = features
        for l2_stage in self.l2_stages:
            paf_pred = l2_stage(x_in)
            x_in = torch.cat([features, paf_pred], 1)
            paf_ret.append(paf_pred)
        for l1_stage in self.l1_stages:
            heat_pred = l1_stage(x_in)
            x_in = torch.cat([features, heat_pred, paf_pred], 1)
            heat_ret.append(heat_pred)
        saved_for_loss.append(paf_ret)
        saved_for_loss.append(heat_ret)
        return [(paf_ret[-2], heat_ret[-2]), (paf_ret[-1], heat_ret[-1])
            ], saved_for_loss

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.001)
            elif isinstance(m, nn.PReLU):
                init.normal_(m.weight, std=0.01)

    def init_w_pretrained_weights(self, pkl_weights=
        '/home/tomas/Desktop/AIFI/internal-repos/aifi-pose/network/weights/openpose/openpose.pkl'
        ):
        with open(pkl_weights, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')
        conv_idxs = [i for i, d in enumerate(weights) if 'conv' in d['name'
            ] and 'split' not in d['name'] and 'concat' not in d['name']]
        prelu_idxs = [i for i, d in enumerate(weights) if 'prelu' in d[
            'name'] and 'split' not in d['name'] and 'concat' not in d['name']]
        conv_idxs = iter(conv_idxs)
        prelu_idxs = iter(prelu_idxs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                idx = next(conv_idxs)
                m.weight = torch.nn.Parameter(torch.Tensor(weights[idx][
                    'weights'][0]))
                m.bias = torch.nn.Parameter(torch.Tensor(weights[idx][
                    'weights'][1]))
            elif isinstance(m, nn.PReLU):
                idx = next(prelu_idxs)
                m.weight = torch.nn.Parameter(torch.Tensor(weights[idx][
                    'weights'][0]))


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Hourglass(nn.Module):

    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


NUM_JOINTS = 18


NUM_LIMBS = 38


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block, num_stacks=2, num_blocks=4, paf_classes=
        NUM_LIMBS * 2, ht_classes=NUM_JOINTS + 1):
        super(HourglassNet, self).__init__()
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        ch = self.num_feats * block.expansion
        hg, res, fc, score_paf, score_ht, fc_, paf_score_, ht_score_ = [], [
            ], [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score_paf.append(nn.Conv2d(ch, paf_classes, kernel_size=1, bias
                =True))
            score_ht.append(nn.Conv2d(ch, ht_classes, kernel_size=1, bias=True)
                )
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                paf_score_.append(nn.Conv2d(paf_classes, ch, kernel_size=1,
                    bias=True))
                ht_score_.append(nn.Conv2d(ht_classes, ch, kernel_size=1,
                    bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score_ht = nn.ModuleList(score_ht)
        self.score_paf = nn.ModuleList(score_paf)
        self.fc_ = nn.ModuleList(fc_)
        self.paf_score_ = nn.ModuleList(paf_score_)
        self.ht_score_ = nn.ModuleList(ht_score_)
        self._initialize_weights_norm()

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        saved_for_loss = []
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score_paf = self.score_paf[i](y)
            score_ht = self.score_ht[i](y)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                paf_score_ = self.paf_score_[i](score_paf)
                ht_score_ = self.ht_score_[i](score_ht)
                x = x + fc_ + paf_score_ + ht_score_
        saved_for_loss.append(score_paf)
        saved_for_loss.append(score_ht)
        return (score_paf, score_ht), saved_for_loss

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.
            last_channel, n_class))
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


class BasicBlock(nn.Module):

    def __init__(self, name, in_channels, out_channels, stride, downsample,
        dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        self.downsample = downsample
        channels = out_channels // 2
        if not self.downsample and self.stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(slim.conv_bn_relu(name + '/conv1',
                channels, channels, 1), slim.conv_bn(name + '/conv2',
                channels, channels, 3, stride=stride, dilation=dilation,
                padding=dilation, groups=channels), slim.conv_bn_relu(name +
                '/conv3', channels, channels, 1))
        else:
            self.conv = nn.Sequential(slim.conv_bn_relu(name + '/conv1',
                in_channels, channels, 1), slim.conv_bn(name + '/conv2',
                channels, channels, 3, stride=stride, dilation=dilation,
                padding=dilation, groups=channels), slim.conv_bn_relu(name +
                '/conv3', channels, channels, 1))
            self.conv0 = nn.Sequential(slim.conv_bn(name + '/conv4',
                in_channels, in_channels, 3, stride=stride, dilation=
                dilation, padding=dilation, groups=in_channels), slim.
                conv_bn_relu(name + '/conv5', in_channels, channels, 1))
        self.shuffle = slim.channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if not self.downsample:
            x1 = x[:, :x.shape[1] // 2, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)

    def generate_caffe_prototxt(self, caffe_net, layer):
        if self.stride == 1:
            layer_x1, layer_x2 = L.Slice(layer, ntop=2, axis=1, slice_point
                =[self.in_channels // 2])
            caffe_net[self.g_name + '/slice1'] = layer_x1
            caffe_net[self.g_name + '/slice2'] = layer_x2
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net,
                layer_x2)
        else:
            layer_x1 = slim.generate_caffe_prototxt(self.conv0, caffe_net,
                layer)
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer
                )
        layer = L.Concat(layer_x1, layer_x2, axis=1)
        caffe_net[self.g_name + '/concat'] = layer
        layer = slim.generate_caffe_prototxt(self.shuffle, caffe_net, layer)
        return layer


_global_config['image_hw'] = 4


class Network(nn.Module):

    def __init__(self, width_multiplier):
        super(Network, self).__init__()
        width_config = {(0.25): (24, 48, 96, 512), (0.33): (32, 64, 128, 
            512), (0.5): (48, 96, 192, 1024), (1.0): (116, 232, 464, 1024),
            (1.5): (176, 352, 704, 1024), (2.0): (244, 488, 976, 2048)}
        width_config = width_config[width_multiplier]
        in_channels = 24
        self.network_config = [g_name('data/bn', nn.BatchNorm2d(3)), slim.
            conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1), g_name(
            'stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)), (
            width_config[0], 2, 1, 4, 'b'), (width_config[1], 1, 1, 8, 'b'),
            (width_config[2], 1, 1, 4, 'b'), slim.conv_bn_relu('conv5',
            width_config[2], width_config[3], 1)]
        self.paf = nn.Conv2d(width_config[3], 38, 1)
        self.heatmap = nn.Conv2d(width_config[3], 19, 1)
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            if stride == 2:
                downsample = True
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels,
                out_channels, stride, downsample, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1),
                    out_channels, out_channels, 1, False, dilation))
            self.network += [nn.Sequential(*blocks)]
            in_channels = out_channels
        self.network = nn.Sequential(*self.network)
        for name, m in self.named_modules():
            if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d,
                nn.Conv2d])):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def trainable_parameters(self):
        parameters = [{'params': self.cls_head_list.parameters(), 'lr_mult':
            1.0}, {'params': self.loc_head_list.parameters(), 'lr_mult': 1.0}]
        for i in range(len(self.network)):
            lr_mult = 0.1 if i in (0, 1, 2, 3, 4, 5) else 1
            parameters.append({'params': self.network[i].parameters(),
                'lr_mult': lr_mult})
        return parameters

    def forward(self, x):
        x = self.network(x)
        PAF = self.paf(x)
        HEAT = self.heatmap(x)
        return [PAF, HEAT], [PAF, HEAT]

    def generate_caffe_prototxt(self, caffe_net, layer):
        data_layer = layer
        network = slim.generate_caffe_prototxt(self.network, caffe_net,
            data_layer)
        return network

    def convert_to_caffe(self, name):
        caffe_net = caffe.NetSpec()
        layer = L.Input(shape=dict(dim=[1, 3, args.image_hw, args.image_hw]))
        caffe_net.tops['data'] = layer
        slim.generate_caffe_prototxt(self, caffe_net, layer)
        None
        with open(name + '.prototxt', 'wb') as f:
            f.write(str(caffe_net.to_proto()).encode())
        caffe_net = caffe.Net(name + '.prototxt', caffe.TEST)
        slim.convert_pytorch_to_caffe(self, caffe_net)
        caffe_net.save(name + '.caffemodel')


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_tensorboy_pytorch_Realtime_Multi_Person_Pose_Estimation(_paritybench_base):
    pass
    def test_000(self):
        self._check(ASPP_ASP(*[], **{'in_': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BasicConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'bn': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ConvBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(InceptionA(*[], **{'in_channels': 4, 'pool_features': 4, 'have_bn': 4, 'have_bias': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(MobileNetV2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_006(self):
        self._check(StageBlock(*[], **{'in_channels': 4, 'inner_channels': 4, 'innerout_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(Ying_model(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_008(self):
        self._check(dilation_layer(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(feature_extractor(*[], **{'have_bn': 4, 'have_bias': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_010(self):
        self._check(stage_block(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

