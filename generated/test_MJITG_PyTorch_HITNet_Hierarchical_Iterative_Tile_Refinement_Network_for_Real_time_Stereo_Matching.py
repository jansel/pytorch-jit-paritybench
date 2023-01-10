import sys
_module = sys.modules[__name__]
del sys
datasets = _module
data_io = _module
kitti_dataset = _module
initialization_loss = _module
propagation_loss = _module
total_loss = _module
main = _module
FE = _module
HITNet = _module
initialization = _module
submodules = _module
tile_update = _module
tile_warping = _module
utils = _module
experiment = _module
metrics = _module
saver = _module
visualization = _module
write_pfm = _module

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


import random


from torch.utils.data import Dataset


import numpy as np


import torchvision.transforms.functional as photometric


import torch


import torch.nn.functional as F


import math


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import torchvision.utils as vutils


import time


from torch.utils.data import DataLoader


import copy


from torch import Tensor


from collections import OrderedDict


import torch.distributed as dist


from torch.autograd import Function


def BasicConv2d(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True, negative_slope=0.2))


def BasicTransposeConv2d(in_channels, out_channels, kernel_size, stride, pad, dilation):
    output_pad = stride + 2 * pad - kernel_size * dilation + dilation - 1
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, output_pad, dilation, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True, negative_slope=0.2))


class unetUp(nn.Module):

    def __init__(self, in_c1, in_c2, out_c):
        super(unetUp, self).__init__()
        self.up_conv1 = BasicTransposeConv2d(in_c1, in_c1 // 2, 2, 2, 0, 1)
        self.reduce_conv2 = BasicConv2d(in_c1 // 2 + in_c2, out_c, 1, 1, 0, 1)
        self.conv = nn.Sequential(BasicConv2d(out_c, out_c, 3, 1, 1, 1))

    def forward(self, inputs1, inputs2):
        layer1 = self.up_conv1(inputs1)
        layer2 = self.reduce_conv2(torch.cat([layer1, inputs2], 1))
        output = self.conv(layer2)
        return output


class feature_extraction_conv(nn.Module):
    """
    UNet for HITNet
    """

    def __init__(self, args):
        super(feature_extraction_conv, self).__init__()
        self.conv1x_0 = nn.Sequential(BasicConv2d(3, 16, 3, 1, 1, 1), BasicConv2d(16, 16, 3, 1, 1, 1))
        self.conv2x_0 = nn.Sequential(BasicConv2d(16, 16, 2, 2, 0, 1), BasicConv2d(16, 16, 3, 1, 1, 1), BasicConv2d(16, 16, 3, 1, 1, 1))
        self.conv4x_0 = nn.Sequential(BasicConv2d(16, 24, 2, 2, 0, 1), BasicConv2d(24, 24, 3, 1, 1, 1), BasicConv2d(24, 24, 3, 1, 1, 1))
        self.conv8x_0 = nn.Sequential(BasicConv2d(24, 24, 2, 2, 0, 1), BasicConv2d(24, 24, 3, 1, 1, 1), BasicConv2d(24, 24, 3, 1, 1, 1))
        self.conv16x_0 = nn.Sequential(BasicConv2d(24, 32, 2, 2, 0, 1), BasicConv2d(32, 32, 3, 1, 1, 1), BasicConv2d(32, 32, 3, 1, 1, 1))
        self.conv16_8x_0 = unetUp(32, 24, 24)
        self.conv8_4x_0 = unetUp(24, 24, 24)
        self.conv4_2x_0 = unetUp(24, 16, 16)
        self.conv2_1x_0 = unetUp(16, 16, 16)
        self.last_conv_1x = nn.Conv2d(16, 16, 1, 1, 0, 1, bias=False)
        self.last_conv_2x = nn.Conv2d(16, 16, 1, 1, 0, 1, bias=False)
        self.last_conv_4x = nn.Conv2d(24, 24, 1, 1, 0, 1, bias=False)
        self.last_conv_8x = nn.Conv2d(24, 24, 1, 1, 0, 1, bias=False)
        self.last_conv_16x = nn.Conv2d(32, 32, 1, 1, 0, 1, bias=False)

    def forward(self, x):
        layer1x_0 = self.conv1x_0(x)
        layer2x_0 = self.conv2x_0(layer1x_0)
        layer4x_0 = self.conv4x_0(layer2x_0)
        layer8x_0 = self.conv8x_0(layer4x_0)
        layer16x_0 = self.conv16x_0(layer8x_0)
        layer8x_1 = self.conv16_8x_0(layer16x_0, layer8x_0)
        layer4x_1 = self.conv8_4x_0(layer8x_1, layer4x_0)
        layer2x_1 = self.conv4_2x_0(layer4x_1, layer2x_0)
        layer1x_1 = self.conv2_1x_0(layer2x_1, layer1x_0)
        layer16x_1 = self.last_conv_16x(layer16x_0)
        layer8x_2 = self.last_conv_8x(layer8x_1)
        layer4x_2 = self.last_conv_4x(layer4x_1)
        layer2x_2 = self.last_conv_2x(layer2x_1)
        layer1x_2 = self.last_conv_1x(layer1x_1)
        return [layer16x_1, layer8x_2, layer4x_2, layer2x_2, layer1x_2]


class DispUpsampleBySlantedPlane(nn.Module):

    def __init__(self, upscale, ts=4):
        super(DispUpsampleBySlantedPlane, self).__init__()
        self.upscale = upscale
        self.center = (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)
        self.ts = ts

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * (self.upscale / self.ts)
        disp0 = []
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i - self.center) * tile_dx + (j - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)
        disp1 = self.DUC(disp0)
        return disp1


class ResBlock(nn.Module):
    """
    Residual Block without BN but with dilation
    """

    def __init__(self, inplanes, out_planes, hid_planes, add_relu=True):
        super(ResBlock, self).__init__()
        self.add_relu = add_relu
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, hid_planes, 3, 1, 1, 1), nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.conv2 = nn.Conv2d(hid_planes, out_planes, 3, 1, 1, 1)
        if add_relu:
            self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        if self.add_relu:
            out = self.relu(out)
        return out


class FinalTileUpdate(nn.Module):
    """
    Final Tile Update: only predicts disp
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num, slant_disp_up, args):
        super(FinalTileUpdate, self).__init__()
        self.disp_upsample = slant_disp_up
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)
        previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)
        previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)
        up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
        up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
        up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
        up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy, up_previous_tile_dscrpt], 1)
        guided_up_previous_tile_hypothesis = torch.cat([up_previous_tile_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = up_previous_tile_d + tile_hypothesis_update
        refined_hypothesis = F.relu(refined_hypothesis.clone())
        return refined_hypothesis


class BuildVolume2d(nn.Module):

    def __init__(self, maxdisp):
        super(BuildVolume2d, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, feat_l, feat_r):
        padded_feat_r = F.pad(feat_r, [self.maxdisp - 1, 0, 0, 0])
        cost = torch.zeros((feat_l.size()[0], self.maxdisp, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, self.maxdisp):
            if i > 0:
                cost[:, i, :, :] = torch.norm(feat_l[:, :, :, :] - padded_feat_r[:, :, :, self.maxdisp - 1 - i:-i:4], 1, 1)
            else:
                cost[:, i, :, :] = torch.norm(feat_l[:, :, :, :] - padded_feat_r[:, :, :, self.maxdisp - 1::4], 1, 1)
        return cost.contiguous()


class INIT(nn.Module):
    """
    Tile hypothesis initialization
    input: dual feature pyramid
    output: initial tile hypothesis pyramid
    """

    def __init__(self, args):
        super().__init__()
        self.maxdisp = args.maxdisp
        fea_c1x = args.fea_c[4]
        fea_c2x = args.fea_c[3]
        fea_c4x = args.fea_c[2]
        fea_c8x = args.fea_c[1]
        fea_c16x = args.fea_c[0]
        self.tile_conv1x = nn.Sequential(BasicConv2d(fea_c1x, fea_c1x, 4, 4, 0, 1), nn.Conv2d(fea_c1x, fea_c1x, 1, 1, 0, bias=False))
        self.tile_conv2x = nn.Sequential(BasicConv2d(fea_c2x, fea_c2x, 4, 4, 0, 1), nn.Conv2d(fea_c2x, fea_c2x, 1, 1, 0, bias=False))
        self.tile_conv4x = nn.Sequential(BasicConv2d(fea_c4x, fea_c4x, 4, 4, 0, 1), nn.Conv2d(fea_c4x, fea_c4x, 1, 1, 0, bias=False))
        self.tile_conv8x = nn.Sequential(BasicConv2d(fea_c8x, fea_c8x, 4, 4, 0, 1), nn.Conv2d(fea_c8x, fea_c8x, 1, 1, 0, bias=False))
        self.tile_conv16x = nn.Sequential(BasicConv2d(fea_c16x, fea_c16x, 4, 4, 0, 1), nn.Conv2d(fea_c16x, fea_c16x, 1, 1, 0, bias=False))
        self.tile_fea_dscrpt16x = BasicConv2d(fea_c16x + 1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt8x = BasicConv2d(fea_c8x + 1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt4x = BasicConv2d(fea_c4x + 1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt2x = BasicConv2d(fea_c2x + 1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt1x = BasicConv2d(fea_c1x + 1, 13, 1, 1, 0, 1)
        self._build_volume_2d16x = BuildVolume2d(self.maxdisp // 16)
        self._build_volume_2d8x = BuildVolume2d(self.maxdisp // 8)
        self._build_volume_2d4x = BuildVolume2d(self.maxdisp // 4)
        self._build_volume_2d2x = BuildVolume2d(self.maxdisp // 2)
        self._build_volume_2d1x = BuildVolume2d(self.maxdisp)

    def tile_features(self, fea_l, fea_r):
        right_fea_pad = [0, 3, 0, 0]
        tile_fea_l1x = self.tile_conv1x(fea_l[-1])
        padded_fea_r1x = F.pad(fea_r[-1], right_fea_pad)
        self.tile_conv1x[0][0].stride = 4, 1
        tile_fea_r1x = self.tile_conv1x(padded_fea_r1x)
        self.tile_conv1x[0][0].stride = 4, 4
        tile_fea_l2x = self.tile_conv2x(fea_l[-2])
        padded_fea_r2x = F.pad(fea_r[-2], right_fea_pad)
        self.tile_conv2x[0][0].stride = 4, 1
        tile_fea_r2x = self.tile_conv2x(padded_fea_r2x)
        self.tile_conv2x[0][0].stride = 4, 4
        tile_fea_l4x = self.tile_conv4x(fea_l[-3])
        padded_fea_r4x = F.pad(fea_r[-3], right_fea_pad)
        self.tile_conv4x[0][0].stride = 4, 1
        tile_fea_r4x = self.tile_conv4x(padded_fea_r4x)
        self.tile_conv4x[0][0].stride = 4, 4
        tile_fea_l8x = self.tile_conv8x(fea_l[-4])
        padded_fea_r8x = F.pad(fea_r[-4], right_fea_pad)
        self.tile_conv8x[0][0].stride = 4, 1
        tile_fea_r8x = self.tile_conv8x(padded_fea_r8x)
        self.tile_conv8x[0][0].stride = 4, 4
        tile_fea_l16x = self.tile_conv16x(fea_l[-5])
        padded_fea_r16x = F.pad(fea_r[-5], right_fea_pad)
        self.tile_conv16x[0][0].stride = 4, 1
        tile_fea_r16x = self.tile_conv16x(padded_fea_r16x)
        self.tile_conv16x[0][0].stride = 4, 4
        return [[tile_fea_l16x, tile_fea_r16x], [tile_fea_l8x, tile_fea_r8x], [tile_fea_l4x, tile_fea_r4x], [tile_fea_l2x, tile_fea_r2x], [tile_fea_l1x, tile_fea_r1x]]

    def tile_hypothesis_pyramid(self, tile_feature_pyramid):
        init_tile_cost16x = self._build_volume_2d16x(tile_feature_pyramid[0][0], tile_feature_pyramid[0][1])
        init_tile_cost8x = self._build_volume_2d8x(tile_feature_pyramid[1][0], tile_feature_pyramid[1][1])
        init_tile_cost4x = self._build_volume_2d4x(tile_feature_pyramid[2][0], tile_feature_pyramid[2][1])
        init_tile_cost2x = self._build_volume_2d2x(tile_feature_pyramid[3][0], tile_feature_pyramid[3][1])
        init_tile_cost1x = self._build_volume_2d1x(tile_feature_pyramid[4][0], tile_feature_pyramid[4][1])
        min_tile_cost16x, min_tile_disp16x = torch.min(init_tile_cost16x, 1)
        min_tile_cost8x, min_tile_disp8x = torch.min(init_tile_cost8x, 1)
        min_tile_cost4x, min_tile_disp4x = torch.min(init_tile_cost4x, 1)
        min_tile_cost2x, min_tile_disp2x = torch.min(init_tile_cost2x, 1)
        min_tile_cost1x, min_tile_disp1x = torch.min(init_tile_cost1x, 1)
        min_tile_cost16x = torch.unsqueeze(min_tile_cost16x, 1)
        min_tile_cost8x = torch.unsqueeze(min_tile_cost8x, 1)
        min_tile_cost4x = torch.unsqueeze(min_tile_cost4x, 1)
        min_tile_cost2x = torch.unsqueeze(min_tile_cost2x, 1)
        min_tile_cost1x = torch.unsqueeze(min_tile_cost1x, 1)
        min_tile_disp16x = min_tile_disp16x.float().unsqueeze(1)
        min_tile_disp8x = min_tile_disp8x.float().unsqueeze(1)
        min_tile_disp4x = min_tile_disp4x.float().unsqueeze(1)
        min_tile_disp2x = min_tile_disp2x.float().unsqueeze(1)
        min_tile_disp1x = min_tile_disp1x.float().unsqueeze(1)
        tile_dscrpt16x = self.tile_fea_dscrpt16x(torch.cat([min_tile_cost16x, tile_feature_pyramid[0][0]], 1))
        tile_dscrpt8x = self.tile_fea_dscrpt8x(torch.cat([min_tile_cost8x, tile_feature_pyramid[1][0]], 1))
        tile_dscrpt4x = self.tile_fea_dscrpt4x(torch.cat([min_tile_cost4x, tile_feature_pyramid[2][0]], 1))
        tile_dscrpt2x = self.tile_fea_dscrpt2x(torch.cat([min_tile_cost2x, tile_feature_pyramid[3][0]], 1))
        tile_dscrpt1x = self.tile_fea_dscrpt1x(torch.cat([min_tile_cost1x, tile_feature_pyramid[4][0]], 1))
        tile_dx16x = torch.zeros_like(min_tile_disp16x)
        tile_dx8x = torch.zeros_like(min_tile_disp8x)
        tile_dx4x = torch.zeros_like(min_tile_disp4x)
        tile_dx2x = torch.zeros_like(min_tile_disp2x)
        tile_dx1x = torch.zeros_like(min_tile_disp1x)
        tile_dy16x = torch.zeros_like(min_tile_disp16x)
        tile_dy8x = torch.zeros_like(min_tile_disp8x)
        tile_dy4x = torch.zeros_like(min_tile_disp4x)
        tile_dy2x = torch.zeros_like(min_tile_disp2x)
        tile_dy1x = torch.zeros_like(min_tile_disp1x)
        tile_hyp16x = torch.cat([min_tile_disp16x, tile_dx16x, tile_dy16x, tile_dscrpt16x], 1)
        tile_hyp8x = torch.cat([min_tile_disp8x, tile_dx8x, tile_dy8x, tile_dscrpt8x], 1)
        tile_hyp4x = torch.cat([min_tile_disp4x, tile_dx4x, tile_dy4x, tile_dscrpt4x], 1)
        tile_hyp2x = torch.cat([min_tile_disp2x, tile_dx2x, tile_dy2x, tile_dscrpt2x], 1)
        tile_hyp1x = torch.cat([min_tile_disp1x, tile_dx1x, tile_dy1x, tile_dscrpt1x], 1)
        return [[init_tile_cost16x, init_tile_cost8x, init_tile_cost4x, init_tile_cost2x, init_tile_cost1x], [tile_hyp16x, tile_hyp8x, tile_hyp4x, tile_hyp2x, tile_hyp1x]]

    def forward(self, fea_l_pyramid, fea_r_pyramid):
        tile_feature_duo_pyramid = self.tile_features(fea_l_pyramid, fea_r_pyramid)
        init_cv_pyramid, init_hypo_pyramid = self.tile_hypothesis_pyramid(tile_feature_duo_pyramid)
        return [init_cv_pyramid, init_hypo_pyramid]


class PostTileUpdate(nn.Module):
    """
    Post Tile Update for a single resolution: decrease tile size, e.g. upsampling tile hypothesis, and do tile warping
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num, slant_disp_up, args):
        super(PostTileUpdate, self).__init__()
        self.disp_upsample = slant_disp_up
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)
        previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)
        previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)
        up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
        up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
        up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
        up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy, up_previous_tile_dscrpt], 1)
        guided_up_previous_tile_hypothesis = torch.cat([up_previous_tile_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = up_previous_tile_hypothesis + tile_hypothesis_update
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())
        return refined_hypothesis


class PostTileUpdateNoUp(nn.Module):
    """
    No hyp upsampling, equal to pure refinement, for 1/4 res
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num, args):
        super(PostTileUpdateNoUp, self).__init__()
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        guided_up_previous_tile_hypothesis = torch.cat([previous_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = previous_hypothesis + tile_hypothesis_update
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())
        return refined_hypothesis


class SlantD2xUpsampleBySlantedPlaneT4T2(nn.Module):
    """
    Slant map upsampling 2x, input tile size = 4x4, output tile size = 2x2
    """

    def __init__(self):
        super(SlantD2xUpsampleBySlantedPlaneT4T2, self).__init__()
        self.DUC = nn.PixelShuffle(2)

    def forward(self, tile_disp, tile_dx, tile_dy):
        disp0 = []
        for i in range(2):
            for j in range(2):
                disp0.append(tile_disp + (i * 2 - 1) * tile_dx + (j * 2 - 1) * tile_dy)
        disp0 = torch.cat(disp0, 1)
        disp1 = self.DUC(disp0)
        return disp1


class SlantDUpsampleBySlantedPlaneT4T4(nn.Module):
    """
    Slant map upsampling, input tile size = 4x4, output tile size = 4x4
    """

    def __init__(self, upscale):
        super(SlantDUpsampleBySlantedPlaneT4T4, self).__init__()
        self.upscale = upscale
        self.center = 4 * (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * self.upscale
        disp0 = []
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i * 4 - self.center) * tile_dx + (j * 4 - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)
        disp1 = self.DUC(disp0)
        return disp1


def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()
    vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    return output


class BuildVolume2dChaos(nn.Module):

    def __init__(self):
        super(BuildVolume2dChaos, self).__init__()

    def forward(self, refimg_fea, targetimg_fea, disps):
        B, C, H, W = refimg_fea.shape
        batch_disp = torch.unsqueeze(disps, dim=2).view(-1, 1, H, W)
        batch_feat_l = refimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
        batch_feat_r = targetimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
        warped_batch_feat_r = warp(batch_feat_r, batch_disp)
        volume = torch.norm(batch_feat_l - warped_batch_feat_r, 1, 1).view(B, disps.shape[1], H, W)
        volume = volume.contiguous()
        return volume


class TileWarping(nn.Module):

    def __init__(self, args):
        super(TileWarping, self).__init__()
        self.disp_up = DispUpsampleBySlantedPlane(4)
        self.build_l1_volume_chaos = BuildVolume2dChaos()

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        tile_d = tile_plane[:, 0, :, :].unsqueeze(1)
        tile_dx = tile_plane[:, 1, :, :].unsqueeze(1)
        tile_dy = tile_plane[:, 2, :, :].unsqueeze(1)
        local_cv = []
        for disp_d in range(-1, 2):
            flatten_local_disp_ws_disp_d = self.disp_up(tile_d + disp_d, tile_dx, tile_dy)
            cv_ws_disp_d = self.build_l1_volume_chaos(fea_l, fea_r, flatten_local_disp_ws_disp_d)
            local_cv_ws_disp_d = []
            for i in range(4):
                for j in range(4):
                    local_cv_ws_disp_d.append(cv_ws_disp_d[:, :, i::4, j::4])
            local_cv_ws_disp_d = torch.cat(local_cv_ws_disp_d, 1)
            local_cv.append(local_cv_ws_disp_d)
        local_cv = torch.cat(local_cv, 1)
        return local_cv


class TileUpdate(nn.Module):
    """
    Tile Update for a single resolution
    forward input: fea duo from current resolution, tile hypothesis from current and previous resolution
    forward output: refined tile hypothesis and confidence (if available)
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num, args):
        super(TileUpdate, self).__init__()
        self.disp_upsample = SlantDUpsampleBySlantedPlaneT4T4(2)
        self.tile_warping = TileWarping(args)
        self.prop_warp0 = BasicConv2d(48, 16, 1, 1, 0, 1)
        self.prop_warp1 = BasicConv2d(48, 16, 1, 1, 0, 1)
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 1, 1, 0, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, fea_l, fea_r, current_hypothesis, previous_hypothesis=None):
        current_tile_local_cv = self.tile_warping(current_hypothesis[:, :3, :, :], fea_l, fea_r)
        current_tile_local_cv = self.prop_warp0(current_tile_local_cv)
        aug_current_tile_hypothesis = torch.cat([current_hypothesis, current_tile_local_cv], 1)
        if previous_hypothesis is None:
            aug_hypothesis_set = aug_current_tile_hypothesis
        else:
            previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)
            previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)
            previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)
            up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
            up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
            up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
            up_previous_tile_dx_dy_dscrpt = torch.cat([up_previous_tile_dx_dy, up_previous_tile_dscrpt], dim=1)
            up_previous_tile_plane = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt[:, :2, :, :]], 1)
            up_previous_tile_local_cv = self.tile_warping(up_previous_tile_plane, fea_l, fea_r)
            up_previous_tile_local_cv = self.prop_warp1(up_previous_tile_local_cv)
            aug_up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt, up_previous_tile_local_cv], 1)
            aug_hypothesis_set = torch.cat([aug_current_tile_hypothesis, aug_up_previous_tile_hypothesis], 1)
        tile_hypothesis_update = self.conv0(aug_hypothesis_set)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        if previous_hypothesis is None:
            refined_hypothesis = current_hypothesis + tile_hypothesis_update
            refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())
            return [refined_hypothesis]
        else:
            conf = tile_hypothesis_update[:, :2, :, :]
            previous_delta_hypothesis = tile_hypothesis_update[:, 2:18, :, :]
            current_delta_hypothesis = tile_hypothesis_update[:, 18:34, :, :]
            _, hypothesis_select_mask = torch.max(conf, dim=1, keepdim=True)
            hypothesis_select_mask = hypothesis_select_mask.float()
            inverse_hypothesis_select_mask = 1 - hypothesis_select_mask
            update_current_hypothesis = current_hypothesis + current_delta_hypothesis
            update_current_hypothesis[:, :1, :, :] = F.relu(update_current_hypothesis[:, :1, :, :].clone())
            update_previous_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt], 1) + previous_delta_hypothesis
            update_previous_hypothesis[:, :1, :, :] = F.relu(update_previous_hypothesis[:, :1, :, :].clone())
            refined_hypothesis = hypothesis_select_mask * update_current_hypothesis + inverse_hypothesis_select_mask * update_previous_hypothesis
            pre_conf = conf[:, :1, :, :]
            cur_conf = conf[:, 1:2, :, :]
            update_current_disp = update_current_hypothesis[:, :1, :, :]
            update_previous_disp = update_previous_hypothesis[:, :1, :, :]
            update_current_dx = update_current_hypothesis[:, 1:2, :, :]
            update_previous_dx = update_previous_hypothesis[:, 1:2, :, :]
            update_current_dy = update_current_hypothesis[:, 2:3, :, :]
            update_previous_dy = update_previous_hypothesis[:, 2:3, :, :]
            return [refined_hypothesis, update_current_disp, update_previous_disp, update_current_dx, update_previous_dx, update_current_dy, update_previous_dy, cur_conf, pre_conf]


class HITNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.feature_extractor = feature_extraction_conv(args)
        self.tile_init = INIT(args)
        self.tile_warp = TileWarping(args)
        self.tile_update0 = TileUpdate(32, 16, 32, 2, args)
        self.tile_update1 = TileUpdate(64, 34, 32, 2, args)
        self.tile_update2 = TileUpdate(64, 34, 32, 2, args)
        self.tile_update3 = TileUpdate(64, 34, 32, 2, args)
        self.tile_update4 = TileUpdate(64, 34, 32, 2, args)
        self.tile_update4_1 = PostTileUpdateNoUp(40, 16, 32, 4, args)
        self.tile_update5 = PostTileUpdate(32, 16, 32, 4, SlantD2xUpsampleBySlantedPlaneT4T2(), args)
        self.tile_update6 = FinalTileUpdate(32, 1, 16, 2, DispUpsampleBySlantedPlane(2, 2), args)
        self.prop_disp_upsample64x = DispUpsampleBySlantedPlane(64)
        self.prop_disp_upsample32x = DispUpsampleBySlantedPlane(32)
        self.prop_disp_upsample16x = DispUpsampleBySlantedPlane(16)
        self.prop_disp_upsample8x = DispUpsampleBySlantedPlane(8)
        self.prop_disp_upsample4x = DispUpsampleBySlantedPlane(4)
        self.prop_disp_upsample2x = DispUpsampleBySlantedPlane(2, 2)
        self.dxdy_upsample64x = nn.UpsamplingNearest2d(scale_factor=64)
        self.dxdy_upsample32x = nn.UpsamplingNearest2d(scale_factor=32)
        self.dxdy_upsample16x = nn.UpsamplingNearest2d(scale_factor=16)
        self.dxdy_upsample8x = nn.UpsamplingNearest2d(scale_factor=8)
        self.dxdy_upsample4x = nn.UpsamplingNearest2d(scale_factor=4)
        self.dxdy_upsample2x = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, left_img, right_img):
        left_fea_pyramid = self.feature_extractor(left_img)
        right_fea_pyramid = self.feature_extractor(right_img)
        init_cv_pyramid, init_tile_pyramid = self.tile_init(left_fea_pyramid, right_fea_pyramid)
        refined_tile16x = self.tile_update0(left_fea_pyramid[0], right_fea_pyramid[0], init_tile_pyramid[0])[0]
        tile_update8x = self.tile_update1(left_fea_pyramid[1], right_fea_pyramid[1], init_tile_pyramid[1], refined_tile16x)
        tile_update4x = self.tile_update2(left_fea_pyramid[2], right_fea_pyramid[2], init_tile_pyramid[2], tile_update8x[0])
        tile_update2x = self.tile_update3(left_fea_pyramid[3], right_fea_pyramid[3], init_tile_pyramid[3], tile_update4x[0])
        tile_update1x = self.tile_update4(left_fea_pyramid[4], right_fea_pyramid[4], init_tile_pyramid[4], tile_update2x[0])
        refined_tile1x = self.tile_update4_1(left_fea_pyramid[2], tile_update1x[0])
        refined_tile05x = self.tile_update5(left_fea_pyramid[3], refined_tile1x)
        refined_tile025x = self.tile_update6(left_fea_pyramid[4], refined_tile05x)
        final_disp = refined_tile025x
        if self.training:
            prop_disp16_fx = self.prop_disp_upsample64x(refined_tile16x[:, :1, :, :], refined_tile16x[:, 1:2, :, :], refined_tile16x[:, 2:3, :, :])
            prop_disp8_fx_cur = self.prop_disp_upsample32x(tile_update8x[1], tile_update8x[3], tile_update8x[5])
            prop_disp8_fx_pre = self.prop_disp_upsample32x(tile_update8x[2], tile_update8x[4], tile_update8x[6])
            prop_disp4_fx_cur = self.prop_disp_upsample16x(tile_update4x[1], tile_update4x[3], tile_update4x[5])
            prop_disp4_fx_pre = self.prop_disp_upsample16x(tile_update4x[2], tile_update4x[4], tile_update4x[6])
            prop_disp2_fx_cur = self.prop_disp_upsample8x(tile_update2x[1], tile_update2x[3], tile_update2x[5])
            prop_disp2_fx_pre = self.prop_disp_upsample8x(tile_update2x[2], tile_update2x[4], tile_update2x[6])
            prop_disp1_fx_cur = self.prop_disp_upsample4x(tile_update1x[1], tile_update1x[3], tile_update1x[5])
            prop_disp1_fx_pre = self.prop_disp_upsample4x(tile_update1x[2], tile_update1x[4], tile_update1x[6])
            prop_disp1_fx = self.prop_disp_upsample4x(refined_tile1x[:, :1, :, :], refined_tile1x[:, 1:2, :, :], refined_tile1x[:, 2:3, :, :])
            prop_disp05_fx = self.prop_disp_upsample2x(refined_tile05x[:, :1, :, :], refined_tile05x[:, 1:2, :, :], refined_tile05x[:, 2:3, :, :])
            prop_disp_pyramid = [prop_disp16_fx, prop_disp8_fx_cur, prop_disp8_fx_pre, prop_disp4_fx_cur, prop_disp4_fx_pre, prop_disp2_fx_cur, prop_disp2_fx_pre, prop_disp1_fx_cur, prop_disp1_fx_pre, prop_disp1_fx, prop_disp05_fx, final_disp]
            dx16_fx = self.dxdy_upsample64x(refined_tile16x[:, 1:2, :, :])
            dx8_fx_cur = self.dxdy_upsample32x(tile_update8x[3])
            dx8_fx_pre = self.dxdy_upsample32x(tile_update8x[4])
            dx4_fx_cur = self.dxdy_upsample16x(tile_update4x[3])
            dx4_fx_pre = self.dxdy_upsample16x(tile_update4x[4])
            dx2_fx_cur = self.dxdy_upsample8x(tile_update2x[3])
            dx2_fx_pre = self.dxdy_upsample8x(tile_update2x[4])
            dx1_fx_cur = self.dxdy_upsample4x(tile_update1x[3])
            dx1_fx_pre = self.dxdy_upsample4x(tile_update1x[4])
            dx1_fx = self.dxdy_upsample4x(refined_tile1x[:, 1:2, :, :])
            dx05_fx = self.dxdy_upsample2x(refined_tile05x[:, 1:2, :, :])
            dx_pyramid = [dx16_fx, dx8_fx_cur, dx8_fx_pre, dx4_fx_cur, dx4_fx_pre, dx2_fx_cur, dx2_fx_pre, dx1_fx_cur, dx1_fx_pre, dx1_fx, dx05_fx]
            dy16_fx = self.dxdy_upsample64x(refined_tile16x[:, 2:3, :, :])
            dy8_fx_cur = self.dxdy_upsample32x(tile_update8x[5])
            dy8_fx_pre = self.dxdy_upsample32x(tile_update8x[6])
            dy4_fx_cur = self.dxdy_upsample16x(tile_update4x[5])
            dy4_fx_pre = self.dxdy_upsample16x(tile_update4x[6])
            dy2_fx_cur = self.dxdy_upsample8x(tile_update2x[5])
            dy2_fx_pre = self.dxdy_upsample8x(tile_update2x[6])
            dy1_fx_cur = self.dxdy_upsample4x(tile_update1x[5])
            dy1_fx_pre = self.dxdy_upsample4x(tile_update1x[6])
            dy1_fx = self.dxdy_upsample4x(refined_tile1x[:, 2:3, :, :])
            dy05_fx = self.dxdy_upsample2x(refined_tile05x[:, 2:3, :, :])
            dy_pyramid = [dy16_fx, dy8_fx_cur, dy8_fx_pre, dy4_fx_cur, dy4_fx_pre, dy2_fx_cur, dy2_fx_pre, dy1_fx_cur, dy1_fx_pre, dy1_fx, dy05_fx]
            conf8_fx_cur = self.dxdy_upsample32x(tile_update8x[7])
            conf8_fx_pre = self.dxdy_upsample32x(tile_update8x[8])
            conf4_fx_cur = self.dxdy_upsample16x(tile_update4x[7])
            conf4_fx_pre = self.dxdy_upsample16x(tile_update4x[8])
            conf2_fx_cur = self.dxdy_upsample8x(tile_update2x[7])
            conf2_fx_pre = self.dxdy_upsample8x(tile_update2x[8])
            conf1_fx_cur = self.dxdy_upsample4x(tile_update1x[7])
            conf1_fx_pre = self.dxdy_upsample4x(tile_update1x[8])
            w_pyramid = [conf8_fx_cur, conf8_fx_pre, conf4_fx_cur, conf4_fx_pre, conf2_fx_cur, conf2_fx_pre, conf1_fx_cur, conf1_fx_pre]
            outputs = {'init_cv_pyramid': init_cv_pyramid, 'prop_disp_pyramid': prop_disp_pyramid, 'dx_pyramid': dx_pyramid, 'dy_pyramid': dy_pyramid, 'w_pyramid': w_pyramid}
            return outputs
        else:
            prop_disp_pyramid = [final_disp]
            return {'prop_disp_pyramid': prop_disp_pyramid}


class TileWarping1(nn.Module):
    """
    Functionality same as TileWarping but with variable tile size
    """

    def __init__(self, tile_size, args):
        super(TileWarping1, self).__init__()
        self.tile_size = tile_size
        self.center = (tile_size - 1) / 2
        self.disp_up = DispUpsampleBySlantedPlane(tile_size)
        self.build_l1_volume_chaos = BuildVolume2dChaos()

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        tile_d = tile_plane[:, 0, :, :].unsqueeze(1)
        tile_dx = tile_plane[:, 1, :, :].unsqueeze(1)
        tile_dy = tile_plane[:, 2, :, :].unsqueeze(1)
        local_cv = []
        for disp_d in range(-1, 2):
            flatten_local_disp_ws_disp_d = self.disp_up(tile_d + disp_d, tile_dx, tile_dy)
            cv_ws_disp_d = self.build_l1_volume_chaos(fea_l, fea_r, flatten_local_disp_ws_disp_d)
            local_cv_ws_disp_d = []
            for i in range(self.tile_size):
                for j in range(self.tile_size):
                    local_cv_ws_disp_d.append(cv_ws_disp_d[:, :, i::self.tile_size, j::self.tile_size])
            local_cv_ws_disp_d = torch.cat(local_cv_ws_disp_d, 1)
            local_cv.append(local_cv_ws_disp_d)
        local_cv = torch.cat(local_cv, 1)
        return local_cv


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BuildVolume2dChaos,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'inplanes': 4, 'out_planes': 4, 'hid_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SlantD2xUpsampleBySlantedPlaneT4T2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TileWarping,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([64, 4, 4, 4]), torch.rand([64, 4, 4, 4])], {}),
     True),
    (TileWarping1,
     lambda: ([], {'tile_size': 4, 'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([64, 4, 4, 4]), torch.rand([64, 4, 4, 4])], {}),
     True),
    (feature_extraction_conv,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_MJITG_PyTorch_HITNet_Hierarchical_Iterative_Tile_Refinement_Network_for_Real_time_Stereo_Matching(_paritybench_base):
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

