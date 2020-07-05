import sys
_module = sys.modules[__name__]
del sys
helpers = _module
plotGT = _module
velpacks2pc = _module
via2label = _module
datagen = _module
logger = _module
loss = _module
main = _module
model = _module
postprocess = _module
sample = _module
run_kitti = _module
testset = _module
utils = _module

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


import torch.nn.functional as F


import numpy as np


import time


import random


from torch.multiprocessing import Pool


import math


import torch.nn


class CustomLoss(nn.Module):

    def __init__(self, device, config, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.alpha = config['alpha']
        self.beta = config['beta']

    def focal_loss(self, x, y):
        """Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2
        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y)
        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)
        loss = -alpha_t * (1 - x_t) ** gamma * x_t.log()
        return loss.mean()

    def cross_entropy(self, x, y):
        return F.binary_cross_entropy(input=x, target=y, reduction='elementwise_mean')

    def forward(self, preds, targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        batch_size = targets.size(0)
        image_size = targets.size(1) * targets.size(2)
        cls_targets, loc_targets = targets.split([1, 6], dim=1)
        if preds.size(1) == 7:
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
        elif preds.size(1) == 15:
            cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)
        cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha
        cls = cls_loss.item()
        pos_pixels = cls_targets.sum()
        if pos_pixels > 0:
            loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
            loc = loc_loss.item()
            loss = loc_loss + cls_loss
        else:
            loc = 0.0
            loss = cls_loss
        return loss, cls, loc


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_bn=True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + out)
        return out


class BackBone(nn.Module):

    def __init__(self, block, num_block, geom, use_bn=True):
        super(BackBone, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(36, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 0 if geom['label_shape'][1] == 175 else 1
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)
        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)
        return p4

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(planes * block.expansion))
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=2, bias=True)
        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y


class Header(nn.Module):

    def __init__(self, use_bn=True):
        super(Header, self).__init__()
        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)
        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, 6, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return cls, reg


class Decoder(nn.Module):

    def __init__(self, geom):
        super(Decoder, self).__init__()
        self.geometry = [geom['L1'], geom['L2'], geom['W1'], geom['W2']]
        self.grid_size = 0.4
        self.target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        self.target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]

    def forward(self, x):
        """

        :param x: Tensor 6-channel geometry
        6 channel map of [cos(yaw), sin(yaw), log(x), log(y), w, l]
        Shape of x: (B, C=6, H=200, W=175)
        :return: Concatenated Tensor of 8 channel geometry map of bounding box corners
        8 channel are [rear_left_x, rear_left_y,
                        rear_right_x, rear_right_y,
                        front_right_x, front_right_y,
                        front_left_x, front_left_y]
        Return tensor has a shape (B, C=8, H=200, W=175), and is located on the same device as x

        """
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        for i in range(6):
            x[:, (i), :, :] = x[:, (i), :, :] * self.target_std_dev[i] + self.target_mean[i]
        cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
        theta = torch.atan2(sin_t, cos_t)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x = torch.arange(self.geometry[2], self.geometry[3], self.grid_size, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[0], self.geometry[1], self.grid_size, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid([y, x])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        rear_left_x = centre_x - l / 2 * cos_t - w / 2 * sin_t
        rear_left_y = centre_y - l / 2 * sin_t + w / 2 * cos_t
        rear_right_x = centre_x - l / 2 * cos_t + w / 2 * sin_t
        rear_right_y = centre_y - l / 2 * sin_t - w / 2 * cos_t
        front_right_x = centre_x + l / 2 * cos_t + w / 2 * sin_t
        front_right_y = centre_y + l / 2 * sin_t - w / 2 * cos_t
        front_left_x = centre_x + l / 2 * cos_t - w / 2 * sin_t
        front_left_y = centre_y + l / 2 * sin_t + w / 2 * cos_t
        decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y, front_right_x, front_right_y, front_left_x, front_left_y], dim=1)
        return decoded_reg


def maskFOV_on_BEV(shape, fov=88.0):
    height = shape[0]
    width = shape[1]
    fov = fov / 2
    x = np.arange(width)
    y = np.arange(-height // 2, height // 2)
    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx) * 180 / np.pi
    in_fov = np.abs(angle) < fov
    in_fov = torch.from_numpy(in_fov.astype(np.float32))
    return in_fov


class PIXOR(nn.Module):
    """
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    """

    def __init__(self, geom, use_bn=True, decode=False):
        super(PIXOR, self).__init__()
        self.backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn)
        self.header = Header(use_bn)
        self.corner_decoder = Decoder(geom)
        self.use_decode = decode
        self.cam_fov_mask = maskFOV_on_BEV(geom['label_shape'])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0 - prior) / prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x):
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        features = self.backbone(x)
        cls, reg = self.header(features)
        self.cam_fov_mask = self.cam_fov_mask
        cls = cls * self.cam_fov_mask
        if self.use_decode:
            decoded = self.corner_decoder(reg)
            pred = torch.cat([cls, reg, decoded], dim=1)
        else:
            pred = torch.cat([cls, reg], dim=1)
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Header,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     True),
]

class Test_philip_huang_PIXOR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

