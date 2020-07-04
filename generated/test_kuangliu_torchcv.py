import sys
_module = sys.modules[__name__]
del sys
demo = _module
eval = _module
get_state_dict = _module
train = _module
get_state_dict = _module
train = _module
demo = _module
eval = _module
train = _module
test_dataloader = _module
test_transform = _module
torchcv = _module
datasets = _module
listdataset = _module
evaluations = _module
voc_eval = _module
loss = _module
focal_loss = _module
ssd_loss = _module
models = _module
fpnssd = _module
box_coder = _module
fpn = _module
net = _module
retinanet = _module
fpn = _module
net = _module
ssd = _module
net = _module
transforms = _module
pad = _module
random_crop = _module
random_distort = _module
random_flip = _module
random_paste = _module
resize = _module
scale_jitter = _module
utils = _module
box = _module
meshgrid = _module
one_hot_embedding = _module
visualizations = _module
vis_image = _module

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


import torch


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import torchvision


from torch.autograd import Variable


import random


import torch.nn as nn


import torch.optim as optim


import math


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


class FocalLoss(nn.Module):

    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def _focal_loss(self, x, y):
        """Focal loss.

        This is described in the original paper.
        With BCELoss, the background should not be counted in num_classes.

        Args:
          x: (tensor) predictions, sized [N,D].
          y: (tensor) targets, sized [N,].

        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2
        t = one_hot_embedding(y - 1, self.num_classes)
        p = x.sigmoid()
        pt = torch.where(t > 0, p, 1 - p)
        w = (1 - pt).pow(gamma)
        w = torch.where(t > 0, alpha * w, (1 - alpha) * w)
        loss = F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        return loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.sum().item()
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask],
            size_average=False)
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self._focal_loss(masked_cls_preds, cls_targets[pos_neg])
        None
        loss = (loc_loss + cls_loss) / num_pos
        return loss


class SSDLoss(nn.Module):

    def __init__(self, num_classes):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes

    def _hard_negative_mining(self, cls_loss, pos):
        """Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        """
        cls_loss = cls_loss * (pos.float() - 1)
        _, idx = cls_loss.sort(1)
        _, rank = idx.sort(1)
        num_neg = 3 * pos.sum(1)
        neg = rank < num_neg[:, (None)]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        """
        pos = cls_targets > 0
        batch_size = pos.size(0)
        num_pos = pos.sum().item()
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask],
            size_average=False)
        cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes),
            cls_targets.view(-1), reduce=False)
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets < 0] = 0
        neg = self._hard_negative_mining(cls_loss, pos)
        cls_loss = cls_loss[pos | neg].sum()
        None
        loss = (loc_loss + cls_loss) / num_pos
        return loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False
            ) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7, p8, p9


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


class FPNSSD512(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super(FPNSSD512, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(x)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1,
                self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1,
                padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1))
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False
            ) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(x)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1,
                self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1,
                padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1))
        return nn.Sequential(*layers)


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = self._make_layers()

    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self):
        """VGG16 layers."""
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=
                    True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding
                    =1), nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


class L2Norm(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[(None), :, (None), (None)]
        return scale * x


class VGG16Extractor300(nn.Module):

    def __init__(self):
        super(VGG16Extractor300, self).__init__()
        self.features = VGG16()
        self.norm4 = L2Norm(512, 20)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x):
        hs = []
        h = self.features(x)
        hs.append(self.norm4(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)
        return hs


class SSD300(nn.Module):
    steps = 8, 16, 32, 64, 100, 300
    box_sizes = 30, 60, 111, 162, 213, 264, 315
    aspect_ratios = (2,), (2, 3), (2, 3), (2, 3), (2,), (2,)
    fm_sizes = 38, 19, 10, 5, 3, 1

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 4, 6, 6, 6, 4, 4
        self.in_channels = 512, 1024, 512, 256, 256, 256
        self.extractor = VGG16Extractor300()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.
                num_anchors[i] * 4, kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], self.
                num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.
                num_classes))
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


class VGG16Extractor512(nn.Module):

    def __init__(self):
        super(VGG16Extractor512, self).__init__()
        self.features = VGG16()
        self.norm4 = L2Norm(512, 20)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv12_2 = nn.Conv2d(128, 256, kernel_size=4, padding=1)

    def forward(self, x):
        hs = []
        h = self.features(x)
        hs.append(self.norm4(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)
        h = F.relu(self.conv12_1(h))
        h = F.relu(self.conv12_2(h))
        hs.append(h)
        return hs


class SSD512(nn.Module):
    steps = 8, 16, 32, 64, 128, 256, 512
    box_sizes = 35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6
    aspect_ratios = (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)
    fm_sizes = 64, 32, 16, 8, 4, 2, 1

    def __init__(self, num_classes):
        super(SSD512, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 4, 6, 6, 6, 6, 4, 4
        self.in_channels = 512, 1024, 512, 256, 256, 256, 256
        self.extractor = VGG16Extractor512()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.
                num_anchors[i] * 4, kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], self.
                num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.
                num_classes))
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kuangliu_torchcv(_paritybench_base):
    pass
    def test_000(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(FPNSSD512(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FocalLoss(*[], **{'num_classes': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.zeros([4, 4], dtype=torch.int64)], {})

    def test_003(self):
        self._check(L2Norm(*[], **{'in_features': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(RetinaNet(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_005(self):
        self._check(VGG16(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

