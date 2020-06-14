import sys
_module = sys.modules[__name__]
del sys
config = _module
data_util = _module
data_utils = _module
eval = _module
setup = _module
hmean = _module
lanms = _module
__main__ = _module
locality_aware_nms = _module
loss = _module
main = _module
model = _module
TL_iou = _module
pyicdartools = _module
evaluation = _module
rrc_evaluation_funcs = _module
init = _module
myzip = _module
save = _module
util = _module

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


import torch


from torch.autograd import Variable


import torch.nn as nn


from torch import nn


from torch.optim import lr_scheduler


from torch.nn.utils.rnn import pack_padded_sequence


from torch.utils.data import DataLoader


import time


import torch.backends.cudnn as cudnn


import warnings


import numpy as np


import math


import torch.utils.model_zoo as model_zoo


from torch.nn import init


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-05
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls *
        training_mask) + eps
    loss = 1.0 - 2 * intersection / union
    return loss


class LossFunc(nn.Module):

    def __init__(self):
        super(LossFunc, self).__init__()
        return

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo,
        training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls,
            training_mask)
        classification_loss *= 0.01
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo
            , 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        return torch.mean(L_g * y_true_cls * training_mask
            ) + classification_loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        """
        f中的每个元素的size分别是 bs 256 w/4 h/4， bs 512 w/8 h/8， 
        bs 1024 w/16 h/16， bs 2048 w/32 h/32
        """
        return x, f


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    image normalization
    :param images: bs * w * h * channel 
    :param means:
    :return:
    """
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:, (i), :, :] -= means[i]
    return images


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class East(nn.Module):

    def __init__(self):
        super(East, self).__init__()
        self.resnet = resnet50(True)
        self.conv1 = nn.Conv2d(3072, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(640, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(320, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv9 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv10 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, images):
        images = mean_image_subtraction(images)
        _, f = self.resnet(images)
        h = f[3]
        g = self.unpool1(h)
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu1(c)
        h = self.conv2(c)
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.unpool2(h)
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)
        h = self.conv4(c)
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.unpool3(h)
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)
        h = self.conv6(c)
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)
        g = self.bn7(g)
        g = self.relu7(g)
        F_score = self.conv8(g)
        F_score = self.sigmoid1(F_score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid2(geo_map) * 512
        angle_map = self.conv10(g)
        angle_map = self.sigmoid3(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2
        F_geometry = torch.cat((geo_map, angle_map), 1)
        return F_score, F_geometry


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_songdejia_EAST(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

