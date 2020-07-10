import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
mfsan = _module
mmd = _module
resnet = _module
data_loader = _module
mfsan = _module
mfsan1 = _module
mmd = _module
resnet = _module
DAN = _module
ResNet = _module
data_loader = _module
mmd = _module
DDC = _module
ResNet = _module
data_loader = _module
mmd = _module
Coral = _module
DeepCoral = _module
ResNet = _module
data_loader = _module
ResNet = _module
RevGrad = _module
data_loader = _module
DAN = _module
ResNet = _module
data_loader = _module
mmd = _module
Config = _module
DSAN = _module
ResNet = _module
Weight = _module
data_loader = _module
mmd = _module
Coral = _module
DeepCoral = _module
ResNet = _module
data_loader = _module
MRAN = _module
ResNet = _module
data_loader = _module
mmd = _module
ResNet = _module
RevGrad = _module
data_loader = _module
loss = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torchvision import datasets


from torchvision import transforms


import torch


import torch.nn.functional as F


from torch.autograd import Variable


import math


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.optim as optim


from torch.utils import model_zoo


import numpy as np


from torch.autograd import Function


import time


import random


import torchvision


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


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out


model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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
        return x

    def output_num(self):
        return resnet50(True).fc.in_features


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [(bandwidth * kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class MFSAN(nn.Module):

    def __init__(self, num_classes=31):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet50(True)
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.sonnet3 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)
            data_tgt_son1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.avgpool(data_tgt_son1)
            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)
            if mark == 1:
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
            if mark == 2:
                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1) - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1) - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
            if mark == 3:
                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son3)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1) - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
        else:
            data = self.sharedNet(data_src)
            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)
            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)
            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)
            return pred1, pred2, pred3


class DANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(DANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        return source, loss


class DDCNet(nn.Module):

    def __init__(self, num_classes=31):
        super(DDCNet, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        source = self.sharedNet(source)
        loss = 0
        if self.training == True:
            target = self.sharedNet(target)
            loss = mmd.mmd_linear(source, target)
        source = self.cls_fc(source)
        return source, loss


def CORAL(source, target):
    d = source.data.shape[1]
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    loss = torch.mean(torch.mul(xc - xct, xc - xct))
    loss = loss / (4 * d * 4)
    return loss


class DeepCoral(nn.Module):

    def __init__(self, num_classes=31):
        super(DeepCoral, self).__init__()
        self.sharedNet = resnet50(True)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            loss += CORAL(source, target)
        source = self.cls_fc(source)
        return source, loss


class AdversarialLayer(torch.autograd.Function):

    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 2000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        return -self.coeff * gradOutput


class AdversarialNetwork(nn.Module):

    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1


class RevGrad(nn.Module):

    def __init__(self, num_classes=31):
        super(RevGrad, self).__init__()
        self.sharedNet = resnet50(True)
        self.cls_fn = nn.Linear(2048, num_classes)
        self.domain_fn = AdversarialNetwork(in_feature=2048)

    def forward(self, data):
        data = self.sharedNet(data)
        clabel_pred = self.cls_fn(data)
        dlabel_pred = self.domain_fn(AdversarialLayer(high_value=1.0)(data))
        return clabel_pred, dlabel_pred


bottle_neck = True


class DSAN(nn.Module):

    def __init__(self, num_classes=31):
        super(DSAN, self).__init__()
        self.feature_layers = resnet50(True)
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        if self.training == True:
            target = self.feature_layers(target)
            if bottle_neck:
                target = self.bottle(target)
            t_label = self.cls_fc(target)
            loss = mmd.lmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        else:
            loss = 0
        return s_pred, loss


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, num_classes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.source_fc = nn.Linear(288, num_classes)

    def forward(self, source, target, s_label):
        s_branch1x1 = self.branch1x1(source)
        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)
        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)
        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)
        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)
        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)
        t_branch1x1 = self.branch1x1(target)
        t_branch5x5 = self.branch5x5_1(target)
        t_branch5x5 = self.branch5x5_2(t_branch5x5)
        t_branch3x3dbl = self.branch3x3dbl_1(target)
        t_branch3x3dbl = self.branch3x3dbl_2(t_branch3x3dbl)
        t_branch3x3dbl = self.branch3x3dbl_3(t_branch3x3dbl)
        t_branch_pool = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        t_branch_pool = self.branch_pool(t_branch_pool)
        t_branch1x1 = self.avg_pool(t_branch1x1)
        t_branch5x5 = self.avg_pool(t_branch5x5)
        t_branch3x3dbl = self.avg_pool(t_branch3x3dbl)
        t_branch_pool = self.avg_pool(t_branch_pool)
        t_branch1x1 = t_branch1x1.view(t_branch1x1.size(0), -1)
        t_branch5x5 = t_branch5x5.view(t_branch5x5.size(0), -1)
        t_branch3x3dbl = t_branch3x3dbl.view(t_branch3x3dbl.size(0), -1)
        t_branch_pool = t_branch_pool.view(t_branch_pool.size(0), -1)
        source = torch.cat([s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool], 1)
        target = torch.cat([t_branch1x1, t_branch5x5, t_branch3x3dbl, t_branch_pool], 1)
        source = self.source_fc(source)
        t_label = self.source_fc(target)
        t_label = t_label.data.max(1)[1]
        loss = torch.Tensor([0])
        loss = loss
        if self.training == True:
            loss += mmd.cmmd(s_branch1x1, t_branch1x1, s_label, t_label)
            loss += mmd.cmmd(s_branch5x5, t_branch5x5, s_label, t_label)
            loss += mmd.cmmd(s_branch3x3dbl, t_branch3x3dbl, s_label, t_label)
            loss += mmd.cmmd(s_branch_pool, t_branch_pool, s_label, t_label)
        return source, loss


class MRANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(MRANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.Inception = InceptionA(2048, 64, num_classes)

    def forward(self, source, target, s_label):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        source, loss = self.Inception(source, target, s_label)
        return source, loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ADDneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AdversarialNetwork,
     lambda: ([], {'in_feature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DANNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DDCNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DSAN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeepCoral,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_easezyc_deep_transfer_learning(_paritybench_base):
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

