import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cityscapes_Dataset = _module
crosscity_Dataset = _module
gta5_Dataset = _module
synthia_Dataset = _module
graphs = _module
ResNet101 = _module
models = _module
deeplab_multi = _module
tools = _module
evaluate = _module
solve_crosscity = _module
solve_gta5 = _module
train_source = _module
utils = _module
eval = _module
loss = _module
train_helper = _module

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


import copy


import random


import scipy.io


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as ttransforms


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import logging


from math import ceil


from math import floor


from torchvision.utils import make_grid


import torch.optim as optim


from torch.autograd import Variable


import time


affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
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


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_momentum=0.1, pretrained=False, output_stride=16, norm_layer=nn.BatchNorm2d):
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        elif output_stride == 32:
            dilations = [1, 1, 1, 1]
            strides = [1, 2, 2, 2]
        else:
            raise Warning('output_stride must be 8 or 16!')
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], bn_momentum=bn_momentum, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], bn_momentum=bn_momentum, norm_layer=norm_layer)
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, bn_momentum=bn_momentum, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, bn_momentum=bn_momentum, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Classifier_Module(nn.Module):

    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer5(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)
        x2 = self.layer4(x)
        x2 = self.layer6(x2)
        x2 = F.interpolate(x2, size=input_size, mode='bilinear', align_corners=True)
        return x2, x1

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr}, {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]


def resnet101(bn_momentum=0.1, pretrained=False, output_stride=16, multi=False, norm_layer=nn.BatchNorm2d):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if not multi:
        model = ResNet(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride, norm_layer=norm_layer)
    else:
        model = ResNetMulti(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride, norm_layer=norm_layer)
    return model


class LabelNet(nn.Module):

    def __init__(self, class_num, pretrained=True, output_stride=8, bn_momentum=0.1, freeze_bn=False):
        super().__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained, output_stride, multi=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, class_num)
        if freeze_bn:
            self.freeze_bn()
            None

    def forward(self, x):
        x, _ = self.Resnet101(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class softCrossEntropy(nn.Module):

    def __init__(self, ignore_index=-1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = target != self.ignore_index
        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])
        return loss


class IWsoftCrossEntropy(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss with image-wise weighting factor
        """
        assert inputs.size() == target.size()
        mask = target != self.ignore_index
        _, argpred = torch.max(inputs, 1)
        weights = []
        batch_size = inputs.size(0)
        for i in range(batch_size):
            hist = torch.histc(argpred[i].cpu().data.float(), bins=self.num_class, min=0, max=self.num_class - 1).float()
            weight = (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1)))[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.sum((torch.mul(-log_likelihood, target) * weights)[mask]) / (batch_size * self.num_class)
        return loss


class IW_MaxSquareloss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        N, C, H, W = prob.size()
        mask = prob != self.ignore_index
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = maxpred != self.ignore_index
        argpred = torch.where(mask_arg, argpred, torch.ones(1) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(), bins=self.num_class + 1, min=-1, max=self.num_class - 1).float()
            hist = hist[1:]
            weight = (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1)))[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss


class MaxSquareloss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class

    def forward(self, pred, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        mask = prob != self.ignore_index
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier_Module,
     lambda: ([], {'inplanes': 4, 'dilation_series': [4, 4], 'padding_series': [4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IWsoftCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxSquareloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (softCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ZJULearning_MaxSquareLoss(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

