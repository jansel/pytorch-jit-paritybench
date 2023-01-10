import sys
_module = sys.modules[__name__]
del sys
LSR = _module
main_visual = _module
model = _module
model = _module
video_cnn = _module
prepare_lrw = _module
prepare_lrw1000 = _module
utils = _module
cvtransforms = _module
dataset = _module
dataset_lrw1000 = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


import math


import numpy as np


import time


import torch.optim as optim


import random


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.autograd import Variable


from torch.utils.data import Dataset


from collections import defaultdict


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added
        one_hot = one_hot
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)
        return one_hot

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'.format(x.size(0), target.size(0)))
        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'.format(x.size(0)))
        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'.format(x.size()))
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(-x * smoothed_target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            out = out * w
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x


class VideoCNN(nn.Module):

    def __init__(self, se=False):
        super(VideoCNN, self).__init__()
        self.frontend3D = nn.Sequential(nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False), nn.BatchNorm3d(64), nn.ReLU(True), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        return x

    def forward(self, x):
        b, t = x.size()[:2]
        x = self.visual_frontend_forward(x)
        feat = x.view(b, -1, 512)
        x = x.view(b, -1, 512)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)
        if self.args.border:
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.v_cls = nn.Linear(1024 * 2, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        self.gru.flatten_parameters()
        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)
        if self.args.border:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:
            h, _ = self.gru(f_v)
        y_v = self.v_cls(self.dropout(h)).mean(1)
        return y_v


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LSR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (VideoCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 1, 64, 64])], {}),
     False),
]

class Test_VIPL_Audio_Visual_Speech_Understanding_learn_an_effective_lip_reading_model_without_pains(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

