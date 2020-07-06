import sys
_module = sys.modules[__name__]
del sys
MiniImagenet = _module
compare = _module
repnet = _module
train = _module
utils = _module

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


import torch


from torch.utils.data import Dataset


from torchvision.transforms import transforms


import numpy as np


import collections


from torch import nn


from torch import optim


from torch.autograd import Variable


from torch.nn import functional as F


import torch.nn as nn


import math


from torch.utils import model_zoo


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
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
        return x


def repnet_deep(pretrained=False, **kwargs):
    """Constructs a ResNet-Mini-Imagenet model.

	Args:
	"""
    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}
    model = ResNet(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


class Compare(nn.Module):
    """
	repnet => feature concat => layer4 & layer5 & avg pooling => fc => sigmoid
	"""

    def __init__(self, n_way, k_shot):
        super(Compare, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.repnet = repnet_deep(False)
        repnet_sz = self.repnet(Variable(torch.rand(2, 3, 224, 224))).size()
        self.c = repnet_sz[1]
        self.d = repnet_sz[2]
        self.inplanes = 2 * self.c
        assert repnet_sz[2] == repnet_sz[3]
        None
        self.layer4 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def _make_layer(self, block, planes, blocks, stride=1):
        """
		make Bottleneck layer * blocks.
		:param block:
		:param planes:
		:param blocks:
		:param stride:
		:return:
		"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, support_x, support_y, query_x, query_y, train=True):
        """

		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
        batchsz, setsz, c_, h, w = support_x.size()
        querysz = query_x.size(1)
        c, d = self.c, self.d
        support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
        query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)
        support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)
        comb = torch.cat([support_xf, query_xf], dim=3)
        comb = self.layer5(self.layer4(comb.view(batchsz * querysz * setsz, 2 * c, d, d)))
        comb = F.avg_pool2d(comb, 3)
        score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3)
        support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
        query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
        label = torch.eq(support_yf, query_yf).float()
        if train:
            loss = torch.pow(label - score, 2).sum() / batchsz
            return loss
        else:
            rn_score_np = score.cpu().data.numpy()
            pred = []
            support_y_np = support_y.cpu().data.numpy()
            for i, batch in enumerate(rn_score_np):
                for j, query in enumerate(batch):
                    sim = []
                    for way in range(self.n_way):
                        sim.append(np.sum(query[way * self.k_shot:(way + 1) * self.k_shot]))
                    idx = np.array(sim).argmax()
                    pred.append(support_y_np[i, idx * self.k_shot])
            pred = Variable(torch.from_numpy(np.array(pred).reshape((batchsz, querysz))))
            correct = torch.eq(pred, query_y).sum()
            return pred, correct


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.Sigmoid()
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dragen1860_LearningToCompare_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

