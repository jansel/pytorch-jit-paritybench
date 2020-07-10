import sys
_module = sys.modules[__name__]
del sys
cifar = _module
datasets = _module
cifar = _module
folder = _module
LinearAverage = _module
NCA = _module
lib = _module
normalize = _module
utils = _module
main = _module
models = _module
resnet = _module
resnet_cifar = _module
test = _module

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


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


import time


import math


import torchvision.datasets as datasets


import torch.utils.data as data


from torch.autograd import Function


from torch import nn


from torch.autograd import Variable


import torch.nn.parallel


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torch.utils.model_zoo as model_zoo


import numpy as np


class LinearAverageOp(Function):

    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)
        out = torch.mm(x.data, memory.t())
        out.div_(T)
        self.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        gradOutput.data.div_(T)
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None


class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


class NCACrossEntropy(nn.Module):
    """ \\sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    """

    def __init__(self, labels, margin=0):
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels', torch.LongTensor(labels.size(0)))
        self.labels = labels
        self.margin = margin

    def forward(self, x, indexes):
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)
        y = torch.index_select(self.labels, 0, indexes.data).view(batchSize, 1)
        same = y.repeat(1, n).eq_(self.labels)
        exp.data.scatter_(1, indexes.data.view(-1, 1), 0)
        p = torch.mul(exp, same.float()).sum(dim=1)
        Z = exp.sum(dim=1)
        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p
        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        loss = prob_masked.log().sum(0)
        return -loss / batchSize


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.l2norm(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearAverage,
     lambda: ([], {'inputSize': 4, 'outputSize': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_microsoft_snca_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

