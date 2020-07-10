import sys
_module = sys.modules[__name__]
del sys
convnet_aig = _module
gumbelmodule = _module
train = _module
train_img = _module
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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import numpy as np


from torch.autograd import Variable


import time


import torch.optim as optim


from torchvision import transforms


import torch.backends.cudnn as cudnn


import torchvision.datasets as datasets


import torch.nn.init as init


class Sequential_ext(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, temperature=1, openings=None):
        gate_activations = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation = module(input, temperature)
            gate_activations.append(gate_activation)
        return input, gate_activations


class GumbleSoftmax(torch.nn.Module):

    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise)
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = -torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()
        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)


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
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.gs = GumbleSoftmax()
        self.gs

    def forward(self, x, temperature=1):
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        w = self.gs(w, temp=temperature, force_hard=True)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out * w[:, (1)].unsqueeze(1)
        out = F.relu(out)
        return out, w[:, (1)]


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
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.gs = GumbleSoftmax()
        self.gs

    def forward(self, x, temperature=1):
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        w = self.gs(w, temp=temperature, force_hard=True)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x) + out * w[:, (1)].unsqueeze(1)
        out = F.relu(out, inplace=True)
        return out, w[:, (1)]


class ResNet_ImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if 'fc2' in str(k):
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, out, temperature=1):
        gate_activations = []
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out, a = self.layer1(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer2(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer3(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer4(out, temperature)
        gate_activations.extend(a)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, gate_activations


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet_cifar(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'fc2' in str(k):
                    m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential_ext(*layers)

    def forward(self, x, temperature=1, openings=None):
        gate_activations = []
        out = F.relu(self.bn1(self.conv1(x)))
        out, a = self.layer1(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer2(out, temperature)
        gate_activations.extend(a)
        out, a = self.layer3(out, temperature)
        gate_activations.extend(a)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, gate_activations


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GumbleSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Sequential_ext,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_andreasveit_convnet_aig(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

