import sys
_module = sys.modules[__name__]
del sys
funcs = _module
models = _module
densenet = _module
wideresnet = _module
prune = _module
train = _module

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


import random


import numpy as np


import torchvision.transforms as transforms


import torchvision


import time


from functools import reduce


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision.datasets as dset


from torch.utils.data import DataLoader


import torchvision.models as models


import math


import torch.utils.model_zoo as model_zoo


import torch.optim.lr_scheduler as lr_scheduler


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x * 0


class ZeroMake(nn.Module):

    def __init__(self, channels, spatial):
        super(ZeroMake, self).__init__()
        self.spatial = spatial
        self.channels = channels

    def forward(self, x):
        return torch.zeros([x.size()[0], self.channels, x.size()[2] // self.spatial, x.size()[3] // self.spatial], dtype=x.dtype, layout=x.layout, device=x.device)


class MaskBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropRate=0.0):
        super(MaskBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_channels == out_channels
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.activation = Identity()
        self.activation.register_backward_hook(self._fisher)
        self.register_buffer('mask', None)
        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.got_shapes = False
        self.running_fisher = 0

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        if self.mask is not None:
            out = out * self.mask[(None), :, (None), (None)]
        else:
            self._create_mask(x, out)
        out = self.activation(out)
        self.act = out
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def _create_mask(self, x, out):
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, notused1, notused2, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()
        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k

    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def cost(self):
        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())
        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()
        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[3] + middle_channels * out_channels * conv2_size[2] * conv2_size[3]
        self.params += 2 * in_channels + 2 * middle_channels
        if not self.equalInOut:
            self.params += in_channels * out_channels
        else:
            self.params += 0

    def compress_weights(self):
        middle_dim = int(self.mask.sum().item())
        None
        if middle_dim is not 0:
            conv1 = nn.Conv2d(self.in_channels, middle_dim, kernel_size=3, stride=self.stride, padding=1, bias=False)
            conv1.weight = nn.Parameter(self.conv1.weight[(self.mask == 1), :, :, :])
            bn2 = nn.BatchNorm2d(middle_dim)
            bn2.weight = nn.Parameter(self.bn2.weight[self.mask == 1])
            bn2.bias = nn.Parameter(self.bn2.bias[self.mask == 1])
            bn2.running_mean = self.bn2.running_mean[self.mask == 1]
            bn2.running_var = self.bn2.running_var[self.mask == 1]
            conv2 = nn.Conv2d(middle_dim, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            conv2.weight = nn.Parameter(self.conv2.weight[:, (self.mask == 1), :, :])
        if middle_dim is 0:
            conv1 = Zero()
            bn2 = Zero()
            conv2 = ZeroMake(channels=self.out_channels, spatial=self.stride)
        self.conv1 = conv1
        self.conv2 = conv2
        self.bn2 = bn2
        if middle_dim is not 0:
            self.mask = torch.ones(middle_dim)
        else:
            self.mask = torch.ones(1)


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate, width=1):
        super(Bottleneck, self).__init__()
        interChannels = int(4 * growthRate * width)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, mask=False, width=1.0):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, mask, width)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, mask, width)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, mask, width)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        self.fixed_params = len(self.conv1.weight.view(-1)) + len(self.bn1.weight) + len(self.bn1.bias) + len(self.fc.weight.view(-1)) + len(self.fc.bias)
        self.fixed_params += len(self.trans1.conv1.weight.view(-1)) + 2 * len(self.trans1.bn1.weight)
        self.fixed_params += len(self.trans2.conv1.weight.view(-1)) + 2 * len(self.trans2.bn1.weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, mask=False, width=1):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck and mask:
                layers.append(MaskBlock(nChannels, growthRate))
            elif bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, width))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_channels == out_channels
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BottleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, stride, dropRate=0.0):
        super(BottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_channels == out_channels
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_channels, out_channels, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_channels, out_channels, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class NetworkBlockBottle(nn.Module):

    def __init__(self, nb_layers, in_channels, out_channels, mid_channels, block, stride, dropRate=0.0):
        super(NetworkBlockBottle, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, mid_channels, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_channels, out_channels, mid_channels, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels, mid_channels, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0, mask=False):
        super(WideResNet, self).__init__()
        nChannels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        if mask == 1:
            block = MaskBlock
        else:
            block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.fixed_params = len(self.conv1.weight.view(-1)) + len(self.bn1.weight) + len(self.bn1.bias) + len(self.fc.weight.view(-1)) + len(self.fc.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNetBottle(nn.Module):

    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0, bottle_mult=0.5):
        super(WideResNetBottle, self).__init__()
        nChannels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BottleBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlockBottle(n, nChannels[0], nChannels[1], int(nChannels[1] * bottle_mult), block, 1, dropRate)
        self.block2 = NetworkBlockBottle(n, nChannels[1], nChannels[2], int(nChannels[2] * bottle_mult), block, 2, dropRate)
        self.block3 = NetworkBlockBottle(n, nChannels[2], nChannels[3], int(nChannels[3] * bottle_mult), block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BottleBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'mid_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_channels': 4, 'out_channels': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetworkBlockBottle,
     lambda: ([], {'nb_layers': 1, 'in_channels': 4, 'out_channels': 4, 'mid_channels': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Zero,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZeroMake,
     lambda: ([], {'channels': 4, 'spatial': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_BayesWatch_pytorch_prunes(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

