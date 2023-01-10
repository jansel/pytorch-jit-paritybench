import sys
_module = sys.modules[__name__]
del sys
acc_printer = _module
batch_sampler = _module
blocks = _module
eval = _module
mini_imagenet_dataset = _module
omniglot_dataset = _module
resnet_blocks = _module
snail = _module
train = _module
utils = _module

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


import random


import numpy as np


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


from collections import OrderedDict


from torch.autograd import Variable


from torch.optim import Adam


import copy


class CasualConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]


class DenseBlock(nn.Module):

    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = F.tanh(xf) * F.sigmoid(xg)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):

    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i + 1), filters) for i in range(int(math.ceil(math.log(seq_length, 2))))])

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        mask = np.array([[(1 if i > j else 0) for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask)
        keys = self.linear_keys(input)
        query = self.linear_query(input)
        values = self.linear_values(input)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        temp = torch.bmm(temp, values)
        return torch.cat((input, temp), dim=2)


def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-05, momentum=0.1):
    size = int(np.prod(np.array(input.data.size()[1])))
    running_mean = torch.zeros(size)
    running_var = torch.ones(size)
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)), ('bn', nn.BatchNorm2d(out_channels, momentum=1)), ('relu', nn.ReLU()), ('pool', nn.MaxPool2d(2))]))


class OmniglotNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(OmniglotNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([('block1', conv_block(x_dim, hid_dim)), ('block2', conv_block(hid_dim, hid_dim)), ('block3', conv_block(hid_dim, hid_dim)), ('block4', conv_block(hid_dim, z_dim))]))

    def forward(self, x, weights=None):
        if weights is None:
            x = self.encoder(x)
        else:
            x = F.conv2d(x, weights['encoder.block1.conv.weight'], weights['encoder.block1.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block1.bn.weight'], bias=weights['encoder.block1.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block2.conv.weight'], weights['encoder.block2.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block2.bn.weight'], bias=weights['encoder.block2.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block3.conv.weight'], weights['encoder.block3.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block3.bn.weight'], bias=weights['encoder.block3.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block4.conv.weight'], weights['encoder.block4.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block4.bn.weight'], bias=weights['encoder.block4.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        return x.view(x.size(0), -1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class ResBlock(nn.Module):

    def __init__(self, in_channels, filters, pool_padding=0):
        super(ResBlock, self).__init__()
        self.conv1 = conv(in_channels, filters)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = conv(filters, filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = conv(filters, filters)
        self.bn3 = nn.BatchNorm2d(filters)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = conv(in_channels, filters, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, padding=pool_padding)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        residual = self.conv4(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out += residual
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class MiniImagenetNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, in_channels=3):
        super(MiniImagenetNet, self).__init__()
        self.block1 = ResBlock(in_channels, 64)
        self.block2 = ResBlock(64, 96)
        self.block3 = ResBlock(96, 128, pool_padding=1)
        self.block4 = ResBlock(128, 256, pool_padding=1)
        self.conv1 = conv(256, 2048, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.conv2 = conv(2048, 384, kernel_size=1, padding=0)

    def forward(self, x, weights=None):
        if weights is None:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)
        else:
            raise ValueError('Not implemented yet')
        return x.view(x.size(0), -1)


class SnailFewShot(nn.Module):

    def __init__(self, N, K, task, use_cuda=True):
        super(SnailFewShot, self).__init__()
        if task == 'omniglot':
            self.encoder = OmniglotNet()
            num_channels = 64 + N
        elif task == 'mini_imagenet':
            self.encoder = MiniImagenetNet()
            num_channels = 384 + N
        else:
            raise ValueError('Not recognized task value')
        num_filters = int(math.ceil(math.log(N * K + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, N)
        self.N = N
        self.K = K
        self.use_cuda = use_cuda

    def forward(self, input, labels):
        x = self.encoder(input)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [((i + 1) * (self.N * self.K + 1) - 1) for i in range(batch_size)]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.N * self.K + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionBlock,
     lambda: ([], {'in_channels': 4, 'key_size': 4, 'value_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CasualConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DenseBlock,
     lambda: ([], {'in_channels': 4, 'dilation': 1, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MiniImagenetNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (OmniglotNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TCBlock,
     lambda: ([], {'in_channels': 4, 'seq_length': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_eambutu_snail_pytorch(_paritybench_base):
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

