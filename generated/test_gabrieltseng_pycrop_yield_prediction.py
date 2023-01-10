import sys
_module = sys.modules[__name__]
del sys
cyp = _module
analysis = _module
counties_plot = _module
data = _module
exporting = _module
feature_engineering = _module
preprocessing = _module
utils = _module
models = _module
base = _module
convnet = _module
gp = _module
loss = _module
rnn = _module
run = _module

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


import matplotlib as mpl


import matplotlib.pyplot as plt


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.utils.data import random_split


import numpy as np


import pandas as pd


from collections import defaultdict


from collections import namedtuple


from torch import nn


import torch.nn.functional as F


import math


def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)
    rows_odd = padding_rows % 2 != 0
    input_cols = input.size(3)
    filter_cols = weight.size(3)
    effective_filter_size_cols = (filter_cols - 1) * dilation[1] + 1
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(0, (out_cols - 1) * stride[1] + effective_filter_size_cols - input_cols)
    cols_odd = padding_cols % 2 != 0
    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride, padding=(padding_rows // 2, padding_cols // 2), dilation=dilation, groups=groups)


class Conv2dSamePadding(nn.Conv2d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867

    This solution is mostly copied from
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036

    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


class ConvBlock(nn.Module):
    """
    A 2D convolution, followed by batchnorm, a ReLU activation, and dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv(x)))
        return self.dropout(x)


class ConvNet(nn.Module):
    """
    A crop yield conv net.

    For a description of the parameters, see the ConvModel class.
    Only handles strides of 1 and 2
    """

    def __init__(self, in_channels=9, dropout=0.5, dense_features=None, time=32):
        super().__init__()
        in_out_channels_list = [in_channels, 128, 256, 256, 512, 512, 512]
        stride_list = [None, 1, 2, 1, 2, 1, 2]
        num_divisors = sum([(1 if i == 2 else 0) for i in stride_list])
        for i in range(num_divisors):
            if time % 2 != 0:
                time += 1
            time /= 2
        if dense_features is None:
            dense_features = [2048, 1]
        dense_features.insert(0, int(in_out_channels_list[-1] * time * 4))
        assert len(stride_list) == len(in_out_channels_list), 'Stride list and out channels list must be the same length!'
        self.convblocks = nn.ModuleList([ConvBlock(in_channels=in_out_channels_list[i - 1], out_channels=in_out_channels_list[i], kernel_size=3, stride=stride_list[i], dropout=dropout) for i in range(1, len(stride_list))])
        self.dense_layers = nn.ModuleList([nn.Linear(in_features=dense_features[i - 1], out_features=dense_features[i]) for i in range(1, len(dense_features))])
        self.initialize_weights()

    def initialize_weights(self):
        for convblock in self.convblocks:
            nn.init.kaiming_uniform_(convblock.conv.weight.data)
            nn.init.constant_(convblock.conv.bias.data, 0)
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x, return_last_dense=False):
        """
        If return_last_dense is true, the feature vector generated by the second to last
        dense layer will also be returned. This is then used to train a Gaussian Process model.
        """
        for block in self.convblocks:
            x = block(x)
        x = x.view(x.shape[0], -1)
        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if return_last_dense and layer_number == len(self.dense_layers) - 2:
                output = x
        if return_last_dense:
            return x, output
        return x


class RNNet(nn.Module):
    """
    A crop yield conv net.

    For a description of the parameters, see the RNNModel class.
    """

    def __init__(self, in_channels=9, num_bins=32, hidden_size=128, num_rnn_layers=1, rnn_dropout=0.25, dense_features=None):
        super().__init__()
        if dense_features is None:
            dense_features = [256, 1]
        dense_features.insert(0, hidden_size)
        self.dropout = nn.Dropout(rnn_dropout)
        self.rnn = nn.LSTM(input_size=in_channels * num_bins, hidden_size=hidden_size, num_layers=num_rnn_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.dense_layers = nn.ModuleList([nn.Linear(in_features=dense_features[i - 1], out_features=dense_features[i]) for i in range(1, len(dense_features))])
        self.initialize_weights()

    def initialize_weights(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.all_weights:
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x, return_last_dense=False):
        """
        If return_last_dense is true, the feature vector generated by the second to last
        dense layer will also be returned. This is then used to train a Gaussian Process model.
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        sequence_length = x.shape[1]
        hidden_state = torch.zeros(1, x.shape[0], self.hidden_size)
        cell_state = torch.zeros(1, x.shape[0], self.hidden_size)
        if x.is_cuda:
            hidden_state = hidden_state
            cell_state = cell_state
        for i in range(sequence_length):
            input_x = x[:, i, :].unsqueeze(1)
            _, (hidden_state, cell_state) = self.rnn(input_x, (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)
        x = hidden_state.squeeze(0)
        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if return_last_dense and layer_number == len(self.dense_layers) - 2:
                output = x
        if return_last_dense:
            return x, output
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2dSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_gabrieltseng_pycrop_yield_prediction(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

