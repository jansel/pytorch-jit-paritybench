import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
grid_sample = _module
mnist_make_gif = _module
mnist_model = _module
mnist_plot_curve = _module
mnist_train = _module
mnist_visualize = _module
single_visualize = _module
tps_grid_gen = _module

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


import random


from torchvision import datasets


from torchvision import transforms


import torch.nn.functional as F


from torch.autograd import Variable


import math


import itertools


import numpy as np


import torch.nn as nn


import torch.optim as optim


import time


from torch.autograd import Function


class CNN(nn.Module):

    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)
        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)
        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, (0)] + pairwise_diff_square[:, :, (1)]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, (-3)].fill_(1)
        forward_kernel[(-3), :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], dim=1)
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


class STNClsNet(nn.Module):

    def __init__(self, args):
        super(STNClsNet, self).__init__()
        self.args = args
        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1
        target_control_points = torch.Tensor(list(itertools.product(np.arange(-r1, r1 + 1e-05, 2.0 * r1 / (args.grid_height - 1)), np.arange(-r2, r2 + 1e-05, 2.0 * r2 / (args.grid_width - 1)))))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        GridLocNet = {'unbounded_stn': UnBoundedGridLocNet, 'bounded_stn': BoundedGridLocNet}[args.model]
        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points)
        self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points)
        self.cls_net = ClsNet()

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.args.image_height, self.args.image_width, 2)
        transformed_x = grid_sample(x, grid)
        logit = self.cls_net(transformed_x)
        return logit

