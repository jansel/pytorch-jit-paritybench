import sys
_module = sys.modules[__name__]
del sys
alex = _module
inference = _module
googlenet = _module
inference = _module
inception = _module
inference = _module
inference = _module
lenet5 = _module
inference = _module
mlp = _module
inference = _module
mnasnet = _module
inference = _module
mobilenetv2 = _module
mobilenetv3 = _module
v3 = _module
bbox = _module
cam_demo = _module
darknet = _module
detect = _module
inference = _module
preprocess = _module
util = _module
video_demo = _module
video_demo_half = _module
inference = _module
resnet18 = _module
inference = _module
resnet34 = _module
inference = _module
resnet50 = _module
inference = _module
resnext50 = _module
inference = _module
shufflenet = _module
inference = _module
squeezenet = _module
inference = _module
vgg = _module

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


from torch import nn


from torch.nn import functional as F


import torchvision


from torch.autograd import Variable


import random


import numpy as np


import time


import torch.nn as nn


import pandas as pd


import torch.nn.functional as F


import matplotlib.pyplot as plt


import itertools


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        None
        x = F.relu(self.conv1(x))
        None
        x = self.pool1(x)
        None
        x = F.relu(self.conv2(x))
        None
        x = self.pool1(x)
        None
        x = x.view(x.size(0), -1)
        None
        x = F.relu(self.fc1(x))
        None
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class LinearRegressionModel(torch.nn.Module):
    """
    Linear Regression Model:
        * - input a number
        * - output its double
    """

    def __init__(self):
        """
        Initialize the model
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        Forward function of linear model
        :param x: input
        :return y_pred: predictions
        """
        y_pred = self.linear(x)
        return y_pred


class test_net(nn.Module):

    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


class MaxPoolStride1(nn.Module):

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode='replicate')
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset
        y_offset = y_offset
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(prediction[:, :, 5:5 + num_classes])
    prediction[:, :, :4] *= stride
    return prediction


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


class ReOrgLayer(nn.Module):

    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert H % hs == 0, 'The stride ' + str(self.stride) + ' is not a proper divisor of height ' + str(H)
        assert W % ws == 0, 'The stride ' + str(self.stride) + ' is not a proper divisor of height ' + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    index = 0
    prev_filters = 3
    output_filters = []
    for x in blocks:
        module = nn.Sequential()
        if x['type'] == 'net':
            continue
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{}'.format(index), upsample)
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif x['type'] == 'shortcut':
            from_ = int(x['from'])
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        elif x['type'] == 'maxpool':
            stride = int(x['stride'])
            size = int(x['size'])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module('maxpool_{}'.format(index), maxpool)
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)
        else:
            None
            assert False
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    return net_info, module_list


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        None
        return self.module_list

    def forward(self, x, CUDA=True):
        detections = []
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i in range(len(modules)):
            module_type = modules[i]['type']
            None
            None
            if module_type == 'convolutional' or module_type == 'upsample' or module_type == 'maxpool':
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == 'route':
                layers = modules[i]['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            elif module_type == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(modules[i]['classes'])
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                None
                None
                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i - 1]
        None
        None
        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, savedfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(savedfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)
                else:
                    cpu(conv.bias.data).numpy().tofile(fp)
                cpu(conv.weight.data).numpy().tofile(fp)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MaxPoolStride1,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReOrgLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wang_xinyu_pytorchx(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

