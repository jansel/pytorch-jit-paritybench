import sys
_module = sys.modules[__name__]
del sys
shufflenet_v2 = _module
slim = _module

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


import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        channels = out_channels // 2
        if stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(slim.conv_bn_relu(name + '/conv1', channels, channels, 1), slim.conv_bn(name + '/conv2', channels, channels, 3, stride=stride, dilation=dilation, padding=dilation, groups=channels), slim.conv_bn_relu(name + '/conv3', channels, channels, 1))
        else:
            self.conv = nn.Sequential(slim.conv_bn_relu(name + '/conv1', in_channels, channels, 1), slim.conv_bn(name + '/conv2', channels, channels, 3, stride=stride, dilation=dilation, padding=dilation, groups=channels), slim.conv_bn_relu(name + '/conv3', channels, channels, 1))
            self.conv0 = nn.Sequential(slim.conv_bn(name + '/conv4', in_channels, in_channels, 3, stride=stride, dilation=dilation, padding=dilation, groups=in_channels), slim.conv_bn_relu(name + '/conv5', in_channels, channels, 1))
        self.shuffle = slim.channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :x.shape[1] // 2, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)

    def generate_caffe_prototxt(self, caffe_net, layer):
        if self.stride == 1:
            layer_x1, layer_x2 = L.Slice(layer, ntop=2, axis=1, slice_point=[self.in_channels // 2])
            caffe_net[self.g_name + '/slice1'] = layer_x1
            caffe_net[self.g_name + '/slice2'] = layer_x2
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer_x2)
        else:
            layer_x1 = slim.generate_caffe_prototxt(self.conv0, caffe_net, layer)
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer)
        layer = L.Concat(layer_x1, layer_x2, axis=1)
        caffe_net[self.g_name + '/concat'] = layer
        layer = slim.generate_caffe_prototxt(self.shuffle, caffe_net, layer)
        return layer


def g_name(g_name, m):
    m.g_name = g_name
    return m


class Network(nn.Module):

    def __init__(self, num_classes, width_multiplier):
        super(Network, self).__init__()
        width_config = {(0.25): (24, 48, 96, 512), (0.33): (32, 64, 128, 512), (0.5): (48, 96, 192, 1024), (1.0): (116, 232, 464, 1024), (1.5): (176, 352, 704, 1024), (2.0): (244, 488, 976, 2048)}
        width_config = width_config[width_multiplier]
        self.num_classes = num_classes
        in_channels = 24
        self.network_config = [g_name('data/bn', nn.BatchNorm2d(3)), slim.conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1), g_name('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)), (width_config[0], 2, 1, 4, 'b'), (width_config[1], 2, 1, 8, 'b'), (width_config[2], 2, 1, 4, 'b'), slim.conv_bn_relu('conv5', width_config[2], width_config[3], 1), g_name('pool', nn.AvgPool2d(7, 1)), g_name('fc', nn.Conv2d(width_config[3], self.num_classes, 1))]
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels, out_channels, stride, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1), out_channels, out_channels, 1, dilation))
            self.network += [nn.Sequential(*blocks)]
            in_channels = out_channels
        self.network = nn.Sequential(*self.network)
        for name, m in self.named_modules():
            if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def trainable_parameters(self):
        parameters = [{'params': self.cls_head_list.parameters(), 'lr_mult': 1.0}, {'params': self.loc_head_list.parameters(), 'lr_mult': 1.0}]
        for i in range(len(self.network)):
            lr_mult = 0.1 if i in (0, 1, 2, 3, 4, 5) else 1
            parameters.append({'params': self.network[i].parameters(), 'lr_mult': lr_mult})
        return parameters

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)

    def generate_caffe_prototxt(self, caffe_net, layer):
        data_layer = layer
        network = slim.generate_caffe_prototxt(self.network, caffe_net, data_layer)
        return network

    def convert_to_caffe(self, name):
        caffe_net = caffe.NetSpec()
        layer = L.Input(shape=dict(dim=[1, 3, args.image_hw, args.image_hw]))
        caffe_net.tops['data'] = layer
        slim.generate_caffe_prototxt(self, caffe_net, layer)
        None
        with open(name + '.prototxt', 'wb') as f:
            f.write(str(caffe_net.to_proto()).encode())
        caffe_net = caffe.Net(name + '.prototxt', caffe.TEST)
        slim.convert_pytorch_to_caffe(self, caffe_net)
        caffe_net.save(name + '.caffemodel')


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.ShuffleChannel(layer, group=self.groups)
        caffe_net[self.g_name] = layer
        return layer


class Permute(nn.Module):

    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x):
        x = x.permute(*self.order).contiguous()
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Permute(layer, order=list(self.order))
        caffe_net[self.g_name] = layer
        return layer


class Flatten(nn.Module):

    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        assert self.axis == 1
        x = x.reshape(x.shape[0], -1)
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Flatten(layer, axis=self.axis)
        caffe_net[self.g_name] = layer
        return layer


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_miaow1988_ShuffleNet_V2_pytorch_caffe(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

