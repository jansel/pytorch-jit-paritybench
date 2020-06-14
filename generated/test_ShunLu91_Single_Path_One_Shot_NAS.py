import sys
_module = sys.modules[__name__]
del sys
block = _module
choice_model = _module
config = _module
model = _module
random_search = _module
supernet = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


class Choice_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True
        ):
        super(Choice_Block, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels
        self.cb_main = nn.Sequential(nn.Conv2d(self.in_channels, self.
            mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine), nn.ReLU(
            inplace=True), nn.Conv2d(self.mid_channels, self.mid_channels,
            kernel_size=kernel, stride=stride, padding=padding, bias=False,
            groups=self.mid_channels), nn.BatchNorm2d(self.mid_channels,
            affine=self.affine), nn.Conv2d(self.mid_channels, self.
            out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine), nn.ReLU(
            inplace=True))
        if stride == 2:
            self.cb_proj = nn.Sequential(nn.Conv2d(self.in_channels, self.
                in_channels, kernel_size=kernel, stride=2, padding=padding,
                bias=False, groups=self.in_channels), nn.BatchNorm2d(self.
                in_channels, affine=self.affine), nn.Conv2d(self.
                in_channels, self.in_channels, kernel_size=1, stride=1,
                padding=0, bias=False), nn.BatchNorm2d(self.in_channels,
                affine=self.affine), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class Choice_Block_x(nn.Module):

    def __init__(self, in_channels, out_channels, stride, supernet=True):
        super(Choice_Block_x, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels
        self.cb_main = nn.Sequential(nn.Conv2d(self.in_channels, self.
            in_channels, kernel_size=3, stride=stride, padding=1, bias=
            False, groups=self.in_channels), nn.BatchNorm2d(self.
            in_channels, affine=self.affine), nn.Conv2d(self.in_channels,
            self.mid_channels, kernel_size=1, stride=1, padding=0, bias=
            False), nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True), nn.Conv2d(self.mid_channels, self.
            mid_channels, kernel_size=3, stride=1, padding=1, bias=False,
            groups=self.mid_channels), nn.BatchNorm2d(self.mid_channels,
            affine=self.affine), nn.Conv2d(self.mid_channels, self.
            mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine), nn.ReLU(
            inplace=True), nn.Conv2d(self.mid_channels, self.mid_channels,
            kernel_size=3, stride=1, padding=1, bias=False, groups=self.
            mid_channels), nn.BatchNorm2d(self.mid_channels, affine=self.
            affine), nn.Conv2d(self.mid_channels, self.out_channels,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (self.out_channels, affine=self.affine), nn.ReLU(inplace=True))
        if stride == 2:
            self.cb_proj = nn.Sequential(nn.Conv2d(self.in_channels, self.
                in_channels, kernel_size=3, stride=2, padding=1, groups=
                self.in_channels, bias=False), nn.BatchNorm2d(self.
                in_channels, affine=self.affine), nn.Conv2d(self.
                in_channels, self.in_channels, kernel_size=1, stride=1,
                padding=0, bias=False), nn.BatchNorm2d(self.in_channels,
                affine=self.affine), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


last_channel = 1024


channel = [16, 64, 64, 64, 64, 160, 160, 160, 160, 320, 320, 320, 320, 320,
    320, 320, 320, 640, 640, 640, 640]


class SinglePath_OneShot(nn.Module):

    def __init__(self, dataset, resize, classes, layers):
        super(SinglePath_OneShot, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']
        self.stem = nn.Sequential(nn.Conv2d(3, channel[0], kernel_size=3,
            stride=first_stride, padding=1, bias=False), nn.BatchNorm2d(
            channel[0], affine=False), nn.ReLU6(inplace=True))
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Block_x(inp, oup, stride=stride))
                else:
                    layer_cb.append(Choice_Block(inp, oup, kernel=j, stride
                        =stride))
            self.choice_block.append(layer_cb)
        self.last_conv = nn.Sequential(nn.Conv2d(channel[-1], last_channel,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (last_channel, affine=False), nn.ReLU6(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x, choice=np.random.randint(4, size=20)):
        x = self.stem(x)
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SinglePath_Network(nn.Module):

    def __init__(self, dataset, resize, classes, layers, choice):
        super(SinglePath_Network, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']
        self.stem = nn.Sequential(nn.Conv2d(3, channel[0], kernel_size=3,
            stride=first_stride, padding=1, bias=False), nn.BatchNorm2d(
            channel[0]), nn.ReLU6(inplace=True))
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            if choice[i] == 3:
                self.choice_block.append(Choice_Block_x(inp, oup, stride=
                    stride, supernet=False))
            else:
                self.choice_block.append(Choice_Block(inp, oup, kernel=self
                    .kernel_list[choice[i]], stride=stride, supernet=False))
        self.last_conv = nn.Sequential(nn.Conv2d(channel[-1], last_channel,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (last_channel), nn.ReLU6(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.layers):
            x = self.choice_block[i](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ShunLu91_Single_Path_One_Shot_NAS(_paritybench_base):
    pass
