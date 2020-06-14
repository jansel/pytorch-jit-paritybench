import sys
_module = sys.modules[__name__]
del sys
cifar100data = _module
cifar10data = _module
layers = _module
main = _module
model = _module
train = _module
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


import torch.nn as nn


import torch.optim


from torch.autograd import Variable


from torch.optim.rmsprop import RMSprop


class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor=6,
        kernel_size=3, stride=2):
        super(InvertedResidualBlock, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError('Stride should be 1 or 2')
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels *
            expansion_factor, 1, bias=False), nn.BatchNorm2d(in_channels *
            expansion_factor), nn.ReLU6(inplace=True), nn.Conv2d(
            in_channels * expansion_factor, in_channels * expansion_factor,
            kernel_size, stride, 1, groups=in_channels * expansion_factor,
            bias=False), nn.BatchNorm2d(in_channels * expansion_factor), nn
            .ReLU6(inplace=True), nn.Conv2d(in_channels * expansion_factor,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block


def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2,
    dropout_prob=0.0):
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
        stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.
        Dropout2d(dropout_prob, inplace=True), nn.ReLU6(inplace=True))


def inverted_residual_sequence(in_channels, out_channels, num_units,
    expansion_factor=6, kernel_size=3, initial_stride=2):
    bottleneck_arr = [InvertedResidualBlock(in_channels, out_channels,
        expansion_factor, kernel_size, initial_stride)]
    for i in range(num_units - 1):
        bottleneck_arr.append(InvertedResidualBlock(out_channels,
            out_channels, expansion_factor, kernel_size, 1))
    return bottleneck_arr


class MobileNetV2(nn.Module):

    def __init__(self, args):
        super(MobileNetV2, self).__init__()
        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1
        self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1}, {'t':
            1, 'c': 16, 'n': 1, 's': 1}, {'t': 6, 'c': 24, 'n': 2, 's': s2},
            {'t': 6, 'c': 32, 'n': 3, 's': 2}, {'t': 6, 'c': 64, 'n': 4,
            's': 2}, {'t': 6, 'c': 96, 'n': 3, 's': 1}, {'t': 6, 'c': 160,
            'n': 3, 's': 2}, {'t': 6, 'c': 320, 'n': 1, 's': 1}, {'t': None,
            'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = args.num_classes
        self.network = [conv2d_bn_relu6(args.num_channels, int(self.
            network_settings[0]['c'] * args.width_multiplier), args.
            kernel_size, self.network_settings[0]['s'], args.dropout_prob)]
        for i in range(1, 8):
            self.network.extend(inverted_residual_sequence(int(self.
                network_settings[i - 1]['c'] * args.width_multiplier), int(
                self.network_settings[i]['c'] * args.width_multiplier),
                self.network_settings[i]['n'], self.network_settings[i]['t'
                ], args.kernel_size, self.network_settings[i]['s']))
        self.network.append(conv2d_bn_relu6(int(self.network_settings[7][
            'c'] * args.width_multiplier), int(self.network_settings[8]['c'
            ] * args.width_multiplier), 1, self.network_settings[8]['s'],
            args.dropout_prob))
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(nn.AvgPool2d((args.img_height // args.
            downsampling, args.img_width // args.downsampling)))
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(nn.Conv2d(int(self.network_settings[8]['c'] *
            args.width_multiplier), self.num_classes, 1, bias=True))
        self.network = nn.Sequential(*self.network)
        self.initialize()

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, self.num_classes)
        return x

    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MG2033_MobileNet_V2(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(InvertedResidualBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

