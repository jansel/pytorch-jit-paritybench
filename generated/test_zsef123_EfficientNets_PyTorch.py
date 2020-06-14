import sys
_module = sys.modules[__name__]
del sys
convert = _module
loader = _module
logger = _module
main = _module
effnet = _module
layers = _module
runner = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import CosineAnnealingLR


import math


import torch.nn.functional as F


import copy


import time


def conv_bn_act(in_, out_, kernel_size, stride=1, groups=1, bias=True, eps=
    0.001, momentum=0.01):
    return nn.Sequential(SamePadConv2d(in_, out_, kernel_size, stride,
        groups=groups, bias=bias), nn.BatchNorm2d(out_, eps, momentum), Swish()
        )


class MBConv(nn.Module):

    def __init__(self, in_, out_, expand, kernel_size, stride, skip,
        se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False
            ) if expand != 1 else nn.Identity()
        self.depth_wise_conv = conv_bn_act(mid_, mid_, kernel_size=
            kernel_size, stride=stride, groups=mid_, bias=False)
        self.se = SEModule(mid_, int(in_ * se_ratio)
            ) if se_ratio > 0 else nn.Identity()
        self.project_conv = nn.Sequential(SamePadConv2d(mid_, out_,
            kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_, 
            0.001, 0.01))
        self.skip = skip and stride == 1 and in_ == out_
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):

    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip,
        se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio,
            drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip,
                se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):

    def __init__(self, width_coeff, depth_coeff, depth_div=8, min_depth=
        None, dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000):
        super().__init__()
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x
            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div *
                depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))
        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2,
            bias=False)
        self.blocks = nn.Sequential(MBBlock(renew_ch(32), renew_ch(16), 1, 
            3, 1, renew_repeat(1), True, 0.25, drop_connect_rate), MBBlock(
            renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 
            0.25, drop_connect_rate), MBBlock(renew_ch(24), renew_ch(40), 6,
            5, 2, renew_repeat(2), True, 0.25, drop_connect_rate), MBBlock(
            renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 
            0.25, drop_connect_rate), MBBlock(renew_ch(80), renew_ch(112), 
            6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4),
            True, 0.25, drop_connect_rate), MBBlock(renew_ch(192), renew_ch
            (320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate))
        self.head = nn.Sequential(*conv_bn_act(renew_ch(320), renew_ch(1280
            ), kernel_size=1, bias=False), nn.AdaptiveAvgPool2d(1), nn.
            Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.
            Identity(), Flatten(), nn.Linear(renew_ch(1280), num_classes))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        head = self.head(x)
        return head


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride +
            effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) *
            dilation + 1 - in_)
        rows_odd = padding_rows % 2 != 0
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.
            shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.
            shape[3], self.stride[1], self.dilation[1])
        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])
        return F.conv2d(x, self.weight, self.bias, self.stride, padding=(
            padding_rows // 2, padding_cols // 2), dilation=self.dilation,
            groups=self.groups)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):

    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_,
            squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(), nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


class DropConnect(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x
        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.
            float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zsef123_EfficientNets_PyTorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(DropConnect(*[], **{'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(EfficientNet(*[], **{'width_coeff': 4, 'depth_coeff': 1}), [torch.rand([4, 3, 4, 4])], {})

    def test_002(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MBBlock(*[], **{'in_': 4, 'out_': 4, 'expand': 4, 'kernel': 4, 'stride': 1, 'num_repeat': 4, 'skip': 4, 'se_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MBConv(*[], **{'in_': 4, 'out_': 4, 'expand': 4, 'kernel_size': 4, 'stride': 1, 'skip': 4, 'se_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SEModule(*[], **{'in_': 4, 'squeeze_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(SamePadConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

