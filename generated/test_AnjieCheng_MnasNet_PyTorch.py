import sys
_module = sys.modules[__name__]
del sys
MnasNet = _module

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


from torch.autograd import Variable


import torch.nn as nn


import torch


import math


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 
            0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(
            inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio,
            kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=
            False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=
            True), nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def Conv_1x1(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def SepConv_3x3(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=
        False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp,
        oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class MnasNet(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MnasNet, self).__init__()
        self.interverted_residual_setting = [[3, 24, 3, 2, 3], [3, 40, 3, 2,
            5], [6, 80, 3, 2, 5], [6, 96, 2, 1, 3], [6, 192, 4, 2, 5], [6, 
            320, 1, 1, 3]]
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult
            ) if width_mult > 1.0 else 1280
        self.features = [Conv_3x3(3, input_channel, 2), SepConv_3x3(
            input_channel, 16)]
        input_channel = 16
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel,
                        output_channel, s, t, k))
                else:
                    self.features.append(InvertedResidual(input_channel,
                        output_channel, 1, t, k))
                input_channel = output_channel
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(self.
            last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_AnjieCheng_MnasNet_PyTorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(MnasNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
