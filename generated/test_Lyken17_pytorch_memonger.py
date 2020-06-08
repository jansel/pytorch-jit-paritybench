import sys
_module = sys.modules[__name__]
del sys
main = _module
memonger = _module
checkpoint = _module
memonger = _module
resnet = _module

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


from torch.utils import checkpoint


import warnings


from math import sqrt


from math import log


from collections import OrderedDict


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.functional as F


def reforwad_momentum_fix(origin_momentum):
    return 1 - sqrt(1 - origin_momentum)


class SublinearSequential(nn.Sequential):

    def __init__(self, *args):
        super(SublinearSequential, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            None
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            None
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):

        def run_function(start, end, functions):

            def forward(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward
        functions = list(self.children())
        segments = int(sqrt(len(functions)))
        segment_size = len(functions) // segments
        end = -1
        if not isinstance(input, tuple):
            inputs = input,
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = inputs,
        output = checkpoint(run_function(end + 1, len(functions) - 1,
            functions), *inputs)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SublinearSequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Lyken17_pytorch_memonger(_paritybench_base):
    pass

    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
