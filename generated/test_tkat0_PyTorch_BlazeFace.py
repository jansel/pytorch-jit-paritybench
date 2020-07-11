import sys
_module = sys.modules[__name__]
del sys
blazeface = _module
mediapipe_implementation = _module
model = _module
setup = _module
tests = _module
test_blazeface = _module

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


from torch import nn


import torch.nn.functional as F


import torch


class BlazeBlock(nn.Module):

    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_double_block = oup2 is not None
        self.use_pooling = self.stride != 1
        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True), nn.BatchNorm2d(inp), nn.Conv2d(inp, oup1, 1, 1, 0, bias=True), nn.BatchNorm2d(oup1))
        self.act = nn.ReLU(inplace=True)
        if self.use_double_block:
            self.conv2 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True), nn.BatchNorm2d(oup1), nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True), nn.BatchNorm2d(oup2))
        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)


def initialize(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


class MediaPipeBlazeFace(nn.Module):
    """Constructs a BlazeFace model of the MediaPipe implementation

    the original implementation
    https://github.com/google/mediapipe/tree/master/mediapipe/models#blazeface-face-detection-model
    """

    def __init__(self):
        super(MediaPipeBlazeFace, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=True), nn.BatchNorm2d(24), nn.ReLU(inplace=True), BlazeBlock(24, 24, kernel_size=3), BlazeBlock(24, 28, kernel_size=3), BlazeBlock(28, 32, kernel_size=3, stride=2), BlazeBlock(32, 36, kernel_size=3), BlazeBlock(36, 42, kernel_size=3), BlazeBlock(42, 48, kernel_size=3, stride=2), BlazeBlock(48, 56, kernel_size=3), BlazeBlock(56, 64, kernel_size=3), BlazeBlock(64, 72, kernel_size=3), BlazeBlock(72, 80, kernel_size=3), BlazeBlock(80, 88, kernel_size=3), BlazeBlock(88, 96, kernel_size=3, stride=2), BlazeBlock(96, 96, kernel_size=3), BlazeBlock(96, 96, kernel_size=3), BlazeBlock(96, 96, kernel_size=3), BlazeBlock(96, 96, kernel_size=3))
        self.apply(initialize)

    def forward(self, x):
        h = self.features(x)
        return h


class BlazeFace(nn.Module):
    """Constructs a BlazeFace model

    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self):
        super(BlazeFace, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=True), nn.BatchNorm2d(24), nn.ReLU(inplace=True), BlazeBlock(24, 24), BlazeBlock(24, 24), BlazeBlock(24, 48, stride=2), BlazeBlock(48, 48), BlazeBlock(48, 48), BlazeBlock(48, 24, 96, stride=2), BlazeBlock(96, 24, 96), BlazeBlock(96, 24, 96), BlazeBlock(96, 24, 96, stride=2), BlazeBlock(96, 24, 96), BlazeBlock(96, 24, 96))
        self.apply(initialize)

    def forward(self, x):
        h = self.features(x)
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BlazeBlock,
     lambda: ([], {'inp': 4, 'oup1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BlazeFace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MediaPipeBlazeFace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_tkat0_PyTorch_BlazeFace(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

