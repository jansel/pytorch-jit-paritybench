import sys
_module = sys.modules[__name__]
del sys
matting = _module
networks = _module
skybox_utils = _module
skyboxengine = _module
skymagic = _module
synrain = _module
train = _module
utils = _module

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


import numpy as np


import matplotlib.pyplot as plt


import torch


import torch.optim as optim


from torch.optim import lr_scheduler


import torch.nn as nn


from torch.nn import init


import functools


from torchvision import models


import math


import random


import torchvision.transforms.functional as TF


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torchvision import transforms


from torchvision import utils


class Identity(nn.Module):

    def forward(self, x):
        return x


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, y_dim, x_dim = input_tensor.size()
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).type_as(input_tensor)
        yy_channel = yy_channel.float() / y_dim
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        ret = torch.cat([input_tensor, yy_channel], dim=1)
        return ret


class CoordConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        in_size = in_channels + 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = AddCoords()(x)
        ret = self.conv(ret)
        return ret


class ResNet50FCN(torch.nn.Module):

    def __init__(self, coordconv=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet50FCN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.coordconv = coordconv
        if coordconv:
            self.conv_in = CoordConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv_fpn1 = CoordConv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = CoordConv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = CoordConv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = CoordConv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = CoordConv2d(64, 1, kernel_size=3, padding=1)
        else:
            self.conv_fpn1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.coordconv:
            x = self.conv_in(x)
        else:
            x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x)
        x_8 = self.resnet.layer2(x_4)
        x_16 = self.resnet.layer3(x_8)
        x_32 = self.resnet.layer4(x_16)
        x = self.upsample(self.relu(self.conv_fpn1(x_32)))
        x = self.upsample(self.relu(self.conv_fpn2(x + x_16)))
        x = self.upsample(self.relu(self.conv_fpn3(x + x_8)))
        x = self.upsample(self.relu(self.conv_fpn4(x + x_4)))
        x = self.upsample(self.relu(self.conv_pred_1(x)))
        x = self.sigmoid(self.conv_pred_2(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet50FCN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_jiupinjia_SkyAR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

