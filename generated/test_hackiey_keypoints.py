import sys
_module = sys.modules[__name__]
del sys
analysis = _module
config = _module
src = _module
dataset = _module
model = _module
prediction = _module
train = _module

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


from torchvision import transforms


from torch.utils.data import DataLoader


import matplotlib.pyplot as plt


import numpy as np


from torch.utils.data import Dataset


from torchvision import utils


import torchvision


import torch.nn as nn


from torch.autograd import Variable


import torch.optim as optim


class Keypoints(nn.Module):

    def __init__(self, num_classes, img_height=353, img_width=257, resnet=18):
        super(Keypoints, self).__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes * 3
        self.img_height = img_height
        self.img_width = img_width
        if resnet == 18:
            self.resnet = torchvision.models.resnet18()
            self.conv1by1 = nn.Conv2d(512, self.num_outputs, (1, 1))
        elif resnet == 101:
            self.resnet = torchvision.models.resnet101()
            self.conv1by1 = nn.Conv2d(2048, self.num_outputs, (1, 1))
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet
        self.conv_transpose = nn.ConvTranspose2d(self.num_outputs, self.num_outputs, kernel_size=32, stride=8)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1by1(x)
        x = self.conv_transpose(x)
        output = nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear')(x)
        maps = self.sigmoid(output[:, :self.num_classes, :, :])
        offsets_x = output[:, self.num_classes:2 * self.num_classes, :, :]
        offsets_y = output[:, 2 * self.num_classes:3 * self.num_classes, :, :]
        maps_pred = self.sigmoid(x[:, :self.num_classes, :, :])
        offsets_x_pred = x[:, self.num_classes:2 * self.num_classes, :, :]
        offsets_y_pred = x[:, 2 * self.num_classes:3 * self.num_classes, :, :]
        return (maps, offsets_x, offsets_y), (maps_pred, offsets_x_pred, offsets_y_pred)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Keypoints,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_hackiey_keypoints(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

