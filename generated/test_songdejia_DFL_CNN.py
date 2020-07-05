import sys
_module = sys.modules[__name__]
del sys
drawrect = _module
main = _module
DFL = _module
train = _module
MyImageFolderWithPaths = _module
init = _module
save = _module
transform = _module
util = _module
validate = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import random


import time


import warnings


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import re


import numpy as np


import torch.nn.functional as F


import torchvision


from torchvision import datasets


from torchvision import transforms


from torchvision import utils


import torchvision.models as models


from torch.nn import init


class DFL_VGG16(nn.Module):

    def __init__(self, k=10, nclass=200):
        super(DFL_VGG16, self).__init__()
        self.k = k
        self.nclass = nclass
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
        conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
        conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
        conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size=1, stride=1, padding=0)
        pool6 = torch.nn.MaxPool2d((56, 56), stride=(56, 56), return_indices=True)
        self.conv1_conv4 = conv1_conv4
        self.conv5 = conv5
        self.cls5 = nn.Sequential(nn.Conv2d(512, 200, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(200), nn.ReLU(True), nn.AdaptiveAvgPool2d((1, 1)))
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(nn.Conv2d(k * nclass, nclass, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d((1, 1)))
        self.cross_channel_pool = nn.AvgPool1d(kernel_size=k, stride=k, padding=0)

    def forward(self, x):
        batchsize = x.size(0)
        inter4 = self.conv1_conv4(x)
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)
        return out1, out2, out3, indices


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DFL_VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     True),
]

class Test_songdejia_DFL_CNN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

