import sys
_module = sys.modules[__name__]
del sys
data = _module
load_data = _module
LPRNet = _module
model = _module
test_LPRNet = _module
train_LPRNet = _module

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


from torch.utils.data import *


import numpy as np


import random


import torch.nn as nn


import torch


from torch.autograd import Variable


import torch.nn.functional as F


from torch import optim


import time


class small_basic_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(ch_in, ch_out // 4, kernel_size=1), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):

    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), nn.BatchNorm2d(num_features=64), nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)), small_basic_block(ch_in=64, ch_out=128), nn.BatchNorm2d(num_features=128), nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)), small_basic_block(ch_in=64, ch_out=256), nn.BatchNorm2d(num_features=256), nn.ReLU(), small_basic_block(ch_in=256, ch_out=256), nn.BatchNorm2d(num_features=256), nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)), nn.Dropout(dropout_rate), nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1), nn.BatchNorm2d(num_features=256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), nn.BatchNorm2d(num_features=class_num), nn.ReLU())
        self.container = nn.Sequential(nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)
        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (small_basic_block,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sirius_ai_LPRNet_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

