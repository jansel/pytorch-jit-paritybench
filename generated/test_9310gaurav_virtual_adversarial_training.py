import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
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


from torchvision import datasets


from torchvision import transforms


import torch.optim as optim


import torch.nn as nn


import torch


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


class VAT(nn.Module):

    def __init__(self, top_bn=True):
        super(VAT, self).__init__()
        self.top_bn = top_bn
        self.main = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2, 1), nn.Dropout2d(), nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2, 1), nn.Dropout2d(), nn.Conv2d(256, 512, 3, 1, 0, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.1), nn.Conv2d(512, 256, 1, 1, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.Conv2d(256, 128, 1, 1, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(128, 10)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        if self.top_bn:
            output = self.bn(output)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (VAT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_9310gaurav_virtual_adversarial_training(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

