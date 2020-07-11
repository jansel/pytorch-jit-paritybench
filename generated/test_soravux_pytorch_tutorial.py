import sys
_module = sys.modules[__name__]
del sys
example1 = _module
example2_adv_example = _module
example2_gradient = _module
example3 = _module
example4 = _module
example5 = _module
example6 = _module
example6_features = _module
example6_gradient = _module
example6_squeezenet = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from matplotlib import pyplot as plt


import numpy as np


import random


import torch.utils.data as data


import torchvision.models as models


from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(512, 2, kernel_size=1)
        self.avgpool = nn.AvgPool2d(13)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = self.conv(x)
        x = self.avgpool(x)
        x = F.log_softmax(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
]

class Test_soravux_pytorch_tutorial(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

