import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
model = _module
psnrmeter = _module
test_image = _module
test_video = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch.nn.functional as F


import torch


import torch.optim as optim


import torchvision.transforms as transforms


from torch.autograd import Variable


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * upscale_factor ** 2, (3, 3), (1, 1),
            (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leftthomas_ESPCN(_paritybench_base):
    pass
    def test_000(self):
        self._check(Net(*[], **{'upscale_factor': 4}), [torch.rand([4, 1, 64, 64])], {})

