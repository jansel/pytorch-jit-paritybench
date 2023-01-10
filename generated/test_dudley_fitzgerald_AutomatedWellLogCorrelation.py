import sys
_module = sys.modules[__name__]
del sys
master = _module
autoWellCorr = _module
createAutoWellProject = _module
createTrainingData = _module
trainModel = _module
vae = _module

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


from itertools import count


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


from scipy.interpolate import interp1d


from scipy.ndimage import gaussian_filter


from scipy.signal import argrelextrema


from scipy.spatial.distance import cdist


from sklearn.linear_model import RANSACRegressor


import torch


from torch import nn


from torch.autograd import Variable


from torch.optim import Adam


class ConvAEDeep(nn.Module):

    def __init__(self):
        super(ConvAEDeep, self).__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding=1)
        self.dec5 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec4 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = nn.ConvTranspose1d(16, 1, 2, stride=2)

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.pool(x)))
        x = self.relu(self.conv3(self.pool(x)))
        x = self.relu(self.conv4(self.pool(x)))
        return x

    def decode(self, x):
        x = self.pool(x)
        x = self.relu(self.dec4(x))
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec1(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = torch.tanh(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvAEDeep,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     True),
]

class Test_dudley_fitzgerald_AutomatedWellLogCorrelation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

