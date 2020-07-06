import sys
_module = sys.modules[__name__]
del sys
main_tensorflow = _module
config = _module
data_preprocess = _module
main_pytorch = _module
network = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from sklearn.preprocessing import StandardScaler


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2), stride=2))
        self.fc1 = nn.Sequential(nn.Linear(in_features=64 * 26, out_features=1000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=500), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(in_features=500, out_features=6))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(-1, 64 * 26)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out

