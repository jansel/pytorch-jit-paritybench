import sys
_module = sys.modules[__name__]
del sys
inference = _module
train = _module
zalo_utils = _module

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


from sklearn.model_selection import train_test_split


import torch


import torch.nn as nn


import copy


import random


from torchvision import transforms


import torch.backends.cudnn as cudnn


from time import time


from time import strftime


from torch.autograd import Variable


from torch.utils.data import Dataset


import torchvision.models as models


import torch.optim as optim


import numpy as np


class MyResNet(nn.Module):

    def __init__(self, depth, num_classes, pretrained=True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)
        self.num_ftrs = model.fc.in_features
        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        None
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                None
                for param in child.parameters():
                    param.requires_grad = False
            else:
                None
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

