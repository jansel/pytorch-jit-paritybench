import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
model = _module
model_arch = _module
tests = _module
model_test = _module
net_arch_test = _module
test_case = _module
trainer = _module
test_model = _module
train_model = _module
utils = _module
writer = _module

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


from enum import Enum


from enum import auto


import torch


import torchvision


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


from collections import OrderedDict


import torch.nn


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.nn as nn


import torch.nn.functional as F


import itertools


import random


import torch.distributed as dist


import torch.multiprocessing as mp


import logging


import numpy as np


from torch.utils.tensorboard import SummaryWriter


class Net_arch(nn.Module):

    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, 3, 2, 1), self.lrelu)
        self.conv2 = nn.Sequential(nn.Conv2d(4, 4, 3, 2, 1), self.lrelu)
        self.fc = nn.Linear(7 * 7 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

