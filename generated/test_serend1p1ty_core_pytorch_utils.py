import sys
_module = sys.modules[__name__]
del sys
cpu = _module
config_parser = _module
distributed = _module
history_buffer = _module
hooks = _module
checkpoint_hook = _module
distributed_hook = _module
eval_hook = _module
hookbase = _module
logger_hook = _module
lr_update_hook = _module
logger = _module
lr_scheduler = _module
misc = _module
trainer = _module
conf = _module
inference_hook = _module
train_minist = _module
train_minist_dist = _module
setup = _module
tests = _module
test_config_parser = _module
test_history_buffer = _module
test_lr_scheduler = _module
test_minist_training = _module
test_trainer = _module

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


import functools


import logging


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import torch


import torch.distributed as dist


from torch import Tensor


from torch._C._distributed_c10d import ProcessGroup


import time


from torch.utils.tensorboard import SummaryWriter


from typing import Union


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


import random


from collections import defaultdict


import numpy as np


import torch.nn as nn


import torch.optim as optim


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.nn.parallel import DistributedDataParallel


from torch.nn.utils import clip_grad_norm_


from torch.utils.data import DataLoader


import matplotlib.pyplot as plt


import torch.nn.functional as F


from torch.optim import Adadelta


from torch.optim.lr_scheduler import StepLR


from torchvision import datasets


from torchvision import transforms


from torch.utils.data.distributed import DistributedSampler


import math


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import MultiStepLR


import re


from torch import nn


class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.device = device
        self

    def forward(self, data):
        img, target = data
        img = img
        target = target
        x = self.conv1(img)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        if self.training:
            loss = F.nll_loss(output, target)
            return loss
        return output


class _SimpleModel(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.device = device
        self

    def forward(self, data):
        x, y = data
        x = x
        y = y
        return F.mse_loss(self.fc(x), y)

