import sys
_module = sys.modules[__name__]
del sys
conf = _module
train_net = _module
train_net_pytorchexperiment = _module
setup = _module
test = _module
test_experimentlogger = _module
test_numpyseabornimageplotlogger = _module
test_pytorchexperiment = _module
test_pytorchexperimentlogger = _module
test_pytorchtensorboardxlogger = _module
test_pytorchvisdomlogger = _module
trixi = _module
browser = _module
experiment = _module
pytorchexperiment = _module
experiment_browser = _module
dataprocessing = _module
experimentreader = _module
logger = _module
abstractlogger = _module
combinedlogger = _module
experimentlogger = _module
pytorchexperimentlogger = _module
file = _module
numpyplotfilelogger = _module
pytorchplotfilelogger = _module
textfilelogger = _module
message = _module
slackmessagelogger = _module
telegrammessagelogger = _module
plt = _module
numpyseabornimageplotlogger = _module
numpyseabornplotlogger = _module
tensorboard = _module
pytorchtensorboardlogger = _module
tensorboardlogger = _module
visdom = _module
numpyvisdomlogger = _module
pytorchvisdomlogger = _module
util = _module
config = _module
extravisdom = _module
gridsearch = _module
metrics = _module
pytorchexperimentstub = _module
pytorchutils = _module
sourcepacker = _module
util = _module

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


import re


import torch


import torchvision


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


from torchvision.transforms import transforms


import time


import numpy as np


from scipy import misc


import random


import string


import warnings


from abc import ABCMeta


from abc import abstractmethod


from functools import wraps


from torchvision.utils import save_image as tv_save_image


from collections import defaultdict


from torch.utils.tensorboard import SummaryWriter


from torchvision.utils import make_grid


from functools import lru_cache


import logging


import math


from collections import deque


from types import FunctionType


from types import ModuleType


class Net(nn.Module):
    """
    Small network to test save/load functionality
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

