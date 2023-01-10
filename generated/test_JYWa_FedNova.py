import sys
_module = sys.modules[__name__]
del sys
comm_helpers = _module
FedNova = _module
FedProx = _module
distoptim = _module
models = _module
vgg = _module
train_LocalSGD = _module
util_v4 = _module

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


import collections


import logging


import math


import copy


import torch


import torch.distributed as dist


import functools


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import numpy as np


import torch.nn as nn


import torch.nn.init as init


import time


from math import ceil


from random import Random


import torch.utils.data.distributed


import torch.nn.functional as F


from torch.multiprocessing import Process


import torchvision


from torchvision import datasets


from torchvision import transforms


import torch.backends.cudnn as cudnn


import torchvision.models as models


import torch.optim as optim


from torch.autograd import Variable


import torchvision.models as IMG_models


class VGG(nn.Module):
    """
    VGG model 
    """

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

