import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
hyperband = _module
main = _module
model = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from torchvision import datasets


from torchvision import transforms


from torch.utils.data.sampler import SubsetRandomSampler


import uuid


import torch.nn as nn


import torch.nn.functional as F


from torch.optim import SGD


from torch.optim import Adam


from torch.autograd import Variable


import time


from numpy.random import uniform


from numpy.random import normal


from numpy.random import randint


from numpy.random import choice


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape, -1)

