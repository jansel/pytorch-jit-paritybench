import sys
_module = sys.modules[__name__]
del sys
client = _module
config = _module
load_data = _module
fl_model = _module
fl_model = _module
fl_model = _module
fl_model = _module
run = _module
analyze_logs = _module
pca = _module
server = _module
accavg = _module
directed = _module
kcenter = _module
kmeans = _module
magavg = _module
server = _module
dists = _module

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


import logging


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


import numpy as np


import random


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

