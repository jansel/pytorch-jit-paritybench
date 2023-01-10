import sys
_module = sys.modules[__name__]
del sys
P10_dataset_transform = _module
P8_Tensorboard = _module
P9_transforms = _module
dataloader = _module
model = _module
model_load = _module
model_pretrained = _module
model_save = _module
nn_conv = _module
nn_conv2d = _module
nn_linear = _module
nn_loss = _module
nn_loss_network = _module
nn_maxpool = _module
nn_module = _module
nn_optim = _module
nn_relu = _module
nn_seq = _module
read_data = _module
test = _module
train = _module
train_gpu_1 = _module
train_gpu_2 = _module

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


import torchvision


from torch.utils.tensorboard import SummaryWriter


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torch


from torch import nn


import torch.nn.functional as F


from torch.nn import Conv2d


from torch.nn import Linear


from torch.nn import L1Loss


from torch.nn import Sequential


from torch.nn import MaxPool2d


from torch.nn import Flatten


from torch.optim.lr_scheduler import StepLR


from torch.nn import ReLU


from torch.nn import Sigmoid


from torchvision.utils import make_grid


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2), nn.MaxPool2d(2), nn.Conv2d(32, 32, 5, 1, 2), nn.MaxPool2d(2), nn.Conv2d(32, 64, 5, 1, 2), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.Linear(64, 10))

    def forward(self, x):
        x = self.model(x)
        return x

