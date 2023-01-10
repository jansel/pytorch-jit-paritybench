import sys
_module = sys.modules[__name__]
del sys
post_gen_project = _module
pre_gen_project = _module
setup = _module
__init__ = _module
data = _module
datamodule = _module
dataset = _module
modules = _module
module = _module
pl_modules = _module
ui = _module
tests = _module
test_nn_core_integration = _module
test_seeding = _module
test_storage = _module
test_training = _module

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


from functools import cached_property


from functools import partial


from typing import List


from typing import Mapping


from typing import Optional


from typing import Sequence


from typing import Union


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torch.utils.data.dataloader import default_collate


from torchvision import transforms


from torchvision.datasets import FashionMNIST


from torch import nn


class CNN(nn.Module):

    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), nn.SiLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(16, 32, 5, 1, 2), nn.SiLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential()
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

