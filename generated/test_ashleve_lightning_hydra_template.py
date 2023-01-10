import sys
_module = sys.modules[__name__]
del sys
setup = _module
src = _module
datamodules = _module
components = _module
mnist_datamodule = _module
eval = _module
models = _module
simple_dense_net = _module
mnist_module = _module
train = _module
utils = _module
pylogger = _module
rich_utils = _module
tests = _module
conftest = _module
helpers = _module
package_available = _module
run_if = _module
run_sh_command = _module
test_configs = _module
test_eval = _module
test_mnist_datamodule = _module
test_sweeps = _module
test_train = _module

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


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


import torch


from torch.utils.data import ConcatDataset


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torchvision.datasets import MNIST


from torchvision.transforms import transforms


from torch import nn


from typing import List


class SimpleDenseNet(nn.Module):

    def __init__(self, input_size: int=784, lin1_size: int=256, lin2_size: int=256, lin3_size: int=256, output_size: int=10):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, lin1_size), nn.BatchNorm1d(lin1_size), nn.ReLU(), nn.Linear(lin1_size, lin2_size), nn.BatchNorm1d(lin2_size), nn.ReLU(), nn.Linear(lin2_size, lin3_size), nn.BatchNorm1d(lin3_size), nn.ReLU(), nn.Linear(lin3_size, output_size))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)

