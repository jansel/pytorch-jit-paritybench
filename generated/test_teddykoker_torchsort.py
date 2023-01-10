import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
cifar10 = _module
setup = _module
test_ops = _module
torchsort = _module
ops = _module

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


from collections import defaultdict


import matplotlib.pyplot as plt


import torch


from functools import partial


import numpy as np


import torch.nn.functional as F


import torchvision


import torchvision.transforms as T


from torch import nn


from torch.utils.data import DataLoader


from torchvision.datasets import CIFAR10


from functools import lru_cache


from torch.utils import cpp_extension

