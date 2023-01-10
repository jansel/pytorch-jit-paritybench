import sys
_module = sys.modules[__name__]
del sys
detector = _module
facedetector = _module
data_preparation = _module
dataset = _module
train = _module
video = _module

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


import numpy as np


from torch import long


from torch import tensor


from torch.utils.data.dataset import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


from typing import Dict


from typing import List


import pandas as pd


import torch


import torch.nn.init as init


from sklearn.model_selection import train_test_split


from torch import Tensor


from torch.nn import Conv2d


from torch.nn import CrossEntropyLoss


from torch.nn import Linear


from torch.nn import MaxPool2d


from torch.nn import ReLU


from torch.nn import Sequential


from torch.optim import Adam


from torch.optim.optimizer import Optimizer


from torch.utils.data import DataLoader

