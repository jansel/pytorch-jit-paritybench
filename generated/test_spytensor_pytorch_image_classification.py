import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
augmentations = _module
dataloader = _module
main = _module
models = _module
model = _module
test = _module
utils = _module
progress_bar = _module
utils = _module

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


from torch.utils.data import Dataset


from torchvision import transforms as T


from itertools import chain


import random


import numpy as np


import pandas as pd


import torch


import time


import torchvision


import warnings


from torch import nn


from torch import optim


from collections import OrderedDict


from torch.autograd import Variable


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


from sklearn.model_selection import StratifiedKFold


import torch.nn.functional as F

