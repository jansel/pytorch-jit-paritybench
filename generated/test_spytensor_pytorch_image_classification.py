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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import random


import time


import torch


import torchvision


import numpy as np


import warnings


from torch import nn


from torch import optim


from collections import OrderedDict


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torch.nn.functional as F

