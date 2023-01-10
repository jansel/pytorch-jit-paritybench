import sys
_module = sys.modules[__name__]
del sys
contrib = _module
app = _module
chat_res = _module
interact = _module
infer = _module
interact = _module
dataset_wb = _module
inputter = _module
data_utils = _module
train = _module

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


import random


from itertools import chain


import torch


import torch.nn.functional as F


import math


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


import numpy as np


from torch.nn.parallel import DistributedDataParallel


from torch.optim.lr_scheduler import LambdaLR

