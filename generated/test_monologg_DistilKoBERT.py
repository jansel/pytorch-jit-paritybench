import sys
_module = sys.modules[__name__]
del sys
distiller = _module
grouped_batch_sampler = _module
lm_seqs_dataset = _module
binarized_data = _module
extract = _module
extract_distilbert = _module
token_counts = _module
tokenization_kobert = _module
train = _module
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


import math


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.optim import AdamW


from torch.utils.data import BatchSampler


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


import copy


from collections import defaultdict


import numpy as np


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from torch.utils.data import Dataset


import logging

