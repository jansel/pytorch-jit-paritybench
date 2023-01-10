import sys
_module = sys.modules[__name__]
del sys
conf = _module
train_cifar = _module
write_datasets = _module
custom_transform = _module
linear_regression = _module
transform_with_inds = _module
ffcv = _module
benchmarks = _module
benchmark = _module
decorator = _module
suites = _module
image_read = _module
jpeg_decode = _module
memory_read = _module
fields = _module
base = _module
basics = _module
bytes = _module
decoders = _module
json = _module
ndarray = _module
rgb_image = _module
libffcv = _module
loader = _module
epoch_iterator = _module
loader = _module
memory_allocator = _module
memory_managers = _module
common = _module
os_cache = _module
process_cache = _module
context = _module
manager = _module
page_reader = _module
schedule = _module
pipeline = _module
allocation_query = _module
compiler = _module
operation = _module
pipeline = _module
state = _module
reader = _module
transforms = _module
cutout = _module
flip = _module
mixup = _module
module = _module
normalize = _module
ops = _module
poisoning = _module
random_resized_crop = _module
replace_label = _module
translate = _module
utils = _module
fast_crop = _module
traversal_order = _module
quasi_random = _module
random = _module
sequential = _module
types = _module
writer = _module
setup = _module
test_array_field = _module
test_augmentations = _module
test_basic_pipeline = _module
test_cuda_nonblocking = _module
test_custom_field = _module
test_image_normalization = _module
test_image_pipeline = _module
test_image_read = _module
test_json_field = _module
test_loader_filter = _module
test_memory_allocation = _module
test_memory_leak = _module
test_memory_reader = _module
test_partial_batches = _module
test_partial_pipeline = _module
test_rrc = _module
test_traversal_orders = _module
test_webdataset = _module
test_writer = _module

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


from typing import List


import time


import numpy as np


import torch as ch


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.nn import CrossEntropyLoss


from torch.nn import Conv2d


from torch.nn import BatchNorm2d


from torch.optim import SGD


from torch.optim import lr_scheduler


import torchvision


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import logging


from time import sleep


from time import time


from torch.utils.data import Dataset


from collections import defaultdict


from queue import Queue


from queue import Full


from typing import Sequence


from typing import TYPE_CHECKING


import enum


from re import sub


from typing import Any


from typing import Callable


from typing import Mapping


from typing import Type


from typing import Union


from typing import Literal


from enum import Enum


from enum import unique


from enum import auto


from typing import Optional


from typing import Tuple


import warnings


import torch.nn.functional as F


from numpy.random import permutation


from numpy.random import rand


from collections.abc import Sequence


from numpy import dtype


import random


from torch.utils.data import DistributedSampler


import uuid


from torchvision import transforms as tvt


from torchvision.datasets import CIFAR10


from torchvision.utils import save_image


from torchvision.utils import make_grid


from torch.utils.data import Subset


import string


from typing import Counter


from torch.utils.data import distributed


from torch.multiprocessing import spawn


from torch.multiprocessing import Queue


from torch.distributed import init_process_group


from numpy.random import shuffle


class Mul(ch.nn.Module):

    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(ch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(ch.nn.Module):

    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mul,
     lambda: ([], {'weight': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_libffcv_ffcv(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

