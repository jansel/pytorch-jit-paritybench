import sys
_module = sys.modules[__name__]
del sys
conf = _module
fbcode = _module
auto_unit_example = _module
train_unit_example = _module
setup = _module
framework = _module
test_csv_writer = _module
test_garbage_collector = _module
test_lambda = _module
test_learning_rate_monitor = _module
test_module_summary = _module
test_pytorch_profiler = _module
test_tensorboard_parameter_monitor = _module
test_torchsnapshot_saver = _module
test_tqdm_progress_bar = _module
test_app_state_mixin = _module
test_auto_unit = _module
test_evaluate = _module
test_fit = _module
test_predict = _module
test_progress = _module
test_state = _module
test_train = _module
test_utils = _module
utils = _module
test_data_prefetcher = _module
test_multi_dataloader = _module
loggers = _module
test_csv = _module
test_in_memory = _module
test_json = _module
test_tensorboard = _module
test_utils = _module
test_device = _module
test_distributed = _module
test_early_stop_checker = _module
test_env = _module
test_fsspec = _module
test_memory = _module
test_misc = _module
test_oom = _module
test_rank_zero_log = _module
test_seed = _module
test_timer = _module
test_version = _module
torchtnt = _module
_test_utils = _module
auto_unit = _module
callback = _module
callbacks = _module
base_csv_writer = _module
garbage_collector = _module
lambda_callback = _module
learning_rate_monitor = _module
module_summary = _module
pytorch_profiler = _module
tensorboard_parameter_monitor = _module
torchsnapshot_saver = _module
tqdm_progress_bar = _module
evaluate = _module
fit = _module
predict = _module
progress = _module
state = _module
train = _module
unit = _module
utils = _module
data = _module
data_prefetcher = _module
iterators = _module
multi_dataloader = _module
device = _module
distributed = _module
early_stop_checker = _module
env = _module
fsspec = _module
csv = _module
file = _module
in_memory = _module
json = _module
logger = _module
tensorboard = _module
utils = _module
memory = _module
misc = _module
oom = _module
rank_zero_log = _module
seed = _module
test_utils = _module
timer = _module
version = _module

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


from typing import Any


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch


import torch.nn as nn


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataset import TensorDataset


from functools import partial


from inspect import getmembers


from inspect import isfunction


from torch.utils.tensorboard import SummaryWriter


from typing import Dict


from torch import nn


import math


import torch._dynamo


from torch.distributed import launcher


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.nn.parallel import DistributedDataParallel as DDP


from typing import Iterator


import torch.distributed as dist


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


import random


from collections import Counter


from typing import Mapping


from torch.utils.data import Dataset


import torch.distributed.launcher as launcher


from torch import distributed as dist


import numpy as np


from collections import defaultdict


from collections import namedtuple


import torch.distributed.launcher as pet


import time


from collections import deque


from typing import Deque


from copy import deepcopy


from random import random


from torch import Tensor


from torch.utils.data import IterableDataset


from torch.utils.data import TensorDataset


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import TypeVar


from torch.cuda.amp import GradScaler


from torch.optim.swa_utils import AveragedModel


from torch.optim.swa_utils import SWALR


from torch.optim.optimizer import Optimizer


from typing import Iterable


from typing import Generic


import collections


import inspect


from enum import Enum


from itertools import cycle


from typing import MutableMapping


from typing import Type


from typing import TYPE_CHECKING


from functools import wraps


import torch.nn.functional as F


import typing as T


from torch.distributed.constants import default_pg_timeout


from numpy import ndarray


from typing import Generator


from typing import Sequence


import uuid


import warnings


from time import perf_counter


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """
    The only difference between :class:`torch.nn.BatchNorm1d`, :class:`torch.nn.BatchNorm2d`,
    :class:`torch.nn.BatchNorm3d`, etc is this method that is overwritten by the sub-class.
    This method is used when calling forward as a sanity check.
    When using :function:`revert_sync_batchnorm` this sanity check is lost.
    """

    def _check_input_dim(self, input: Tensor) ->None:
        return


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_BatchNormXd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pytorch_tnt(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

