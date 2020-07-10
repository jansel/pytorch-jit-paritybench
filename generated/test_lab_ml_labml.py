import sys
_module = sys.modules[__name__]
del sys
labml = _module
analytics = _module
matplotlib = _module
cli = _module
configs = _module
experiment = _module
helpers = _module
pytorch = _module
datasets = _module
cifar10 = _module
mnist = _module
device = _module
module = _module
seed = _module
train_valid = _module
training_loop = _module
internal = _module
altair = _module
binned_heatmap = _module
density = _module
histogram = _module
scatter = _module
utils = _module
cache = _module
indicators = _module
tb = _module
sqlite = _module
tensorboard = _module
base = _module
calculator = _module
config_function = _module
config_item = _module
dependency_parser = _module
eval_function = _module
parser = _module
processor = _module
processor_dict = _module
experiment_run = _module
pytorch = _module
sklearn = _module
lab = _module
logger = _module
destinations = _module
console = _module
factory = _module
ipynb = _module
inspect = _module
iterator = _module
loop = _module
sections = _module
store = _module
artifacts = _module
indicators = _module
namespace = _module
util = _module
writers = _module
colors = _module
monit = _module
tracker = _module
data = _module
pytorch = _module
delayed_keyboard_interrupt = _module
errors = _module
pytorch = _module
setup = _module
test = _module
configs_dict = _module
configs_new_api = _module
inspect = _module
tracker_perf = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from typing import Tuple


from typing import Optional


from typing import List


from typing import overload


from typing import Union


from typing import TYPE_CHECKING


import numpy as np


from typing import Set


from typing import Dict


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import torch


import torch.nn


from typing import Callable


import torch.optim


import torch.utils.data


from torch import nn


from abc import ABC


from collections import OrderedDict


from typing import Any


from typing import OrderedDict as OrderedDictType


from uuid import uuid1


from collections import deque


from torch.utils.data import TensorDataset


class Module(torch.nn.Module):
    """
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.
    """

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            return
        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f'Unable to determine device of {self.__class__.__name__}') from None

