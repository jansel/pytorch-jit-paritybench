import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
cachers_test = _module
datasets = _module
datasets_test = _module
maps_test = _module
modifiers_test = _module
samplers_test = _module
torchdata_test = _module
utils = _module
torchdata = _module
_base = _module
_dev_utils = _module
cachers = _module
datasets = _module
maps = _module
modifiers = _module
samplers = _module

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


import torch


import torchvision


import time


import abc


import functools


import typing


from torch.utils.data import ConcatDataset as TorchConcatDataset


from torch.utils.data import Dataset as TorchDataset


from torch.utils.data import ChainDataset as TorchChain


from torch.utils.data import IterableDataset as TorchIterable


from torch.utils.data import TensorDataset as TorchTensorDataset


from torch.utils.data import RandomSampler


from torch.utils.data import Sampler


from torch.utils.data import SubsetRandomSampler

