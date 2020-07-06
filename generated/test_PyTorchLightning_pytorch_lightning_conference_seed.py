import sys
_module = sys.modules[__name__]
del sys
setup = _module
src = _module
production_mnist = _module
mnist = _module
mnist_trainer = _module
research_mnist = _module
mnist = _module
mnist_trainer = _module
simplest_mnist = _module

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


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torchvision.datasets import MNIST


import torchvision.transforms as transforms

