import sys
_module = sys.modules[__name__]
del sys
main = _module
tfpack2lmdb = _module
folder2lmdb = _module

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


import random


import time


import warnings


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import string


import torch.utils.data as data


from torch.utils.data import DataLoader


from torchvision.transforms import transforms


from torchvision.datasets import ImageFolder


from torchvision import transforms


from torchvision import datasets

