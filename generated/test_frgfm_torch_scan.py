import sys
_module = sys.modules[__name__]
del sys
collect_env = _module
verify_labels = _module
conf = _module
benchmark = _module
setup = _module
test_crawler = _module
test_modules = _module
test_process = _module
test_utils = _module
torchscan = _module
crawler = _module
modules = _module
flops = _module
macs = _module
memory = _module
receptive = _module
process = _module
memory = _module
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


import re


from collections import namedtuple


from typing import Any


from typing import Set


from typing import Tuple


import torch


from torchvision import models


from collections import OrderedDict


import torch.nn as nn


from torch import nn


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Optional


from typing import Union


from torch.nn import Module


import warnings


from functools import reduce


from torch import Tensor


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeNd


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


import math


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

