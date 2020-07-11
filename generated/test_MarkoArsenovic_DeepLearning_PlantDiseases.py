import sys
_module = sys.modules[__name__]
del sys
plot = _module
train = _module
occlusion = _module
saliency = _module
torchvis = _module
util = _module

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


import time


import numpy as np


import torch


import torch.optim as optim


from torch import nn


from torch.autograd import Variable


import torchvision


import torchvision.models as models


import torch.utils.model_zoo as model_zoo


import torchvision.transforms as transforms


from torchvision import datasets


from itertools import accumulate


from functools import reduce


from torchvision import models


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


import copy


import math


from collections import OrderedDict


from enum import Enum


from functools import partial

