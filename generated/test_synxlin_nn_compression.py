import sys
_module = sys.modules[__name__]
del sys
master = _module
examples = _module
channel_pruning = _module
prune_train = _module
vgg_pruner = _module
deep_compression = _module
decode = _module
encode = _module
prune_train = _module
quantize_train = _module
sensitivity_scan = _module
slender = _module
coding = _module
codec = _module
encode = _module
prune = _module
channel = _module
vanilla = _module
quantize = _module
fixed_point = _module
kmeans = _module
linear = _module
quantizer = _module
replicate = _module
utils = _module
test = _module
test_coding = _module
test_quantize = _module
test_vanilla_prune = _module

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


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import re


from torchvision.models import VGG


import numpy as np


import math


from collections import Counter


from itertools import repeat


import random


from sklearn.linear_model import Lasso


from collections import Iterable


from sklearn.cluster import KMeans

