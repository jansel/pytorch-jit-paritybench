import sys
_module = sys.modules[__name__]
del sys
conf = _module
mnist = _module
mnist_with_meterlogger = _module
mnist_with_visdom = _module
setup = _module
test_datasets = _module
test_meters = _module
test_transforms = _module
torchnet = _module
dataset = _module
batchdataset = _module
concatdataset = _module
dataset = _module
listdataset = _module
resampledataset = _module
shuffledataset = _module
splitdataset = _module
tensordataset = _module
transformdataset = _module
engine = _module
engine = _module
logger = _module
meterlogger = _module
visdomlogger = _module
meter = _module
apmeter = _module
aucmeter = _module
averagevaluemeter = _module
classerrormeter = _module
confusionmeter = _module
mapmeter = _module
movingaveragevaluemeter = _module
msemeter = _module
timemeter = _module
transform = _module
utils = _module
multitaskdataloader = _module
resultswriter = _module
table = _module

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


import torch


import torch.optim


from torchvision.datasets.mnist import MNIST


from torch.autograd import Variable


import torch.nn.functional as F


from torch.nn.init import kaiming_normal


import numpy as np


import math


from torch.utils.data import DataLoader


import numbers


from itertools import islice


from itertools import chain


from itertools import repeat


import torch.utils.data

