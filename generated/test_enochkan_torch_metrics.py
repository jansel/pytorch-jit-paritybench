import sys
_module = sys.modules[__name__]
del sys
setup = _module
input_data = _module
test_classification_metrics = _module
test_regression_metrics = _module
torch_metrics = _module
classification = _module
accuracy = _module
dice = _module
f1 = _module
hinge = _module
kldivergence = _module
pr = _module
squarehinge = _module
regression = _module
huber = _module
logcosh = _module
mae = _module
meaniou = _module
mse = _module
msle = _module
rmse = _module
rsquared = _module
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


from collections import namedtuple


import torch


import numpy as np


from numpy.testing import assert_allclose


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


from sklearn.metrics import precision_score


from sklearn.metrics import mean_absolute_error


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_squared_log_error

