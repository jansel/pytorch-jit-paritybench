import sys
_module = sys.modules[__name__]
del sys
convert_weights = _module
detoxify = _module
detoxify = _module
hubconf = _module
compute_bias_metric = _module
compute_language_breakdown = _module
evaluate = _module
utils = _module
preprocessing_utils = _module
run_prediction = _module
setup = _module
src = _module
data_loaders = _module
utils = _module
tests = _module
test_models = _module
test_torch_hub = _module
test_trainer = _module
train = _module

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


from collections import OrderedDict


import torch


import warnings


import numpy as np


import pandas as pd


from sklearn.metrics import roc_auc_score


from torch.utils.data import DataLoader


from torch.utils.data.dataset import Dataset


from torch.nn import functional as F

