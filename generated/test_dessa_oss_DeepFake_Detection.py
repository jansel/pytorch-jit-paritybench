import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
faceforensics_download = _module
hparams_search = _module
main = _module
model = _module
train = _module
utils = _module

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


import numpy as np


import torch


from torch.utils.data import Dataset


from torchvision import transforms


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.optim as optim


from torch.optim import lr_scheduler


import torchvision.models as models


from sklearn.metrics import roc_auc_score as extra_metric

