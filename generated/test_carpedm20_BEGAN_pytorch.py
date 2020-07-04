import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
download = _module
folder = _module
main = _module
models = _module
trainer = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


from torch import nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import scipy.misc


from itertools import chain


from collections import deque


import torch.nn.parallel


import torchvision.utils as vutils


class BaseModel(nn.Module):

    def forward(self, x):
        gpu_ids = None
        if isinstance(x.data, torch.FloatTensor) and self.num_gpu > 1:
            gpu_ids = range(self.num_gpu)
        if gpu_ids:
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)


class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_carpedm20_BEGAN_pytorch(_paritybench_base):
    pass
