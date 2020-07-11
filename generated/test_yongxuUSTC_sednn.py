import sys
_module = sys.modules[__name__]
del sys
config = _module
get_csv = _module
istft = _module
main_train_sednn_keras_v2 = _module
prepare_fea = _module
preprocessing = _module
spectrogram_to_wave = _module
wav_reconstruction = _module
data_generator = _module
evaluate = _module
main_dnn = _module
prepare_data = _module
crash = _module
prepare_data = _module
stft = _module
test9 = _module
tmp01 = _module
tmp01b = _module
tmp01c = _module
tmp01d = _module
tmp01e = _module
tmp01f = _module
tmp02 = _module
tmp02b = _module
tmp03 = _module
tmp03b = _module
tmp04 = _module
tmp04b = _module

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


import numpy as np


import time


import matplotlib.pyplot as plt


from scipy import signal


from sklearn import preprocessing


import torch


from torch.autograd import Variable


import math


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


import torch.optim as optim


class DNN(nn.Module):

    def __init__(self, stack_num, n_freq):
        super(DNN, self).__init__()
        n_hid = 2048
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_freq)

    def forward(self, x):
        drop_p = 0.2
        _, stack_num, n_freq = x.size()
        x = x.view(-1, stack_num * n_freq)
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
        return x5


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DNN,
     lambda: ([], {'stack_num': 4, 'n_freq': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_yongxuUSTC_sednn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

