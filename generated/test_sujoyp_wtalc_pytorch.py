import sys
_module = sys.modules[__name__]
del sys
classificationMAP = _module
detectionMAP = _module
main = _module
model = _module
options = _module
test = _module
train = _module
utils = _module
video_dataset = _module

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


import torch.optim as optim


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as torch_init


from torch.autograd import Variable


import scipy.io as sio


import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):

    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.fc1 = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        return x, self.classifier(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Model,
     lambda: ([], {'n_feature': 4, 'n_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_sujoyp_wtalc_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

