import sys
_module = sys.modules[__name__]
del sys
conf = _module
mnist = _module
viz_optimizers = _module
setup = _module
conftest = _module
test_basic = _module
test_optimizer = _module
test_optimizer_with_nn = _module
test_param_validation = _module
utils = _module
torch_optimizer = _module
accsgd = _module
adabound = _module
adamod = _module
diffgrad = _module
lamb = _module
lookahead = _module
novograd = _module
pid = _module
qhadam = _module
qhm = _module
radam = _module
sgdw = _module
shampoo = _module
types = _module
yogi = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


from torchvision import utils


from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter


import math


import numpy as np


from torch import nn


from torch.optim.optimizer import Optimizer


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jettify_pytorch_optimizer(_paritybench_base):
    pass
    def test_000(self):
        self._check(LogisticRegression(*[], **{}), [torch.rand([2, 2])], {})

