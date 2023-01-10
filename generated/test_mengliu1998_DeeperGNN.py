import sys
_module = sys.modules[__name__]
del sys
dagnn = _module
datasets = _module
main_ogbnarxiv = _module
train_eval = _module

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


from torch.nn import Linear


import torch.nn.functional as F


import warnings


import time


from torch import tensor


from torch.optim import Adam


import numpy as np


class Net(torch.nn.Module):

    def __init__(self, num_features, num_classes, hidden, K, dropout):
        super(Net, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.bn = torch.nn.BatchNorm1d(hidden)
        self.prop = Prop(num_classes, K)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        x = F.relu(self.bn(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, norm)
        return F.log_softmax(x, dim=1)

