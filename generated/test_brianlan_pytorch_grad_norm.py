import sys
_module = sys.modules[__name__]
del sys
dataset = _module
model = _module
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


import torch


import numpy as np


from torch.utils import data


from torch.nn.modules.loss import MSELoss


import torch.nn.functional as F


import matplotlib.pyplot as plt


from torch.autograd import Variable


class RegressionTrain(torch.nn.Module):
    """
    """

    def __init__(self, model):
        """
        """
        super(RegressionTrain, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        self.mse_loss = MSELoss()

    def forward(self, x, ts):
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)
        assert ys.size()[1] == n_tasks
        task_loss = []
        for i in range(n_tasks):
            task_loss.append(self.mse_loss(ys[:, i, :], ts[:, i, :]))
        task_loss = torch.stack(task_loss)
        return task_loss

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


class RegressionModel(torch.nn.Module):
    """
    """

    def __init__(self, n_tasks):
        """
        Constructor of the architecture.
        Input:
            n_tasks: number of tasks to solve ($T$ in the paper)
        """
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks
        self.l1 = torch.nn.Linear(250, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, 100)
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), torch.nn.Linear(100, 100))

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))
        return torch.stack(outs, dim=1)

    def get_last_shared_layer(self):
        return self.l4

