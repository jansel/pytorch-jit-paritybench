import sys
_module = sys.modules[__name__]
del sys
data = _module
celeba = _module
cifar = _module
kdd = _module
mnist = _module
mnist_anomaly = _module
toy = _module
networks = _module
celeba = _module
cifar = _module
kdd = _module
mnist = _module
regularizers = _module
toy = _module
evals = _module
inception_score = _module
sampler = _module
eval_metrics_cifar = _module
latent_space_mcmc = _module
anomaly_kdd = _module
anomaly_mnist = _module
classifier_mnist = _module
ebm_celeba = _module
ebm_cifar = _module
ebm_mnist = _module
ebm_toy = _module
functions = _module
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


import torch


from torchvision import transforms


from torchvision import datasets


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torchvision.utils import save_image


import matplotlib.pyplot as plt


import time


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import auc


import torch.optim as optim


from torchvision.utils import make_grid


class Generator(nn.Module):

    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(z_dim, dim), nn.ReLU(True), nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, 2))

    def forward(self, z):
        return self.main(z)


class EnergyModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(2, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, 1))

    def forward(self, x):
        return self.main(x).squeeze(-1)


class StatisticsNetwork(nn.Module):

    def __init__(self, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(2 + z_dim, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, 1))

    def forward(self, x, z):
        x = torch.cat([x, z], -1)
        return self.main(x).squeeze(-1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'z_dim': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ritheshkumar95_energy_based_generative_models(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

