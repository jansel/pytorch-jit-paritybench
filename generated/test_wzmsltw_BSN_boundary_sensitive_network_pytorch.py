import sys
_module = sys.modules[__name__]
del sys
eval_proposal = _module
utils = _module
data_process = _module
ldb_process = _module
dataset = _module
eval = _module
loss_function = _module
main = _module
models = _module
opts = _module
pgm = _module
post_processing = _module

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


import torch.nn.functional as F


import torchvision


import torch.nn.parallel


import torch.optim as optim


from torch.autograd import Variable


import torch.nn as nn


from torch.nn import init


class TEM(torch.nn.Module):

    def __init__(self, opt):
        super(TEM, self).__init__()
        self.feat_dim = opt['tem_feat_dim']
        self.temporal_dim = opt['temporal_scale']
        self.batch_size = opt['tem_batch_size']
        self.c_hidden = opt['tem_hidden_dim']
        self.tem_best_loss = 10000000
        self.output_dim = 3
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,
            out_channels=self.c_hidden, kernel_size=3, stride=1, padding=1,
            groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,
            out_channels=self.c_hidden, kernel_size=3, stride=1, padding=1,
            groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,
            out_channels=self.output_dim, kernel_size=1, stride=1, padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.01 * self.conv3(x))
        return x


class PEM(torch.nn.Module):

    def __init__(self, opt):
        super(PEM, self).__init__()
        self.feat_dim = opt['pem_feat_dim']
        self.batch_size = opt['pem_batch_size']
        self.hidden_dim = opt['pem_hidden_dim']
        self.u_ratio_m = opt['pem_u_ratio_m']
        self.u_ratio_l = opt['pem_u_ratio_l']
        self.output_dim = 1
        self.pem_best_loss = 1000000
        self.fc1 = torch.nn.Linear(in_features=self.feat_dim, out_features=
            self.hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim,
            out_features=self.output_dim, bias=True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(0.1 * self.fc1(x))
        x = torch.sigmoid(0.1 * self.fc2(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_wzmsltw_BSN_boundary_sensitive_network_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(PEM(*[], **{'opt': _mock_config(pem_feat_dim=4, pem_batch_size=4, pem_hidden_dim=4, pem_u_ratio_m=4, pem_u_ratio_l=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(TEM(*[], **{'opt': _mock_config(tem_feat_dim=4, temporal_scale=1.0, tem_batch_size=4, tem_hidden_dim=4)}), [torch.rand([4, 4, 64])], {})

