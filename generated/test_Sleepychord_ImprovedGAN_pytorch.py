import sys
_module = sys.modules[__name__]
del sys
Datasets = _module
ImprovedGAN = _module
Nets = _module
functional = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.nn.parameter import Parameter


from torch import nn


from torch.nn import functional as F


import math


class Discriminator(nn.Module):

    def __init__(self, input_dim=28 ** 2, output_dim=10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([LinearWeightNorm(input_dim, 1000
            ), LinearWeightNorm(1000, 500), LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250), LinearWeightNorm(250, 250)])
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)

    def forward(self, x, feature=False, cuda=False):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor(
            [0])
        if cuda:
            noise = noise
        x = x + Variable(noise, requires_grad=False)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.relu(m(x))
            noise = torch.randn(x_f.size()
                ) * 0.5 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise
            x = x_f + Variable(noise, requires_grad=False)
        if feature:
            return x_f, self.final(x)
        return self.final(x)


class Generator(nn.Module):

    def __init__(self, z_dim, output_dim=28 ** 2):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias=False)
        self.bn1 = nn.BatchNorm1d(500, affine=False, eps=1e-06, momentum=0.5)
        self.fc2 = nn.Linear(500, 500, bias=False)
        self.bn2 = nn.BatchNorm1d(500, affine=False, eps=1e-06, momentum=0.5)
        self.fc3 = LinearWeightNorm(500, output_dim, weight_scale=1)
        self.bn1_b = Parameter(torch.zeros(500))
        self.bn2_b = Parameter(torch.zeros(500))
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, batch_size, cuda=False):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad=
            False, volatile=not self.training)
        if cuda:
            x = x
        x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.fc3(x))
        return x


class LinearWeightNorm(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, weight_scale=
        None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) *
            weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) *
                weight_scale)
        else:
            self.weight_scale = 1

    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.
            weight ** 2, dim=1, keepdim=True))
        return F.linear(x, W, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', weight_scale=' + str(self.weight_scale) + ')'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Sleepychord_ImprovedGAN_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(LinearWeightNorm(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

