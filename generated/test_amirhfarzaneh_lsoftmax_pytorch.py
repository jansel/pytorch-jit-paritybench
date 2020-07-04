import sys
_module = sys.modules[__name__]
del sys
plot_logs = _module
lsoftmax = _module
models = _module
train_mnist = _module

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


import math


import torch


from torch import nn


from scipy.special import binom


import torch.nn as nn


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import numpy as np


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features
        self.output_dim = output_features
        self.margin = margin
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(input_features,
            output_features))
        self.divisor = math.pi / self.margin
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2)))
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2))
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers)))
        self.signs = torch.ones(margin // 2 + 1)
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)
        sin2_terms = sin2_theta.unsqueeze(1) ** self.sin2_powers.unsqueeze(0)
        cos_m_theta = (self.signs.unsqueeze(0) * self.C_m_2n.unsqueeze(0) *
            cos_terms * sin2_terms).sum(1)
        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        eps = 1e-07
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]
            w_target_norm = w[:, (target)].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)
            k = self.find_k(cos_theta_target)
            logit_target_updated = w_target_norm * x_norm * ((-1) ** k *
                cos_m_theta_target - 2 * k)
            logit_target_updated_beta = (logit_target_updated + beta *
                logit[indexes, target]) / (1 + beta)
            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


class MNISTNet(nn.Module):

    def __init__(self, margin, device):
        super(MNISTNet, self).__init__()
        self.margin = margin
        self.device = device
        self.conv_0 = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 64, 3),
            nn.PReLU(), nn.BatchNorm2d(64))
        self.conv_1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.
            PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.conv_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.
            PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.
            PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(576, 256), nn.BatchNorm1d(256))
        self.lsoftmax_linear = LSoftmaxLinear(input_features=256,
            output_features=10, margin=margin, device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 576)
        x = self.fc(x)
        logit = self.lsoftmax_linear(input=x, target=target)
        return logit


class MNISTFIG2Net(nn.Module):

    def __init__(self, margin, device):
        super(MNISTFIG2Net, self).__init__()
        self.margin = margin
        self.device = device
        self.conv_1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.
            PReLU(), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 5, padding=2),
            nn.PReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.
            PReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 5, padding=2),
            nn.PReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2))
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 128, 5, padding=2), nn.
            PReLU(), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 5, padding=2),
            nn.PReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(1152, 2), nn.BatchNorm1d(2))
        self.lsoftmax_linear = LSoftmaxLinear(input_features=2,
            output_features=10, margin=margin, device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 1152)
        x = self.fc(x)
        logit = self.lsoftmax_linear(input=x, target=target)
        return logit, x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_amirhfarzaneh_lsoftmax_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LSoftmaxLinear(*[], **{'input_features': 4, 'output_features': 4, 'margin': 4, 'device': 0}), [torch.rand([4, 4])], {})

