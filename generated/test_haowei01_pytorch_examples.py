import sys
_module = sys.modules[__name__]
del sys
LambdaRank = _module
ListNet = _module
RankNet = _module
ranking = _module
data_loaders = _module
load_expedia = _module
load_mslr = _module
metrics = _module
positional_bias = _module
utils = _module
uplift_model = _module
eval_uplift_effect = _module
rank_uplift = _module

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


import pandas as pd


import torch


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


import torch.nn.functional as F


from collections import defaultdict


from matplotlib import pyplot as plt


class LambdaRank(nn.Module):

    def __init__(self, net_structures, leaky_relu=False, sigma=1.0, double_precision=False):
        """Fully Connected Layers with Sigmoid activation at the last layer

        :param net_structures: list of int for LambdaRank FC width
        """
        super(LambdaRank, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(self, 'fc' + str(i + 1), nn.Linear(net_structures[i], net_structures[i + 1]))
            if leaky_relu:
                setattr(self, 'act' + str(i + 1), nn.LeakyReLU())
            else:
                setattr(self, 'act' + str(i + 1), nn.ReLU())
        setattr(self, 'fc' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))
        if double_precision:
            for i in range(1, len(net_structures) + 1):
                setattr(self, 'fc' + str(i), getattr(self, 'fc' + str(i)).double())
        self.sigma = sigma
        self.activation = nn.ReLU6()

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            act = getattr(self, 'act' + str(i))
            input1 = act(fc(input1))
        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1)) * self.sigma

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            None
            fc = getattr(self, 'fc' + str(i))
            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0
            except Exception:
                ipdb.set_trace()
            None
            None


class RankNet(nn.Module):

    def __init__(self, net_structures, double_precision=False):
        """
        :param net_structures: list of int for RankNet FC width
        """
        super(RankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i + 1])
            if double_precision:
                layer = layer.double()
            setattr(self, 'fc' + str(i + 1), layer)
        last_layer = nn.Linear(net_structures[-1], 1)
        if double_precision:
            last_layer = last_layer.double()
        setattr(self, 'fc' + str(len(net_structures)), last_layer)
        self.activation = nn.ReLU6()

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1))

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            None
            fc = getattr(self, 'fc' + str(i))
            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0
            except Exception:
                ipdb.set_trace()
            None
            None


class RankNetPairs(RankNet):

    def __init__(self, net_structures, double_precision=False):
        super(RankNetPairs, self).__init__(net_structures, double_precision)

    def forward(self, input1, input2):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))
        fc = getattr(self, 'fc' + str(self.fc_layers))
        input1 = self.activation(fc(input1))
        input2 = self.activation(fc(input2))
        return torch.sigmoid(input1 - input2)


class UpliftRanker(nn.Module):

    def __init__(self, net_structures):
        """
        :param list net_structures: width of each FC layer
        """
        super(UpliftRanker, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i + 1])
            setattr(self, 'fc' + str(i + 1), layer)
        last_layer = nn.Linear(net_structures[-1], 1)
        setattr(self, 'fc' + str(len(net_structures)), last_layer)
        self.activation = torch.tanh

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1))

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            None
            fc = getattr(self, 'fc' + str(i))
            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0
            except Exception:
                ipdb.set_trace()
            None
            None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LambdaRank,
     lambda: ([], {'net_structures': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RankNet,
     lambda: ([], {'net_structures': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RankNetPairs,
     lambda: ([], {'net_structures': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UpliftRanker,
     lambda: ([], {'net_structures': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_haowei01_pytorch_examples(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

