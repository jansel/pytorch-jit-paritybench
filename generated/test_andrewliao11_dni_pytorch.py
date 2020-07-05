import sys
_module = sys.modules[__name__]
del sys
cnn = _module
dataset = _module
dni = _module
main = _module
mlp = _module
model = _module
plot = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
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


import torchvision.datasets as dsets


import torchvision.transforms as transforms


from torch.autograd import Variable


import numpy as np


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class dni_linear(nn.Module):

    def __init__(self, input_dims, num_classes, dni_hidden_size=1024, conditioned=False):
        super(dni_linear, self).__init__()
        self.conditioned = conditioned
        if self.conditioned:
            dni_input_dims = input_dims + num_classes
        else:
            dni_input_dims = input_dims
        self.layer1 = nn.Sequential(nn.Linear(dni_input_dims, dni_hidden_size), nn.BatchNorm1d(dni_hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(dni_hidden_size, dni_hidden_size), nn.BatchNorm1d(dni_hidden_size), nn.ReLU())
        self.layer3 = nn.Linear(dni_hidden_size, input_dims)

    def forward(self, x, y):
        if self.conditioned:
            assert y is not None
            x = torch.cat((x, y), 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class dni_Conv2d(nn.Module):

    def __init__(self, input_dims, input_size, num_classes, dni_hidden_size=64, conditioned=False):
        super(dni_Conv2d, self).__init__()
        self.conditioned = conditioned
        if self.conditioned:
            dni_input_dims = input_dims + 1
        else:
            dni_input_dims = input_dims
        self.input_size = list(input_size)
        self.label_emb = nn.Linear(num_classes, np.prod(np.array(input_size)))
        self.layer1 = nn.Sequential(nn.Conv2d(dni_input_dims, dni_hidden_size, kernel_size=5, padding=2), nn.BatchNorm2d(dni_hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(dni_hidden_size, dni_hidden_size, kernel_size=5, padding=2), nn.BatchNorm2d(dni_hidden_size), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(dni_hidden_size, input_dims, kernel_size=5, padding=2))

    def forward(self, x, y):
        if self.conditioned:
            assert y is not None
            y = self.label_emb(y)
            y = y.view([-1, 1] + self.input_size)
            x = torch.cat((x, y), 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class cnn(nn.Module):

    def __init__(self, in_channel, conditioned_DNI, num_classes):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channel, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self._layer1 = dni_Conv2d(16, (14, 14), num_classes, conditioned=conditioned_DNI)
        self._layer2 = dni_Conv2d(32, (7, 7), num_classes, conditioned=conditioned_DNI)
        self._fc = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)
        self.cnn = nn.Sequential(self.layer1, self.layer2, self.fc)
        self.dni = nn.Sequential(self._layer1, self._layer2, self._fc)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()

    def init_optimzers(self, learning_rate=0.001):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_fc)

    def forward_layer1(self, x, y=None):
        out = self.layer1(x)
        grad = self._layer1(out, y)
        return out, grad

    def forward_layer2(self, x, y=None):
        out = self.layer2(x)
        grad = self._layer2(out, y)
        return out, grad

    def forward_fc(self, x, y=None):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        grad = self._fc(out, y)
        return out, grad

    def forward(self, x, y=None):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer2_flat = layer2.view(layer2.size(0), -1)
        fc = self.fc(layer2_flat)
        if y is not None:
            grad_layer1 = self._layer1(layer1, y)
            grad_layer2 = self._layer2(layer2, y)
            grad_fc = self._fc(fc, y)
            return (layer1, layer2, fc), (grad_layer1, grad_layer2, grad_fc)
        else:
            return layer1, layer2, fc


class mlp(nn.Module):

    def __init__(self, conditioned_DNI, input_size, num_classes, hidden_size=256):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._fc1 = dni_linear(hidden_size, num_classes, conditioned=conditioned_DNI)
        self._fc2 = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)
        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
        self.dni = nn.Sequential(self._fc1, self._fc2)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()

    def init_optimzers(self, learning_rate=3e-05):
        self.optimizers.append(torch.optim.Adam(self.fc1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc2.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_fc1)
        self.forwards.append(self.forward_fc2)

    def forward_fc1(self, x, y=None):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        grad = self._fc1(out, y)
        return out, grad

    def forward_fc2(self, x, y=None):
        x = self.relu(x)
        out = self.fc2(x)
        grad = self._fc2(out, y)
        return out, grad

    def forward(self, x, y=None):
        x = x.view(-1, 28 * 28)
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)
        if y is not None:
            grad_fc1 = self._fc1(fc1, y)
            grad_fc2 = self._fc2(fc2, y)
            return (fc1, fc2), (grad_fc1, grad_fc2)
        else:
            return fc1, fc2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (dni_Conv2d,
     lambda: ([], {'input_dims': 4, 'input_size': [4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (dni_linear,
     lambda: ([], {'input_dims': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_andrewliao11_dni_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

