import sys
_module = sys.modules[__name__]
del sys
conf = _module
omniglot_loaders = _module
higher = _module
optim = _module
patch = _module
utils = _module
setup = _module
tests = _module
test_higher = _module
test_optim = _module
test_patch = _module

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


import typing


import torch


from torch import nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import time


import numpy as np


import typing as _typing


import torch as _torch


import abc as _abc


import collections as _collections


import copy as _copy


import math as _math


import warnings as _warnings


from collections import OrderedDict as _OrderedDict


import copy


from collections import OrderedDict


from torch import optim


from torch.nn import functional as F


class EnergyNet(nn.Module):
    """An energy function E(x, y) for visual single-label classification.

    An energy function takes an image x and label y
    as the input and outputs a real number.
    We use a LeNet-style architecture to extract an embedding from x
    that is that concatenated with y and passed through a single hidden
    layer fully-connected network.

    Args:
        n_fc_hidden (int): The number of hidden units the
          fully-connected layers have.
        n_cls (int): The number of classes.
    """

    def __init__(self, n_fc_hidden: int=500, n_cls: int=10):
        super(EnergyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, n_fc_hidden)
        self.fc2 = nn.Linear(n_fc_hidden, n_fc_hidden)
        self.fce1 = nn.Linear(n_fc_hidden + n_cls, n_fc_hidden)
        self.fce2 = nn.Linear(n_fc_hidden, 1)

    def forward(self, x, y):
        z = F.softplus(self.conv1(x))
        z = F.max_pool2d(z, 2, 2)
        z = F.softplus(self.conv2(z))
        z = F.max_pool2d(z, 2, 2)
        z = z.view(-1, 4 * 4 * 50)
        z = F.softplus(self.fc1(z))
        z = self.fc2(z)
        v = torch.cat((z, y), dim=1)
        v = F.softplus(self.fce1(v))
        E = self.fce2(v).squeeze()
        return E


class UnrollEnergy(nn.Module):
    """A deep energy module that unrolls an optimizer over the energy function.

    This module takes a grayscale 28x28 image x as the input and
    outputs a class prediction by (approximately) solving the
    optimization problem

        \\hat y = argmin_y E_	heta(x, y)

    with a fixed number of gradient steps.

    Args:
        Enet: The energy network.
        n_cls (int): The number of classes.
        n_inner_iter (int): The number of optimization steps to take.
    """

    def __init__(self, Enet: EnergyNet, n_cls: int=10, n_inner_iter: int=5):
        super(UnrollEnergy, self).__init__()
        self.Enet = Enet
        self.n_cls = n_cls
        self.n_inner_iter = n_inner_iter

    def forward(self, x):
        assert x.ndimension() == 4
        nbatch = x.size(0)
        y = torch.zeros(nbatch, self.n_cls, device=x.device, requires_grad=True)
        inner_opt = higher.get_diff_optim(torch.optim.SGD([y], lr=0.1), [y], device=x.device)
        for _ in range(self.n_inner_iter):
            E = self.Enet(x, y)
            y, = inner_opt.step(E.sum(), params=[y])
        return y


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class _MonkeyPatchBase(_abc.ABC, _torch.nn.Module):

    @_abc.abstractmethod
    def __init__(self) ->None:
        self._param_mapping: _typing.List[int] = []
        self._being_modified_internally: bool = True
        self._track_higher_grads: bool = True

    def forward(self):
        raise NotImplementedError("The monkey-patching logic has failed to override self.forward on the new module, or you tried calling forward on a patched version of a module which doesn't have forward (e.g. ModuleList).")

    def _expand_params(self, params: _typing.List[_torch.Tensor]) ->_typing.List[_torch.Tensor]:
        expanded = []
        for index in self._param_mapping:
            expanded.append(params[index])
        return expanded

    @property
    def init_fast_params(self):
        if not self.track_higher_grads:
            raise Exception('Cannot get initial parameters when not tracking higher gradients.')
        return self._fast_params[0]

    @property
    def fast_params(self):
        return None if self._fast_params is None else self._fast_params[-1]

    @fast_params.setter
    def fast_params(self, value):
        value = list(value)
        if self._fast_params is None:
            self._fast_params = []
        if self.track_higher_grads:
            self._fast_params.append(value)
        else:
            self._fast_params[0] = value

    @property
    def track_higher_grads(self):
        return self._track_higher_grads

    @track_higher_grads.setter
    def track_higher_grads(self, value):
        if not isinstance(value, bool):
            raise ValueError('Expected boolean argument. Got: {}.'.format(type(value)))
        self._track_higher_grads = value


class _ReferenceNet(nn.Module):

    def __init__(self, features, fc):
        super().__init__()
        self.features = features
        self.add_module('fc', fc)

    def batch_norm(self, inputs, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-05, momentum=0.1):
        running_mean = torch.zeros(np.prod(np.array(inputs.data.size()[1])))
        running_var = torch.ones(np.prod(np.array(inputs.data.size()[1])))
        return F.batch_norm(inputs, running_mean, running_var, weight, bias, training, momentum, eps)

    def maxpool(self, input, kernel_size, stride=None):
        return F.max_pool2d(input, kernel_size, stride)

    def forward(self, x, params=None):
        if params is None:
            x = self.features(x).view(x.size(0), 64)
            x = self.fc(x)
        else:
            x = F.conv2d(x, params['features.conv1.weight'], params['features.conv1.bias'])
            x = self.batch_norm(x, weight=params['features.bn1.weight'], bias=params['features.bn1.bias'], momentum=1)
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = F.conv2d(x, params['features.conv2.weight'], params['features.conv2.bias'])
            x = self.batch_norm(x, weight=params['features.bn2.weight'], bias=params['features.bn2.bias'], momentum=1)
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = F.conv2d(x, params['features.conv3.weight'], params['features.conv3.bias'])
            x = self.batch_norm(x, weight=params['features.bn3.weight'], bias=params['features.bn3.bias'], momentum=1)
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), 64)
            x = F.linear(x, params['fc.weight'], params['fc.bias'])
        return x

    def get_fast_weights(self):
        fast_weights = OrderedDict((name, param) for name, param in self.named_parameters())
        return fast_weights


class _TargetNet(nn.Module):

    def __init__(self, features, fc):
        super().__init__()
        self.features = features
        self.add_module('fc', fc)

    def forward(self, x):
        x = self.features(x).view(x.size(0), 64)
        x = self.fc(x)
        return x


class _NestedEnc(torch.nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class _Enc(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.e1 = _NestedEnc(torch.nn.Linear(4, 2))
        self.e2 = _NestedEnc(self.e1.f)

    def forward(self, x):
        return self.e1(x) + self.e2(x)


class _PartiallyUsed(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))

    def forward(self, x):
        return x @ self.a


class _NestedEnc(torch.nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class _Enc(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.e1 = _NestedEnc(torch.nn.Linear(4, 2))
        self.e2 = _NestedEnc(self.e1.f)

    def forward(self, x):
        return self.e1(x) + self.e2(x)


class _PartiallyUsed(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))

    def forward(self, x):
        return x @ self.a


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Enc,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_NestedEnc,
     lambda: ([], {'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_PartiallyUsed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ReferenceNet,
     lambda: ([], {'features': _mock_layer(), 'fc': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_TargetNet,
     lambda: ([], {'features': _mock_layer(), 'fc': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_higher(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

