import sys
_module = sys.modules[__name__]
del sys
cca_zoo = _module
data = _module
deep = _module
simulated = _module
deepmodels = _module
_base = _module
_discriminative = _module
_dcca = _module
_dcca_barlow_twins = _module
_dcca_eigengame = _module
_dcca_noi = _module
_dcca_sdl = _module
_dtcca = _module
_generative = _module
_base = _module
_dccae = _module
_dvcca = _module
_splitae = _module
architectures = _module
callbacks = _module
objectives = _module
model_selection = _module
_search = _module
_validation = _module
models = _module
_grcca = _module
_iterative = _module
_altmaxvar = _module
_elastic = _module
_pddgcca = _module
_pls_als = _module
_pmd = _module
_scca_admm = _module
_scca_parkhomenko = _module
_spancca = _module
_swcca = _module
_multiview = _module
_gcca = _module
_mcca = _module
_tcca = _module
_ncca = _module
_partialcca = _module
_prcca = _module
_proximal_operators = _module
_rcca = _module
_stochastic = _module
_base = _module
_eigengame = _module
_ghagep = _module
_incrementalpls = _module
_stochasticpls = _module
plotting = _module
probabilisticmodels = _module
_probabilisticcca = _module
test = _module
test_deepmodels = _module
test_models = _module
utils = _module
check_values = _module
conf = _module
examples = _module
plot_dcca = _module
plot_dcca_custom_data = _module
plot_dcca_multi = _module
plot_dvcca = _module
plot_hyperparameter_selection = _module
plot_kernel_cca = _module
plot_many_views = _module
plot_plotting = _module
plot_sparse_cca = _module
plot_validation = _module
setup = _module

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


from typing import Iterable


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from abc import abstractmethod


import torch


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiStepLR


import torch.nn.functional as F


from torch.nn import functional as F


from math import sqrt


from torch import nn


import torchvision


from torch.autograd import Variable


from scipy.linalg import block_diag


from torch.utils import data


from sklearn.utils.validation import check_random_state


from torch import manual_seed


from torch.utils.data import random_split


import scipy.sparse as sp


from sklearn.utils.fixes import loguniform


import warnings


from torch.utils.data import Subset


class _BaseEncoder(torch.nn.Module):

    @abstractmethod
    def __init__(self, latent_dims: int, variational: bool=False):
        super(_BaseEncoder, self).__init__()
        self.variational = variational
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class _BaseDecoder(torch.nn.Module):

    @abstractmethod
    def __init__(self, latent_dims: int):
        super(_BaseDecoder, self).__init__()
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class Encoder(_BaseEncoder):

    def __init__(self, latent_dims: int, variational: bool=False, feature_size: int=784, layer_sizes: tuple=None, activation=nn.LeakyReLU(), dropout=0):
        super(Encoder, self).__init__(latent_dims, variational=variational)
        if layer_sizes is None:
            layer_sizes = 128,
        layer_sizes = (feature_size,) + layer_sizes + (latent_dims,)
        layers = []
        for l_id in range(len(layer_sizes) - 2):
            layers.append(torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]), activation))
        self.layers = torch.nn.Sequential(*layers)
        if self.variational:
            self.fc_mu = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.fc_var = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        else:
            self.fc = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        x = self.layers(x)
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class Decoder(_BaseDecoder):

    def __init__(self, latent_dims: int, feature_size: int=784, layer_sizes: tuple=None, activation=nn.LeakyReLU(), dropout=0):
        super(Decoder, self).__init__(latent_dims)
        if layer_sizes is None:
            layer_sizes = 128,
        layer_sizes = (latent_dims,) + layer_sizes + (feature_size,)
        layers = []
        for l_id in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]), activation))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CNNEncoder(_BaseEncoder):

    def __init__(self, latent_dims: int, variational: bool=False, feature_size: Iterable=(28, 28), channels: tuple=None, kernel_sizes: tuple=None, stride: tuple=None, padding: tuple=None, activation=nn.LeakyReLU(), dropout=0):
        super(CNNEncoder, self).__init__(latent_dims, variational=variational)
        if channels is None:
            channels = 1, 1
        if kernel_sizes is None:
            kernel_sizes = (5,) * len(channels)
        if stride is None:
            stride = (1,) * len(channels)
        if padding is None:
            padding = (2,) * len(channels)
        conv_layers = []
        current_size = feature_size[0]
        current_channels = 1
        for l_id in range(len(channels) - 1):
            conv_layers.append(torch.nn.Sequential(torch.nn.Conv2d(in_channels=current_channels, out_channels=channels[l_id], kernel_size=kernel_sizes[l_id], stride=stride[l_id], padding=padding[l_id]), activation))
            current_size = current_size
            current_channels = channels[l_id]
        if self.variational:
            self.fc_mu = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(int(current_size * current_size * current_channels), latent_dims))
            self.fc_var = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(int(current_size * current_size * current_channels), latent_dims))
        else:
            self.fc = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(int(current_size * current_size * current_channels), latent_dims))
        self.conv_layers = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape((x.shape[0], -1))
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class CNNDecoder(_BaseDecoder):

    def __init__(self, latent_dims: int, feature_size: Iterable=(28, 28), channels: tuple=None, kernel_sizes=None, strides=None, paddings=None, activation=nn.LeakyReLU(), dropout=0):
        super(CNNDecoder, self).__init__(latent_dims)
        if channels is None:
            channels = 1, 1
        if kernel_sizes is None:
            kernel_sizes = (5,) * len(channels)
        if strides is None:
            strides = (1,) * len(channels)
        if paddings is None:
            paddings = (2,) * len(channels)
        conv_layers = []
        current_channels = 1
        current_size = feature_size[0]
        for l_id, (channel, kernel, stride, padding) in reversed(list(enumerate(zip(channels, kernel_sizes, strides, paddings)))):
            conv_layers.append(torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=channel, out_channels=current_channels, kernel_size=kernel_sizes[l_id], stride=strides[l_id], padding=paddings[l_id]), activation))
            current_size = current_size
            current_channels = channel
        self.conv_layers = torch.nn.Sequential(*conv_layers[::-1])
        self.fc_layer = torch.nn.Sequential(nn.Dropout(p=dropout), torch.nn.Linear(latent_dims, int(current_size * current_size * current_channels)), activation)

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.reshape((x.shape[0], self.conv_layers[0][0].in_channels, -1))
        x = x.reshape((x.shape[0], self.conv_layers[0][0].in_channels, int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1]))))
        x = self.conv_layers(x)
        return x


class LinearEncoder(_BaseEncoder):

    def __init__(self, latent_dims: int, feature_size: int, variational: bool=False):
        super(LinearEncoder, self).__init__(latent_dims, variational=variational)
        self.variational = variational
        if self.variational:
            self.fc_mu = torch.nn.Linear(feature_size, latent_dims)
            self.fc_var = torch.nn.Linear(feature_size, latent_dims)
        else:
            self.fc = torch.nn.Linear(feature_size, latent_dims)

    def forward(self, x):
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class LinearDecoder(_BaseDecoder):

    def __init__(self, latent_dims: int, feature_size: int):
        super(LinearDecoder, self).__init__(latent_dims)
        self.linear = torch.nn.Linear(latent_dims, feature_size)

    def forward(self, x):
        out = self.linear(x)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNNDecoder,
     lambda: ([], {'latent_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'latent_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearDecoder,
     lambda: ([], {'latent_dims': 4, 'feature_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearEncoder,
     lambda: ([], {'latent_dims': 4, 'feature_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_BaseDecoder,
     lambda: ([], {'latent_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BaseEncoder,
     lambda: ([], {'latent_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jameschapman19_cca_zoo(_paritybench_base):
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

