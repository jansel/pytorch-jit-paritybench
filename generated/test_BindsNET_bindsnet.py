import sys
_module = sys.modules[__name__]
del sys
bindsnet = _module
analysis = _module
pipeline_analysis = _module
plotting = _module
visualization = _module
conversion = _module
conversion = _module
datasets = _module
alov300 = _module
collate = _module
dataloader = _module
davis = _module
preprocess = _module
spoken_mnist = _module
torchvision_wrapper = _module
encoding = _module
encoders = _module
encodings = _module
loaders = _module
environment = _module
evaluation = _module
learning = _module
reward = _module
models = _module
models = _module
network = _module
monitors = _module
network = _module
nodes = _module
topology = _module
pipeline = _module
action = _module
base_pipeline = _module
dataloader_pipeline = _module
environment_pipeline = _module
preprocessing = _module
utils = _module
conf = _module
annarchy = _module
benchmark = _module
gpu_annarchy = _module
plot_benchmark = _module
breakout = _module
breakout_stdp = _module
play_breakout_from_ANN = _module
random_baseline = _module
random_network_baseline = _module
batch_eth_mnist = _module
conv_mnist = _module
eth_mnist = _module
reservoir = _module
supervised_mnist = _module
tensorboard = _module
setup = _module
test_analyzers = _module
test_conversion = _module
test_encoding = _module
test_import = _module
test_models = _module
test_connections = _module
test_learning = _module
test_monitors = _module
test_network = _module
test_nodes = _module

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


import numpy as np


from torch.nn.modules.utils import _pair


from typing import Tuple


from typing import List


from typing import Optional


from typing import Sized


from typing import Dict


from typing import Union


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from typing import Sequence


from typing import Iterable


from scipy.spatial.distance import euclidean


from torchvision import models


from typing import Type


from abc import ABC


from abc import abstractmethod


from functools import reduce


from torch.nn import Module


from torch.nn import Parameter


import math


from torch import Tensor


from numpy import ndarray


from torchvision import transforms


class Permute(nn.Module):
    """
    PyTorch module for the explicit permutation of a tensor's dimensions in a parent
    module's ``forward`` pass (as opposed to ``torch.permute``).
    """

    def __init__(self, dims):
        """
        Constructor for ``Permute`` module.

        :param dims: Ordering of dimensions for permutation.
        """
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        Forward pass of permutation module.

        :param x: Input tensor to permute.
        :return: Permuted input tensor.
        """
        return x.permute(*self.dims).contiguous()


class FeatureExtractor(nn.Module):
    """
    Special-purpose PyTorch module for the extraction of child module's activations.
    """

    def __init__(self, submodule):
        """
        Constructor for ``FeatureExtractor`` module.

        :param submodule: The module who's children modules are to be extracted.
        """
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) ->Dict[nn.Module, torch.Tensor]:
        """
        Forward pass of the feature extractor.

        :param x: Input data for the ``submodule''.
        :return: A dictionary mapping
        """
        activations = {'input': x}
        for name, module in self.submodule._modules.items():
            if isinstance(module, nn.Linear):
                x = x.view(-1, module.in_features)
            x = module(x)
            activations[name] = x
        return activations


class AbstractMonitor(ABC):
    """
    Abstract base class for state variable monitors.
    """


class AbstractReward(ABC):
    """
    Abstract base class for reward computation.
    """

    @abstractmethod
    def compute(self, **kwargs) ->None:
        """
        Computes/modifies reward.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) ->None:
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        pass


class Nodes(torch.nn.Module):
    """
    Abstract base class for groups of neurons.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, learning: bool=True, **kwargs) ->None:
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param learning: Whether to be in learning or testing.
        """
        super().__init__()
        assert n is not None or shape is not None, 'Must provide either no. of neurons or shape of layer'
        if n is None:
            self.n = reduce(mul, shape)
        else:
            self.n = n
        if shape is None:
            self.shape = [self.n]
        else:
            self.shape = shape
        assert self.n == reduce(mul, self.shape), 'No. of neurons and shape do not match'
        self.traces = traces
        self.traces_additive = traces_additive
        self.register_buffer('s', torch.ByteTensor())
        self.sum_input = sum_input
        if self.traces:
            self.register_buffer('x', torch.Tensor())
            self.register_buffer('tc_trace', torch.tensor(tc_trace))
            if self.traces_additive:
                self.register_buffer('trace_scale', torch.tensor(trace_scale))
            self.register_buffer('trace_decay', torch.empty_like(self.tc_trace))
        if self.sum_input:
            self.register_buffer('summed', torch.FloatTensor())
        self.dt = None
        self.batch_size = None
        self.trace_decay = None
        self.learning = learning

    @abstractmethod
    def forward(self, x: torch.Tensor) ->None:
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            self.x *= self.trace_decay
            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s != 0, 1)
        if self.sum_input:
            self.summed += x.float()

    def reset_state_variables(self) ->None:
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()
        if self.traces:
            self.x.zero_()
        if self.sum_input:
            self.summed.zero_()

    def compute_decays(self, dt) ->None:
        """
        Abstract base class method for setting decays.
        """
        self.dt = dt
        if self.traces:
            self.trace_decay = torch.exp(-self.dt / self.tc_trace)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)
        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        if self.sum_input:
            self.summed = torch.zeros(batch_size, *self.shape, device=self.summed.device)

    def train(self, mode: bool=True) ->'Nodes':
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)


class AbstractConnection(ABC, Module):
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        super().__init__()
        assert isinstance(source, Nodes), 'Source is not a Nodes object'
        assert isinstance(target, Nodes), 'Target is not a Nodes object'
        self.source = source
        self.target = target
        self.nu = nu
        self.weight_decay = weight_decay
        self.reduction = reduction
        self.update_rule = kwargs.get('update_rule', NoOp)
        self.wmin = kwargs.get('wmin', -np.inf)
        self.wmax = kwargs.get('wmax', np.inf)
        self.norm = kwargs.get('norm', None)
        self.decay = kwargs.get('decay', None)
        if self.update_rule is None:
            self.update_rule = NoOp
        self.update_rule = self.update_rule(connection=self, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)

    @abstractmethod
    def compute(self, s: torch.Tensor) ->None:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get('learning', True)
        if learning:
            self.update_rule.update(**kwargs)
        mask = kwargs.get('mask', None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        pass


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        return out


class FullyConnectedNetwork(nn.Module):
    """
    Simply fully-connected network implemented in PyTorch.
    """

    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeatureExtractor,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FullyConnectedNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([784, 784])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([6400, 6400])], {}),
     True),
]

class Test_BindsNET_bindsnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

