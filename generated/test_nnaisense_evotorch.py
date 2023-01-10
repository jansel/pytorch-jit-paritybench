import sys
_module = sys.modules[__name__]
del sys
gen_doc_stubs = _module
bbo_vectorized = _module
moo_parallel = _module
rl_clipup = _module
rl_enjoy = _module
rl_gym = _module
tinytraj_humanoid_bullet = _module
setup = _module
evotorch = _module
algorithms = _module
cmaes = _module
distributed = _module
gaussian = _module
ga = _module
searchalgorithm = _module
core = _module
decorators = _module
distributions = _module
logging = _module
neuroevolution = _module
baseneproblem = _module
gymne = _module
neproblem = _module
net = _module
functional = _module
layers = _module
misc = _module
multilayered = _module
parser = _module
rl = _module
runningnorm = _module
runningstat = _module
statefulmodule = _module
vecrl = _module
supervisedne = _module
vecgymne = _module
operators = _module
base = _module
real = _module
sequence = _module
optimizers = _module
testing = _module
tools = _module
cloning = _module
hook = _module
immutable = _module
misc = _module
objectarray = _module
ranking = _module
readonlytensor = _module
recursiveprintable = _module
tensormaker = _module
conftest = _module
test_cloning = _module
test_core = _module
test_decorators = _module
test_examples = _module
test_hook = _module
test_logging = _module
test_net = _module
test_neuroevolution_net_parser = _module
test_neuroevolution_vecgymne = _module
test_normalization = _module
test_objectarray = _module
test_optimizers = _module
test_parallelization = _module
test_pareto_sorting = _module
test_ranking = _module
test_read_only_tensor = _module
test_tensor_making = _module
test_tensors_in_container = _module
test_tools_misc = _module
test_tools_recursiveprintable = _module
test_vecrl = _module

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


from time import sleep


from typing import Optional


from typing import Union


import numpy as np


import math


from copy import copy


from copy import deepcopy


from typing import Any


from typing import Callable


from typing import Iterable


from typing import List


from collections.abc import Mapping


from typing import Type


import logging


import random


from collections.abc import Sequence


from typing import NamedTuple


from typing import Tuple


from torch import nn


import torch.nn.functional as nnf


from collections import namedtuple


from torch.nn import utils as nnu


from warnings import warn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from numbers import Real


from collections import OrderedDict


from numbers import Number


from collections.abc import Iterable


from collections.abc import Set


import functools


import inspect


from numbers import Integral


from typing import Dict


from typing import Mapping


import itertools


from itertools import product


from torch.utils.data import TensorDataset


from torch import FloatTensor


class Clip(nn.Module):
    """A small torch module for clipping the values of tensors"""

    def __init__(self, lb: float, ub: float):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound. Values less than this will be clipped.
            ub: Upper bound. Values greater than this will be clipped.
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)

    def forward(self, x: torch.Tensor):
        return x.clamp(self._lb, self._ub)

    def extra_repr(self):
        return 'lb={}, ub={}'.format(self._lb, self._ub)


class Bin(nn.Module):
    """A small torch module for binning the values of tensors.

    In more details, considering a lower bound value lb,
    an upper bound value ub, and an input tensor x,
    each value within x closer to lb will be converted to lb
    and each value within x closer to ub will be converted to ub.
    """

    def __init__(self, lb: float, ub: float):
        """`__init__(...)`: Initialize the Clip operator.

        Args:
            lb: Lower bound
            ub: Upper bound
        """
        nn.Module.__init__(self)
        self._lb = float(lb)
        self._ub = float(ub)
        self._interval_size = self._ub - self._lb
        self._shrink_amount = self._interval_size / 2.0
        self._shift_amount = (self._ub + self._lb) / 2.0

    def forward(self, x: torch.Tensor):
        x = x - self._shift_amount
        x = x / self._shrink_amount
        x = torch.sign(x)
        x = x * self._shrink_amount
        x = x + self._shift_amount
        return x

    def extra_repr(self):
        return 'lb={}, ub={}'.format(self._lb, self._ub)


class Slice(nn.Module):
    """A small torch module for getting the slice of an input tensor"""

    def __init__(self, from_index: int, to_index: int):
        """`__init__(...)`: Initialize the Slice operator.

        Args:
            from_index: The index from which the slice begins.
            to_index: The exclusive index at which the slice ends.
        """
        nn.Module.__init__(self)
        self._from_index = from_index
        self._to_index = to_index

    def forward(self, x):
        return x[self._from_index:self._to_index]

    def extra_repr(self):
        return 'from_index={}, to_index={}'.format(self._from_index, self._to_index)


class Round(nn.Module):
    """A small torch module for rounding the values of an input tensor"""

    def __init__(self, ndigits: int=0):
        nn.Module.__init__(self)
        self._ndigits = int(ndigits)
        self._q = 10.0 ** self._ndigits

    def forward(self, x):
        x = x * self._q
        x = torch.round(x)
        x = x / self._q
        return x

    def extra_repr(self):
        return 'ndigits=' + str(self._ndigits)


class Apply(nn.Module):
    """A torch module for applying an arithmetic operator on an input tensor"""

    def __init__(self, operator: str, argument: float):
        """`__init__(...)`: Initialize the Apply module.

        Args:
            operator: Must be '+', '-', '*', '/', or '**'.
                Indicates which operation will be done
                on the input tensor.
            argument: Expected as a float, represents
                the right-argument of the operation
                (the left-argument being the input
                tensor).
        """
        nn.Module.__init__(self)
        self._operator = str(operator)
        assert self._operator in ('+', '-', '*', '/', '**')
        self._argument = float(argument)

    def forward(self, x):
        op = self._operator
        arg = self._argument
        if op == '+':
            return x + arg
        elif op == '-':
            return x - arg
        elif op == '*':
            return x * arg
        elif op == '/':
            return x / arg
        elif op == '**':
            return x ** arg
        else:
            raise ValueError('Unknown operator:' + repr(op))

    def extra_repr(self):
        return 'operator={}, argument={}'.format(repr(self._operator), self._argument)


class RNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str='tanh', *, dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu'):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)
        nonlinearity = str(nonlinearity)
        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype, device=device))
        self.W2 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.b2 = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        if nonlinearity == 'tanh':
            self.actfunc = torch.tanh
        else:
            self.actfunc = getattr(nnf, nonlinearity)
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]=None) ->tuple:
        if h is None:
            h = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        act = self.actfunc
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1.unsqueeze(-1)
        b2 = self.b2.unsqueeze(-1)
        x = x.unsqueeze(-1)
        h = h.unsqueeze(-1)
        y = act(W1 @ x + b1 + (W2 @ h + b2))
        y = y.squeeze(-1)
        return y, y

    def __repr__(self) ->str:
        clsname = type(self).__name__
        return f'{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size}, nonlinearity={repr(self.nonlinearity)})'


class LSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, *, dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu'):
        super().__init__()
        input_size = int(input_size)
        hidden_size = int(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        def input_weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.input_size, dtype=dtype, device=device))

        def weight():
            return nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, dtype=dtype, device=device))

        def bias():
            return nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype, device=device))
        self.W_ii = input_weight()
        self.W_if = input_weight()
        self.W_ig = input_weight()
        self.W_io = input_weight()
        self.W_hi = weight()
        self.W_hf = weight()
        self.W_hg = weight()
        self.W_ho = weight()
        self.b_ii = bias()
        self.b_if = bias()
        self.b_ig = bias()
        self.b_io = bias()
        self.b_hi = bias()
        self.b_hf = bias()
        self.b_hg = bias()
        self.b_ho = bias()

    def forward(self, x: torch.Tensor, hidden=None) ->tuple:
        sigm = torch.sigmoid
        tanh = torch.tanh
        if hidden is None:
            h_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
            c_prev = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_prev, c_prev = hidden
        i_t = sigm(self.W_ii @ x + self.b_ii + self.W_hi @ h_prev + self.b_hi)
        f_t = sigm(self.W_if @ x + self.b_if + self.W_hf @ h_prev + self.b_hf)
        g_t = tanh(self.W_ig @ x + self.b_ig + self.W_hg @ h_prev + self.b_hg)
        o_t = sigm(self.W_io @ x + self.b_io + self.W_ho @ h_prev + self.b_ho)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)
        return h_t, (h_t, c_t)

    def __repr__(self) ->str:
        clsname = type(self).__name__
        return f'{clsname}(input_size={self.input_size}, hidden_size={self.hidden_size})'


class FeedForwardNet(nn.Module):
    """
    Representation of a feed forward neural network as a torch Module.

    An example initialization of a FeedForwardNet is as follows:

        net = drt.FeedForwardNet(4, [(8, 'tanh'), (6, 'tanh')])

    which means that we would like to have a network which expects an input
    vector of length 4 and passes its input through 2 tanh-activated hidden
    layers (with neurons count 8 and 6, respectively).
    The output of the last hidden layer (of length 6) is the final
    output vector.

    The string representation of the module obtained via the example above
    is:

        FeedForwardNet(
          (layer_0): Linear(in_features=4, out_features=8, bias=True)
          (actfunc_0): Tanh()
          (layer_1): Linear(in_features=8, out_features=6, bias=True)
          (actfunc_1): Tanh()
        )
    """
    LengthActTuple = Tuple[int, Union[str, Callable]]
    LengthActBiasTuple = Tuple[int, Union[str, Callable], Union[bool]]

    def __init__(self, input_size: int, layers: List[Union[LengthActTuple, LengthActBiasTuple]]):
        """`__init__(...)`: Initialize the FeedForward network.

        Args:
            input_size: Input size of the network, expected as an int.
            layers: Expected as a list of tuples,
                where each tuple is either of the form
                `(layer_size, activation_function)`
                or of the form
                `(layer_size, activation_function, bias)`
                in which
                (i) `layer_size` is an int, specifying the number of neurons;
                (ii) `activation_function` is None, or a callable object,
                or a string containing the name of the activation function
                ('relu', 'selu', 'elu', 'tanh', 'hardtanh', or 'sigmoid');
                (iii) `bias` is a boolean, specifying whether the layer
                is to have a bias or not.
                When omitted, bias is set to True.
        """
        nn.Module.__init__(self)
        for i, layer in enumerate(layers):
            if len(layer) == 2:
                size, actfunc = layer
                bias = True
            elif len(layer) == 3:
                size, actfunc, bias = layer
            else:
                assert False, 'A layer tuple of invalid size is encountered'
            setattr(self, 'layer_' + str(i), nn.Linear(input_size, size, bias=bias))
            if isinstance(actfunc, str):
                if actfunc == 'relu':
                    actfunc = nn.ReLU()
                elif actfunc == 'selu':
                    actfunc = nn.SELU()
                elif actfunc == 'elu':
                    actfunc = nn.ELU()
                elif actfunc == 'tanh':
                    actfunc = nn.Tanh()
                elif actfunc == 'hardtanh':
                    actfunc = nn.Hardtanh()
                elif actfunc == 'sigmoid':
                    actfunc = nn.Sigmoid()
                elif actfunc == 'round':
                    actfunc = Round()
                else:
                    raise ValueError('Unknown activation function: ' + repr(actfunc))
            setattr(self, 'actfunc_' + str(i), actfunc)
            input_size = size

    def forward(self, x):
        i = 0
        while hasattr(self, 'layer_' + str(i)):
            x = getattr(self, 'layer_' + str(i))(x)
            f = getattr(self, 'actfunc_' + str(i))
            if f is not None:
                x = f(x)
            i += 1
        return x


class StructuredControlNet(nn.Module):
    """Structured Control Net.

    This is a control network consisting of two components:
    (i) a non-linear component, which is a feed-forward network; and
    (ii) a linear component, which is a linear layer.
    Both components take the input vector provided to the
    structured control network.
    The final output is the sum of the outputs of both components.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(self, *, in_features: int, out_features: int, num_layers: int, hidden_size: int, bias: bool=True, nonlinearity: Union[str, Callable]='tanh'):
        """`__init__(...)`: Initialize the structured control net.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            num_layers: Number of hidden layers for the non-linear component
            hidden_size: Number of neurons in a hidden layer of the
                non-linear component
            bias: Whether or not the linear component is to have bias
            nonlinearity: Activation function
        """
        nn.Module.__init__(self)
        self._in_features = in_features
        self._out_features = out_features
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._bias = bias
        self._nonlinearity = nonlinearity
        self._linear_component = nn.Linear(in_features=self._in_features, out_features=self._out_features, bias=self._bias)
        self._nonlinear_component = FeedForwardNet(input_size=self._in_features, layers=list((self._hidden_size, self._nonlinearity) for _ in range(self._num_layers)) + [(self._out_features, self._nonlinearity)])

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """TODO: documentation"""
        return self._linear_component(x) + self._nonlinear_component(x)

    @property
    def in_features(self):
        """TODO: documentation"""
        return self._in_features

    @property
    def out_features(self):
        """TODO: documentation"""
        return self._out_features

    @property
    def num_layers(self):
        """TODO: documentation"""
        return self._num_layers

    @property
    def hidden_size(self):
        """TODO: documentation"""
        return self._hidden_size

    @property
    def bias(self):
        """TODO: documentation"""
        return self._bias

    @property
    def nonlinearity(self):
        """TODO: documentation"""
        return self._nonlinearity


class LocomotorNet(nn.Module):
    """LocomotorNet: A locomotion-specific structured control net.

    This is a control network which consists of two components:
    one linear, and one non-linear. The non-linear component
    is an input-independent set of sinusoidals waves whose
    amplitudes, frequencies and phases are trainable.
    Upon execution of a forward pass, the output of the non-linear
    component is the sum of all these sinusoidal waves.
    The linear component is a linear layer (optionally with bias)
    whose weights (and biases) are trainable.
    The final output of the LocomotorNet at the end of a forward pass
    is the sum of the linear and the non-linear components.

    Note that this is a stateful network, where the only state
    is the timestep t, which starts from 0 and gets incremented by 1
    at the end of each forward pass. The `reset()` method resets
    t back to 0.

    Reference:
        Mario Srouji, Jian Zhang, Ruslan Salakhutdinov (2018).
        Structured Control Nets for Deep Reinforcement Learning.
    """

    def __init__(self, *, in_features: int, out_features: int, bias: bool=True, num_sinusoids=16):
        """`__init__(...)`: Initialize the LocomotorNet.

        Args:
            in_features: Length of the input vector
            out_features: Length of the output vector
            bias: Whether or not the linear component is to have a bias
            num_sinusoids: Number of sinusoidal waves
        """
        nn.Module.__init__(self)
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._num_sinusoids = num_sinusoids
        self._linear_component = nn.Linear(in_features=self._in_features, out_features=self._out_features, bias=self._bias)
        self._amplitudes = nn.ParameterList()
        self._frequencies = nn.ParameterList()
        self._phases = nn.ParameterList()
        for _ in range(self._num_sinusoids):
            for paramlist in (self._amplitudes, self._frequencies, self._phases):
                paramlist.append(nn.Parameter(torch.randn(self._out_features, dtype=torch.float32)))
        self.reset()

    def reset(self):
        """Set the timestep t to 0"""
        self._t = 0

    @property
    def t(self) ->int:
        """The current timestep t"""
        return self._t

    @property
    def in_features(self) ->int:
        """Get the length of the input vector"""
        return self._in_features

    @property
    def out_features(self) ->int:
        """Get the length of the output vector"""
        return self._out_features

    @property
    def num_sinusoids(self) ->int:
        """Get the number of sinusoidal waves of the non-linear component"""
        return self._num_sinusoids

    @property
    def bias(self) ->bool:
        """Get whether or not the linear component has bias"""
        return self._bias

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Execute a forward pass"""
        u_linear = self._linear_component(x)
        t = self._t
        u_nonlinear = torch.zeros(self._out_features)
        for i in range(self._num_sinusoids):
            A = self._amplitudes[i]
            w = self._frequencies[i]
            phi = self._phases[i]
            u_nonlinear = u_nonlinear + A * torch.sin(w * t + phi)
        self._t += 1
        return u_linear + u_nonlinear


class MultiLayered(nn.Module):

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self._submodules = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, h: Optional[dict]=None):
        if h is None:
            h = {}
        new_h = {}
        for i, layer in enumerate(self._submodules):
            layer_h = h.get(i, None)
            if layer_h is None:
                layer_result = layer(x)
            else:
                layer_result = layer(x, h[i])
            if isinstance(layer_result, tuple):
                if len(layer_result) == 2:
                    x, layer_new_h = layer_result
                else:
                    raise ValueError(f'The layer number {i} returned a tuple of length {len(layer_result)}. A tensor or a tuple of two elements was expected.')
            elif isinstance(layer_result, torch.Tensor):
                x = layer_result
                layer_new_h = None
            else:
                raise TypeError(f'The layer number {i} returned an object of type {type(layer_result)}. A tensor or a tuple of two elements was expected.')
            if layer_new_h is not None:
                new_h[i] = layer_new_h
        if len(new_h) == 0:
            return x
        else:
            return x, new_h

    def __iter__(self):
        return self._submodules.__iter__()

    def __getitem__(self, i):
        return self._submodules[i]

    def __len__(self):
        return len(self._submodules)

    def append(self, module: nn.Module):
        self._submodules.append(module)


def device_of_module(m: nn.Module, default: Optional[Union[str, torch.device]]=None) ->torch.device:
    """
    Get the device in which the module exists.

    This function looks at the first parameter of the module, and returns
    its device. This function is not meant to be used on modules whose
    parameters exist on different devices.

    Args:
        m: The module whose device is being queried.
        default: The fallback device to return if the module has no
            parameters. If this is left as None, the fallback device
            is assumed to be "cpu".
    Returns:
        The device of the module, determined from its first parameter.
    """
    if default is None:
        default = torch.device('cpu')
    device = default
    for p in m.parameters():
        device = p.device
        break
    return device


CollectedStats = namedtuple('CollectedStats', ['mean', 'stdev'])


DType = Union[str, torch.dtype, np.dtype, Type]


Device = Union[str, torch.device]


def _clamp(x: torch.Tensor, min: Optional[float], max: Optional[float]) ->torch.Tensor:
    """
    Clamp the tensor x according to the given min and max values.
    Unlike PyTorch's clamp, this function allows both min and max
    to be None, in which case no clamping will be done.

    Args:
        x: The tensor subject to the clamp operation.
        min: The minimum value.
        max: The maximum value.
    Returns:
        The result of the clamp operation, as a tensor.
        If both min and max were None, the returned object is x itself.
    """
    if min is None and max is None:
        return x
    else:
        return torch.clamp(x, min, max)


class ObsNormLayer(nn.Module):
    """
    An observation normalizer which behaves as a PyTorch Module.
    """

    def __init__(self, mean: torch.Tensor, stdev: torch.Tensor, low: Optional[float]=None, high: Optional[float]=None) ->None:
        """
        `__init__(...)`: Initialize the ObsNormLayer.

        Args:
            mean: The mean according to which the observations are to be
                normalized.
            stdev: The standard deviation according to which the observations
                are to be normalized.
            low: Optionally a real number if the result of the normalization
                is to be clipped. Represents the lower bound for the clipping
                operation.
            high: Optionally a real number if the result of the normalization
                is to be clipped. Represents the upper bound for the clipping
                operation.
        """
        super().__init__()
        self.register_buffer('_mean', mean)
        self.register_buffer('_stdev', stdev)
        self._lb = None if low is None else float(low)
        self._ub = None if high is None else float(high)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Normalize an observation or a batch of observations.

        Args:
            x: The observation(s).
        Returns:
            The normalized counterpart of the observation(s).
        """
        return _clamp((x - self._mean) / self._stdev, self._lb, self._ub)


def to_torch_dtype(dtype: DType) ->torch.dtype:
    """
    Convert the given string or the given numpy dtype to a PyTorch dtype.
    If the argument is already a PyTorch dtype, then the argument is returned
    as it is.

    Returns:
        The dtype, converted to a PyTorch dtype.
    """
    if isinstance(dtype, str) and hasattr(torch, dtype):
        attrib_within_torch = getattr(torch, dtype)
    else:
        attrib_within_torch = None
    if isinstance(attrib_within_torch, torch.dtype):
        return attrib_within_torch
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif dtype is Any or dtype is object:
        raise TypeError(f'Cannot make a numeric tensor with dtype {repr(dtype)}')
    else:
        return torch.from_numpy(np.array([], dtype=dtype)).dtype


class RunningNorm:
    """
    An online observation normalization tool
    """

    def __init__(self, *, shape: Union[tuple, int], dtype: DType, device: Optional[Device]=None, min_variance: float=0.01, clip: Optional[tuple]=None) ->None:
        """
        `__init__(...)`: Initialize the RunningNorm

        Args:
            shape: Observation shape. Can be an integer or a tuple.
            dtype: The dtype of the observations.
            device: The device in which the observation stats are held.
                If left as None, the device is assumed to be "cpu".
            min_variance: A lower bound for the variance to be used in
                the normalization computations.
                In other words, if the computed variance according to the
                collected observations ends up lower than `min_variance`,
                this `min_variance` will be used instead (in an elementwise
                manner) while computing the normalized observations.
                As in Salimans et al. (2017), the default is 1e-2.
            clip: Can be left as None (which is the default), or can be
                given as a pair of real numbers.
                This is used for clipping the observations after the
                normalization operation.
                In Salimans et al. (2017), (-5.0, +5.0) was used.
        """
        if isinstance(shape, Iterable):
            self._shape = torch.Size(shape)
        else:
            self._shape = torch.Size([int(shape)])
        self._ndim = len(self._shape)
        self._dtype = to_torch_dtype(dtype)
        self._device = 'cpu' if device is None else device
        self._sum: Optional[torch.Tensor] = None
        self._sum_of_squares: Optional[torch.Tensor] = None
        self._count: int = 0
        self._min_variance = float(min_variance)
        if clip is not None:
            lb, ub = clip
            self._lb = float(lb)
            self._ub = float(ub)
        else:
            self._lb = None
            self._ub = None

    def to(self, device: Device) ->'RunningNorm':
        """
        If the target device is a different device, then make a copy of this
        RunningNorm instance on the target device.
        If the target device is the same with this RunningNorm's device, then
        return this RunningNorm itself.

        Args:
            device: The target device.
        Returns:
            The RunningNorm on the target device. This can be a copy, or the
            original RunningNorm instance itself.
        """
        if torch.device(device) == torch.device(self.device):
            return self
        else:
            new_running_norm = object.__new__(type(self))
            already_handled = {'_sum', '_sum_of_squares', '_device'}
            new_running_norm._sum = self._sum
            new_running_norm._sum_of_squares = self._sum_of_squares
            new_running_norm._device = device
            for k, v in self.__dict__.items():
                if k not in already_handled:
                    setattr(new_running_norm, k, deepcopy(v))
            return new_running_norm

    @property
    def device(self) ->Device:
        """
        The device in which the observation stats are held
        """
        return self._device

    @property
    def dtype(self) ->DType:
        """
        The dtype of the stored observation stats
        """
        return self._dtype

    @property
    def shape(self) ->tuple:
        """
        Observation shape
        """
        return self._shape

    @property
    def min_variance(self) ->float:
        """
        Minimum variance
        """
        return self._min_variance

    @property
    def low(self) ->Optional[float]:
        """
        The lower component of the bounds given in the `clip` tuple.
        If `clip` was initialized as None, this is also None.
        """
        return self._lb

    @property
    def high(self) ->Optional[float]:
        """
        The higher (upper) component of the bounds given in the `clip` tuple.
        If `clip` was initialized as None, this is also None.
        """
        return self._ub

    def _like_its_own(self, x: Iterable) ->torch.Tensor:
        return torch.as_tensor(x, dtype=self._dtype, device=self._device)

    def _verify(self, x: Iterable) ->torch.Tensor:
        x = self._like_its_own(x)
        if x.ndim == self._ndim:
            if x.shape != self._shape:
                raise ValueError(f'This RunningNorm instance was initialized with shape: {self._shape}. However, the provided tensor has an incompatible shape: {x._shape}.')
        elif x.ndim == self._ndim + 1:
            if x.shape[1:] != self._shape:
                raise ValueError(f"This RunningNorm instance was initialized with shape: {self._shape}. The provided tensor is shaped {x.shape}. Accepting the tensor's leftmost dimension as the batch size, the remaining shape is incompatible: {x.shape[1:]}")
        else:
            raise ValueError(f'This RunningNorm instance was initialized with shape: {self._shape}. The provided tensor is shaped {x.shape}. The number of dimensions of the given tensor is incompatible.')
        return x

    def _has_no_data(self) ->bool:
        return self._sum is None and self._sum_of_squares is None and self._count == 0

    def _has_data(self) ->bool:
        return self._sum is not None and self._sum_of_squares is not None and self._count > 0

    def reset(self):
        """
        Remove all the collected observation data.
        """
        self._sum = None
        self._sum_of_squares = None
        self._count = 0

    @torch.no_grad()
    def update(self, x: Union[Iterable, 'RunningNorm'], mask: Optional[Iterable]=None, *, verify: bool=True):
        """
        Update the stored stats with new observation data.

        Args:
            x: The new observation(s), as a PyTorch tensor, or any Iterable
                that can be converted to a PyTorch tensor, or another
                RunningNorm instance.
                If given as a tensor or as an Iterable, the shape of `x` can
                be the same with observation shape, or it can be augmented
                with an extra leftmost dimension.
                In the case of augmented dimension, `x` is interpreted not as
                a single observation, but as a batch of observations.
                If `x` is another RunningNorm instance, the stats stored by
                this RunningNorm instance will be updated with all the data
                stored by `x`.
            mask: Can be given as a 1-dimensional Iterable of booleans ONLY
                if `x` represents a batch of observations.
                If a `mask` is provided, the i-th observation within the
                observation batch `x` will be taken into account only if
                the i-th item of the `mask` is True.
            verify: Whether or not to verify the shape of the given Iterable
                objects. The default is True.
        """
        if isinstance(x, RunningNorm):
            if x._count > 0:
                if mask is not None:
                    raise ValueError('The `mask` argument is expected as None if the first argument is a RunningNorm. However, `mask` is found as something other than None.')
                if self._shape != x._shape:
                    raise ValueError(f'The RunningNorm to be updated has the shape {self._shape} The other RunningNorm has the shape {self._shape} These shapes are incompatible.')
                if self._has_no_data():
                    self._sum = self._like_its_own(x._sum.clone())
                    self._sum_of_squares = self._like_its_own(x._sum_of_squares.clone())
                    self._count = x._count
                elif self._has_data():
                    self._sum += self._like_its_own(x._sum)
                    self._sum_of_squares += self._like_its_own(x._sum_of_squares)
                    self._count += x._count
                else:
                    assert False, 'RunningNorm is in an invalid state! This might be a bug.'
        else:
            if verify:
                x = self._verify(x)
            if x.ndim == self._ndim:
                if mask is not None:
                    raise ValueError('The `mask` argument is expected as None if the first argument is a single observation (i.e. not a batch of observations, with an extra leftmost dimension). However, `mask` is found as something other than None.')
                sum_of_x = x
                sum_of_x_squared = x.square()
                n = 1
            elif x.ndim == self._ndim + 1:
                if mask is not None:
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=self._device)
                    if mask.ndim != 1:
                        raise ValueError(f'The `mask` tensor was expected as a 1-dimensional tensor. However, its shape is {mask.shape}.')
                    if len(mask) != x.shape[0]:
                        raise ValueError(f'The shape of the given tensor is {x.shape}. Therefore, the batch size of observations is {x.shape[0]}. However, the given `mask` tensor does not has an incompatible length: {len(mask)}.')
                    n = int(torch.sum(torch.as_tensor(mask, dtype=torch.int64, device=self._device)))
                    mask = self._like_its_own(mask).reshape(torch.Size([x.shape[0]] + [1] * (x.ndim - 1)))
                    x = x * mask
                else:
                    n = x.shape[0]
                sum_of_x = torch.sum(x, dim=0)
                sum_of_x_squared = torch.sum(x.square(), dim=0)
            else:
                raise ValueError(f'Invalid shape: {x.shape}')
            if self._has_no_data():
                self._sum = sum_of_x
                self._sum_of_squares = sum_of_x_squared
                self._count = n
            elif self._has_data():
                self._sum += sum_of_x
                self._sum_of_squares += sum_of_x_squared
                self._count += n
            else:
                assert False, 'RunningNorm is in an invalid state! This might be a bug.'

    @property
    @torch.no_grad()
    def stats(self) ->CollectedStats:
        """
        The collected data's mean and standard deviation (stdev) in a tuple
        """
        E_x = self._sum / self._count
        E_x2 = self._sum_of_squares / self._count
        mean = E_x
        variance = _clamp(E_x2 - E_x.square(), self._min_variance, None)
        stdev = torch.sqrt(variance)
        return CollectedStats(mean=mean, stdev=stdev)

    @property
    def mean(self) ->torch.Tensor:
        """
        The collected data's mean
        """
        return self._sum / self._count

    @property
    def stdev(self) ->torch.Tensor:
        """
        The collected data's standard deviation
        """
        return self.stats.stdev

    @property
    def sum(self) ->torch.Tensor:
        """
        The collected data's sum
        """
        return self._sum

    @property
    def sum_of_squares(self) ->torch.Tensor:
        """
        Sum of squares of the collected data
        """
        return self._sum_of_squares

    @property
    def count(self) ->int:
        """
        Number of observations encountered
        """
        return self._count

    @torch.no_grad()
    def normalize(self, x: Iterable, *, result_as_numpy: Optional[bool]=None, verify: bool=True) ->Iterable:
        """
        Normalize the given observation x.

        Args:
            x: The observation(s), as a PyTorch tensor, or any Iterable
                that is convertable to a PyTorch tensor.
                `x` can be a single observation, or it can be a batch
                of observations (with an extra leftmost dimension).
            result_as_numpy: Whether or not to return the normalized
                observation as a numpy array.
                If left as None (which is the default), then the returned
                type depends on x: a PyTorch tensor is returned if x is a
                PyTorch tensor, and a numpy array is returned otherwise.
                If True, the result is always a numpy array.
                If False, the result is always a PyTorch tensor.
            verify: Whether or not to check the type and dimensions of x.
                This is True by default.
                Note that, if `verify` is False, this function will not
                properly check the type of `x` and will assume that `x`
                is a PyTorch tensor.
        Returns:
            The normalized observation, as a PyTorch tensor or a numpy array.
        """
        if self._count == 0:
            raise ValueError('Cannot do normalization because no data is collected yet.')
        if verify:
            if result_as_numpy is None:
                result_as_numpy = not isinstance(x, torch.Tensor)
            else:
                result_as_numpy = bool(result_as_numpy)
            x = self._verify(x)
        mean, stdev = self.stats
        result = _clamp((x - mean) / stdev, self._lb, self._ub)
        if result_as_numpy:
            result = result.cpu().numpy()
        return result

    @torch.no_grad()
    def update_and_normalize(self, x: Iterable, mask: Optional[Iterable]=None) ->Iterable:
        """
        Update the observation stats according to x, then normalize x.

        Args:
            x: The observation(s), as a PyTorch tensor, or as an Iterable
                which can be converted to a PyTorch tensor.
                The shape of x can be the same with the observaiton shape,
                or it can be augmented with an extra leftmost dimension
                to express a batch of observations.
            mask: Can be given as a 1-dimensional Iterable of booleans ONLY
                if `x` represents a batch of observations.
                If a `mask` is provided, the i-th observation within the
                observation batch `x` will be taken into account only if
                the the i-th item of the `mask` is True.
        Returns:
            The normalized counterpart of the observation(s) expressed by x.
        """
        result_as_numpy = not isinstance(x, torch.Tensor)
        x = self._verify(x)
        self.update(x, mask, verify=False)
        result = self.normalize(x, verify=False)
        if result_as_numpy:
            result = result.cpu().numpy()
        return result

    def to_layer(self) ->'ObsNormLayer':
        """
        Make a PyTorch module which normalizes the its inputs.

        Returns:
            An ObsNormLayer instance.
        """
        mean, stdev = self.stats
        low = self.low
        high = self.high
        return ObsNormLayer(mean=mean, stdev=stdev, low=low, high=high)

    def __repr__(self) ->str:
        return f'<{self.__class__.__name__}, count: {self.count}>'

    def __copy__(self) ->'RunningNorm':
        return deepcopy(self)


class RunningStat:
    """
    Tool for efficiently computing the mean and stdev of arrays.
    The arrays themselves are not stored separately,
    instead, they are accumulated.

    This RunningStat is implemented as a wrapper around RunningNorm.
    The difference is that the interface of RunningStat is simplified
    to expect only numpy arrays, and expect only non-vectorized
    observations.
    With this simplified interface, RunningStat is meant to be used
    by GymNE, on classical non-vectorized gym tasks.
    """

    def __init__(self):
        """
        `__init__(...)`: Initialize the RunningStat.
        """
        self._rn: Optional[RunningNorm] = None
        self.reset()

    def reset(self):
        """
        Reset the RunningStat to its initial state.
        """
        self._rn = None

    @property
    def count(self) ->int:
        """
        Get the number of arrays accumulated.
        """
        if self._rn is None:
            return 0
        else:
            return self._rn.count

    @property
    def sum(self) ->np.ndarray:
        """
        Get the sum of all accumulated arrays.
        """
        return self._rn.sum.numpy()

    @property
    def sum_of_squares(self) ->np.ndarray:
        """
        Get the sum of squares of all accumulated arrays.
        """
        return self._rn.sum_of_squares.numpy()

    @property
    def mean(self) ->np.ndarray:
        """
        Get the mean of all accumulated arrays.
        """
        return self._rn.mean.numpy()

    @property
    def stdev(self) ->np.ndarray:
        """
        Get the standard deviation of all accumulated arrays.
        """
        return self._rn.stdev.numpy()

    def update(self, x: Union[np.ndarray, 'RunningStat']):
        """
        Accumulate more data into the RunningStat object.
        If the argument is an array, that array is added
        as one more data element.
        If the argument is another RunningStat instance,
        all the stats accumulated by that RunningStat object
        are added into this RunningStat object.
        """
        if isinstance(x, RunningStat):
            if x.count > 0:
                if self._rn is None:
                    self._rn = deepcopy(x._rn)
                else:
                    self._rn.update(x._rn)
        else:
            if self._rn is None:
                x = np.array(x, dtype='float32')
                self._rn = RunningNorm(shape=x.shape, dtype='float32', device='cpu')
            self._rn.update(x)

    def normalize(self, x: Union[np.ndarray, list]) ->np.ndarray:
        """
        Normalize the array x according to the accumulated stats.
        """
        if self._rn is None:
            return x
        else:
            x = np.array(x, dtype='float32')
            return self._rn.normalize(x)

    def __copy__(self):
        return deepcopy(self)

    def __repr__(self) ->str:
        return f'<{self.__class__.__name__}, count: {self.count}>'

    def to(self, device: Union[str, torch.device]) ->'RunningStat':
        """
        If the target device is cpu, return this RunningStat instance itself.
        A RunningStat object is meant to work with numpy arrays. Therefore,
        any device other than the cpu will trigger an error.

        Args:
            device: The target device. Only cpu is supported.
        Returns:
            The original RunningStat.
        """
        if torch.device(device) == torch.device('cpu'):
            return self
        else:
            raise ValueError(f'The received target device is {repr(device)}. However, RunningStat can only work on a cpu.')

    def to_layer(self) ->nn.Module:
        """
        Make a PyTorch module which normalizes the its inputs.

        Returns:
            An ObsNormLayer instance.
        """
        return self._rn.to_layer()


class ObsNormWrapperModule(nn.Module):

    def __init__(self, wrapped_module: nn.Module, rn: Union[RunningStat, RunningNorm]):
        super().__init__()
        device = device_of_module(wrapped_module)
        self.wrapped_module = wrapped_module
        with torch.no_grad():
            normalizer = deepcopy(rn.to_layer())
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor, h: Any=None) ->Union[torch.Tensor, tuple]:
        x = self.normalizer(x)
        if h is None:
            result = self.wrapped_module(x)
        else:
            result = self.wrapped_module(x, h)
        if isinstance(result, tuple):
            x, h = result
            got_h = True
        else:
            x = result
            h = None
            got_h = False
        if got_h:
            return x, h
        else:
            return x


class StatefulModule(nn.Module):
    """
    A wrapper that provides a stateful interface for recurrent torch modules.

    If the torch module to be wrapped is non-recurrent and its forward method
    has a single input (the input tensor) and a single output (the output
    tensor), then this wrapper module acts as a no-op wrapper.

    If the torch module to be wrapped is recurrent and its forward method has
    two inputs (the input tensor and an optional second argument for the hidden
    state) and two outputs (the output tensor and the new hidden state), then
    this wrapper brings a new forward-passing interface. In this new interface,
    the forward method has a single input (the input tensor) and a single
    output (the output tensor). The hidden states, instead of being
    explicitly requested via a second argument and returned as a second
    result, are stored and used by the wrapper.
    When a new series of inputs is to be used, one has to call the `reset()`
    method of this wrapper.
    """

    def __init__(self, wrapped_module: nn.Module):
        """
        `__init__(...)`: Initialize the StatefulModule.

        Args:
            wrapped_module: The `torch.nn.Module` instance to wrap.
        """
        super().__init__()
        self._hidden: Any = None
        self.wrapped_module = wrapped_module

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self._hidden is None:
            out = self.wrapped_module(x)
        else:
            out = self.wrapped_module(x, self._hidden)
        if isinstance(out, tuple):
            y, self._hidden = out
        else:
            y = out
            self._hidden = None
        return y

    def reset(self):
        """
        Reset the hidden state, if any.
        """
        self._hidden = None


class DummyRecurrentNet(nn.Module):

    def __init__(self, first_value: int=1):
        super().__init__()
        self.first_value = int(first_value)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]=None) ->tuple:
        if h is None:
            h = torch.tensor(self.first_value, dtype=torch.int64, device=x.device)
        return x * torch.as_tensor(h, dtype=x.dtype, device=x.device), h + 1


class Unbatched:


    class LSTM(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.lstm = nn.LSTM(*args, **kwargs)

        def forward(self, x: torch.Tensor, h: Optional[tuple]=None) ->tuple:
            if h is not None:
                a, b = h
                a = a.reshape(1, 1, -1)
                b = b.reshape(1, 1, -1)
                h = a, b
            x = x.reshape(1, 1, -1)
            x, h = self.lstm(x, h)
            x = x.reshape(-1)
            a, b = h
            a = a.reshape(-1)
            b = b.reshape(-1)
            h = a, b
            return x, h


    class RNN(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.rnn = nn.RNN(*args, **kwargs)

        def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]=None) ->tuple:
            if h is not None:
                h = h.reshape(1, 1, -1)
            x = x.reshape(1, 1, -1)
            x, h = self.rnn(x, h)
            x = x.reshape(-1)
            h = h.reshape(-1)
            return x, h


class DummyComposedRecurrent(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Unbatched.RNN(3, 5), nn.Linear(5, 8), Unbatched.LSTM(8, 2)])

    def forward(self, x: torch.Tensor, h: Optional[dict]=None) ->tuple:
        if h is None:
            h = {(0): None, (2): None}
        x, h[0] = self.layers[0](x, h[0])
        x = self.layers[1](x)
        x = torch.tanh(x)
        x, h[2] = self.layers[2](x, h[2])
        return x, h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Apply,
     lambda: ([], {'operator': '+', 'argument': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bin,
     lambda: ([], {'lb': 4, 'ub': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Clip,
     lambda: ([], {'lb': 4, 'ub': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyRecurrentNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LocomotorNet,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiLayered,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Round,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Slice,
     lambda: ([], {'from_index': 4, 'to_index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StatefulModule,
     lambda: ([], {'wrapped_module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StructuredControlNet,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'num_layers': 1, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_nnaisense_evotorch(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

