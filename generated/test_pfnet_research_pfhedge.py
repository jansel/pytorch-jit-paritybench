import sys
_module = sys.modules[__name__]
del sys
conf = _module
example_adding_clause = _module
example_autogreek = _module
example_black_scholes = _module
example_expected_shortfall = _module
example_hedging_variance_swap = _module
example_heston_iv = _module
example_knockout = _module
example_minimal = _module
example_module_output = _module
example_multi_layer_perceptron = _module
example_multiple_hedge = _module
example_no_transaction_band = _module
example_plot_american_binary = _module
example_plot_european = _module
example_plot_european_binary = _module
example_plot_lookback = _module
example_readme = _module
example_svi = _module
example_whalley_wilmott = _module
pfhedge = _module
_utils = _module
bisect = _module
doc = _module
hook = _module
lazy = _module
operations = _module
parse = _module
str = _module
testing = _module
typing = _module
autogreek = _module
features = _module
_base = _module
_getter = _module
container = _module
features = _module
instruments = _module
base = _module
derivative = _module
american_binary = _module
base = _module
cliquet = _module
european = _module
european_binary = _module
lookback = _module
variance_swap = _module
primary = _module
base = _module
brownian = _module
cir = _module
heston = _module
local_volatility = _module
vasicek = _module
nn = _module
functional = _module
modules = _module
bs = _module
_base = _module
american_binary = _module
black_scholes = _module
european = _module
european_binary = _module
lookback = _module
clamp = _module
hedger = _module
loss = _module
mlp = _module
naked = _module
svi = _module
ww = _module
stochastic = _module
_utils = _module
brownian = _module
cir = _module
engine = _module
heston = _module
local_volatility = _module
random = _module
vasicek = _module
version = _module
tests = _module
test_bisect = _module
test_lazy = _module
test_operations = _module
test_parse = _module
test_features = _module
test_getter = _module
test_american_binary = _module
test_cliquet = _module
test_european = _module
test_european_binary = _module
test_lookback = _module
test_variance_swap = _module
test_base = _module
test_brownian = _module
test_cir = _module
test_heston = _module
test_local_volatility = _module
test_vasicek = _module
_base = _module
_utils = _module
test_american_binary = _module
test_bs = _module
test_european = _module
test_european_binary = _module
test_lookback = _module
test_clamp = _module
test_hedger = _module
test_loss = _module
test_mlp = _module
test_naked = _module
test_svi = _module
test_ww = _module
test_functional = _module
test_brownian = _module
test_cir = _module
test_heston = _module
test_randn = _module
test_autogreek = _module
test_examples = _module
test_version = _module

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


from math import sqrt


import torch


import matplotlib.pyplot as plt


from math import exp


from torch import Tensor


import torch.nn.functional as fn


from torch.nn import Module


from typing import Any


from typing import Callable


from typing import Union


from typing import Optional


from torch.nn.parameter import is_lazy


from numbers import Real


from torch.testing import assert_close


from inspect import signature


import copy


from abc import ABC


from abc import abstractmethod


from typing import TypeVar


from typing import List


from typing import Type


from typing import no_type_check


from collections import OrderedDict


from typing import Dict


from typing import Iterator


from typing import Tuple


from math import floor


from math import ceil


from typing import cast


from math import pi as kPI


from torch.distributions.normal import Normal


from torch.distributions.utils import broadcast_all


from torch.optim import Adam


from torch.optim import Optimizer


from torch.nn.parameter import Parameter


from copy import deepcopy


from typing import Sequence


from torch.nn import Identity


from torch.nn import LazyLinear


from torch.nn import Linear


from torch.nn import ReLU


from torch.nn import Sequential


from torch.quasirandom import SobolEngine


from collections import namedtuple


from torch.nn.functional import relu


from torch.testing import assert_allclose


from torch.distributions.gamma import Gamma


def leaky_clamp(input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None, clamped_slope: float=0.01, inverted_output: str='mean') ->Tensor:
    """Leakily clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    See :class:`pfhedge.nn.LeakyClamp` for details.
    """
    x = input
    if min is not None:
        min = torch.as_tensor(min)
        x = x.maximum(min + clamped_slope * (x - min))
    if max is not None:
        max = torch.as_tensor(max)
        x = x.minimum(max + clamped_slope * (x - max))
    if min is not None and max is not None:
        if inverted_output == 'mean':
            y = (min + max) / 2
        elif inverted_output == 'max':
            y = max
        else:
            raise ValueError("inverted_output must be 'mean' or 'max'.")
        x = x.where(min <= max, y)
    return x


def clamp(input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None, inverted_output: str='mean') ->Tensor:
    """Clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    See :class:`pfhedge.nn.Clamp` for details.
    """
    if inverted_output == 'mean':
        output = leaky_clamp(input, min, max, clamped_slope=0.0, inverted_output='mean')
    elif inverted_output == 'max':
        output = torch.clamp(input, min, max)
    else:
        raise ValueError("inverted_output must be 'mean' or 'max'.")
    return output


class Clamp(Module):
    """Clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    If :math:`\\min \\leq \\max`:

    .. math::

        \\text{output} = \\begin{cases}
        \\min & \\text{input} < \\min \\\\
        \\text{input} & \\min \\leq \\text{input} \\leq \\max \\\\
        \\max & \\max < \\text{input}
        \\end{cases}

    If :math:`\\min > \\max`:

    .. math::

        \\text{output} = \\begin{cases}
            \\frac12 (\\min + \\max)
            & \\text{inverted_output} = \\text{'mean'} \\\\
            \\max
            & \\text{inverted_output} = \\text{'max'} \\\\
        \\end{cases}

    .. seealso::
        - :func:`torch.clamp`
        - :func:`pfhedge.nn.functional.clamp`

    Args:
        inverted_output ({'mean', ''max'}, default='mean'):
            Controls the output when :math:`\\min > \\max`.
            'max' is consistent with :func:`torch.clamp`.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - min: :math:`(N, *)` or any size broadcastable to ``input``.
        - max: :math:`(N, *)` or any size broadcastable to ``input``.
        - output: :math:`(N, *)`, same shape as the input.

    Examples:

        >>> import torch
        >>> from pfhedge.nn import Clamp
        >>>
        >>> m = Clamp()
        >>> input = torch.linspace(-2, 12, 15) * 0.1
        >>> input
        tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
                 0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
        >>> m(input, 0.0, 1.0)
        tensor([0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
                0.7000, 0.8000, 0.9000, 1.0000, 1.0000, 1.0000])

        When :math:`\\min > \\max`, returns the mean of :math:`\\min` and :math:`\\max`.

        >>> input = torch.tensor([1.0, 0.0])
        >>> min = torch.tensor([0.0, 1.0])
        >>> max = torch.tensor([0.0, 0.0])
        >>> m(input, min, max)
        tensor([0.0000, 0.5000])
    """

    def forward(self, input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None) ->Tensor:
        """Clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

        Args:
            input (torch.Tensor): The input tensor.
            min (torch.Tensor, optional): Lower-bound of the range to be clamped to.
            max (torch.Tensor, optional): Upper-bound of the range to be clamped to.

        Returns:
            torch.Tensor
        """
        return clamp(input, min=min, max=max)


class MultiLayerPerceptron(Sequential):
    """Creates a multilayer perceptron.

    Number of input features is lazily determined.

    Args:
        in_features (int, optional): Size of each input sample.
            If ``None`` (default), the number of input features will be
            will be inferred from the ``input.shape[-1]`` after the first call to
            ``forward`` is done. Also, before the first ``forward`` parameters in the
            module are of :class:`torch.nn.UninitializedParameter` class.
        out_features (int, default=1): Size of each output sample.
        n_layers (int, default=4): The number of hidden layers.
        n_units (int or tuple[int], default=32): The number of units in
            each hidden layer.
            If ``tuple[int]``, it specifies different number of units for each layer.
        activation (torch.nn.Module, default=torch.nn.ReLU()):
            The activation module of the hidden layers.
            Default is a :class:`torch.nn.ReLU` instance.
        out_activation (torch.nn.Module, default=torch.nn.Identity()):
            The activation module of the output layer.
            Default is a :class:`torch.nn.Identity` instance.

    Shape:
        - Input: :math:`(N, *, H_{\\text{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\\text{in}}` is the number of input features.
        - Output: :math:`(N, *, H_{\\text{out}})` where
          all but the last dimension are the same shape as the input and
          :math:`H_{\\text{out}}` is the number of output features.

    Examples:

        By default, ``in_features`` is lazily determined:

        >>> import torch
        >>> from pfhedge.nn import MultiLayerPerceptron
        >>>
        >>> m = MultiLayerPerceptron()
        >>> m
        MultiLayerPerceptron(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )
        >>> _ = m(torch.zeros(3, 2))
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=2, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )

        Specify different number of layers for each layer:

        >>> m = MultiLayerPerceptron(1, 1, n_layers=2, n_units=(16, 32))
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=1, out_features=16, bias=True)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=1, bias=True)
          (5): Identity()
        )
    """

    def __init__(self, in_features: Optional[int]=None, out_features: int=1, n_layers: int=4, n_units: Union[int, Sequence[int]]=32, activation: Module=ReLU(), out_activation: Module=Identity()):
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units
        layers: List[Module] = []
        for i in range(n_layers):
            if i == 0:
                if in_features is None:
                    layers.append(LazyLinear(n_units[0]))
                else:
                    layers.append(Linear(in_features, n_units[0]))
            else:
                layers.append(Linear(n_units[i - 1], n_units[i]))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units[-1], out_features))
        layers.append(deepcopy(out_activation))
        super().__init__(*layers)


class NoTransactionBandNet(Module):
    """Initialize a no-transaction band network.

    The `forward` method returns the next hedge ratio.

    Args:
        derivative (pfhedge.instruments.BaseDerivative): The derivative to hedge.

    Shape:
        - Input: :math:`(N, H_{\\text{in}})`, where :math:`(N, H_{\\text{in}})` is the
        number of input features. See `inputs()` for the names of input features.
        - Output: :math:`(N, 1)`.

    Examples:

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption

        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> m = NoTransactionBandNet(derivative)
        >>> m.inputs()
        ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
        >>> input = torch.tensor([
        ...     [-0.05, 0.1, 0.2, 0.5],
        ...     [-0.01, 0.1, 0.2, 0.5],
        ...     [ 0.00, 0.1, 0.2, 0.5],
        ...     [ 0.01, 0.1, 0.2, 0.5],
        ...     [ 0.05, 0.1, 0.2, 0.5]])
        >>> m(input)
        tensor([[0.2232],
                [0.4489],
                [0.5000],
                [0.5111],
                [0.7310]], grad_fn=<SWhereBackward>)
    """

    def __init__(self, derivative):
        super().__init__()
        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ['prev_hedge']

    def forward(self, input: Tensor) ->Tensor:
        prev_hedge = input[..., [-1]]
        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])
        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])
        return self.clamp(prev_hedge, min=min, max=max)


T = TypeVar('T', bound='BasePrimary')


TensorOrScalar = Union[Tensor, float, int]


Clause = Callable[[T, Tensor], Tensor]


def _addindent(string: str, n_spaces: int=2) ->str:
    return '\n'.join(' ' * n_spaces + line for line in string.split('\n'))


TM = TypeVar('TM', bound='ModuleOutput')


def _format_float(value: float) ->str:
    """
    >>> _format_float(1)
    '1'
    >>> _format_float(1.0)
    '1.'
    >>> _format_float(1e-4)
    '1.0000e-04'
    """
    tensor = torch.tensor([value])
    return torch._tensor_str._Formatter(tensor).format(value)


def american_binary_payoff(input: Tensor, call: bool=True, strike: float=1.0) ->Tensor:
    """Returns the payoff of an American binary option.

    .. seealso::
        - :class:`pfhedge.instruments.AmericanBinaryOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return input.max(dim=-1).values >= strike
    else:
        return input.min(dim=-1).values <= strike


def european_binary_payoff(input: Tensor, call: bool=True, strike: float=1.0) ->Tensor:
    """Returns the payoff of a European binary option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanBinaryOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return input[..., -1] >= strike
    else:
        return input[..., -1] <= strike


def european_payoff(input: Tensor, call: bool=True, strike: float=1.0) ->Tensor:
    """Returns the payoff of a European option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input[..., -1] - strike)
    else:
        return fn.relu(strike - input[..., -1])


def lookback_payoff(input: Tensor, call: bool=True, strike: float=1.0) ->Tensor:
    """Returns the payoff of a lookback option with a fixed strike.

    .. seealso::
        - :class:`pfhedge.instruments.LookbackOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input.max(dim=-1).values - strike)
    else:
        return fn.relu(strike - input.min(dim=-1).values)


def d1(log_moneyness: TensorOrScalar, time_to_maturity: TensorOrScalar, volatility: TensorOrScalar) ->Tensor:
    """Returns :math:`d_1` in the Black-Scholes formula.

    .. math::
        d_1 = \\frac{s}{\\sigma \\sqrt{t}} + \\frac{\\sigma \\sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError('all elements in time_to_maturity have to be non-negative')
    if not (v >= 0).all():
        raise ValueError('all elements in volatility have to be non-negative')
    variance = v * t.sqrt()
    output = s / variance + variance / 2
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def d2(log_moneyness: TensorOrScalar, time_to_maturity: TensorOrScalar, volatility: TensorOrScalar) ->Tensor:
    """Returns :math:`d_2` in the Black-Scholes formula.

    .. math::
        d_2 = \\frac{s}{\\sigma \\sqrt{t}} - \\frac{\\sigma \\sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError('all elements in time_to_maturity have to be non-negative')
    if not (v >= 0).all():
        raise ValueError('all elements in volatility have to be non-negative')
    variance = v * t.sqrt()
    output = s / variance - variance / 2
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def ncdf(input: Tensor) ->Tensor:
    """Returns a new tensor with the normal cumulative distribution function.

    .. math::
        \\text{ncdf}(x) =
            \\int_{-\\infty}^x
            \\frac{1}{\\sqrt{2 \\pi}} e^{-\\frac{y^2}{2}} dy

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import ncdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> ncdf(input)
        tensor([0.1587, 0.5000, 1.0000])
    """
    return Normal(0.0, 1.0).cdf(input)


def npdf(input: Tensor) ->Tensor:
    """Returns a new tensor with the normal distribution function.

    .. math::
        \\text{npdf}(x) = \\frac{1}{\\sqrt{2 \\pi}} e^{-\\frac{x^2}{2}}

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import npdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> npdf(input)
        tensor([2.4197e-01, 3.9894e-01, 7.6946e-23])
    """
    return Normal(0.0, 1.0).log_prob(input).exp()


def bs_american_binary_delta(log_moneyness: Tensor, max_log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar) ->Tensor:
    """Returns Black-Scholes delta of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.delta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike
    d1_tensor = d1(s, t, v)
    d2_tensor = d2(s, t, v)
    w = v * t.sqrt()
    p = npdf(d2_tensor).div(spot * w) + ncdf(d1_tensor).div(strike) + npdf(d1_tensor).div(strike * w)
    return p.where(max_log_moneyness < 0, torch.zeros_like(p))


def bs_american_binary_price(log_moneyness: Tensor, max_log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor) ->Tensor:
    """Returns Black-Scholes price of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.price` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    p = ncdf(d2(s, t, v)) + s.exp() * ncdf(d1(s, t, v))
    return p.where(max_log_moneyness < 0, torch.ones_like(p))


def bisect(fn: Callable[[Tensor], Tensor], target: Tensor, lower: Union[float, Tensor], upper: Union[float, Tensor], precision: float=1e-06, max_iter: int=100000) ->Tensor:
    """Perform binary search over a tensor.

    The output tensor approximately satisfies the following relation:

    .. code-block::

        fn(output) = target

    Args:
        fn (callable[[Tensor], Tensor]): A monotone function.
        target (Tensor): Target of function values.
        lower (Tensor or float): Lower bound of binary search.
        upper (Tensor or float): Upper bound of binary search.
        precision (float, default=1e-6): Precision of output.
        max_iter (int, default 100000): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor

    Raises:
        RuntimeError: If the number of iteration exceeds ``max_iter``.

    Examples:

        >>> target = torch.tensor([-1.0, 0.0, 1.0])
        >>> fn = torch.log
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([0.3679, 1.0000, 2.7183])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True

        Monotone decreasing function:

        >>> fn = lambda input: -torch.log(input)
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([2.7183, 1.0000, 0.3679])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True
    """
    lower, upper = map(torch.as_tensor, (lower, upper))
    if not (lower < upper).all():
        raise ValueError('condition lower < upper should be satisfied.')
    if (fn(lower) > fn(upper)).all():
        mf = lambda input: -fn(input)
        return bisect(mf, -target, lower, upper, precision=precision, max_iter=max_iter)
    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(f'Aborting since iteration exceeds max_iter={max_iter}.')
        m = (lower + upper) / 2
        output = fn(m)
        lower = lower.where(output >= target, m)
        upper = upper.where(output < target, m)
    return upper


def find_implied_volatility(pricer: Callable, price: Tensor, lower: float=0.001, upper: float=1.0, precision: float=1e-06, max_iter: int=100, **params: Any) ->Tensor:
    """Find implied volatility by binary search.

    Args:
        pricer (callable): Pricing formula of a derivative.
        price (Tensor): The price of the derivative.
        lower (float, default=0.001): Lower bound of binary search.
        upper (float, default=1.000): Upper bound of binary search.
        precision (float, default=1e-6): Computational precision of the implied
            volatility.
        max_iter (int, default 100): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor
    """
    fn = lambda volatility: pricer(volatility=volatility, **params)
    return bisect(fn, price, lower, upper, precision=precision, max_iter=max_iter)


def bs_european_delta(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True) ->Tensor:
    """Returns Black-Scholes delta of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.delta`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_delta
        ...
        >>> bs_european_delta(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([0.3446, 0.5398, 0.7257])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    delta = ncdf(d1(s, t, v))
    delta = delta - 1 if not call else delta
    return delta


def bs_european_gamma(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar=1.0) ->Tensor:
    """Returns Black-Scholes gamma of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.gamma`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_gamma
        ...
        >>> bs_european_gamma(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([2.0350, 1.9848, 1.5076])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = strike * s.exp()
    numerator = npdf(d1(s, t, v))
    denominator = spot * v * t.sqrt()
    output = numerator / denominator
    return torch.where((numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output)


def bs_european_price(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar=1.0, call: bool=True) ->Tensor:
    """Returns Black-Scholes price of a European option.

    .. seealso::
        - :func:`pfhedge.nn.BSEuropeanOption.price`

    Args:
        log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
        volatility (torch.Tensor, optional): Volatility of the underlying asset.

    Shape:
        - log_moneyness: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - time_to_maturity: :math:`(N, *)`
        - volatility: :math:`(N, *)`
        - output: :math:`(N, *)`

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import bs_european_price
        ...
        >>> bs_european_price(torch.tensor([-0.1, 0.0, 0.1]), 1.0, 0.2)
        tensor([0.0375, 0.0797, 0.1467])
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike
    price = spot * ncdf(d1(s, t, v)) - strike * ncdf(d2(s, t, v))
    price = price + strike * (1 - s.exp()) if not call else price
    return price


def bs_european_theta(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar) ->Tensor:
    """Returns Black-Scholes theta of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.theta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    numerator = -npdf(d1(s, t, v)) * price * v
    denominator = 2 * t.sqrt()
    output = numerator / denominator
    return torch.where((numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output)


def bs_european_vega(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar) ->Tensor:
    """Returns Black-Scholes vega of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.vega` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    return npdf(d1(s, t, v)) * price * t.sqrt()


def bs_european_binary_delta(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True, strike: TensorOrScalar=1.0) ->Tensor:
    """Returns Black-Scholes delta of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.delta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike
    numerator = npdf(d2(s, t, v))
    denominator = spot * v * t.sqrt()
    delta = numerator / denominator
    delta = torch.where((numerator == 0).logical_and(denominator == 0), torch.zeros_like(delta), delta)
    delta = -delta if not call else delta
    return delta


def bs_european_binary_gamma(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True, strike: TensorOrScalar=1.0) ->Tensor:
    """Returns Black-Scholes gamma of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.gamma` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike
    d2_tensor = d2(s, t, v)
    w = volatility * time_to_maturity.square()
    gamma = -npdf(d2_tensor).div(w * spot.square()) * (1 + d2_tensor.div(w))
    gamma = -gamma if not call else gamma
    return gamma


def bs_european_binary_price(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True) ->Tensor:
    """Returns Black-Scholes price of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.price` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = ncdf(d2(s, t, v))
    price = 1.0 - price if not call else price
    return price


def _bs_theta_gamma_relation(gamma: Tensor, spot: Tensor, volatility: Tensor) ->Tensor:
    return -gamma * volatility.square() * spot.square() / 2


def bs_european_binary_theta(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True, strike: TensorOrScalar=1.0) ->Tensor:
    """Returns Black-Scholes theta of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.theta` for details.
    """
    gamma = bs_european_binary_gamma(log_moneyness=log_moneyness, time_to_maturity=time_to_maturity, volatility=volatility, call=call, strike=strike)
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def _bs_vega_gamma_relation(gamma: Tensor, spot: Tensor, time_to_maturity: Tensor, volatility: Tensor) ->Tensor:
    return gamma * volatility * spot.square() * time_to_maturity


def bs_european_binary_vega(log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, call: bool=True, strike: TensorOrScalar=1.0) ->Tensor:
    """Returns Black-Scholes vega of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.vega` for details.
    """
    gamma = bs_european_binary_gamma(log_moneyness=log_moneyness, time_to_maturity=time_to_maturity, volatility=volatility, call=call, strike=strike)
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility)


def bs_lookback_price(log_moneyness: Tensor, max_log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor, strike: TensorOrScalar) ->Tensor:
    """Returns Black-Scholes price of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.price` for details.
    """
    s, m, t, v = map(torch.as_tensor, (log_moneyness, max_log_moneyness, time_to_maturity, volatility))
    spot = s.exp() * strike
    max = m.exp() * strike
    d1_value = d1(s, t, v)
    d2_value = d2(s, t, v)
    m1 = d1(s - m, t, v)
    m2 = d2(s - m, t, v)
    price_0 = spot * (ncdf(d1_value) + v * t.sqrt() * (d1_value * ncdf(d1_value) + npdf(d1_value))) - strike * ncdf(d2_value)
    price_1 = spot * (ncdf(m1) + v * t.sqrt() * (m1 * ncdf(m1) + npdf(m1))) - strike + max * (1 - ncdf(m2))
    return torch.where(max < strike, price_0, price_1)


class LeakyClamp(Module):
    """Leakily clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    If :math:`\\min \\leq \\max`:

    .. math::
        \\text{output} = \\begin{cases}
            \\min + \\text{clampled_slope} * (\\text{input} - \\min) &
            \\text{input} < \\min \\\\
            \\text{input} & \\min \\leq \\text{input} \\leq \\max \\\\
            \\max + \\text{clampled_slope} * (\\text{input} - \\max) &
            \\max < \\text{input}
        \\end{cases}

    If :math:`\\min > \\max`:

    .. math::

        \\text{output} = \\begin{cases}
            \\frac12 (\\min + \\max)
            & \\text{inverted_output} = \\text{'mean'} \\\\
            \\max
            & \\text{inverted_output} = \\text{'max'} \\\\
        \\end{cases}

    .. seealso::
        - :func:`pfhedge.nn.functional.leaky_clamp`

    Args:
        clamped_slope (float, default=0.01):
            Controls the slope in the clampled regions.
        inverted_output ({'mean', ''max'}, default='mean'):
            Controls the output when :math:`\\min > \\max`.
            'max' is consistent with :func:`torch.clamp`.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - min: :math:`(N, *)` or any size broadcastable to ``input``.
        - max: :math:`(N, *)` or any size broadcastable to ``input``.
        - output: :math:`(N, *)`, same shape as the input.

    Examples:
        >>> import torch
        >>> from pfhedge.nn import LeakyClamp
        >>> m = LeakyClamp()
        >>> input = torch.linspace(-2, 12, 15) * 0.1
        >>> input
        tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
                 0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
        >>> m(input, 0.0, 1.0)
        tensor([-2.0000e-03, -1.0000e-03,  0.0000e+00,  1.0000e-01,  2.0000e-01,
                 3.0000e-01,  4.0000e-01,  5.0000e-01,  6.0000e-01,  7.0000e-01,
                 8.0000e-01,  9.0000e-01,  1.0000e+00,  1.0010e+00,  1.0020e+00])
    """

    def __init__(self, clamped_slope: float=0.01, inverted_output: str='mean'):
        super().__init__()
        self.clamped_slope = clamped_slope
        self.inverted_output = inverted_output

    def extra_repr(self) ->str:
        return 'clamped_slope=' + _format_float(self.clamped_slope)

    def forward(self, input: Tensor, min: Optional[Tensor]=None, max: Optional[Tensor]=None) ->Tensor:
        """Clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

        Args:
            input (torch.Tensor): The input tensor.
            min (torch.Tensor, optional): Lower-bound of the range to be clamped to.
            max (torch.Tensor, optional): Upper-bound of the range to be clamped to.

        Returns:
            torch.Tensor
        """
        return leaky_clamp(input, min=min, max=max, clamped_slope=self.clamped_slope)


class HedgeLoss(Module, ABC):
    """Base class for hedging criteria."""

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        """Returns the loss of the profit-loss distribution.

        This method should be overridden.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.
            target (torch.Tensor or float, default=0): The target portfolio to replicate.
                Typically, target is the payoff of a derivative.

        Shape:
            - input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - target: :math:`(N, *)`
            - output: :math:`(*)`

        Returns:
            torch.Tensor
        """

    def cash(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        """Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        The output ``cash`` is expected to satisfy the following relation:

        .. code::

            loss(torch.full_like(pl, cash)) = loss(pl)

        By default, the output is computed by binary search.
        If analytic form is known, it is recommended to override this method
        for faster computation.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.
            target (torch.Tensor or float, default=0): The target portfolio to replicate.
                Typically, target is the payoff of a derivative.

        Shape:
            - input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - target: :math:`(N, *)`
            - output: :math:`(*)`

        Returns:
            torch.Tensor
        """
        pl = input - target
        return bisect(self, self(pl), pl.min(), pl.max())


def exp_utility(input: Tensor, a: float=1.0) ->Tensor:
    """Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math::

        u(x) = -\\exp(-a x) \\,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -(-a * input).exp()


def entropic_risk_measure(input: Tensor, a: float=1.0) ->Tensor:
    """Returns the entropic risk measure.

    See :class:`pfhedge.nn.EntropicRiskMeasure` for details.
    """
    return (-exp_utility(input, a=a).mean(0)).log() / a


class EntropicRiskMeasure(HedgeLoss):
    """Creates a criterion that measures
    the entropic risk measure.

    The entropic risk measure of the profit-loss distribution
    :math:`\\text{pl}` is given by:

    .. math::
        \\text{loss}(\\text{PL}) = \\frac{1}{a}
        \\log(- \\mathbf{E}[u(\\text{PL})]) \\,,
        \\quad
        u(x) = -\\exp(-a x) \\,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float, default=1.0): Risk aversion coefficient of
            the exponential utility.
            This parameter should be positive.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import EntropicRiskMeasure
        ...
        >>> loss = EntropicRiskMeasure()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.0539)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float=1.0) ->None:
        if not a > 0:
            raise ValueError('Risk aversion coefficient should be positive.')
        super().__init__()
        self.a = a

    def extra_repr(self) ->str:
        return 'a=' + _format_float(self.a) if self.a != 1 else ''

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return entropic_risk_measure(input - target, a=self.a)

    def cash(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return -self(input - target)


def ensemble_mean(function: Callable[..., Tensor], n_times: int=1, *args: Any, **kwargs: Any) ->Tensor:
    """Compute ensemble mean from function.

    Args:
        function (callable[..., torch.Tensor]): Function to evaluate.
        n_times (int, default=1): Number of times to evaluate.
        *args, **kwargs
            Arguments passed to the function.

    Returns:
        torch.Tensor

    Examples:
        >>> function = lambda: torch.tensor([1.0, 2.0])
        >>> ensemble_mean(function, 5)
        tensor([1., 2.])

        >>> _ = torch.manual_seed(42)
        >>> function = lambda: torch.randn(2)
        >>> ensemble_mean(function, 5)
        tensor([ 0.4236, -0.0396])
    """
    if n_times == 1:
        return function(*args, **kwargs)
    else:
        stack = torch.stack([function(*args, **kwargs) for _ in range(n_times)])
        return stack.mean(dim=0)


def has_lazy(module: Module) ->bool:
    return any(map(is_lazy, module.parameters()))


def pl(spot: Tensor, unit: Tensor, cost: Optional[List[float]]=None, payoff: Optional[Tensor]=None, deduct_first_cost: bool=True, deduct_final_cost: bool=False) ->Tensor:
    """Returns the final profit and loss of hedging.

    For
    hedging instruments indexed by :math:`h = 1, \\dots, H` and
    time steps :math:`i = 1, \\dots, T`,
    the final profit and loss is given by

    .. math::
        \\text{PL}(Z, \\delta, S) =
            - Z
            + \\sum_{h = 1}^{H} \\sum_{t = 1}^{T} \\left[
                    \\delta^{(h)}_{t - 1} (S^{(h)}_{t} - S^{(h)}_{t - 1})
                    - c^{(h)} |\\delta^{(h)}_{t} - \\delta^{(h)}_{t - 1}| S^{(h)}_{t}
                \\right] ,

    where
    :math:`Z` is the payoff of the derivative.
    For each hedging instrument,
    :math:`\\{S^{(h)}_t ; t = 1, \\dots, T\\}` is the spot price,
    :math:`\\{\\delta^{(h)}_t ; t = 1, \\dots, T\\}` is the number of shares
    held at each time step.
    We define :math:`\\delta^{(h)}_0 = 0` for notational convenience.

    A hedger sells the derivative to its customer and
    obliges to settle the payoff at maturity.
    The dealer hedges the risk of this liability
    by trading the underlying instrument of the derivative.
    The resulting profit and loss is obtained by adding up the payoff to the
    customer, capital gains from the underlying asset, and the transaction cost.

    References:
        - Buehler, H., Gonon, L., Teichmann, J. and Wood, B., 2019.
          Deep hedging. Quantitative Finance, 19(8), pp.1271-1291.
          [arXiv:`1802.03042 <https://arxiv.org/abs/1802.03042>`_ [q-fin]]

    Args:
        spot (torch.Tensor): The spot price of the underlying asset :math:`S`.
        unit (torch.Tensor): The signed number of shares of the underlying asset
            :math:`\\delta`.
        cost (list[float], default=None): The proportional transaction cost rate of
            the underlying assets.
        payoff (torch.Tensor, optional): The payoff of the derivative :math:`Z`.
        deduct_first_cost (bool, default=True): Whether to deduct the transaction
            cost of the stock at the first time step.
            If ``False``, :math:`- c |\\delta_0| S_1` is omitted the above
            equation of the terminal value.

    Shape:
        - spot: :math:`(N, H, T)` where
          :math:`N` is the number of paths,
          :math:`H` is the number of hedging instruments, and
          :math:`T` is the number of time steps.
        - unit: :math:`(N, H, T)`
        - payoff: :math:`(N)`
        - output: :math:`(N)`.

    Returns:
        torch.Tensor
    """
    assert not deduct_final_cost, 'not supported'
    if spot.size() != unit.size():
        raise RuntimeError(f'unmatched sizes: spot {spot.size()}, unit {unit.size()}')
    if payoff is not None:
        if payoff.dim() != 1 or spot.size(0) != payoff.size(0):
            raise RuntimeError(f'unmatched sizes: spot {spot.size()}, payoff {payoff.size()}')
    output = unit[..., :-1].mul(spot.diff(dim=-1)).sum(dim=(-2, -1))
    if payoff is not None:
        output -= payoff
    if cost is not None:
        c = torch.tensor(cost).unsqueeze(0).unsqueeze(-1)
        output -= (spot[..., 1:] * unit.diff(dim=-1).abs() * c).sum(dim=(-2, -1))
        if deduct_first_cost:
            output -= (spot[..., [0]] * unit[..., [0]].abs() * c).sum(dim=(-2, -1))
    return output


def save_prev_output(module: Module, input: Optional[Tensor], output: Optional[Tensor]) ->None:
    """A hook to save previous output as a buffer named ``prev_output``.

    Examples:

        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> m = torch.nn.Linear(3, 2)
        >>> hook = m.register_forward_hook(save_prev_output)
        >>> input = torch.randn(1, 3)
        >>> m(input)
        tensor([[-1.1647,  0.0244]], ...)
        >>> m.prev_output
        tensor([[-1.1647,  0.0244]], ...)
    """
    module.register_buffer('prev_output', output, persistent=False)


class EntropicLoss(HedgeLoss):
    """Creates a criterion that measures the expected exponential utility.

    The loss of the profit-loss :math:`\\text{PL}` is given by:

    .. math::
        \\text{loss}(\\text{PL}) = -\\mathbf{E}[u(\\text{PL})] \\,,
        \\quad
        u(x) = -\\exp(-a x) \\,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float > 0, default=1.0): Risk aversion coefficient of
            the exponential utility.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import EntropicLoss
        ...
        >>> loss = EntropicLoss()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(7.7982)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float=1.0) ->None:
        if not a > 0:
            raise ValueError('Risk aversion coefficient should be positive.')
        super().__init__()
        self.a = a

    def extra_repr(self) ->str:
        return 'a=' + _format_float(self.a) if self.a != 1 else ''

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return -exp_utility(input - target, a=self.a).mean(0)

    def cash(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return -(-exp_utility(input - target, a=self.a).mean(0)).log() / self.a


def isoelastic_utility(input: Tensor, a: float) ->Tensor:
    """Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

    .. math::

        u(x) = \\begin{cases}
        x^{1 - a} & a \\neq 1 \\\\
        \\log{x} & a = 1
        \\end{cases} \\,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float): Relative risk aversion coefficient of the isoelastic
            utility.

    Returns:
        torch.Tensor
    """
    if a == 1.0:
        return input.log()
    else:
        return input.pow(1.0 - a)


class IsoelasticLoss(HedgeLoss):
    """Creates a criterion that measures the expected isoelastic utility.

    The loss of the profit-loss :math:`\\text{PL}` is given by:

    .. math::
        \\text{loss}(\\text{PL}) = -\\mathbf{E}[u(\\text{PL})] \\,,
        \\quad
        u(x) = \\begin{cases}
        x^{1 - a} & a \\neq 1 \\\\
        \\log{x} & a = 1
        \\end{cases} \\,.

    .. seealso::
        - :func:`pfhedge.nn.functional.isoelastic_utility`:
          The corresponding utility function.

    Args:
        a (float): Relative risk aversion coefficient of the isoelastic utility.
            This parameter should satisfy :math:`0 < a \\leq 1`.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import IsoelasticLoss
        ...
        >>> loss = IsoelasticLoss(0.5)
        >>> input = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-1.5366)
        >>> loss.cash(input)
        tensor(2.3610)

        >>> loss = IsoelasticLoss(1.0)
        >>> pl = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-0.7945)
        >>> loss.cash(input)
        tensor(2.2134)
    """

    def __init__(self, a: float) ->None:
        if not 0 < a <= 1:
            raise ValueError('Relative risk aversion coefficient should satisfy 0 < a <= 1.')
        super().__init__()
        self.a = a

    def extra_repr(self) ->str:
        return 'a=' + _format_float(self.a) if self.a != 1 else ''

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return -isoelastic_utility(input - target, a=self.a).mean(0)


def topp(input: Tensor, p: float, dim: Optional[int]=None, largest: bool=True):
    """Returns the largest :math:`p * N` elements of the given input tensor,
    where :math:`N` stands for the total number of elements in the input tensor.

    If ``dim`` is not given, the last dimension of the ``input`` is chosen.

    If ``largest`` is ``False`` then the smallest elements are returned.

    A namedtuple of ``(values, indices)`` is returned, where the ``indices``
    are the indices of the elements in the original ``input`` tensor.

    .. seealso::
        - :func:`torch.topk`: Returns the ``k`` largest elements of the given input tensor
          along a given dimension.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import topp
        >>>
        >>> input = torch.arange(1.0, 6.0)
        >>> input
        tensor([1., 2., 3., 4., 5.])
        >>> topp(input, 3 / 5)
        torch.return_types.topk(
        values=tensor([5., 4., 3.]),
        indices=tensor([4, 3, 2]))
    """
    if dim is None:
        return input.topk(ceil(p * input.numel()), largest=largest)
    else:
        return input.topk(ceil(p * input.size(dim)), dim=dim, largest=largest)


def expected_shortfall(input: Tensor, p: float, dim: Optional[int]=None) ->Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import expected_shortfall
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> expected_shortfall(input, 0.3)
        tensor(8.)
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


class ExpectedShortfall(HedgeLoss):
    """Creates a criterion that measures the expected shortfall.

    .. seealso::
        - :func:`pfhedge.nn.functional.expected_shortfall`

    Args:
        p (float, default=0.1): Quantile level.
            This parameter should satisfy :math:`0 < p \\leq 1`.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import ExpectedShortfall
        ...
        >>> loss = ExpectedShortfall(0.5)
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.5000)
        >>> loss.cash(input)
        tensor(-2.5000)
    """

    def __init__(self, p: float=0.1):
        if not 0 < p <= 1:
            raise ValueError('The quantile level should satisfy 0 < p <= 1.')
        super().__init__()
        self.p = p

    def extra_repr(self) ->str:
        return str(self.p)

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return expected_shortfall(input - target, p=self.p, dim=0)

    def cash(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return -self(input - target)


class OCE(HedgeLoss):
    """Creates a criterion that measures the optimized certainty equivalent.

    The certainty equivalent is given by:

    .. math::
        \\text{loss}(X, w) = w - \\mathrm{E}[u(X + w)]

    Minimization of loss gives the optimized certainty equivalent.

    .. math::
        \\rho_u(X) = \\inf_w \\text{loss}(X, w)

    Args:
        utility (callable): Utility function.

    Attributes:
        w (torch.nn.Parameter): Represents wealth.

    Examples:
        >>> from pfhedge.nn.modules.loss import OCE
        ...
        >>> _ = torch.manual_seed(42)
        >>> m = OCE(lambda x: 1 - (-x).exp())
        >>> pl = torch.randn(10)
        >>> m(pl)
        tensor(0.0855, grad_fn=<SubBackward0>)
        >>> m.cash(pl)
        tensor(-0.0821)
    """

    def __init__(self, utility: Callable[[Tensor], Tensor]) ->None:
        super().__init__()
        self.utility = utility
        self.w = Parameter(torch.tensor(0.0))

    def extra_repr(self) ->str:
        w = float(self.w.item())
        return self.utility.__name__ + ', w=' + _format_float(w)

    def forward(self, input: Tensor, target: TensorOrScalar=0.0) ->Tensor:
        return self.w - self.utility(input - target + self.w).mean(0)


class Naked(Module):
    """Returns a tensor filled with the scalar value zero.

    Args:
        out_features (int, default=1): Size of each output sample.

    Shape:
        - Input: :math:`(N, *, H_{\\text{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\\text{in}}` is the number of input features.
        - Output: :math:`(N, *, H_{\\text{out}})` where
          all but the last dimension are the same shape as the input and
          :math:`H_{\\text{out}}` is the number of output features.

    Examples:

        >>> from pfhedge.nn import Naked
        >>>
        >>> m = Naked()
        >>> input = torch.zeros((2, 3))
        >>> m(input)
        tensor([[0.],
                [0.]])
    """

    def __init__(self, out_features: int=1):
        super().__init__()
        self.out_features = out_features

    def forward(self, input: Tensor) ->Tensor:
        return input.new_zeros(input.size()[:-1] + (self.out_features,))


def svi_variance(input: TensorOrScalar, a: TensorOrScalar, b: TensorOrScalar, rho: TensorOrScalar, m: TensorOrScalar, sigma: TensorOrScalar) ->Tensor:
    """Returns variance in the SVI model.

    See :class:`pfhedge.nn.SVIVariance` for details.

    Args:
        input (torch.Tensor or float): Log strike of the underlying asset.
            That is, :math:`k = \\log(K / S)` for spot :math:`S` and strike :math:`K`.
        a (torch.Tensor or float): The parameter :math:`a`.
        b (torch.Tensor or float): The parameter :math:`b`.
        rho (torch.Tensor or float): The parameter :math:`\\rho`.
        m (torch.Tensor or float): The parameter :math:`m`.
        sigma (torch.Tensor or float): The parameter :math:`s`.

    Returns:
        torch.Tensor
    """
    k_m = torch.as_tensor(input - m)
    return a + b * (rho * k_m + (k_m.square() + sigma ** 2).sqrt())


class SVIVariance(Module):
    """Returns total variance in the SVI model.

    The total variance for log strike :math:`k = \\log(K / S)`,
    where :math:`K` and :math:`S` are strike and spot, reads:

    .. math::
        w = a + b \\left[ \\rho (k - m) + \\sqrt{(k - m)^2 + \\sigma^2} \\right] .

    References:
        - Jim Gatheral and Antoine Jacquier,
          Arbitrage-free SVI volatility surfaces.
          [arXiv:`1204.0646 <https://arxiv.org/abs/1204.0646>`_ [q-fin.PR]]

    Args:
        a (torch.Tensor or float): The parameter :math:`a`.
        b (torch.Tensor or float): The parameter :math:`b`.
        rho (torch.Tensor or float): The parameter :math:`\\rho`.
        m (torch.Tensor or float): The parameter :math:`m`.
        sigma (torch.Tensor or float): The parameter :math:`\\sigma`.

    Examples:
        >>> import torch
        >>>
        >>> a, b, rho, m, sigma = 0.03, 0.10, 0.10, 0.00, 0.10
        >>> module = SVIVariance(a, b, rho, m, sigma)
        >>> input = torch.tensor([-0.10, -0.01, 0.00, 0.01, 0.10])
        >>> module(input)
        tensor([0.0431, 0.0399, 0.0400, 0.0401, 0.0451])
    """

    def __init__(self, a: TensorOrScalar, b: TensorOrScalar, rho: TensorOrScalar, m: TensorOrScalar, sigma: TensorOrScalar) ->None:
        super().__init__()
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def forward(self, input: Tensor) ->Tensor:
        return svi_variance(input, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma)

    def extra_repr(self) ->str:
        params = f'a={self.a}', f'b={self.b}', f'rho={self.rho}', f'm={self.m}', f'sigma={self.sigma}'
        return ', '.join(params)


def ww_width(gamma: Tensor, spot: Tensor, cost: TensorOrScalar, a: TensorOrScalar=1.0) ->Tensor:
    """Returns half-width of the no-transaction band for
    Whalley-Wilmott's hedging strategy.

    See :class:`pfhedge.nn.WhalleyWilmott` for details.

    Args:
        gamma (torch.Tensor): The gamma of the derivative,
        spot (torch.Tensor): The spot price of the underlier.
        cost (torch.Tensor or float): The cost rate of the underlier.
        a (torch.Tensor or float, default=1.0): Risk aversion parameter in exponential utility.

    Returns:
        torch.Tensor
    """
    return (cost * (3 / 2) * gamma.square() * spot / a).pow(1 / 3)


class FakeModule(Module):

    def __init__(self, output: Tensor):
        super().__init__()
        self.i = 0
        self.register_buffer('output', output)

    def forward(self, input: Tensor):
        output = self.get_buffer('output')[:, [self.i]]
        self.i += 1
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Clamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EntropicLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EntropicRiskMeasure,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ExpectedShortfall,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HedgeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LeakyClamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiLayerPerceptron,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Naked,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OCE,
     lambda: ([], {'utility': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SVIVariance,
     lambda: ([], {'a': 4, 'b': 4, 'rho': 4, 'm': 4, 'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_pfnet_research_pfhedge(_paritybench_base):
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

