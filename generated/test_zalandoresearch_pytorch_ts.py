import sys
_module = sys.modules[__name__]
del sys
pts = _module
core = _module
_base = _module
component = _module
serde = _module
dataset = _module
artificial = _module
common = _module
file_dataset = _module
list_dataset = _module
loader = _module
multivariate_grouper = _module
process = _module
recipe = _module
repository = _module
_artificial = _module
_gp_copula_2019 = _module
_lstnet = _module
_m4 = _module
_util = _module
datasets = _module
stat = _module
transformed_iterable_dataset = _module
utils = _module
evaluation = _module
backtest = _module
evaluator = _module
exception = _module
feature = _module
holiday = _module
lag = _module
time_feature = _module
model = _module
deepar = _module
deepar_estimator = _module
deepar_network = _module
deepvar = _module
deepvar_estimator = _module
deepvar_network = _module
estimator = _module
forecast = _module
forecast_generator = _module
lstnet = _module
lstnet_estimator = _module
lstnet_network = _module
n_beats = _module
n_beats_ensemble = _module
n_beats_estimator = _module
n_beats_network = _module
predictor = _module
quantile = _module
simple_feedforward = _module
simple_feedforward_estimator = _module
simple_feedforward_network = _module
tempflow = _module
tempflow_estimator = _module
tempflow_network = _module
transformer = _module
transformer_estimator = _module
transformer_network = _module
transformer_tempflow = _module
transformer_tempflow_estimator = _module
transformer_tempflow_network = _module
utils = _module
modules = _module
distribution_output = _module
feature = _module
flows = _module
lambda_layer = _module
scaler = _module
trainer = _module
transform = _module
convert = _module
field = _module
sampler = _module
split = _module
setup = _module
test_common = _module
test_multivariate_grouper = _module
test_process = _module
test_stat = _module
test_evaluator = _module
test_holiday = _module
test_lag = _module
test_auxillary_outputs = _module
test_lags = _module
test_deepvar = _module
test_forecast = _module
test_lstnet = _module
test_distribution_output = _module
test_feature = _module
test_scaler = _module
test_transform = _module

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


import functools


import inspect


from collections import OrderedDict


from typing import Any


import torch


import itertools


from collections import defaultdict


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


import numpy as np


import torch.nn as nn


from typing import Tuple


from typing import Union


from torch.distributions import Distribution


from abc import ABC


from abc import abstractmethod


from typing import NamedTuple


from torch.utils.data import DataLoader


from enum import Enum


from typing import Set


from typing import Callable


import pandas as pd


import torch.nn.functional as F


from abc import abstractclassmethod


from torch.distributions import Beta


from torch.distributions import NegativeBinomial


from torch.distributions import StudentT


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.distributions import MixtureSameFamily


from torch.distributions import Independent


from torch.distributions import LowRankMultivariateNormal


from torch.distributions import MultivariateNormal


from torch.distributions import TransformedDistribution


from torch.distributions import AffineTransform


import copy


import math


import time


from torch.utils.tensorboard import SummaryWriter


from scipy.special import erf


from scipy.special import erfinv


from itertools import islice


from torch.distributions import Uniform


from torch.nn.utils import clip_grad_norm_


from torch.optim import SGD


from torch.utils.data import TensorDataset


from itertools import chain


from itertools import combinations


class ArgProj(nn.Module):

    def __init__(self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], dtype: np.dtype=np.float32, prefix: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.dtype = dtype
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Module):

    def __init__(self, function):
        super().__init__()
        self._func = function

    def forward(self, x, *args):
        return self._func(x, *args)


class Output(ABC):
    in_features: int
    args_dim: Dict[str, int]
    _dtype: np.dtype = np.float32

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype):
        self._dtype = dtype

    def get_args_proj(self, in_features: int, prefix: Optional[str]=None) ->ArgProj:
        return ArgProj(in_features=in_features, args_dim=self.args_dim, domain_map=LambdaLayer(self.domain_map), prefix=prefix, dtype=self.dtype)

    @abstractclassmethod
    def domain_map(cls, *args: torch.Tensor):
        pass


def fqname_for(cls: type) ->str:
    """
    Returns the fully qualified name of ``cls``.

    Parameters
    ----------
    cls
        The class we are interested in.

    Returns
    -------
    str
        The fully qualified name of ``cls``.
    """
    return f'{cls.__module__}.{cls.__qualname__}'


kind_inst = 'instance'


kind_type = 'type'


def dump_code(o: Any) ->str:
    """
    Serializes an object to a Python code string.

    Parameters
    ----------
    o
        The object to serialize.

    Returns
    -------
    str
        A string representing the object as Python code.

    See Also
    --------
    load_code
        Inverse function.
    """

    def _dump_code(x: Any) ->str:
        if type(x) == dict and x.get('__kind__') == kind_inst:
            args = x.get('args', [])
            kwargs = x.get('kwargs', {})
            fqname = x['class']
            bindings = ', '.join(itertools.chain(map(_dump_code, args), [f'{k}={_dump_code(v)}' for k, v in kwargs.items()]))
            return f'{fqname}({bindings})'
        if type(x) == dict and x.get('__kind__') == kind_type:
            return x['class']
        if isinstance(x, dict):
            inner = ', '.join(f'{_dump_code(k)}: {_dump_code(v)}' for k, v in x.items())
            return f'{{{inner}}}'
        if isinstance(x, list):
            inner = ', '.join(list(map(dump_code, x)))
            return f'[{inner}]'
        if isinstance(x, tuple):
            inner = ', '.join(list(map(dump_code, x)))
            if len(x) == 1:
                inner += ','
            return f'({inner})'
        if isinstance(x, str):
            return json.dumps(x)
        if isinstance(x, float) or np.issubdtype(type(x), np.inexact):
            if math.isfinite(x):
                return str(x)
            else:
                return 'float("{x}")'
        if isinstance(x, int) or np.issubdtype(type(x), np.integer):
            return str(x)
        if x is None:
            return str(x)
        raise RuntimeError(f'Unexpected element type {fqname_for(x.__class__)}')
    return _dump_code(encode(o))


def validated(base_model=None):
    """
    Decorates an ``__init__`` method with typed parameters with validation
    and auto-conversion logic.

    >>> class ComplexNumber:
    ...     @validated()
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y

    Classes with decorated initializers can be instantiated using arguments of
    another type (e.g. an ``y`` argument of type ``str`` ). The decorator
    handles the type conversion logic.

    >>> c = ComplexNumber(y='42')
    >>> (c.x, c.y)
    (0.0, 42.0)

    If the bound argument cannot be converted, the decorator throws an error.

    >>> c = ComplexNumber(y=None)
    Traceback (most recent call last):
        ...
    pydantic.error_wrappers.ValidationError: 1 validation error for ComplexNumberModel
    y
      none is not an allowed value (type=type_error.none.not_allowed)

    Internally, the decorator delegates all validation and conversion logic to
    `a Pydantic model <https://pydantic-docs.helpmanual.io/>`_, which can be
    accessed through the ``Model`` attribute of the decorated initiazlier.

    >>> ComplexNumber.__init__.Model
    <class 'ComplexNumberModel'>

    The Pydantic model is synthesized automatically from on the parameter
    names and types of the decorated initializer. In the ``ComplexNumber``
    example, the synthesized Pydantic model corresponds to the following
    definition.

    >>> class ComplexNumberModel(BaseValidatedInitializerModel):
    ...     x: float = 0.0
    ...     y: float = 0.0


    Clients can optionally customize the base class of the synthesized
    Pydantic model using the ``base_model`` decorator parameter. The default
    behavior uses :class:`BaseValidatedInitializerModel` and its
    `model config <https://pydantic-docs.helpmanual.io/#config>`_.

    See Also
    --------
    BaseValidatedInitializerModel
        Default base class for all synthesized Pydantic models.
    """

    def validator(init):
        init_qualname = dict(inspect.getmembers(init))['__qualname__']
        init_clsnme = init_qualname.split('.')[0]
        init_params = inspect.signature(init).parameters
        init_fields = {param.name: (param.annotation if param.annotation != inspect.Parameter.empty else Any, param.default if param.default != inspect.Parameter.empty else ...) for param in init_params.values() if param.name != 'self' and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD}
        if base_model is None:
            PydanticModel = create_model(f'{init_clsnme}Model', __config__=BaseValidatedInitializerModel.Config, **init_fields)
        else:
            PydanticModel = create_model(f'{init_clsnme}Model', __base__=base_model, **init_fields)

        def validated_repr(self) ->str:
            return dump_code(self)

        def validated_getnewargs_ex(self):
            return (), self.__init_args__

        @functools.wraps(init)
        def init_wrapper(*args, **kwargs):
            self, *args = args
            nmargs = {name: arg for (name, param), arg in zip(list(init_params.items()), [self] + args) if name != 'self'}
            model = PydanticModel(**{**nmargs, **kwargs})
            all_args = {**nmargs, **kwargs, **model.__dict__}
            if not getattr(self, '__init_args__', {}):
                self.__init_args__ = OrderedDict({name: arg for name, arg in sorted(all_args.items()) if type(arg) != torch.nn.ParameterDict})
                self.__class__.__getnewargs_ex__ = validated_getnewargs_ex
                self.__class__.__repr__ = validated_repr
            return init(self, **all_args)
        setattr(init_wrapper, 'Model', PydanticModel)
        return init_wrapper
    return validator


class FeatureEmbedder(nn.Module):

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) ->None:
        super().__init__()
        self.__num_features = len(cardinalities)

        def create_embedding(c: int, d: int) ->nn.Embedding:
            embedding = nn.Embedding(c, d)
            return embedding
        self.__embedders = nn.ModuleList([create_embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) ->torch.Tensor:
        if self.__num_features > 1:
            cat_feature_slices = torch.chunk(features, self.__num_features, dim=-1)
        else:
            cat_feature_slices = [features]
        return torch.cat([embed(cat_feature_slice.squeeze(-1)) for embed, cat_feature_slice in zip(self.__embedders, cat_feature_slices)], dim=-1)


class Scaler(ABC, nn.Module):

    def __init__(self, keepdim: bool=False, time_first: bool=True):
        super().__init__()
        self.keepdim = keepdim
        self.time_first = time_first

    @abstractmethod
    def compute_scale(self, data: torch.Tensor, observed_indicator: torch.Tensor) ->torch.Tensor:
        pass

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data
            tensor of shape (N, T, C) if ``time_first == True`` or (N, C, T)
            if ``time_first == False`` containing the data to be scaled

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            Tensor containing the "scaled" data, shape: (N, T, C) or (N, C, T).
        Tensor
            Tensor containing the scale, of shape (N, C) if ``keepdim == False``, 
            and shape (N, 1, C) or (N, C, 1) if ``keepdim == True``.
        """
        scale = self.compute_scale(data, observed_indicator)
        if self.time_first:
            dim = 1
        else:
            dim = 2
        if self.keepdim:
            scale = scale.unsqueeze(dim=dim)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(dim=dim), scale


class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    def __init__(self, minimum_scale: float=1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('minimum_scale', torch.tensor(minimum_scale))

    def compute_scale(self, data: torch.Tensor, observed_indicator: torch.Tensor) ->torch.Tensor:
        if self.time_first:
            dim = 1
        else:
            dim = 2
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator
        scale = torch.where(sum_observed > torch.zeros_like(sum_observed), scale, default_scale * torch.ones_like(num_observed))
        return torch.max(scale, self.minimum_scale).detach()


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(self, data: torch.Tensor, observed_indicator: torch.Tensor) ->torch.Tensor:
        if self.time_first:
            dim = 1
        else:
            dim = 2
        return torch.ones_like(data).mean(dim=dim)


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def weighted_average(tensor: torch.Tensor, weights: Optional[torch.Tensor]=None, dim=None):
    if weights is not None:
        weighted_tensor = tensor * weights
        if dim is not None:
            sum_weights = torch.sum(weights, dim)
            sum_weighted_tensor = torch.sum(weighted_tensor, dim)
        else:
            sum_weights = weights.sum()
            sum_weighted_tensor = weighted_tensor.sum()
        sum_weights = torch.max(torch.ones_like(sum_weights), sum_weights)
        return sum_weighted_tensor / sum_weights
    elif dim is not None:
        return torch.mean(tensor, dim=dim)
    else:
        return tensor.mean()


class LSTNetBase(nn.Module):

    def __init__(self, num_series: int, channels: int, kernel_size: int, rnn_cell_type: str, rnn_num_cells: int, skip_rnn_cell_type: str, skip_rnn_num_cells: int, skip_size: int, ar_window: int, context_length: int, horizon: Optional[int], prediction_length: Optional[int], dropout_rate: float, output_activation: Optional[str], scaling: bool, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.num_series = num_series
        self.channels = channels
        assert channels % skip_size == 0, 'number of conv1d `channels` must be divisible by the `skip_size`'
        self.skip_size = skip_size
        assert ar_window > 0, 'auto-regressive window must be a positive integer'
        self.ar_window = ar_window
        assert not (horizon is None) == (prediction_length is None), 'Exactly one of `horizon` and `prediction_length` must be set at a time'
        assert horizon is None or horizon > 0, '`horizon` must be greater than zero'
        assert prediction_length is None or prediction_length > 0, '`prediction_length` must be greater than zero'
        self.prediction_length = prediction_length
        self.horizon = horizon
        assert context_length > 0, '`context_length` must be greater than zero'
        self.context_length = context_length
        if output_activation is not None:
            assert output_activation in ['sigmoid', 'tanh'], "`output_activation` must be either 'sigmiod' or 'tanh' "
        self.output_activation = output_activation
        assert rnn_cell_type in ['GRU', 'LSTM'], "`rnn_cell_type` must be either 'GRU' or 'LSTM' "
        assert skip_rnn_cell_type in ['GRU', 'LSTM'], "`skip_rnn_cell_type` must be either 'GRU' or 'LSTM' "
        conv_out = context_length - kernel_size
        self.conv_skip = conv_out // skip_size
        assert self.conv_skip > 0, 'conv1d output size must be greater than or equal to `skip_size`\nChoose a smaller `kernel_size` or bigger `context_length`'
        self.cnn = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(num_series, kernel_size))
        self.dropout = nn.Dropout(p=dropout_rate)
        rnn = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_cell_type]
        self.rnn = rnn(input_size=channels, hidden_size=rnn_num_cells)
        skip_rnn = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[skip_rnn_cell_type]
        self.skip_rnn_num_cells = skip_rnn_num_cells
        self.skip_rnn = skip_rnn(input_size=channels, hidden_size=skip_rnn_num_cells)
        self.fc = nn.Linear(rnn_num_cells + skip_size * skip_rnn_num_cells, num_series)
        if self.horizon:
            self.ar_fc = nn.Linear(ar_window, 1)
        else:
            self.ar_fc = nn.Linear(ar_window, prediction_length)
        if scaling:
            self.scaler = MeanScaler(keepdim=True, time_first=False)
        else:
            self.scaler = NOPScaler(keepdim=True, time_first=False)

    def forward(self, past_target: torch.Tensor, past_observed_values: torch.Tensor) ->torch.Tensor:
        scaled_past_target, scale = self.scaler(past_target[(...), -self.context_length:], past_observed_values[(...), -self.context_length:])
        c = F.relu(self.cnn(scaled_past_target.unsqueeze(1)))
        c = self.dropout(c)
        c = c.squeeze(2)
        r = c.permute(2, 0, 1)
        _, r = self.rnn(r)
        r = self.dropout(r.squeeze(0))
        skip_c = c[(...), -self.conv_skip * self.skip_size:]
        skip_c = skip_c.reshape(-1, self.channels, self.conv_skip, self.skip_size)
        skip_c = skip_c.permute(2, 0, 3, 1)
        skip_c = skip_c.reshape((self.conv_skip, -1, self.channels))
        _, skip_c = self.skip_rnn(skip_c)
        skip_c = skip_c.reshape((-1, self.skip_size * self.skip_rnn_num_cells))
        skip_c = self.dropout(skip_c)
        res = self.fc(torch.cat((r, skip_c), 1)).unsqueeze(-1)
        ar_x = scaled_past_target[(...), -self.ar_window:]
        ar_x = ar_x.reshape(-1, self.ar_window)
        ar_x = self.ar_fc(ar_x)
        if self.horizon:
            ar_x = ar_x.reshape(-1, self.num_series, 1)
        else:
            ar_x = ar_x.reshape(-1, self.num_series, self.prediction_length)
        out = res + ar_x
        if self.output_activation is None:
            return out, scale
        return torch.sigmoid(out) if self.output_activation == 'sigmoid' else torch.tanh(out), scale


class LSTNetTrain(LSTNetBase):

    def __init__(self, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.L1Loss()

    def forward(self, past_target: torch.Tensor, past_observed_values: torch.Tensor, future_target: torch.Tensor) ->torch.Tensor:
        ret, scale = super().forward(past_target, past_observed_values)
        if self.horizon:
            future_target = future_target[(...), -1:]
        loss = self.loss_fn(ret * scale, future_target)
        return loss


class LSTNetPredict(LSTNetBase):

    def forward(self, past_target: torch.Tensor, past_observed_values: torch.Tensor) ->torch.Tensor:
        ret, scale = super().forward(past_target, past_observed_values)
        ret = (ret * scale).permute(0, 2, 1)
        return ret.unsqueeze(1)


class NBEATSBlock(nn.Module):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, share_thetas=False):
        super(NBEATSBlock, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        fc_stack = [nn.Linear(backcast_length, units), nn.ReLU()]
        for _ in range(num_block_layers - 1):
            fc_stack.append(nn.Linear(units, units))
            fc_stack.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_stack)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


def linspace(backcast_length: int, forecast_length: int) ->Tuple[np.ndarray, np.ndarray]:
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSSeasonalBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim=None, num_block_layers=4, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        super(NBEATSSeasonalBlock, self).__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length, share_thetas=True)
        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length)
        p1, p2 = (thetas_dim // 2, thetas_dim // 2) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1)
        s1_b = torch.tensor([np.cos(2 * np.pi * i * backcast_linspace) for i in range(p1)]).float()
        s2_b = torch.tensor([np.sin(2 * np.pi * i * backcast_linspace) for i in range(p2)]).float()
        self.register_buffer('S_backcast', torch.cat([s1_b, s2_b]))
        s1_f = torch.tensor([np.cos(2 * np.pi * i * forecast_linspace) for i in range(p1)]).float()
        s2_f = torch.tensor([np.sin(2 * np.pi * i * forecast_linspace) for i in range(p2)]).float()
        self.register_buffer('S_forecast', torch.cat([s1_f, s2_f]))

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.S_backcast)
        forecast = self.theta_f_fc(x).mm(self.S_forecast)
        return backcast, forecast


class NBEATSTrendBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(NBEATSTrendBlock, self).__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length, share_thetas=True)
        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length)
        self.register_buffer('T_backcast', torch.tensor([(backcast_linspace ** i) for i in range(thetas_dim)]).float())
        self.register_buffer('T_forecast', torch.tensor([(forecast_linspace ** i) for i in range(thetas_dim)]).float())

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5):
        super(NBEATSGenericBlock, self).__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length)
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class NBEATSNetwork(nn.Module):

    def __init__(self, prediction_length: int, context_length: int, num_stacks: int, widths: List[int], num_blocks: List[int], num_block_layers: List[int], expansion_coefficient_lengths: List[int], sharing: List[bool], stack_types: List[str], **kwargs) ->None:
        super(NBEATSNetwork, self).__init__()
        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types = stack_types
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.net_blocks = nn.ModuleList()
        for stack_id in range(num_stacks):
            for block_id in range(num_blocks[stack_id]):
                if self.stack_types[stack_id] == 'G':
                    net_block = NBEATSGenericBlock(units=self.widths[stack_id], thetas_dim=self.expansion_coefficient_lengths[stack_id], num_block_layers=self.num_block_layers[stack_id], backcast_length=context_length, forecast_length=prediction_length)
                elif self.stack_types[stack_id] == 'S':
                    net_block = NBEATSSeasonalBlock(units=self.widths[stack_id], num_block_layers=self.num_block_layers[stack_id], backcast_length=context_length, forecast_length=prediction_length)
                else:
                    net_block = NBEATSTrendBlock(units=self.widths[stack_id], thetas_dim=self.expansion_coefficient_lengths[stack_id], num_block_layers=self.num_block_layers[stack_id], backcast_length=context_length, forecast_length=prediction_length)
                self.net_blocks.append(net_block)

    def forward(self, past_target: torch.Tensor):
        if len(self.net_blocks) == 1:
            _, forecast = self.net_blocks[0](past_target)
            return forecast
        else:
            backcast, forecast = self.net_blocks[0](past_target)
            backcast = past_target - backcast
            for i in range(1, len(self.net_blocks) - 1):
                b, f = self.net_blocks[i](backcast)
                backcast = backcast - b
                forecast = forecast + f
            _, last_forecast = self.net_blocks[-1](backcast)
            return forecast + last_forecast

    def smape_loss(self, forecast: torch.Tensor, future_target: torch.Tensor) ->torch.Tensor:
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = denominator == 0
        return 200 / self.prediction_length * torch.mean(torch.abs(future_target - forecast) * torch.logical_not(flag) / (denominator + flag), dim=1)

    def mape_loss(self, forecast: torch.Tensor, future_target: torch.Tensor) ->torch.Tensor:
        denominator = torch.abs(future_target)
        flag = denominator == 0
        return 100 / self.prediction_length * torch.mean(torch.abs(future_target - forecast) * torch.logical_not(flag) / (denominator + flag), dim=1)

    def mase_loss(self, forecast: torch.Tensor, future_target: torch.Tensor, past_target: torch.Tensor, periodicity: int) ->torch.Tensor:
        factor = 1 / (self.context_length + self.prediction_length - periodicity)
        whole_target = torch.cat((past_target, future_target), dim=1)
        seasonal_error = factor * torch.mean(torch.abs(whole_target[:, periodicity:, (...)] - whole_target[:, :-periodicity, (...)]), dim=1)
        flag = seasonal_error == 0
        return torch.mean(torch.abs(future_target - forecast), dim=1) * torch.logical_not(flag) / (seasonal_error + flag)


class NBEATSTrainingNetwork(NBEATSNetwork):

    def __init__(self, loss_function: str, freq: str, *args, **kwargs) ->None:
        super(NBEATSTrainingNetwork, self).__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.freq = freq
        self.periodicity = get_seasonality(self.freq)
        if self.loss_function == 'MASE':
            assert self.periodicity < self.context_length + self.prediction_length, "If the 'periodicity' of your data is less than 'context_length' + 'prediction_length' the seasonal_error cannot be calculated and thus 'MASE' cannot be used for optimization."

    def forward(self, past_target: torch.Tensor, future_target: torch.Tensor) ->torch.Tensor:
        forecast = super().forward(past_target=past_target)
        if self.loss_function == 'sMAPE':
            loss = self.smape_loss(forecast, future_target)
        elif self.loss_function == 'MAPE':
            loss = self.mape_loss(forecast, future_target)
        elif self.loss_function == 'MASE':
            loss = self.mase_loss(forecast, future_target, past_target, self.periodicity)
        else:
            raise ValueError(f'Invalid value {self.loss_function} for argument loss_function.')
        return loss.mean()


class NBEATSPredictionNetwork(NBEATSNetwork):

    def __init__(self, *args, **kwargs) ->None:
        super(NBEATSPredictionNetwork, self).__init__(*args, **kwargs)

    def forward(self, past_target: torch.Tensor, future_target: torch.Tensor=None) ->torch.Tensor:
        forecasts = super().forward(past_target=past_target)
        return forecasts.unsqueeze(1)


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-05):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            self.batch_var = x.view(-1, x.shape[-1]).var(0)
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det_jacobian.expand_as(x)


class Flow(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.__scale = None
        self.net = None
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def forward(self, x, cond):
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        return u, log_abs_det_jacobian

    def inverse(self, u, cond):
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))
        return x, log_abs_det_jacobian

    def log_prob(self, x, cond):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1)

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape
        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer('mask', mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    degrees = []
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]
    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
    return masks, degrees[0]


class MADE(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='ReLU', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of MADEs
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)
        if activation == 'ReLU':
            activation_fn = nn.ReLU()
        elif activation == 'Tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        x = torch.zeros_like(u)
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[..., i] = u[..., i] * torch.exp(loga[..., i]) + m[..., i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class MAF(Flow):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='ReLU', input_order='sequential', batch_norm=True):
        super().__init__(input_size)
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()
        self.register_buffer('mask', mask)
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)
        self.t_net = copy.deepcopy(self.s_net)
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        mx = x * self.mask
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (1 - self.mask)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        log_abs_det_jacobian = log_s
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        mu = u * self.mask
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (1 - self.mask)
        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        log_abs_det_jacobian = -log_s
        return x, log_abs_det_jacobian


class RealNVP(Flow):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__(input_size)
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)


class FeatureAssembler(nn.Module):

    def __init__(self, T: int, embed_static: Optional[FeatureEmbedder]=None, embed_dynamic: Optional[FeatureEmbedder]=None) ->None:
        super().__init__()
        self.T = T
        self.embeddings = nn.ModuleDict({'embed_static': embed_static, 'embed_dynamic': embed_dynamic})

    def forward(self, feat_static_cat: torch.Tensor, feat_static_real: torch.Tensor, feat_dynamic_cat: torch.Tensor, feat_dynamic_real: torch.Tensor) ->torch.Tensor:
        processed_features = [self.process_static_cat(feat_static_cat), self.process_static_real(feat_static_real), self.process_dynamic_cat(feat_dynamic_cat), self.process_dynamic_real(feat_dynamic_real)]
        return torch.cat(processed_features, dim=-1)

    def process_static_cat(self, feature: torch.Tensor) ->torch.Tensor:
        if self.embeddings['embed_static'] is not None:
            feature = self.embeddings['embed_static'](feature)
        return feature.unsqueeze(1).expand(-1, self.T, -1).float()

    def process_dynamic_cat(self, feature: torch.Tensor) ->torch.Tensor:
        if self.embeddings['embed_dynamic'] is None:
            return feature.float()
        else:
            return self.embeddings['embed_dynamic'](feature)

    def process_static_real(self, feature: torch.Tensor) ->torch.Tensor:
        return feature.unsqueeze(1).expand(-1, self.T, -1)

    def process_dynamic_real(self, feature: torch.Tensor) ->torch.Tensor:
        return feature


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeatureAssembler,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FeatureEmbedder,
     lambda: ([], {'cardinalities': [4, 4], 'embedding_dims': [4, 4]}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (FlowSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'function': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MADE,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NOPScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zalandoresearch_pytorch_ts(_paritybench_base):
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

