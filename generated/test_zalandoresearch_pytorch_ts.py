import sys
_module = sys.modules[__name__]
del sys
pts = _module
dataset = _module
loader = _module
repository = _module
_m5 = _module
datasets = _module
distributions = _module
implicit_quantile = _module
piecewise_linear = _module
utils = _module
zero_inflated = _module
feature = _module
fourier_date_feature = _module
holiday = _module
lags = _module
model = _module
causal_deepar = _module
causal_deepar_estimator = _module
causal_deepar_network = _module
deepar = _module
deepar_estimator = _module
deepar_network = _module
deepvar = _module
deepvar_estimator = _module
deepvar_network = _module
estimator = _module
lstnet = _module
lstnet_estimator = _module
lstnet_network = _module
n_beats = _module
n_beats_ensemble = _module
n_beats_estimator = _module
n_beats_network = _module
simple_feedforward = _module
simple_feedforward_estimator = _module
simple_feedforward_network = _module
tempflow = _module
tempflow_estimator = _module
tempflow_network = _module
tft = _module
tft_estimator = _module
tft_modules = _module
tft_network = _module
tft_output = _module
tft_transform = _module
time_grad = _module
epsilon_theta = _module
time_grad_estimator = _module
time_grad_network = _module
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
gaussian_diffusion = _module
iqn_modules = _module
scaler = _module
trainer = _module
setup = _module
test_piecewise_linear = _module
test_zero_inflated = _module
test_holiday = _module
test_auxillary_outputs = _module
test_lags = _module
test_deepvar = _module
test_forecast = _module
test_lstnet = _module
test_distribution_output = _module
test_feature = _module
test_implicit_quantile_distr_output = _module
test_scaler = _module

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


from typing import Optional


import itertools


from torch.utils.data import IterableDataset


import torch


from torch.distributions import Distribution


from torch.distributions import TransformedDistribution


from torch.distributions import AffineTransform


import torch.nn.functional as F


from torch.distributions import constraints


from torch.distributions import NegativeBinomial


from torch.distributions import Poisson


from torch.distributions.utils import broadcast_all


from torch.distributions.utils import lazy_property


from typing import List


import numpy as np


import torch.nn as nn


from typing import Tuple


from typing import Union


from typing import Callable


from typing import NamedTuple


from functools import partial


from torch.utils import data


from torch.utils.data import DataLoader


from itertools import chain


from typing import Dict


import math


from torch import nn


from torch.nn.modules import loss


import inspect


from abc import ABC


from abc import abstractclassmethod


import warnings


from torch.distributions import Beta


from torch.distributions import StudentT


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.distributions import MixtureSameFamily


from torch.distributions import Independent


from torch.distributions import LowRankMultivariateNormal


from torch.distributions import MultivariateNormal


import copy


from inspect import isfunction


from torch import einsum


from math import pi


from torch import nn as nn


from abc import abstractmethod


import time


from torch.optim import Adam


from torch.optim.lr_scheduler import OneCycleLR


from numpy.testing import assert_allclose as assert_close


from itertools import islice


import pandas as pd


from torch.distributions import Uniform


from torch.nn.utils import clip_grad_norm_


from torch.optim import SGD


from torch.utils.data import TensorDataset


from itertools import combinations


from torch.distributions import Bernoulli


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


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def weighted_average(x: torch.Tensor, weights: Optional[torch.Tensor]=None, dim=None) ->torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return x.mean(dim=dim)


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
        scaled_past_target, scale = self.scaler(past_target[..., -self.context_length:], past_observed_values[..., -self.context_length:])
        c = F.relu(self.cnn(scaled_past_target.unsqueeze(1)))
        c = self.dropout(c)
        c = c.squeeze(2)
        r = c.permute(2, 0, 1)
        _, r = self.rnn(r)
        r = self.dropout(r.squeeze(0))
        skip_c = c[..., -self.conv_skip * self.skip_size:]
        skip_c = skip_c.reshape(-1, self.channels, self.conv_skip, self.skip_size)
        skip_c = skip_c.permute(2, 0, 3, 1)
        skip_c = skip_c.reshape((self.conv_skip, -1, self.channels))
        _, skip_c = self.skip_rnn(skip_c)
        skip_c = skip_c.reshape((-1, self.skip_size * self.skip_rnn_num_cells))
        skip_c = self.dropout(skip_c)
        res = self.fc(torch.cat((r, skip_c), 1)).unsqueeze(-1)
        ar_x = scaled_past_target[..., -self.ar_window:]
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
            future_target = future_target[..., -1:]
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
        seasonal_error = factor * torch.mean(torch.abs(whole_target[:, periodicity:, ...] - whole_target[:, :-periodicity, ...]), dim=1)
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


class FeatureProjector(nn.Module):

    def __init__(self, feature_dims: List[int], embedding_dims: List[int]):
        super().__init__()
        self.__num_features = len(feature_dims)
        if self.__num_features > 1:
            self.feature_slices = feature_dims[0:1] + np.cumsum(feature_dims)[:-1].tolist()
        else:
            self.feature_slices = feature_dims
        self.feature_dims = feature_dims
        self._projector = nn.ModuleList([nn.Linear(in_features=in_feature, out_features=out_features) for in_feature, out_features in zip(self.feature_dims, embedding_dims)])

    def forward(self, features: torch.Tensor) ->List[torch.Tensor]:
        if self.__num_features > 1:
            real_feature_slices = torch.tensor_split(features, self.feature_slices[1:], dim=-1)
        else:
            real_feature_slices = [features]
        return [proj(real_feature_slice) for proj, real_feature_slice in zip(self._projector, real_feature_slices)]


class GatedLinearUnit(nn.Module):

    def __init__(self, dim: int=-1, nonlinear: bool=True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        val, gate = torch.chunk(x, 2, dim=self.dim)
        if self.nonlinear:
            val = torch.tanh(val)
        return torch.sigmoid(gate) * val


class GatedResidualNetwork(nn.Module):

    def __init__(self, d_hidden: int, d_input: Optional[int]=None, d_output: Optional[int]=None, d_static: Optional[int]=None, dropout: float=0.0):
        super().__init__()
        d_input = d_input or d_hidden
        d_static = d_static or 0
        if d_output is None:
            d_output = d_input
            self.add_skip = False
        elif d_output != d_input:
            self.add_skip = True
            self.skip_proj = nn.Linear(in_features=d_input, out_features=d_output)
        else:
            self.add_skip = False
        self.mlp = nn.Sequential(nn.Linear(in_features=d_input + d_static, out_features=d_hidden), nn.ELU(), nn.Linear(in_features=d_hidden, out_features=d_hidden), nn.Dropout(p=dropout), nn.Linear(in_features=d_hidden, out_features=d_output * 2), GatedLinearUnit(nonlinear=False))
        self.lnorm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor]=None) ->torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.mlp(x)
        x = self.lnorm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):

    def __init__(self, d_hidden: int, n_vars: int, dropout: float=0.0, add_static: bool=False):
        super().__init__()
        self.weight_network = GatedResidualNetwork(d_hidden=d_hidden, d_input=d_hidden * n_vars, d_output=n_vars, d_static=d_hidden if add_static else None, dropout=dropout)
        self.variable_network = nn.ModuleList([GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout) for _ in range(n_vars)])

    def forward(self, variables: List[torch.Tensor], static: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)
        var_encodings = [net(var) for var, net in zip(variables, self.variable_network)]
        var_encodings = torch.stack(var_encodings, dim=-1)
        var_encodings = torch.sum(var_encodings * weight, dim=-1)
        return var_encodings, weight


class TemporalFusionEncoder(nn.Module):

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(in_features=d_hidden, out_features=d_hidden * 2), GatedLinearUnit(nonlinear=False))
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(in_features=d_input, out_features=d_hidden)
            self.add_skip = True
        else:
            self.add_skip = False
        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(self, ctx_input: torch.Tensor, tgt_input: torch.Tensor, states: List[torch.Tensor]):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)
        tgt_encodings, _ = self.decoder_lstm(tgt_input, states)
        encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
        skip = torch.cat((ctx_input, tgt_input), dim=1)
        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):

    def __init__(self, context_length: int, prediction_length: int, d_hidden: int, d_var: int, n_head: int, dropout: float=0.0):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.enrich = GatedResidualNetwork(d_hidden=d_hidden, d_static=d_var, dropout=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=n_head, dropout=dropout)
        self.att_net = nn.Sequential(nn.Linear(in_features=d_hidden, out_features=d_hidden * 2), GatedLinearUnit(nonlinear=False))
        self.att_lnorm = nn.LayerNorm(d_hidden)
        self.ff_net = nn.Sequential(GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout), nn.Linear(in_features=d_hidden, out_features=d_hidden * 2), GatedLinearUnit(nonlinear=False))
        self.ff_lnorm = nn.LayerNorm(d_hidden)
        self.register_buffer('attn_mask', self._generate_subsequent_mask(prediction_length, prediction_length + context_length))

    @staticmethod
    def _generate_subsequent_mask(target_length: int, source_length: int) ->torch.Tensor:
        mask = (torch.triu(torch.ones(source_length, target_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        static = static.repeat((1, self.context_length + self.prediction_length, 1))
        skip = x[:, self.context_length:, ...]
        x = self.enrich(x, static)
        mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        mask_pad = mask_pad.repeat((1, self.prediction_length))
        key_padding_mask = torch.cat((mask, mask_pad), dim=1).bool()
        query_key_value = x.permute(1, 0, 2)
        attn_output, _ = self.attention(query=query_key_value[-self.prediction_length:, ...], key=query_key_value, value=query_key_value, attn_mask=self.attn_mask)
        att = self.att_net(attn_output.permute(1, 0, 2))
        x = x[:, self.context_length:, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)
        return x


class DiffusionEmbedding(nn.Module):

    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(dim, max_steps), persistent=False)
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(dim).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation, padding_mode='circular')
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1, padding=2, padding_mode='circular')
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):

    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):

    def __init__(self, target_dim, cond_length, time_emb_dim=16, residual_layers=8, residual_channels=8, dilation_cycle_length=2, residual_hidden=64):
        super().__init__()
        self.input_projection = nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode='circular')
        self.diffusion_embedding = DiffusionEmbedding(time_emb_dim, proj_dim=residual_hidden)
        self.cond_upsampler = CondUpsampler(target_dim=target_dim, cond_length=cond_length)
        self.residual_layers = nn.ModuleList([ResidualBlock(residual_channels=residual_channels, dilation=2 ** (i % dilation_cycle_length), hidden_size=residual_hidden) for i in range(residual_layers)])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)
        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos((x / timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0, 0.999)


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda : torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda : torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):

    def __init__(self, denoise_fn, input_size, beta_end=0.1, diff_steps=100, loss_type='l2', betas=None, beta_schedule='linear'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale = None
        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        elif beta_schedule == 'linear':
            betas = np.linspace(0.0001, beta_end, diff_steps)
        elif beta_schedule == 'quad':
            betas = np.linspace(0.0001 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
        elif beta_schedule == 'const':
            betas = beta_end * np.ones(diff_steps)
        elif beta_schedule == 'jsd':
            betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
        elif beta_schedule == 'sigmoid':
            betas = np.linspace(-6, 6, diff_steps)
            betas = (beta_end - 0.0001) / (np.exp(-betas) + 1) + 0.0001
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(diff_steps)
        else:
            raise NotImplementedError(beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, cond=cond))
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, cond, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
        else:
            shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)
        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)
        if self.loss_type == 'l1':
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()
        return loss

    def log_prob(self, x, cond, *args, **kwargs):
        if self.scale is not None:
            x /= self.scale
        B, T, _ = x.shape
        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs)
        return loss


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


class QuantileLayer(nn.Module):
    """Define quantile embedding layer, i.e. phi in the IQN paper (arXiv: 1806.06923)."""

    def __init__(self, num_output):
        super(QuantileLayer, self).__init__()
        self.n_cos_embedding = 64
        self.num_output = num_output
        self.output_layer = nn.Sequential(nn.Linear(self.n_cos_embedding, self.n_cos_embedding), nn.PReLU(), nn.Linear(self.n_cos_embedding, num_output))

    def forward(self, tau):
        cos_embedded_tau = self.cos_embed(tau)
        final_output = self.output_layer(cos_embedded_tau)
        return final_output

    def cos_embed(self, tau):
        integers = torch.repeat_interleave(torch.arange(0, self.n_cos_embedding).unsqueeze(dim=0), repeats=tau.shape[-1], dim=0)
        return torch.cos(pi * tau.unsqueeze(dim=-1) * integers)


class ImplicitQuantileModule(nn.Module):
    """See arXiv: 1806.06923
    This module, in combination with quantile loss,
    learns how to generate the quantile of the distribution of the target.
    A quantile value, tau, is randomly generated with a Uniform([0, 1])).
    This quantile value is embedded in this module and also passed to the quantile loss:
    this should force the model to learn the appropriate quantile.
    """

    def __init__(self, in_features, output_domain_cls):
        super(ImplicitQuantileModule, self).__init__()
        self.in_features = in_features
        self.quantile_layer = QuantileLayer(in_features)
        self.output_layer = nn.Sequential(nn.Linear(in_features, in_features), nn.Softplus(), nn.Linear(in_features, 1), output_domain_cls())

    def forward(self, input_data, tau):
        embedded_tau = self.quantile_layer(tau)
        new_input_data = input_data * (torch.ones_like(embedded_tau) + embedded_tau)
        return self.output_layer(new_input_data).squeeze(-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CondUpsampler,
     lambda: ([], {'cond_length': 4, 'target_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiffusionEmbedding,
     lambda: ([], {'dim': 4, 'proj_dim': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     True),
    (FeatureAssembler,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FlowSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedLinearUnit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedResidualNetwork,
     lambda: ([], {'d_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ImplicitQuantileModule,
     lambda: ([], {'in_features': 4, 'output_domain_cls': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MADE,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantileLayer,
     lambda: ([], {'num_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

