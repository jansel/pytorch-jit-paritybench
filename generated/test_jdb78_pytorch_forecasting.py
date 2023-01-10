import sys
_module = sys.modules[__name__]
del sys
conf = _module
ar = _module
nbeats = _module
stallion = _module
pytorch_forecasting = _module
data = _module
encoders = _module
examples = _module
samplers = _module
timeseries = _module
metrics = _module
_mqf2_utils = _module
base_metrics = _module
distributions = _module
point = _module
quantile = _module
models = _module
base_model = _module
baseline = _module
deepar = _module
mlp = _module
submodules = _module
nbeats = _module
sub_modules = _module
nhits = _module
sub_modules = _module
nn = _module
embeddings = _module
rnn = _module
rnn = _module
temporal_fusion_transformer = _module
sub_modules = _module
tuning = _module
optim = _module
utils = _module
conftest = _module
test_encoders = _module
test_samplers = _module
test_timeseries = _module
test_metrics = _module
conftest = _module
test_baseline = _module
test_deepar = _module
test_mlp = _module
test_nbeats = _module
test_nhits = _module
test_embeddings = _module
test_rnn = _module
test_rnn_model = _module
test_temporal_fusion_transformer = _module
test_autocorrelation = _module

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


import warnings


import numpy as np


import pandas as pd


from pandas.core.common import SettingWithCopyWarning


import torch


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Tuple


from typing import Union


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


from torch.distributions import constraints


from torch.distributions.transforms import ExpTransform


from torch.distributions.transforms import PowerTransform


from torch.distributions.transforms import SigmoidTransform


from torch.distributions.transforms import Transform


from torch.distributions.transforms import _clipped_sigmoid


from torch.distributions.transforms import identity_transform


import torch.nn.functional as F


from torch.nn.utils import rnn


from sklearn.utils import shuffle


from torch.utils.data.sampler import Sampler


from copy import copy as _copy


from copy import deepcopy


from functools import lru_cache


import inspect


import matplotlib.pyplot as plt


from sklearn.exceptions import NotFittedError


from sklearn.preprocessing import RobustScaler


from sklearn.preprocessing import StandardScaler


from sklearn.utils.validation import check_is_fitted


from torch.distributions import Beta


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from typing import Optional


from torch.distributions import AffineTransform


from torch.distributions import Distribution


from torch.distributions import Normal


from torch.distributions import TransformedDistribution


from typing import Type


from torch import distributions


from torch import nn


import scipy.stats


from collections import namedtuple


import copy


from numpy.lib.function_base import iterable


import torch.nn as nn


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from copy import copy


from matplotlib.pyplot import plot_date


import torch.distributions as dists


from torch.utils.data.dataloader import DataLoader


from matplotlib import pyplot as plt


from functools import partial


from re import S


from torch import embedding


from abc import ABC


from abc import abstractmethod


import math


import logging


from torch.optim.optimizer import Optimizer


from torch.fft import irfft


from torch.fft import rfft


import itertools


from sklearn.utils.validation import NotFittedError


from torch.optim import SGD


class ImplicitQuantileNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.quantile_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU(), nn.Linear(hidden_size, input_size))
        self.output_layer = nn.Sequential(nn.Linear(input_size, input_size), nn.PReLU(), nn.Linear(input_size, 1))
        self.register_buffer('cos_multipliers', torch.arange(0, hidden_size) * torch.pi)

    def forward(self, x: torch.Tensor, quantiles: torch.Tensor) ->torch.Tensor:
        cos_emb_tau = torch.cos(quantiles[:, None] * self.cos_multipliers[None])
        cos_emb_tau = self.quantile_layer(cos_emb_tau)
        emb_inputs = x.unsqueeze(-2) * (1.0 + cos_emb_tau)
        emb_outputs = self.output_layer(emb_inputs).squeeze(-1)
        return emb_outputs


class FullyConnectedModule(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, activation_class: nn.ReLU, dropout: float=None, norm: bool=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_class = activation_class
        self.dropout = dropout
        self.norm = norm
        module_list = [nn.Linear(input_size, hidden_size), activation_class()]
        if dropout is not None:
            module_list.append(nn.Dropout(dropout))
        if norm:
            module_list.append(nn.LayerNorm(hidden_size))
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), activation_class()])
            if dropout is not None:
                module_list.append(nn.Dropout(dropout))
            if norm:
                module_list.append(nn.LayerNorm(hidden_size))
        module_list.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.sequential(x)


def linear(input_size, output_size, bias=True, dropout: int=None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


class NBEATSBlock(nn.Module):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, share_thetas=False, dropout=0.1):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        fc_stack = [nn.Linear(backcast_length, units), nn.ReLU()]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


def linspace(backcast_length: int, forecast_length: int, centered: bool=False) ->Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSSeasonalBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim=None, num_block_layers=4, backcast_length=10, forecast_length=5, nb_harmonics=None, min_period=1, dropout=0.1):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        self.min_period = min_period
        super().__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length, share_thetas=True, dropout=dropout)
        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=False)
        p1, p2 = (thetas_dim // 2, thetas_dim // 2) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1)
        s1_b = torch.tensor([np.cos(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32)
        s2_b = torch.tensor([np.sin(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32)
        self.register_buffer('S_backcast', torch.cat([s1_b, s2_b]))
        s1_f = torch.tensor([np.cos(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p1)], dtype=torch.float32)
        s2_f = torch.tensor([np.sin(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p2)], dtype=torch.float32)
        self.register_buffer('S_forecast', torch.cat([s1_f, s2_f]))

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        amplitudes_backward = self.theta_b_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = self.theta_f_fc(x)
        forecast = amplitudes_forward.mm(self.S_forecast)
        return backcast, forecast

    def get_frequencies(self, n):
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)


class NBEATSTrendBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, dropout=0.1):
        super().__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length, share_thetas=True, dropout=dropout)
        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
        norm = np.sqrt(forecast_length / thetas_dim)
        coefficients = torch.tensor([(backcast_linspace ** i) for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer('T_backcast', coefficients * norm)
        coefficients = torch.tensor([(forecast_linspace ** i) for i in range(thetas_dim)], dtype=torch.float32)
        self.register_buffer('T_forecast', coefficients * norm)

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4, backcast_length=10, forecast_length=5, dropout=0.1):
        super().__init__(units=units, thetas_dim=thetas_dim, num_block_layers=num_block_layers, backcast_length=backcast_length, forecast_length=forecast_length, dropout=dropout)
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class StaticFeaturesEncoder(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        layers = [nn.Dropout(p=0.5), nn.Linear(in_features=in_features, out_features=out_features), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class IdentityBasis(nn.Module):

    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert interpolation_mode in ['linear', 'nearest'] or 'cubic' in interpolation_mode
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, backcast_theta: torch.Tensor, forecast_theta: torch.Tensor, encoder_x_t: torch.Tensor, decoder_x_t: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        backcast = backcast_theta
        knots = forecast_theta
        if self.interpolation_mode == 'nearest':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == 'linear':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif 'cubic' in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split('-')[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size))
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i * batch_size:(i + 1) * batch_size], size=self.forecast_size, mode='bicubic')
                forecast[i * batch_size:(i + 1) * batch_size] += forecast_i[:, 0, 0, :]
        return backcast, forecast


ACTIVATIONS = ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']


class NHiTSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(self, context_length: int, prediction_length: int, output_size: int, covariate_size: int, static_size: int, static_hidden_size: int, n_theta: int, hidden_size: List[int], pooling_sizes: int, pooling_mode: str, basis: nn.Module, n_layers: int, batch_normalization: bool, dropout: float, activation: str):
        super().__init__()
        assert pooling_mode in ['max', 'average']
        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))
        if static_size == 0:
            static_hidden_size = 0
        self.context_length = context_length
        self.output_size = output_size
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.static_size = static_size
        self.static_hidden_size = static_hidden_size
        self.covariate_size = covariate_size
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.hidden_size = [self.context_length_pooled * len(self.output_size) + (self.context_length + self.prediction_length) * self.covariate_size + self.static_hidden_size] + hidden_size
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()
        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1]))
            hidden_layers.append(activ)
            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i + 1]))
            if self.dropout > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout))
        output_layer = [nn.Linear(in_features=self.hidden_size[-1], out_features=context_length * len(output_size) + n_theta * sum(output_size))]
        layers = hidden_layers + output_layer
        if self.static_size > 0 and self.static_hidden_size > 0:
            self.static_encoder = StaticFeaturesEncoder(in_features=static_size, out_features=static_hidden_size)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, encoder_y: torch.Tensor, encoder_x_t: torch.Tensor, decoder_x_t: torch.Tensor, x_s: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)
        encoder_y = encoder_y.transpose(1, 2)
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)
        if self.covariate_size > 0:
            encoder_y = torch.cat((encoder_y, encoder_x_t.reshape(batch_size, -1), decoder_x_t.reshape(batch_size, -1)), 1)
        if self.static_size > 0 and self.static_hidden_size > 0:
            x_s = self.static_encoder(x_s)
            encoder_y = torch.cat((encoder_y, x_s), 1)
        theta = self.layers(encoder_y)
        backcast_theta = theta[:, :self.context_length * len(self.output_size)].reshape(-1, self.context_length)
        forecast_theta = theta[:, self.context_length * len(self.output_size):].reshape(-1, self.n_theta)
        backcast, forecast = self.basis(backcast_theta, forecast_theta, encoder_x_t, decoder_x_t)
        backcast = backcast.reshape(-1, len(self.output_size), self.context_length).transpose(1, 2)
        forecast = forecast.reshape(-1, sum(self.output_size), self.prediction_length).transpose(1, 2)
        return backcast, forecast


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == 'orthogonal':
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass
        else:
            assert 1 < 0, f'Initialization {initialization} not found'


class NHiTS(nn.Module):
    """
    N-HiTS Model.
    """

    def __init__(self, context_length, prediction_length, output_size: int, static_size, covariate_size, static_hidden_size, n_blocks: list, n_layers: list, hidden_size: list, pooling_sizes: list, downsample_frequencies: list, pooling_mode, interpolation_mode, dropout, activation, initialization, batch_normalization, shared_weights, naive_level: bool):
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_size = output_size
        self.naive_level = naive_level
        blocks = self.create_stack(n_blocks=n_blocks, context_length=context_length, prediction_length=prediction_length, output_size=output_size, covariate_size=covariate_size, static_size=static_size, static_hidden_size=static_hidden_size, n_layers=n_layers, hidden_size=hidden_size, pooling_sizes=pooling_sizes, downsample_frequencies=downsample_frequencies, pooling_mode=pooling_mode, interpolation_mode=interpolation_mode, batch_normalization=batch_normalization, dropout=dropout, activation=activation, shared_weights=shared_weights, initialization=initialization)
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(self, n_blocks, context_length, prediction_length, output_size, covariate_size, static_size, static_hidden_size, n_layers, hidden_size, pooling_sizes, downsample_frequencies, pooling_mode, interpolation_mode, batch_normalization, dropout, activation, shared_weights, initialization):
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):
                if len(block_list) == 0 and batch_normalization:
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)
                    basis = IdentityBasis(backcast_size=context_length, forecast_size=prediction_length, interpolation_mode=interpolation_mode)
                    nbeats_block = NHiTSBlock(context_length=context_length, prediction_length=prediction_length, output_size=output_size, covariate_size=covariate_size, static_size=static_size, static_hidden_size=static_hidden_size, n_theta=n_theta, hidden_size=hidden_size[i], pooling_sizes=pooling_sizes[i], pooling_mode=pooling_mode, basis=basis, n_layers=n_layers[i], batch_normalization=batch_normalization_block, dropout=dropout, activation=activation)
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, encoder_y, encoder_mask, encoder_x_t, decoder_x_t, x_s):
        residuals = encoder_y
        encoder_mask = encoder_mask.unsqueeze(-1)
        level = encoder_y[:, -1:].repeat(1, self.prediction_length, 1)
        forecast_level = level.repeat_interleave(torch.tensor(self.output_size, device=level.device), dim=2)
        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [encoder_y[:, -1:].repeat(1, self.context_length, 1)]
            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)
        for block in self.blocks:
            block_backcast, block_forecast = block(encoder_y=residuals, encoder_x_t=encoder_x_t, decoder_x_t=decoder_x_t, x_s=x_s)
            residuals = (residuals - block_backcast) * encoder_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)
        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        backcast = residuals
        return forecast, backcast, block_forecasts, block_backcasts


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):

    def __init__(self, *args, batch_first: bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = super().forward(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


def get_embedding_size(n: int, max_size: int=100) ->int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n ** 0.56), max_size)
    else:
        return 1


class MultiEmbedding(nn.Module):
    concat_output: bool

    def __init__(self, embedding_sizes: Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]], x_categoricals: List[str]=None, categorical_groups: Dict[str, List[str]]={}, embedding_paddings: List[str]=[], max_embedding_size: int=None):
        """Embedding layer for categorical variables including groups of categorical variables.

        Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).

        Args:
            embedding_sizes (Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]]):
                either

                * dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
                  indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding
                  dimensions. Use :py:func:`~pytorch_forecasting.utils.get_embedding_size` to automatically obtain
                  reasonable embedding sizes depending on the number of categories.
                * dictionary of categorical sizes, e.g. ``{'cat1': 10}`` where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size`.
                * list of embedding and categorical sizes, e.g. ``[(10, 3), (20, 2)]`` (requires ``x_categoricals`` to
                  be empty)
                * list of categorical sizes where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size` (requires ``x_categoricals`` to be empty).

                If input is provided as list, output will be a single tensor of shape batch x (optional) time x
                sum(embedding_sizes). Otherwise, output is a dictionary of embedding tensors.
            x_categoricals (List[str]): list of categorical variables that are used as input.
            categorical_groups (Dict[str, List[str]]): dictionary of categories that should be summed up in an
                embedding bag, e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates that a new categorical variable ``'cat1'``
                is mapped to an embedding bag containing the second and third categorical variables.
                Defaults to empty dictionary.
            embedding_paddings (List[str]): list of categorical variables for which the value 0 is mapped to a zero
                embedding vector. Defaults to empty list.
            max_embedding_size (int, optional): if embedding size defined by ``embedding_sizes`` is larger than
                ``max_embedding_size``, it will be constrained. Defaults to None.
        """
        super().__init__()
        if isinstance(embedding_sizes, dict):
            self.concat_output = False
            assert x_categoricals is not None, 'x_categoricals must be provided.'
            categorical_group_variables = [name for names in categorical_groups.values() for name in names]
            if len(categorical_groups) > 0:
                assert all(name in embedding_sizes for name in categorical_groups), 'categorical_groups must be in embedding_sizes.'
                assert not any(name in embedding_sizes for name in categorical_group_variables), 'group variables in categorical_groups must not be in embedding_sizes.'
                assert all(name in x_categoricals for name in categorical_group_variables), 'group variables in categorical_groups must be in x_categoricals.'
            assert all(name in embedding_sizes for name in embedding_sizes if name not in categorical_group_variables), 'all variables in embedding_sizes must be in x_categoricals - but only ifnot already in categorical_groups.'
        else:
            assert x_categoricals is None and len(categorical_groups) == 0, 'If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty.'
            embedding_sizes = {str(name): size for name, size in enumerate(embedding_sizes)}
            x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True
        self.embedding_sizes = {name: ((size, get_embedding_size(size)) if isinstance(size, int) else size) for name, size in embedding_sizes.items()}
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals
        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size
            if name in self.categorical_groups:
                self.embeddings[name] = TimeDistributedEmbeddingBag(self.embedding_sizes[name][0], embedding_size, mode='sum', batch_first=True)
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(self.embedding_sizes[name][0], embedding_size, padding_idx=padding_idx)

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) ->int:
        return len(self.x_categoricals)

    @property
    def output_size(self) ->Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) ->Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape batch x (optional) time x categoricals in the order of
                ``x_categoricals``.

        Returns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: dictionary of category names to embeddings
                of shape batch x (optional) time x embedding_size if ``embedding_size`` is given as dictionary.
                Otherwise, returns the embedding of shape batch x (optional) time x sum(embedding_sizes).
                Query attribute ``output_size`` to get the size of the output(s).
        """
        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(x[..., [self.x_categoricals.index(cat_name) for cat_name in self.categorical_groups[name]]])
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])
        if self.concat_output:
            return torch.cat(list(input_vectors.values()), dim=-1)
        else:
            return input_vectors


HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class RNN(ABC, nn.RNNBase):
    """
    Base class flexible RNNs.

    Forward function can handle sequences of length 0.
    """

    @abstractmethod
    def handle_no_encoding(self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState) ->HiddenState:
        """
        Mask the hidden_state where there is no encoding.

        Args:
            hidden_state (HiddenState): hidden state where some entries need replacement
            no_encoding (torch.BoolTensor): positions that need replacement
            initial_hidden_state (HiddenState): hidden state to use for replacement

        Returns:
            HiddenState: hidden state with propagated initial hidden state where appropriate
        """
        pass

    @abstractmethod
    def init_hidden_state(self, x: torch.Tensor) ->HiddenState:
        """
        Initialise a hidden_state.

        Args:
            x (torch.Tensor): network input

        Returns:
            HiddenState: default (zero-like) hidden state
        """
        pass

    @abstractmethod
    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) ->HiddenState:
        """
        Duplicate the hidden_state n_samples times.

        Args:
            hidden_state (HiddenState): hidden state to repeat
            n_samples (int): number of repetitions

        Returns:
            HiddenState: repeated hidden state
        """
        pass

    def forward(self, x: Union[rnn.PackedSequence, torch.Tensor], hx: HiddenState=None, lengths: torch.LongTensor=None, enforce_sorted: bool=True) ->Tuple[Union[rnn.PackedSequence, torch.Tensor], HiddenState]:
        """
        Forward function of rnn that allows zero-length sequences.

        Functions as normal for RNN. Only changes output if lengths are defined.

        Args:
            x (Union[rnn.PackedSequence, torch.Tensor]): input to RNN. either packed sequence or tensor of
                padded sequences
            hx (HiddenState, optional): hidden state. Defaults to None.
            lengths (torch.LongTensor, optional): lengths of sequences. If not None, used to determine correct returned
                hidden state. Can contain zeros. Defaults to None.
            enforce_sorted (bool, optional): if lengths are passed, determines if RNN expects them to be sorted.
                Defaults to True.

        Returns:
            Tuple[Union[rnn.PackedSequence, torch.Tensor], HiddenState]: output and hidden state.
                Output is packed sequence if input has been a packed sequence.
        """
        if isinstance(x, rnn.PackedSequence) or lengths is None:
            assert lengths is None, 'cannot combine x of type PackedSequence with lengths argument'
            return super().forward(x, hx=hx)
        else:
            min_length = lengths.min()
            max_length = lengths.max()
            assert min_length >= 0, 'sequence lengths must be great equals 0'
            if max_length == 0:
                hidden_state = self.init_hidden_state(x)
                if self.batch_first:
                    out = torch.zeros(lengths.size(0), x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
                else:
                    out = torch.zeros(x.size(0), lengths.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
                return out, hidden_state
            else:
                pack_lengths = lengths.where(lengths > 0, torch.ones_like(lengths))
                packed_out, hidden_state = super().forward(rnn.pack_padded_sequence(x, pack_lengths.cpu(), enforce_sorted=enforce_sorted, batch_first=self.batch_first), hx=hx)
                if min_length == 0:
                    no_encoding = (lengths == 0)[None, :, None]
                    if hx is None:
                        initial_hidden_state = self.init_hidden_state(x)
                    else:
                        initial_hidden_state = hx
                    hidden_state = self.handle_no_encoding(hidden_state, no_encoding, initial_hidden_state)
                out, _ = rnn.pad_packed_sequence(packed_out, batch_first=self.batch_first)
                return out, hidden_state


class LSTM(RNN, nn.LSTM):
    """LSTM that can handle zero-length sequences"""

    def handle_no_encoding(self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState) ->HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.masked_scatter(no_encoding, initial_hidden_state[0])
        cell = cell.masked_scatter(no_encoding, initial_hidden_state[0])
        return hidden, cell

    def init_hidden_state(self, x: torch.Tensor) ->HiddenState:
        num_directions = 2 if self.bidirectional else 1
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        hidden = torch.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        cell = torch.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        return hidden, cell

    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) ->HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, 1)
        cell = cell.repeat_interleave(n_samples, 1)
        return hidden, cell


class GRU(RNN, nn.GRU):
    """GRU that can handle zero-length sequences"""

    def handle_no_encoding(self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState) ->HiddenState:
        return hidden_state.masked_scatter(no_encoding, initial_hidden_state)

    def init_hidden_state(self, x: torch.Tensor) ->HiddenState:
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        hidden = torch.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        return hidden

    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) ->HiddenState:
        return hidden_state.repeat_interleave(n_samples, 1)


class TimeDistributed(nn.Module):

    def __init__(self, module: nn.Module, batch_first: bool=False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class TimeDistributedInterpolation(nn.Module):

    def __init__(self, output_size: int, batch_first: bool=False, trainable: bool=False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode='linear', align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.interpolate(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int=None, dropout: float=None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'fc' in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class ResampleNorm(nn.Module):

    def __init__(self, input_size: int, output_size: int=None, trainable_add: bool=True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size
        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)
        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0
        output = self.norm(x)
        return output


class AddNorm(nn.Module):

    def __init__(self, input_size: int, skip_size: int=None, trainable_add: bool=True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):

    def __init__(self, input_size: int, hidden_size: int=None, skip_size: int=None, trainable_add: bool=False, dropout: float=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout
        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float=0.1, context_size: int=None, residual: bool=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual
        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size
        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()
        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()
        self.gate_norm = GateAddNorm(input_size=self.hidden_size, skip_size=self.output_size, hidden_size=self.output_size, dropout=self.dropout, trainable_add=False)

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(p)
            elif 'fc1' in name or 'fc2' in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'context' in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x
        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):

    def __init__(self, input_sizes: Dict[str, int], hidden_size: int, input_embedding_flags: Dict[str, bool]={}, dropout: float=0.1, context_size: int=None, single_variable_grns: Dict[str, GatedResidualNetwork]={}, prescalers: Dict[str, nn.Linear]={}):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size
        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(self.input_size_total, min(self.hidden_size, self.num_inputs), self.num_inputs, self.dropout, self.context_size, residual=False)
            else:
                self.flattened_grn = GatedResidualNetwork(self.input_size_total, min(self.hidden_size, self.num_inputs), self.num_inputs, self.dropout, residual=False)
        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, self.hidden_size), output_size=self.hidden_size, dropout=self.dropout)
            if name in prescalers:
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)
        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor=None):
        if self.num_inputs > 1:
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)
            if outputs.ndim == 3:
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)
            else:
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparse_weights


class PositionalEncoder(torch.nn.Module):

    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, 'model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))'
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
                pe[pos, i + 1] = math.cos(pos / 10000 ** (2 * (i + 1) / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float=None, scale: bool=True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))
        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension
        if mask is not None:
            attn = attn.masked_fill(mask, -1000000000.0)
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):

    def __init__(self, n_head: int, d_model: int, dropout: float=0.0):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) ->Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        return outputs, attn


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FullyConnectedModule,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_size': 4, 'n_hidden_layers': 1, 'activation_class': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GateAddNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedLinearUnit,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedResidualNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ImplicitQuantileNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (InterpretableMultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (LSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PositionalEncoder,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResampleNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (StaticFeaturesEncoder,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeDistributed,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeDistributedInterpolation,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_jdb78_pytorch_forecasting(_paritybench_base):
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

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

