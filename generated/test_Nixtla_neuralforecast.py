import sys
_module = sys.modules[__name__]
del sys
cv_models = _module
data = _module
evaluation = _module
models = _module
neuralforecast = _module
_modidx = _module
auto = _module
common = _module
_base_auto = _module
_base_recurrent = _module
_base_windows = _module
_modules = _module
_scalers = _module
core = _module
losses = _module
numpy = _module
pytorch = _module
dilated_rnn = _module
gru = _module
lstm = _module
mlp = _module
nbeats = _module
nbeatsx = _module
nhits = _module
rnn = _module
tcn = _module
tft = _module
tsdataset = _module
utils = _module
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


import torch


from copy import deepcopy


import random


import numpy as np


import torch.nn as nn


from typing import Optional


from typing import Union


from typing import Tuple


import math


import torch.nn.functional as F


from torch.distributions import Normal


from torch.distributions import StudentT


from torch.distributions import Poisson


from torch.distributions import AffineTransform


from torch.distributions import Distribution


from torch.distributions import TransformedDistribution


from typing import List


from torch import Tensor


from torch.nn import LayerNorm


import logging


import warnings


from collections.abc import Mapping


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


def identity_scaler(x, mask, dim=-1, eps=1e-06):
    """Identity Scaler

    A placeholder identity scaler, that is argument insensitive.

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `x`: original torch.Tensor `x`.
    """
    x_shift = torch.zeros_like(x)[:, [0], :]
    x_scale = torch.ones_like(x)[:, [0], :]
    return x, x_shift, x_scale


def inv_identity_scaler(z, x_shift, x_scale):
    return z * x_scale + x_shift


def inv_invariant_scaler(z, x_median, x_mad):
    return torch.sinh(z) * x_mad + x_median


def inv_minmax1_scaler(z, x_min, x_range):
    z = (z + 1) / 2
    return z * x_range + x_min


def inv_minmax_scaler(z, x_min, x_range):
    return z * x_range + x_min


def inv_robust_scaler(z, x_median, x_mad):
    return z * x_mad + x_median


def inv_std_scaler(z, x_mean, x_std):
    return z * x_std + x_mean


def masked_mean(x, mask, dim=-1, keepdim=True):
    """Masked  Mean

    Compute the mean of tensor `x` along dimension, ignoring values where
    `mask` is False. `x` and `mask` need to be broadcastable.

    **Parameters:**<br>
    `x`: torch.Tensor to compute mean of along `dim` dimension.<br>
    `mask`: torch Tensor bool with same shape as `x`, where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `dim` (int, optional): Dimension to take mean of. Defaults to -1.<br>
    `keepdim` (bool, optional): Keep dimension of `x` or not. Defaults to True.<br>

    **Returns:**<br>
    `x_mean`: torch.Tensor with normalized values.
    """
    x_nan = x.float().masked_fill(mask < 1, float('nan'))
    x_mean = x_nan.nanmean(dim=dim, keepdim=keepdim)
    return x_mean


def masked_median(x, mask, dim=-1, keepdim=True):
    """Masked Median

    Compute the median of tensor `x` along dim, ignoring values where
    `mask` is False. `x` and `mask` need to be broadcastable.

    **Parameters:**<br>
    `x`: torch.Tensor to compute median of along `dim` dimension.<br>
    `mask`: torch Tensor bool with same shape as `x`, where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `dim` (int, optional): Dimension to take median of. Defaults to -1.<br>
    `keepdim` (bool, optional): Keep dimension of `x` or not. Defaults to True.<br>

    **Returns:**<br>
    `x_median`: torch.Tensor with normalized values.
    """
    x_nan = x.float().masked_fill(mask < 1, float('nan'))
    x_median, _ = x_nan.nanmedian(dim=dim, keepdim=keepdim)
    return x_median


def invariant_scaler(x, mask, dim=-1, eps=1e-06):
    """Invariant Median Scaler

    Standardizes features by removing the median and scaling
    with the mean absolute deviation (mad) a robust estimator of variance.
    Aditionally it complements the transformation with the arcsinh transformation.

    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\\mathbf{z} = (\\mathbf{x}_{[B,T,C]}-\\textrm{median}(\\mathbf{x})_{[B,1,C]})/\\textrm{mad}(\\mathbf{x})_{[B,1,C]}$$

    $$\\mathbf{z} = \\textrm{arcsinh}(\\mathbf{z})$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_median = masked_median(x=x, mask=mask, dim=dim)
    x_mad = masked_median(x=torch.abs(x - x_median), mask=mask, dim=dim)
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x - x_means) ** 2, mask=mask, dim=dim))
    x_mad_aux = x_stds / 0.6744897501960817
    x_mad = x_mad * (x_mad > 0) + x_mad_aux * (x_mad == 0)
    x_mad[x_mad == 0] = 1.0
    x_mad = x_mad + eps
    z = torch.arcsinh((x - x_median) / x_mad)
    return z, x_median, x_mad


def minmax1_scaler(x, mask, eps=1e-06, dim=-1):
    """MinMax1 Scaler

    Standardizes temporal features by ensuring its range dweels between
    [-1,1] range. This transformation is often used as an alternative
    to the standard scaler or classic Min Max Scaler.
    The scaled features are obtained as:

    $$\\mathbf{z} = 2 (\\mathbf{x}_{[B,T,C]}-\\mathrm{min}({\\mathbf{x}})_{[B,1,C]})/ (\\mathrm{max}({\\mathbf{x}})_{[B,1,C]}- \\mathrm{min}({\\mathbf{x}})_{[B,1,C]})-1$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute min and max. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    max_mask = (mask == 0) * -1000000000000.0
    min_mask = (mask == 0) * 1000000000000.0
    x_max = torch.max(x + max_mask, dim=dim, keepdim=True)[0]
    x_min = torch.min(x + min_mask, dim=dim, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    x_range = x_range + eps
    x = (x - x_min) / x_range
    z = x * 2 - 1
    return z, x_min, x_range


def minmax_scaler(x, mask, eps=1e-06, dim=-1):
    """MinMax Scaler

    Standardizes temporal features by ensuring its range dweels between
    [0,1] range. This transformation is often used as an alternative
    to the standard scaler. The scaled features are obtained as:

    $$\\mathbf{z} = (\\mathbf{x}_{[B,T,C]}-\\mathrm{min}({\\mathbf{x}})_{[B,1,C]})/
        (\\mathrm{max}({\\mathbf{x}})_{[B,1,C]}- \\mathrm{min}({\\mathbf{x}})_{[B,1,C]})$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute min and max. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    max_mask = (mask == 0) * -1000000000000.0
    min_mask = (mask == 0) * 1000000000000.0
    x_max = torch.max(x + max_mask, dim=dim, keepdim=True)[0]
    x_min = torch.min(x + min_mask, dim=dim, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    x_range = x_range + eps
    z = (x - x_min) / x_range
    return z, x_min, x_range


def robust_scaler(x, mask, dim=-1, eps=1e-06):
    """Robust Median Scaler

    Standardizes features by removing the median and scaling
    with the mean absolute deviation (mad) a robust estimator of variance.
    This scaler is particularly useful with noisy data where outliers can
    heavily influence the sample mean / variance in a negative way.
    In these scenarios the median and amd give better results.

    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\\mathbf{z} = (\\mathbf{x}_{[B,T,C]}-\\textrm{median}(\\mathbf{x})_{[B,1,C]})/\\textrm{mad}(\\mathbf{x})_{[B,1,C]}$$

    $$\\textrm{mad}(\\mathbf{x}) = \\frac{1}{N} \\sum_{}|\\mathbf{x} - \\mathrm{median}(x)|$$

    **Parameters:**<br>
    `x`: torch.Tensor input tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute median and mad. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_median = masked_median(x=x, mask=mask, dim=dim)
    x_mad = masked_median(x=torch.abs(x - x_median), mask=mask, dim=dim)
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x - x_means) ** 2, mask=mask, dim=dim))
    x_mad_aux = x_stds / 0.6744897501960817
    x_mad = x_mad * (x_mad > 0) + x_mad_aux * (x_mad == 0)
    x_mad[x_mad == 0] = 1.0
    x_mad = x_mad + eps
    z = (x - x_median) / x_mad
    return z, x_median, x_mad


def std_scaler(x, mask, dim=-1, eps=1e-06):
    """Standard Scaler

    Standardizes features by removing the mean and scaling
    to unit variance along the `dim` dimension.

    For example, for `base_windows` models, the scaled features are obtained as (with dim=1):

    $$\\mathbf{z} = (\\mathbf{x}_{[B,T,C]}-\\bar{\\mathbf{x}}_{[B,1,C]})/\\hat{\\sigma}_{[B,1,C]}$$

    **Parameters:**<br>
    `x`: torch.Tensor.<br>
    `mask`: torch Tensor bool, same dimension as `x`, indicates where `x` is valid and False
            where `x` should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>
    `dim` (int, optional): Dimension over to compute mean and std. Defaults to -1.<br>

    **Returns:**<br>
    `z`: torch.Tensor same shape as `x`, except scaled.
    """
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x - x_means) ** 2, mask=mask, dim=dim))
    x_stds[x_stds == 0] = 1.0
    x_stds = x_stds + eps
    z = (x - x_means) / x_stds
    return z, x_means, x_stds


class TemporalNorm(nn.Module):
    """Temporal Normalization

    Standardization of the features is a common requirement for many
    machine learning estimators, and it is commonly achieved by removing
    the level and scaling its variance. The `TemporalNorm` module applies
    temporal normalization over the batch of inputs as defined by the type of scaler.

    $$\\mathbf{z}_{[B,T,C]} = \\textrm{Scaler}(\\mathbf{x}_{[B,T,C]})$$

    **Parameters:**<br>
    `scaler_type`: str, defines the type of scaler used by TemporalNorm.
                    available [`identity`, `standard`, `robust`, `minmax`, `minmax1`, `invariant`].<br>
    `dim` (int, optional): Dimension over to compute scale and shift. Defaults to -1.<br>
    `eps` (float, optional): Small value to avoid division by zero. Defaults to 1e-6.<br>

    """

    def __init__(self, scaler_type='robust', dim=-1, eps=1e-06):
        super().__init__()
        scalers = {None: identity_scaler, 'identity': identity_scaler, 'standard': std_scaler, 'robust': robust_scaler, 'minmax': minmax_scaler, 'minmax1': minmax1_scaler, 'invariant': invariant_scaler}
        inverse_scalers = {None: inv_identity_scaler, 'identity': inv_identity_scaler, 'standard': inv_std_scaler, 'robust': inv_robust_scaler, 'minmax': inv_minmax_scaler, 'minmax1': inv_minmax1_scaler, 'invariant': inv_invariant_scaler}
        assert scaler_type in scalers.keys(), f'{scaler_type} not defined'
        self.scaler = scalers[scaler_type]
        self.inverse_scaler = inverse_scalers[scaler_type]
        self.scaler_type = scaler_type
        self.dim = dim
        self.eps = eps

    def transform(self, x, mask):
        """Center and scale the data.

        **Parameters:**<br>
        `x`: torch.Tensor shape [batch, time, channels].<br>
        `mask`: torch Tensor bool, shape  [batch, time] where `x` is valid and False
                where `x` should be masked. Mask should not be all False in any column of
                dimension dim to avoid NaNs from zero division.<br>

        **Returns:**<br>
        `z`: torch.Tensor same shape as `x`, except scaled.
        """
        z, x_shift, x_scale = self.scaler(x=x, mask=mask, dim=self.dim, eps=self.eps)
        self.x_shift = x_shift
        self.x_scale = x_scale
        return z

    def inverse_transform(self, z, x_shift=None, x_scale=None):
        """Scale back the data to the original representation.

        **Parameters:**<br>
        `z`: torch.Tensor shape [batch, time, channels], scaled.<br>

        **Returns:**<br>
        `x`: torch.Tensor original data.
        """
        if x_shift is None:
            x_shift = self.x_shift
        if x_scale is None:
            x_scale = self.x_scale
        x = self.inverse_scaler(z, x_shift, x_scale)
        return x


class TimeSeriesDataset(Dataset):

    def __init__(self, temporal, temporal_cols, indptr, max_size, static=None, static_cols=None, sorted=False):
        super().__init__()
        self.temporal = torch.tensor(temporal, dtype=torch.float)
        self.temporal_cols = pd.Index(list(temporal_cols) + ['available_mask'])
        if static is not None:
            self.static = torch.tensor(static, dtype=torch.float)
            self.static_cols = static_cols
        else:
            self.static = static
            self.static_cols = static_cols
        self.indptr = indptr
        self.n_groups = self.indptr.size - 1
        self.max_size = max_size
        self.updated = False
        self.sorted = sorted

    def __getitem__(self, idx):
        if isinstance(idx, int):
            temporal = torch.zeros(size=(len(self.temporal_cols), self.max_size), dtype=torch.float32)
            ts = self.temporal[self.indptr[idx]:self.indptr[idx + 1], :]
            temporal[:len(self.temporal_cols) - 1, -len(ts):] = ts.permute(1, 0)
            temporal[len(self.temporal_cols) - 1, -len(ts):] = 1
            static = None if self.static is None else self.static[idx, :]
            item = dict(temporal=temporal, temporal_cols=self.temporal_cols, static=static, static_cols=self.static_cols)
            return item
        raise ValueError(f'idx must be int, got {type(idx)}')

    def __len__(self):
        return self.n_groups

    def __repr__(self):
        return f'TimeSeriesDataset(n_data={self.data.size:,}, n_groups={self.n_groups:,})'

    def __eq__(self, other):
        if not hasattr(other, 'data') or not hasattr(other, 'indptr'):
            return False
        return np.allclose(self.data, other.data) and np.array_equal(self.indptr, other.indptr)

    @staticmethod
    def update_dataset(dataset, future_df):
        """Add future observations to the dataset."""
        temporal_cols = dataset.temporal_cols.copy()
        temporal_cols = temporal_cols.delete(len(temporal_cols) - 1)
        for col in temporal_cols:
            if col not in future_df.columns:
                future_df[col] = None
        future_df = future_df[['unique_id', 'ds'] + temporal_cols.tolist()]
        futr_dataset, indices, futr_dates, futr_index = dataset.from_df(df=future_df, sort_df=dataset.sorted)
        len_temporal, col_temporal = dataset.temporal.shape
        new_temporal = torch.zeros(size=(len_temporal + len(future_df), col_temporal))
        new_indptr = [0]
        new_max_size = 0
        acum = 0
        for i in range(dataset.n_groups):
            series_length = dataset.indptr[i + 1] - dataset.indptr[i]
            new_length = series_length + futr_dataset.indptr[i + 1] - futr_dataset.indptr[i]
            new_temporal[acum:acum + series_length, :] = dataset.temporal[dataset.indptr[i]:dataset.indptr[i + 1], :]
            new_temporal[acum + series_length:acum + new_length, :] = futr_dataset.temporal[futr_dataset.indptr[i]:futr_dataset.indptr[i + 1], :]
            acum += new_length
            new_indptr.append(acum)
            if new_length > new_max_size:
                new_max_size = new_length
        updated_dataset = TimeSeriesDataset(temporal=new_temporal, temporal_cols=temporal_cols, indptr=np.array(new_indptr).astype(np.int32), max_size=new_max_size, static=dataset.static, static_cols=dataset.static_cols, sorted=dataset.sorted)
        return updated_dataset

    @staticmethod
    def from_df(df, static_df=None, sort_df=False):
        if df.index.name != 'unique_id':
            df = df.set_index('unique_id')
            if static_df is not None:
                static_df = static_df.set_index('unique_id')
        df = df.set_index('ds', append=True)
        if not df.index.is_monotonic_increasing and sort_df:
            df = df.sort_index()
            if static_df is not None:
                static_df = static_df.sort_index()
        temporal = df.values.astype(np.float32)
        temporal_cols = df.columns
        indices_sizes = df.index.get_level_values('unique_id').value_counts(sort=False)
        indices = indices_sizes.index
        sizes = indices_sizes.values
        max_size = max(sizes)
        cum_sizes = sizes.cumsum()
        dates = df.index.get_level_values('ds')[cum_sizes - 1]
        indptr = np.append(0, cum_sizes).astype(np.int32)
        if static_df is not None:
            static = static_df.values
            static_cols = static_df.columns
        else:
            static = None
            static_cols = None
        dataset = TimeSeriesDataset(temporal=temporal, temporal_cols=temporal_cols, static=static, static_cols=static_cols, indptr=indptr, max_size=max_size, sorted=sort_df)
        return dataset, indices, dates, df.index


class TimeSeriesLoader(DataLoader):
    """TimeSeriesLoader DataLoader.
    [Source code](https://github.com/Nixtla/neuralforecast1/blob/main/neuralforecast/tsdataset.py).

    Small change to PyTorch's Data loader.
    Combines a dataset and a sampler, and provides an iterable over the given dataset.

    The class `~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    **Parameters:**<br>
    `batch_size`: (int, optional): how many samples per batch to load (default: 1).<br>
    `shuffle`: (bool, optional): set to `True` to have the data reshuffled at every epoch (default: `False`).<br>
    `sampler`: (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.<br>
                Can be any `Iterable` with `__len__` implemented. If specified, `shuffle` must not be specified.<br>
    """

    def __init__(self, dataset, **kwargs):
        if 'collate_fn' in kwargs:
            kwargs.pop('collate_fn')
        kwargs_ = {**kwargs, **dict(collate_fn=self._collate_fn)}
        DataLoader.__init__(self, dataset=dataset, **kwargs_)

    def _collate_fn(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        elif isinstance(elem, Mapping):
            if elem['static'] is None:
                return dict(temporal=self.collate_fn([d['temporal'] for d in batch]), temporal_cols=elem['temporal_cols'])
            return dict(static=self.collate_fn([d['static'] for d in batch]), static_cols=elem['static_cols'], temporal=self.collate_fn([d['temporal'] for d in batch]), temporal_cols=elem['temporal_cols'])
        raise TypeError(f'Unknown {elem_type}')


class MAE(torch.nn.Module):
    """Mean Absolute Error

    Calculates Mean Absolute Error between
    `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    $$ \\mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} |y_{\\tau} - \\hat{y}_{\\tau}| $$
    """

    def __init__(self):
        super(MAE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mae`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y)
        mae = torch.abs(y - y_hat) * mask
        mae = torch.mean(mae)
        return mae


class Chomp1d(nn.Module):
    """Chomp1d

    Receives `x` input of dim [N,C,T], and trims it so that only
    'time available' information is used.
    Used by one dimensional causal convolutions `CausalConv1d`.

    **Parameters:**<br>
    `horizon`: int, length of outsample values to skip.
    """

    def __init__(self, horizon):
        super(Chomp1d, self).__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, :, :-self.horizon].contiguous()


ACTIVATIONS = ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']


class CausalConv1d(nn.Module):
    """Causal Convolution 1d

    Receives `x` input of dim [N,C_in,T], and computes a causal convolution
    in the time dimension. Skipping the H steps of the forecast horizon, through
    its dilation.
    Consider a batch of one element, the dilated convolution operation on the
    $t$ time step is defined:

    $\\mathrm{Conv1D}(\\mathbf{x},\\mathbf{w})(t) = (\\mathbf{x}_{[*d]} \\mathbf{w})(t) = \\sum^{K}_{k=1} w_{k} \\mathbf{x}_{t-dk}$

    where $d$ is the dilation factor, $K$ is the kernel size, $t-dk$ is the index of
    the considered past observation. The dilation effectively applies a filter with skip
    connections. If $d=1$ one recovers a normal convolution.

    **Parameters:**<br>
    `in_channels`: int, dimension of `x` input's initial channels.<br>
    `out_channels`: int, dimension of `x` outputs's channels.<br>
    `activation`: str, identifying activations from PyTorch activations.
        select from 'ReLU','Softplus','Tanh','SELU', 'LeakyReLU','PReLU','Sigmoid'.<br>
    `padding`: int, number of zero padding used to the left.<br>
    `kernel_size`: int, convolution's kernel size.<br>
    `dilation`: int, dilation skip connections.<br>

    **Returns:**<br>
    `x`: tensor, torch tensor of dim [N,C_out,T] activation(conv1d(inputs, kernel) + bias). <br>
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, activation, stride: int=1):
        super(CausalConv1d, self).__init__()
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.activation = getattr(nn, activation)()
        self.causalconv = nn.Sequential(self.conv, self.chomp, self.activation)

    def forward(self, x):
        return self.causalconv(x)


class TemporalConvolutionEncoder(nn.Module):
    """Temporal Convolution Encoder

    Receives `x` input of dim [N,T,C_in], permutes it to  [N,C_in,T]
    applies a deep stack of exponentially dilated causal convolutions.
    The exponentially increasing dilations of the convolutions allow for
    the creation of weighted averages of exponentially large long-term memory.

    **Parameters:**<br>
    `in_channels`: int, dimension of `x` input's initial channels.<br>
    `out_channels`: int, dimension of `x` outputs's channels.<br>
    `kernel_size`: int, size of the convolving kernel.<br>
    `dilations`: int list, controls the temporal spacing between the kernel points.<br>
    `activation`: str, identifying activations from PyTorch activations.
        select from 'ReLU','Softplus','Tanh','SELU', 'LeakyReLU','PReLU','Sigmoid'.<br>

    **Returns:**<br>
    `x`: tensor, torch tensor of dim [N,T,C_out].<br>
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilations, activation: str='ReLU'):
        super(TemporalConvolutionEncoder, self).__init__()
        layers = []
        for dilation in dilations:
            layers.append(CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * dilation, activation=activation, dilation=dilation))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.tcn(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class MSE(torch.nn.Module):
    """Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.

    $$ \\mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \\hat{y}_{\\tau})^{2} $$
    """

    def __init__(self):
        super(MSE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mse`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        mse = (y - y_hat) ** 2
        mse = mask * mse
        mse = torch.mean(mse)
        return mse


class RMSE(torch.nn.Module):
    """Root Mean Squared Error

    Calculates Root Mean Squared Error between
    `y` and `y_hat`. RMSE measures the relative prediction
    accuracy of a forecasting method by calculating the squared deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.
    RMSE has a direct connection to the L2 norm.

    $$ \\mathrm{RMSE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) = \\sqrt{\\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \\hat{y}_{\\tau})^{2}} $$
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `rmse`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        mse = (y - y_hat) ** 2
        mse = mask * mse
        mse = torch.mean(mse)
        mse = torch.sqrt(mse)
        return mse


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) ->torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div


class MAPE(torch.nn.Module):
    """Mean Absolute Percentage Error

    Calculates Mean Absolute Percentage Error  between
    `y` and `y_hat`. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty MAPE loss
    assigns to the corresponding error.

    $$ \\mathrm{MAPE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\\hat{y}_{\\tau}|}{|y_{\\tau}|} $$
    """

    def __init__(self):
        super(MAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mape`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        mask = _divide_no_nan(mask, torch.abs(y))
        mape = torch.abs(y - y_hat) * mask
        mape = torch.mean(mape)
        return mape


class SMAPE(torch.nn.Module):
    """Symmetric Mean Absolute Percentage Error

    Calculates Symmetric Mean Absolute Percentage Error between
    `y` and `y_hat`. SMAPE measures the relative prediction
    accuracy of a forecasting method by calculating the relative deviation
    of the prediction and the observed value scaled by the sum of the
    absolute values for the prediction and observed value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined when the target is zero.

    $$ \\mathrm{sMAPE}_{2}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\\hat{y}_{\\tau}|}{|y_{\\tau}|+|\\hat{y}_{\\tau}|} $$

    **References:**<br>
    [Makridakis S., "Accuracy measures: theoretical and practical concerns".](https://www.sciencedirect.com/science/article/pii/0169207093900793)
    """

    def __init__(self):
        super(SMAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `smape`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        delta_y = torch.abs(y - y_hat)
        scale = torch.abs(y) + torch.abs(y_hat)
        smape = _divide_no_nan(delta_y, scale)
        smape = smape * mask
        smape = 2 * torch.mean(smape)
        return smape


class MASE(torch.nn.Module):
    """Mean Absolute Scaled Error
    Calculates the Mean Absolute Scaled Error between
    `y` and `y_hat`. MASE measures the relative prediction
    accuracy of a forecasting method by comparinng the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    $$ \\mathrm{MASE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}, \\mathbf{\\hat{y}}^{season}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\frac{|y_{\\tau}-\\hat{y}_{\\tau}|}{\\mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}^{season}_{\\tau})} $$

    **Parameters:**<br>
    `seasonality`: int. Main frequency of the time series; Hourly 24,  Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.

    **References:**<br>
    [Rob J. Hyndman, & Koehler, A. B. "Another look at measures of forecast accuracy".](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
    [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods".](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
    """

    def __init__(self, seasonality: int):
        super(MASE, self).__init__()
        self.outputsize_multiplier = 1
        self.seasonality = seasonality
        self.output_names = ['']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, y_insample: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor (batch_size, output_size), Actual values.<br>
        `y_hat`: tensor (batch_size, output_size)), Predicted values.<br>
        `y_insample`: tensor (batch_size, input_size), Actual insample Seasonal Naive predictions.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mase`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        delta_y = torch.abs(y - y_hat)
        scale = torch.mean(torch.abs(y_insample[:, self.seasonality:] - y_insample[:, :-self.seasonality]), axis=1)
        mase = _divide_no_nan(delta_y, scale[:, None])
        mase = mase * mask
        mase = torch.mean(mase)
        return mase


class QuantileLoss(torch.nn.Module):
    """Quantile Loss

    Computes the quantile loss between `y` and `y_hat`.
    QL measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for q is 0.5 for the deviation from the median (Pinball loss).

    $$ \\mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}^{(q)}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} \\Big( (1-q)\\,( \\hat{y}^{(q)}_{\\tau} - y_{\\tau} )_{+} + q\\,( y_{\\tau} - \\hat{y}^{(q)}_{\\tau} )_{+} \\Big) $$

    **Parameters:**<br>
    `q`: float, between 0 and 1. The slope of the quantile loss, in the context of quantile regression, the q determines the conditional quantile level.<br>

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)
    """

    def __init__(self, q):
        super(QuantileLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.q = q
        self.output_names = [f'_ql{q}']
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `quantile_loss`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        delta_y = y - y_hat
        loss = torch.max(torch.mul(self.q, delta_y), torch.mul(self.q - 1, delta_y))
        loss = loss * mask
        quantile_loss = torch.mean(loss)
        return quantile_loss


def level_to_outputs(level):
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f'-lo-{l}', f'-hi-{l}'] for l in level], [])
    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]
    quantiles = np.concatenate([np.array([50]), quantiles])
    quantiles = torch.Tensor(quantiles) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, '-median')
    return quantiles, output_names


def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.5:
            output_names.append(f'-lo-{np.round(100 - 200 * q, 2)}')
        elif q > 0.5:
            output_names.append(f'-hi-{np.round(100 - 200 * (1 - q), 2)}')
        else:
            output_names.append('-median')
    return quantiles, output_names


class MQLoss(torch.nn.Module):
    """Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    $$ \\mathrm{MQL}(\\mathbf{y}_{\\tau},[\\mathbf{\\hat{y}}^{(q_{1})}_{\\tau}, ... ,\\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \\mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}^{(q_{i})}_{\\tau}) $$

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\\mathbf{\\hat{F}}_{\\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    $$ \\mathrm{CRPS}(y_{\\tau}, \\mathbf{\\hat{F}}_{\\tau}) = \\int^{1}_{0} \\mathrm{QL}(y_{\\tau}, \\hat{y}^{(q)}_{\\tau}) dq $$

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """

    def __init__(self, level=[80, 90], quantiles=None):
        super(MQLoss, self).__init__()
        if level:
            qs, self.output_names = level_to_outputs(level)
            quantiles = torch.Tensor(qs)
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            quantiles = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(quantiles, requires_grad=False)
        self.outputsize_multiplier = len(self.quantiles)
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Identity domain map [B,T,H,Q]/[B,H,Q]
        """
        return y_hat

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mqloss`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        n_q = len(self.quantiles)
        error = y_hat - y.unsqueeze(-1)
        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))
        mqloss = self.quantiles * sq + (1 - self.quantiles) * s1_q
        mask = mask / torch.sum(mask)
        mask = mask.unsqueeze(-1)
        mqloss = 1 / n_q * mqloss * mask
        return torch.sum(mqloss)


class wMQLoss(torch.nn.Module):
    """Weighted Multi-Quantile loss

    Calculates the Weighted Multi-Quantile loss (WMQL) between `y` and `y_hat`.
    WMQL calculates the weighted average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    $$ \\mathrm{wMQL}(\\mathbf{y}_{\\tau},[\\mathbf{\\hat{y}}^{(q_{1})}_{\\tau}, ... ,\\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \\frac{\\mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}^{(q_{i})}_{\\tau})}{\\sum^{t+H}_{\\tau=t+1} |y_{\\tau}|} $$

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals (Defaults median).
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """

    def __init__(self, level=[80, 90], quantiles=None):
        super(wMQLoss, self).__init__()
        if level:
            qs, self.output_names = level_to_outputs(level)
            quantiles = torch.Tensor(qs)
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            quantiles = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(quantiles, requires_grad=False)
        self.outputsize_multiplier = len(self.quantiles)
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Identity domain map [B,T,H,Q]/[B,H,Q]
        """
        return y_hat

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `mqloss`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)
        error = y_hat - y.unsqueeze(-1)
        sq = torch.maximum(-error, torch.zeros_like(error))
        s1_q = torch.maximum(error, torch.zeros_like(error))
        loss = self.quantiles * sq + (1 - self.quantiles) * s1_q
        mask = mask.unsqueeze(-1)
        wmqloss = _divide_no_nan(torch.sum(loss * mask, axis=-2), torch.sum(torch.abs(y.unsqueeze(-1)) * mask, axis=-2))
        return torch.mean(wmqloss)


def normal_domain_map(input: torch.Tensor, eps: float=0.1):
    """
    Maps input into distribution constraints, by construction input's
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>
    `eps`: float, helps the initialization of scale for easier optimization.<br>

    **Returns:**<br>
    `(loc, scale)`: tuple with tensors of Normal distribution arguments.<br>
    """
    loc, scale = torch.tensor_split(input, 2, dim=-1)
    scale = F.softplus(scale) + eps
    return loc.squeeze(-1), scale.squeeze(-1)


def poisson_domain_map(input: torch.Tensor):
    """
    Maps input into distribution constraints, by construction input's
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>

    **Returns:**<br>
    `(loc,)`: tuple with tensors of Poisson distribution arguments.<br>
    """
    rate_pos = F.softplus(input).clone()
    return rate_pos.squeeze(-1),


def student_domain_map(input: torch.Tensor, eps: float=0.1):
    """
    Maps input into distribution constraints, by construction input's
    last dimension is of matching `distr_args` length.

    **Parameters:**<br>
    `input`: tensor, of dimensions [B,T,H,theta] or [B,H,theta].<br>
    `eps`: float, helps the initialization of scale for easier optimization.<br>

    **Returns:**<br>
    `(df, loc, scale)`: tuple with tensors of StudentT distribution arguments.<br>
    """
    df, loc, scale = torch.tensor_split(input, 3, dim=-1)
    scale = F.softplus(scale) + eps
    df = 2.0 + F.softplus(df)
    return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


def weighted_average(x: torch.Tensor, weights: Optional[torch.Tensor]=None, dim=None) ->torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    **Parameters:**<br>
    `x`: Input tensor, of which the average must be computed.<br>
    `weights`: Weights tensor, of the same shape as `x`.<br>
    `dim`: The dim along which to average `x`.<br>

    **Returns:**<br>
    `Tensor`: The tensor with values averaged along the specified `dim`.<br>
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return x.mean(dim=dim)


class DistributionLoss(torch.nn.Module):
    """DistributionLoss

    This PyTorch module wraps the `torch.distribution` classes allowing it to
    interact with NeuralForecast models modularly. It shares the negative
    log-likelihood as the optimization objective and a sample method to
    generate empirically the quantiles defined by the `level` list.

    Additionally, it implements a distribution transformation that factorizes the
    scale-dependent likelihood parameters into a base scale and a multiplier
    efficiently learnable within the network's non-linearities operating ranges.

    Available distributions:
    - Poisson
    - Normal
    - StudentT

    **Parameters:**<br>
    `distribution`: str, identifier of a torch.distributions.Distribution class.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `num_samples`: int=500, number of samples for the empirical quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br><br>

    **References:**<br>
    - [PyTorch Probability Distributions Package: StudentT.](https://pytorch.org/docs/stable/distributions.html#studentt)<br>
    - [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020).
       "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207019301888)<br>
    """

    def __init__(self, distribution, level=[80, 90], quantiles=None, num_samples=500, return_params=False):
        super(DistributionLoss, self).__init__()
        available_distributions = dict(Normal=Normal, Poisson=Poisson, StudentT=StudentT)
        domain_maps = dict(Normal=normal_domain_map, Poisson=poisson_domain_map, StudentT=student_domain_map)
        param_names = dict(Normal=['-loc', '-scale'], Poisson=['-loc'], StudentT=['-df', '-loc', '-scale'])
        assert distribution in available_distributions.keys(), f'{distribution} not available'
        self._base_distribution = available_distributions[distribution]
        self.domain_map = domain_maps[distribution]
        self.param_names = param_names[distribution]
        if level:
            qs, self.output_names = level_to_outputs(level)
            quantiles = torch.Tensor(qs)
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            quantiles = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(quantiles, requires_grad=False)
        self.num_samples = num_samples
        self.return_params = return_params
        if self.return_params:
            self.output_names = self.output_names + self.param_names
        self.outputsize_multiplier = len(self._base_distribution.arg_constraints.keys())
        self.is_distribution_output = True

    def get_distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None) ->Distribution:
        """
        Construct the associated Pytorch Distribution, given the collection of
        constructor arguments and, optionally, location and scale tensors.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution.<br>

        **Returns**<br>
        `Distribution`: AffineTransformed distribution.<br>
        """
        distr = self._base_distribution(*distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return TransformedDistribution(distr, [AffineTransform(loc=loc, scale=scale)])

    def sample(self, distr_args: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, num_samples: Optional[int]=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution.<br>
        `num_samples`: int=500, overwrite number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples
        B, H = distr_args[0].size()
        Q = len(self.quantiles)
        distr = self.get_distribution(distr_args=distr_args, loc=loc, scale=scale)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(1, 2, 0)
        samples = samples
        samples = samples.view(B * H, num_samples)
        quantiles_device = self.quantiles
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1, 0))
        samples = samples.view(B, H, num_samples)
        quants = quants.view(B, H, Q)
        return samples, quants

    def __call__(self, y: torch.Tensor, distr_args: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        """
        Computes the negative log-likelihood objective function.
        To estimate the following predictive distribution:

        $$\\mathrm{P}(\\mathbf{y}_{\\tau}\\,|\\,\\theta) \\quad \\mathrm{and} \\quad -\\log(\\mathrm{P}(\\mathbf{y}_{\\tau}\\,|\\,\\theta))$$

        where $\\theta$ represents the distributions parameters. It aditionally
        summarizes the objective signal using a weighted average using the `mask` tensor.

        **Parameters**<br>
        `y`: tensor, Actual values.<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns**<br>
        `loss`: scalar, weighted loss function against which backpropagation will be performed.<br>
        """
        distr = self.get_distribution(distr_args=distr_args, loc=loc, scale=scale)
        loss_values = -distr.log_prob(y)
        loss_weights = mask
        return weighted_average(loss_values, weights=loss_weights)


class PMM(torch.nn.Module):
    """Poisson Mixture Mesh

    This Poisson Mixture statistical model assumes independence across groups of
    data $\\mathcal{G}=\\{[g_{i}]\\}$, and estimates relationships within the group.

    $$ \\mathrm{P}\\left(\\mathbf{y}_{[b][t+1:t+H]}\\right) =
    \\prod_{ [g_{i}] \\in \\mathcal{G}} \\mathrm{P} \\left(\\mathbf{y}_{[g_{i}][\\tau]} \\right) =
    \\prod_{\\beta\\in[g_{i}]}
    \\left(\\sum_{k=1}^{K} w_k \\prod_{(\\beta,\\tau) \\in [g_i][t+1:t+H]} \\mathrm{Poisson}(y_{\\beta,\\tau}, \\hat{\\lambda}_{\\beta,\\tau,k}) \\right)$$

    **Parameters:**<br>
    `n_components`: int=10, the number of mixture components.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br><br>

    **References:**<br>
    [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker.
    Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International
    Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(self, n_components=10, level=[80, 90], quantiles=None, num_samples=500, return_params=False):
        super(PMM, self).__init__()
        if level:
            qs, self.output_names = level_to_outputs(level)
            quantiles = torch.Tensor(qs)
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            quantiles = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(quantiles, requires_grad=False)
        self.num_samples = num_samples
        self.return_params = return_params
        if self.return_params:
            self.param_names = [f'-lambda-{i}' for i in range(1, n_components + 1)]
            self.output_names = self.output_names + self.param_names
        self.outputsize_multiplier = n_components
        self.is_distribution_output = True

    def get_distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None):
        if loc is None and scale is None:
            return distr_args
        elif loc is not None and scale is not None:
            loc = loc.view(distr_args[0].size(dim=0), 1, -1)
            scale = scale.view(distr_args[0].size(dim=0), 1, -1)
            lambda_scaled = distr_args[0] * scale + loc
            return lambda_scaled,

    def domain_map(self, lambdas_hat: torch.Tensor):
        lambdas_hat = F.softplus(lambdas_hat)
        return lambdas_hat,

    def sample(self, distr_args, loc=None, scale=None, num_samples=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution.<br>
        `num_samples`: int=500, overwrites number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples
        lambdas = self.get_distribution(distr_args, loc, scale)[0]
        B, H, K = lambdas.size()
        Q = len(self.quantiles)
        weights = 1 / K * torch.ones_like(lambdas)
        weights = weights.reshape(-1, K)
        lambdas = lambdas.flatten()
        sample_idxs = torch.multinomial(input=weights, num_samples=num_samples, replacement=True)
        aux_col_idx = torch.unsqueeze(torch.arange(B * H), -1) * K
        sample_idxs = sample_idxs
        aux_col_idx = aux_col_idx
        sample_idxs = sample_idxs + aux_col_idx
        sample_idxs = sample_idxs.flatten()
        sample_lambdas = lambdas[sample_idxs]
        samples = torch.poisson(sample_lambdas)
        samples = samples.view(B * H, num_samples)
        quantiles_device = self.quantiles
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1, 0))
        samples = samples.view(B, H, num_samples)
        quants = quants.view(B, H, Q)
        return samples, quants

    def neglog_likelihood(self, y: torch.Tensor, distr_args: Tuple[torch.Tensor], mask: Union[torch.Tensor, None]=None, loc: Union[torch.Tensor, None]=None, scale: Union[torch.Tensor, None]=None):
        if mask is None:
            mask = torch.ones_like(y)
        eps = 1e-10
        lambdas = self.get_distribution(distr_args, loc, scale)[0]
        B, H, K = lambdas.size()
        weights = 1 / K * torch.ones_like(lambdas)
        y = y[:, :, None]
        mask = mask[:, :, None]
        log = y * torch.log(lambdas + eps) - lambdas - (y * torch.log(y + eps) - y)
        log_max = torch.amax(log, dim=2, keepdim=True)
        lik = weights * torch.exp(log - log_max)
        loglik = torch.log(torch.sum(lik, dim=2, keepdim=True)) + log_max
        loglik = loglik * mask
        loss = -torch.mean(loglik)
        return loss

    def __call__(self, y: torch.Tensor, distr_args: Tuple[torch.Tensor], mask: Union[torch.Tensor, None]=None, loc: Union[torch.Tensor, None]=None, scale: Union[torch.Tensor, None]=None):
        return self.neglog_likelihood(y=y, distr_args=distr_args, mask=mask, loc=loc, scale=scale)


class GMM(torch.nn.Module):
    """Gaussian Mixture Mesh

    This Gaussian Mixture statistical model assumes independence across groups of
    data $\\mathcal{G}=\\{[g_{i}]\\}$, and estimates relationships within the group.

    $$ \\mathrm{P}\\left(\\mathbf{y}_{[b][t+1:t+H]}\\right) =
    \\prod_{ [g_{i}] \\in \\mathcal{G}} \\mathrm{P}\\left(\\mathbf{y}_{[g_{i}][\\tau]}\\right)=
    \\prod_{\\beta\\in[g_{i}]}
    \\left(\\sum_{k=1}^{K} w_k \\prod_{(\\beta,\\tau) \\in [g_i][t+1:t+H]}
    \\mathrm{Gaussian}(y_{\\beta,\\tau}, \\hat{\\mu}_{\\beta,\\tau,k}, \\sigma_{\\beta,\\tau,k})\\right)$$

    **Parameters:**<br>
    `n_components`: int=10, the number of mixture components.<br>
    `level`: float list [0,100], confidence levels for prediction intervals.<br>
    `quantiles`: float list [0,1], alternative to level list, target quantiles.<br>
    `return_params`: bool=False, wether or not return the Distribution parameters.<br><br>

    **References:**<br>
    [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker.
    Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. Submitted to the International
    Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """

    def __init__(self, n_components=1, level=[80, 90], quantiles=None, num_samples=500, return_params=False):
        super(GMM, self).__init__()
        if level:
            qs, self.output_names = level_to_outputs(level)
            quantiles = torch.Tensor(qs)
        if quantiles is not None:
            _, self.output_names = quantiles_to_outputs(quantiles)
            quantiles = torch.Tensor(quantiles)
        self.quantiles = torch.nn.Parameter(quantiles, requires_grad=False)
        self.num_samples = num_samples
        self.return_params = return_params
        if self.return_params:
            mu_names = [f'-mu-{i}' for i in range(1, n_components + 1)]
            std_names = [f'-std-{i}' for i in range(1, n_components + 1)]
            self.output_names = self.output_names + mu_names + std_names
        self.outputsize_multiplier = 2 * n_components
        self.is_distribution_output = True

    def get_distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None):
        if loc is None and scale is None:
            return distr_args
        elif loc is not None and scale is not None:
            loc = loc.view(distr_args[0].size(dim=0), 1, -1)
            scale = scale.view(distr_args[1].size(dim=0), 1, -1)
            mu_scaled = distr_args[0] * scale + loc
            std_scaled = distr_args[1] * scale
            return mu_scaled, std_scaled

    def domain_map(self, params_hat: torch.Tensor, eps: float=0.2):
        loc, scale = torch.tensor_split(params_hat, 2, dim=-1)
        scale = F.softplus(scale) + eps
        return loc, scale

    def sample(self, distr_args, loc=None, scale=None, num_samples=None):
        """
        Construct the empirical quantiles from the estimated Distribution,
        sampling from it `num_samples` independently.

        **Parameters**<br>
        `distr_args`: Constructor arguments for the underlying Distribution type.<br>
        `loc`: Optional tensor, of the same shape as the batch_shape + event_shape
               of the resulting distribution.<br>
        `scale`: Optional tensor, of the same shape as the batch_shape+event_shape
               of the resulting distribution.<br>
        `num_samples`: int=500, number of samples for the empirical quantiles.<br>

        **Returns**<br>
        `samples`: tensor, shape [B,H,`num_samples`].<br>
        `quantiles`: tensor, empirical quantiles defined by `levels`.<br>
        """
        if num_samples is None:
            num_samples = self.num_samples
        means, stds = self.get_distribution(distr_args, loc, scale)
        B, H, K = means.size()
        Q = len(self.quantiles)
        assert means.shape == stds.shape
        weights = 1 / K * torch.ones_like(means)
        weights = weights.reshape(-1, K)
        means = means.flatten()
        stds = stds.flatten()
        sample_idxs = torch.multinomial(input=weights, num_samples=num_samples, replacement=True)
        aux_col_idx = torch.unsqueeze(torch.arange(B * H), -1) * K
        sample_idxs = sample_idxs
        aux_col_idx = aux_col_idx
        sample_idxs = sample_idxs + aux_col_idx
        sample_idxs = sample_idxs.flatten()
        sample_means = means[sample_idxs]
        sample_stds = stds[sample_idxs]
        samples = torch.normal(sample_means, sample_stds)
        samples = samples.view(B * H, num_samples)
        quantiles_device = self.quantiles
        quants = torch.quantile(input=samples, q=quantiles_device, dim=1)
        quants = quants.permute((1, 0))
        samples = samples.view(B, H, num_samples)
        quants = quants.view(B, H, Q)
        return samples, quants

    def neglog_likelihood(self, y: torch.Tensor, distr_args: Tuple[torch.Tensor, torch.Tensor], mask: Union[torch.Tensor, None]=None, loc: Union[torch.Tensor, None]=None, scale: Union[torch.Tensor, None]=None):
        if mask is None:
            mask = torch.ones_like(y)
        means, stds = self.get_distribution(distr_args, loc, scale)
        B, H, K = means.size()
        weights = 1 / K * torch.ones_like(means)
        y = y[:, :, None]
        mask = mask[:, :, None]
        log = -0.5 * (1 / stds * (y - means)) ** 2 - torch.log((2 * math.pi) ** 0.5 * stds)
        log_max = torch.amax(log, dim=2, keepdim=True)
        lik = weights * torch.exp(log - log_max)
        loglik = torch.log(torch.sum(lik, dim=2, keepdim=True)) + log_max
        loglik = loglik * mask
        loss = -torch.mean(loglik)
        return loss

    def __call__(self, y: torch.Tensor, distr_args: Tuple[torch.Tensor, torch.Tensor], mask: Union[torch.Tensor, None]=None, loc: Union[torch.Tensor, None]=None, scale: Union[torch.Tensor, None]=None):
        return self.neglog_likelihood(y=y, distr_args=distr_args, mask=mask, loc=loc, scale=scale)


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
        gates = torch.matmul(inputs, self.weight_ih.t()) + self.bias_ih + torch.matmul(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, (hy, cy)


class ResLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
        ifo_gates = torch.matmul(inputs, self.weight_ii.t()) + self.bias_ii + torch.matmul(hx, self.weight_ih.t()) + self.bias_ih + torch.matmul(cx, self.weight_ic.t()) + self.bias_ic
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)
        cellgate = torch.matmul(hx, self.weight_hh.t()) + self.bias_hh
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        ry = torch.tanh(cy)
        if self.input_size == self.hidden_size:
            hy = outgate * (ry + inputs)
        else:
            hy = outgate * (ry + torch.matmul(inputs, self.weight_ir.t()))
        return hy, (hy, cy)


class ResLSTMLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.0)

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, hidden = self.cell(inputs[i], hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden


class AttentiveLSTMLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(AttentiveLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        attention_hsize = hidden_size
        self.attention_hsize = attention_hsize
        self.cell = LSTMCell(input_size, hidden_size)
        self.attn_layer = nn.Sequential(nn.Linear(2 * hidden_size + input_size, attention_hsize), nn.Tanh(), nn.Linear(attention_hsize, 1))
        self.softmax = nn.Softmax(dim=0)
        self.dropout = dropout

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []
        for t in range(len(inputs)):
            hx, cx = (tensor.squeeze(0) for tensor in hidden)
            hx_rep = hx.repeat(len(inputs), 1, 1)
            cx_rep = cx.repeat(len(inputs), 1, 1)
            x = torch.cat((inputs, hx_rep, cx_rep), dim=-1)
            l = self.attn_layer(x)
            beta = self.softmax(l)
            context = torch.bmm(beta.permute(1, 2, 0), inputs.permute(1, 0, 2)).squeeze(1)
            out, hidden = self.cell(context, hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=True):
        super(DRNN, self).__init__()
        self.dilations = dilations
        self.cell_type = cell_type
        self.batch_first = batch_first
        layers = []
        if self.cell_type == 'GRU':
            cell = nn.GRU
        elif self.cell_type == 'RNN':
            cell = nn.RNN
        elif self.cell_type == 'LSTM':
            cell = nn.LSTM
        elif self.cell_type == 'ResLSTM':
            cell = ResLSTMLayer
        elif self.cell_type == 'AttentiveLSTM':
            cell = AttentiveLSTMLayer
        else:
            raise NotImplementedError
        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
            outputs.append(inputs[-dilation:])
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            hidden = torch.zeros(batch_size * rate, hidden_size, dtype=dilated_inputs.dtype, device=dilated_inputs.device)
            hidden = hidden.unsqueeze(0)
            if self.cell_type in ['LSTM', 'ResLSTM', 'AttentiveLSTM']:
                hidden = hidden, hidden
        dilated_outputs, hidden = cell(dilated_inputs, hidden)
        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate
        blocks = [dilated_outputs[:, i * batchsize:(i + 1) * batchsize, :] for i in range(rate)]
        interleaved = torch.stack(blocks).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate, batchsize, dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = n_steps % rate == 0
        if not iseven:
            dilated_steps = n_steps // rate + 1
            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0), inputs.size(1), inputs.size(2), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate
        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs


class IdentityBasis(nn.Module):

    def __init__(self, backcast_size: int, forecast_size: int, out_features: int=1):
        super().__init__()
        self.out_features = out_features
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]
        forecast = forecast.reshape(len(forecast), -1, self.out_features)
        return backcast, forecast


class TrendBasis(nn.Module):

    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int, out_features: int=1):
        super().__init__()
        self.out_features = out_features
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :] for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :] for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, polynomial_size:]
        forecast_theta = forecast_theta.reshape(len(forecast_theta), polynomial_size, -1)
        backcast = torch.einsum('bp,pt->bt', backcast_theta, self.backcast_basis)
        forecast = torch.einsum('bpq,pt->btq', forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int, out_features: int=1):
        super().__init__()
        self.out_features = out_features
        frequency = np.append(np.zeros(1, dtype=float), np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (np.arange(backcast_size, dtype=float)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (np.arange(forecast_size, dtype=float)[:, None] / forecast_size) * frequency
        backcast_cos_template = torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32)
        backcast_sin_template = torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32)
        backcast_template = torch.cat([backcast_cos_template, backcast_sin_template], dim=0)
        forecast_cos_template = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
        forecast_sin_template = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
        forecast_template = torch.cat([forecast_cos_template, forecast_sin_template], dim=0)
        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, harmonic_size:]
        forecast_theta = forecast_theta.reshape(len(forecast_theta), harmonic_size, -1)
        backcast = torch.einsum('bp,pt->bt', backcast_theta, self.backcast_basis)
        forecast = torch.einsum('bpq,pt->btq', forecast_theta, self.forecast_basis)
        return backcast, forecast


class NBEATSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self, input_size: int, h: int, futr_input_size: int, hist_input_size: int, stat_input_size: int, n_theta: int, mlp_units: list, basis: nn.Module, dropout_prob: float, activation: str):
        """ """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()
        input_size = input_size + hist_input_size * input_size + futr_input_size * (input_size + h) + stat_input_size
        hidden_layers = [nn.Linear(in_features=input_size, out_features=mlp_units[0][0])]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)
            if self.dropout_prob > 0:
                raise NotImplementedError('dropout')
        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, futr_exog: torch.Tensor, hist_exog: torch.Tensor, stat_exog: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            insample_y = torch.cat((insample_y, hist_exog.reshape(batch_size, -1)), dim=1)
        if self.futr_input_size > 0:
            insample_y = torch.cat((insample_y, futr_exog.reshape(batch_size, -1)), dim=1)
        if self.stat_input_size > 0:
            insample_y = torch.cat((insample_y, stat_exog.reshape(batch_size, -1)), dim=1)
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class _IdentityBasis(nn.Module):

    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str, out_features: int=1):
        super().__init__()
        assert interpolation_mode in ['linear', 'nearest'] or 'cubic' in interpolation_mode
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ['nearest', 'linear']:
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
        elif 'cubic' in self.interpolation_mode:
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros((len(knots), self.forecast_size))
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i * batch_size:(i + 1) * batch_size], size=self.forecast_size, mode='bicubic')
                forecast[i * batch_size:(i + 1) * batch_size] += forecast_i[:, 0, :, :]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


POOLING = ['MaxPool1d', 'AvgPool1d']


class NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(self, input_size: int, h: int, n_theta: int, mlp_units: list, basis: nn.Module, futr_input_size: int, hist_input_size: int, stat_input_size: int, n_pool_kernel_size: int, pooling_mode: str, dropout_prob: float, activation: str):
        super().__init__()
        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))
        input_size = pooled_hist_size + hist_input_size * pooled_hist_size + futr_input_size * pooled_futr_size + stat_input_size
        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        assert pooling_mode in POOLING, f'{pooling_mode} is not in {POOLING}'
        activ = getattr(nn, activation)()
        self.pooling_layer = getattr(nn, pooling_mode)(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        hidden_layers = [nn.Linear(in_features=input_size, out_features=mlp_units[0][0])]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)
            if self.dropout_prob > 0:
                raise NotImplementedError('dropout')
        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, futr_exog: torch.Tensor, hist_exog: torch.Tensor, stat_exog: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)
            insample_y = torch.cat((insample_y, hist_exog.reshape(batch_size, -1)), dim=1)
        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)
            insample_y = torch.cat((insample_y, futr_exog.reshape(batch_size, -1)), dim=1)
        if self.stat_input_size > 0:
            insample_y = torch.cat((insample_y, stat_exog.reshape(batch_size, -1)), dim=1)
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class MaybeLayerNorm(nn.Module):

    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) ->Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class GRN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=None, context_hidden_size=None, dropout=0):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=0.001)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor]=None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x


class TFTEmbedding(nn.Module):

    def __init__(self, hidden_size, stat_input_size, futr_input_size, hist_input_size, tgt_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.stat_input_size = stat_input_size
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.tgt_size = tgt_size
        for attr, size in [('stat_exog_embedding', stat_input_size), ('futr_exog_embedding', futr_input_size), ('hist_exog_embedding', hist_input_size), ('tgt_embedding', tgt_size)]:
            if size:
                vectors = nn.Parameter(torch.Tensor(size, hidden_size))
                bias = nn.Parameter(torch.zeros(size, hidden_size))
                torch.nn.init.xavier_normal_(vectors)
                setattr(self, attr + '_vectors', vectors)
                setattr(self, attr + '_bias', bias)
            else:
                setattr(self, attr + '_vectors', None)
                setattr(self, attr + '_bias', None)

    def _apply_embedding(self, cont: Optional[Tensor], cont_emb: Tensor, cont_bias: Tensor):
        if cont is not None:
            e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
            e_cont = e_cont + cont_bias
            return e_cont
        return None

    def forward(self, target_inp, stat_exog=None, futr_exog=None, hist_exog=None):
        stat_exog = stat_exog[:, :] if stat_exog is not None else None
        s_inp = self._apply_embedding(cont=stat_exog, cont_emb=self.stat_exog_embedding_vectors, cont_bias=self.stat_exog_embedding_bias)
        k_inp = self._apply_embedding(cont=futr_exog, cont_emb=self.futr_exog_embedding_vectors, cont_bias=self.futr_exog_embedding_bias)
        o_inp = self._apply_embedding(cont=hist_exog, cont_emb=self.hist_exog_embedding_vectors, cont_bias=self.hist_exog_embedding_bias)
        target_inp = torch.matmul(target_inp.unsqueeze(3).unsqueeze(4), self.tgt_embedding_vectors.unsqueeze(1)).squeeze(3)
        target_inp = target_inp + self.tgt_embedding_bias
        return s_inp, k_inp, o_inp, target_inp


class VariableSelectionNetwork(nn.Module):

    def __init__(self, hidden_size, num_inputs, dropout):
        super().__init__()
        self.joint_grn = GRN(input_size=hidden_size * num_inputs, hidden_size=hidden_size, output_size=num_inputs, context_hidden_size=hidden_size)
        self.var_grns = nn.ModuleList([GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout) for _ in range(num_inputs)])

    def forward(self, x: Tensor, context: Optional[Tensor]=None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)
        return variable_ctx, sparse_weights


class InterpretableMultiHeadAttention(nn.Module):

    def __init__(self, n_head, hidden_size, example_length, attn_dropout, dropout):
        super().__init__()
        self.n_head = n_head
        assert hidden_size % n_head == 0
        self.d_head = hidden_size // n_head
        self.qkv_linears = nn.Linear(hidden_size, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        self.register_buffer('_mask', torch.triu(torch.full((example_length, example_length), float('-inf')), 1).unsqueeze(0))

    def forward(self, x: Tensor, mask_future_timesteps: bool=True) ->Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)
        if mask_future_timesteps:
            attn_score = attn_score + self._mask
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)
        return out, attn_vec


class StaticCovariateEncoder(nn.Module):

    def __init__(self, hidden_size, num_static_vars, dropout):
        super().__init__()
        self.vsn = VariableSelectionNetwork(hidden_size=hidden_size, num_inputs=num_static_vars, dropout=dropout)
        self.context_grns = nn.ModuleList([GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout) for _ in range(4)])

    def forward(self, x: Tensor) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)
        cs, ce, ch, cc = tuple(m(variable_ctx) for m in self.context_grns)
        return cs, ce, ch, cc


class TemporalCovariateEncoder(nn.Module):

    def __init__(self, hidden_size, num_historic_vars, num_future_vars, dropout):
        super(TemporalCovariateEncoder, self).__init__()
        self.history_vsn = VariableSelectionNetwork(hidden_size=hidden_size, num_inputs=num_historic_vars, dropout=dropout)
        self.history_encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(hidden_size=hidden_size, num_inputs=num_future_vars, dropout=dropout)
        self.future_encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.input_gate = GLU(hidden_size, hidden_size)
        self.input_gate_ln = LayerNorm(hidden_size, eps=0.001)

    def forward(self, historical_inputs, future_inputs, cs, ch, cc):
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)
        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)
        return temporal_features


class TemporalFusionDecoder(nn.Module):

    def __init__(self, n_head, hidden_size, example_length, encoder_length, attn_dropout, dropout):
        super(TemporalFusionDecoder, self).__init__()
        self.encoder_length = encoder_length
        self.enrichment_grn = GRN(input_size=hidden_size, hidden_size=hidden_size, context_hidden_size=hidden_size, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(n_head=n_head, hidden_size=hidden_size, example_length=example_length, attn_dropout=attn_dropout, dropout=dropout)
        self.attention_gate = GLU(hidden_size, hidden_size)
        self.attention_ln = LayerNorm(normalized_shape=hidden_size, eps=0.001)
        self.positionwise_grn = GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        self.decoder_gate = GLU(hidden_size, hidden_size)
        self.decoder_ln = LayerNorm(normalized_shape=hidden_size, eps=0.001)

    def forward(self, temporal_features, ce):
        enriched = self.enrichment_grn(temporal_features, c=ce)
        x, _ = self.attention(enriched, mask_future_timesteps=True)
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]
        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)
        x = self.positionwise_grn(x)
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1, 'activation': 'ReLU'}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Chomp1d,
     lambda: ([], {'horizon': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GLU,
     lambda: ([], {'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IdentityBasis,
     lambda: ([], {'backcast_size': 4, 'forecast_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InterpretableMultiHeadAttention,
     lambda: ([], {'n_head': 4, 'hidden_size': 4, 'example_length': 4, 'attn_dropout': 0.5, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MAE,
     lambda: ([], {}),
     lambda: ([], {'y': torch.rand([4, 4]), 'y_hat': 4}),
     True),
    (MAPE,
     lambda: ([], {}),
     lambda: ([], {'y': torch.rand([4, 4]), 'y_hat': torch.rand([4, 4])}),
     True),
    (MSE,
     lambda: ([], {}),
     lambda: ([], {'y': 4, 'y_hat': torch.rand([4, 4])}),
     True),
    (MaybeLayerNorm,
     lambda: ([], {'output_size': 4, 'hidden_size': 4, 'eps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuantileLoss,
     lambda: ([], {'q': 4}),
     lambda: ([], {'y': 4, 'y_hat': torch.rand([4, 4])}),
     True),
    (RMSE,
     lambda: ([], {}),
     lambda: ([], {'y': 4, 'y_hat': torch.rand([4, 4])}),
     True),
    (ResLSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (ResLSTMLayer,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (SMAPE,
     lambda: ([], {}),
     lambda: ([], {'y': torch.rand([4, 4]), 'y_hat': torch.rand([4, 4])}),
     True),
    (StaticCovariateEncoder,
     lambda: ([], {'hidden_size': 4, 'num_static_vars': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TFTEmbedding,
     lambda: ([], {'hidden_size': 4, 'stat_input_size': 4, 'futr_input_size': 4, 'hist_input_size': 4, 'tgt_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (TemporalConvolutionEncoder,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (VariableSelectionNetwork,
     lambda: ([], {'hidden_size': 4, 'num_inputs': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Nixtla_neuralforecast(_paritybench_base):
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

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

