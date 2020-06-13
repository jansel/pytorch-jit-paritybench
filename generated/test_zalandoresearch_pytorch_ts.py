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

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import functools


import inspect


from collections import OrderedDict


from typing import Any


import torch


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


from typing import Callable


from typing import Iterator


import torch.nn.functional as F


from abc import abstractclassmethod


from typing import Dict


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


from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils import clip_grad_norm_


from torch.optim import SGD


from torch.utils.data import TensorDataset


from itertools import chain


from itertools import combinations


class NBEATSBlock(nn.Module):

    def __init__(self, units, thetas_dim, num_block_layers=4,
        backcast_length=10, forecast_length=5, share_thetas=False):
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
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim,
                bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


def linspace(backcast_length: int, forecast_length: int) ->Tuple[np.ndarray,
    np.ndarray]:
    lin_space = np.linspace(-backcast_length, forecast_length, 
        backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSTrendBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4,
        backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(NBEATSTrendBlock, self).__init__(units=units, thetas_dim=
            thetas_dim, num_block_layers=num_block_layers, backcast_length=
            backcast_length, forecast_length=forecast_length, share_thetas=True
            )
        backcast_linspace, forecast_linspace = linspace(backcast_length,
            forecast_length)
        self.register_buffer('T_backcast', torch.tensor([(backcast_linspace **
            i) for i in range(thetas_dim)]).float())
        self.register_buffer('T_forecast', torch.tensor([(forecast_linspace **
            i) for i in range(thetas_dim)]).float())

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSSeasonalBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim=None, num_block_layers=4,
        backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        super(NBEATSSeasonalBlock, self).__init__(units=units, thetas_dim=
            thetas_dim, num_block_layers=num_block_layers, backcast_length=
            backcast_length, forecast_length=forecast_length, share_thetas=True
            )
        backcast_linspace, forecast_linspace = linspace(backcast_length,
            forecast_length)
        p1, p2 = (thetas_dim // 2, thetas_dim // 2
            ) if thetas_dim % 2 == 0 else (thetas_dim // 2, thetas_dim // 2 + 1
            )
        s1_b = torch.tensor([np.cos(2 * np.pi * i * backcast_linspace) for
            i in range(p1)]).float()
        s2_b = torch.tensor([np.sin(2 * np.pi * i * backcast_linspace) for
            i in range(p2)]).float()
        self.register_buffer('S_backcast', torch.cat([s1_b, s2_b]))
        s1_f = torch.tensor([np.cos(2 * np.pi * i * forecast_linspace) for
            i in range(p1)]).float()
        s2_f = torch.tensor([np.sin(2 * np.pi * i * forecast_linspace) for
            i in range(p2)]).float()
        self.register_buffer('S_forecast', torch.cat([s1_f, s2_f]))

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.S_backcast)
        forecast = self.theta_f_fc(x).mm(self.S_forecast)
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):

    def __init__(self, units, thetas_dim, num_block_layers=4,
        backcast_length=10, forecast_length=5):
        super(NBEATSGenericBlock, self).__init__(units=units, thetas_dim=
            thetas_dim, num_block_layers=num_block_layers, backcast_length=
            backcast_length, forecast_length=forecast_length)
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class NBEATSNetwork(nn.Module):

    def __init__(self, prediction_length: int, context_length: int,
        num_stacks: int, widths: List[int], num_blocks: List[int],
        num_block_layers: List[int], expansion_coefficient_lengths: List[
        int], sharing: List[bool], stack_types: List[str], **kwargs) ->None:
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
                    net_block = NBEATSGenericBlock(units=self.widths[
                        stack_id], thetas_dim=self.
                        expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=context_length, forecast_length=
                        prediction_length)
                elif self.stack_types[stack_id] == 'S':
                    net_block = NBEATSSeasonalBlock(units=self.widths[
                        stack_id], num_block_layers=self.num_block_layers[
                        stack_id], backcast_length=context_length,
                        forecast_length=prediction_length)
                else:
                    net_block = NBEATSTrendBlock(units=self.widths[stack_id
                        ], thetas_dim=self.expansion_coefficient_lengths[
                        stack_id], num_block_layers=self.num_block_layers[
                        stack_id], backcast_length=context_length,
                        forecast_length=prediction_length)
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

    def smape_loss(self, forecast: torch.Tensor, future_target: torch.Tensor
        ) ->torch.Tensor:
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = denominator == 0
        return 200 / self.prediction_length * torch.mean(torch.abs(
            future_target - forecast) * torch.logical_not(flag) / (
            denominator + flag), dim=1)

    def mape_loss(self, forecast: torch.Tensor, future_target: torch.Tensor
        ) ->torch.Tensor:
        denominator = torch.abs(future_target)
        flag = denominator == 0
        return 100 / self.prediction_length * torch.mean(torch.abs(
            future_target - forecast) * torch.logical_not(flag) / (
            denominator + flag), dim=1)

    def mase_loss(self, forecast: torch.Tensor, future_target: torch.Tensor,
        past_target: torch.Tensor, periodicity: int) ->torch.Tensor:
        factor = 1 / (self.context_length + self.prediction_length -
            periodicity)
        whole_target = torch.cat((past_target, future_target), dim=1)
        seasonal_error = factor * torch.mean(torch.abs(whole_target[:,
            periodicity:, (...)] - whole_target[:, :-periodicity, (...)]),
            dim=1)
        flag = seasonal_error == 0
        return torch.mean(torch.abs(future_target - forecast), dim=1
            ) * torch.logical_not(flag) / (seasonal_error + flag)


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class ArgProj(nn.Module):

    def __init__(self, in_features: int, args_dim: Dict[str, int],
        domain_map: Callable[..., Tuple[torch.Tensor]], dtype: np.dtype=np.
        float32, prefix: Optional[str]=None, **kwargs):
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.dtype = dtype
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in
            args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)


class FeatureEmbedder(nn.Module):

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]
        ) ->None:
        super().__init__()
        self.__num_features = len(cardinalities)

        def create_embedding(c: int, d: int) ->nn.Embedding:
            embedding = nn.Embedding(c, d)
            return embedding
        self.__embedders = nn.ModuleList([create_embedding(c, d) for c, d in
            zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) ->torch.Tensor:
        if self.__num_features > 1:
            cat_feature_slices = torch.chunk(features, self.__num_features,
                dim=-1)
        else:
            cat_feature_slices = [features]
        return torch.cat([embed(cat_feature_slice.squeeze(-1)) for embed,
            cat_feature_slice in zip(self.__embedders, cat_feature_slices)],
            dim=-1)


class FeatureAssembler(nn.Module):

    def __init__(self, T: int, embed_static: Optional[FeatureEmbedder]=None,
        embed_dynamic: Optional[FeatureEmbedder]=None) ->None:
        super().__init__()
        self.T = T
        self.embeddings = nn.ModuleDict({'embed_static': embed_static,
            'embed_dynamic': embed_dynamic})

    def forward(self, feat_static_cat: torch.Tensor, feat_static_real:
        torch.Tensor, feat_dynamic_cat: torch.Tensor, feat_dynamic_real:
        torch.Tensor) ->torch.Tensor:
        processed_features = [self.process_static_cat(feat_static_cat),
            self.process_static_real(feat_static_real), self.
            process_dynamic_cat(feat_dynamic_cat), self.
            process_dynamic_real(feat_dynamic_real)]
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
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data *
                (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data *
                (1 - self.momentum))
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


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask,
        cond_label_size=None):
        super().__init__()
        self.register_buffer('mask', mask)
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size
             is not None else 0), hidden_size)]
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
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        log_abs_det_jacobian = log_s
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        mu = u * self.mask
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask)
        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        log_abs_det_jacobian = -log_s
        return x, log_abs_det_jacobian


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer('mask', mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs,
                cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


def create_masks(input_size, hidden_size, n_hidden, input_order=
    'sequential', input_degrees=None):
    degrees = []
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [
            input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1
            ] if input_degrees is None else [input_degrees % input_size - 1]
    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [
            input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (
                hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,
            )) - 1] if input_degrees is None else [input_degrees - 1]
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
    return masks, degrees[0]


class MADE(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=
        None, activation='ReLU', input_order='sequential', input_degrees=None):
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
        masks, self.input_degrees = create_masks(input_size, hidden_size,
            n_hidden, input_order, input_degrees)
        if activation == 'ReLU':
            activation_fn = nn.ReLU()
        elif activation == 'Tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0],
            cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size,
                hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 *
            input_size, masks[-1].repeat(2, 1))]
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
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian,
            dim=-1)


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
        return torch.sum(self.base_dist.log_prob(u) +
            sum_log_abs_det_jacobians, dim=-1)

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape
        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class LambdaLayer(nn.Module):

    def __init__(self, function):
        super().__init__()
        self._func = function

    def forward(self, x, *args):
        return self._func(x, *args)


class Scaler(ABC, nn.Module):

    def __init__(self, keepdim: bool=False, time_first: bool=True):
        super().__init__()
        self.keepdim = keepdim
        self.time_first = time_first

    @abstractmethod
    def compute_scale(self, data: torch.Tensor, observed_indicator: torch.
        Tensor) ->torch.Tensor:
        pass

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor
        ) ->Tuple[torch.Tensor, torch.Tensor]:
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zalandoresearch_pytorch_ts(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BatchNorm(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(FeatureEmbedder(*[], **{'cardinalities': [4, 4], 'embedding_dims': [4, 4]}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(FlowSequential(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MADE(*[], **{'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(NBEATSBlock(*[], **{'units': 4, 'thetas_dim': 4}), [torch.rand([10, 10])], {})

    @_fails_compile()
    def test_005(self):
        self._check(NBEATSGenericBlock(*[], **{'units': 4, 'thetas_dim': 4}), [torch.rand([10, 10])], {})

    @_fails_compile()
    def test_006(self):
        self._check(NBEATSNetwork(*[], **{'prediction_length': 4, 'context_length': 4, 'num_stacks': 4, 'widths': [4, 4, 4, 4], 'num_blocks': [4, 4, 4, 4], 'num_block_layers': [4, 4, 4, 4], 'expansion_coefficient_lengths': [4, 4, 4, 4], 'sharing': 4, 'stack_types': [4, 4, 4, 4]}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(NBEATSSeasonalBlock(*[], **{'units': 4}), [torch.rand([10, 10])], {})

    @_fails_compile()
    def test_008(self):
        self._check(NBEATSTrendBlock(*[], **{'units': 4, 'thetas_dim': 4}), [torch.rand([10, 10])], {})

