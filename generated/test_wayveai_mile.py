import sys
_module = sys.modules[__name__]
del sys
mile_agent = _module
mile_wrapper = _module
distributions = _module
ppo = _module
ppo_buffer = _module
ppo_policy = _module
torch_layers = _module
torch_util = _module
rl_birdview_agent = _module
rl_birdview_wrapper = _module
wandb_callback = _module
carla_gym = _module
carla_multi_agent_env = _module
control = _module
route = _module
speed = _module
velocity = _module
chauffeurnet = _module
chauffeurnet_label = _module
rgb = _module
gnss = _module
waypoint_plan = _module
ego = _module
pedestrian = _module
stop_sign = _module
traffic_light_new = _module
vehicle = _module
obs_manager = _module
obs_manager_handler = _module
blocked = _module
collision = _module
encounter_light = _module
outside_route_lane = _module
route_deviation = _module
run_red_light = _module
run_stop_sign = _module
global_route_planner = _module
map_utils = _module
route_manipulation = _module
task_vehicle = _module
ego_vehicle_handler = _module
valeo_action = _module
leaderboard = _module
leaderboard_dagger = _module
valeo = _module
valeo_no_det_px = _module
basic_agent = _module
constant_speed_agent = _module
controller = _module
local_planner = _module
misc = _module
scenario_actor_handler = _module
zombie_vehicle = _module
zombie_vehicle_handler = _module
zombie_walker = _module
zombie_walker_handler = _module
envs = _module
endless_env = _module
leaderboard_env = _module
birdview_map = _module
config_utils = _module
dynamic_weather = _module
gps_utils = _module
hazard_actor = _module
traffic_light = _module
transforms = _module
data_collect = _module
evaluate = _module
config = _module
constants = _module
dataset = _module
dataset_utils = _module
layers = _module
losses = _module
metrics = _module
common = _module
frustum_pooling = _module
mile = _module
preprocess = _module
transition = _module
utils = _module
trainer = _module
carla_utils = _module
geometry_utils = _module
instance_utils = _module
network_utils = _module
visualisation = _module
train = _module
saving_utils = _module
server_utils = _module

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


import logging


import torch


from collections import deque


import numpy as np


from typing import Optional


from typing import Tuple


import torch as th


import torch.nn as nn


from torch.distributions import Beta


from torch.distributions import Normal


from torch.nn import functional as F


import time


from typing import Generator


from typing import NamedTuple


from typing import Dict


from typing import List


import queue


from typing import Union


from typing import Any


from functools import partial


from torch import nn


import math


import torch.nn.functional as F


from time import time


import pandas as pd


import scipy.ndimage


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from collections import OrderedDict


from torch import Tensor


import torchvision.transforms as transforms


import torchvision.transforms.functional as tvf


import torchvision


import matplotlib.pylab


def load_entry_point(name):
    mod_name, attr_name = name.split(':')
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class XtMaCNN(nn.Module):
    """
    Inspired by https://github.com/xtma/pytorch_car_caring
    """

    def __init__(self, observation_space, n_input_frames=1, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = n_input_frames * observation_space['birdview'].shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(8, 16, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten())
        with th.no_grad():
            n_flatten = 1024
        self.linear = nn.Sequential(nn.Linear(n_flatten + states_neurons[-1], 512), nn.ReLU(), nn.Linear(512, features_dim), nn.ReLU())
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)
        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class StateEncoder(nn.Module):

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim
        n_flatten = 256 * 6 * 6
        self.linear = nn.Sequential(nn.Linear(n_flatten + states_neurons[-1], 512), nn.ReLU(), nn.Linear(512, features_dim), nn.ReLU())
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview_state, state):
        batch_size = state.shape[0]
        birdview_state = birdview_state.view(batch_size, -1)
        latent_state = self.state_linear(state)
        x = th.cat((birdview_state, latent_state), dim=1)
        x = self.linear(x)
        return x


class ImpalaCNN(nn.Module):

    def __init__(self, observation_space, chans=(16, 32, 32, 64, 64), states_neurons=[256], features_dim=256, nblock=2, batch_norm=False, final_relu=True):
        super().__init__()
        self.features_dim = features_dim
        self.final_relu = final_relu
        curshape = observation_space['birdview'].shape
        s = 1 / np.sqrt(len(chans))
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = tu.CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, batch_norm=batch_norm)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        n_image_latent = tu.intprod(curshape)
        self.dense = tu.NormedLinear(n_image_latent + states_neurons[-1], features_dim, scale=1.4)
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(tu.NormedLinear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview, state):
        for layer in self.stacks:
            birdview = layer(birdview)
        x = th.flatten(birdview, 1)
        x = th.relu(x)
        latent_state = self.state_linear(state)
        x = th.cat((x, latent_state), dim=1)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x


def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get('bias', True):
        out.bias.data *= 0
    return out


class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        if getattr(self, 'batch_norm', False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, 'batch_norm', False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList([CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)])

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, 'pool', True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, 'pool', True):
            return self.outchan, (h + 1) // 2, (w + 1) // 2
        else:
            return self.outchan, h, w


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample_conv(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=2, dilation=1, first_dilation=first_dilation, norm_layer=nn.BatchNorm2d)
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


class RestrictionActivation(nn.Module):
    """ Constrain output to be between min_value and max_value."""

    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.scale = (max_value - min_value) / 2
        self.offset = min_value

    def forward(self, x):
        x = torch.tanh(x) + 1
        x = self.scale * x + self.offset
        return x


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm='bn', activation='relu', bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Interpolate(nn.Module):

    def __init__(self, scale_factor: int=2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)


class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1, groups=1, upsample=False, downsample=False, dropout=0.0):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, bias=False, dilation=1, stride=2, output_padding=padding_size, padding=padding_size, groups=groups)
        elif downsample:
            bottleneck_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, bias=False, dilation=dilation, stride=2, padding=padding_size, groups=groups)
        else:
            bottleneck_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size, groups=groups)
        self.layers = nn.Sequential(OrderedDict([('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)), ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))), ('conv', bottleneck_conv), ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))), ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)), ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))), ('dropout', nn.Dropout2d(p=dropout))]))
        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update({'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 'bn_skip_proj': nn.BatchNorm2d(out_channels)})
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        x, = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False), nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.upsample_layer(x)
        return x


class UpsamplingAdd(nn.Module):

    def __init__(self, in_channels, action_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False), nn.Conv2d(in_channels + action_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, x_skip, action):
        b, _, h, w = x.shape
        action = action.view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([x, action], dim=1)
        x = self.upsample_layer(x)
        return x + x_skip


class UpsamplingConcat(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class ActivatedNormLinear(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.module(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.mean(dim=(-1, -2))


SEMANTIC_SEG_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0])


class SegmentationLoss(nn.Module):

    def __init__(self, use_top_k=False, top_k_ratio=1.0, use_weights=False, poly_one=False, poly_one_coefficient=0.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient
        if self.use_weights:
            self.weights = SEMANTIC_SEG_WEIGHTS

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        if self.use_weights:
            weights = torch.tensor(self.weights, dtype=prediction.dtype, device=prediction.device)
        else:
            weights = None
        loss = F.cross_entropy(prediction, target, reduction='none', weight=weights)
        if self.poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1 - prob)
            loss = loss + loss_poly_one
        loss = loss.view(b, s, -1)
        if self.use_top_k:
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]
        return torch.mean(loss)


class RegressionLoss(nn.Module):

    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim
        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):

    def __init__(self, norm, ignore_index=255):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()
        loss = self.loss_fn(prediction, target, reduction='none')
        loss = torch.sum(loss, dim=-3, keepdims=True)
        return loss[mask].mean()


class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""

    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma[:, 1:] ** 2
        prior_var = prior_sigma[:, 1:] ** 2
        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])
        kl_div = prior_log_sigma - posterior_log_sigma - 0.5 + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        first_kl = -posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


class KLLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=True)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)
        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss


class RouteEncode(nn.Module):

    def __init__(self, out_channels, backbone='resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=[4])
        self.out_channels = out_channels
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fc = nn.Linear(feature_info[-1]['num_chs'], out_channels)

    def forward(self, route):
        x = self.backbone(route)[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class GRUCellLayerNorm(nn.Module):

    def __init__(self, input_size, hidden_size, reset_bias=1.0):
        super().__init__()
        self.reset_bias = reset_bias
        self.update_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.update_norm = nn.LayerNorm(hidden_size)
        self.reset_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.reset_norm = nn.LayerNorm(hidden_size)
        self.proposal_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.proposal_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, state):
        update = self.update_layer(torch.cat([inputs, state], -1))
        update = torch.sigmoid(self.update_norm(update))
        reset = self.reset_layer(torch.cat([inputs, state], -1))
        reset = torch.sigmoid(self.reset_norm(reset) + self.reset_bias)
        h_n = self.proposal_layer(torch.cat([inputs, reset * state], -1))
        h_n = torch.tanh(self.proposal_norm(h_n))
        output = (1 - update) * h_n + update * state
        return output


class Policy(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU(True), nn.Linear(in_channels, in_channels), nn.ReLU(True), nn.Linear(in_channels, in_channels // 2), nn.ReLU(True), nn.Linear(in_channels // 2, 2), nn.Tanh())

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):

    def __init__(self, feature_info, out_channels):
        super().__init__()
        n_upsample_skip_convs = len(feature_info) - 1
        self.conv1 = nn.Sequential(nn.Conv2d(feature_info[-1]['num_chs'], out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.upsample_skip_convs = nn.ModuleList(nn.Sequential(nn.Conv2d(feature_info[-i]['num_chs'], out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)) for i in range(2, n_upsample_skip_convs + 2))
        self.out_channels = out_channels

    def forward(self, xs: List[Tensor]) ->Tensor:
        x = self.conv1(xs[-1])
        for i, conv in enumerate(self.upsample_skip_convs):
            size = xs[-(i + 2)].shape[-2:]
            x = conv(xs[-(i + 2)]) + F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, latent_n_channels, out_channels, epsilon=1e-08):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)

    def forward(self, x, style):
        mean = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2), keepdim=True) + self.epsilon)
        x = x / std
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class ConvInstanceNorm(nn.Module):

    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.adaptive_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.segmentation_head = nn.Sequential(nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0))
        self.instance_offset_head = nn.Sequential(nn.Conv2d(in_channels, 2, kernel_size=1, padding=0))
        self.instance_center_head = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        output = {f'bev_segmentation_{self.downsample_factor}': self.segmentation_head(x), f'bev_instance_offset_{self.downsample_factor}': self.instance_offset_head(x), f'bev_instance_center_{self.downsample_factor}': self.instance_center_head(x)}
        return output


class RGBHead(nn.Module):

    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.rgb_head = nn.Sequential(nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0))

    def forward(self, x):
        output = {f'rgb_{self.downsample_factor}': self.rgb_head(x)}
        return output


class BevDecoder(nn.Module):

    def __init__(self, latent_n_channels, semantic_n_channels, constant_size=(3, 3), is_segmentation=True):
        super().__init__()
        n_channels = 512
        self.constant_tensor = nn.Parameter(torch.randn((n_channels, *constant_size), dtype=torch.float32))
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels, n_channels, latent_n_channels)
        self.middle_conv = nn.ModuleList([DecoderBlock(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(3)])
        head_module = SegmentationHead if is_segmentation else RGBHead
        self.conv1 = DecoderBlock(n_channels, 256, latent_n_channels, upsample=True)
        self.head_4 = head_module(256, semantic_n_channels, downsample_factor=4)
        self.conv2 = DecoderBlock(256, 128, latent_n_channels, upsample=True)
        self.head_2 = head_module(128, semantic_n_channels, downsample_factor=2)
        self.conv3 = DecoderBlock(128, 64, latent_n_channels, upsample=True)
        self.head_1 = head_module(64, semantic_n_channels, downsample_factor=1)

    def forward(self, w: Tensor) ->Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1])
        x = self.first_norm(x, w)
        x = self.first_conv(x, w)
        for module in self.middle_conv:
            x = module(x, w)
        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)
        output = {**output_4, **output_2, **output_1}
        return output


def bev_params_to_intrinsics(size, scale, offsetx):
    """
        size: number of pixels (width, height)
        scale: pixel size (in meters)
        offsetx: offset in x direction (direction of car travel)
    """
    intrinsics_bev = np.array([[1 / scale, 0, size[0] / 2 + offsetx], [0, -1 / scale, size[1] / 2], [0, 0, 1]], dtype=np.float32)
    return intrinsics_bev


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


def gen_dx_bx(size, scale, offsetx):
    xbound = [-size[0] * scale / 2 - offsetx * scale, size[0] * scale / 2 - offsetx * scale, scale]
    ybound = [-size[1] * scale / 2, size[1] * scale / 2, scale]
    zbound = [-10.0, 10.0, 20.0]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([(row[0] + row[2] / 2.0) for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([np.round((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def intrinsics_inverse(intrinsics):
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    one = torch.ones_like(fx)
    zero = torch.zeros_like(fx)
    intrinsics_inv = torch.stack((torch.stack((1 / fx, zero, -cx / fx), -1), torch.stack((zero, 1 / fy, -cy / fy), -1), torch.stack((zero, zero, one), -1)), -2)
    return intrinsics_inv


class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


def quick_cumsum(x, geom_feats, ranks):
    return QuickCumsum.apply(x, geom_feats, ranks)


class FrustumPooling(nn.Module):

    def __init__(self, size, scale, offsetx, dbound, downsample, use_quickcumsum=True):
        """ Pools camera frustums into Birds Eye View

        Args:
            size: (width, height) size of voxel grid
            scale: size of pixel in m
            offsetx: egocar offset (forwards) from center of bev in px
            dbound: depth planes in camera frustum (min, max, step)
            downsample: fraction of the size of the feature maps (stride of backbone)
        """
        super().__init__()
        self.register_buffer('bev_intrinsics', torch.tensor(bev_params_to_intrinsics(size, scale, offsetx)))
        dx, bx, nx = gen_dx_bx(size, scale, offsetx)
        self.nx_constant = nx.numpy().tolist()
        self.register_buffer('dx', dx, persistent=False)
        self.register_buffer('bx', bx, persistent=False)
        self.register_buffer('nx', nx, persistent=False)
        self.use_quickcumsum = use_quickcumsum
        self.dbound = dbound
        ds = torch.arange(self.dbound[0], self.dbound[1], self.dbound[2], dtype=torch.float32)
        self.D = len(ds)
        self.register_buffer('ds', ds, persistent=False)
        self.downsample = downsample
        self.register_buffer('frustum', torch.zeros(0), persistent=False)

    def initialize_frustum(self, image):
        if self.frustum.shape[0] == 0:
            device = image.device
            fH, fW = image.shape[-3:-1]
            ogfH, ogfW = fH * self.downsample, fW * self.downsample
            ds = self.ds.view(-1, 1, 1).expand(-1, fH, fW)
            xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float, device=device).view(1, 1, fW).expand(self.D, fH, fW)
            ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float, device=device).view(1, fH, 1).expand(self.D, fH, fW)
            self.frustum = torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N = trans.shape[:2]
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(intrinsics_inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def voxel_pooling(self, geom_feats, x, mask):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)
        geom_feats = geom_feats.view(Nprime, 3)
        geom_feats[:, 0] = geom_feats[:, 0] * self.bev_intrinsics[0, 0] + self.bev_intrinsics[0, 2]
        geom_feats[:, 1] = geom_feats[:, 1] * self.bev_intrinsics[1, 1] + self.bev_intrinsics[1, 2]
        geom_feats[:, 2] = (geom_feats[:, 2] - self.bx[2] + self.dx[2] / 2.0) / self.dx[2]
        geom_feats = geom_feats.long()
        batch_ix = torch.cat([torch.full(size=(Nprime // B, 1), fill_value=ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        if len(mask) > 0:
            mask = mask.view(Nprime)
            x = x[mask]
            geom_feats = geom_feats[mask]
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) + geom_feats[:, 1] * (self.nx[2] * B) + geom_feats[:, 2] * B + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        if self.use_quickcumsum and self.training:
            x, geom_feats = quick_cumsum(x, geom_feats, ranks)
        else:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        final = torch.zeros((B, C, self.nx_constant[2], self.nx_constant[1], self.nx_constant[0]), dtype=x.dtype, device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def forward(self, x, intrinsics, pose, mask=torch.zeros(0)):
        """
        Args:
            x: (B x N x D x H x W x C) frustum feature maps
            intrinsics: (B x N x 3 x 3) camera intrinsics (of input image prior to downsampling by backbone)
            pose: (B x N x 4 x 4) camera pose matrix
        """
        self.initialize_frustum(x)
        rots = pose[..., :3, :3]
        trans = pose[..., :3, 3:]
        geom = self.get_geometry(rots, trans, intrinsics)
        x = self.voxel_pooling(geom, x, mask).type_as(x)
        return x

    def get_depth_map(self, depth):
        """ Convert depth probibility distribution to depth """
        ds = self.ds.view(1, -1, 1, 1)
        depth = (ds * depth).sum(1, keepdim=True)
        depth = nn.functional.interpolate(depth, scale_factor=float(self.downsample), mode='bilinear', align_corners=False)
        return depth


CARLA_FPS = 25


DISPLAY_SEGMENTATION = True


class RepresentationModel(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1
        self.module = nn.Sequential(nn.Linear(in_channels, in_channels), nn.LeakyReLU(True), nn.Linear(in_channels, 2 * self.latent_dim))

    def forward(self, x):

        def sigmoid2(tensor: torch.Tensor, min_value: float) ->torch.Tensor:
            return 2 * torch.sigmoid(tensor / 2) + min_value
        mu_log_sigma = self.module(x)
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)
        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSM(nn.Module):

    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim, action_latent_dim, receptive_field, use_dropout=False, dropout_probability=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.action_latent_dim = action_latent_dim
        self.receptive_field = receptive_field
        self.use_dropout = use_dropout
        self.dropout_probability = dropout_probability
        self.pre_gru_net = nn.Sequential(nn.Linear(state_dim, hidden_state_dim), nn.LeakyReLU(True))
        self.recurrent_model = nn.GRUCell(input_size=hidden_state_dim, hidden_size=hidden_state_dim)
        self.posterior_action_module = nn.Sequential(nn.Linear(action_dim, self.action_latent_dim), nn.LeakyReLU(True))
        self.posterior = RepresentationModel(in_channels=hidden_state_dim + embedding_dim + self.action_latent_dim, latent_dim=state_dim)
        self.prior_action_module = nn.Sequential(nn.Linear(action_dim, self.action_latent_dim), nn.LeakyReLU(True))
        self.prior = RepresentationModel(in_channels=hidden_state_dim + self.action_latent_dim, latent_dim=state_dim)
        self.active_inference = False
        if self.active_inference:
            None

    def forward(self, input_embedding, action, use_sample=True, policy=None):
        """
        Inputs
        ------
            input_embedding: torch.Tensor size (B, S, C)
            action: torch.Tensor size (B, S, 2)
            use_sample: bool
                whether to use sample from the distributions, or taking the mean

        Returns
        -------
            output: dict
                prior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
                posterior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
        """
        output = {'prior': [], 'posterior': []}
        batch_size, sequence_length, _ = input_embedding.shape
        h_t = input_embedding.new_zeros((batch_size, self.hidden_state_dim))
        sample_t = input_embedding.new_zeros((batch_size, self.state_dim))
        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:, 0])
            else:
                action_t = action[:, t - 1]
            output_t = self.observe_step(h_t, sample_t, action_t, input_embedding[:, t], use_sample=use_sample, policy=policy)
            use_prior = self.training and self.use_dropout and torch.rand(1).item() < self.dropout_probability and t > 0
            if use_prior:
                sample_t = output_t['prior']['sample']
            else:
                sample_t = output_t['posterior']['sample']
            h_t = output_t['prior']['hidden_state']
            for key, value in output_t.items():
                output[key].append(value)
        output = self.stack_list_of_dict_tensor(output, dim=1)
        return output

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        imagine_output = self.imagine_step(h_t, sample_t, action_t, use_sample, policy=policy)
        latent_action_t = self.posterior_action_module(action_t)
        posterior_mu_t, posterior_sigma_t = self.posterior(torch.cat([imagine_output['hidden_state'], embedding_t, latent_action_t], dim=-1))
        sample_t = self.sample_from_distribution(posterior_mu_t, posterior_sigma_t, use_sample=use_sample)
        posterior_output = {'hidden_state': imagine_output['hidden_state'], 'sample': sample_t, 'mu': posterior_mu_t, 'sigma': posterior_sigma_t}
        output = {'prior': imagine_output, 'posterior': posterior_output}
        return output

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        if self.active_inference:
            action_t = policy(torch.cat([h_t, sample_t], dim=-1))
        latent_action_t = self.prior_action_module(action_t)
        input_t = self.pre_gru_net(sample_t)
        h_t = self.recurrent_model(input_t, h_t)
        prior_mu_t, prior_sigma_t = self.prior(torch.cat([h_t, latent_action_t], dim=-1))
        sample_t = self.sample_from_distribution(prior_mu_t, prior_sigma_t, use_sample=use_sample)
        imagine_output = {'hidden_state': h_t, 'sample': sample_t, 'mu': prior_mu_t, 'sigma': prior_sigma_t}
        return imagine_output

    @staticmethod
    def sample_from_distribution(mu, sigma, use_sample):
        sample = mu
        if use_sample:
            noise = torch.randn_like(sample)
            sample = sample + sigma * noise
        return sample

    @staticmethod
    def stack_list_of_dict_tensor(output, dim=1):
        new_output = {}
        for outter_key, outter_value in output.items():
            if len(outter_value) > 0:
                new_output[outter_key] = dict()
                for inner_key in outter_value[0].keys():
                    new_output[outter_key][inner_key] = torch.stack([x[inner_key] for x in outter_value], dim=dim)
        return new_output


def pack_sequence_dim(x):
    """ Does not create a copy."""
    if isinstance(x, torch.Tensor):
        b, s = x.shape[:2]
        return x.view(b * s, *x.shape[2:])
    if isinstance(x, list):
        return [pack_sequence_dim(elt) for elt in x]
    output = {}
    for key, value in x.items():
        output[key] = pack_sequence_dim(value)
    return output


def remove_past(x, receptive_field):
    """ Removes past tensors. The past is indicated by the receptive field. Creates a copy."""
    if isinstance(x, torch.Tensor):
        return x[:, receptive_field - 1:].contiguous()
    output = {}
    for key, value in x.items():
        output[key] = remove_past(value, receptive_field)
    return output


def unpack_sequence_dim(x, b, s):
    """ Does not create a copy."""
    if isinstance(x, torch.Tensor):
        return x.view(b, s, *x.shape[1:])
    if isinstance(x, list):
        return [unpack_sequence_dim(elt, b, s) for elt in x]
    output = {}
    for key, value in x.items():
        output[key] = unpack_sequence_dim(value, b, s)
    return output


class Mile(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD
        if self.cfg.MODEL.ENCODER.NAME == 'resnet18':
            self.encoder = timm.create_model(cfg.MODEL.ENCODER.NAME, pretrained=True, features_only=True, out_indices=[2, 3, 4])
            feature_info = self.encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.feat_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
        if not self.cfg.EVAL.NO_LIFTING:
            bev_downsample = cfg.BEV.FEATURE_DOWNSAMPLE
            self.frustum_pooling = FrustumPooling(size=(cfg.BEV.SIZE[0] // bev_downsample, cfg.BEV.SIZE[1] // bev_downsample), scale=cfg.BEV.RESOLUTION * bev_downsample, offsetx=cfg.BEV.OFFSET_FORWARD / bev_downsample, dbound=cfg.BEV.FRUSTUM_POOL.D_BOUND, downsample=8)
            self.depth_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
            self.depth = nn.Conv2d(self.depth_decoder.out_channels, self.frustum_pooling.D, kernel_size=1)
            self.sparse_depth = cfg.BEV.FRUSTUM_POOL.SPARSE
            self.sparse_depth_count = cfg.BEV.FRUSTUM_POOL.SPARSE_COUNT
        backbone_bev_in_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.cfg.MODEL.ROUTE.ENABLED:
            self.backbone_route = RouteEncode(cfg.MODEL.ROUTE.CHANNELS, cfg.MODEL.ROUTE.BACKBONE)
            backbone_bev_in_channels += self.backbone_route.out_channels
        if self.cfg.MODEL.MEASUREMENTS.ENABLED:
            self.command_encoder = nn.Sequential(nn.Embedding(6, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.ReLU(True), nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.ReLU(True))
            self.command_next_encoder = nn.Sequential(nn.Embedding(6, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.ReLU(True), nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS), nn.ReLU(True))
            self.gps_encoder = nn.Sequential(nn.Linear(2 * 2, self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS), nn.ReLU(True), nn.Linear(self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS, self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS), nn.ReLU(True))
            backbone_bev_in_channels += 2 * self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS
            backbone_bev_in_channels += self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS
        self.speed_enc = nn.Sequential(nn.Linear(1, cfg.MODEL.SPEED.CHANNELS), nn.ReLU(True), nn.Linear(cfg.MODEL.SPEED.CHANNELS, cfg.MODEL.SPEED.CHANNELS), nn.ReLU(True))
        backbone_bev_in_channels += cfg.MODEL.SPEED.CHANNELS
        self.speed_normalisation = cfg.SPEED.NORMALISATION
        self.backbone_bev = timm.create_model(cfg.MODEL.BEV.BACKBONE, in_chans=backbone_bev_in_channels, pretrained=True, features_only=True, out_indices=[3])
        feature_info_bev = self.backbone_bev.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        embedding_n_channels = self.cfg.MODEL.EMBEDDING_DIM
        self.final_state_conv = nn.Sequential(BasicBlock(feature_info_bev[-1]['num_chs'], embedding_n_channels, stride=2, downsample=True), BasicBlock(embedding_n_channels, embedding_n_channels), nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(start_dim=1))
        self.receptive_field = self.cfg.RECEPTIVE_FIELD
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.rssm = RSSM(embedding_dim=embedding_n_channels, action_dim=self.cfg.MODEL.ACTION_DIM, hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM, state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM, action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM, receptive_field=self.receptive_field, use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT, dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY)
        if self.cfg.MODEL.TRANSITION.ENABLED:
            state_dim = self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + self.cfg.MODEL.TRANSITION.STATE_DIM
        else:
            state_dim = embedding_n_channels
        self.policy = Policy(in_channels=state_dim)
        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.bev_decoder = BevDecoder(latent_n_channels=state_dim, semantic_n_channels=self.cfg.SEMANTIC_SEG.N_CHANNELS)
        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_decoder = BevDecoder(latent_n_channels=state_dim, semantic_n_channels=3, constant_size=(5, 13), is_segmentation=False)
        self.last_h = None
        self.last_sample = None
        self.last_action = None
        self.count = 0

    def forward(self, batch, deployment=False):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        b, s = batch['image'].shape[:2]
        embedding = self.encode(batch)
        output = dict()
        if self.cfg.MODEL.TRANSITION.ENABLED:
            if deployment:
                action = batch['action']
            else:
                action = torch.cat([batch['throttle_brake'], batch['steering']], dim=-1)
            state_dict = self.rssm(embedding, action, use_sample=not deployment, policy=self.policy)
            if deployment:
                state_dict = remove_past(state_dict, s)
                s = 1
            output = {**output, **state_dict}
            state = torch.cat([state_dict['posterior']['hidden_state'], state_dict['posterior']['sample']], dim=-1)
        else:
            state = embedding
        state = pack_sequence_dim(state)
        output_policy = self.policy(state)
        throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
        output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
        output['steering'] = unpack_sequence_dim(steering, b, s)
        if self.cfg.SEMANTIC_SEG.ENABLED:
            if not deployment or deployment and DISPLAY_SEGMENTATION:
                bev_decoder_output = self.bev_decoder(state)
                bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
                output = {**output, **bev_decoder_output}
        if self.cfg.EVAL.RGB_SUPERVISION:
            rgb_decoder_output = self.rgb_decoder(state)
            rgb_decoder_output = unpack_sequence_dim(rgb_decoder_output, b, s)
            output = {**output, **rgb_decoder_output}
        return output

    def encode(self, batch):
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(batch['image'])
        speed = pack_sequence_dim(batch['speed'])
        intrinsics = pack_sequence_dim(batch['intrinsics'])
        extrinsics = pack_sequence_dim(batch['extrinsics'])
        xs = self.encoder(image)
        x = self.feat_decoder(xs)
        if not self.cfg.EVAL.NO_LIFTING:
            depth = self.depth(self.depth_decoder(xs)).softmax(dim=1)
            if self.sparse_depth:
                topk_bins = depth.topk(self.sparse_depth_count, dim=1)[1]
                depth_mask = torch.zeros(depth.shape, device=depth.device, dtype=torch.bool)
                depth_mask.scatter_(1, topk_bins, 1)
            else:
                depth_mask = torch.zeros(0, device=depth.device)
            x = (depth.unsqueeze(1) * x.unsqueeze(2)).type_as(x)
            x = x.unsqueeze(1)
            x = x.permute(0, 1, 3, 4, 5, 2)
            x = self.frustum_pooling(x, intrinsics.unsqueeze(1), extrinsics.unsqueeze(1), depth_mask)
        if self.cfg.MODEL.ROUTE.ENABLED:
            route_map = pack_sequence_dim(batch['route_map'])
            route_map_features = self.backbone_route(route_map)
            route_map_features = route_map_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, route_map_features], dim=1)
        if self.cfg.MODEL.MEASUREMENTS.ENABLED:
            route_command = pack_sequence_dim(batch['route_command'])
            gps_vector = pack_sequence_dim(batch['gps_vector'])
            route_command_next = pack_sequence_dim(batch['route_command_next'])
            gps_vector_next = pack_sequence_dim(batch['gps_vector_next'])
            command_features = self.command_encoder(route_command)
            command_features = command_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, command_features], dim=1)
            command_next_features = self.command_next_encoder(route_command_next)
            command_next_features = command_next_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, command_next_features], dim=1)
            gps_features = self.gps_encoder(torch.cat([gps_vector, gps_vector_next], dim=-1))
            gps_features = gps_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, gps_features], dim=1)
        speed_features = self.speed_enc(speed / self.speed_normalisation)
        speed_features = speed_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, speed_features), 1)
        embedding = self.backbone_bev(x)[-1]
        embedding = self.final_state_conv(embedding)
        embedding = unpack_sequence_dim(embedding, b, s)
        return embedding

    def observe_and_imagine(self, batch, predict_action=False, future_horizon=None):
        """ This is only used for visualisation of future prediction"""
        assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
        if future_horizon is None:
            future_horizon = self.cfg.FUTURE_HORIZON
        b, s = batch['image'].shape[:2]
        if not predict_action:
            assert batch['throttle_brake'].shape[1] == s + future_horizon
            assert batch['steering'].shape[1] == s + future_horizon
        output_observe = self.forward(batch)
        output_imagine = {'action': [], 'state': [], 'hidden': [], 'sample': []}
        h_t = output_observe['posterior']['hidden_state'][:, -1]
        sample_t = output_observe['posterior']['sample'][:, -1]
        for t in range(future_horizon):
            if predict_action:
                action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
            else:
                action_t = torch.cat([batch['throttle_brake'][:, s + t], batch['steering'][:, s + t]], dim=-1)
            prior_t = self.rssm.imagine_step(h_t, sample_t, action_t, use_sample=True, policy=self.policy)
            sample_t = prior_t['sample']
            h_t = prior_t['hidden_state']
            output_imagine['action'].append(action_t)
            output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
            output_imagine['hidden'].append(h_t)
            output_imagine['sample'].append(sample_t)
        for k, v in output_imagine.items():
            output_imagine[k] = torch.stack(v, dim=1)
        bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
        output_imagine = {**output_imagine, **bev_decoder_output}
        return output_observe, output_imagine

    def imagine(self, batch, predict_action=False, future_horizon=None):
        """ This is only used for visualisation of future prediction"""
        assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
        if future_horizon is None:
            future_horizon = self.cfg.FUTURE_HORIZON
        output_imagine = {'action': [], 'state': [], 'hidden': [], 'sample': []}
        h_t = batch['hidden_state']
        sample_t = batch['sample']
        b = h_t.shape[0]
        for t in range(future_horizon):
            if predict_action:
                action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
            else:
                action_t = torch.cat([batch['throttle_brake'][:, t], batch['steering'][:, t]], dim=-1)
            prior_t = self.rssm.imagine_step(h_t, sample_t, action_t, use_sample=True, policy=self.policy)
            sample_t = prior_t['sample']
            h_t = prior_t['hidden_state']
            output_imagine['action'].append(action_t)
            output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
            output_imagine['hidden'].append(h_t)
            output_imagine['sample'].append(sample_t)
        for k, v in output_imagine.items():
            output_imagine[k] = torch.stack(v, dim=1)
        bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
        output_imagine = {**output_imagine, **bev_decoder_output}
        return output_imagine

    def deployment_forward(self, batch, is_dreaming):
        """
        Keep latent states in memory for fast inference.

        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        assert self.cfg.MODEL.TRANSITION.ENABLED
        b = batch['image'].shape[0]
        if self.count == 0:
            s = batch['image'].shape[1]
            action_t = batch['action'][:, -2]
            batch = remove_past(batch, s)
            embedding_t = self.encode(batch)[:, -1]
            if self.last_h is None:
                h_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM)
                sample_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.STATE_DIM)
            else:
                h_t = self.last_h
                sample_t = self.last_sample
            if is_dreaming:
                rssm_output = self.rssm.imagine_step(h_t, sample_t, action_t, use_sample=False, policy=self.policy)
            else:
                rssm_output = self.rssm.observe_step(h_t, sample_t, action_t, embedding_t, use_sample=False, policy=self.policy)['posterior']
            sample_t = rssm_output['sample']
            h_t = rssm_output['hidden_state']
            self.last_h = h_t
            self.last_sample = sample_t
            game_frequency = CARLA_FPS
            model_stride_sec = self.cfg.DATASET.STRIDE_SEC
            n_image_per_stride = int(game_frequency * model_stride_sec)
            self.count = n_image_per_stride - 1
        else:
            self.count -= 1
        s = 1
        state = torch.cat([self.last_h, self.last_sample], dim=-1)
        output_policy = self.policy(state)
        throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
        output = dict()
        output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
        output['steering'] = unpack_sequence_dim(steering, b, s)
        output['hidden_state'] = self.last_h
        output['sample'] = self.last_sample
        if self.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            bev_decoder_output = self.bev_decoder(state)
            bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
            output = {**output, **bev_decoder_output}
        return output


class PixelAugmentation(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.blur_prob = cfg.IMAGE.AUGMENTATION.BLUR_PROB
        self.sharpen_prob = cfg.IMAGE.AUGMENTATION.SHARPEN_PROB
        self.blur_window = cfg.IMAGE.AUGMENTATION.BLUR_WINDOW
        self.blur_std = cfg.IMAGE.AUGMENTATION.BLUR_STD
        self.sharpen_factor = cfg.IMAGE.AUGMENTATION.SHARPEN_FACTOR
        assert self.blur_prob + self.sharpen_prob <= 1
        self.color_jitter = transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(cfg.IMAGE.AUGMENTATION.COLOR_JITTER_BRIGHTNESS, cfg.IMAGE.AUGMENTATION.COLOR_JITTER_CONTRAST, cfg.IMAGE.AUGMENTATION.COLOR_JITTER_SATURATION, cfg.IMAGE.AUGMENTATION.COLOR_JITTER_HUE)]), cfg.IMAGE.AUGMENTATION.COLOR_PROB)

    def forward(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rand_value = torch.rand(1)
                if rand_value < self.blur_prob:
                    std = torch.empty(1).uniform_(self.blur_std[0], self.blur_std[1]).item()
                    image[i, j] = tvf.gaussian_blur(image[i, j], self.blur_window, std)
                elif rand_value < self.blur_prob + self.sharpen_prob:
                    factor = torch.empty(1).uniform_(self.sharpen_factor[0], self.sharpen_factor[1]).item()
                    image[i, j] = tvf.adjust_sharpness(image[i, j], factor)
                image[i, j] = self.color_jitter(image[i, j])
        batch['image'] = image
        return batch


class RouteAugmentation(nn.Module):

    def __init__(self, drop=0.025, end_of_route=0.025, small_rotation=0.025, large_rotation=0.025, degrees=8.0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=(0.1, 0.1)):
        super().__init__()
        assert drop + end_of_route + small_rotation + large_rotation <= 1
        self.drop = drop
        self.end_of_route = end_of_route
        self.small_rotation = small_rotation
        self.large_rotation = large_rotation
        self.small_perturbation = transforms.RandomAffine(degrees, translate, scale, shear)
        self.large_perturbation = transforms.RandomAffine(180, translate, scale, shear)

    def forward(self, batch):
        if 'route_map' in batch:
            route_map = batch['route_map']
            for i in range(route_map.shape[0]):
                rand_value = torch.rand(1)
                if rand_value < self.drop:
                    route_map[i] = torch.zeros_like(route_map[i])
                elif rand_value < self.drop + self.end_of_route:
                    height = torch.randint(route_map[i].shape[-2], (1,))
                    route_map[i][:, :, :height] = 0
                elif rand_value < self.drop + self.end_of_route + self.small_rotation:
                    route_map[i] = self.small_perturbation(route_map[i])
                elif rand_value < self.drop + self.end_of_route + self.small_rotation + self.large_rotation:
                    route_map[i] = self.large_perturbation(route_map[i])
            batch['route_map'] = route_map
        return batch


def convert_instance_mask_to_center_and_offset_label(instance_label, ignore_index=255, sigma=3):
    instance_label = instance_label.squeeze(2)
    batch_size, seq_len, h, w = instance_label.shape
    center_label = torch.zeros(batch_size, seq_len, 1, h, w, device=instance_label.device)
    offset_label = ignore_index * torch.ones(batch_size, seq_len, 2, h, w, device=instance_label.device)
    x, y = torch.meshgrid(torch.arange(h, dtype=torch.float, device=instance_label.device), torch.arange(w, dtype=torch.float, device=instance_label.device))
    for b in range(batch_size):
        num_instances = instance_label[b].max()
        for instance_id in range(1, num_instances + 1):
            for t in range(seq_len):
                instance_mask = instance_label[b, t] == instance_id
                if instance_mask.sum() == 0:
                    continue
                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()
                off_x = xc - x
                off_y = yc - y
                g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
                center_label[b, t, 0] = torch.maximum(center_label[b, t, 0], g)
                offset_label[b, t, 0, instance_mask] = off_x[instance_mask]
                offset_label[b, t, 1, instance_mask] = off_y[instance_mask]
    return center_label, offset_label


def functional_crop(batch: Dict[str, torch.Tensor], crop: Tuple[int, int, int, int]):
    left, top, right, bottom = crop
    height = bottom - top
    width = right - left
    if 'image' in batch:
        batch['image'] = tvf.crop(batch['image'], top, left, height, width)
    if 'depth' in batch:
        batch['depth'] = tvf.crop(batch['depth'], top, left, height, width)
    if 'semseg' in batch:
        batch['semseg'] = tvf.crop(batch['semseg'], top, left, height, width)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., 0, 2] -= left
        intrinsics[..., 1, 2] -= top
        batch['intrinsics'] = intrinsics
    return batch


def functional_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    b, s, c, h, w = x.shape
    x = x.view(b * s, c, h, w)
    x = tvf.resize(x, size, interpolation=mode)
    x = x.view(b, s, c, *size)
    return x


def functional_resize_batch(batch, scale):
    b, s, c, h, w = batch['image'].shape
    h1, w1 = int(round(h * scale)), int(round(w * scale))
    size = h1, w1
    if 'image' in batch:
        image = batch['image'].view(b * s, c, h, w)
        image = tvf.resize(image, size, antialias=True)
        batch['image'] = image.view(b, s, c, h1, w1)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., :2, :] *= scale
        batch['intrinsics'] = intrinsics
    return batch


def get_out_of_view_mask(cfg):
    """ Returns a mask of everything that is not visible from the image given a certain bird's-eye view grid."""
    fov = cfg.IMAGE.FOV
    w = cfg.IMAGE.SIZE[1]
    resolution = cfg.BEV.RESOLUTION
    f = w / (2 * np.tan(fov * np.pi / 360.0))
    c_u = w / 2 - cfg.IMAGE.CROP[0]
    bev_left = -np.round(cfg.BEV.SIZE[0] // 2 * resolution, decimals=1)
    bev_right = np.round(cfg.BEV.SIZE[0] // 2 * resolution, decimals=1)
    bev_bottom = 0.01
    camera_offset = (cfg.BEV.SIZE[1] / 2 + cfg.BEV.OFFSET_FORWARD) * resolution + cfg.IMAGE.CAMERA_POSITION[0]
    bev_top = np.round(cfg.BEV.SIZE[1] * resolution - camera_offset, decimals=1)
    x, z = np.arange(bev_left, bev_right, resolution), np.arange(bev_bottom, bev_top, resolution)
    ucoords = x / z[:, None] * f + c_u
    new_w = cfg.IMAGE.CROP[2] - cfg.IMAGE.CROP[0]
    mask = (ucoords >= 0) & (ucoords < new_w)
    mask = ~mask[::-1]
    mask_behind_ego_vehicle = np.ones((int(camera_offset / resolution), mask.shape[1]), dtype=np.bool)
    return np.vstack([mask, mask_behind_ego_vehicle])


class PreProcess(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.crop = tuple(cfg.IMAGE.CROP)
        self.route_map_size = cfg.ROUTE.SIZE
        self.bev_out_of_view_mask = get_out_of_view_mask(cfg)
        self.center_sigma = cfg.INSTANCE_SEG.CENTER_LABEL_SIGMA_PX
        self.ignore_index = cfg.INSTANCE_SEG.IGNORE_INDEX
        self.min_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[0]
        self.max_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[1]
        self.pixel_augmentation = PixelAugmentation(cfg)
        self.route_augmentation = RouteAugmentation(cfg.ROUTE.AUGMENTATION_DROPOUT, cfg.ROUTE.AUGMENTATION_END_OF_ROUTE, cfg.ROUTE.AUGMENTATION_SMALL_ROTATION, cfg.ROUTE.AUGMENTATION_LARGE_ROTATION, cfg.ROUTE.AUGMENTATION_DEGREES, cfg.ROUTE.AUGMENTATION_TRANSLATE, cfg.ROUTE.AUGMENTATION_SCALE, cfg.ROUTE.AUGMENTATION_SHEAR)
        self.register_buffer('image_mean', torch.tensor(cfg.IMAGE.IMAGENET_MEAN).unsqueeze(1).unsqueeze(1))
        self.register_buffer('image_std', torch.tensor(cfg.IMAGE.IMAGENET_STD).unsqueeze(1).unsqueeze(1))

    def augmentation(self, batch: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        batch = self.pixel_augmentation(batch)
        batch = self.route_augmentation(batch)
        return batch

    def prepare_bev_labels(self, batch):
        if 'birdview_label' in batch:
            batch['birdview_label'][:, :, :, self.bev_out_of_view_mask] = 0
            batch['birdview_label'] = torch.rot90(batch['birdview_label'], k=-1, dims=[3, 4]).contiguous()
            batch['birdview_label_1'] = batch['birdview_label']
            h, w = batch['birdview_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'birdview_label_{downsample_factor}'] = functional_resize(batch[f'birdview_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST)
        if 'instance_label' in batch:
            batch['instance_label'][:, :, :, self.bev_out_of_view_mask] = 0
            batch['instance_label'] = torch.rot90(batch['instance_label'], k=-1, dims=[3, 4]).contiguous()
            center_label, offset_label = convert_instance_mask_to_center_and_offset_label(batch['instance_label'], ignore_index=self.ignore_index, sigma=self.center_sigma)
            batch['center_label'] = center_label
            batch['offset_label'] = offset_label
            batch['instance_label_1'] = batch['instance_label']
            batch['center_label_1'] = batch['center_label']
            batch['offset_label_1'] = batch['offset_label']
            h, w = batch['instance_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'instance_label_{downsample_factor}'] = functional_resize(batch[f'instance_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST)
                center_label, offset_label = convert_instance_mask_to_center_and_offset_label(batch[f'instance_label_{downsample_factor}'], ignore_index=self.ignore_index, sigma=self.center_sigma / downsample_factor)
                batch[f'center_label_{downsample_factor}'] = center_label
                batch[f'offset_label_{downsample_factor}'] = offset_label
        if self.cfg.EVAL.RGB_SUPERVISION:
            batch['rgb_label_1'] = batch['image']
            h, w = batch['rgb_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'rgb_label_{downsample_factor}'] = functional_resize(batch[f'rgb_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.BILINEAR)
        return batch

    def forward(self, batch: Dict[str, torch.Tensor]):
        batch['image'] = batch['image'].float() / 255
        if 'route_map' in batch:
            batch['route_map'] = batch['route_map'].float() / 255
            batch['route_map'] = functional_resize(batch['route_map'], size=(self.route_map_size, self.route_map_size))
        batch = functional_crop(batch, self.crop)
        if self.cfg.EVAL.RESOLUTION.ENABLED:
            batch = functional_resize_batch(batch, scale=1 / self.cfg.EVAL.RESOLUTION.FACTOR)
        batch = self.prepare_bev_labels(batch)
        if self.training:
            batch = self.augmentation(batch)
        batch['image'] = (batch['image'] - self.image_mean) / self.image_std
        if 'route_map' in batch:
            batch['route_map'] = (batch['route_map'] - self.image_mean) / self.image_std
        if 'depth' in batch:
            batch['depth_mask'] = (batch['depth'] > self.min_depth) & (batch['depth'] < self.max_depth)
        return batch


class Concat(nn.Module):

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=-1)


class PixelNorm(nn.Module):

    def forward(self, x, epsilon=1e-08):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdims=True) + epsilon)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActivatedNormLinear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (AdaptiveInstanceNorm,
     lambda: ([], {'latent_n_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BevDecoder,
     lambda: ([], {'latent_n_channels': 4, 'semantic_n_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (CnnBasicBlock,
     lambda: ([], {'inchan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CnnDownStack,
     lambda: ([], {'inchan': 4, 'nblock': 4, 'outchan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Concat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRUCellLayerNorm,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Interpolate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Policy,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProbabilisticLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RGBHead,
     lambda: ([], {'in_channels': 4, 'n_classes': 4, 'downsample_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RSSM,
     lambda: ([], {'embedding_dim': 4, 'action_dim': 4, 'hidden_state_dim': 4, 'state_dim': 4, 'action_latent_dim': 4, 'receptive_field': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (RepresentationModel,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RestrictionActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SegmentationHead,
     lambda: ([], {'in_channels': 4, 'n_classes': 4, 'downsample_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SegmentationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.ones([16, 4, 4], dtype=torch.int64)], {}),
     False),
    (Upsampling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wayveai_mile(_paritybench_base):
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

    def test_020(self):
        self._check(*TESTCASES[20])

