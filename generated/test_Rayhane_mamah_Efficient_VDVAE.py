import sys
_module = sys.modules[__name__]
del sys
preprocess_celeba64 = _module
random_split = _module
utility_resize = _module
efficient_vdvae_jax = _module
data = _module
cifar10_data_loader = _module
generic_data_loader = _module
imagenet_data_loader = _module
mnist_data_loader = _module
model = _module
adamax = _module
autoencoder = _module
conv2d = _module
div_stats_utils = _module
latent_layers = _module
layers = _module
losses = _module
model = _module
optimizers = _module
schedules = _module
ssim = _module
synthesize = _module
train = _module
utils = _module
denormalizer = _module
ema_train_state = _module
inference_helpers = _module
normalizer = _module
temperature_functions = _module
train_helpers = _module
efficient_vdvae_torch = _module
cifar10_data_loader = _module
generic_data_loader = _module
imagenet_data_loader = _module
mnist_data_loader = _module
adamax = _module
autoencoder = _module
conv2d = _module
def_model = _module
div_stats_utils = _module
latent_layers = _module
layers = _module
losses = _module
model = _module
schedules = _module
ssim = _module
synthesize = _module
train = _module
utils = _module

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


from typing import Union


from typing import Tuple


from typing import Callable


from typing import Any


import numpy as np


import torch.utils.data


import tensorflow as tf


import torchvision.transforms as transforms


from sklearn.utils import shuffle


from torch.utils.data.distributed import DistributedSampler


import torch


from torch.optim import Optimizer


import torch.nn as nn


import torch.nn.init as init


import torch.nn.functional as F


from torch import nn


import time


import torch.distributed as dist


import typing


from typing import List


from torch.distributions.bernoulli import Bernoulli


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import CosineAnnealingLR


import warnings


from numpy.random import seed


import random


import copy


from torch.nn.parallel.distributed import DistributedDataParallel


from collections import defaultdict


from torch.utils.tensorboard import SummaryWriter


def get_causal_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("can't have the stride and dilation over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        p_ += p, 0
    return p_


def get_same_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("Can't have the stride and dilation rate over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        if p % 2 == 0:
            p = p // 2, p // 2
        else:
            p = int(np.ceil(p / 2)), int(np.floor(p / 2))
        p_ += p
    return tuple(p_)


def get_valid_padding(n_dims=2):
    p_ = (0,) * 2 * n_dims
    return p_


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, *args, **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        self.stride = stride
        self.padding_str = padding.upper()
        if self.padding_str == 'SAME':
            self.pad_values = get_same_padding(kernel_size, stride, dilation)
        elif self.padding_str == 'VALID':
            self.pad_values = get_valid_padding()
        elif self.padding_str == 'CAUSAL':
            self.pad_values = get_causal_padding(kernel_size, stride, dilation)
        else:
            raise ValueError
        self.condition = np.sum(self.pad_values) != 0
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, *args, **kwargs)

    def reset_parameters(self) ->None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        if self.condition:
            x = F.pad(x, self.pad_values)
        x = super(Conv2d, self).forward(x)
        return x


class PoolLayer(nn.Module):

    def __init__(self, in_filters, filters, strides):
        super(PoolLayer, self).__init__()
        self.filters = filters
        if isinstance(strides, int):
            strides = strides, strides
        ops = [Conv2d(in_channels=in_filters, out_channels=filters, kernel_size=strides, stride=strides, padding='same'), nn.LeakyReLU(negative_slope=0.1)]
        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class ResidualConvCell(nn.Module):

    def __init__(self, n_layers, in_filters, bottleneck_ratio, kernel_size, init_scaler, residual=True, use_1x1=True, output_ratio=1.0):
        super(ResidualConvCell, self).__init__()
        self.residual = residual
        self.output_ratio = output_ratio
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        if self.residual:
            assert self.output_ratio == 1
        output_filters = int(in_filters * output_ratio)
        bottlneck_filters = int(in_filters * bottleneck_ratio)
        convs = [nn.SiLU(inplace=False), nn.Conv2d(in_channels=in_filters, out_channels=bottlneck_filters, kernel_size=(1, 1) if use_1x1 else kernel_size, stride=(1, 1), padding='same')]
        for _ in range(n_layers):
            convs.append(nn.SiLU(inplace=False))
            convs.append(Conv2d(in_channels=bottlneck_filters, out_channels=bottlneck_filters, kernel_size=kernel_size, stride=(1, 1), padding='same'))
        convs += [nn.SiLU(inplace=False), Conv2d(in_channels=bottlneck_filters, out_channels=output_filters, kernel_size=(1, 1) if use_1x1 else kernel_size, stride=(1, 1), padding='same')]
        convs[-1].weight.data *= init_scaler
        self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        x = inputs
        x = self.convs(x)
        if self.residual:
            outputs = inputs + x
        else:
            outputs = x
        return outputs


class LevelBlockUp(nn.Module):

    def __init__(self, n_blocks, n_layers, in_filters, filters, bottleneck_ratio, kernel_size, strides, skip_filters, use_skip):
        super(LevelBlockUp, self).__init__()
        self.strides = strides
        self.use_skip = use_skip
        self.residual_block = nn.Sequential(*[ResidualConvCell(n_layers=n_layers, in_filters=in_filters, bottleneck_ratio=bottleneck_ratio, kernel_size=kernel_size, init_scaler=np.sqrt(1.0 / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))) if hparams.model.stable_init else 1.0, use_1x1=hparams.model.use_1x1_conv) for _ in range(n_blocks)])
        if self.use_skip:
            self.skip_projection = Conv2d(in_channels=in_filters, out_channels=skip_filters, kernel_size=(1, 1), stride=(1, 1), padding='same')
        if self.strides > 1:
            self.pool = PoolLayer(in_filters, filters, strides)

    def forward(self, x):
        x = self.residual_block(x)
        if self.use_skip:
            skip_output = self.skip_projection(x)
        else:
            skip_output = x
        if self.strides > 1:
            x = self.pool(x)
        return x, skip_output


class BottomUp(torch.nn.Module):

    def __init__(self):
        super(BottomUp, self).__init__()
        in_channels_up = [hparams.model.input_conv_filters] + hparams.model.up_filters[0:-1]
        self.levels_up = nn.ModuleList([])
        self.levels_up_downsample = nn.ModuleList([])
        for i, stride in enumerate(hparams.model.up_strides):
            elements = nn.ModuleList([])
            for j in range(hparams.model.up_n_blocks_per_res[i]):
                elements.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i], n_layers=hparams.model.up_n_layers[i], in_filters=in_channels_up[i], filters=hparams.model.up_filters[i], bottleneck_ratio=hparams.model.up_mid_filters_ratio[i], kernel_size=hparams.model.up_kernel_size[i], strides=1, skip_filters=hparams.model.up_skip_filters[i], use_skip=False)])
            self.levels_up.extend([elements])
            self.levels_up_downsample.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i], n_layers=hparams.model.up_n_layers[i], in_filters=in_channels_up[i], filters=hparams.model.up_filters[i], bottleneck_ratio=hparams.model.up_mid_filters_ratio[i], kernel_size=hparams.model.up_kernel_size[i], strides=stride, skip_filters=hparams.model.up_skip_filters[i], use_skip=True)])
        self.input_conv = Conv2d(in_channels=hparams.data.channels, out_channels=hparams.model.input_conv_filters, kernel_size=hparams.model.input_kernel_size, stride=(1, 1), padding='same')

    def forward(self, x):
        x = self.input_conv(x)
        skip_list = []
        for i, (level_up, level_up_downsample) in enumerate(zip(self.levels_up, self.levels_up_downsample)):
            for layer in level_up:
                x, _ = layer(x)
            x, skip_out = level_up_downsample(x)
            skip_list.append(skip_out)
        skip_list = skip_list[::-1]
        return skip_list


def _logstd_mode(x, prior_stats):
    mean, logstd = torch.chunk(x, chunks=2, dim=1)
    if prior_stats is not None:
        mean = mean + prior_stats[0]
        logstd = logstd + prior_stats[1]
    std = torch.exp(hparams.model.gradient_smoothing_beta * logstd)
    stats = [mean, logstd]
    return mean, std, stats


def _std_mode(x, prior_stats, softplus):
    mean, std = torch.chunk(x, chunks=2, dim=1)
    std = softplus(std)
    if prior_stats is not None:
        mean = mean + prior_stats[0]
        std = std * prior_stats[1]
    stats = [mean, std]
    return mean, std, stats


@torch.jit.script
def calculate_z(mean, std):
    eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0.0, 1.0)
    z = eps * std + mean
    return z, mean, std


class GaussianLatentLayer(nn.Module):

    def __init__(self, in_filters, num_variates, min_std=np.exp(-2)):
        super(GaussianLatentLayer, self).__init__()
        self.projection = Conv2d(in_channels=in_filters, out_channels=num_variates * 2, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.min_std = min_std
        self.softplus = torch.nn.Softplus(beta=hparams.model.gradient_smoothing_beta)

    def forward(self, x, temperature=None, prior_stats=None, return_sample=True):
        x = self.projection(x)
        if hparams.model.distribution_base == 'std':
            mean, std, stats = _std_mode(x, prior_stats, self.softplus)
        elif hparams.model.distribution_base == 'logstd':
            mean, std, stats = _logstd_mode(x, prior_stats)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')
        if temperature is not None:
            std = std * temperature
        if return_sample:
            z, mean, std = calculate_z(mean, std)
            return z, stats
        return stats


class Interpolate(nn.Module):

    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Unpoolayer(nn.Module):

    def __init__(self, in_filters, filters, strides):
        super(Unpoolayer, self).__init__()
        self.filters = filters
        if isinstance(strides, int):
            self.strides = strides, strides
        else:
            self.strides = strides
        ops = [Conv2d(in_channels=in_filters, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), padding='same'), nn.LeakyReLU(negative_slope=0.1), Interpolate(scale=self.strides)]
        self.register_parameter('scale_bias', None)
        self.ops = nn.Sequential(*ops)

    def reset_parameters(self, inputs):
        B, C, H, W = inputs.shape
        self.scale_bias = nn.Parameter(torch.zeros(size=(1, C, H, W), device='cuda'), requires_grad=True)

    def forward(self, x):
        x = self.ops(x)
        if self.scale_bias is None:
            self.reset_parameters(x)
        x = x + self.scale_bias
        return x


class LevelBlockDown(nn.Module):

    def __init__(self, n_blocks, n_layers, in_filters, filters, bottleneck_ratio, kernel_size, strides, skip_filters, latent_variates, first_block, last_block):
        super(LevelBlockDown, self).__init__()
        self.first_block = first_block
        self.last_block = last_block
        self.strides = strides
        self.filters = filters
        assert not (self.first_block and self.last_block)
        if self.strides > 1:
            self.unpool = Unpoolayer(in_filters, filters, strides)
            in_filters = filters
        self.residual_block = nn.Sequential(*[ResidualConvCell(n_layers=n_layers, in_filters=in_filters, bottleneck_ratio=bottleneck_ratio, kernel_size=kernel_size, init_scaler=np.sqrt(1.0 / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))) if hparams.model.stable_init else 1.0, use_1x1=hparams.model.use_1x1_conv) for _ in range(n_blocks)])
        self.posterior_net = nn.Sequential(ResidualConvCell(n_layers=n_layers, in_filters=in_filters + skip_filters, bottleneck_ratio=bottleneck_ratio * 0.5, kernel_size=kernel_size, init_scaler=1.0, residual=False, use_1x1=hparams.model.use_1x1_conv, output_ratio=0.5))
        self.prior_net = nn.Sequential(ResidualConvCell(n_layers=n_layers, in_filters=in_filters, bottleneck_ratio=bottleneck_ratio, kernel_size=kernel_size, init_scaler=0.0 if hparams.model.initialize_prior_weights_as_zero else 1.0, residual=False, use_1x1=hparams.model.use_1x1_conv, output_ratio=2.0))
        self.prior_layer = GaussianLatentLayer(in_filters=in_filters, num_variates=latent_variates)
        self.posterior_layer = GaussianLatentLayer(in_filters=in_filters, num_variates=latent_variates)
        self.latent_embeddings = None
        self.z_projection = Conv2d(in_channels=latent_variates, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.z_projection.weight.data *= np.sqrt(1.0 / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)))

    def sampler(self, latent_fn, y, prior_stats=None, temperature=None):
        z, dist = latent_fn(y, prior_stats=prior_stats, temperature=temperature)
        return z, dist

    def get_analytical_distribution(self, latent_fn, y, prior_stats=None):
        dist = latent_fn(y, prior_stats=prior_stats, return_sample=False)
        return None, dist

    def sample_from_weights(self, latent_fn, y, attention_weights, latent_embeddings):
        z = latent_fn.extract_and_project_memory(attention_weights, latent_embeddings)
        B, _, H, W = y.size()
        z = z.reshape(B, z.size()[1], H, W)
        return z

    def forward(self, x_skip, y, variate_mask=None):
        if self.strides > 1:
            y = self.unpool(y)
        y_prior_kl = self.prior_net(y)
        kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=1)
        y_post = torch.cat([y, x_skip], dim=1)
        y_post = self.posterior_net(y_post)
        if variate_mask is None:
            z_prior_kl, prior_kl_dist = self.get_analytical_distribution(self.prior_layer, y_prior_kl)
        else:
            z_prior_kl, prior_kl_dist = self.sampler(self.prior_layer, y_prior_kl)
        z_post, posterior_dist = self.sampler(self.posterior_layer, y_post, prior_stats=prior_kl_dist if hparams.model.use_residual_distribution else None)
        if variate_mask is not None:
            variate_mask = torch.Tensor(variate_mask)[None, :, None, None]
            z_post = variate_mask * z_post + (1.0 - variate_mask) * z_prior_kl
        y = y + kl_residual
        z_post = self.z_projection(z_post)
        y = y + z_post
        y = self.residual_block(y)
        return y, posterior_dist, prior_kl_dist

    def sample_from_prior(self, y, temperature):
        if self.strides > 1:
            y = self.unpool(y)
        y_prior = self.prior_net(y)
        kl_residual, y_prior = torch.chunk(y_prior, chunks=2, dim=1)
        y = y + kl_residual
        z, _ = self.sampler(self.prior_layer, y_prior, temperature=temperature)
        proj_z = self.z_projection(z)
        y = y + proj_z
        y = self.residual_block(y)
        return y, z


def compute_latent_dimension():
    assert np.prod(hparams.model.down_strides) == np.prod(hparams.model.up_strides)
    return hparams.data.target_res // np.prod(hparams.model.down_strides)


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=torch.device('cuda'))
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot


def scale_pixels(img):
    img = np.floor(img / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale
    return img


class TopDown(torch.nn.Module):

    def __init__(self):
        super(TopDown, self).__init__()
        self.min_pix_value = scale_pixels(0.0)
        self.max_pix_value = scale_pixels(255.0)
        H = W = compute_latent_dimension()
        self.trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, hparams.model.down_filters[0], H, W)), requires_grad=True)
        nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        in_channels_down = [hparams.model.down_filters[0]] + hparams.model.down_filters[0:-1]
        self.levels_down, self.levels_down_upsample = nn.ModuleList([]), nn.ModuleList([])
        for i, stride in enumerate(hparams.model.down_strides):
            self.levels_down_upsample.extend([LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i], n_layers=hparams.model.down_n_layers[i], in_filters=in_channels_down[i], filters=hparams.model.down_filters[i], bottleneck_ratio=hparams.model.down_mid_filters_ratio[i], kernel_size=hparams.model.down_kernel_size[i], strides=stride, skip_filters=hparams.model.up_skip_filters[::-1][i], latent_variates=hparams.model.down_latent_variates[i], first_block=i == 0, last_block=False)])
            self.levels_down.extend([nn.ModuleList([LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i], n_layers=hparams.model.down_n_layers[i], in_filters=hparams.model.down_filters[i], filters=hparams.model.down_filters[i], bottleneck_ratio=hparams.model.down_mid_filters_ratio[i], kernel_size=hparams.model.down_kernel_size[i], strides=1, skip_filters=hparams.model.up_skip_filters[::-1][i], latent_variates=hparams.model.down_latent_variates[i], first_block=False, last_block=i == len(hparams.model.down_strides) - 1 and j == hparams.model.down_n_blocks_per_res[i] - 1) for j in range(hparams.model.down_n_blocks_per_res[i])])])
        self.output_conv = Conv2d(in_channels=hparams.model.down_filters[-1], out_channels=1 if hparams.data.dataset_source == 'binarized_mnist' else hparams.model.num_output_mixtures * (3 * hparams.data.channels + 1), kernel_size=hparams.model.output_kernel_size, stride=(1, 1), padding='same')

    def sample(self, logits):
        if hparams.data.dataset_source == 'binarized_mnist':
            return self._sample_from_bernoulli(logits)
        else:
            return self._sample_from_mol(logits)

    def _sample_from_bernoulli(self, logits):
        logits = logits[:, :, 2:30, 2:30]
        probs = torch.sigmoid(logits)
        return torch.Tensor(logits.size()).bernoulli_(probs)

    def _compute_scales(self, logits):
        softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
        if hparams.model.output_distribution_base == 'std':
            scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))
        elif hparams.model.output_distribution_base == 'logstd':
            log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
            scales = torch.exp(hparams.model.output_gradient_smoothing_beta * log_scales)
        else:
            raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')
        return scales

    def _sample_from_mol(self, logits):
        B, _, H, W = logits.size()
        logit_probs = logits[:, :hparams.model.num_output_mixtures, :, :]
        l = logits[:, hparams.model.num_output_mixtures:, :, :]
        l = l.reshape(B, hparams.data.channels, 3 * hparams.model.num_output_mixtures, H, W)
        model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]
        scales = self._compute_scales(l[:, :, hparams.model.num_output_mixtures:2 * hparams.model.num_output_mixtures, :, :])
        model_coeffs = torch.tanh(l[:, :, 2 * hparams.model.num_output_mixtures:3 * hparams.model.num_output_mixtures, :, :])
        gumbel_noise = -torch.log(-torch.log(torch.Tensor(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)))
        logit_probs = logit_probs / hparams.synthesis.output_temperature + gumbel_noise
        lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size()[1], dim=1)
        lambda_ = lambda_.unsqueeze(1)
        means = torch.sum(model_means * lambda_, dim=2)
        scales = torch.sum(scales * lambda_, dim=2)
        coeffs = torch.sum(model_coeffs * lambda_, dim=2)
        u = torch.Tensor(means.size()).uniform_(1e-05, 1.0 - 1e-05)
        x = means + scales * hparams.synthesis.output_temperature * (torch.log(u) - torch.log(1.0 - u))
        x0 = torch.clamp(x[:, 0:1, :, :], min=self.min_pix_value, max=self.max_pix_value)
        x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, min=self.min_pix_value, max=self.max_pix_value)
        x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1, min=self.min_pix_value, max=self.max_pix_value)
        x = torch.cat([x0, x1, x2], dim=1)
        return x

    def forward(self, skip_list, variate_masks):
        y = torch.tile(self.trainable_h, (skip_list[0].size()[0], 1, 1, 1))
        posterior_dist_list = []
        prior_kl_dist_list = []
        layer_idx = 0
        for i, (level_down_upsample, level_down, skip_input) in enumerate(zip(self.levels_down_upsample, self.levels_down, skip_list)):
            y, posterior_dist, prior_kl_dist = level_down_upsample(skip_input, y, variate_mask=variate_masks[layer_idx])
            layer_idx += 1
            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]
            for j, layer in enumerate(level_down):
                y, posterior_dist, prior_kl_dist = layer(skip_input, y, variate_mask=variate_masks[layer_idx])
                layer_idx += 1
                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)
            posterior_dist_list += resolution_posterior_dist
            prior_kl_dist_list += resolution_prior_kl_dist
        y = self.output_conv(y)
        return y, posterior_dist_list, prior_kl_dist_list

    def sample_from_prior(self, batch_size, temperatures):
        with torch.no_grad():
            y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))
            prior_zs = []
            for i, (level_down_upsample, level_down, temperature) in enumerate(zip(self.levels_down_upsample, self.levels_down, temperatures)):
                y, z = level_down_upsample.sample_from_prior(y, temperature=temperature)
                level_z = [z]
                for _, layer in enumerate(level_down):
                    y, z = layer.sample_from_prior(y, temperature=temperature)
                    level_z.append(z)
                prior_zs += level_z
            y = self.output_conv(y)
        return y, prior_zs


Array = Any


Dtype = Any


PRNGKey = Any


Shape = Iterable[int]


class UniversalAutoEncoder(nn.Module):

    def __init__(self):
        super(UniversalAutoEncoder, self).__init__()
        self.bottom_up = BottomUp()
        self.top_down = TopDown()

    def forward(self, x, variate_masks=None):
        """
        x: (batch_size, time, H, W, C). In train, this is the shifted version of the target
        In slow synthesis, it would be the concatenated previous outputs
        """
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)
        skip_list = self.bottom_up(x)
        outputs, posterior_dist_list, prior_kl_dist_list = self.top_down(skip_list, variate_masks=variate_masks)
        return outputs, posterior_dist_list, prior_kl_dist_list


def effective_pixels():
    if hparams.data.dataset_source == 'binarized_mnist':
        return 28 * 28 * hparams.data.channels
    else:
        return hparams.data.target_res * hparams.data.target_res * hparams.data.channels


class BernoulliLoss(nn.Module):

    def __init__(self):
        super(BernoulliLoss, self).__init__()

    def forward(self, targets, logits, global_batch_size):
        targets = targets[:, :, 2:30, 2:30]
        logits = logits[:, :, 2:30, 2:30]
        loss_value = Bernoulli(logits=logits)
        recon = loss_value.log_prob(targets)
        mean_axis = list(range(1, len(recon.size())))
        per_example_loss = -torch.sum(recon, dim=mean_axis)
        scalar = global_batch_size * effective_pixels()
        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(per_example_loss) / global_batch_size
        model_means, log_scales = None, None
        return loss, avg_loss, model_means, log_scales


def _compute_inv_stdv(logits):
    softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
    if hparams.model.output_distribution_base == 'std':
        scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))
        inv_stdv = 1.0 / scales
        log_scales = torch.log(scales)
    elif hparams.model.output_distribution_base == 'logstd':
        log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
        inv_stdv = torch.exp(-hparams.model.output_gradient_smoothing_beta * log_scales)
    else:
        raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')
    return inv_stdv, log_scales


class DiscMixLogistic(nn.Module):

    def __init__(self):
        super(DiscMixLogistic, self).__init__()
        self.num_classes = 2.0 ** hparams.data.num_bits - 1.0
        self.min_pix_value = scale_pixels(0.0)
        self.max_pix_value = scale_pixels(255.0)

    def forward(self, targets, logits, global_batch_size):
        assert len(targets.shape) == 4
        B, C, H, W = targets.size()
        assert C == 3
        targets = targets.unsqueeze(2)
        logit_probs = logits[:, :hparams.model.num_output_mixtures, :, :]
        l = logits[:, hparams.model.num_output_mixtures:, :, :]
        l = l.reshape(B, hparams.data.channels, 3 * hparams.model.num_output_mixtures, H, W)
        model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]
        inv_stdv, log_scales = _compute_inv_stdv(l[:, :, hparams.model.num_output_mixtures:2 * hparams.model.num_output_mixtures, :, :])
        model_coeffs = torch.tanh(l[:, :, 2 * hparams.model.num_output_mixtures:3 * hparams.model.num_output_mixtures, :, :])
        mean1 = model_means[:, 0:1, :, :, :]
        mean2 = model_means[:, 1:2, :, :, :] + model_coeffs[:, 0:1, :, :, :] * targets[:, 0:1, :, :, :]
        mean3 = model_means[:, 2:3, :, :, :] + model_coeffs[:, 1:2, :, :, :] * targets[:, 0:1, :, :, :] + model_coeffs[:, 2:3, :, :, :] * targets[:, 1:2, :, :, :]
        means = torch.cat([mean1, mean2, mean3], dim=1)
        centered = targets - means
        plus_in = inv_stdv * (centered + 1.0 / self.num_classes)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1.0 / self.num_classes)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        broadcast_targets = torch.broadcast_to(targets, size=[B, C, hparams.model.num_output_mixtures, H, W])
        log_probs = torch.where(broadcast_targets == self.min_pix_value, log_cdf_plus, torch.where(broadcast_targets == self.max_pix_value, log_one_minus_cdf_min, torch.where(cdf_delta > 1e-05, torch.log(torch.clamp(cdf_delta, min=1e-12)), log_pdf_mid - np.log(self.num_classes / 2))))
        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(logit_probs, dim=1)
        negative_log_probs = -torch.logsumexp(log_probs, dim=1)
        mean_axis = list(range(1, len(negative_log_probs.size())))
        per_example_loss = torch.sum(negative_log_probs, dim=mean_axis)
        avg_per_example_loss = per_example_loss / (np.prod([negative_log_probs.size()[i] for i in mean_axis]) * hparams.data.channels)
        assert len(per_example_loss.size()) == len(avg_per_example_loss.size()) == 1
        scalar = global_batch_size * hparams.data.target_res * hparams.data.target_res * hparams.data.channels
        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(2))
        return loss, avg_loss, model_means, log_scales


def calculate_logstd_loss(p, q):
    q_logstd = q[1]
    p_logstd = p[1]
    p_std = torch.exp(hparams.model.gradient_smoothing_beta * p_logstd)
    inv_q_std = torch.exp(-hparams.model.gradient_smoothing_beta * q_logstd)
    term1 = (p[0] - q[0]) * inv_q_std
    term2 = p_std * inv_q_std
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    return loss


@torch.jit.script
def calculate_std_loss(p: List[torch.Tensor], q: List[torch.Tensor]):
    q_std = q[1]
    p_std = p[1]
    term1 = (p[0] - q[0]) / q_std
    term2 = p_std / q_std
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    return loss


class KLDivergence(nn.Module):

    def forward(self, p, q, global_batch_size):
        if hparams.model.distribution_base == 'std':
            loss = calculate_std_loss(p, q)
        elif hparams.model.distribution_base == 'logstd':
            loss = calculate_logstd_loss(p, q)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')
        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])
        avg_per_example_loss = per_example_loss / n_mean_elems
        assert len(per_example_loss.shape) == 1
        scalar = global_batch_size * effective_pixels()
        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(2))
        return loss, avg_loss


class SSIM(torch.nn.Module):

    def __init__(self, image_channels, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        self.max_val = max_val
        self.k1 = k1
        self.k2 = k2
        self.filter_size = filter_size
        self.compensation = 1.0
        self.kernel = SSIM._fspecial_gauss(filter_size, filter_sigma, image_channels)

    @staticmethod
    def _fspecial_gauss(filter_size, filter_sigma, image_channels):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = torch.arange(0, filter_size, dtype=torch.float32)
        coords -= (filter_size - 1.0) / 2.0
        g = torch.square(coords)
        g *= -0.5 / np.square(filter_sigma)
        g = torch.reshape(g, shape=(1, -1)) + torch.reshape(g, shape=(-1, 1))
        g = torch.reshape(g, shape=(1, -1))
        g = F.softmax(g, dim=-1)
        g = torch.reshape(g, shape=(1, 1, filter_size, filter_size))
        return torch.tile(g, (image_channels, 1, 1, 1))

    def _apply_filter(self, x):
        shape = list(x.size())
        x = torch.reshape(x, shape=[-1] + shape[-3:])
        y = F.conv2d(x, weight=self.kernel, stride=1, padding=(self.filter_size - 1) // 2, groups=x.shape[1])
        return torch.reshape(y, shape[:-3] + list(y.size()[1:]))

    def _compute_luminance_contrast_structure(self, x, y):
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2
        mean0 = self._apply_filter(x)
        mean1 = self._apply_filter(y)
        num0 = mean0 * mean1 * 2.0
        den0 = torch.square(mean0) + torch.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)
        num1 = self._apply_filter(x * y) * 2.0
        den1 = self._apply_filter(torch.square(x) + torch.square(y))
        c2 *= self.compensation
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)
        return luminance, cs

    def _compute_one_channel_ssim(self, x, y):
        luminance, contrast_structure = self._compute_luminance_contrast_structure(x, y)
        return (luminance * contrast_structure).mean(dim=(-2, -1))

    def forward(self, targets, outputs):
        ssim_per_channel = self._compute_one_channel_ssim(targets, outputs)
        return ssim_per_channel.mean(dim=-1)


class StructureSimilarityIndexMap(nn.Module):

    def __init__(self, image_channels, unnormalized_max=255.0, filter_size=11):
        super(StructureSimilarityIndexMap, self).__init__()
        self.ssim = SSIM(image_channels=image_channels, max_val=unnormalized_max, filter_size=filter_size)

    def forward(self, targets, outputs, global_batch_size):
        if hparams.data.dataset_source == 'binarized_mnist':
            return 0.0
        targets = targets * 127.5 + 127.5
        outputs = outputs * 127.5 + 127.5
        assert targets.size() == outputs.size()
        per_example_ssim = self.ssim(targets, outputs)
        mean_axis = list(range(1, len(per_example_ssim.size())))
        per_example_ssim = torch.sum(per_example_ssim, dim=mean_axis)
        loss = torch.sum(per_example_ssim) / global_batch_size
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Interpolate,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoolLayer,
     lambda: ([], {'in_filters': 4, 'filters': 4, 'strides': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualConvCell,
     lambda: ([], {'n_layers': 1, 'in_filters': 4, 'bottleneck_ratio': 4, 'kernel_size': 4, 'init_scaler': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {'image_channels': 4, 'max_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Rayhane_mamah_Efficient_VDVAE(_paritybench_base):
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

