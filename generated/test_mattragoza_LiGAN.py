import sys
_module = sys.modules[__name__]
del sys
generate = _module
ligan = _module
atom_fitting = _module
atom_grids = _module
atom_structs = _module
atom_types = _module
bond_adding = _module
common = _module
data = _module
dkoes_fitting = _module
generating = _module
interpolation = _module
loss_fns = _module
metrics = _module
models = _module
molecules = _module
training = _module
isoslider = _module
pymol_util = _module
benchmark = _module
convert_checkpoint = _module
convert_types = _module
count_atom_types = _module
create_cond_datasets = _module
create_mutants = _module
interrupt = _module
memory = _module
sanitize_mol = _module
split_sdf = _module
valid_mols = _module
valid_novel_unique = _module
fit = _module
grid = _module
hang = _module
test_atom_fitting = _module
test_atom_grids = _module
test_atom_structs = _module
test_atom_types = _module
test_bond_adding = _module
test_data = _module
test_generating = _module
test_loss_fns = _module
test_models = _module
test_molecules = _module
test_training = _module
train = _module

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


import time


import numpy as np


import torch


import torch.nn.functional as F


from collections import namedtuple


from collections import defaultdict


from functools import lru_cache


import random


import re


import pandas as pd


from torch import nn


from torch import utils


import itertools


from collections import OrderedDict as odict


from math import pi


import scipy as sp


import scipy.optimize


from collections import OrderedDict


from scipy import stats


from functools import partial


from torch import optim


import copy


from numpy import isclose


from numpy import allclose


from numpy import inf


from numpy import isnan


import numpy as numpy


from numpy.linalg import norm


def lerp(v0, v1, t):
    """
    Linear interpolation between
    vectors v0 and v1 at steps t.
    """
    k0, k1 = 1 - t, t
    return k0 * v0 + k1 * v1


def slerp(v0, v1, t, center=None):
    """
    Spherical linear interpolation between
    vectors v0 and v1 at steps t.
    """
    eps = 1e-06
    if center is not None:
        v0 -= center
        v1 -= center
    norm_v0 = v0.norm(dim=1, keepdim=True)
    norm_v1 = v1.norm(dim=1, keepdim=True)
    dot_v0_v1 = (v0 * v1).sum(dim=1, keepdim=True)
    cos_theta = dot_v0_v1 / (norm_v0 * norm_v1)
    theta = torch.acos(cos_theta) + eps
    sin_theta = torch.sin(theta)
    k0 = torch.sin((1 - t) * theta) / sin_theta
    k1 = torch.sin(t * theta) / sin_theta
    if center is not None:
        return k0 * v0 + k1 * v1 + center
    else:
        return k0 * v0 + k1 * v1


class Interpolation(torch.nn.Module):

    def __init__(self, n_samples):
        super().__init__()
        self.endpoints = None
        self.center = None
        self.n_samples = n_samples
        self.curr_step = 0

    @property
    def is_initialized(self):
        return self.endpoints is not None

    def initialize(self, init_point, center=None):
        assert not self.is_initialized
        self.endpoints = init_point.unsqueeze(0)
        if center is not None:
            self.center = center.unsqueeze(0)

    def forward(self, inputs, spherical=False):
        assert len(inputs.shape) == 2, 'inputs must be vectors'
        batch_size = inputs.shape[0]
        batch_idx = torch.arange(batch_size, device=inputs.device)
        is_endpoint = (self.curr_step + batch_idx) % self.n_samples == 0
        self.endpoints = torch.cat([self.endpoints, inputs[is_endpoint]])
        start_idx = (self.curr_step + batch_idx) // self.n_samples
        start_points = self.endpoints[start_idx]
        stop_points = self.endpoints[start_idx + 1]
        k_interp = ((self.curr_step + batch_idx) % self.n_samples + 1).unsqueeze(1) / self.n_samples
        if spherical:
            outputs = slerp(start_points, stop_points, k_interp, self.center)
        else:
            outputs = lerp(start_points, stop_points, k_interp)
        assert not outputs.isnan().any()
        self.curr_step += batch_size
        return outputs


class TransformInterpolation(Interpolation):

    def initialize(self, example):
        rec_coord_set, lig_coord_set = example.coord_sets
        rec_center = tuple(rec_coord_set.center())
        lig_center = tuple(lig_coord_set.center())
        super().initialize(init_point=torch.as_tensor(lig_center), center=torch.as_tensor(rec_center))

    def forward(self, transforms, **kwargs):
        centers = torch.tensor([tuple(t.get_rotation_center()) for t in transforms], dtype=float)
        centers = super().forward(centers, **kwargs)
        return [molgrid.Transform(t.get_quaternion(), tuple(center.numpy()), t.get_translation()) for t, center in zip(transforms, centers)]


def wasserstein_loss(predictions, labels):
    labels = 2 * labels - 1
    return (labels * predictions).sum() / labels.shape[0]


def get_gan_loss_fn(type):
    assert type in {'x', 'w'}, type
    if type == 'w':
        return wasserstein_loss
    else:
        return torch.nn.BCEWithLogitsLoss()


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars_ = stds * stds
    return (-log_stds + means2 / 2 + vars_ / 2 - 0.5).sum() / means.shape[0]


def get_kldiv_loss_fn(type):
    assert type == 'k', type
    return kl_divergence


def get_loss_schedule(start_wt, start_iter=500000, end_wt=None, period=100000, type='d'):
    """
    Return a function that takes the
    iteration as input and returns a
    modulated loss weight as output.
    """
    no_schedule = type == 'n' or end_wt is None or end_wt == start_wt
    assert no_schedule or period > 0, period
    assert type in {'n', 'd', 'c', 'r'}, type
    periodic = type == 'c ' or type == 'r'
    restart = type == 'r'
    end_iter = start_iter + period

    def loss_schedule(iteration, use_loss_wt):
        if not use_loss_wt:
            return torch.tensor(1)
        if no_schedule or iteration < start_iter:
            return torch.as_tensor(start_wt)
        if iteration >= end_iter and not periodic:
            return torch.as_tensor(end_wt)
        wt_range = end_wt - start_wt
        theta = (iteration - start_iter) / period * pi
        if restart:
            theta %= pi
        return end_wt - wt_range * 0.5 * (1 + torch.cos(theta))
    return loss_schedule, end_wt


def L1_loss(predictions, labels, log_var=0):
    return torch.sum(((labels - predictions) / torch.exp(log_var)).abs() + log_var) / labels.shape[0]


def L2_loss(predictions, labels, log_var=0):
    return torch.sum(((labels - predictions) / torch.exp(log_var)) ** 2 / 2.0 + log_var) / labels.shape[0]


def get_recon_loss_fn(type):
    assert type in {'1', '2'}, type
    if type == '1':
        return L1_loss
    else:
        return L2_loss


def product_loss(rec_grids, lig_grids):
    """
    Minimize receptor-ligand overlap
    by summing the pointwise products
    of total density at each point.
    """
    return (rec_grids.sum(dim=1) * lig_grids.clamp(min=0).sum(dim=1)).sum() / lig_grids.shape[0]


def get_steric_loss_fn(type):
    assert type == 'p', type
    return product_loss


has_both = lambda a, b: a is not None and b is not None


class LossFunction(nn.Module):
    """
    A multi-task loss function for training
    generative models of atomic density grids.

    The loss function combines KL divergence,
    reconstruction, GAN discrimination, and/or
    receptor-ligand steric clash.

    Each term can have a different loss weight
    associated with it, and different types of
    loss functions are available for each term.

    The different loss terms are computed based
    on the input provided to the forward method.
    """

    def __init__(self, types, weights, schedules={}, device='cuda'):
        super().__init__()
        self.init_loss_weights(**weights)
        self.init_loss_types(**types)
        self.init_loss_schedules(**schedules)
        self.device = device

    def init_loss_weights(self, kldiv_loss=0, recon_loss=0, gan_loss=0, steric_loss=0, kldiv2_loss=0, recon2_loss=0):
        self.kldiv_loss_wt = float(kldiv_loss)
        self.recon_loss_wt = float(recon_loss)
        self.gan_loss_wt = float(gan_loss)
        self.steric_loss_wt = float(steric_loss)
        self.kldiv2_loss_wt = float(kldiv2_loss)
        self.recon2_loss_wt = float(recon2_loss)

    def init_loss_types(self, kldiv_loss='k', recon_loss='2', gan_loss='x', steric_loss='p', kldiv2_loss='k', recon2_loss='2'):
        self.kldiv_loss_fn = get_kldiv_loss_fn(kldiv_loss)
        self.recon_loss_fn = get_recon_loss_fn(recon_loss)
        self.gan_loss_fn = get_gan_loss_fn(gan_loss)
        self.steric_loss_fn = get_steric_loss_fn(steric_loss)
        self.kldiv2_loss_fn = get_kldiv_loss_fn(kldiv2_loss)
        self.recon2_loss_fn = get_recon_loss_fn(recon2_loss)

    def init_loss_schedules(self, kldiv_loss={}, recon_loss={}, gan_loss={}, steric_loss={}, kldiv2_loss={}, recon2_loss={}):
        self.kldiv_loss_schedule, _ = get_loss_schedule(start_wt=self.kldiv_loss_wt, **kldiv_loss)
        self.recon_loss_schedule, _ = get_loss_schedule(start_wt=self.recon_loss_wt, **recon_loss)
        self.gan_loss_schedule, self.end_gan_loss_wt = get_loss_schedule(start_wt=self.gan_loss_wt, **gan_loss)
        self.steric_loss_schedule, self.end_steric_loss_wt = get_loss_schedule(start_wt=self.steric_loss_wt, **steric_loss)
        self.kldiv2_loss_schedule, _ = get_loss_schedule(start_wt=self.kldiv2_loss_wt, **kldiv2_loss)
        self.recon2_loss_schedule, _ = get_loss_schedule(start_wt=self.recon2_loss_wt, **recon2_loss)

    @property
    def has_prior_loss(self):
        """
        Whether the loss function ever has
        non-zero value on prior samples.
        """
        return bool(self.gan_loss_wt or self.end_gan_loss_wt or self.steric_loss_wt or self.end_steric_loss_wt)

    def forward(self, latent_means=None, latent_log_stds=None, lig_grids=None, lig_gen_grids=None, disc_labels=None, disc_preds=None, rec_grids=None, rec_lig_grids=None, latent2_means=None, latent2_log_stds=None, real_latents=None, gen_latents=None, gen_log_var=torch.zeros(1), prior_log_var=torch.zeros(1), use_loss_wt=True, iteration=0):
        """
        Computes the loss as follows:

        = kldiv_loss_wt * 
            kldiv_loss_fn(latent_means, latent_log_stds)
        + recon_loss_wt * 
            recon_loss_fn(lig_gen_grids, lig_grids)
        + gan_loss_wt * 
            gan_loss_fn(disc_preds, disc_labels)
        + steric_loss_wt * 
            steric_loss_fn(rec_grids, rec_lig_grids)
        + ...

        Each term is computed iff both of its inputs are
        provided to the method, and each computed term is
        also returned as values in an OrderedDict.
        """
        loss = torch.zeros(1, device=self.device)
        losses = odict()
        if has_both(lig_grids, lig_gen_grids):
            recon_loss = self.recon_loss_fn(lig_gen_grids, lig_grids, gen_log_var)
            recon_loss_wt = self.recon_loss_schedule(iteration, use_loss_wt)
            loss += recon_loss_wt * recon_loss
            losses['recon_loss'] = recon_loss.item()
            losses['recon_loss_wt'] = recon_loss_wt.item()
            losses['recon_log_var'] = gen_log_var.item()
        if has_both(latent_means, latent_log_stds):
            kldiv_loss = self.kldiv_loss_fn(latent_means, latent_log_stds)
            kldiv_loss_wt = self.kldiv_loss_schedule(iteration, use_loss_wt)
            loss += kldiv_loss_wt * kldiv_loss
            losses['kldiv_loss'] = kldiv_loss.item()
            losses['kldiv_loss_wt'] = kldiv_loss_wt.item()
        if has_both(disc_labels, disc_preds):
            gan_loss = self.gan_loss_fn(disc_preds, disc_labels)
            gan_loss_wt = self.gan_loss_schedule(iteration, use_loss_wt)
            loss += gan_loss_wt * gan_loss
            losses['gan_loss'] = gan_loss.item()
            losses['gan_loss_wt'] = gan_loss_wt.item()
        if has_both(rec_grids, rec_lig_grids):
            steric_loss = self.steric_loss_fn(rec_grids, rec_lig_grids)
            steric_loss_wt = self.steric_loss_schedule(iteration, use_loss_wt)
            loss += steric_loss_wt * steric_loss
            losses['steric_loss'] = steric_loss.item()
            losses['steric_loss_wt'] = steric_loss_wt.item()
        if has_both(latent2_means, latent2_log_stds):
            kldiv2_loss = self.kldiv2_loss_fn(latent2_means, latent2_log_stds)
            kldiv2_loss_wt = self.kldiv2_loss_schedule(iteration, use_loss_wt)
            loss += kldiv2_loss_wt * kldiv2_loss
            losses['kldiv2_loss'] = kldiv2_loss.item()
            losses['kldiv2_loss_wt'] = kldiv2_loss_wt.item()
        if has_both(real_latents, gen_latents):
            recon2_loss = self.recon2_loss_fn(gen_latents, real_latents, prior_log_var)
            recon2_loss_wt = self.recon2_loss_schedule(iteration, use_loss_wt)
            loss += recon2_loss_wt * recon2_loss
            losses['recon2_loss'] = recon2_loss.item()
            losses['recon2_loss_wt'] = recon2_loss_wt.item()
            losses['recon2_log_var'] = prior_log_var.item()
        losses['loss'] = loss.item()
        return loss, losses


class Conv3DReLU(nn.Sequential):
    """
    A 3D convolutional layer followed by leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    """
    conv_type = nn.Conv3d

    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, relu_leak=0.1, batch_norm=False, spectral_norm=False):
        modules = [self.conv_type(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size, padding=kernel_size // 2), nn.LeakyReLU(negative_slope=relu_leak, inplace=True)]
        if batch_norm > 0:
            modules.insert(batch_norm, nn.BatchNorm3d(n_channels_out))
        if spectral_norm > 0:
            modules[0] = nn.utils.spectral_norm(modules[0], n_power_iterations=spectral_norm)
        super().__init__(*modules)


class TConv3DReLU(Conv3DReLU):
    """
    A 3D transposed convolution layer and leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    """
    conv_type = nn.ConvTranspose3d


class Conv3DBlock(nn.Module):
    """
    A sequence of n_convs ConvReLUs with the same settings.
    """
    conv_type = Conv3DReLU

    def __init__(self, n_convs, n_channels_in, n_channels_out, block_type='c', growth_rate=8, bottleneck_factor=0, debug=False, **kwargs):
        super().__init__()
        assert block_type in {'c', 'r', 'd'}, block_type
        self.residual = block_type == 'r'
        self.dense = block_type == 'd'
        if self.residual:
            self.init_skip_conv(n_channels_in=n_channels_in, n_channels_out=n_channels_out, **kwargs)
        if self.dense:
            self.init_final_conv(n_channels_in=n_channels_in, n_convs=n_convs, growth_rate=growth_rate, n_channels_out=n_channels_out, **kwargs)
            n_channels_out = growth_rate
        self.init_conv_sequence(n_convs=n_convs, n_channels_in=n_channels_in, n_channels_out=n_channels_out, bottleneck_factor=bottleneck_factor, **kwargs)

    def init_skip_conv(self, n_channels_in, n_channels_out, kernel_size, **kwargs):
        if n_channels_out != n_channels_in:
            self.skip_conv = self.conv_type(n_channels_in=n_channels_in, n_channels_out=n_channels_out, kernel_size=1, **kwargs)
        else:
            self.skip_conv = nn.Identity()

    def init_final_conv(self, n_channels_in, n_convs, growth_rate, n_channels_out, kernel_size, **kwargs):
        self.final_conv = self.conv_type(n_channels_in=n_channels_in + n_convs * growth_rate, n_channels_out=n_channels_out, kernel_size=1, **kwargs)

    def bottleneck_conv(self, n_channels_in, n_channels_bn, n_channels_out, kernel_size, **kwargs):
        assert n_channels_bn > 0, (n_channels_in, n_channels_bn, n_channels_out)
        return nn.Sequential(self.conv_type(n_channels_in=n_channels_in, n_channels_out=n_channels_bn, kernel_size=1, **kwargs), self.conv_type(n_channels_in=n_channels_bn, n_channels_out=n_channels_bn, kernel_size=kernel_size, **kwargs), self.conv_type(n_channels_in=n_channels_bn, n_channels_out=n_channels_out, kernel_size=1, **kwargs))

    def init_conv_sequence(self, n_convs, n_channels_in, n_channels_out, bottleneck_factor, **kwargs):
        self.conv_modules = []
        for i in range(n_convs):
            if bottleneck_factor:
                conv = self.bottleneck_conv(n_channels_in=n_channels_in, n_channels_bn=n_channels_in // bottleneck_factor, n_channels_out=n_channels_out, **kwargs)
            else:
                conv = self.conv_type(n_channels_in=n_channels_in, n_channels_out=n_channels_out, **kwargs)
            self.conv_modules.append(conv)
            self.add_module(str(i), conv)
            if self.dense:
                n_channels_in += n_channels_out
            else:
                n_channels_in = n_channels_out

    def __len__(self):
        return len(self.conv_modules)

    def forward(self, inputs):
        if not self.conv_modules:
            return inputs
        if self.dense:
            all_inputs = [inputs]
        for i, f in enumerate(self.conv_modules):
            if self.residual:
                identity = self.skip_conv(inputs) if i == 0 else inputs
                outputs = f(inputs) + identity
            else:
                outputs = f(inputs)
            if self.dense:
                all_inputs.append(outputs)
                inputs = torch.cat(all_inputs, dim=1)
            else:
                inputs = outputs
        if self.dense:
            outputs = self.final_conv(inputs)
        return outputs


class TConv3DBlock(Conv3DBlock):
    """
    A sequence of n_convs TConvReLUs with the same settings.
    """
    conv_type = TConv3DReLU


class Pool3D(nn.Sequential):
    """
    A layer that decreases 3D spatial dimensions,
    either by max pooling (pool_type=m), average
    pooling (pool_type=a), or strided convolution
    (pool_type=c).
    """

    def __init__(self, n_channels, pool_type, pool_factor):
        if pool_type == 'm':
            pool = nn.MaxPool3d(kernel_size=pool_factor, stride=pool_factor)
        elif pool_type == 'a':
            pool = nn.AvgPool3d(kernel_size=pool_factor, stride=pool_factor)
        elif pool_type == 'c':
            pool = nn.Conv3d(in_channels=n_channels, out_channels=n_channels, groups=n_channels, kernel_size=pool_factor, stride=pool_factor)
        else:
            raise ValueError('unknown pool_type ' + repr(pool_type))
        super().__init__(pool)


unpool_type_map = dict(n='nearest', t='trilinear')


class Unpool3D(nn.Sequential):
    """
    A layer that increases the 3D spatial dimensions,
    either by nearest neighbor (unpool_type=n), tri-
    linear interpolation (unpool_type=t), or strided
    transposed convolution (unpool_type=c).
    """

    def __init__(self, n_channels, unpool_type, unpool_factor):
        if unpool_type in unpool_type_map:
            unpool = nn.Upsample(scale_factor=unpool_factor, mode=unpool_type_map[unpool_type])
        elif unpool_type == 'c':
            unpool = nn.ConvTranspose3d(in_channels=n_channels, out_channels=n_channels, groups=n_channels, kernel_size=unpool_factor, stride=unpool_factor)
        else:
            raise ValueError('unknown unpool_type ' + repr(unpool_type))
        super().__init__(unpool)


class Reshape(nn.Module):
    """
    A layer that reshapes the input.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def __repr__(self):
        return 'Reshape(shape={})'.format(self.shape)

    def forward(self, x):
        return x.reshape(self.shape)


class Grid2Vec(nn.Sequential):
    """
    A fully connected layer applied to a
    flattened version of the input, for
    transforming from grids to vectors.
    """

    def __init__(self, in_shape, n_output, activ_fn=None, spectral_norm=0):
        n_input = np.prod(in_shape)
        modules = [Reshape(shape=(-1, n_input)), nn.Linear(n_input, n_output)]
        if activ_fn:
            modules.append(activ_fn)
        if spectral_norm > 0:
            modules[1] = nn.utils.spectral_norm(modules[1], n_power_iterations=spectral_norm)
        super().__init__(*modules)


class Vec2Grid(nn.Sequential):
    """
    A fully connected layer followed by
    reshaping the output, for transforming
    from vectors to grids.
    """

    def __init__(self, n_input, out_shape, relu_leak, batch_norm, spectral_norm):
        n_output = np.prod(out_shape)
        modules = [nn.Linear(n_input, n_output), Reshape(shape=(-1, *out_shape)), nn.LeakyReLU(negative_slope=relu_leak, inplace=True)]
        if batch_norm > 0:
            modules.insert(batch_norm + 1, nn.BatchNorm3d(out_shape[0]))
        if spectral_norm > 0:
            modules[0] = nn.utils.spectral_norm(modules[0], n_power_iterations=spectral_norm)
        super().__init__(*modules)


def as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def reduce_list(obj):
    return obj[0] if isinstance(obj, list) and len(obj) == 1 else obj


class GridEncoder(nn.Module):
    """
    A sequence of 3D convolution blocks and
    pooling layers, followed by one or more
    fully connected output tasks.
    """

    def __init__(self, n_channels, grid_size=48, n_filters=32, width_factor=2, n_levels=4, conv_per_level=3, kernel_size=3, relu_leak=0.1, batch_norm=0, spectral_norm=0, pool_type='a', pool_factor=2, n_output=1, output_activ_fn=None, init_conv_pool=False, block_type='c', growth_rate=8, bottleneck_factor=0, debug=False):
        super().__init__()
        self.debug = debug
        self.grid_modules = []
        self.n_channels = n_channels
        self.grid_size = grid_size
        if init_conv_pool:
            self.add_conv3d(name='init_conv', n_filters=n_filters, kernel_size=kernel_size + 2, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm)
            self.add_pool3d(name='init_pool', pool_type=pool_type, pool_factor=pool_factor)
            n_filters *= width_factor
        for i in range(n_levels):
            if i > 0:
                self.add_pool3d(name='level' + str(i) + '_pool', pool_type=pool_type, pool_factor=pool_factor)
                n_filters *= width_factor
            self.add_conv3d_block(name='level' + str(i), n_convs=conv_per_level, n_filters=n_filters, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, debug=debug)
        n_output = as_list(n_output)
        assert n_output and all(n_o > 0 for n_o in n_output)
        output_activ_fn = as_list(output_activ_fn)
        if len(output_activ_fn) == 1:
            output_activ_fn *= len(n_output)
        assert len(output_activ_fn) == len(n_output)
        self.n_tasks = len(n_output)
        self.task_modules = []
        for i, (n_output_i, activ_fn_i) in enumerate(zip(n_output, output_activ_fn)):
            self.add_grid2vec(name='fc' + str(i), n_output=n_output_i, activ_fn=activ_fn_i, spectral_norm=spectral_norm)

    def print(self, *args, **kwargs):
        if self.debug:
            None

    def add_conv3d(self, name, n_filters, **kwargs):
        conv = Conv3DReLU(n_channels_in=self.n_channels, n_channels_out=n_filters, **kwargs)
        self.add_module(name, conv)
        self.grid_modules.append(conv)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_pool3d(self, name, pool_factor, **kwargs):
        assert self.grid_size % pool_factor == 0, 'cannot pool remaining spatial dims ({} % {})'.format(self.grid_size, pool_factor)
        pool = Pool3D(n_channels=self.n_channels, pool_factor=pool_factor, **kwargs)
        self.add_module(name, pool)
        self.grid_modules.append(pool)
        self.grid_size //= pool_factor
        self.print(name, self.n_channels, self.grid_size)

    def add_conv3d_block(self, name, n_filters, **kwargs):
        conv_block = Conv3DBlock(n_channels_in=self.n_channels, n_channels_out=n_filters, **kwargs)
        self.add_module(name, conv_block)
        self.grid_modules.append(conv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_grid2vec(self, name, **kwargs):
        fc = Grid2Vec(in_shape=(self.n_channels,) + (self.grid_size,) * 3, **kwargs)
        self.add_module(name, fc)
        self.task_modules.append(fc)
        self.print(name, self.n_channels, self.grid_size)

    def forward(self, inputs):
        conv_features = []
        for f in self.grid_modules:
            outputs = f(inputs)
            self.print(inputs.shape, '->', f, '->', outputs.shape)
            if not isinstance(f, Pool3D):
                conv_features.append(outputs)
            inputs = outputs
        outputs = [f(inputs) for f in self.task_modules]
        outputs_shape = [o.shape for o in outputs]
        self.print(inputs.shape, '->', self.task_modules, '->', outputs_shape)
        return reduce_list(outputs), conv_features


class Discriminator(GridEncoder):
    pass


class GridDecoder(nn.Module):
    """
    A fully connected layer followed by a
    sequence of 3D transposed convolution
    blocks and unpooling layers.
    """

    def __init__(self, n_input, grid_size, n_channels, width_factor, n_levels, tconv_per_level, kernel_size, relu_leak, batch_norm, spectral_norm, unpool_type, unpool_factor, n_channels_out, final_unpool=False, skip_connect=False, block_type='c', growth_rate=8, bottleneck_factor=0, debug=False):
        super().__init__()
        self.skip_connect = bool(skip_connect)
        self.debug = debug
        self.fc_modules = []
        self.n_input = n_input
        self.add_vec2grid(name='fc', n_input=n_input, n_channels=n_channels, grid_size=grid_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm)
        n_filters = n_channels
        self.grid_modules = []
        for i in reversed(range(n_levels)):
            if i + 1 < n_levels:
                unpool_name = 'level' + str(i) + '_unpool'
                self.add_unpool3d(name=unpool_name, unpool_type=unpool_type, unpool_factor=unpool_factor)
                n_filters //= width_factor
            if skip_connect:
                n_skip_channels = self.n_channels
                if i < n_levels - 1:
                    n_skip_channels //= width_factor
            else:
                n_skip_channels = 0
            tconv_block_name = 'level' + str(i)
            self.add_tconv3d_block(name=tconv_block_name, n_convs=tconv_per_level, n_filters=n_filters, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, n_skip_channels=n_skip_channels, debug=debug)
        if final_unpool:
            self.add_unpool3d(name='final_unpool', unpool_type=unpool_type, unpool_factor=unpool_factor)
            n_skip_channels //= width_factor
            self.add_tconv3d_block(name='final_conv', n_convs=tconv_per_level, n_filters=n_channels_out, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, n_skip_channels=n_skip_channels, debug=debug)
        else:
            self.add_tconv3d(name='final_conv', n_filters=n_channels_out, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm)

    def print(self, *args, **kwargs):
        if self.debug:
            None

    def add_vec2grid(self, name, n_channels, grid_size, **kwargs):
        vec2grid = Vec2Grid(out_shape=(n_channels,) + (grid_size,) * 3, **kwargs)
        self.add_module(name, vec2grid)
        self.fc_modules.append(vec2grid)
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.print(name, self.n_channels, self.grid_size)

    def add_unpool3d(self, name, unpool_factor, **kwargs):
        unpool = Unpool3D(n_channels=self.n_channels, unpool_factor=unpool_factor, **kwargs)
        self.add_module(name, unpool)
        self.grid_modules.append(unpool)
        self.grid_size *= unpool_factor
        self.print(name, self.n_channels, self.grid_size)

    def add_tconv3d(self, name, n_filters, **kwargs):
        tconv = TConv3DReLU(n_channels_in=self.n_channels, n_channels_out=n_filters, **kwargs)
        self.add_module(name, tconv)
        self.grid_modules.append(tconv)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_tconv3d_block(self, name, n_filters, n_skip_channels, **kwargs):
        tconv_block = TConv3DBlock(n_channels_in=self.n_channels + n_skip_channels, n_channels_out=n_filters, **kwargs)
        self.add_module(name, tconv_block)
        self.grid_modules.append(tconv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def forward(self, inputs, skip_features=None):
        for f in self.fc_modules:
            outputs = f(inputs)
            self.print(inputs.shape, '->', f, '->', outputs.shape)
            inputs = outputs
        for f in self.grid_modules:
            if self.skip_connect and isinstance(f, TConv3DBlock):
                skip_inputs = skip_features.pop()
                inputs = torch.cat([inputs, skip_inputs], dim=1)
                inputs_shape = [inputs.shape, skip_inputs.shape]
            else:
                inputs_shape = inputs.shape
            outputs = f(inputs)
            self.print(inputs_shape, '->', f, '->', outputs.shape)
            inputs = outputs
        return outputs


def is_positive_int(x):
    return isinstance(x, int) and x > 0


class GridGenerator(nn.Sequential):
    """
    A generative model of 3D grids that can take the form
    of an encoder-decoder architecture (e.g. AE, VAE) or
    a decoder-only architecture (e.g. GAN). The model can
    also have a conditional encoder (e.g. CE, CVAE, CGAN).
    """
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = False
    has_stage2 = False

    def __init__(self, n_channels_in=None, n_channels_cond=None, n_channels_out=19, grid_size=48, n_filters=32, width_factor=2, n_levels=4, conv_per_level=3, kernel_size=3, relu_leak=0.1, batch_norm=0, spectral_norm=0, pool_type='a', unpool_type='n', pool_factor=2, n_latent=1024, init_conv_pool=False, skip_connect=False, block_type='c', growth_rate=8, bottleneck_factor=0, n_samples=0, device='cuda', debug=False):
        assert type(self) != GridGenerator, 'GridGenerator is abstract'
        self.debug = debug
        super().__init__()
        self.check_encoder_channels(n_channels_in, n_channels_cond)
        assert is_positive_int(n_channels_out)
        assert is_positive_int(n_latent)
        self.n_channels_in = n_channels_in
        self.n_channels_cond = n_channels_cond
        self.n_channels_out = n_channels_out
        self.n_latent = n_latent
        if self.has_input_encoder:
            if self.is_variational:
                encoder_output = [n_latent, n_latent]
            else:
                encoder_output = n_latent
            self.input_encoder = GridEncoder(n_channels=n_channels_in, grid_size=grid_size, n_filters=n_filters, width_factor=width_factor, n_levels=n_levels, conv_per_level=conv_per_level, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, pool_type=pool_type, pool_factor=pool_factor, n_output=encoder_output, init_conv_pool=init_conv_pool, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, debug=debug)
        if self.has_conditional_encoder:
            self.conditional_encoder = GridEncoder(n_channels=n_channels_cond, grid_size=grid_size, n_filters=n_filters, width_factor=width_factor, n_levels=n_levels, conv_per_level=conv_per_level, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, pool_type=pool_type, pool_factor=pool_factor, n_output=n_latent, init_conv_pool=init_conv_pool, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, debug=debug)
        n_pools = n_levels - 1 + init_conv_pool
        self.decoder = GridDecoder(n_input=self.n_decoder_input, grid_size=grid_size // pool_factor ** n_pools, n_channels=n_filters * width_factor ** n_pools, width_factor=width_factor, n_levels=n_levels, tconv_per_level=conv_per_level, kernel_size=kernel_size, relu_leak=relu_leak, batch_norm=batch_norm, spectral_norm=spectral_norm, unpool_type=unpool_type, unpool_factor=pool_factor, n_channels_out=n_channels_out, final_unpool=init_conv_pool, skip_connect=skip_connect, block_type=block_type, growth_rate=growth_rate, bottleneck_factor=bottleneck_factor, debug=debug)
        self.latent_interp = Interpolation(n_samples=n_samples)
        super()
        self.device = device

    def check_encoder_channels(self, n_channels_in, n_channels_cond):
        if self.has_input_encoder:
            assert is_positive_int(n_channels_in), n_channels_in
        else:
            assert n_channels_in is None, n_channels_in
        if self.has_conditional_encoder:
            assert is_positive_int(n_channels_cond), n_channels_cond
        else:
            assert n_channels_cond is None, n_channels_cond

    @property
    def n_decoder_input(self):
        n = 0
        if self.has_input_encoder or self.is_variational:
            n += self.n_latent
        if self.has_conditional_encoder:
            n += self.n_latent
        return n

    def sample_latent(self, batch_size, means=None, log_stds=None, interpolate=False, spherical=False, **kwargs):
        latent_vecs = sample_latent(batch_size=batch_size, n_latent=self.n_latent, means=means, log_stds=log_stds, device=self.device, **kwargs)
        if interpolate:
            if not self.latent_interp.is_initialized:
                self.latent_interp.initialize(sample_latent(batch_size=1, n_latent=self.n_latent, device=self.device, **kwargs)[0])
            latent_vecs = self.latent_interp(latent_vecs, spherical=spherical)
        return latent_vecs


class AE(GridGenerator):
    is_variational = False
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):
        if inputs is None:
            in_latents = self.sample_latent(batch_size)
        else:
            in_latents, _ = self.input_encoder(inputs)
        outputs = self.decoder(inputs=in_latents)
        return outputs, in_latents, None, None


class VAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):
        if inputs is None:
            means, log_stds = None, None
        else:
            (means, log_stds), _ = self.input_encoder(inputs)
        var_latents = self.sample_latent(batch_size, means, log_stds)
        outputs = self.decoder(inputs=var_latents)
        return outputs, var_latents, means, log_stds


class CE(GridGenerator):
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        cond_latents, cond_features = self.conditional_encoder(conditions)
        outputs = self.decoder(inputs=cond_latents, skip_features=cond_features)
        return outputs, cond_latents, None, None


class CVAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None, **kwargs):
        if inputs is None:
            means, log_stds = None, None
        else:
            (means, log_stds), _ = self.input_encoder(inputs)
        in_latents = self.sample_latent(batch_size, means, log_stds, **kwargs)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([in_latents, cond_latents], dim=1)
        outputs = self.decoder(inputs=cat_latents, skip_features=cond_features)
        return outputs, in_latents, means, log_stds


class GAN(GridGenerator):
    is_variational = True
    has_input_encoder = False
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latent(batch_size)
        outputs = self.decoder(inputs=var_latents)
        return outputs, var_latents, None, None


class CGAN(GridGenerator):
    is_variational = True
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latent(batch_size)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([var_latents, cond_latents], dim=1)
        outputs = self.decoder(inputs=cat_latents, skip_features=cond_features)
        return outputs, var_latents, None, None


class VAE2(VAE):
    has_stage2 = True
    """
    This is a module that allows insertion of
    a prior model, aka stage 2 VAE, into an
    existing VAE model, aka a two-stage VAE.
    """

    def forward2(self, prior_model, inputs=None, conditions=None, batch_size=None, **kwargs):
        if inputs is None:
            var_latents = means = log_stds = None
        else:
            (means, log_stds), _ = self.input_encoder(inputs)
            var_latents = self.sample_latent(batch_size, means, log_stds, **kwargs)
        gen_latents, _, means2, log_stds2 = prior_model(inputs=var_latents, batch_size=batch_size)
        outputs = self.decoder(inputs=gen_latents)
        return outputs, var_latents, means, log_stds, gen_latents, means2, log_stds2


class CVAE2(CVAE):
    has_stage2 = True
    """
    Two-stage CVAE.
    """

    def forward2(self, prior_model, inputs=None, conditions=None, batch_size=None, **kwargs):
        if inputs is None:
            in_latents = means = log_stds = None
        else:
            (means, log_stds), _ = self.input_encoder(inputs)
            in_latents = self.sample_latent(batch_size, means, log_stds, **kwargs)
        gen_latents, _, means2, log_stds2 = prior_model(inputs=in_latents, batch_size=batch_size)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([gen_latents, cond_latents], dim=1)
        outputs = self.decoder(inputs=cat_latents, skip_features=cond_features)
        return outputs, in_latents, means, log_stds, gen_latents, means2, log_stds2


class Stage2VAE(nn.Module):

    def __init__(self, n_input, n_h_layers, n_h_units, n_latent, relu_leak=0.1, device='cuda'):
        super().__init__()
        n_inputs = []
        modules = []
        for i in range(n_h_layers):
            modules.append(nn.Linear(n_input, n_h_units))
            modules.append(nn.LeakyReLU(negative_slope=relu_leak))
            n_inputs.append(n_input)
            n_input = n_h_units
        self.encoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(n_input, n_latent)
        self.fc_log_std = nn.Linear(n_input, n_latent)
        modules = [nn.Linear(n_latent, n_input), nn.LeakyReLU(negative_slope=relu_leak)]
        for n_output in reversed(n_inputs):
            modules.append(nn.Linear(n_input, n_output))
            modules.append(nn.LeakyReLU(negative_slope=relu_leak))
            n_input = n_output
        self.decoder = nn.Sequential(*modules)
        self.n_latent = n_latent
        self.device = device

    def forward(self, inputs=None, batch_size=None):
        if inputs is None:
            means, log_stds = None, None
        else:
            enc_outputs = self.encoder(inputs)
            means = self.fc_mean(enc_outputs)
            log_stds = self.fc_log_std(enc_outputs)
        var_latents = self.sample_latent(batch_size, means, log_stds)
        outputs = self.decoder(var_latents)
        return outputs, var_latents, means, log_stds

    def sample_latent(self, batch_size, means=None, log_stds=None, **kwargs):
        return sample_latent(batch_size=batch_size, n_latent=self.n_latent, means=means, log_stds=log_stds, device=self.device, **kwargs)


MB = 1024 ** 2


def compute_grid_variance(grids):
    mean_grid = grids.detach().mean(dim=0)
    return (((grids.detach() - mean_grid) ** 2).sum() / grids.shape[0]).item()


def compute_mean_grid_norm(grids):
    dim = tuple(range(1, grids.ndim))
    return grids.detach().norm(p=2, dim=dim).mean().item()


def compute_grid_metrics(grid_type, grids):
    m = OrderedDict()
    m[grid_type + '_norm'] = compute_mean_grid_norm(grids)
    m[grid_type + '_variance'] = compute_grid_variance(grids)
    return m


def compute_L2_loss(grids, ref_grids):
    return ((ref_grids.detach() - grids.detach()) ** 2).sum().item() / 2 / grids.shape[0]


def compute_paired_grid_metrics(grid_type, grids, ref_grid_type, ref_grids):
    m = compute_grid_metrics(ref_grid_type, ref_grids)
    m.update(compute_grid_metrics(grid_type, grids))
    m[grid_type + '_L2_loss'] = compute_L2_loss(grids, ref_grids)
    return m


def compute_min_rmsd(coords1, types1, coords2, types2):
    """
    Compute an RMSD between two sets of positions of the same
    atom types with no prior mapping between particular atom
    positions of a given type. Returns the minimum RMSD across
    all permutations of this mapping.
    """
    n1, n2 = len(coords1), len(coords2)
    assert n1 == n2, 'structs must have same num atoms ({} vs. {})'.format(n1, n2)
    n_atoms = len(coords1)
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    types1 = np.array(types1)
    types2 = np.array(types2)
    type_counts1 = types1.sum(axis=0)
    type_counts2 = types2.sum(axis=0)
    type_diff = np.abs(type_counts1 - type_counts2).sum()
    assert (type_counts1 == type_counts2).all(), 'structs must have same type counts ({:.2f})'.format(type_diff)
    ssd = 0.0
    nax = np.newaxis
    for t in np.unique(types1, axis=0):
        coords1_t = coords1[(types1 == t[nax, :]).all(axis=1)]
        coords2_t = coords2[(types2 == t[nax, :]).all(axis=1)]
        assert len(coords1_t) == len(coords2_t), 'structs must have same num atoms of each type'
        dist2_t = ((coords1_t[:, nax, :] - coords2_t[nax, :, :]) ** 2).sum(axis=2)
        idx1, idx2 = sp.optimize.linear_sum_assignment(dist2_t)
        ssd += dist2_t[idx1, idx2].sum()
    return np.sqrt(ssd / n_atoms)


def compute_struct_rmsd(struct1, struct2, catch_exc=True):
    assert struct1.typer == struct2.typer, 'structs have different typers'
    n_elem_types = struct1.typer.n_elem_types
    try:
        return compute_min_rmsd(struct1.coords.cpu(), struct1.types[:, :n_elem_types].cpu(), struct2.coords.cpu(), struct2.types[:, :n_elem_types].cpu())
    except (AssertionError, ZeroDivisionError):
        if catch_exc:
            return np.nan
        raise


def compute_mean_atom_rmsd(structs, ref_structs):
    atom_rmsds = [compute_struct_rmsd(s, r) for s, r in zip(structs, ref_structs)]
    return np.mean(atom_rmsds)


def compute_mean_type_diff(structs, ref_structs, which=None):
    if which is None:
        type_counts = [s.type_counts for s in structs]
        ref_type_counts = [s.type_counts for s in ref_structs]
    elif which == 'elem':
        type_counts = [s.elem_counts for s in structs]
        ref_type_counts = [s.elem_counts for s in ref_structs]
    elif which == 'prop':
        type_counts = [s.prop_counts for s in structs]
        ref_type_counts = [s.prop_counts for s in ref_structs]
    type_diffs = np.array([(t - r).norm(p=1).item() for t, r in zip(type_counts, ref_type_counts)])
    return np.mean(type_diffs), np.mean(type_diffs == 0)


def compute_mean_n_atoms(structs):
    return np.mean([s.n_atoms for s in structs])


def compute_mean_radius(structs):
    return np.mean([s.radius for s in structs])


def compute_n_atoms_variance(structs):
    m = np.mean([s.n_atoms for s in structs])
    return np.mean([((s.n_atoms - m) ** 2) for s in structs])


def compute_type_variance(structs, which=None):
    if which is None:
        type_counts = [s.type_counts for s in structs]
    elif which == 'elem':
        type_counts = [s.elem_counts for s in structs]
    elif which == 'prop':
        type_counts = [s.prop_counts for s in structs]
    m = torch.stack(type_counts).mean(dim=0)
    return np.mean([(t - m).norm(p=1).item() for t in type_counts])


def compute_struct_metrics(struct_type, structs):
    m = OrderedDict()
    m[struct_type + '_n_atoms'] = compute_mean_n_atoms(structs)
    m[struct_type + '_n_atoms_variance'] = compute_n_atoms_variance(structs)
    m[struct_type + '_radius'] = compute_mean_radius(structs)
    m[struct_type + '_type_variance'] = compute_type_variance(structs)
    m[struct_type + '_elem_variance'] = compute_type_variance(structs, which='elem')
    m[struct_type + '_prop_variance'] = compute_type_variance(structs, which='prop')
    return m


def compute_paired_struct_metrics(struct_type, structs, ref_struct_type, ref_structs):
    m = compute_struct_metrics(struct_type, structs)
    m.update(compute_struct_metrics(ref_struct_type, ref_structs))
    m[struct_type + '_type_diff'], m[struct_type + '_exact_types'] = compute_mean_type_diff(structs, ref_structs)
    m[struct_type + '_elem_diff'], m[struct_type + '_exact_elems'] = compute_mean_type_diff(structs, ref_structs, which='elem')
    m[struct_type + '_prop_diff'], m[struct_type + '_exact_props'] = compute_mean_type_diff(structs, ref_structs, which='prop')
    m[struct_type + '_atom_rmsd'] = compute_mean_atom_rmsd(structs, ref_structs)
    return m


def compute_scalar_metrics(scalar_type, scalars):
    m = OrderedDict()
    m[scalar_type + '_mean'] = scalars.mean().item()
    m[scalar_type + '_variance'] = scalars.var(unbiased=False).item()
    return m


def get_memory_used():
    return psutil.Process(os.getpid()).memory_info().rss


def get_state_prefix(out_prefix, iter_):
    return '{}_iter_{}'.format(out_prefix, iter_)


def save_on_exception(method):

    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except:
            self.save_state_and_metrics()
            raise
    return wrapper


class GenerativeSolver(nn.Module):
    """
    Base class for training models that
    generate ligand atomic density grids.
    """
    gen_model_type = None
    has_disc_model = False
    has_prior_model = False
    has_complex_input = False

    def __init__(self, out_prefix, data_kws={}, wandb_kws={}, gen_model_kws={}, disc_model_kws={}, prior_model_kws={}, loss_fn_kws={}, gen_optim_kws={}, disc_optim_kws={}, prior_optim_kws={}, atom_fitting_kws={}, bond_adding_kws={}, device='cuda', debug=False, sync_cuda=False):
        super().__init__()
        self.device = device
        None
        self.init_data(device=device, **data_kws)
        self.learn_recon_var = loss_fn_kws.pop('learn_recon_var', False)
        None
        self.init_gen_model(device=device, **gen_model_kws)
        self.init_gen_optimizer(**gen_optim_kws)
        if self.has_disc_model:
            None
            self.init_disc_model(device=device, **disc_model_kws)
            self.init_disc_optimizer(**disc_optim_kws)
        else:
            self.disc_iter = 0
        if self.has_prior_model:
            None
            self.init_prior_model(device=device, **prior_model_kws)
            self.init_prior_optimizer(**prior_optim_kws)
        else:
            self.prior_iter = 0
        self.init_loss_fn(device=device, **loss_fn_kws)
        None
        self.atom_fitter = atom_fitting.AtomFitter(device=device, **atom_fitting_kws)
        self.bond_adder = bond_adding.BondAdder(debug=debug, **bond_adding_kws)
        self.index_cols = ['iteration', 'disc_iter', 'data_phase', 'model_phase', 'grid_phase', 'batch']
        self.metrics = pd.DataFrame(columns=self.index_cols)
        self.metrics.set_index(self.index_cols, inplace=True)
        self.out_prefix = out_prefix
        self.debug = debug
        self.sync_cuda = sync_cuda
        self.wandb_kws = wandb_kws
        self.use_wandb = self.wandb_kws.get('use_wandb', False)
        if self.use_wandb:
            try:
                wandb
            except NameError:
                raise ImportError('wandb is not installed')

    def init_data(self, device, train_file, test_file, **data_kws):
        self.train_data = data.AtomGridData(device=device, data_file=train_file, **data_kws)
        self.test_data = data.AtomGridData(device=device, data_file=test_file, **data_kws)

    def init_gen_model(self, device, caffe_init=False, state=None, **gen_model_kws):
        self.gen_model = self.gen_model_type(n_channels_in=self.n_channels_in, n_channels_cond=self.n_channels_cond, n_channels_out=self.n_channels_out, grid_size=self.train_data.grid_size, device=device, **gen_model_kws)
        if caffe_init:
            self.gen_model.apply(models.caffe_init_weights)
        if self.learn_recon_var:
            self.gen_model.log_recon_var = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.gen_model.log_recon_var = torch.zeros(1, device=device)
        if state:
            self.gen_model.load_state_dict(torch.load(state))

    def init_disc_model(self, device, caffe_init=False, state=None, **disc_model_kws):
        self.disc_model = models.Discriminator(n_channels=self.n_channels_disc, grid_size=self.train_data.grid_size, **disc_model_kws)
        if caffe_init:
            self.disc_model.apply(models.caffe_init_weights)
        if state:
            self.disc_model.load_state_dict(torch.load(state))

    def init_prior_model(self, device, caffe_init=False, state=None, **prior_model_kws):
        self.prior_model = models.Stage2VAE(n_input=self.gen_model.n_latent, **prior_model_kws)
        if caffe_init:
            self.prior_model.apply(models.caffe_init_weights)
        if self.learn_recon_var:
            self.prior_model.log_recon_var = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.prior_model.log_recon_var = torch.zeros(1, device=device)
        if state:
            self.prior_model.load_state_dict(torch.load(state))

    def init_gen_optimizer(self, type, n_train_iters=1, clip_gradient=0, **gen_optim_kws):
        self.n_gen_train_iters = n_train_iters
        self.gen_clip_grad = clip_gradient
        self.gen_optimizer = getattr(optim, type)(self.gen_model.parameters(), **gen_optim_kws)
        self.gen_iter = 0

    def init_disc_optimizer(self, type, n_train_iters=2, clip_gradient=0, **disc_optim_kws):
        self.n_disc_train_iters = n_train_iters
        self.disc_clip_grad = clip_gradient
        self.disc_optimizer = getattr(optim, type)(self.disc_model.parameters(), **disc_optim_kws)
        self.disc_iter = 0

    def init_prior_optimizer(self, type, n_train_iters=1, clip_gradient=0, **prior_optim_kws):
        assert n_train_iters == self.n_gen_train_iters
        self.n_prior_train_iters = n_train_iters
        self.prior_clip_grad = clip_gradient
        self.prior_optimizer = getattr(optim, type)(self.prior_model.parameters(), **prior_optim_kws)
        self.prior_iter = 0

    def init_loss_fn(self, device, balance=False, **loss_fn_kws):
        self.loss_fn = loss_fns.LossFunction(device=device, **loss_fn_kws)
        if self.has_disc_model:
            assert self.loss_fn.gan_loss_wt != 0, 'GAN loss weight is zero'
            if balance:
                self.disc_gan_loss = -1
                self.gen_gan_loss = 0
        else:
            assert self.loss_fn.gan_loss_wt == 0, 'non-zero GAN loss weight in non-GAN model'
            assert balance == False, 'can only balance GAN loss'
        if self.has_prior_model:
            assert self.loss_fn.kldiv2_loss_wt != 0, '2-stage VAE kldiv2 loss weight is zero'
            assert self.loss_fn.recon2_loss_wt != 0, '2-stage VAE recon2 loss weight is zero'
        else:
            assert self.loss_fn.kldiv2_loss_wt == 0, 'non-zero kldiv2 weight, but no stage 2 VAE'
            assert self.loss_fn.recon2_loss_wt == 0, 'non-zero recon2 weight, but no stage 2 VAE'
        if not self.gen_model_type.has_conditional_encoder:
            assert self.loss_fn.steric_loss_wt == 0, 'non-zero steric loss but no rec'
        self.balance = balance

    @property
    def n_channels_in(self):
        if self.gen_model_type.has_input_encoder:
            data = self.train_data
            if self.has_complex_input:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    @property
    def n_channels_cond(self):
        if self.gen_model_type.has_conditional_encoder:
            return self.train_data.n_rec_channels

    @property
    def n_channels_out(self):
        return self.train_data.n_lig_channels

    @property
    def n_channels_disc(self):
        if self.has_disc_model:
            data = self.train_data
            if self.gen_model_type.has_conditional_encoder:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    @property
    def state_prefix(self):
        return get_state_prefix(self.out_prefix, self.gen_iter)

    @property
    def gen_model_state_file(self):
        return self.state_prefix + '.gen_model_state'

    @property
    def gen_solver_state_file(self):
        return self.state_prefix + '.gen_solver_state'

    @property
    def disc_model_state_file(self):
        return self.state_prefix + '.disc_model_state'

    @property
    def disc_solver_state_file(self):
        return self.state_prefix + '.disc_solver_state'

    @property
    def prior_model_state_file(self):
        return self.state_prefix + '.prior_model_state'

    @property
    def prior_solver_state_file(self):
        return self.state_prefix + '.prior_solver_state'

    @property
    def metrics_file(self):
        return self.out_prefix + '.train_metrics'

    def save_state(self):
        self.gen_model.cpu()
        state_file = self.gen_model_state_file
        None
        torch.save(self.gen_model.state_dict(), state_file)
        state_file = self.gen_solver_state_file
        None
        state_dict = OrderedDict()
        state_dict['optim_state'] = self.gen_optimizer.state_dict()
        state_dict['iter'] = self.gen_iter
        torch.save(state_dict, state_file)
        self.gen_model
        if self.has_disc_model:
            self.disc_model.cpu()
            state_file = self.disc_model_state_file
            None
            torch.save(self.disc_model.state_dict(), state_file)
            state_file = self.disc_solver_state_file
            None
            state_dict = OrderedDict()
            state_dict['optim_state'] = self.disc_optimizer.state_dict()
            state_dict['iter'] = self.disc_iter
            torch.save(state_dict, state_file)
            self.disc_model
        if self.has_prior_model:
            self.prior_model.cpu()
            state_file = self.prior_model_state_file
            None
            torch.save(self.prior_model.state_dict(), state_file)
            state_file = self.prior_solver_state_file
            None
            state_dict = OrderedDict()
            state_dict['optim_state'] = self.prior_optimizer.state_dict()
            state_dict['iter'] = self.prior_iter
            torch.save(state_dict, state_file)
            self.prior_model

    def load_state(self, cont_iter=None):
        if cont_iter is None:
            self.gen_iter = self.find_last_iter()
        else:
            self.gen_iter = cont_iter
        state_file = self.state_prefix + '.gen_model_state'
        None
        self.gen_model.load_state_dict(torch.load(state_file))
        state_file = self.state_prefix + '.gen_solver_state'
        None
        state_dict = torch.load(state_file)
        self.gen_optimizer.load_state_dict(state_dict['optim_state'])
        self.gen_iter = state_dict['iter']
        if self.has_disc_model:
            state_file = self.state_prefix + '.disc_model_state'
            None
            self.disc_model.load_state_dict(torch.load(state_file))
            state_file = self.state_prefix + '.disc_solver_state'
            None
            state_dict = torch.load(state_file)
            self.disc_optimizer.load_state_dict(state_dict['optim_state'])
            self.disc_iter = state_dict['iter']
        if self.has_prior_model:
            state_file = self.state_prefix + '.prior_model_state'
            None
            self.prior_model.load_state_dict(torch.load(state_file))
            state_file = self.state_prefix + '.prior_solver_state'
            None
            state_dict = torch.load(state_file)
            self.prior_optimizer.load_state_dict(state_dict['optim_state'])
            self.prior_iter = state_dict['iter']

    def find_last_iter(self):
        return find_last_iter(self.out_prefix)

    def save_metrics(self):
        csv_file = self.metrics_file
        None
        self.metrics.to_csv(csv_file, sep=' ')

    def load_metrics(self):
        csv_file = self.metrics_file
        None
        self.metrics = pd.read_csv(csv_file, sep=' ').set_index(self.index_cols)

    def load_state_and_metrics(self, cont_iter=None):
        self.load_state(cont_iter)
        try:
            self.load_metrics()
        except FileNotFoundError:
            if self.gen_iter > 0:
                raise

    def save_state_and_metrics(self):
        self.save_metrics()
        self.save_state()

    def print_metrics(self, idx, metrics):
        index_str = ' '.join('{}={}'.format(*kv) for kv in zip(self.index_cols, idx))
        metrics_str = ' '.join('{}={:.4f}'.format(*kv) for kv in metrics.items())
        None

    def insert_metrics(self, idx, metrics):
        for k, v in metrics.items():
            try:
                self.metrics.loc[idx, k] = v
            except AttributeError:
                None
                raise

    def save_mols(self, mols, grid_type):
        sdf_file = '{}_iter_{}_{}.sdf'.format(self.out_prefix, self.gen_iter, grid_type)
        None
        molecules.write_rd_mols_to_sdf_file(sdf_file, mols, kekulize=False)

    @property
    def has_prior_phase(self):
        return self.gen_model_type.is_variational

    @property
    def has_posterior_phase(self):
        return self.gen_model_type.has_input_encoder or not self.has_prior_phase

    def get_gen_grid_phase(self, batch_idx, test=False):
        """
        Determine whether to sample prior or
        posterior grids in the next gen batch.
        """
        has_prior_phase = self.has_prior_phase
        if not test:
            has_prior_phase &= self.loss_fn.has_prior_loss
        has_posterior_phase = self.has_posterior_phase
        assert has_prior_phase or has_posterior_phase, 'no gen grid phases'
        grid_phases = []
        if has_posterior_phase:
            grid_phases.append('poster')
        if has_prior_phase:
            grid_phases.append('prior')
        if self.has_prior_model and test:
            grid_phases.extend(['poster2', 'prior2'])
        phase_idx = self.gen_iter + batch_idx
        return grid_phases[phase_idx % len(grid_phases)]

    def get_disc_grid_phase(self, batch_idx, test=False):
        """
        Determine whether to sample real, prior,
        or posterior grids in the next disc batch.

        NOT integrated with stage-2 VAE.
        """
        has_prior_phase = self.has_prior_phase
        has_posterior_phase = self.has_posterior_phase
        assert has_prior_phase or has_posterior_phase, 'no disc grid phases'
        grid_phases = []
        if has_posterior_phase:
            grid_phases += ['real', 'poster']
        if has_prior_phase:
            grid_phases += ['real', 'prior']
        phase_idx = self.disc_iter + batch_idx
        return grid_phases[phase_idx % len(grid_phases)]

    def gen_forward(self, data, grid_type, fit_atoms=False):
        """
        Compute loss and other metrics for the
        generative model's ability to produce
        realistic atomic density grids.
        """
        is_varial = self.gen_model.is_variational
        has_input = self.gen_model.has_input_encoder
        has_cond = self.gen_model_type.has_conditional_encoder
        has_disc = self.has_disc_model
        valid_grid_types = set()
        if self.has_prior_phase:
            valid_grid_types.add('prior')
        if self.has_posterior_phase:
            valid_grid_types.add('poster')
        if self.has_prior_model:
            valid_grid_types.add('prior2')
            valid_grid_types.add('poster2')
        assert grid_type in valid_grid_types, 'invalid grid type ' + repr(grid_type)
        if grid_type[-1] == '2':
            grid_type = grid_type[:-1]
            decode_stage2_vecs = True
        else:
            decode_stage2_vecs = False
        prior = grid_type == 'prior'
        posterior = grid_type == 'poster'
        compute_stage2_loss = posterior and self.has_prior_model
        t0 = time.time()
        if posterior or has_cond:
            input_grids, cond_grids, input_structs, cond_structs, _, _ = data.forward()
            input_rec_structs, input_lig_structs = input_structs
            cond_rec_structs, cond_lig_structs = cond_structs
            input_rec_grids, input_lig_grids = data.split_channels(input_grids)
            cond_rec_grids, cond_lig_grids = data.split_channels(cond_grids)
        if self.sync_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        if posterior:
            gen_input_grids = input_grids if self.has_complex_input else input_lig_grids
        if decode_stage2_vecs:
            lig_gen_grids, latent_vecs, latent_means, latent_log_stds, latent_vecs_gen, latent2_means, latent2_log_stds = self.gen_model.forward2(prior_model=self.prior_model, inputs=gen_input_grids if posterior else None, conditions=cond_rec_grids if has_cond else None, batch_size=data.batch_size)
        else:
            lig_gen_grids, latent_vecs, latent_means, latent_log_stds = self.gen_model(inputs=gen_input_grids if posterior else None, conditions=cond_rec_grids if has_cond else None, batch_size=data.batch_size)
            if compute_stage2_loss:
                latent_vecs = latent_vecs.detach()
                latent_vecs_gen, _, latent2_means, latent2_log_stds = self.prior_model(inputs=latent_vecs, batch_size=data.batch_size)
        if self.sync_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        if has_disc:
            if has_cond:
                disc_input_grids = torch.cat([cond_rec_grids, lig_gen_grids], dim=1)
            else:
                disc_input_grids = lig_gen_grids
            disc_labels = torch.ones(data.batch_size, 1, device=self.device)
            disc_preds, _ = self.disc_model(inputs=disc_input_grids)
        loss, metrics = self.loss_fn(lig_grids=cond_lig_grids if posterior else None, lig_gen_grids=lig_gen_grids if posterior else None, disc_labels=disc_labels if has_disc else None, disc_preds=disc_preds if has_disc else None, latent_means=latent_means if posterior else None, latent_log_stds=latent_log_stds if posterior else None, rec_grids=cond_rec_grids if has_cond else None, rec_lig_grids=lig_gen_grids if has_cond else None, latent2_means=latent2_means if compute_stage2_loss else None, latent2_log_stds=latent2_log_stds if compute_stage2_loss else None, real_latents=latent_vecs if compute_stage2_loss else None, gen_latents=latent_vecs_gen if compute_stage2_loss else None, gen_log_var=self.gen_model.log_recon_var if posterior else None, prior_log_var=self.prior_model.log_recon_var if compute_stage2_loss else None, iteration=self.gen_iter)
        if self.sync_cuda:
            torch.cuda.synchronize()
        t3 = time.time()
        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(batch_values=lig_gen_grids, center=torch.zeros(3), resolution=data.resolution, typer=data.lig_typer)
            lig_gen_fit_mols, _ = self.bond_adder.make_batch(structs=lig_gen_fit_structs)
            self.save_mols(lig_gen_fit_mols, grid_type)
        if self.sync_cuda:
            torch.cuda.synchronize()
        t4 = time.time()
        if posterior:
            metrics.update(compute_paired_grid_metrics('lig_gen', lig_gen_grids, 'lig', input_lig_grids))
        else:
            metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))
        if has_cond:
            metrics.update(compute_paired_grid_metrics('lig_gen', lig_gen_grids, 'cond_lig', cond_lig_grids))
        if has_disc:
            metrics.update(compute_scalar_metrics('pred', disc_preds))
        if fit_atoms:
            if posterior:
                metrics.update(compute_paired_struct_metrics('lig_gen_fit', lig_gen_fit_structs, 'lig', input_lig_structs))
            else:
                metrics.update(compute_struct_metrics('lig_gen_fit', lig_gen_fit_structs))
            if has_cond:
                metrics.update(compute_paired_struct_metrics('lig_gen_fit', lig_gen_fit_structs, 'cond_lig', cond_lig_structs))
        if self.sync_cuda:
            torch.cuda.synchronize()
        t5 = time.time()
        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_fit_time'] = t4 - t3
        metrics['forward_metrics_time'] = t5 - t4
        return loss, metrics

    def disc_forward(self, data, grid_type):
        """
        Compute loss and other metrics for the
        discriminative model's ability to tell
        apart real and generated data.
        """
        is_varial = self.gen_model.is_variational
        has_input = self.gen_model.has_input_encoder
        has_cond = self.gen_model_type.has_conditional_encoder
        valid_grid_types = {'real'}
        if is_varial:
            valid_grid_types.add('prior')
        if has_input:
            valid_grid_types.add('poster')
        assert grid_type in valid_grid_types, 'invalid grid type'
        real = grid_type == 'real'
        prior = grid_type == 'prior'
        posterior = grid_type == 'poster'
        t0 = time.time()
        with torch.no_grad():
            if real or posterior or has_cond:
                input_grids, cond_grids, input_structs, cond_structs, _, _ = data.forward()
                rec_structs, lig_structs = input_structs
                input_rec_grids, input_lig_grids = data.split_channels(input_grids)
                if data.diff_cond_transform:
                    cond_rec_grids, cond_lig_grids = data.split_channels(cond_grids)
                else:
                    cond_grids = input_grids
                    cond_rec_grids = input_rec_grids
                    cond_lig_grids = input_lig_grids
            t1 = time.time()
            if not real:
                if posterior:
                    gen_input_grids = input_grids if self.has_complex_input else input_lig_grids
                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = self.gen_model(inputs=gen_input_grids if posterior else None, conditions=cond_rec_grids if has_cond else None, batch_size=data.batch_size)
            t2 = time.time()
        if real:
            disc_grids = cond_grids if has_cond else cond_lig_grids
        elif has_cond:
            disc_grids = torch.cat([cond_rec_grids, lig_gen_grids], dim=1)
        else:
            disc_grids = lig_gen_grids
        disc_labels = torch.full((data.batch_size, 1), real, device=self.device)
        disc_preds, _ = self.disc_model(inputs=disc_grids)
        loss, metrics = self.loss_fn(disc_labels=disc_labels, disc_preds=disc_preds, use_loss_wt=False)
        t3 = time.time()
        metrics.update(compute_grid_metrics('lig' if real else 'lig_gen', cond_lig_grids if real else lig_gen_grids))
        metrics.update(compute_scalar_metrics('disc_pred', disc_preds))
        t4 = time.time()
        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    def gen_backward(self, loss, update=False, compute_norm=False):
        """
        Backpropagate loss gradient onto
        generative model parameters, op-
        tionally computing the gradient
        norm and/or updating parameters.
        """
        metrics = OrderedDict()
        t0 = time.time()
        self.gen_optimizer.zero_grad()
        loss.backward()
        t1 = time.time()
        if self.gen_clip_grad:
            models.clip_grad_norm(self.gen_model, self.gen_clip_grad)
        if self.has_prior_model and self.prior_clip_grad:
            models.clip_grad_norm(self.prior_model, self.prior_clip_grad)
        if compute_norm:
            grad_norm = models.compute_grad_norm(self.gen_model)
            if self.has_prior_model:
                prior_grad_norm = models.compute_grad_norm(self.prior_model)
        t2 = time.time()
        if update:
            self.gen_optimizer.step()
            self.gen_iter += 1
            if self.has_prior_model:
                self.prior_optimizer.step()
                self.prior_iter += 1
        t3 = time.time()
        if compute_norm:
            metrics['gen_grad_norm'] = grad_norm
            if self.has_prior_model:
                metrics['prior_grad_norm'] = prior_grad_norm
        metrics['backward_grad_time'] = t1 - t0
        metrics['backward_norm_time'] = t2 - t1
        metrics['backward_update_time'] = t3 - t2
        return metrics

    def disc_backward(self, loss, update=False, compute_norm=False):
        """
        Backpropagate loss gradient onto
        discriminative model parameters,
        optionally computing the gradient
        norm and/or updating parameters.
        """
        metrics = OrderedDict()
        t0 = time.time()
        self.disc_optimizer.zero_grad()
        loss.backward()
        t1 = time.time()
        if self.disc_clip_grad:
            models.clip_grad_norm(self.disc_model, self.disc_clip_grad)
        if compute_norm:
            grad_norm = models.compute_grad_norm(self.disc_model)
        t2 = time.time()
        if update:
            self.disc_optimizer.step()
            self.disc_iter += 1
        t3 = time.time()
        if compute_norm:
            metrics['disc_grad_norm'] = grad_norm
        metrics['backward_grad_time'] = t1 - t0
        metrics['backward_norm_time'] = t2 - t1
        metrics['backward_update_time'] = t3 - t2
        return metrics

    def gen_step(self, grid_type, update=True, compute_norm=True, batch_idx=0):
        """
        Perform a single forward-backward pass
        on the generative model, optionally
        updating model parameters and/or comp-
        uting the parameter gradient norm.
        """
        idx = self.gen_iter, self.disc_iter, 'train', 'gen', grid_type, batch_idx
        need_gradient = update or compute_norm
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()
        loss, metrics = self.gen_forward(self.train_data, grid_type)
        if self.sync_cuda:
            torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()
        if need_gradient:
            metrics.update(self.gen_backward(loss, update, compute_norm))
        if self.sync_cuda:
            torch.cuda.synchronize()
        m2 = torch.cuda.max_memory_allocated()
        t2 = time.time()
        metrics['memory'] = get_memory_used() / MB
        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['backward_time'] = t2 - t1
        metrics['backward_gpu'] = m2 / MB
        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        assert not loss.isnan(), 'generator loss is nan'
        if compute_norm:
            grad_norm = metrics['gen_grad_norm']
            assert not np.isnan(grad_norm), 'generator gradient is nan'
            assert not np.isclose(0, grad_norm), 'generator gradient is zero'
        return metrics

    def disc_step(self, grid_type, update=True, compute_norm=True, batch_idx=0):
        """
        Perform a single forward-backward pass
        on the discriminative model, optionally
        updating model parameters and/or comp-
        uting the parameter gradient norm.
        """
        idx = self.gen_iter, self.disc_iter, 'train', 'disc', grid_type, batch_idx
        need_gradient = update or compute_norm
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()
        loss, metrics = self.disc_forward(self.train_data, grid_type)
        if self.sync_cuda:
            torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()
        if need_gradient:
            metrics.update(self.disc_backward(loss, update, compute_norm))
        if self.sync_cuda:
            torch.cuda.synchronize()
        m2 = torch.cuda.max_memory_allocated()
        t2 = time.time()
        metrics['memory'] = get_memory_used() / MB
        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['backward_time'] = t2 - t1
        metrics['backward_gpu'] = m2 / MB
        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        assert not loss.isnan(), 'discriminator loss is nan'
        if compute_norm:
            grad_norm = metrics['disc_grad_norm']
            assert not np.isnan(grad_norm), 'discriminator gradient is nan'
        return metrics

    def test_model(self, n_batches, model_type, fit_atoms=False):
        """
        Evaluate a model's performance on
        n_batches of test data, optionally
        performing atom fitting.
        """
        valid_model_types = {'gen'}
        if self.has_disc_model:
            valid_model_types.add('disc')
        test_disc = model_type == 'disc'
        for i in range(n_batches):
            torch.cuda.reset_max_memory_allocated()
            t0 = time.time()
            if test_disc:
                grid_type = self.get_disc_grid_phase(i, test=True)
                loss, metrics = self.disc_forward(data=self.test_data, grid_type=grid_type)
            else:
                grid_type = self.get_gen_grid_phase(i, test=True)
                loss, metrics = self.gen_forward(data=self.test_data, grid_type=grid_type, fit_atoms=fit_atoms)
            metrics['memory'] = get_memory_used() / MB
            metrics['forward_time'] = time.time() - t0
            metrics['forward_gpu'] = torch.cuda.max_memory_allocated() / MB
            idx = self.gen_iter, self.disc_iter, 'test', model_type, grid_type, i
            self.insert_metrics(idx, metrics)
            if self.use_wandb:
                wandb_metrics = metrics.copy()
                wandb_metrics.update(dict(zip(self.index_cols, idx)))
                wandb.log(wandb_metrics)
        idx = idx[:-1]
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

    def test_models(self, n_batches, fit_atoms):
        """
        Evaluate each model on n_batches of test
        data, optionally performing atom fitting.
        """
        if self.has_disc_model:
            self.test_model(n_batches=n_batches, model_type='disc')
        self.test_model(n_batches=n_batches, model_type='gen', fit_atoms=fit_atoms)
        self.save_metrics()

    def train_model(self, n_iters, model_type, update=True, compute_norm=True):
        """
        Perform n_iters forward-backward passes
        on one of the models, optionally updating
        its parameters.
        """
        valid_model_types = {'gen'}
        if self.has_disc_model:
            valid_model_types.add('disc')
        train_disc = model_type == 'disc'
        for i in range(n_iters):
            batch_idx = 0 if update else i
            if train_disc:
                grid_type = self.get_disc_grid_phase(batch_idx, test=False)
                metrics = self.disc_step(grid_type=grid_type, update=update, compute_norm=compute_norm, batch_idx=batch_idx)
                if grid_type == 'real':
                    disc_gan_loss = metrics.get('gan_loss', -1)
                    self.disc_gan_loss = disc_gan_loss
            else:
                grid_type = self.get_gen_grid_phase(batch_idx, test=False)
                metrics = self.gen_step(grid_type=grid_type, update=update, compute_norm=compute_norm, batch_idx=batch_idx)
                gen_gan_loss = metrics.get('gan_loss', 0)
                self.gen_gan_loss = gen_gan_loss

    def train_models(self, update=True, compute_norm=False):
        """
        Train each model on training data for
        a pre-determined number of iterations.
        """
        if update:
            if self.balance:
                update_disc = True
                update_gen = self.disc_gan_loss < self.gen_gan_loss
            else:
                update_disc = update_gen = True
        else:
            update_disc = update_gen = False
        if self.has_disc_model:
            self.train_model(n_iters=self.n_disc_train_iters, model_type='disc', update=update_disc, compute_norm=compute_norm)
        self.train_model(n_iters=self.n_gen_train_iters, model_type='gen', update=update_gen, compute_norm=compute_norm)

    @save_on_exception
    def train_and_test(self, max_iter, test_interval, n_test_batches, fit_interval, norm_interval, save_interval):
        init_iter = self.gen_iter
        last_save = None
        last_test = None
        divides = lambda d, n: n % d == 0
        while self.gen_iter <= max_iter:
            i = self.gen_iter
            if last_save != i and divides(save_interval, i):
                self.save_state()
                last_save = i
            if last_test != i and divides(test_interval, i):
                fit_atoms = fit_interval > 0 and divides(fit_interval, i)
                self.test_models(n_batches=n_test_batches, fit_atoms=fit_atoms)
                last_test = i
            update = i < max_iter
            compute_norm = norm_interval > 0 and divides(norm_interval, i)
            self.train_models(update=update, compute_norm=compute_norm)
            if i == max_iter:
                break
        self.save_state_and_metrics()


class AESolver(GenerativeSolver):
    gen_model_type = models.AE


class VAESolver(GenerativeSolver):
    gen_model_type = models.VAE


class CESolver(GenerativeSolver):
    gen_model_type = models.CE


class CVAESolver(GenerativeSolver):
    gen_model_type = models.CVAE
    has_complex_input = True


class GANSolver(GenerativeSolver):
    gen_model_type = models.GAN
    has_disc_model = True


class CGANSolver(GenerativeSolver):
    gen_model_type = models.CGAN
    has_disc_model = True


class VAEGANSolver(GenerativeSolver):
    gen_model_type = models.VAE
    has_disc_model = True


class CVAEGANSolver(GenerativeSolver):
    gen_model_type = models.CVAE
    has_complex_input = True
    has_disc_model = True


class VAE2Solver(GenerativeSolver):
    gen_model_type = models.VAE2
    has_prior_model = True


class CVAE2Solver(GenerativeSolver):
    gen_model_type = models.CVAE2
    has_complex_input = True
    has_prior_model = True


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv3DBlock,
     lambda: ([], {'n_convs': 4, 'n_channels_in': 4, 'n_channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3DReLU,
     lambda: ([], {'n_channels_in': 4, 'n_channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Grid2Vec,
     lambda: ([], {'in_shape': 4, 'n_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reshape,
     lambda: ([], {'shape': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (TConv3DBlock,
     lambda: ([], {'n_convs': 4, 'n_channels_in': 4, 'n_channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TConv3DReLU,
     lambda: ([], {'n_channels_in': 4, 'n_channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mattragoza_LiGAN(_paritybench_base):
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

