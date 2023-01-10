import sys
_module = sys.modules[__name__]
del sys
BigGAN = _module
DiffAugment_pytorch = _module
animal_hash = _module
datasets = _module
dnnlib = _module
submission = _module
internal = _module
local = _module
run_context = _module
submit = _module
tflib = _module
autosummary = _module
network = _module
optimizer = _module
tfutil = _module
eval = _module
inception_tf = _module
layers = _module
losses = _module
train = _module
train_fns = _module
utils = _module
DiffAugment_tf = _module
compare_gan = _module
architectures = _module
abstract_arch = _module
arch_ops = _module
arch_ops_test = _module
arch_ops_tpu_test = _module
architectures_test = _module
dcgan = _module
infogan = _module
resnet30 = _module
resnet5 = _module
resnet_biggan = _module
resnet_biggan_deep = _module
resnet_biggan_deep_test = _module
resnet_biggan_test = _module
resnet_cifar = _module
resnet_init_test = _module
resnet_norm_test = _module
resnet_ops = _module
resnet_stl = _module
sndcgan = _module
datasets_test = _module
eval_gan_lib = _module
eval_gan_lib_test = _module
eval_utils = _module
gans = _module
abstract_gan = _module
consts = _module
loss_lib = _module
modular_gan = _module
modular_gan_conditional_test = _module
modular_gan_test = _module
modular_gan_tpu_test = _module
ops = _module
penalty_lib = _module
s3gan = _module
s3gan_test = _module
ssgan = _module
ssgan_test = _module
hooks = _module
main = _module
metrics = _module
accuracy = _module
eval_task = _module
fid_score = _module
fid_score_test = _module
fractal_dimension = _module
fractal_dimension_test = _module
gilbo = _module
image_similarity = _module
inception_score = _module
jacobian_conditioning = _module
jacobian_conditioning_test = _module
kid_score = _module
ms_ssim_score = _module
ms_ssim_score_test = _module
prd_score = _module
prd_score_test = _module
runner_lib = _module
runner_lib_test = _module
test_utils = _module
tpu = _module
tpu_ops = _module
tpu_ops_test = _module
tpu_random = _module
tpu_random_test = _module
tpu_summaries = _module
generate = _module
setup = _module
DiffAugment_pytorch = _module
calc_metrics = _module
dataset_tool = _module
generate = _module
generate_gif = _module
projector = _module
style_mixing = _module
custom_ops = _module
bias_act = _module
conv2d_gradfix = _module
conv2d_resample = _module
fma = _module
grid_sample_gradfix = _module
upfirdn2d = _module
training_stats = _module
train = _module
training = _module
fused_bias_act = _module
upfirdn_2d = _module
frechet_inception_distance = _module
metric_base = _module
metric_defaults = _module
run_cifar = _module
run_ffhq = _module
run_low_shot = _module
dataset = _module
loss = _module
misc = _module
training_loop = _module
DiffAugment_pytorch = _module

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


import numpy as np


import math


import functools


import torch


import torch.nn as nn


from torch.nn import init


import torch.optim as optim


import torch.nn.functional as F


from torch.nn import Parameter as P


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import check_integrity


import torch.utils.data as data


from torch.utils.data import DataLoader


import inspect


import scipy


import types


import re


import uuid


from typing import Any


from typing import List


from typing import Tuple


from typing import Union


import torchvision


from torch.optim.optimizer import Optimizer


import time


import copy


from typing import Optional


from time import perf_counter


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


import warnings


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels': [(ch * item) for item in [16, 16, 8, 8, 4, 2, 1]], 'out_channels': [(ch * item) for item in [16, 8, 8, 4, 2, 1, 1]], 'upsample': [True] * 7, 'resolution': [8, 16, 32, 64, 128, 256, 512], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 10)}}
    arch[256] = {'in_channels': [(ch * item) for item in [16, 16, 8, 8, 4, 2]], 'out_channels': [(ch * item) for item in [16, 8, 8, 4, 2, 1]], 'upsample': [True] * 6, 'resolution': [8, 16, 32, 64, 128, 256], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 9)}}
    arch[128] = {'in_channels': [(ch * item) for item in [16, 16, 8, 4, 2]], 'out_channels': [(ch * item) for item in [16, 8, 4, 2, 1]], 'upsample': [True] * 5, 'resolution': [8, 16, 32, 64, 128], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 8)}}
    arch[64] = {'in_channels': [(ch * item) for item in [16, 16, 8, 4]], 'out_channels': [(ch * item) for item in [16, 8, 4, 2]], 'upsample': [True] * 4, 'resolution': [8, 16, 32, 64], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 7)}}
    arch[32] = {'in_channels': [(ch * item) for item in [4, 4, 4]], 'out_channels': [(ch * item) for item in [4, 4, 4]], 'upsample': [True] * 3, 'resolution': [8, 16, 32], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 6)}}
    return arch


class Generator(nn.Module):

    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128, G_kernel_size=3, G_attn='64', n_classes=1000, num_G_SVs=1, num_G_SV_itrs=1, G_shared=True, shared_dim=0, hier=False, cross_replica=False, mybn=False, G_activation=nn.ReLU(inplace=False), G_lr=5e-05, G_B1=0.0, G_B2=0.999, adam_eps=1e-08, BN_eps=1e-05, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False, G_init='ortho', skip_init=False, no_optim=False, G_param='SN', norm_style='bn', **kwargs):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.G_shared = G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        self.hier = hier
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.norm_style = norm_style
        self.BN_eps = BN_eps
        self.SN_eps = SN_eps
        self.fp16 = G_fp16
        self.arch = G_arch(self.ch, self.attention)[resolution]
        if self.hier:
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = self.dim_z // self.num_slots
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d, kernel_size=3, padding=1, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        bn_linear = functools.partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding
        self.which_bn = functools.partial(layers.ccbn, which_linear=bn_linear, cross_replica=self.cross_replica, mybn=self.mybn, input_size=self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes, norm_style=self.norm_style, eps=self.BN_eps)
        self.shared = self.which_embedding(n_classes, self.shared_dim) if G_shared else layers.identity()
        self.linear = self.which_linear(self.dim_z // self.num_slots, self.arch['in_channels'][0] * self.bottom_width ** 2)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index], out_channels=self.arch['out_channels'][index], which_conv=self.which_conv, which_bn=self.which_bn, activation=self.activation, upsample=functools.partial(F.interpolate, scale_factor=2) if self.arch['upsample'][index] else None)]]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1], cross_replica=self.cross_replica, mybn=self.mybn), self.activation, self.which_conv(self.arch['out_channels'][-1], 3))
        if not skip_init:
            self.init_weights()
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            None
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        None

    def forward(self, z, y):
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, ys[index])
        return torch.tanh(self.output_layer(h))


def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [3] + [(ch * item) for item in [1, 2, 4, 8, 8, 16]], 'out_channels': [(item * ch) for item in [1, 2, 4, 8, 8, 16, 16]], 'downsample': [True] * 6 + [False], 'resolution': [128, 64, 32, 16, 8, 4, 4], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(2, 8)}}
    arch[128] = {'in_channels': [3] + [(ch * item) for item in [1, 2, 4, 8, 16]], 'out_channels': [(item * ch) for item in [1, 2, 4, 8, 16, 16]], 'downsample': [True] * 5 + [False], 'resolution': [64, 32, 16, 8, 4, 4], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(2, 8)}}
    arch[64] = {'in_channels': [3] + [(ch * item) for item in [1, 2, 4, 8]], 'out_channels': [(item * ch) for item in [1, 2, 4, 8, 16]], 'downsample': [True] * 4 + [False], 'resolution': [32, 16, 8, 4, 4], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(2, 7)}}
    arch[32] = {'in_channels': [3] + [(item * ch) for item in [4, 4, 4]], 'out_channels': [(item * ch) for item in [4, 4, 4, 4]], 'downsample': [True, True, False, False], 'resolution': [16, 16, 16, 16], 'attention': {(2 ** i): (2 ** i in [int(item) for item in attention.split('_')]) for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128, D_kernel_size=3, D_attn='64', n_classes=1000, num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), D_lr=0.0002, D_B1=0.0, D_B2=0.999, adam_eps=1e-08, SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False, D_init='ortho', skip_init=False, D_param='SN', **kwargs):
        super(Discriminator, self).__init__()
        self.ch = D_ch
        self.D_wide = D_wide
        self.resolution = resolution
        self.kernel_size = D_kernel_size
        self.attention = D_attn
        self.n_classes = n_classes
        self.activation = D_activation
        self.init = D_init
        self.D_param = D_param
        self.SN_eps = SN_eps
        self.fp16 = D_fp16
        self.arch = D_arch(self.ch, self.attention)[resolution]
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d, kernel_size=3, padding=1, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            self.which_embedding = nn.Embedding
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index], out_channels=self.arch['out_channels'][index], which_conv=self.which_conv, wide=self.D_wide, activation=self.activation, preactivation=index > 0, downsample=nn.AvgPool2d(2) if self.arch['downsample'][index] else None)]]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        if not skip_init:
            self.init_weights()
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            None
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        None

    def forward(self, x, y=None):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        out = self.linear(h)
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(cutout_size[0], dtype=torch.long, device=x.device), torch.arange(cutout_size[1], dtype=torch.long, device=x.device))
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(x.size(2), dtype=torch.long, device=x.device), torch.arange(x.size(3), dtype=torch.long, device=x.device))
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


AUGMENT_FNS = {'color': [rand_brightness, rand_saturation, rand_contrast], 'translation': [rand_translation], 'cutout': [rand_cutout]}


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


class G_D(nn.Module):

    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False, policy=False, CR=False, CR_augment=None):
        if z is not None:
            with torch.set_grad_enabled(train_G):
                G_z = self.G(z, self.G.shared(gy))
                if self.G.fp16 and not self.D.fp16:
                    G_z = G_z.float()
                if self.D.fp16 and not self.G.fp16:
                    G_z = G_z.half()
        else:
            G_z = None
        D_input = torch.cat([img for img in [G_z, x] if img is not None], 0)
        D_class = torch.cat([label for label in [gy, dy] if label is not None], 0)
        D_input = DiffAugment(D_input, policy=policy)
        if CR:
            if CR_augment:
                x_CR_aug = torch.split(D_input, [G_z.shape[0], x.shape[0]])[1]
                if CR_augment.startswith('flip,'):
                    x_CR_aug = torch.where(torch.randint(0, 2, size=[x_CR_aug.size(0), 1, 1, 1], device=x_CR_aug.device) > 0, x_CR_aug.flip(3), x_CR_aug)
                x_CR_aug = DiffAugment(x_CR_aug, policy=CR_augment.replace('flip,', ''))
                D_input = torch.cat([D_input, x_CR_aug], 0)
            else:
                D_input = torch.cat([D_input, x], 0)
            D_class = torch.cat([D_class, dy], 0)
        D_out = self.D(D_input, D_class)
        if G_z is None:
            return D_out
        elif x is not None:
            if CR:
                return torch.split(D_out, [G_z.shape[0], x.shape[0], x.shape[0]])
            else:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
        elif return_G_z:
            return D_out, G_z
        else:
            return D_out


class identity(nn.Module):

    def forward(self, input):
        return input


def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


class SN(object):

    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):

    def __init__(self, in_features, out_features, bias=True, num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None, num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


class Attention(nn.Module):

    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-05):
    scale = torch.rsqrt(var + eps)
    if gain is not None:
        scale = scale * gain
    shift = mean * scale
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-05):
    float_x = x.float()
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    var = m2 - m ** 2
    var = var.type(x.type())
    m = m.type(x.type())
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


class myBN(nn.Module):

    def __init__(self, num_channels, eps=1e-05, momentum=0.1):
        super(myBN, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var', torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        self.accumulate_standing = False

    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


def groupnorm(x, norm_style):
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    else:
        groups = 16
    return F.group_norm(x, groups)


class ccbn(nn.Module):

    def __init__(self, output_size, input_size, which_linear, eps=1e-05, momentum=0.1, cross_replica=False, mybn=False, norm_style='bn'):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.norm_style = norm_style
        if self.cross_replica:
            assert False
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


class bn(nn.Module):

    def __init__(self, output_size, eps=1e-05, momentum=0.1, cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        if self.cross_replica:
            assert False
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain, self.bias, self.training, self.momentum, self.eps)


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=bn, activation=None, upsample=None):
        super(GBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True, preactivation=False, activation=None, downsample=None):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if in_channels != out_channels or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SNConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SNLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (bn,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (myBN,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_mit_han_lab_data_efficient_gans(_paritybench_base):
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

