import sys
_module = sys.modules[__name__]
del sys
BigGAN = _module
BigGANdeep = _module
biggan_v1 = _module
converter = _module
animal_hash = _module
calculate_inception_moments = _module
datasets = _module
inception_tf13 = _module
inception_utils = _module
layers = _module
losses = _module
make_hdf5 = _module
sample = _module
sync_batchnorm = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
train = _module
train_fns = _module
utils = _module

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


import numpy as np


import math


import functools


import torch


import torch.nn as nn


from torch.nn import init


import torch.optim as optim


import torch.nn.functional as F


from torch.nn import Parameter as P


from scipy.stats import truncnorm


from torch import nn


from torch.nn import Parameter


from torch.nn import functional as F


from scipy import linalg


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import torch.nn.init as init


from torch.nn.parallel.data_parallel import DataParallel


from torch.utils.data import DataLoader


from torch.optim.optimizer import Optimizer


def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [(ch * item) for item in [16, 16, 8, 8, 4, 
        2]], 'out_channels': [(ch * item) for item in [16, 8, 8, 4, 2, 1]],
        'upsample': [True] * 6, 'resolution': [8, 16, 32, 64, 128, 256],
        'attention': {(2 ** i): (2 ** i in [int(item) for item in attention
        .split('_')]) for i in range(3, 9)}}
    arch[128] = {'in_channels': [(ch * item) for item in [16, 16, 8, 4, 2]],
        'out_channels': [(ch * item) for item in [16, 8, 4, 2, 1]],
        'upsample': [True] * 5, 'resolution': [8, 16, 32, 64, 128],
        'attention': {(2 ** i): (2 ** i in [int(item) for item in attention
        .split('_')]) for i in range(3, 8)}}
    arch[64] = {'in_channels': [(ch * item) for item in [16, 16, 8, 4]],
        'out_channels': [(ch * item) for item in [16, 8, 4, 2]], 'upsample':
        [True] * 4, 'resolution': [8, 16, 32, 64], 'attention': {(2 ** i):
        (2 ** i in [int(item) for item in attention.split('_')]) for i in
        range(3, 7)}}
    arch[32] = {'in_channels': [(ch * item) for item in [4, 4, 4]],
        'out_channels': [(ch * item) for item in [4, 4, 4]], 'upsample': [
        True] * 3, 'resolution': [8, 16, 32], 'attention': {(2 ** i): (2 **
        i in [int(item) for item in attention.split('_')]) for i in range(3,
        6)}}
    return arch


class Generator(nn.Module):

    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
        G_kernel_size=3, G_attn='64', n_classes=1000, num_G_SVs=1,
        num_G_SV_itrs=1, G_shared=True, shared_dim=0, hier=False,
        cross_replica=False, mybn=False, G_activation=nn.ReLU(inplace=False
        ), G_lr=5e-05, G_B1=0.0, G_B2=0.999, adam_eps=1e-08, BN_eps=1e-05,
        SN_eps=1e-12, G_mixed_precision=False, G_fp16=False, G_init='ortho',
        skip_init=False, no_optim=False, G_param='SN', norm_style='bn', **
        kwargs):
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
            self.which_conv = functools.partial(layers.SNConv2d,
                kernel_size=3, padding=1, num_svs=num_G_SVs, num_itrs=
                num_G_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=
                num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3,
                padding=1)
            self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        bn_linear = functools.partial(self.which_linear, bias=False
            ) if self.G_shared else self.which_embedding
        self.which_bn = functools.partial(layers.ccbn, which_linear=
            bn_linear, cross_replica=self.cross_replica, mybn=self.mybn,
            input_size=self.shared_dim + self.z_chunk_size if self.G_shared
             else self.n_classes, norm_style=self.norm_style, eps=self.BN_eps)
        self.shared = self.which_embedding(n_classes, self.shared_dim
            ) if G_shared else layers.identity()
        self.linear = self.which_linear(self.dim_z // self.num_slots, self.
            arch['in_channels'][0] * self.bottom_width ** 2)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch[
                'in_channels'][index], out_channels=self.arch[
                'out_channels'][index], which_conv=self.which_conv,
                which_bn=self.which_bn, activation=self.activation,
                upsample=functools.partial(F.interpolate, scale_factor=2) if
                self.arch['upsample'][index] else None)]]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch[
                    'out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self
            .blocks])
        self.output_layer = nn.Sequential(layers.bn(self.arch[
            'out_channels'][-1], cross_replica=self.cross_replica, mybn=
            self.mybn), self.activation, self.which_conv(self.arch[
            'out_channels'][-1], 3))
        if not skip_init:
            self.init_weights()
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            NoneNone
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear
                ) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.
                    parameters()])
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
    arch[256] = {'in_channels': [(item * ch) for item in [1, 2, 4, 8, 8, 16
        ]], 'out_channels': [(item * ch) for item in [2, 4, 8, 8, 16, 16]],
        'downsample': [True] * 6 + [False], 'resolution': [128, 64, 32, 16,
        8, 4, 4], 'attention': {(2 ** i): (2 ** i in [int(item) for item in
        attention.split('_')]) for i in range(2, 8)}}
    arch[128] = {'in_channels': [(item * ch) for item in [1, 2, 4, 8, 16]],
        'out_channels': [(item * ch) for item in [2, 4, 8, 16, 16]],
        'downsample': [True] * 5 + [False], 'resolution': [64, 32, 16, 8, 4,
        4], 'attention': {(2 ** i): (2 ** i in [int(item) for item in
        attention.split('_')]) for i in range(2, 8)}}
    arch[64] = {'in_channels': [(item * ch) for item in [1, 2, 4, 8]],
        'out_channels': [(item * ch) for item in [2, 4, 8, 16]],
        'downsample': [True] * 4 + [False], 'resolution': [32, 16, 8, 4, 4],
        'attention': {(2 ** i): (2 ** i in [int(item) for item in attention
        .split('_')]) for i in range(2, 7)}}
    arch[32] = {'in_channels': [(item * ch) for item in [4, 4, 4]],
        'out_channels': [(item * ch) for item in [4, 4, 4]], 'downsample':
        [True, True, False, False], 'resolution': [16, 16, 16, 16],
        'attention': {(2 ** i): (2 ** i in [int(item) for item in attention
        .split('_')]) for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128, D_kernel_size=
        3, D_attn='64', n_classes=1000, num_D_SVs=1, num_D_SV_itrs=1,
        D_activation=nn.ReLU(inplace=False), D_lr=0.0002, D_B1=0.0, D_B2=
        0.999, adam_eps=1e-08, SN_eps=1e-12, output_dim=1,
        D_mixed_precision=False, D_fp16=False, D_init='ortho', skip_init=
        False, D_param='SN', **kwargs):
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
            self.which_conv = functools.partial(layers.SNConv2d,
                kernel_size=3, padding=1, num_svs=num_D_SVs, num_itrs=
                num_D_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=
                num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch[
                'in_channels'][index], out_channels=self.arch[
                'out_channels'][index], which_conv=self.which_conv, wide=
                self.D_wide, activation=self.activation, preactivation=
                index > 0, downsample=nn.AvgPool2d(2) if self.arch[
                'downsample'][index] else None)]]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch[
                    'out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self
            .blocks])
        self.linear = self.which_linear(self.arch['out_channels'][-1],
            output_dim)
        self.embed = self.which_embedding(self.n_classes, self.arch[
            'out_channels'][-1])
        if not skip_init:
            self.init_weights()
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            NoneNone
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear
                ) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.
                    parameters()])
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


class G_D(nn.Module):

    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=
        False, split_D=False):
        with torch.set_grad_enabled(train_G):
            G_z = self.G(z, self.G.shared(gy))
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            elif return_G_z:
                return D_fake, G_z
            else:
                return D_fake
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
            elif return_G_z:
                return D_out, G_z
            else:
                return D_out


class Generator(nn.Module):

    def __init__(self, G_ch=64, G_depth=2, dim_z=128, bottom_width=4,
        resolution=128, G_kernel_size=3, G_attn='64', n_classes=1000,
        num_G_SVs=1, num_G_SV_itrs=1, G_shared=True, shared_dim=0, hier=
        False, cross_replica=False, mybn=False, G_activation=nn.ReLU(
        inplace=False), G_lr=5e-05, G_B1=0.0, G_B2=0.999, adam_eps=1e-08,
        BN_eps=1e-05, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
        G_init='ortho', skip_init=False, no_optim=False, G_param='SN',
        norm_style='bn', **kwargs):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.G_depth = G_depth
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
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                kernel_size=3, padding=1, num_svs=num_G_SVs, num_itrs=
                num_G_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=
                num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3,
                padding=1)
            self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding
        bn_linear = functools.partial(self.which_linear, bias=False
            ) if self.G_shared else self.which_embedding
        self.which_bn = functools.partial(layers.ccbn, which_linear=
            bn_linear, cross_replica=self.cross_replica, mybn=self.mybn,
            input_size=self.shared_dim + self.dim_z if self.G_shared else
            self.n_classes, norm_style=self.norm_style, eps=self.BN_eps)
        self.shared = self.which_embedding(n_classes, self.shared_dim
            ) if G_shared else layers.identity()
        self.linear = self.which_linear(self.dim_z + self.shared_dim, self.
            arch['in_channels'][0] * self.bottom_width ** 2)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][
                index], out_channels=self.arch['in_channels'][index] if 
                g_index == 0 else self.arch['out_channels'][index],
                which_conv=self.which_conv, which_bn=self.which_bn,
                activation=self.activation, upsample=functools.partial(F.
                interpolate, scale_factor=2) if self.arch['upsample'][index
                ] and g_index == self.G_depth - 1 else None)] for g_index in
                range(self.G_depth)]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch[
                    'out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self
            .blocks])
        self.output_layer = nn.Sequential(layers.bn(self.arch[
            'out_channels'][-1], cross_replica=self.cross_replica, mybn=
            self.mybn), self.activation, self.which_conv(self.arch[
            'out_channels'][-1], 3))
        if not skip_init:
            self.init_weights()
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            NoneNone
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear
                ) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.
                    parameters()])
        None

    def forward(self, z, y):
        if self.hier:
            z = torch.cat([y, z], 1)
            y = z
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, y)
        return torch.tanh(self.output_layer(h))


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, D_depth=2, resolution=128,
        D_kernel_size=3, D_attn='64', n_classes=1000, num_D_SVs=1,
        num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), D_lr=0.0002,
        D_B1=0.0, D_B2=0.999, adam_eps=1e-08, SN_eps=1e-12, output_dim=1,
        D_mixed_precision=False, D_fp16=False, D_init='ortho', skip_init=
        False, D_param='SN', **kwargs):
        super(Discriminator, self).__init__()
        self.ch = D_ch
        self.D_wide = D_wide
        self.D_depth = D_depth
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
            self.which_conv = functools.partial(layers.SNConv2d,
                kernel_size=3, padding=1, num_svs=num_D_SVs, num_itrs=
                num_D_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=
                num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
        self.input_conv = self.which_conv(3, self.arch['in_channels'][0])
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][
                index] if d_index == 0 else self.arch['out_channels'][index
                ], out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv, wide=self.D_wide, activation=
                self.activation, preactivation=True, downsample=nn.
                AvgPool2d(2) if self.arch['downsample'][index] and d_index ==
                0 else None) for d_index in range(self.D_depth)]]
            if self.arch['attention'][self.arch['resolution'][index]]:
                None
                self.blocks[-1] += [layers.Attention(self.arch[
                    'out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self
            .blocks])
        self.linear = self.which_linear(self.arch['out_channels'][-1],
            output_dim)
        self.embed = self.which_embedding(self.n_classes, self.arch[
            'out_channels'][-1])
        if not skip_init:
            self.init_weights()
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            NoneNone
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear
                ) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    None
                self.param_count += sum([p.data.nelement() for p in module.
                    parameters()])
        None

    def forward(self, x, y=None):
        h = self.input_conv(x)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        out = self.linear(h)
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


class G_D(nn.Module):

    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=
        False, split_D=False):
        with torch.set_grad_enabled(train_G):
            G_z = self.G(z, self.G.shared(gy))
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            elif return_G_z:
                return D_fake, G_z
            else:
                return D_fake
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
            elif return_G_z:
                return D_out, G_z
            else:
                return D_out


def l2normalize(v, eps=0.0001):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))
        sigma = u.dot(_w.mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + '_u')
            getattr(self.module, self.name + '_v')
            getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """ Self Attention Layer"""

    def __init__(self, in_dim, activation=F.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim,
            out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=
            in_dim // 8, kernel_size=1, bias=False))
        self.pool = nn.MaxPool2d(2, 2)
        self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=
            in_dim // 2, kernel_size=1, bias=False))
        self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2,
            out_channels=in_dim, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        N = height * width
        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = phi.view(m_batchsize, -1, N // 4)
        theta = theta.view(m_batchsize, -1, N)
        theta = theta.permute(0, 2, 1)
        attention = self.softmax(torch.bmm(theta, phi))
        g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
        attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize,
            -1, width, height)
        out = self.o_conv(attn_g)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes, eps=0.0001, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps,
            momentum=momentum)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features,
            bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features,
            bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1,
            self.num_features, 1, 1)
        return out


class GBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3], padding
        =1, stride=1, n_class=None, bn=True, activation=F.relu, upsample=
        True, downsample=False, z_dim=148):
        super().__init__()
        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
            kernel_size, stride, padding, bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
            kernel_size, stride, padding, bias=True if bn else True))
        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 
                1, 1, 0))
            self.skip_proj = True
        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
            self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)

    def forward(self, input, condition=None):
        out = input
        if self.bn:
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip


class Generator128(nn.Module):

    def __init__(self, code_dim=120, n_class=1000, chn=96, debug=False):
        super().__init__()
        self.linear = nn.Linear(n_class, 128, bias=False)
        if debug:
            chn = 8
        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))
        z_dim = code_dim + 28
        self.GBlock = nn.ModuleList([GBlock(16 * chn, 16 * chn, n_class=
            n_class, z_dim=z_dim), GBlock(16 * chn, 8 * chn, n_class=
            n_class, z_dim=z_dim), GBlock(8 * chn, 4 * chn, n_class=n_class,
            z_dim=z_dim), GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=
            z_dim), GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim)])
        self.sa_id = 4
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(2 * chn)
        self.ScaledCrossReplicaBN = nn.BatchNorm2d(1 * chn, eps=0.0001)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.chunk(input, self.num_split, 1)
        class_emb = self.linear(class_id)
        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)
        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)


class Generator256(nn.Module):

    def __init__(self, code_dim=140, n_class=1000, chn=96, debug=False):
        super().__init__()
        self.linear = nn.Linear(n_class, 128, bias=False)
        if debug:
            chn = 8
        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))
        self.GBlock = nn.ModuleList([GBlock(16 * chn, 16 * chn, n_class=
            n_class), GBlock(16 * chn, 8 * chn, n_class=n_class), GBlock(8 *
            chn, 8 * chn, n_class=n_class), GBlock(8 * chn, 4 * chn,
            n_class=n_class), GBlock(4 * chn, 2 * chn, n_class=n_class),
            GBlock(2 * chn, 1 * chn, n_class=n_class)])
        self.sa_id = 5
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(2 * chn)
        self.ScaledCrossReplicaBN = nn.BatchNorm2d(1 * chn, eps=0.0001)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.chunk(input, self.num_split, 1)
        class_emb = self.linear(class_id)
        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)
        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)


class Generator512(nn.Module):

    def __init__(self, code_dim=128, n_class=1000, chn=96, debug=False):
        super().__init__()
        self.linear = nn.Linear(n_class, 128, bias=False)
        if debug:
            chn = 8
        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(16, 4 * 4 * 16 * chn))
        z_dim = code_dim + 16
        self.GBlock = nn.ModuleList([GBlock(16 * chn, 16 * chn, n_class=
            n_class, z_dim=z_dim), GBlock(16 * chn, 8 * chn, n_class=
            n_class, z_dim=z_dim), GBlock(8 * chn, 8 * chn, n_class=n_class,
            z_dim=z_dim), GBlock(8 * chn, 4 * chn, n_class=n_class, z_dim=
            z_dim), GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=z_dim),
            GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim), GBlock(
            1 * chn, 1 * chn, n_class=n_class, z_dim=z_dim)])
        self.sa_id = 4
        self.num_split = len(self.GBlock) + 1
        self.attention = SelfAttention(4 * chn)
        self.ScaledCrossReplicaBN = nn.BatchNorm2d(1 * chn)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.chunk(input, self.num_split, 1)
        class_emb = self.linear(class_id)
        out = self.G_linear(codes[0])
        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            out = GBlock(out, condition)
        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return torch.tanh(out)


class Discriminator(nn.Module):

    def __init__(self, n_class=1000, chn=96, debug=False):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel, bn=False, upsample=False,
                downsample=downsample)
        if debug:
            chn = 8
        self.debug = debug
        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 1 * chn, 3,
            padding=1)), nn.ReLU(), SpectralNorm(nn.Conv2d(1 * chn, 1 * chn,
            3, padding=1)), nn.AvgPool2d(2))
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 1 * chn, 1))
        self.conv = nn.Sequential(conv(1 * chn, 1 * chn, downsample=True),
            conv(1 * chn, 2 * chn, downsample=True), SelfAttention(2 * chn),
            conv(2 * chn, 2 * chn, downsample=True), conv(2 * chn, 4 * chn,
            downsample=True), conv(4 * chn, 8 * chn, downsample=True), conv
            (8 * chn, 8 * chn, downsample=True), conv(8 * chn, 16 * chn,
            downsample=True), conv(16 * chn, 16 * chn, downsample=False))
        self.linear = SpectralNorm(nn.Linear(16 * chn, 1))
        self.embed = nn.Embedding(n_class, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = SpectralNorm(self.embed)

    def forward(self, input, class_id):
        out = self.pre_conv(input)
        out += self.pre_skip(F.avg_pool2d(input, 2))
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)
        return out_linear + prod


class WrapInception(nn.Module):

    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
            requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
            requires_grad=False)

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear',
                align_corners=True)
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size
            (0), -1))
        return pool, logits


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

    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps
        =1e-12):
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
            svs, us, vs = power_iteration(W_mat, self.u, update=self.
                training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, num_svs=1, num_itrs=1,
        eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, self.padding,
            self.dilation, self.groups)


class SNLinear(nn.Linear, SN):

    def __init__(self, in_features, out_features, bias=True, num_svs=1,
        num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
        max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False,
        _weight=None, num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim,
            padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
            _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


class Attention(nn.Module):

    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1,
            padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1,
            padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1,
            padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1,
            padding=0, bias=False)
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2,
            x.shape[2], x.shape[3]))
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
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True,
                eps=self.eps)
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum
                    ) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum
                    ) + var * self.momentum
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

    def __init__(self, output_size, input_size, which_linear, eps=1e-05,
        momentum=0.1, cross_replica=False, mybn=False, norm_style='bn'):
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
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.
                momentum, affine=False)
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
                out = F.batch_norm(x, self.stored_mean, self.stored_var,
                    None, None, self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var,
                    None, None, self.training, 0.1, self.eps)
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

    def __init__(self, output_size, eps=1e-05, momentum=0.1, cross_replica=
        False, mybn=False):
        super(bn, self).__init__()
        self.output_size = output_size
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.
                momentum, affine=False)
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
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.
                gain, self.bias, self.training, self.momentum, self.eps)


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d,
        which_bn=bn, activation=None, upsample=None):
        super(GBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                kernel_size=1, padding=0)
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

    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide
        =True, preactivation=False, activation=None, downsample=None):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = (True if in_channels != out_channels or
            downsample else False)
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                kernel_size=1, padding=0)

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


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)
