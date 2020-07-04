import sys
_module = sys.modules[__name__]
del sys
cfg = _module
datasets = _module
functions = _module
models = _module
autogan_cifar10_a = _module
autogan_cifar10_b = _module
autogan_cifar10_c = _module
building_blocks = _module
models_search = _module
building_blocks_search = _module
controller = _module
shared_gan = _module
search = _module
test = _module
train = _module
train_derived = _module
utils = _module
cal_fid_stat = _module
fid_score = _module
inception_score = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import logging


from copy import deepcopy


import numpy as np


import torch


import torch.nn as nn


from torchvision.utils import make_grid


from torch import nn


import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, self.bottom_width ** 2 * args.
            gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            0, short_cut=True)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in
            =1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            2, short_cut=False)
        self.to_rgb = nn.Sequential(nn.BatchNorm2d(args.gf_dim), nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)
        return output


class Discriminator(nn.Module):

    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.sum(2).sum(2)
        output = self.l5(h)
        return output


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, self.bottom_width ** 2 * args.
            gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            0, short_cut=True)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            2, short_cut=True)
        self.to_rgb = nn.Sequential(nn.BatchNorm2d(args.gf_dim), nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)
        return output


class Discriminator(nn.Module):

    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.sum(2).sum(2)
        output = self.l5(h)
        return output


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, self.bottom_width ** 2 * args.
            gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            0, short_cut=True, norm='bn')
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in
            =1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=
            2, short_cut=False)
        self.to_rgb = nn.Sequential(nn.BatchNorm2d(args.gf_dim), nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)
        return output


class Discriminator(nn.Module):

    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.sum(2).sum(2)
        output = self.l5(h)
        return output


NORMS = ['in', 'bn']


UP_MODES = ['nearest', 'bilinear']


class Cell(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode, ksize=3,
        num_skip_in=0, short_cut=False, norm=None):
        super(Cell, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize //
            2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=
            ksize // 2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels,
                out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None):
        residual = x
        if self.norm:
            residual = self.n1(residual)
        h = nn.ReLU()(residual)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.
                    up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=
                self.up_mode))
        return h_skip_out, final_out


def _downsample(x):
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1,
        activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, hidden_channels=
        None, ksize=3, pad=1, activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = (in_channels if hidden_channels is None else
            hidden_channels)
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=
            ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


NORM_TYPE = {(0): None, (1): 'bn', (2): 'in'}


UP_TYPE = {(0): 'bilinear', (1): 'nearest', (2): 'deconv'}


class PreGenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_block, ksize=3):
        super(PreGenBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.inn = nn.InstanceNorm2d(in_channels)
        self.up_block = up_block
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels,
            kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=
            ksize // 2)

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(x)
            elif self.norm_type == 'in':
                h = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x
        h = nn.ReLU()(h)
        if self.up_block:
            if self.up_type == 'deconv':
                h = self.deconv(h)
            else:
                h = F.interpolate(h, scale_factor=2, mode=self.up_type)
        out = self.conv(h)
        return out


class PostGenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_block, ksize=3):
        super(PostGenBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels,
            kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, padding=
            ksize // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.inn = nn.InstanceNorm2d(out_channels)
        self.up_block = up_block

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        if self.up_block:
            if self.up_type == 'deconv':
                h = self.deconv(x)
            else:
                h = F.interpolate(x, scale_factor=2, mode=self.up_type)
        else:
            h = x
        h = self.conv(h)
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(h)
            elif self.norm_type == 'in':
                h = self.inn(h)
            else:
                raise NotImplementedError(self.norm_type)
        out = nn.ReLU()(h)
        return out


CONV_TYPE = {(0): 'post', (1): 'pre'}


SHORT_CUT_TYPE = {(0): False, (1): True}


def decimal2binary(n):
    return bin(n).replace('0b', '')


class Cell(nn.Module):

    def __init__(self, in_channels, out_channels, num_skip_in, ksize=3):
        super(Cell, self).__init__()
        self.post_conv1 = PostGenBlock(in_channels, out_channels, ksize=
            ksize, up_block=True)
        self.pre_conv1 = PreGenBlock(in_channels, out_channels, ksize=ksize,
            up_block=True)
        self.post_conv2 = PostGenBlock(out_channels, out_channels, ksize=
            ksize, up_block=False)
        self.pre_conv2 = PreGenBlock(out_channels, out_channels, ksize=
            ksize, up_block=False)
        self.deconv_sc = nn.ConvTranspose2d(in_channels, in_channels,
            kernel_size=2, stride=2)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_deconvx2 = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=2, stride=2)
        self.skip_deconvx4 = nn.Sequential(nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size=2, stride=2), nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2))
        self.num_skip_in = num_skip_in
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(in_channels,
                out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def set_arch(self, conv_id, norm_id, up_id, short_cut_id, skip_ins):
        self.post_conv1.set_arch(up_id, norm_id)
        self.pre_conv1.set_arch(up_id, norm_id)
        self.post_conv2.set_arch(up_id, norm_id)
        self.pre_conv2.set_arch(up_id, norm_id)
        if self.num_skip_in:
            self.skip_ins = [(0) for _ in range(self.num_skip_in)]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)
        self.conv_type = CONV_TYPE[conv_id]
        self.up_type = UP_TYPE[up_id]
        self.short_cut = SHORT_CUT_TYPE[short_cut_id]

    def forward(self, x, skip_ft=None):
        residual = x
        if self.conv_type == 'post':
            h = self.post_conv1(residual)
        elif self.conv_type == 'pre':
            h = self.pre_conv1(residual)
        else:
            raise NotImplementedError(self.norm_type)
        _, _, ht, wt = h.size()
        h_skip_out = h
        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft,
                self.skip_in_ops):
                if skip_flag:
                    if self.up_type != 'deconv':
                        h += skip_in_op(F.interpolate(ft, size=(ht, wt),
                            mode=self.up_type))
                    else:
                        scale = wt // ft.size()[-1]
                        h += skip_in_op(getattr(self,
                            f'skip_deconvx{scale}')(ft))
        if self.conv_type == 'post':
            final_out = self.post_conv2(h)
        elif self.conv_type == 'pre':
            final_out = self.pre_conv2(h)
        else:
            raise NotImplementedError(self.norm_type)
        if self.short_cut:
            if self.up_type != 'deconv':
                final_out += self.c_sc(F.interpolate(x, scale_factor=2,
                    mode=self.up_type))
            else:
                final_out += self.c_sc(self.deconv_sc(x))
        return h_skip_out, final_out


class OptimizedDisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1,
        activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, hidden_channels=
        None, ksize=3, pad=1, activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = (in_channels if hidden_channels is None else
            hidden_channels)
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=
            ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


SKIP_TYPE = {(0): False, (1): True}


class Controller(nn.Module):

    def __init__(self, args, cur_stage):
        """
        init
        :param args:
        :param cur_stage: varies from 0 to ...
        """
        super(Controller, self).__init__()
        self.hid_size = args.hid_size
        self.cur_stage = cur_stage
        self.lstm = torch.nn.LSTMCell(self.hid_size, self.hid_size)
        if cur_stage:
            self.tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE),
                len(SHORT_CUT_TYPE), len(SKIP_TYPE) ** cur_stage]
        else:
            self.tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE),
                len(SHORT_CUT_TYPE)]
        self.encoder = nn.Embedding(sum(self.tokens), self.hid_size)
        self.decoders = nn.ModuleList([nn.Linear(self.hid_size, token) for
            token in self.tokens])

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hid_size, requires_grad=False)

    def forward(self, x, hidden, index):
        if index == 0:
            embed = x
        else:
            embed = self.encoder(x)
        hx, cx = self.lstm(embed, hidden)
        logit = self.decoders[index](hx)
        return logit, (hx, cx)

    def sample(self, batch_size, with_hidden=False, prev_hiddens=None,
        prev_archs=None):
        x = self.initHidden(batch_size)
        if prev_hiddens:
            assert prev_archs
            prev_hxs, prev_cxs = prev_hiddens
            selected_idx = np.random.choice(len(prev_archs), batch_size)
            selected_idx = [int(x) for x in selected_idx]
            selected_archs = []
            selected_hxs = []
            selected_cxs = []
            for s_idx in selected_idx:
                selected_archs.append(prev_archs[s_idx].unsqueeze(0))
                selected_hxs.append(prev_hxs[s_idx].unsqueeze(0))
                selected_cxs.append(prev_cxs[s_idx].unsqueeze(0))
            selected_archs = torch.cat(selected_archs, 0)
            hidden = torch.cat(selected_hxs, 0), torch.cat(selected_cxs, 0)
        else:
            hidden = self.initHidden(batch_size), self.initHidden(batch_size)
        entropies = []
        actions = []
        selected_log_probs = []
        for decode_idx in range(len(self.decoders)):
            logit, hidden = self.forward(x, hidden, decode_idx)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropies.append(-(log_prob * prob).sum(1, keepdim=True))
            action = prob.multinomial(1)
            actions.append(action)
            selected_log_prob = log_prob.gather(1, action.data)
            selected_log_probs.append(selected_log_prob)
            x = action.view(batch_size) + sum(self.tokens[:decode_idx])
            x = x.requires_grad_(False)
        archs = torch.cat(actions, -1)
        selected_log_probs = torch.cat(selected_log_probs, -1)
        entropies = torch.cat(entropies, 0)
        if prev_hiddens:
            archs = torch.cat([selected_archs, archs], -1)
        if with_hidden:
            return archs, selected_log_probs, entropies, hidden
        return archs, selected_log_probs, entropies


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, self.bottom_width ** 2 * args.
            gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, num_skip_in=2)
        self.to_rgb = nn.Sequential(nn.BatchNorm2d(args.gf_dim), nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1), nn.Tanh())

    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        arch_stage1 = arch_id[:4]
        self.cell1.set_arch(conv_id=arch_stage1[0], norm_id=arch_stage1[1],
            up_id=arch_stage1[2], short_cut_id=arch_stage1[3], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[4:9]
            self.cell2.set_arch(conv_id=arch_stage2[0], norm_id=arch_stage2
                [1], up_id=arch_stage2[2], short_cut_id=arch_stage2[3],
                skip_ins=arch_stage2[4])
        if cur_stage == 2:
            arch_stage3 = arch_id[9:]
            self.cell3.set_arch(conv_id=arch_stage3[0], norm_id=arch_stage3
                [1], up_id=arch_stage3[2], short_cut_id=arch_stage3[3],
                skip_ins=arch_stage3[4])

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        if self.cur_stage == 0:
            return self.to_rgb(h1)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        if self.cur_stage == 1:
            return self.to_rgb(h2)
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        if self.cur_stage == 2:
            return self.to_rgb(h3)


class OptimizedDisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1,
        activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):

    def __init__(self, args, in_channels, out_channels, hidden_channels=
        None, ksize=3, pad=1, activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = (in_channels if hidden_channels is None else
            hidden_channels)
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=
            ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):

    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=
            activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)
        self.cur_stage = 0

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        variable_model = nn.Sequential(*layers[:self.cur_stage + 1])
        h = variable_model(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.sum(2).sum(2)
        output = self.l5(h)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_TAMU_VITA_AutoGAN(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DisBlock(*[], **{'args': _mock_config(d_spectral_norm=4), 'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Discriminator(*[], **{'args': _mock_config(df_dim=4, d_spectral_norm=4)}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(OptimizedDisBlock(*[], **{'args': _mock_config(d_spectral_norm=4), 'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

