import sys
_module = sys.modules[__name__]
del sys
setup = _module
stylegan2_pytorch = _module
stylegan2_pytorch = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


from math import floor


from math import log2


from random import random


from functools import partial


import numpy as np


import torch


from torch import nn


from torch.utils import data


import torch.nn.functional as F


from torch.autograd import grad as torch_grad


import torchvision


from torchvision import transforms


class Flatten(nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def is_empty(t):
    return t.nelement() == 0


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(1 - decay, new)


def laplace_smoothing(x, n_categories, eps=1e-05):
    return (x + eps) / (x.sum() + n_categories * eps)


class VectorQuantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.8, commitment=1.0, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        dtype = input.dtype
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))
        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 3, 1, 2)
        return quantize, loss


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class StyleVectorizer(nn.Module):

    def __init__(self, emb, depth):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


EPS = 1e-08


class Conv2DMod(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape
        w1 = y[:, (None), :, (None), (None)]
        w2 = self.weight[(None), :, :, :, :]
        weights = w2 * (w1 + 1)
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x


class RGBBlock(nn.Module):

    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)
        if prev_rgb is not None:
            x = x + prev_rgb
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class GeneratorBlock(nn.Module):

    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)
        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(nn.Module):

    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1)
        self.net = nn.Sequential(nn.Conv2d(input_channels, filters, 3, padding=1), leaky_relu(), nn.Conv2d(filters, filters, 3, padding=1), leaky_relu())
        self.downsample = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = x + res
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Generator(nn.Module):

    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        init_channels = 4 * network_capacity
        self.initial_block = nn.Parameter(torch.randn((init_channels, 4, 4)))
        filters = [init_channels] + [(network_capacity * 2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        in_out_pairs = zip(filters[0:-1], filters[1:])
        self.blocks = nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != self.num_layers - 1
            block = GeneratorBlock(latent_dim, in_chan, out_chan, upsample=not_first, upsample_rgb=not_last, rgba=transparent)
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size
        x = self.initial_block.expand(batch_size, -1, -1, -1)
        styles = styles.transpose(0, 1)
        rgb = None
        for style, block in zip(styles, self.blocks):
            x, rgb = block(x, rgb, style, input_noise)
        return rgb


class Discriminator(nn.Module):

    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, transparent=False):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity * 2 ** i) for i in range(num_layers + 1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))
        blocks = []
        quantize_blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind < len(chan_in_out) - 1
            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)
            quantize_fn = VectorQuantize(out_chan, fq_dict_size) if ind + 1 in fq_layers else None
            quantize_blocks.append(quantize_fn)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)
        self.blocks = nn.ModuleList(blocks)
        latent_dim = 2 * 2 * filters[-1]
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape
        quantize_loss = torch.zeros(1)
        for block, q_block in zip(self.blocks, self.quantize_blocks):
            x = block(x)
            if q_block is not None:
                x, loss = q_block(x)
                quantize_loss += loss
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


def ema_inplace_module(moving_avg_module, new, decay):
    for current_params, ma_params in zip(moving_avg_module.parameters(), new.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ema_inplace(old_weight, up_weight, decay)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


class StyleGAN2(nn.Module):

    def __init__(self, image_size, latent_dim=512, style_depth=8, network_capacity=16, transparent=False, fp16=False, cl_reg=False, steps=1, lr=0.0001, fq_layers=[], fq_dict_size=256):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_decay = 0.995
        self.S = StyleVectorizer(latent_dim, style_depth)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent)
        self.D = Discriminator(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size, transparent=transparent)
        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent)
        self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten') if cl_reg else None
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = DiffGrad(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = DiffGrad(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self._init_weights()
        self.reset_parameter_averaging()
        self
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O2')

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        ema_inplace_module(self.SE, self.S, self.ema_decay)
        ema_inplace_module(self.GE, self.G, self.ema_decay)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'image_size': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (DiscriminatorBlock,
     lambda: ([], {'input_channels': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RGBBlock,
     lambda: ([], {'latent_dim': 4, 'input_channel': 4, 'upsample': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (StyleVectorizer,
     lambda: ([], {'emb': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VectorQuantize,
     lambda: ([], {'dim': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_stylegan2_pytorch(_paritybench_base):
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

