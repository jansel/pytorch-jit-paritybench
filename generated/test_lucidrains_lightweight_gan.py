import sys
_module = sys.modules[__name__]
del sys
lightweight_gan = _module
cli = _module
diff_augment = _module
diff_augment_test = _module
lightweight_gan = _module
version = _module
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


import random


from functools import wraps


import torch


import torch.multiprocessing as mp


import torch.distributed as dist


import numpy as np


import torch.nn.functional as F


import torchvision


from torch import nn


from torch.utils.data import DataLoader


from random import random


import math


from math import log2


from math import floor


from functools import lru_cache


from functools import partial


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.optim import Adam


from torch import einsum


from torch.utils.data import Dataset


from torch.autograd import grad as torch_grad


from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


from torchvision import transforms


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RandomApply(nn.Module):

    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class ChanNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):

    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class Blur(nn.Module):

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


def exists(val):
    return val is not None


class Noise(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        b, _, h, w, device = *x.shape, x.device
        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device=device)
        return x + self.weight * noise


class DepthWiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias), nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias))

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, kernel_size=3):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.kernel_size = kernel_size
        self.nonlin = nn.GELU()
        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=1, bias=False)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        lin_q, lin_k, lin_v = self.to_lin_q(fmap), *self.to_lin_kv(fmap).chunk(2, dim=1)
        lin_q, lin_k, lin_v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (lin_q, lin_k, lin_v))
        lin_q = lin_q.softmax(dim=-1)
        lin_k = lin_k.softmax(dim=-2)
        lin_q = lin_q * self.scale
        context = einsum('b n d, b n e -> b d e', lin_k, lin_v)
        lin_out = einsum('b n d, b d e -> b n e', lin_q, context)
        lin_out = rearrange(lin_out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)
        q, k, v = self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h=h), (q, k, v))
        k = F.unfold(k, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        v = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        k, v = map(lambda t: rearrange(t, 'b (d j) n -> b n j d', d=self.dim_head), (k, v))
        q = rearrange(q, 'b c ... -> b (...) c') * self.scale
        sim = einsum('b i d, b i j d -> b i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        full_out = einsum('b i j, b i j d -> b i d', attn, v)
        full_out = rearrange(full_out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)
        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim=1)
        return self.to_out(out)


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
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(cutout_size[0], dtype=torch.long, device=x.device), torch.arange(cutout_size[1], dtype=torch.long, device=x.device), indexing='ij')
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)
    imgs = []
    for img in x.unbind(dim=0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)
        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v
        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)
        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)
        imgs.append(img)
    return torch.stack(imgs)


def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)


def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(x.size(2), dtype=torch.long, device=x.device), torch.arange(x.size(3), dtype=torch.long, device=x.device), indexing='ij')
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


AUGMENT_FNS = {'color': [rand_brightness, rand_saturation, rand_contrast], 'offset': [rand_offset], 'offset_h': [rand_offset_h], 'offset_v': [rand_offset_v], 'translation': [rand_translation], 'cutout': [rand_cutout]}


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):

    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0.0, types=[], detach=False, **kwargs):
        context = torch.no_grad if detach else null_context
        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)
        return self.D(images, **kwargs)


def default(val, d):
    return val if exists(val) else d


class PixelShuffleUpsample(nn.Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class GlobalContext(nn.Module):

    def __init__(self, *, chan_in, chan_out):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)
        self.net = nn.Sequential(nn.Conv2d(chan_in, chan_intermediate, 1), nn.LeakyReLU(0.1), nn.Conv2d(chan_intermediate, chan_out, 1), nn.Sigmoid())

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part:(i + 1) * c_part, x, y] = coor_value
    return dct_weights


class FCANet(nn.Module):

    def __init__(self, *, chan_in, chan_out, reduction=4, width):
        super().__init__()
        freq_w, freq_h = [0] * 8, list(range(8))
        dct_weights = get_dct_weights(width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)
        chan_intermediate = max(3, chan_out // reduction)
        self.net = nn.Sequential(nn.Conv2d(chan_in, chan_intermediate, 1), nn.LeakyReLU(0.1), nn.Conv2d(chan_intermediate, chan_out, 1), nn.Sigmoid())

    def forward(self, x):
        x = reduce(x * self.dct_weights, 'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1=1, w1=1)
        return self.net(x)


def Conv2dSame(dim_in, dim_out, kernel_size, bias=True):
    pad_left = kernel_size // 2
    pad_right = pad_left - 1 if kernel_size % 2 == 0 else pad_left
    return nn.Sequential(nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)), nn.Conv2d(dim_in, dim_out, kernel_size, bias=bias))


def is_power_of_two(val):
    return log2(val).is_integer()


norm_class = nn.BatchNorm2d


class Generator(nn.Module):

    def __init__(self, *, image_size, latent_dim=256, fmap_max=512, fmap_inverse_coef=12, transparent=False, greyscale=False, attn_res_layers=[], freq_chan_attn=False):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3
        fmap_max = default(fmap_max, latent_dim)
        self.initial_conv = nn.Sequential(nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4), norm_class(latent_dim * 2), nn.GLU(dim=1))
        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]
        in_out_features = list(zip(features[:-1], features[1:]))
        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))
        self.sle_map = (3, 7), (4, 8), (5, 9), (6, 10)
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)
        self.num_layers_spatial_res = 1
        for res, (chan_in, chan_out) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res
            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))
            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]
                if freq_chan_attn:
                    sle = FCANet(chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1))
                else:
                    sle = GlobalContext(chan_in=chan_out, chan_out=sle_chan_out)
            layer = nn.ModuleList([nn.Sequential(PixelShuffleUpsample(chan_in), Blur(), Conv2dSame(chan_in, chan_out * 2, 4), Noise(), norm_class(chan_out * 2), nn.GLU(dim=1)), sle, attn])
            self.layers.append(layer)
        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)
        residuals = dict()
        for res, (up, sle, attn) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x
            x = up(x)
            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual
            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]
        return self.out_conv(x)


class SimpleDecoder(nn.Module):

    def __init__(self, *, chan_in, chan_out=3, num_upsamples=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in
        for ind in range(num_upsamples):
            last_layer = ind == num_upsamples - 1
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(PixelShuffleUpsample(chans), nn.Conv2d(chans, chan_out, 3, padding=1), nn.GLU(dim=1))
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def SPConvDownsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2), nn.Conv2d(dim * 4, dim_out, 1))


class Discriminator(nn.Module):

    def __init__(self, *, image_size, fmap_max=512, fmap_inverse_coef=12, transparent=False, greyscale=False, disc_output_size=5, attn_res_layers=[]):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'
        resolution = int(resolution)
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3
        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3
        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = res, init_channel
        chan_in_out = list(zip(features[:-1], features[1:]))
        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == num_non_residual_layers - 1
            chan_out = features[0][-1] if last_layer else init_channel
            self.non_residual_layers.append(nn.Sequential(Blur(), nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1), nn.LeakyReLU(0.1)))
        self.residual_layers = nn.ModuleList([])
        for res, ((_, chan_in), (_, chan_out)) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res
            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))
            self.residual_layers.append(nn.ModuleList([SumBranches([nn.Sequential(Blur(), SPConvDownsample(chan_in, chan_out), nn.LeakyReLU(0.1), nn.Conv2d(chan_out, chan_out, 3, padding=1), nn.LeakyReLU(0.1)), nn.Sequential(Blur(), nn.AvgPool2d(2), nn.Conv2d(chan_in, chan_out, 1), nn.LeakyReLU(0.1))]), attn]))
        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(nn.Conv2d(last_chan, last_chan, 1), nn.LeakyReLU(0.1), nn.Conv2d(last_chan, 1, 4))
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(Blur(), nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1), nn.LeakyReLU(0.1), nn.Conv2d(last_chan, 1, 4))
        self.to_shape_disc_out = nn.Sequential(nn.Conv2d(init_channel, 64, 3, padding=1), Residual(PreNorm(64, LinearAttention(64))), SumBranches([nn.Sequential(Blur(), SPConvDownsample(64, 32), nn.LeakyReLU(0.1), nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.1)), nn.Sequential(Blur(), nn.AvgPool2d(2), nn.Conv2d(64, 32, 1), nn.LeakyReLU(0.1))]), Residual(PreNorm(32, LinearAttention(32))), nn.AdaptiveAvgPool2d((4, 4)), nn.Conv2d(32, 1, 4))
        self.decoder1 = SimpleDecoder(chan_in=last_chan, chan_out=init_channel)
        self.decoder2 = SimpleDecoder(chan_in=features[-2][-1], chan_out=init_channel) if resolution >= 9 else None

    def forward(self, x, calc_aux_loss=False):
        orig_img = x
        for layer in self.non_residual_layers:
            x = layer(x)
        layer_outputs = []
        for net, attn in self.residual_layers:
            if exists(attn):
                x = attn(x) + x
            x = net(x)
            layer_outputs.append(x)
        out = self.to_logits(x).flatten(1)
        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)
        if not calc_aux_loss:
            return out, out_32x32, None
        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]
        recon_img_8x8 = self.decoder1(layer_8x8)
        aux_loss = F.mse_loss(recon_img_8x8, F.interpolate(orig_img, size=recon_img_8x8.shape[2:]))
        if exists(self.decoder2):
            select_random_quadrant = lambda rand_quadrant, img: rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))
            recon_img_16x16 = self.decoder2(layer_16x16_part)
            aux_loss_16x16 = F.mse_loss(recon_img_16x16, F.interpolate(img_part, size=recon_img_16x16.shape[2:]))
            aux_loss = aux_loss + aux_loss_16x16
        return out, out_32x32, aux_loss


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


class LightweightGAN(nn.Module):

    def __init__(self, *, latent_dim, image_size, optimizer='adam', fmap_max=512, fmap_inverse_coef=12, transparent=False, greyscale=False, disc_output_size=5, attn_res_layers=[], freq_chan_attn=False, ttur_mult=1.0, lr=0.0002, rank=0, ddp=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        G_kwargs = dict(image_size=image_size, latent_dim=latent_dim, fmap_max=fmap_max, fmap_inverse_coef=fmap_inverse_coef, transparent=transparent, greyscale=greyscale, attn_res_layers=attn_res_layers, freq_chan_attn=freq_chan_attn)
        self.G = Generator(**G_kwargs)
        self.D = Discriminator(image_size=image_size, fmap_max=fmap_max, fmap_inverse_coef=fmap_inverse_coef, transparent=transparent, greyscale=greyscale, attn_res_layers=attn_res_layers, disc_output_size=disc_output_size)
        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)
        if optimizer == 'adam':
            self.G_opt = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr=lr * ttur_mult, betas=(0.5, 0.9))
        elif optimizer == 'adabelief':
            self.G_opt = AdaBelief(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(), lr=lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, 'No valid optimizer is given'
        self.apply(self._init_weights)
        self.reset_parameter_averaging()
        self
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):

        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)
            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChanNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthWiseConv2d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalContext,
     lambda: ([], {'chan_in': 4, 'chan_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Noise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SumBranches,
     lambda: ([], {'branches': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_lightweight_gan(_paritybench_base):
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

