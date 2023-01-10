import sys
_module = sys.modules[__name__]
del sys
cfg_modify_image = _module
cfg_sample = _module
clip_sample = _module
diffusion = _module
models = _module
cc12m_1 = _module
danbooru_128 = _module
imagenet_128 = _module
wikiart_128 = _module
wikiart_256 = _module
yfcc_1 = _module
yfcc_2 = _module
sampling = _module
utils = _module
make_grid = _module
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


from functools import partial


import torch


from torch import nn


from torch.nn import functional as F


from torchvision import transforms


from torchvision.transforms import functional as TF


import math


class MakeCutouts(nn.Module):

    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


class ResidualBlock(nn.Module):

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):

    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([nn.Linear(f_in, f_mid), nn.ReLU(inplace=True), nn.Linear(f_mid, f_out), nn.ReLU(inplace=True) if not is_last else nn.Identity()], skip)


class Modulation2d(nn.Module):

    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ResModConvBlock(ResidualBlock):

    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([nn.Conv2d(c_in, c_mid, 3, padding=1), nn.GroupNorm(1, c_mid, affine=False), Modulation2d(state, feats_in, c_mid), nn.ReLU(inplace=True), nn.Conv2d(c_mid, c_out, 3, padding=1), nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(), Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(), nn.ReLU(inplace=True) if not is_last else nn.Identity()], skip)


class SkipBlock(nn.Module):

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):

    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = (q * scale @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class CC12M1Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 256, 256
        self.clip_model = 'ViT-B/16'
        self.min_t = 0.0
        self.max_t = 1.0
        c = 128
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]
        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(ResLinearBlock(512 + 128, 1024, 1024), ResLinearBlock(1024, 1024, 1024, is_last=True))
        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5 ** 0.5
        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.net = nn.Sequential(conv_block(3 + 16, cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), SkipBlock([self.down, conv_block(cs[0], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), SkipBlock([self.down, conv_block(cs[1], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), SkipBlock([self.down, conv_block(cs[2], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), SkipBlock([self.down, conv_block(cs[3], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), SkipBlock([self.down, conv_block(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), SkipBlock([self.down, conv_block(cs[5], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), conv_block(cs[6], cs[6], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), self.up]), conv_block(cs[5] * 2, cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), conv_block(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), self.up]), conv_block(cs[4] * 2, cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), conv_block(cs[4], cs[4], cs[3]), SelfAttention2d(cs[3], cs[3] // 64), self.up]), conv_block(cs[3] * 2, cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[3]), conv_block(cs[3], cs[3], cs[2]), self.up]), conv_block(cs[2] * 2, cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[2]), conv_block(cs[2], cs[2], cs[1]), self.up]), conv_block(cs[1] * 2, cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[1]), conv_block(cs[1], cs[1], cs[0]), self.up]), conv_block(cs[0] * 2, cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], cs[0]), conv_block(cs[0], cs[0], 3, is_last=True))
        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5 ** 0.5

    def forward(self, input, t, clip_embed):
        clip_embed = F.normalize(clip_embed, dim=-1) * clip_embed.shape[-1] ** 0.5
        mapping_timestep_embed = self.mapping_timestep_embed(t[:, None])
        self.state['cond'] = self.mapping(torch.cat([clip_embed, mapping_timestep_embed], dim=1))
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out


class ResConvBlock(ResidualBlock):

    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([nn.Conv2d(c_in, c_mid, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c_mid, c_out, 3, padding=1), nn.ReLU(inplace=True) if not is_last else nn.Identity()], skip)


class Danbooru128Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 128, 128
        self.min_t = utils.get_ddpm_schedule(torch.tensor(0.0)).item()
        self.max_t = utils.get_ddpm_schedule(torch.tensor(1.0)).item()
        c = 256
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.net = nn.Sequential(ResConvBlock(3 + 16, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), SkipBlock([self.down, ResConvBlock(cs[0], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), SkipBlock([self.down, ResConvBlock(cs[1], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), SkipBlock([self.down, ResConvBlock(cs[2], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), SkipBlock([self.down, ResConvBlock(cs[3], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), SkipBlock([self.down, ResConvBlock(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), self.up]), ResConvBlock(cs[4] * 2, cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), self.up]), ResConvBlock(cs[3] * 2, cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[2]), SelfAttention2d(cs[2], cs[2] // 128), self.up]), ResConvBlock(cs[2] * 2, cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[1]), self.up]), ResConvBlock(cs[1] * 2, cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[0]), self.up]), ResConvBlock(cs[0] * 2, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], 3))

    def forward(self, input, t):
        log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(self.timestep_embed(log_snr[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


class ImageNet128Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 128, 128
        self.min_t = utils.get_ddpm_schedule(torch.tensor(0.0)).item()
        self.max_t = utils.get_ddpm_schedule(torch.tensor(1.0)).item()
        c = 128
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.net = nn.Sequential(ResConvBlock(3 + 16, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), SkipBlock([self.down, ResConvBlock(cs[0], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), SkipBlock([self.down, ResConvBlock(cs[1], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), SkipBlock([self.down, ResConvBlock(cs[2], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), SkipBlock([self.down, ResConvBlock(cs[3], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), SkipBlock([self.down, ResConvBlock(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), self.up]), ResConvBlock(cs[4] * 2, cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), self.up]), ResConvBlock(cs[3] * 2, cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), ResConvBlock(cs[3], cs[3], cs[2]), SelfAttention2d(cs[2], cs[2] // 128), self.up]), ResConvBlock(cs[2] * 2, cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[1]), self.up]), ResConvBlock(cs[1] * 2, cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[0]), self.up]), ResConvBlock(cs[0] * 2, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], 3, is_last=True))

    def forward(self, input, t):
        log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(self.timestep_embed(log_snr[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


class WikiArt128Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 128, 128
        self.min_t = utils.get_ddpm_schedule(torch.tensor(0.0)).item()
        self.max_t = utils.get_ddpm_schedule(torch.tensor(1.0)).item()
        c = 128
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.net = nn.Sequential(ResConvBlock(3 + 16, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), SkipBlock([self.down, ResConvBlock(cs[0], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), SkipBlock([self.down, ResConvBlock(cs[1], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), SkipBlock([self.down, ResConvBlock(cs[2], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), SkipBlock([self.down, ResConvBlock(cs[3], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), SkipBlock([self.down, ResConvBlock(cs[4], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[5]), ResConvBlock(cs[5], cs[5], cs[4]), self.up]), ResConvBlock(cs[4] * 2, cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[3]), self.up]), ResConvBlock(cs[3] * 2, cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[2]), self.up]), ResConvBlock(cs[2] * 2, cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[1]), self.up]), ResConvBlock(cs[1] * 2, cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[0]), self.up]), ResConvBlock(cs[0] * 2, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], 3))

    def forward(self, input, t):
        log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(self.timestep_embed(log_snr[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


class WikiArt256Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 256, 256
        self.min_t = utils.get_ddpm_schedule(torch.tensor(0.0)).item()
        self.max_t = utils.get_ddpm_schedule(torch.tensor(1.0)).item()
        c = 128
        cs = [c // 2, c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.net = nn.Sequential(ResConvBlock(3 + 16, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), SkipBlock([self.down, ResConvBlock(cs[0], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), SkipBlock([self.down, ResConvBlock(cs[1], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), SkipBlock([self.down, ResConvBlock(cs[2], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), SkipBlock([self.down, ResConvBlock(cs[3], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), SkipBlock([self.down, ResConvBlock(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), SkipBlock([self.down, ResConvBlock(cs[5], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 128), ResConvBlock(cs[6], cs[6], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), self.up]), ResConvBlock(cs[5] * 2, cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 128), ResConvBlock(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), self.up]), ResConvBlock(cs[4] * 2, cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[4]), SelfAttention2d(cs[4], cs[4] // 128), ResConvBlock(cs[4], cs[4], cs[3]), SelfAttention2d(cs[3], cs[3] // 128), self.up]), ResConvBlock(cs[3] * 2, cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[2]), self.up]), ResConvBlock(cs[2] * 2, cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[1]), self.up]), ResConvBlock(cs[1] * 2, cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[0]), self.up]), ResConvBlock(cs[0] * 2, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), ResConvBlock(cs[0], cs[0], 3, is_last=True))

    def forward(self, input, t):
        log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(self.timestep_embed(log_snr[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


class YFCC1Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 512, 512
        self.min_t = 0.0
        self.max_t = 1.0
        c = 128
        self.timestep_embed = FourierFeatures(1, 16)
        self.net = nn.Sequential(ResConvBlock(3 + 16, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c * 2, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 4), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c * 4, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), SkipBlock([nn.AvgPool2d(2), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 16, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 8), SelfAttention2d(c * 8, c * 8 // 64), ResConvBlock(c * 8, c * 8, c * 4), SelfAttention2d(c * 4, c * 4 // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 8, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), ResConvBlock(c * 4, c * 4, c * 4), SelfAttention2d(c * 4, c * 4 // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 8, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 4), ResConvBlock(c * 4, c * 4, c * 2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 4, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 4, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c * 2), ResConvBlock(c * 2, c * 2, c), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 2, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(c * 2, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, c), ResConvBlock(c, c, 3, dropout_last=False))

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


class YFCC2Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape = 3, 512, 512
        self.min_t = 0.0
        self.max_t = 1.0
        c = 256
        cs = [c // 2, c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]
        self.timestep_embed = FourierFeatures(1, 16)
        self.net = nn.Sequential(ResConvBlock(3 + 16, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], cs[0]), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[0], cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[1]), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[1], cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[2]), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[2], cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[3]), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[3], cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[4]), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[4], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), ResConvBlock(cs[5], cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[5], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), ResConvBlock(cs[6], cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), SkipBlock([nn.AvgPool2d(2), ResConvBlock(cs[6], cs[7], cs[7]), SelfAttention2d(cs[7], cs[7] // 64), ResConvBlock(cs[7], cs[7], cs[7]), SelfAttention2d(cs[7], cs[7] // 64), ResConvBlock(cs[7], cs[7], cs[7]), SelfAttention2d(cs[7], cs[7] // 64), ResConvBlock(cs[7], cs[7], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[6] * 2, cs[6], cs[6]), SelfAttention2d(cs[6], cs[6] // 64), ResConvBlock(cs[6], cs[6], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[5] * 2, cs[5], cs[5]), SelfAttention2d(cs[5], cs[5] // 64), ResConvBlock(cs[5], cs[5], cs[4]), SelfAttention2d(cs[4], cs[4] // 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[4] * 2, cs[4], cs[4]), ResConvBlock(cs[4], cs[4], cs[3]), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[3] * 2, cs[3], cs[3]), ResConvBlock(cs[3], cs[3], cs[2]), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[2] * 2, cs[2], cs[2]), ResConvBlock(cs[2], cs[2], cs[1]), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[1] * 2, cs[1], cs[1]), ResConvBlock(cs[1], cs[1], cs[0]), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]), ResConvBlock(cs[0] * 2, cs[0], cs[0]), ResConvBlock(cs[0], cs[0], 3, is_last=True))

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FourierFeatures,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MakeCutouts,
     lambda: ([], {'cut_size': 4, 'cutn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResConvBlock,
     lambda: ([], {'c_in': 4, 'c_mid': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResLinearBlock,
     lambda: ([], {'f_in': 4, 'f_mid': 4, 'f_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention2d,
     lambda: ([], {'c_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_crowsonkb_v_diffusion_pytorch(_paritybench_base):
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

