import sys
_module = sys.modules[__name__]
del sys
eval_vae = _module
callbacks = _module
loader = _module
vqvae = _module
lpips = _module
quantize = _module
vae = _module
util = _module
train_vae = _module

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


import torch


from typing import Any


from typing import Optional


from typing import Tuple


import torch.nn.functional as F


import torchvision


import torchvision.transforms.functional as TF


from random import randint


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms as T


from torchvision.datasets import ImageFolder


from torchvision.datasets import FakeData


import torch.nn as nn


import math


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision import models


from collections import namedtuple


import numpy as np


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers = layers + [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


CKPT_MAP = {'vgg_lpips': 'vgg.pth'}


MD5_MAP = {'vgg_lpips': 'd507d7349b931f0638a25a48a722f98a'}


URL_MAP = {'vgg_lpips': 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or check and not md5_hash(path) == MD5_MAP[name]:
        None
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name='vgg_lpips'):
        ckpt = get_ckpt_path(name, 'latent_verse/modules/autoencoder/lpips')
        self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        None

    @classmethod
    def from_pretrained(cls, name='vgg_lpips'):
        if name != 'vgg_lpips':
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val = val + res[l]
        return val


class RecPerceptualLoss(nn.Module):

    def __init__(self, loss_type='lpips_l1', perceptual_weight=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.perceptual_weight = perceptual_weight
        if loss_type == 'lpips_l1':
            self.rec_loss = nn.L1Loss(reduction='none')
        else:
            self.rec_loss = nn.MSELoss(reduction='none')
        self.perceptual_loss = LPIPS().eval()

    def forward(self, xrec, x):
        p_loss = self.perceptual_loss(x, xrec)
        rec_loss = self.rec_loss(xrec, x)
        loss = rec_loss + self.perceptual_weight * p_loss
        loss = torch.mean(loss)
        return loss


class VectorQuantizer(nn.Module):

    def __init__(self, num_tokens, codebook_dim, beta, normalized=False, contrast=False):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        loss = self.beta * F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)


class EmbeddingEMA(nn.Module):

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-05):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(nn.Module):

    def __init__(self, num_tokens, codebook_dim, beta, decay=0.99, eps=1e-05):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.training and self.embedding.update:
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            if stride == 4:
                self.conv_2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        self.stride = stride

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
            if self.stride == 4:
                x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
                x = self.conv_2(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return x


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):

    def __init__(self, *, hidden_dim, in_channels, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, resolution, z_channels, double_z=True, use_attn=False, **ignore_kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_attn = use_attn
        self.conv_in = torch.nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            if self.use_attn:
                attn = nn.ModuleList()
            block_in = hidden_dim * in_ch_mult[i_level]
            block_out = hidden_dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if self.use_attn:
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            if self.use_attn:
                down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        if self.use_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if self.use_attn:
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        if self.use_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):

    def __init__(self, *, hidden_dim, out_channels, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, use_attn=False, **ignorekwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.use_attn = use_attn
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = hidden_dim * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        None
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        if self.use_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            if self.use_attn:
                attn = nn.ModuleList()
            block_out = hidden_dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if self.use_attn:
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            if self.use_attn:
                up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        if self.use_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if self.use_attn:
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ActNorm(nn.Module):

    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-06))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        _, _, height, width = input.shape
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        h = self.scale * (input + self.loc)
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0])
            return h, logdet
        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError('Initializing ActNorm in reverse direction is disabled by default. Use allow_reverse_init=True to enable.')
            else:
                self.initialize(output)
                self.initialized.fill_(1)
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        h = output / self.scale - self.loc
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_tgisaturday_L_Verse(_paritybench_base):
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

