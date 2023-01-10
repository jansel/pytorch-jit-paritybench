import sys
_module = sys.modules[__name__]
del sys
cvivit = _module
distributed = _module
discriminator = _module
loss = _module
maskgit = _module
paella = _module
test_reconstructions = _module
train_maskgit = _module
train_vivq = _module
utils = _module
vivq = _module

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


import torch.nn as nn


import torch.nn.functional as F


import math


from torch import nn


from torch import einsum


import numpy as np


import random


import torchvision.io


import torchvision.utils as vutils


from torch.utils.data import DataLoader


from matplotlib import pyplot as plt


import time


from torch import optim


from torch.nn.parallel import DistributedDataParallel


import matplotlib.pyplot as plt


import torch.multiprocessing as mp


import torch.distributed as dist


import torchvision


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


class TemporalSpatialAttention(nn.Module):

    def __init__(self, channels, size, frames, num_layers=4, num_heads=4, spatial_first=True, pos_encodings=True):
        super(TemporalSpatialAttention, self).__init__()
        self.spatial_first = spatial_first
        self.pos_encodings = pos_encodings
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=channels * 4, nhead=num_heads, batch_first=True, norm_first=True, activation='gelu')
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers, norm=nn.LayerNorm(channels))
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers, norm=nn.LayerNorm(channels))
        if pos_encodings:
            self.spatial_positional_encoding = nn.Parameter(torch.randn(1, 1, size * size, channels) / channels ** 0.5)
            self.temporal_positional_encoding = nn.Parameter(torch.randn(1, frames, 1, channels) / channels ** 0.5)

    def _spatial_attn(self, x, base_shape):
        x = x.view(-1, *x.shape[2:])
        x = self.spatial_transformer(x)
        x = x.view(base_shape[0], base_shape[1], *x.shape[1:])
        return x

    def _temporal_attn(self, x, base_shape):
        mask = torch.triu(torch.ones(base_shape[1], base_shape[1]) * float('-inf'), diagonal=1)
        x = x.permute(0, 2, 1, 3).view(-1, x.size(1), x.size(-1))
        x = self.temporal_transformer(x, mask=mask)
        x = x.view(base_shape[0], -1, *x.shape[1:]).permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        base_shape = x.shape
        if self.spatial_first:
            x += self.spatial_positional_encoding
            x = self._spatial_attn(x, base_shape)
            x += self.temporal_positional_encoding
            x = self._temporal_attn(x, base_shape)
        else:
            x += self.temporal_positional_encoding
            x = self._temporal_attn(x, base_shape)
            x += self.spatial_positional_encoding
            x = self._spatial_attn(x, base_shape)
        return x


class ResBlockvq(nn.Module):

    def __init__(self, c, c_hidden, c_cond=0, scaler=None, kernel_size=3):
        super().__init__()
        self.resblock = nn.Sequential(nn.GELU(), nn.Conv3d(c + c_cond, c_hidden, kernel_size=1), nn.GELU(), nn.ReplicationPad3d(kernel_size // 2), nn.Conv3d(c_hidden, c_hidden, kernel_size=kernel_size, groups=c_hidden), nn.GELU(), nn.Conv3d(c_hidden, c, kernel_size=1))
        self.scaler = scaler

    def forward(self, x, s=None, encoder=True, i=None):
        res = x
        if s is not None:
            x = torch.cat([x, s], dim=1)
        x = res + self.resblock(x)
        if self.scaler is not None:
            if encoder:
                x = x.permute(0, 2, 1, 3, 4)
                x = self.scaler(x)
                x = x.permute(0, 2, 1, 3, 4)
            else:
                x = self.scaler(x)
                if i == 1:
                    x = x[:, :, 1:]
        return x


class Encoder(nn.Module):

    def __init__(self, c_in, c_hidden=256, levels=4, blocks_per_level=1, c_min=4, bottleneck_blocks=8):
        super().__init__()
        levels = levels - 2
        c_first = max(c_hidden // 4 ** max(levels - 1, 0), c_min)
        self.stem = nn.Sequential(nn.Conv3d(c_in, c_first, kernel_size=(2, 4, 4), stride=(2, 4, 4)))
        self.encoder = nn.ModuleList()
        self.remapper = nn.ModuleList()
        for i in range(levels):
            for j in range(blocks_per_level):
                bc_in_raw = c_hidden // 4 ** (levels - i - 1)
                bc_in_next_raw = c_hidden // 4 ** (levels - i)
                bc_in = max(bc_in_raw, c_min)
                bc_in_next = max(bc_in_next_raw, c_min)
                bc_hiden = bc_in * 4
                bc_cond = 0 if i == 0 and j == 0 else bc_in if j > 0 else bc_in_next
                scaler = nn.PixelUnshuffle(2) if i < levels - 1 and j == blocks_per_level - 1 else None
                if scaler is not None and bc_in_raw < bc_in:
                    self.remapper.append(nn.Sequential(nn.Conv3d(bc_in * 4, bc_in, kernel_size=1)))
                else:
                    self.remapper.append(nn.Identity())
                self.encoder.append(ResBlockvq(bc_in, bc_hiden, c_cond=bc_cond, scaler=scaler))
        for block in self.encoder:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))
        self.bottleneck = nn.Sequential(*[ResBlockvq(c_hidden, c_hidden * 4) for _ in range(bottleneck_blocks)])
        for block in self.bottleneck:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))
        self.learned_frame = nn.Parameter(torch.randn(3, 1, 128, 128) / c_hidden ** 0.5)

    def forward(self, image, video=None):
        image = torch.cat([self.learned_frame.unsqueeze(0).expand(image.shape[0], -1, -1, -1, -1), image.unsqueeze(2)], dim=2)
        if video is not None:
            video = video.permute(0, 2, 1, 3, 4)
            video = torch.cat([image, video], dim=2)
        else:
            video = image
        video = self.stem(video)
        s = None
        if len(self.encoder) > 0:
            for block, remapper in zip(self.encoder, self.remapper):
                if block.scaler is not None:
                    prev_s = nn.functional.interpolate(video, scale_factor=(1, 0.5, 0.5), recompute_scale_factor=False)
                else:
                    prev_s = video
                video = block(video, s)
                if block.scaler is not None:
                    video = remapper(video)
                s = prev_s
        video = self.bottleneck(video)
        return video


class Decoder(nn.Module):

    def __init__(self, c_out, c_hidden=256, levels=4, blocks_per_level=2, bottleneck_blocks=8, c_min=4, ucm=4, out_ks=1):
        super().__init__()
        self.bottleneck = nn.Sequential(*[ResBlockvq(c_hidden, c_hidden * 4) for _ in range(bottleneck_blocks)])
        for block in self.bottleneck:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))
        self.decoder = nn.ModuleList()
        self.blocks_per_level = blocks_per_level
        for i in range(levels):
            for j in range(blocks_per_level):
                bc_in_raw = c_hidden // ucm ** i
                bc_in_prev_raw = c_hidden // ucm ** (i - 1)
                bc_in = max(bc_in_raw, c_min)
                bc_in_prev = max(bc_in_prev_raw, c_min)
                bc_hiden = bc_in * 4
                bc_cond = 0 if i == 0 and j == 0 else bc_in if j > 0 else bc_in_prev
                if i < levels - 1 and j == blocks_per_level - 1:
                    if i == 0:
                        scaler = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2)), nn.Conv3d(bc_in, max(bc_in // ucm, c_min), kernel_size=1))
                    else:
                        scaler = nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2)), nn.Conv3d(bc_in, max(bc_in // ucm, c_min), kernel_size=1))
                else:
                    scaler = None
                block = ResBlockvq(bc_in, bc_hiden, c_cond=bc_cond, scaler=scaler)
                block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))
                self.decoder.append(block)
        self.output_convs = nn.ModuleList()
        for i in range(levels):
            bc_in = max(c_hidden // ucm ** min(i + 1, levels - 1), c_min)
            self.output_convs.append(nn.Sequential(nn.ReflectionPad3d(out_ks // 2), nn.Conv3d(bc_in, c_out, kernel_size=out_ks)))

    def forward(self, x):
        x = self.bottleneck(x)
        s = None
        outs = []
        for i, block in enumerate(self.decoder):
            if block.scaler is not None:
                if i == 1:
                    prev_s = nn.functional.interpolate(x, scale_factor=(2, 2, 2), recompute_scale_factor=False)[:, :, 1:]
                else:
                    prev_s = nn.functional.interpolate(x, scale_factor=(1, 2, 2), recompute_scale_factor=False)
            else:
                prev_s = x
            x = block(x, s, encoder=False, i=i)
            s = prev_s
            if block.scaler is not None or i == len(self.decoder) - 1:
                outs.append(self.output_convs[i // self.blocks_per_level](x))
        for i in range(len(outs) - 1):
            if outs[i].size(3) < outs[-1].size(3):
                outs[i] = nn.functional.interpolate(outs[i], size=(outs[-1].size(2), outs[-1].size(3), outs[-1].size(4)), mode='nearest')
        x = torch.stack(outs, dim=0).sum(0)
        return x.sigmoid().permute(0, 2, 1, 3, 4)


class VQModule(nn.Module):

    def __init__(self, c_hidden, k, q_init, q_refresh_step, q_refresh_end, reservoir_size=int(90000.0)):
        super().__init__()
        self.vquantizer = VectorQuantize(c_hidden, k=k, ema_loss=True)
        self.codebook_size = k
        self.q_init, self.q_refresh_step, self.q_refresh_end = q_init, q_refresh_step, q_refresh_end
        self.register_buffer('q_step_counter', torch.tensor(0))
        self.reservoir = None
        self.reservoir_size = reservoir_size

    def forward(self, x, dim=-1):
        if self.training:
            qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)
        elif self.q_step_counter < self.q_init:
            qe, commit_loss, indices = x, x.new_tensor(0), None
        else:
            qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)
        return qe, commit_loss, indices


class VIVIT(nn.Module):

    def __init__(self, patch_size=(5, 8, 8), compressed_frames=20, latent_size=32, c_hidden=64, c_codebook=16, codebook_size=1024, num_layers_enc=4, num_layers_dec=4, num_heads=4):
        super().__init__()
        self.encoder = Encoder(patch_size=patch_size, hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames, num_layers=num_layers_enc, num_heads=num_heads)
        self.cod_mapper = nn.Linear(c_hidden, c_codebook)
        self.batchnorm = nn.BatchNorm2d(c_codebook)
        self.cod_unmapper = nn.Linear(c_codebook, c_hidden)
        self.decoder = Decoder(patch_size=patch_size, hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames, num_layers=num_layers_dec, num_heads=num_heads)
        self.codebook_size = codebook_size
        self.vqmodule = VQModule(c_codebook, k=codebook_size, q_init=0, q_refresh_step=15010, q_refresh_end=15010 * 130)

    def encode(self, image, video):
        x = self.encoder(image, video)
        x = self.cod_mapper(x)
        x = self.batchnorm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        qe, commit_loss, indices = self.vqmodule(x, dim=-1)
        return (x, qe), commit_loss, indices

    def decode(self, x):
        x = self.cod_unmapper(x)
        x = self.decoder(x)
        return x

    def decode_indices(self, x):
        return self.decode(self.vqmodule.vquantizer.idx2vq(x, dim=-1))

    def forward(self, image, video=None):
        (_, qe), commit_loss, _ = self.encode(image, video)
        decoded = self.decode(qe)
        return decoded, commit_loss


class Discriminator(nn.Module):

    def __init__(self, c_in=3, c_cond=0, c_hidden=256, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [nn.utils.spectral_norm(nn.Conv2d(c_in + c_cond, c_hidden // 2 ** d, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2)]
        for i in range(depth - 1):
            c_in = c_hidden // 2 ** max(d - i, 0)
            c_out = c_hidden // 2 ** max(d - 1 - i, 0)
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(c_hidden, 1, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        x = self.discriminator(x)
        return x


class FirstStageLoss(nn.Module):

    def __init__(self, mse_weight=1.0, vq_weight=0.1, adv_weight=0.1, perc_weight=0.1, start_disc=0, device='cpu'):
        super(FirstStageLoss, self).__init__()
        self.mse_weight = mse_weight
        self.vq_weight = vq_weight
        self.adv_weight = adv_weight
        self.perc_weight = perc_weight
        self.start_disc = start_disc
        self.discriminator = Discriminator()
        self.lpips = lpips.LPIPS(net='vgg').requires_grad_(False)

    def forward(self, images, videos, reconstructions, vq_loss, step):
        if videos is not None:
            videos = torch.cat([images.unsqueeze(1), videos], dim=1).view(-1, *videos.shape[2:])
        else:
            videos = images
        reconstructions = reconstructions.contiguous().view(-1, *reconstructions.shape[2:])
        mse_loss = F.mse_loss(videos, reconstructions)
        if step >= self.start_disc:
            d_real = self.discriminator(videos)
            d_real_loss = F.binary_cross_entropy(d_real, torch.zeros_like(d_real) + 0.1)
            d_fake = self.discriminator(reconstructions.detach())
            d_fake_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake) - 0.1)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_recon = self.discriminator(reconstructions)
            g_loss = F.binary_cross_entropy(d_recon, torch.zeros_like(d_recon) + 0.1)
        else:
            g_loss = reconstructions.new_tensor(0)
            d_loss = None
        lpips_loss = self.lpips(videos, reconstructions).mean()
        loss = self.mse_weight * mse_loss + self.adv_weight * g_loss + self.perc_weight * lpips_loss + self.vq_weight * vq_loss
        return loss, d_loss


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):

    def __init__(self, dim, dim_context=None, dim_head=64, heads=8, causal=False, num_null_kv=2):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        self.norm = nn.LayerNorm(dim)
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        kv_input = default(context, x)
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale
        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]
        if exists(mask):
            mask = F.pad(mask, (j - mask.shape[-1], 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(mult * dim)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False))


class Transformer(nn.Module):

    def __init__(self, dim, *, depth, dim_context=None, causal=False, dim_head=64, heads=8, ff_mult=4, attn_num_null_kv=2, has_cross_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal), Attention(dim=dim, dim_head=dim_head, dim_context=dim_context, heads=heads, causal=False, num_null_kv=attn_num_null_kv) if has_cross_attn else None, FeedForward(dim=dim, mult=ff_mult)]))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=None) + x
            x = ff(x) + x
        return self.norm_out(x)


class MaskGit(nn.Module):

    def __init__(self, *, dim, num_tokens, max_seq_len, **kwargs):
        super().__init__()
        self.mask_id = num_tokens
        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.transformer = Transformer(dim=dim, attn_num_null_kv=2, has_cross_attn=True, **kwargs)
        self.to_logits = nn.Linear(dim, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r):
        r = self.gamma(r)[:, None]
        mask = torch.bernoulli(r * torch.ones_like(x))
        mask = mask.round().bool()
        x = x * ~mask + self.mask_id * mask
        return x, mask

    def forward(self, x, text_cond=None, **kwargs):
        b, n = x.shape
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=x.device)) + x
        x = self.transformer(x, text_cond, **kwargs)
        return self.to_logits(x)

    def loss(self, pred, video_indices, mask):
        acc = (pred.permute(0, 2, 1).argmax(1) == video_indices).float().mean()
        video_indices = video_indices[mask]
        mask = mask.flatten()
        pred = pred.view(-1, pred.shape[-1])[mask]
        return self.loss_fn(pred, video_indices), acc


class MaskGitTrainWrapper(nn.Module):

    def __init__(self, maskgit, *, steps):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id
        self.steps = steps

    def forward(self, x, **kwargs):
        batch, seq, device = *x.shape, x.device
        self.maskgit.train()
        rand_step = torch.randint(0, self.steps, (1,), device=device)
        num_tokens_mask = (seq * torch.cos(rand_step * math.pi * 0.5 / self.steps)).round().long().clamp(min=1)
        _, indices = torch.randn((batch, seq), device=device).topk(num_tokens_mask.item(), dim=-1)
        mask = torch.zeros((batch, seq), device=device).scatter(1, indices, 1.0).bool()
        masked_input = torch.where(mask, self.mask_id, x)
        logits = self.maskgit(masked_input, **kwargs)
        loss = F.cross_entropy(logits[mask], x[mask])
        return loss


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)


class Phenaki(nn.Module):

    def __init__(self, *, maskgit: MaskGit, steps=18, sample_temperature=0.0, text_embed_dim=None, cond_drop_prob=0.25, max_text_len=128):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id
        self.maskgit_trainer = MaskGitTrainWrapper(maskgit, steps=steps)
        self.steps = steps
        self.sample_temperature = sample_temperature
        assert cond_drop_prob > 0.0
        self.cond_drop_prob = cond_drop_prob

    @torch.no_grad()
    def sample(self, *, text, num_frames, cond_scale=3.0, starting_temperature=0.9, noise_K=1.0):
        device = next(self.parameters()).device
        num_tokens = self.cvivit.num_tokens_per_frames(num_frames)
        with torch.no_grad():
            text_embeds = self.encode_texts([text], output_device=device)
            text_mask = torch.any(text_embeds != 0, dim=-1)
        shape = 1, num_tokens
        video_token_ids = torch.full(shape, self.mask_id, device=device)
        mask = torch.ones(shape, device=device, dtype=torch.bool)
        scores = None
        for step in range(self.steps):
            is_first_step = step == 0
            is_last_step = step == self.steps - 1
            steps_til_x0 = self.steps - (step + 1)
            if not is_first_step and exists(scores):
                time = torch.full((1,), step / self.steps, device=device)
                num_tokens_mask = (num_tokens * torch.cos(time * math.pi * 0.5)).round().long().clamp(min=1)
                _, indices = scores.topk(num_tokens_mask.item(), dim=-1)
                mask = torch.zeros(shape, device=device).scatter(1, indices, 1).bool()
            video_token_ids = torch.where(mask, self.mask_id, video_token_ids)
            logits = self.maskgit.forward_with_cond_scale(video_token_ids, context=text_embeds, mask=text_mask, cond_scale=cond_scale)
            temperature = starting_temperature * (step / steps_til_x0)
            pred_video_ids = gumbel_sample(logits, temperature=temperature)
            video_token_ids = torch.where(mask, pred_video_ids, video_token_ids)
            if not is_last_step:
                scores = logits.gather(2, rearrange(pred_video_ids, '... -> ... 1'))
                scores = 1 - rearrange(scores, '... 1 -> ...')
                scores = torch.where(mask, scores, -10000.0)
        video = self.cvivit.decode_from_codebook_indices(video_token_ids)
        return video


class ModulatedLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-06, channels_first=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.channels_first = channels_first

    def forward(self, x, w=None):
        """
        x: B x C x T x H x W
        x -> B x T x H x W x C
        """
        x = x.permute(0, 2, 3, 4, 1) if self.channels_first else x
        if w is None:
            x = self.ln(x)
        else:
            x = self.gamma * w * self.ln(x) + self.beta * w
        x = x.permute(0, 4, 1, 2, 3) if self.channels_first else x
        return x


class ResBlock(nn.Module):

    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-06):
        super().__init__()
        self.depthwise = nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(c, c, kernel_size=3, groups=c))
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        self.channelwise = nn.Sequential(nn.Linear(c + c_skip, c_hidden), nn.GELU(), nn.Linear(c_hidden, c))
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c), requires_grad=True) if layer_scale_init_value > 0 else None
        self.scaler = scaler
        if c_cond > 0:
            self.cond_mapper = nn.Linear(c_cond, c)

    def forward(self, x, s=None, skip=None):
        """
        x: B x 1280 x 6 x 16 x 16
        s: B x E x 1 x 1 x 1
        s -> B x E x T x H x W
        """
        res = x
        x = self.depthwise(x)
        if s is not None:
            s = self.cond_mapper(s.permute(0, 2, 3, 4, 1))
            if s.size(1) == s.size(2) == x.size(3) == 1:
                s = s.expand(-1, x.size(2), x.size(3), x.size(4), -1)
        x = self.ln(x.permute(0, 2, 3, 4, 1), s)
        if skip is not None:
            x = torch.cat([x, skip.permute(0, 2, 3, 4, 1)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma * x if self.gamma is not None else x
        x = res + x.permute(0, 4, 1, 2, 3)
        if self.scaler is not None:
            x = self.scaler(x)
        return x


class DenoiseUNet(nn.Module):

    def __init__(self, num_labels, c_hidden=1280, c_clip=1024, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4]):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [(c_hidden // 2 ** i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels, c_levels[0])
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv3d(c_levels[i - 1], c_levels[i], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1))
            for _ in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_clip + c_r)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 1 - i] * 4, c_clip + c_r, c_levels[len(c_levels) - 1 - i] if j == 0 and i > 0 else 0)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(nn.ConvTranspose3d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))
        self.clf = nn.Conv3d(c_levels[0], num_labels, kernel_size=1)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r, random_x=None):
        r = self.gamma(r)[:, None, None, None]
        mask = torch.bernoulli(r * torch.ones_like(x))
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def gen_r_embedding(self, r, max_positions=10000):
        dtype = r.dtype
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    x = block(x, s)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
        return x

    def forward(self, x, c, r):
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 4, 1, 2, 3)
        s = torch.cat([c, r_embed], dim=-1)[:, :, None, None, None]
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x

    def loss(self, pred, video_indices):
        acc = (pred.argmax(1) == video_indices).float().mean()
        return self.loss_fn(pred, video_indices), acc


BASE_SHAPE = 6, 16, 16


class VIVQ(nn.Module):

    def __init__(self, base_channels=3, c_hidden=512, c_codebook=16, codebook_size=1024):
        super().__init__()
        self.encoder = Encoder(base_channels, c_hidden=c_hidden)
        self.cod_mapper = nn.Sequential(nn.Conv3d(c_hidden, c_codebook, kernel_size=1), nn.BatchNorm3d(c_codebook))
        self.cod_unmapper = nn.Conv3d(c_codebook, c_hidden, kernel_size=1)
        self.decoder = Decoder(base_channels, c_hidden=c_hidden)
        self.codebook_size = codebook_size
        self.vqmodule = VQModule(c_codebook, k=codebook_size, q_init=1000, q_refresh_step=1000, q_refresh_end=5000)

    def encode(self, image, video):
        x = self.encoder(image, video)
        x = self.cod_mapper(x)
        shape = x.shape
        x = x.view(*x.shape[:3], x.shape[3] * x.shape[4]).permute(0, 2, 3, 1)
        qe, commit_loss, indices = self.vqmodule(x, dim=-1)
        if video is not None:
            indices = indices.view(image.shape[0], *BASE_SHAPE)
        else:
            indices = indices.view(image.shape[0], *BASE_SHAPE[1:]).unsqueeze(1)
        return (x, qe), commit_loss, indices, shape

    def decode(self, x, shape=None):
        if shape is not None:
            x = x.permute(0, 3, 1, 2).view(shape)
        x = self.cod_unmapper(x)
        x = self.decoder(x)
        return x

    def decode_indices(self, x, shape=None):
        if shape is not None:
            x = x.view(x.shape[0], *shape)
        return self.decode(self.vqmodule.vquantizer.idx2vq(x, dim=-1).permute(0, 4, 1, 2, 3))

    def forward(self, image, video=None):
        (_, qe), commit_loss, _, shape = self.encode(image, video)
        decoded = self.decode(qe, shape)
        return decoded, commit_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResBlockvq,
     lambda: ([], {'c': 4, 'c_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_LAION_AI_phenaki(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

