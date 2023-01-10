import sys
_module = sys.modules[__name__]
del sys
setup = _module
vector_quantize_pytorch = _module
residual_vq = _module
vector_quantize_pytorch = _module

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


from random import randrange


import torch


from torch import nn


import torch.nn.functional as F


from torch import einsum


import torch.distributed as distributed


from torch.cuda.amp import autocast


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d=dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
    return embeds.gather(2, indices)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)
    return (t / temperature + gumbel_noise(t)).argmax(dim=dim)


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def noop(*args, **kwargs):
    pass


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False, sample_fn=batched_sample_vectors, all_reduce_fn=noop):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    means = sample_fn(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p=2)
        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(rearrange(zero_mask, '... -> ... 1'), means, new_means)
    return means, bins


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def pad_shape(shape, size, dim=0):
    return [(size if i == dim else s) for i, s in enumerate(shape)]


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []
    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    distributed.barrier()
    return all_x


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()
    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)
    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    return sample


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)
    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)
    return rearrange(out, '... -> 1 ...')


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


class CosineSimCodebook(nn.Module):

    def __init__(self, dim, codebook_size, num_codebooks=1, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, decay=0.8, eps=1e-05, threshold_ema_dead_code=2, use_ddp=False, learnable_codebook=False, sample_codebook_temp=0.0):
        super().__init__()
        self.decay = decay
        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim=True, sample_fn=self.sample_fn, all_reduce_fn=self.kmeans_all_reduce_fn)
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)
        self.init_embed_(flatten)
        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)
        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, self.embed)
        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)
            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(rearrange(zero_mask, '... -> ... 1'), embed, embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)
        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        return quantize, embed_ind


def laplace_smoothing(x, n_categories, eps=1e-05):
    return (x + eps) / (x.sum() + n_categories * eps)


class EuclideanCodebook(nn.Module):

    def __init__(self, dim, codebook_size, num_codebooks=1, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, decay=0.8, eps=1e-05, threshold_ema_dead_code=2, use_ddp=False, learnable_codebook=False, sample_codebook_temp=0):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, sample_fn=self.sample_fn, all_reduce_fn=self.kmeans_all_reduce_fn)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        self.init_embed_(flatten)
        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        dist = -torch.cdist(flatten, embed, p=2)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, self.embed)
        if self.training:
            cluster_size = embed_onehot.sum(dim=1)
            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size, cluster_size, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)
        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        return quantize, embed_ind


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def orthogonal_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - 1 / n


class VectorQuantize(nn.Module):

    def __init__(self, dim, codebook_size, codebook_dim=None, heads=1, separate_codebook_per_head=False, decay=0.8, eps=1e-05, kmeans_init=False, kmeans_iters=10, sync_kmeans=True, use_cosine_sim=False, threshold_ema_dead_code=0, channel_last=True, accept_image_fmap=False, commitment_weight=1.0, orthogonal_reg_weight=0.0, orthogonal_reg_active_codes_only=False, orthogonal_reg_max_codes=None, sample_codebook_temp=0.0, sync_codebook=False):
        super().__init__()
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.eps = eps
        self.commitment_weight = commitment_weight
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook
        self._codebook = codebook_class(dim=codebook_dim, num_codebooks=heads if separate_codebook_per_head else 1, codebook_size=codebook_size, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sync_kmeans=sync_kmeans, decay=decay, eps=eps, threshold_ema_dead_code=threshold_ema_dead_code, use_ddp=sync_codebook, learnable_codebook=has_codebook_orthogonal_loss, sample_codebook_temp=sample_codebook_temp)
        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook
        return rearrange(codebook, '1 ... -> ...')

    def forward(self, x, mask=None):
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        need_transpose = not self.channel_last and not self.accept_image_fmap
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=heads)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach()
                if exists(mask):
                    commit_loss = F.mse_loss(detached_quantize, x, reduction='none')
                    if is_multiheaded:
                        mask = repeat(mask, 'b n -> c (b h) n', c=commit_loss.shape[0], h=commit_loss.shape[1] // mask.shape[0])
                    commit_loss = commit_loss[mask].mean()
                else:
                    commit_loss = F.mse_loss(detached_quantize, x)
                loss = loss + commit_loss * self.commitment_weight
            if self.orthogonal_reg_weight > 0:
                codebook = self._codebook.embed
                if self.orthogonal_reg_active_codes_only:
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]
                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=heads)
        quantize = self.project_out(quantize)
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)
        return quantize, embed_ind, loss


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(self, *, num_quantizers, shared_codebook=False, heads=1, quantize_dropout=False, quantize_dropout_cutoff_index=0, **kwargs):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])
        self.quantize_dropout = quantize_dropout
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        if not shared_codebook:
            return
        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook
        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        codebooks = rearrange(codebooks, 'q 1 c d -> q c d')
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])
        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(mask, 0)
        all_codes = codebooks.gather(2, gather_indices)
        all_codes = all_codes.masked_fill(mask, 0.0)
        return all_codes

    def forward(self, x, return_all_codes=False):
        b, n, *_, num_quant, device = *x.shape, self.num_quantizers, x.device
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        should_quantize_dropout = self.training and self.quantize_dropout
        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                null_indices = torch.full((b, n), -1.0, device=device, dtype=torch.long)
                null_loss = torch.full((1,), 0.0, device=device, dtype=x.dtype)
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))
        ret = quantized_out, all_indices, all_losses
        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = *ret, all_codes
        return ret

