import sys
_module = sys.modules[__name__]
del sys
train = _module
train = _module
enc_dec_copy = _module
enc_dec_copy_apex = _module
performer_pytorch = _module
autoregressive_wrapper = _module
performer_enc_dec = _module
performer_pytorch = _module
reversible = _module
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


import numpy as np


import torch


import torch.optim as optim


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from functools import partial


from torch import nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence


import re


import math


import torch.nn as nn


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


def exists(val):
    return val is not None


def repetition_penalty_fn(logits, ctx, theta=1.2):
    w = torch.ones(logits.shape[-1], dtype=torch.float, device=logits.device)
    for i in torch.unique(ctx):
        w[i] = theta
    return logits / w


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):

    def __init__(self, net, ignore_index=0, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_logits_fn=top_k, filter_thres=0.9, repetition_penalty=1.0, repetition_penalty_ctx=32, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]
        b, t = start_tokens.shape
        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('mask', None)
        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)
        context_mask = kwargs.pop('context_mask', None)
        if 'context' in kwargs and not exists(context_mask):
            context = kwargs['context']
            context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=out.device)
        kwargs.update(context_mask=context_mask)
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]
            logits = self.net(x, mask=input_mask, **kwargs)[:, -1, :]
            if repetition_penalty > 1.0:
                logits = repetition_penalty_fn(logits, out[-repetition_penalty_ctx:], theta=repetition_penalty)
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)
            if eos_token is not None and (sample == eos_token).all():
                break
        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]
        mask = kwargs.pop('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
        kwargs.update(mask=mask)
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
        return loss


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class Always(nn.Module):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j->ij', position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :]


class Chunk(nn.Module):

    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


def causal_linear_attention(q, k, v, eps=1e-06):
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled=False)
    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply
    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1.0 / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))
        out = causal_dot_product_fn(q, k, v)
    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out


def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-06):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []
    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
        D_inv = 1.0 / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)
        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)
    return torch.cat(outs, dim=-2)


def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t, (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=0.0001, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0 * data_normalizer ** 2
    diag_data = diag_data.unsqueeze(dim=-1)
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)
    return data_dash.type_as(data)


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal
        if causal:
            try:
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                None
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2), (sin, cos))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    return q, k


def empty(tensor):
    return tensor.numel() == 0


class Attention(nn.Module):

    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), dropout=0.0, no_projection=False, qkv_bias=False, attn_out_bias=True):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, no_projection=no_projection)
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None
        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)
            out = self.fast_attention(q, k, v)
            attn_outs.append(out)
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)
        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class CrossAttention(Attention):

    def forward(self, *args, context=None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x


class PreLayerNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreScaleNorm(nn.Module):

    def __init__(self, dim, fn, eps=1e-05):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)
    return F.pad(t, (0, 0, amount, -amount), value=0.0)


class PreShiftTokens(nn.Module):

    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


def get_module_device(module):
    return next(module.parameters()).device


class ProjectionUpdater(nn.Module):

    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance
        if not self.training:
            return
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)
            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)
            self.calls_since_last_redraw.zero_()
            return
        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented


class ReZero(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(0.001))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class Deterministic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


class _ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]
    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: {key: val} if route else {}, routes)
            routed_args[depth] = {**f_args, **new_f_args}, {**g_args, **new_g_args}
    return routed_args


class ReversibleSequence(nn.Module):

    def __init__(self, blocks, args_route={}):
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))
        out = _ReversibleFunction.apply(x, blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)


class SelfAttention(Attention):

    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)


class SequentialSequence(nn.Module):

    def __init__(self, layers, args_route={}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))
        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


class Performer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, local_attn_heads=0, local_window_size=256, causal=False, ff_mult=4, nb_features=None, feature_redraw_interval=1000, reversible=False, ff_chunks=1, generalized_attention=False, kernel_fn=nn.ReLU(), use_scalenorm=False, use_rezero=False, ff_glu=False, ff_dropout=0.0, attn_dropout=0.0, cross_attend=False, no_projection=False, auto_check_redraw=True, qkv_bias=True, attn_out_bias=True, shift_tokens=False):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'
        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)
        for _, local_heads in zip(range(depth), local_attn_heads):
            attn = SelfAttention(dim, causal=causal, heads=heads, dim_head=dim_head, local_heads=local_heads, local_window_size=local_window_size, nb_features=nb_features, generalized_attention=generalized_attention, kernel_fn=kernel_fn, dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)
            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))
            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))
            if not cross_attend:
                continue
            layers.append(nn.ModuleList([wrapper_fn(CrossAttention(dim, heads=heads, dim_head=dim_head, nb_features=nb_features, generalized_attention=generalized_attention, kernel_fn=kernel_fn, dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)), wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))]))
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)


class PerformerLM(nn.Module):

    def __init__(self, *, num_tokens, max_seq_len, dim, depth, heads, dim_head=64, local_attn_heads=0, local_window_size=256, causal=False, ff_mult=4, nb_features=None, feature_redraw_interval=1000, reversible=False, ff_chunks=1, ff_glu=False, emb_dropout=0.0, ff_dropout=0.0, attn_dropout=0.0, generalized_attention=False, kernel_fn=nn.ReLU(), use_scalenorm=False, use_rezero=False, cross_attend=False, no_projection=False, tie_embed=False, rotary_position_emb=True, axial_position_emb=False, axial_position_shape=None, auto_check_redraw=True, qkv_bias=False, attn_out_bias=False, shift_tokens=False):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        self.dropout = nn.Dropout(emb_dropout)
        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias, attn_out_bias, shift_tokens)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.dropout(x)
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)
        x = self.norm(x)
        if return_encodings:
            return x
        if exists(self.to_out):
            return self.to_out(x)
        return x @ self.token_emb.weight.t()


DEC_PREFIX = 'dec_'


ENC_PREFIX = 'enc_'


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs


def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    return enc_kwargs, dec_kwargs, kwargs


class PerformerEncDec(nn.Module):

    def __init__(self, dim, ignore_index=0, pad_value=0, tie_token_embeds=False, no_projection=False, **kwargs):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'
        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = dec_kwargs['no_projection'] = no_projection
        dec_kwargs['causal'] = True
        dec_kwargs['cross_attend'] = True
        enc = PerformerLM(**enc_kwargs)
        dec = PerformerLM(**dec_kwargs)
        if tie_token_embeds:
            enc.token_emb = dec.token_emb
        self.enc = enc
        self.dec = AutoregressiveWrapper(dec, ignore_index=ignore_index, pad_value=pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        encodings = self.enc(seq_in, return_encodings=True, **enc_kwargs)
        return self.dec.generate(seq_out_start, seq_len, context=encodings, **{**dec_kwargs, **kwargs})

    def forward(self, seq_in, seq_out, enc_mask=None, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        encodings = self.enc(seq_in, mask=enc_mask, return_encodings=True, **enc_kwargs)
        return self.dec(seq_out, context=encodings, context_mask=enc_mask, **dec_kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AbsolutePositionalEmbedding,
     lambda: ([], {'dim': 4, 'max_seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Always,
     lambda: ([], {'val': 4}),
     lambda: ([], {}),
     False),
    (Chunk,
     lambda: ([], {'chunks': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Deterministic,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FixedPositionalEmbedding,
     lambda: ([], {'dim': 4, 'max_seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreLayerNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreScaleNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreShiftTokens,
     lambda: ([], {'shifts': [4, 4], 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReZero,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_performer_pytorch(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

